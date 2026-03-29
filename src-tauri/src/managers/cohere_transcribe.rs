use anyhow::{Context, Result};
use log::{debug, info, warn};
use serde_json::{json, Value};
use std::collections::VecDeque;
use std::ffi::OsStr;
use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command, Stdio};
use std::thread;
use tauri::{AppHandle, Manager};
use transcribe_rs::TranscriptionResult;

pub const COHERE_MODEL_ID: &str = "cohere-transcribe-03-2026";
pub const COHERE_MODEL_NAME: &str = "Cohere Transcribe";
pub const COHERE_MODEL_DIRNAME: &str = "cohere-transcribe-03-2026";
pub const COHERE_INSTRUCTIONS_URL: &str =
    "https://huggingface.co/CohereLabs/cohere-transcribe-03-2026";
pub const COHERE_DEFAULT_LANGUAGE: &str = "en";

const COHERE_RUNTIME_DIRNAME: &str = "cohere-transcribe-runtime";
const COHERE_SETUP_README: &str = "README.txt";
const COHERE_REQUIREMENTS_RESOURCE: &str = "resources/scripts/cohere_transcribe_requirements.txt";
const COHERE_WORKER_RESOURCE: &str = "resources/scripts/cohere_transcribe_worker.py";
const COHERE_SAMPLE_RATE: u32 = 16_000;
const COHERE_MIN_PYTHON_MINOR: u32 = 10;

pub fn supported_languages() -> Vec<String> {
    vec![
        "en", "fr", "de", "it", "es", "pt", "el", "nl", "pl", "zh-Hans", "zh-Hant", "ja", "ko",
        "vi", "ar",
    ]
    .into_iter()
    .map(String::from)
    .collect()
}

pub fn map_ui_language(language: &str) -> Option<&'static str> {
    match language {
        "en" => Some("en"),
        "fr" => Some("fr"),
        "de" => Some("de"),
        "it" => Some("it"),
        "es" => Some("es"),
        "pt" => Some("pt"),
        "el" => Some("el"),
        "nl" => Some("nl"),
        "pl" => Some("pl"),
        "zh-Hans" | "zh-Hant" => Some("zh"),
        "ja" => Some("ja"),
        "ko" => Some("ko"),
        "vi" => Some("vi"),
        "ar" => Some("ar"),
        _ => None,
    }
}

pub fn model_dir(app: &AppHandle) -> Result<PathBuf> {
    Ok(crate::portable::app_data_dir(app)
        .map_err(|e| anyhow::anyhow!("Failed to get app data directory: {}", e))?
        .join("models")
        .join(COHERE_MODEL_DIRNAME))
}

pub fn prepare_manual_install_dir(app: &AppHandle) -> Result<PathBuf> {
    let dir = model_dir(app)?;
    fs::create_dir_all(&dir)?;

    let readme_path = dir.join(COHERE_SETUP_README);
    fs::write(&readme_path, setup_readme_contents(&dir))?;

    Ok(dir)
}

pub fn find_model_root(base_dir: &Path) -> Option<PathBuf> {
    if !base_dir.exists() || !base_dir.is_dir() {
        return None;
    }

    let mut queue = VecDeque::from([(base_dir.to_path_buf(), 0usize)]);
    while let Some((dir, depth)) = queue.pop_front() {
        if is_valid_model_root(&dir) {
            return Some(dir);
        }

        if depth >= 3 {
            continue;
        }

        let entries = fs::read_dir(&dir).ok()?;
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                queue.push_back((path, depth + 1));
            }
        }
    }

    None
}

pub struct CohereTranscribeEngine {
    child: Child,
    stdin: ChildStdin,
    stdout: BufReader<ChildStdout>,
}

impl CohereTranscribeEngine {
    pub fn load(app: &AppHandle, model_root: &Path) -> Result<Self> {
        let runtime = ensure_runtime(app)?;
        let worker = app
            .path()
            .resolve(COHERE_WORKER_RESOURCE, tauri::path::BaseDirectory::Resource)
            .map_err(|e| anyhow::anyhow!("Failed to resolve Cohere worker path: {}", e))?;

        let mut command = Command::new(&runtime.python);
        command
            .arg(&worker)
            .arg("--model-dir")
            .arg(model_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .env("PYTHONUNBUFFERED", "1")
            .env("PYTHONNOUSERSITE", "1");

        let mut child = command.spawn().with_context(|| {
            format!(
                "Failed to start Cohere Transcribe worker with Python at {}",
                runtime.python.display()
            )
        })?;

        let stderr = child
            .stderr
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture Cohere worker stderr"))?;
        spawn_stderr_logger(stderr);

        let stdin = child
            .stdin
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture Cohere worker stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| anyhow::anyhow!("Failed to capture Cohere worker stdout"))?;
        let mut stdout = BufReader::new(stdout);

        let ready = read_json_line(&mut stdout)
            .context("Cohere Transcribe worker failed before signalling readiness")?;
        if ready.get("status").and_then(Value::as_str) != Some("ready") {
            let error = ready
                .get("error")
                .and_then(Value::as_str)
                .unwrap_or("unknown startup failure");
            anyhow::bail!("Cohere Transcribe worker failed to start: {}", error);
        }

        let device = ready
            .get("device")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        let backend = ready
            .get("backend")
            .and_then(Value::as_str)
            .unwrap_or("unknown");
        info!(
            "Cohere Transcribe worker ready on {} using {} backend",
            device, backend
        );

        Ok(Self {
            child,
            stdin,
            stdout,
        })
    }

    pub fn transcribe(&mut self, audio: &[f32], language: &str) -> Result<TranscriptionResult> {
        let mapped_language = map_ui_language(language)
            .ok_or_else(|| anyhow::anyhow!("Unsupported Cohere language: {}", language))?;

        let request = json!({
            "command": "transcribe",
            "language": mapped_language,
            "sample_rate": COHERE_SAMPLE_RATE,
            "audio_bytes_len": audio.len() * std::mem::size_of::<f32>(),
        });
        write_audio_request(&mut self.stdin, &request, audio)?;

        let response = read_json_line(&mut self.stdout)
            .context("Cohere worker stopped responding during transcription")?;

        match response.get("status").and_then(Value::as_str) {
            Some("ok") => {
                let text = response
                    .get("text")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string();
                Ok(TranscriptionResult {
                    text,
                    segments: None,
                })
            }
            _ => {
                let error = response
                    .get("error")
                    .and_then(Value::as_str)
                    .unwrap_or("Unknown Cohere worker error");
                anyhow::bail!("Cohere transcription failed: {}", error);
            }
        }
    }
}

impl Drop for CohereTranscribeEngine {
    fn drop(&mut self) {
        let _ = write_json_line(&mut self.stdin, &json!({ "command": "shutdown" }));
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

struct RuntimePaths {
    python: PathBuf,
}

struct PythonCandidate {
    executable: String,
    major: u32,
    minor: u32,
}

fn is_valid_model_root(dir: &Path) -> bool {
    if !dir.is_dir() {
        return false;
    }

    let has_required_configs = [
        "config.json",
        "preprocessor_config.json",
        "processor_config.json",
        "tokenizer_config.json",
    ]
    .into_iter()
    .all(|name| dir.join(name).exists());

    if !has_required_configs {
        return false;
    }

    let has_tokenizer = dir.join("tokenizer.json").exists()
        || dir.join("tokenizer.model").exists()
        || dir.join("spiece.model").exists();
    if !has_tokenizer {
        return false;
    }

    let has_weights = dir.join("model.safetensors.index.json").exists()
        || fs::read_dir(dir)
            .ok()
            .map(|entries| {
                entries.flatten().any(|entry| {
                    entry
                        .path()
                        .extension()
                        .and_then(|ext| ext.to_str())
                        .map(|ext| ext == "safetensors")
                        .unwrap_or(false)
                })
            })
            .unwrap_or(false);

    has_weights
}

fn setup_readme_contents(dir: &Path) -> String {
    format!(
        "Handy Cohere Transcribe beta setup\n\n\
1. Request access to the gated model at:\n\
   {url}\n\n\
2. Install the Hugging Face CLI:\n\
   https://huggingface.co/docs/huggingface_hub/en/guides/cli\n\
3. Sign in with the Hugging Face CLI after your access request is approved:\n\
   hf auth login\n\n\
4. From this folder, download the latest model snapshot directly into place:\n\
   cd \"{dir}\"\n\
   hf download CohereLabs/cohere-transcribe-03-2026 --local-dir .\n\n\
When the files are present, Handy will detect the model automatically. If it is\n\
already open, return to the app after copying the files and wait a few seconds.\n",
        url = COHERE_INSTRUCTIONS_URL,
        dir = dir.display()
    )
}

fn ensure_runtime(app: &AppHandle) -> Result<RuntimePaths> {
    let app_data = crate::portable::app_data_dir(app)
        .map_err(|e| anyhow::anyhow!("Failed to get app data directory: {}", e))?;
    let runtime_dir = app_data.join(COHERE_RUNTIME_DIRNAME);
    let venv_dir = runtime_dir.join("venv");
    let python_path = venv_dir.join("bin").join("python");
    let marker_path = runtime_dir.join("requirements.marker");
    let resolved_requirements_path = runtime_dir.join("requirements.resolved.txt");

    fs::create_dir_all(&runtime_dir)?;

    let requirements_path = app
        .path()
        .resolve(
            COHERE_REQUIREMENTS_RESOURCE,
            tauri::path::BaseDirectory::Resource,
        )
        .map_err(|e| anyhow::anyhow!("Failed to resolve Cohere requirements path: {}", e))?;
    let requirements_contents = fs::read_to_string(&requirements_path)?;
    let system_python = find_system_python()?;
    let expected_python = (system_python.major, system_python.minor);
    let existing_python = detect_python_version(&python_path);
    let should_recreate_venv = python_path.exists() && existing_python != Some(expected_python);
    let needs_install = !python_path.exists()
        || should_recreate_venv
        || fs::read_to_string(&marker_path).unwrap_or_default() != requirements_contents;

    if needs_install {
        info!(
            "Preparing Cohere Transcribe Python runtime with {} ({}.{}).",
            system_python.executable, system_python.major, system_python.minor
        );
        fs::write(&resolved_requirements_path, &requirements_contents)?;

        if !python_path.exists() || should_recreate_venv {
            if venv_dir.exists() {
                info!(
                    "Recreating Cohere Python virtual environment to match preferred interpreter {}.{}",
                    system_python.major, system_python.minor
                );
                let _ = fs::remove_dir_all(&venv_dir);
            }
            run_command(
                Command::new(&system_python.executable)
                    .args(["-m", "venv"])
                    .arg(&venv_dir),
                "create Cohere Python virtual environment",
            )?;
        }

        run_command(
            Command::new(&python_path)
                .env("PIP_DISABLE_PIP_VERSION_CHECK", "1")
                .args(["-m", "pip", "install", "--upgrade", "pip"]),
            "upgrade pip for Cohere runtime",
        )?;

        run_command(
            Command::new(&python_path)
                .env("PIP_DISABLE_PIP_VERSION_CHECK", "1")
                .args(["-m", "pip", "install", "-r"])
                .arg(&resolved_requirements_path),
            "install Cohere runtime dependencies",
        )?;

        fs::write(&marker_path, requirements_contents)?;
    }

    Ok(RuntimePaths {
        python: python_path,
    })
}

fn find_system_python() -> Result<PythonCandidate> {
    let versioned = ["python3.12", "python3.11", "python3.10"];
    let unversioned = ["python3", "python"];

    // First try bare names from PATH (works in dev / terminal-launched builds).
    for candidate in versioned.iter().chain(unversioned.iter()) {
        if let Some(c) = check_python_candidate(candidate) {
            return Ok(c);
        }
    }

    // GUI .app bundles on macOS inherit a minimal PATH that excludes most
    // user-installed Pythons. Probe well-known directories explicitly.
    for dir in wellknown_python_dirs() {
        for name in versioned.iter().chain(unversioned.iter()) {
            let full = dir.join(name);
            if full.exists() {
                if let Some(c) = check_python_candidate(&full) {
                    return Ok(c);
                }
            }
        }
    }

    anyhow::bail!(
        "Python 3.10 or later is required for Cohere Transcribe. \
         Install Python 3.10+ and try again."
    )
}

fn check_python_candidate(command: impl AsRef<OsStr> + std::fmt::Debug) -> Option<PythonCandidate> {
    let (major, minor) = detect_python_version(&command)?;
    if major == 3 && minor >= COHERE_MIN_PYTHON_MINOR {
        let executable = command
            .as_ref()
            .to_str()
            .unwrap_or_default()
            .to_string();
        Some(PythonCandidate {
            executable,
            major,
            minor,
        })
    } else {
        None
    }
}

/// Returns well-known directories where Python may be installed on macOS /
/// Linux, ordered by preference (newest / most common first).
fn wellknown_python_dirs() -> Vec<PathBuf> {
    let mut dirs: Vec<PathBuf> = Vec::new();

    if let Ok(home) = std::env::var("HOME").map(PathBuf::from) {
        // uv / pipx: ~/.local/bin
        dirs.push(home.join(".local/bin"));

        // uv managed installs: ~/.local/share/uv/python/cpython-3.*/bin
        if let Ok(entries) = fs::read_dir(home.join(".local/share/uv/python")) {
            let mut uv_bins: Vec<PathBuf> = entries
                .flatten()
                .filter_map(|e| {
                    let p = e.path();
                    if p.is_dir() {
                        Some(p.join("bin"))
                    } else {
                        None
                    }
                })
                .collect();
            uv_bins.sort();
            uv_bins.reverse(); // prefer newer versions
            dirs.extend(uv_bins);
        }

        // pyenv: ~/.pyenv/shims
        dirs.push(home.join(".pyenv/shims"));

        // mise / rtx: ~/.local/share/mise/shims
        dirs.push(home.join(".local/share/mise/shims"));
    }

    // Homebrew (Apple Silicon then Intel)
    dirs.push(PathBuf::from("/opt/homebrew/bin"));
    dirs.push(PathBuf::from("/usr/local/bin"));

    // python.org framework installs
    for minor in (COHERE_MIN_PYTHON_MINOR..=14).rev() {
        dirs.push(PathBuf::from(format!(
            "/Library/Frameworks/Python.framework/Versions/3.{minor}/bin"
        )));
    }

    dirs
}

fn detect_python_version(command: impl AsRef<OsStr>) -> Option<(u32, u32)> {
    let output = Command::new(command).arg("--version").output().ok()?;
    if !output.status.success() {
        return None;
    }

    let text = if !output.stdout.is_empty() {
        String::from_utf8_lossy(&output.stdout).into_owned()
    } else {
        String::from_utf8_lossy(&output.stderr).into_owned()
    };

    let version = text
        .split_whitespace()
        .find(|part| part.chars().next().is_some_and(|ch| ch.is_ascii_digit()))?;
    let mut parts = version.split('.');
    let major = parts.next()?.parse().ok()?;
    let minor = parts.next()?.parse().ok()?;
    Some((major, minor))
}

fn run_command(command: &mut Command, description: &str) -> Result<()> {
    let output = command
        .output()
        .with_context(|| format!("Failed to {}", description))?;
    if output.status.success() {
        return Ok(());
    }

    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();
    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let summary = summarize_command_output(&stderr, &stdout);

    if !stderr.is_empty() {
        debug!("[cohere-runtime][stderr] {}", stderr);
    }
    if !stdout.is_empty() {
        debug!("[cohere-runtime][stdout] {}", stdout);
    }

    let mut message = format!(
        "{} failed{}",
        description,
        output
            .status
            .code()
            .map(|code| format!(" with exit code {}", code))
            .unwrap_or_default()
    );
    if let Some(summary) = summary {
        message.push_str(&format!(". {}", summary));
    }
    if description == "install Cohere runtime dependencies" {
        message.push_str(" See the terminal logs for full pip output.");
    }
    anyhow::bail!(message)
}

fn summarize_command_output(stderr: &str, stdout: &str) -> Option<String> {
    for source in [stderr, stdout] {
        if let Some(line) = source
            .lines()
            .find(|line| line.trim_start().starts_with("ERROR:"))
        {
            return Some(line.trim().to_string());
        }
    }

    for source in [stderr, stdout] {
        if let Some(line) = source.lines().rev().find(|line| !line.trim().is_empty()) {
            let line = line.trim();
            let shortened: String = line.chars().take(220).collect();
            if shortened.len() < line.len() {
                return Some(format!("{}...", shortened));
            }
            return Some(shortened);
        }
    }

    None
}

fn write_json_line(stdin: &mut ChildStdin, value: &Value) -> Result<()> {
    let payload = serde_json::to_string(value)?;
    stdin.write_all(payload.as_bytes())?;
    stdin.write_all(b"\n")?;
    stdin.flush()?;
    Ok(())
}

fn write_audio_request(stdin: &mut ChildStdin, header: &Value, audio: &[f32]) -> Result<()> {
    let payload = serde_json::to_string(header)?;
    stdin.write_all(payload.as_bytes())?;
    stdin.write_all(b"\n")?;

    #[cfg(target_endian = "little")]
    {
        // Handy already holds 16 kHz mono f32 samples in memory, so on
        // little-endian targets we can stream the raw bytes directly.
        let bytes = unsafe {
            std::slice::from_raw_parts(audio.as_ptr() as *const u8, std::mem::size_of_val(audio))
        };
        stdin.write_all(bytes)?;
    }

    #[cfg(not(target_endian = "little"))]
    {
        for sample in audio {
            stdin.write_all(&sample.to_le_bytes())?;
        }
    }

    stdin.flush()?;
    Ok(())
}

fn read_json_line(stdout: &mut BufReader<ChildStdout>) -> Result<Value> {
    let mut line = String::new();
    let read = stdout.read_line(&mut line)?;
    if read == 0 {
        anyhow::bail!("Unexpected EOF from Cohere worker");
    }
    let trimmed = line.trim();
    serde_json::from_str(trimmed)
        .with_context(|| format!("Failed to parse Cohere worker response: {}", trimmed))
}

fn spawn_stderr_logger(stderr: ChildStderr) {
    thread::spawn(move || {
        let reader = BufReader::new(stderr);
        for line in reader.lines() {
            match line {
                Ok(line) if !line.trim().is_empty() => debug!("[cohere-worker] {}", line),
                Ok(_) => {}
                Err(err) => {
                    warn!("Failed to read Cohere worker stderr: {}", err);
                    break;
                }
            }
        }
    });
}
