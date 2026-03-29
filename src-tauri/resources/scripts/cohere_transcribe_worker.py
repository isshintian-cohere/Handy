#!/usr/bin/env python3

import argparse
import json
import os
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def _patch_cohere_audio_frontend() -> None:
    """Replace CohereAudioFrontend.load_buffers_from_checkpoint with an
    mx.load()-based version so that PyTorch is not required at runtime.

    See: https://github.com/Blaizzy/mlx-audio/pull/605
    Patch PR in review: https://github.com/Blaizzy/mlx-audio/pull/616
    """
    try:
        import mlx.core as mx
        from mlx_audio.stt.models.cohere_asr.audio import CohereAudioFrontend
    except ImportError:
        return

    def _load_buffers(self, model_path):
        safetensor_path = Path(model_path) / "model.safetensors"
        if not safetensor_path.exists():
            return

        weights = mx.load(str(safetensor_path))

        fb_key = "preprocessor.featurizer.fb"
        if fb_key in weights:
            fb = weights[fb_key]
            if fb.ndim == 3:
                fb = fb.squeeze(0)
            self.fb = fb.astype(mx.float32)

        win_key = "preprocessor.featurizer.window"
        if win_key in weights:
            self.window = weights[win_key].astype(mx.float32)

    CohereAudioFrontend.load_buffers_from_checkpoint = _load_buffers


def load_runtime(model_dir: str):
    _patch_cohere_audio_frontend()

    from mlx_audio.stt.utils import load

    model = load(model_dir)
    return model


def transcribe(model, audio: np.ndarray, sample_rate: int, language: str) -> str:
    import soundfile as sf

    fd, tmp_path = tempfile.mkstemp(suffix=".wav")
    try:
        os.close(fd)
        sf.write(tmp_path, audio, sample_rate)
        result = model.generate(tmp_path, language=language, verbose=False)
        return result.text.strip()
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def read_exact(stream, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError(
                f"Unexpected EOF while reading audio payload: expected {size} bytes, got {len(chunks)}"
            )
        chunks.extend(chunk)
    return bytes(chunks)


def handle_loop(model) -> None:
    stdin = sys.stdin.buffer

    while True:
        raw_line = stdin.readline()
        if not raw_line:
            return

        line = raw_line.decode("utf-8").strip()
        if not line:
            continue

        try:
            payload = json.loads(line)
        except json.JSONDecodeError as err:
            emit({"status": "error", "error": f"Invalid JSON request: {err}"})
            continue

        command = payload.get("command")
        if command == "shutdown":
            emit({"status": "ok"})
            return

        if command != "transcribe":
            emit({"status": "error", "error": f"Unknown command: {command}"})
            continue

        language = payload.get("language")
        sample_rate = payload.get("sample_rate")
        audio_bytes_len = payload.get("audio_bytes_len")
        if not language or sample_rate is None or audio_bytes_len is None:
            emit(
                {
                    "status": "error",
                    "error": "Transcribe request requires language, sample_rate, and audio_bytes_len",
                }
            )
            continue

        try:
            sample_rate = int(sample_rate)
            audio_bytes_len = int(audio_bytes_len)
            raw_audio = read_exact(stdin, audio_bytes_len)
            audio = np.frombuffer(raw_audio, dtype="<f4").copy()
            text = transcribe(model, audio, sample_rate, language)
            emit({"status": "ok", "text": text})
        except Exception as err:
            emit(
                {
                    "status": "error",
                    "error": f"{err}",
                    "traceback": traceback.format_exc(),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    args = parser.parse_args()

    model_dir = str(Path(args.model_dir).expanduser().resolve())
    try:
        model = load_runtime(model_dir)
        emit(
            {
                "status": "ready",
                "device": "metal",
                "backend": "mlx-audio",
                "model_dir": model_dir,
            }
        )
        handle_loop(model)
        return 0
    except Exception as err:
        emit(
            {
                "status": "error",
                "error": str(err),
                "traceback": traceback.format_exc(),
            }
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
