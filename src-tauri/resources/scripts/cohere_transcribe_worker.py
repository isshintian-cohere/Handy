#!/usr/bin/env python3

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def log(message: str) -> None:
    sys.stderr.write(message + "\n")
    sys.stderr.flush()


def choose_device() -> str:
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_for_device(device: str, load_model):
    model = load_model()
    model.to(device)
    model.eval()
    return model


def load_runtime_native(model_dir: str):
    from transformers import CohereAsrForConditionalGeneration

    processor = AutoProcessor.from_pretrained(model_dir)
    preferred_device = choose_device()

    def construct_model():
        # MPS half precision overflows on this model during generation.
        # Stick to float32 for macOS so transcription remains reliable.
        return CohereAsrForConditionalGeneration.from_pretrained(
            model_dir,
            dtype=torch.float32,
        )

    try:
        model = load_for_device(preferred_device, construct_model)
        return processor, model, preferred_device, "native"
    except Exception:
        if preferred_device != "cpu":
            log(
                "Falling back to CPU for native Cohere ASR after failure on "
                f"{preferred_device}:\n{traceback.format_exc()}"
            )
            model = load_for_device("cpu", construct_model)
            return processor, model, "cpu", "native"
        raise


def load_runtime_compat(model_dir: str):
    processor = AutoProcessor.from_pretrained(model_dir, trust_remote_code=True)
    preferred_device = choose_device()

    def construct_model():
        return AutoModelForSpeechSeq2Seq.from_pretrained(
            model_dir,
            dtype=torch.float32,
            trust_remote_code=True,
        )

    try:
        model = load_for_device(preferred_device, construct_model)
        return processor, model, preferred_device, "trust_remote_code"
    except Exception:
        if preferred_device != "cpu":
            log(
                "Falling back to CPU for compatibility mode after failure on "
                f"{preferred_device}:\n{traceback.format_exc()}"
            )
            model = load_for_device("cpu", construct_model)
            return processor, model, "cpu", "trust_remote_code"
        raise


def load_runtime(model_dir: str):
    try:
        return load_runtime_native(model_dir)
    except Exception:
        log(
            "Native Cohere ASR path unavailable; falling back to trust_remote_code:\n"
            f"{traceback.format_exc()}"
        )
        return load_runtime_compat(model_dir)


def move_inputs_to_device(inputs: dict, model):
    prepared = {}
    model_dtype = getattr(model, "dtype", None)

    for key, value in inputs.items():
        if key == "audio_chunk_index":
            continue
        if hasattr(value, "to"):
            if getattr(value, "is_floating_point", lambda: False)() and model_dtype is not None:
                prepared[key] = value.to(model.device, dtype=model_dtype)
            else:
                prepared[key] = value.to(model.device)
        else:
            prepared[key] = value

    return prepared


def transcribe_native(processor, model, audio: np.ndarray, sample_rate: int, language: str) -> str:
    inputs = processor(
        audio=audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        language=language,
    )
    audio_chunk_index = inputs.get("audio_chunk_index")
    prepared_inputs = move_inputs_to_device(inputs, model)

    with torch.inference_mode():
        outputs = model.generate(**prepared_inputs, max_new_tokens=256)
        if hasattr(outputs, "cpu"):
            outputs = outputs.cpu()

    if audio_chunk_index is not None:
        decoded = processor.decode(
            outputs,
            skip_special_tokens=True,
            audio_chunk_index=audio_chunk_index,
            language=language,
        )
    else:
        decoded = processor.decode(outputs, skip_special_tokens=True)

    if isinstance(decoded, list):
        if not decoded:
            return ""
        return str(decoded[0]).strip()

    return str(decoded).strip()


def transcribe_compat(processor, model, audio: np.ndarray, sample_rate: int, language: str) -> str:
    with torch.inference_mode():
        texts = model.transcribe(
            processor=processor,
            audio_arrays=[audio],
            sample_rates=[sample_rate],
            language=language,
        )

    if isinstance(texts, list):
        if not texts:
            return ""
        return str(texts[0]).strip()

    return str(texts).strip()


def transcribe(processor, model, audio: np.ndarray, sample_rate: int, language: str, backend: str) -> str:
    if backend == "native":
        return transcribe_native(processor, model, audio, sample_rate, language)
    return transcribe_compat(processor, model, audio, sample_rate, language)


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


def handle_loop(processor, model, backend: str) -> None:
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
            text = transcribe(processor, model, audio, sample_rate, language, backend)
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
        processor, model, device, backend = load_runtime(model_dir)
        emit(
            {
                "status": "ready",
                "device": device,
                "backend": backend,
                "model_dir": model_dir,
            }
        )
        handle_loop(processor, model, backend)
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
