from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable


LOCAL_MODELS_ENV = "WORLDMM_HF_MODELS_DIR"

MODEL_PATH_ALIASES = {
    "Qwen/Qwen3-Embedding-4B": ("Qwen/Qwen3-Embedding-4B", "Qwen3-Embedding-4B", "Qwen"),
    "VLM2Vec/VLM2Vec-V2.0": ("VLM2Vec/VLM2Vec-V2.0", "VLM2Vec-V2.0", "VLM2Vec"),
    "Qwen/Qwen3-VL-2B-Instruct": ("Qwen/Qwen3-VL-2B-Instruct", "Qwen3-VL-2B-Instruct"),
    "Qwen/Qwen3-VL-4B-Instruct": ("Qwen/Qwen3-VL-4B-Instruct", "Qwen3-VL-4B-Instruct"),
    "Qwen/Qwen3-VL-8B-Instruct": ("Qwen/Qwen3-VL-8B-Instruct", "Qwen3-VL-8B-Instruct"),
}

WHISPER_PATH_ALIASES = {
    "distil-large-v3.5": (
        "distil-whisper/distil-large-v3.5-ct2",
        "distil-large-v3.5-ct2",
        "distil-whisper",
    ),
    "distil-whisper/distil-large-v3.5-ct2": (
        "distil-whisper/distil-large-v3.5-ct2",
        "distil-large-v3.5-ct2",
        "distil-whisper",
    ),
}


def _has_required_files(path: Path, required_files: Iterable[str] | None) -> bool:
    if required_files is None:
        return True
    return all((path / file_name).exists() for file_name in required_files)


def resolve_local_model_path(
    model_name: str,
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
    required_files: Iterable[str] | None = None,
    models_dir_env: str = LOCAL_MODELS_ENV,
) -> str:
    """Resolve a Hugging Face model id to a local directory when configured.

    Set WORLDMM_HF_MODELS_DIR to a directory containing local model downloads.
    If no matching local path exists, the original model name is returned so the
    normal Hugging Face lookup behavior is preserved.
    """
    model_path = Path(model_name).expanduser()
    if model_path.exists() and _has_required_files(model_path, required_files):
        return str(model_path)

    models_root_value = os.getenv(models_dir_env)
    if not models_root_value:
        return model_name

    models_root = Path(models_root_value).expanduser()
    if not models_root.exists():
        return model_name

    names: list[str] = []
    if aliases and model_name in aliases:
        names.extend(aliases[model_name])
    names.extend(
        [
            model_name,
            model_name.split("/")[-1],
            model_name.replace("/", "--"),
        ]
    )

    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        candidate = models_root / name
        if candidate.exists() and _has_required_files(candidate, required_files):
            return str(candidate)

    return model_name


def resolve_hf_model_path(model_name: str) -> str:
    return resolve_local_model_path(model_name, aliases=MODEL_PATH_ALIASES)


def resolve_whisper_model_path(model_name: str) -> str:
    return resolve_local_model_path(
        model_name,
        aliases=WHISPER_PATH_ALIASES,
        required_files=("model.bin",),
    )
