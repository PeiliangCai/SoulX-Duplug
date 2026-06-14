from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def resolve_path(value: str | Path, *, base_dir: str | Path | None = None) -> Path:
    text = str(value)
    text = text.replace("${DATA_ROOT}", str(Path(os.environ.get("DATA_ROOT", "/root/SoulX-Duplug/datasets"))))
    text = text.replace("${MODEL_ROOT}", str(Path(os.environ.get("MODEL_ROOT", "/root/autodl-tmp/models"))))
    text = text.replace("${CACHE_ROOT}", str(Path(os.environ.get("CACHE_ROOT", "/root/autodl-tmp/cache"))))
    text = text.replace("${OUTPUT_ROOT}", str(Path(os.environ.get("OUTPUT_ROOT", "/root/SoulX-Duplug/outputs"))))
    path = Path(os.path.expandvars(os.path.expanduser(text)))
    if not path.is_absolute() and base_dir is not None:
        path = Path(base_dir) / path
    return path


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return data


def save_yaml(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result
