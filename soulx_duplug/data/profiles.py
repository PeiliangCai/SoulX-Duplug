from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from soulx_duplug.config import load_yaml


@dataclass(frozen=True)
class DatasetProfileEntry:
    dataset_id: str
    enabled: bool = True
    lang: str | None = None
    provider: str | None = None
    subset: str | None = None
    download: dict[str, Any] = field(default_factory=dict)
    manifest: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DatasetProfile:
    name: str
    datasets: list[DatasetProfileEntry]
    dedup: str = "source_text"
    target_hours: dict[str, float] = field(default_factory=dict)


def _entry_from_obj(obj: Any) -> DatasetProfileEntry:
    if isinstance(obj, str):
        return DatasetProfileEntry(dataset_id=obj)
    if not isinstance(obj, dict):
        raise ValueError(f"dataset profile entry must be string or mapping, got {type(obj).__name__}")
    dataset_id = obj.get("id") or obj.get("dataset") or obj.get("name")
    if not dataset_id:
        raise ValueError(f"dataset profile entry has no id: {obj}")
    return DatasetProfileEntry(
        dataset_id=str(dataset_id),
        enabled=bool(obj.get("enabled", True)),
        lang=str(obj["lang"]) if obj.get("lang") is not None else None,
        provider=str(obj["provider"]) if obj.get("provider") is not None else None,
        subset=str(obj["subset"]) if obj.get("subset") is not None else None,
        download=dict(obj.get("download") or {}),
        manifest=dict(obj.get("manifest") or {}),
    )


def load_dataset_profile(path: str | Path) -> DatasetProfile:
    data = load_yaml(path)
    entries = [_entry_from_obj(item) for item in data.get("datasets", [])]
    if not entries:
        raise ValueError(f"dataset profile has no datasets: {path}")
    target_hours = {
        str(lang): float(hours)
        for lang, hours in dict(data.get("target_hours") or {}).items()
        if hours is not None
    }
    return DatasetProfile(
        name=str(data.get("name") or Path(path).stem),
        datasets=entries,
        dedup=str(data.get("dedup", "source_text")),
        target_hours=target_hours,
    )


def enabled_dataset_ids(profile: DatasetProfile) -> list[str]:
    return [
        entry.dataset_id
        for entry in profile.datasets
        if entry.enabled and bool(entry.manifest.get("enabled", True))
    ]
