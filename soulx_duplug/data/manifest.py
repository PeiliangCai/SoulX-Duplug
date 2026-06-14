from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator


@dataclass(frozen=True)
class Stage1Record:
    utt_id: str
    dataset: str
    split: str
    lang: str
    audio_path: str
    text: str
    speaker_id: str = ""
    duration: float | None = None
    sample_rate: int | None = None
    channels: int | None = None
    normalized_path: str | None = None
    segment_start: float | None = None
    segment_end: float | None = None

    def with_split(self, split: str) -> "Stage1Record":
        return replace(self, split=split)

    def with_audio_info(
        self,
        *,
        duration: float | None,
        sample_rate: int | None,
        channels: int | None,
        normalized_path: str | None = None,
        segment_start: float | None = None,
        segment_end: float | None = None,
    ) -> "Stage1Record":
        return replace(
            self,
            duration=duration,
            sample_rate=sample_rate,
            channels=channels,
            normalized_path=normalized_path if normalized_path is not None else self.normalized_path,
            segment_start=segment_start if segment_start is not None else self.segment_start,
            segment_end=segment_end if segment_end is not None else self.segment_end,
        )


def stable_score(value: str) -> float:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()
    return int(digest[:12], 16) / float(16**12)


def coarse_split(split: str) -> str:
    lowered = split.lower()
    if lowered.startswith(("dev", "valid", "val")):
        return "dev"
    if lowered.startswith("test"):
        return "test"
    return "train"


def read_manifest(path: str | Path) -> Iterator[Stage1Record]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                yield Stage1Record(**data)
            except Exception as exc:  # pragma: no cover - message path matters more.
                raise ValueError(f"invalid manifest line {line_no} in {path}: {exc}") from exc


def write_manifest(records: Iterable[Stage1Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def assign_dev_split(records: list[Stage1Record], dev_ratio: float) -> list[Stage1Record]:
    if dev_ratio <= 0:
        return records

    existing_dev_by_dataset = {record.dataset for record in records if record.split == "dev"}
    result: list[Stage1Record] = []
    for record in records:
        if record.split != "train" or record.dataset in existing_dev_by_dataset:
            result.append(record)
            continue
        key = record.speaker_id or record.utt_id
        if stable_score(f"{record.dataset}:{key}") < dev_ratio:
            result.append(record.with_split("dev"))
        else:
            result.append(record)
    return result


def group_by_split(records: Iterable[Stage1Record]) -> dict[str, list[Stage1Record]]:
    grouped: dict[str, list[Stage1Record]] = {"train": [], "dev": [], "test": []}
    for record in records:
        grouped.setdefault(record.split, []).append(record)
    return grouped
