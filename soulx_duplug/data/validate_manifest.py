from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from soulx_duplug.data.manifest import read_manifest


def validate_manifest(path: Path, *, prefer_normalized: bool = True) -> None:
    counts: Counter[str] = Counter()
    durations: Counter[str] = Counter()
    missing = 0
    empty_text = 0
    total_duration = 0.0
    for record in read_manifest(path):
        counts[f"split:{record.split}"] += 1
        counts[f"dataset:{record.dataset}"] += 1
        counts[f"lang:{record.lang}"] += 1
        if record.sample_rate is not None:
            counts[f"sample_rate:{record.sample_rate}"] += 1
        if record.channels is not None:
            counts[f"channels:{record.channels}"] += 1
        if record.duration:
            total_duration += record.duration
        audio_path = Path(record.normalized_path or record.audio_path) if prefer_normalized else Path(record.audio_path)
        if not audio_path.exists():
            missing += 1
        if not record.text:
            empty_text += 1
    print(f"manifest: {path}")
    print(f"records: {sum(v for k, v in counts.items() if k.startswith('split:'))}")
    print(f"hours: {total_duration / 3600.0:.3f}")
    print(f"missing_audio: {missing}")
    print(f"empty_text: {empty_text}")
    for key, value in sorted(counts.items()):
        print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate Stage 1 manifest files.")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--raw-audio", action="store_true", help="Validate audio_path instead of normalized_path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_manifest(args.manifest, prefer_normalized=not args.raw_audio)


if __name__ == "__main__":
    main()
