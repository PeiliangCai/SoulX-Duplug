from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


DEFAULT_DATA_ROOT = Path("/root/SoulX-Duplug/datasets")


@dataclass
class DatasetCheck:
    name: str
    ok: bool
    message: str


def _load_state(dataset_dir: Path) -> dict:
    state_path = dataset_dir / ".download-state.json"
    if not state_path.exists():
        return {}
    try:
        return json.loads(state_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def check_aishell1(data_root: Path) -> list[DatasetCheck]:
    dataset_dir = data_root / "aishell1"
    state = _load_state(dataset_dir)
    checks = [
        DatasetCheck("aishell1.state", bool(state.get("files", {}).get("data")), "data_aishell.tgz recorded complete"),
        DatasetCheck("aishell1.resource", bool(state.get("files", {}).get("resource")), "resource_aishell.tgz recorded complete"),
    ]
    base = dataset_dir / "extracted" / "data_aishell"
    transcript = base / "transcript" / "aishell_transcript_v0.8.txt"
    checks.append(DatasetCheck("aishell1.transcript", transcript.exists(), str(transcript)))
    wav_dir = base / "wav"
    inner_archives = list(wav_dir.glob("S*.tar.gz")) if wav_dir.exists() else []
    wav_count = sum(1 for _ in wav_dir.rglob("*.wav")) if wav_dir.exists() else 0
    checks.append(DatasetCheck("aishell1.inner_archives", len(inner_archives) == 0, f"{len(inner_archives)} nested archives remaining"))
    checks.append(DatasetCheck("aishell1.wav", wav_count > 0, f"{wav_count} wav files found"))
    return checks


def check_aishell3(data_root: Path) -> list[DatasetCheck]:
    dataset_dir = data_root / "aishell3"
    state = _load_state(dataset_dir)
    base = dataset_dir / "extracted"
    train_content = base / "train" / "content.txt"
    test_content = base / "test" / "content.txt"
    wav_count = sum(1 for _ in base.rglob("*.wav")) if base.exists() else 0
    return [
        DatasetCheck("aishell3.state", bool(state.get("files", {}).get("data")), "data_aishell3.tgz recorded complete"),
        DatasetCheck("aishell3.train_content", train_content.exists(), str(train_content)),
        DatasetCheck("aishell3.test_content", test_content.exists(), str(test_content)),
        DatasetCheck("aishell3.wav", wav_count > 0, f"{wav_count} wav files found"),
    ]


def check_generic_dataset(data_root: Path, dataset: str) -> list[DatasetCheck]:
    dataset_dir = data_root / dataset
    alt_dir = data_root / dataset.replace("-", "_")
    roots = [path for path in (dataset_dir, alt_dir) if path.exists()]
    if dataset in {"commonvoice-cn", "commonvoice-en", "emilia-cn", "emilia-en"}:
        voxbox = data_root / "voxbox"
        if voxbox.exists():
            roots.append(voxbox)
    root_exists = bool(roots)
    audio_count = 0
    metadata_count = 0
    for root in roots:
        audio_count += sum(1 for path in root.rglob("*") if path.suffix.lower() in {".wav", ".flac", ".mp3", ".opus", ".ogg", ".m4a"})
        metadata_count += sum(1 for path in root.rglob("*") if path.suffix.lower() in {".json", ".jsonl", ".tsv", ".gz"})
    return [
        DatasetCheck(f"{dataset}.dir", root_exists, ", ".join(str(root) for root in roots) if roots else str(dataset_dir)),
        DatasetCheck(f"{dataset}.metadata", metadata_count > 0, f"{metadata_count} metadata files found"),
        DatasetCheck(f"{dataset}.audio", audio_count > 0, f"{audio_count} audio files found"),
    ]


def generic_checker(dataset: str):
    def _check(data_root: Path) -> list[DatasetCheck]:
        return check_generic_dataset(data_root, dataset)

    return _check


CHECKERS = {
    "aishell1": check_aishell1,
    "aishell3": check_aishell3,
    "wenetspeech": generic_checker("wenetspeech"),
    "commonvoice-cn": generic_checker("commonvoice-cn"),
    "emilia-cn": generic_checker("emilia-cn"),
    "magicdata": generic_checker("magicdata"),
    "librispeech": generic_checker("librispeech"),
    "gigaspeech": generic_checker("gigaspeech"),
    "commonvoice-en": generic_checker("commonvoice-en"),
    "emilia-en": generic_checker("emilia-en"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify local dataset download/extraction state.")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--datasets", default="aishell1,aishell3")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero if any check fails.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    failed = False
    for dataset in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        if dataset not in CHECKERS:
            raise ValueError(f"unsupported checker: {dataset}")
        for check in CHECKERS[dataset](args.data_root):
            status = "ok" if check.ok else "fail"
            print(f"[{status}] {check.name}: {check.message}")
            failed = failed or not check.ok
    if failed and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
