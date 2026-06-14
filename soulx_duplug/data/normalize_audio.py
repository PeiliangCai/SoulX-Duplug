from __future__ import annotations

import argparse
from pathlib import Path

from soulx_duplug.data.audio import normalize_audio_file
from soulx_duplug.data.manifest import Stage1Record, read_manifest, write_manifest


def normalized_path_for(record: Stage1Record, output_root: Path) -> Path:
    return output_root / record.dataset / record.split / f"{record.utt_id}.wav"


def normalize_manifest(
    manifest_path: Path,
    output_root: Path,
    *,
    out_manifest: Path | None = None,
    target_sample_rate: int = 16000,
    force: bool = False,
    limit: int | None = None,
    strict: bool = False,
) -> int:
    normalized: list[Stage1Record] = []
    failures = 0
    for idx, record in enumerate(read_manifest(manifest_path), start=1):
        if limit is not None and len(normalized) >= limit:
            break
        target_path = normalized_path_for(record, output_root)
        try:
            metadata = normalize_audio_file(
                record.audio_path,
                target_path,
                target_sample_rate=target_sample_rate,
                force=force,
                segment_start=record.segment_start,
                segment_end=record.segment_end,
            )
        except Exception as exc:
            failures += 1
            message = f"[warn] normalize failed for {record.utt_id}: {exc}"
            if strict:
                raise RuntimeError(message) from exc
            print(message)
            continue
        normalized.append(
            record.with_audio_info(
                duration=metadata.duration,
                sample_rate=metadata.sample_rate,
                channels=metadata.channels,
                normalized_path=str(target_path),
            )
        )
        if idx % 1000 == 0:
            print(f"[progress] processed {idx} rows")

    out_manifest = out_manifest or manifest_path.with_name(f"{manifest_path.stem}.normalized.jsonl")
    count = write_manifest(normalized, out_manifest)
    print(f"[write] {out_manifest}: {count} records; failures={failures}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize Stage 1 manifest audio to 16 kHz mono PCM WAV.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path)
    parser.add_argument("--output-root", type=Path, default=Path("/root/autodl-tmp/datasets/normalized/stage1"))
    parser.add_argument("--target-sample-rate", type=int, default=16000)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalize_manifest(
        args.manifest,
        args.output_root,
        out_manifest=args.out_manifest,
        target_sample_rate=args.target_sample_rate,
        force=args.force,
        limit=args.limit,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
