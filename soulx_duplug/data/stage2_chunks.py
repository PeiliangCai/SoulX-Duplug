from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

from soulx_duplug.data.audio import read_audio
from soulx_duplug.data.manifest import Stage1Record, read_manifest


ASR_EOS_TOKEN = "<asr_eos>"


@dataclass(frozen=True)
class AlignmentToken:
    text: str
    start: float
    end: float
    confidence: float | None = None


@dataclass(frozen=True)
class Stage2Chunk:
    index: int
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Stage2Record:
    utt_id: str
    dataset: str
    split: str
    lang: str
    audio_path: str
    text: str
    duration: float
    chunks: list[Stage2Chunk]
    speaker_id: str = ""
    segment_start: float | None = None
    segment_end: float | None = None


def _token_from_obj(obj: dict) -> AlignmentToken:
    text = str(obj.get("text", obj.get("char", obj.get("token", ""))))
    if not text:
        raise ValueError(f"alignment token has no text: {obj}")
    return AlignmentToken(
        text=text,
        start=float(obj["start"]),
        end=float(obj["end"]),
        confidence=float(obj["confidence"]) if obj.get("confidence") is not None else None,
    )


def read_alignment_file(path: str | Path) -> dict[str, list[AlignmentToken]]:
    alignments: dict[str, list[AlignmentToken]] = {}
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            utt_id = str(data["utt_id"])
            if "tokens" in data:
                tokens = [_token_from_obj(item) for item in data["tokens"]]
                alignments[utt_id] = tokens
            else:
                alignments.setdefault(utt_id, []).append(_token_from_obj(data))
    for utt_id, tokens in alignments.items():
        alignments[utt_id] = sorted(tokens, key=lambda item: (item.start, item.end))
    return alignments


def write_alignment_file(alignments: dict[str, list[AlignmentToken]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for utt_id, tokens in alignments.items():
            f.write(
                json.dumps(
                    {"utt_id": utt_id, "tokens": [asdict(token) for token in tokens]},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def infer_duration(record: Stage1Record) -> float:
    duration = float(record.duration or 0.0)
    if duration > 0.0:
        return duration
    if record.segment_start is not None and record.segment_end is not None:
        return max(0.0, float(record.segment_end) - float(record.segment_start))
    audio = read_audio(record.normalized_path or record.audio_path, ffmpeg_sample_rate=16000)
    return audio.duration


def make_uniform_alignment(record: Stage1Record, duration: float | None = None) -> list[AlignmentToken]:
    duration = float(duration if duration is not None else record.duration or 0.0)
    text = record.text or ""
    if duration <= 0.0 or not text:
        return []
    step = duration / max(1, len(text))
    tokens = []
    for idx, char in enumerate(text):
        tokens.append(AlignmentToken(text=char, start=idx * step, end=(idx + 1) * step))
    return tokens


def tokens_to_chunks(tokens: list[AlignmentToken], duration: float, *, chunk_seconds: float = 0.16) -> list[Stage2Chunk]:
    num_chunks = max(1, int(math.ceil(duration / chunk_seconds)))
    buckets = ["" for _ in range(num_chunks)]
    for token in tokens:
        center = max(0.0, (token.start + token.end) / 2.0)
        index = min(num_chunks - 1, int(center // chunk_seconds))
        buckets[index] += token.text
    chunks = []
    for index, text in enumerate(buckets):
        start = index * chunk_seconds
        end = min(duration, (index + 1) * chunk_seconds)
        chunks.append(Stage2Chunk(index=index, start=start, end=end, text=text))
    return chunks


def build_stage2_records(
    manifest: str | Path,
    *,
    alignment_path: str | Path | None = None,
    chunk_seconds: float = 0.16,
    allow_uniform_fallback: bool = False,
    limit: int | None = None,
) -> list[Stage2Record]:
    alignments = read_alignment_file(alignment_path) if alignment_path else {}
    records = []
    missing_alignment = 0
    missing_duration = 0
    for record in read_manifest(manifest):
        if limit is not None and len(records) >= limit:
            break
        tokens = alignments.get(record.utt_id)
        if tokens is None:
            if not allow_uniform_fallback:
                missing_alignment += 1
                continue
        try:
            duration = infer_duration(record)
        except Exception as exc:
            missing_duration += 1
            print(f"[warn] failed to infer duration for {record.utt_id}: {exc}")
            continue
        if duration <= 0.0:
            missing_duration += 1
            continue
        if tokens is None:
            tokens = make_uniform_alignment(record, duration=duration)
        records.append(
            Stage2Record(
                utt_id=record.utt_id,
                dataset=record.dataset,
                split=record.split,
                lang=record.lang,
                audio_path=record.normalized_path or record.audio_path,
                text=record.text,
                duration=duration,
                chunks=tokens_to_chunks(tokens, duration, chunk_seconds=chunk_seconds),
                speaker_id=record.speaker_id,
                segment_start=None if record.normalized_path else record.segment_start,
                segment_end=None if record.normalized_path else record.segment_end,
            )
        )
    if missing_alignment:
        print(f"[warn] skipped {missing_alignment} records without alignment")
    if missing_duration:
        print(f"[warn] skipped {missing_duration} records without duration")
    return records


def read_stage2_manifest(path: str | Path) -> Iterator[Stage2Record]:
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data["chunks"] = [Stage2Chunk(**chunk) for chunk in data["chunks"]]
            try:
                yield Stage2Record(**data)
            except TypeError as exc:
                raise ValueError(f"invalid Stage 2 manifest line {line_no}: {exc}") from exc


def write_stage2_manifest(records: Iterable[Stage2Record], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            data = asdict(record)
            f.write(json.dumps(data, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Stage 2 chunk-aligned streaming ASR manifest.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--alignment", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--chunk-seconds", type=float, default=0.16)
    parser.add_argument("--allow-uniform-fallback", action="store_true")
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = build_stage2_records(
        args.manifest,
        alignment_path=args.alignment,
        chunk_seconds=args.chunk_seconds,
        allow_uniform_fallback=args.allow_uniform_fallback,
        limit=args.limit,
    )
    count = write_stage2_manifest(records, args.out)
    print(f"[write] {args.out}: {count}")


if __name__ == "__main__":
    main()
