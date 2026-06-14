from __future__ import annotations

import argparse
from pathlib import Path

from soulx_duplug.data.manifest import Stage1Record, read_manifest
from soulx_duplug.data.stage2_chunks import AlignmentToken, write_alignment_file


def _tokens_from_whisperx_result(result: dict, *, prefer_chars: bool = False) -> list[AlignmentToken]:
    items = []
    if prefer_chars:
        items = list(result.get("char_segments") or [])
        if not items:
            for segment in result.get("segments") or []:
                items.extend(segment.get("chars") or [])
    if not items:
        items = list(result.get("word_segments") or [])
        if not items:
            for segment in result.get("segments") or []:
                items.extend(segment.get("words") or [])

    tokens: list[AlignmentToken] = []
    for item in items:
        raw_text = str(item.get("char") or item.get("word") or item.get("text") or "")
        text = raw_text if prefer_chars else raw_text.strip()
        if not text or item.get("start") is None or item.get("end") is None:
            continue
        if not prefer_chars:
            text = f"{text} "
        tokens.append(
            AlignmentToken(
                text=text,
                start=float(item["start"]),
                end=float(item["end"]),
                confidence=float(item["score"]) if item.get("score") is not None else None,
            )
        )
    return tokens


def _record_audio_and_duration(record: Stage1Record, whisperx_module) -> tuple[object, float]:
    audio = whisperx_module.load_audio(record.normalized_path or record.audio_path)
    sample_rate = 16000
    if record.normalized_path is None and record.segment_start is not None and record.segment_end is not None:
        start = max(0, int(float(record.segment_start) * sample_rate))
        end = max(start, int(float(record.segment_end) * sample_rate))
        audio = audio[start:end]
    duration = float(record.duration or 0.0)
    if duration <= 0.0 and record.segment_start is not None and record.segment_end is not None:
        duration = max(0.0, float(record.segment_end) - float(record.segment_start))
    if duration <= 0.0:
        duration = len(audio) / sample_rate
    return audio, duration


def generate_alignments(
    manifest: Path,
    out: Path,
    *,
    language: str = "en",
    device: str = "cuda",
    align_model: str | None = None,
    limit: int | None = None,
    prefer_chars: bool = False,
    progress_every: int = 100,
) -> int:
    try:
        import whisperx  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "WhisperX alignment generation requires whisperx. "
            "Install it or pass an external --alignment file to Stage 2 manifest generation."
        ) from exc

    align_kwargs = {"language_code": language, "device": device}
    if align_model:
        align_kwargs["model_name"] = align_model
    model, metadata = whisperx.load_align_model(**align_kwargs)
    alignments: dict[str, list[AlignmentToken]] = {}
    records = [record for record in read_manifest(manifest) if record.lang == language]
    if limit is not None:
        records = records[:limit]
    for idx, record in enumerate(records, start=1):
        audio, duration = _record_audio_and_duration(record, whisperx)
        if duration <= 0.0:
            print(f"[warn] no duration for {record.utt_id}")
            continue
        segments = [{"start": 0.0, "end": duration, "text": record.text}]
        result = whisperx.align(
            segments,
            model,
            metadata,
            audio,
            device,
            return_char_alignments=prefer_chars,
        )
        tokens = _tokens_from_whisperx_result(result, prefer_chars=prefer_chars)
        if not tokens:
            print(f"[warn] no WhisperX alignment tokens for {record.utt_id}")
            continue
        alignments[record.utt_id] = tokens
        if progress_every > 0 and idx % progress_every == 0:
            print(f"[progress] aligned {idx}/{len(records)}")
    return write_alignment_file(alignments, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate English word-level alignments with WhisperX.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--align-model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--prefer-chars", action="store_true", help="Ask WhisperX for char alignments and emit chars when available.")
    parser.add_argument("--progress-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = generate_alignments(
        args.manifest,
        args.out,
        language=args.language,
        device=args.device,
        align_model=args.align_model,
        limit=args.limit,
        prefer_chars=args.prefer_chars,
        progress_every=args.progress_every,
    )
    print(f"[write] {args.out}: {count}")


if __name__ == "__main__":
    main()
