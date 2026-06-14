from __future__ import annotations

import argparse
from pathlib import Path

from soulx_duplug.data.manifest import read_manifest
from soulx_duplug.data.stage2_chunks import AlignmentToken, write_alignment_file


DEFAULT_MODEL = "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch"


def _tokens_from_funasr_result(result: dict) -> list[AlignmentToken]:
    text = str(result.get("text", "")).replace(" ", "")
    timestamps = result.get("timestamp") or result.get("timestamps") or []
    tokens: list[AlignmentToken] = []
    if timestamps and text:
        for idx, item in enumerate(timestamps[: len(text)]):
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                start, end = float(item[0]) / 1000.0, float(item[1]) / 1000.0
            elif isinstance(item, dict):
                start, end = float(item["start"]) / 1000.0, float(item["end"]) / 1000.0
            else:
                continue
            tokens.append(AlignmentToken(text=text[idx], start=start, end=end))
    return tokens


def generate_alignments(
    manifest: Path,
    out: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    language: str = "zh",
    limit: int | None = None,
    batch_size: int = 1,
) -> int:
    try:
        from funasr import AutoModel  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Paraformer alignment generation requires funasr/modelscope. "
            "Install optional ASR dependencies or pass external --alignment to Stage 2."
        ) from exc

    model = AutoModel(model=model_name)
    alignments: dict[str, list[AlignmentToken]] = {}
    records = [record for record in read_manifest(manifest) if record.lang == language]
    if limit is not None:
        records = records[:limit]
    for idx, record in enumerate(records, start=1):
        audio_path = record.normalized_path or record.audio_path
        result = model.generate(input=audio_path, batch_size=batch_size)
        first = result[0] if isinstance(result, list) and result else result
        if not isinstance(first, dict):
            print(f"[warn] unexpected FunASR result for {record.utt_id}: {type(first)}")
            continue
        tokens = _tokens_from_funasr_result(first)
        if not tokens:
            print(f"[warn] no timestamp tokens for {record.utt_id}")
            continue
        alignments[record.utt_id] = tokens
        if idx % 100 == 0:
            print(f"[progress] aligned {idx}/{len(records)}")
    return write_alignment_file(alignments, out)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Mandarin char-level alignments with Paraformer/FunASR.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--language", default="zh")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    count = generate_alignments(
        args.manifest,
        args.out,
        model_name=args.model,
        language=args.language,
        limit=args.limit,
        batch_size=args.batch_size,
    )
    print(f"[write] {args.out}: {count}")


if __name__ == "__main__":
    main()
