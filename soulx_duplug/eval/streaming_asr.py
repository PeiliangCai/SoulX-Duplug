from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from soulx_duplug.config import load_yaml, resolve_path
from soulx_duplug.data.stage2_chunks import ASR_EOS_TOKEN, read_stage2_manifest
from soulx_duplug.models.speech_tokenizer import build_speech_tokenizer
from soulx_duplug.models.streaming_asr_model import build_interleaved_asr_model
from soulx_duplug.models.text_tokenizer import load_text_tokenizer
from soulx_duplug.train.stage2_streaming_asr import (
    Stage2StreamingAsrDataset,
    collate_stage2,
    evaluate_loader,
)


def evaluate_checkpoint(
    checkpoint: Path,
    manifest: Path,
    *,
    limit: int | None = None,
    device_name: str | None = None,
) -> dict[str, float]:
    checkpoint = resolve_path(checkpoint)
    manifest = resolve_path(manifest)
    config = load_yaml(checkpoint / "config.yaml")
    tokenizer = load_text_tokenizer(checkpoint / "text_tokenizer.json")
    asr_eos_id = tokenizer.ensure_token(ASR_EOS_TOKEN)
    speech_tokenizer = build_speech_tokenizer(config.get("tokenizer", {}))
    records = list(read_stage2_manifest(manifest))
    if limit is not None:
        records = records[:limit]

    audio_cfg = config.get("audio", {})
    data_cfg = config.get("data", {})
    dataset = Stage2StreamingAsrDataset(
        records,
        speech_tokenizer=speech_tokenizer,
        text_tokenizer=tokenizer,
        asr_eos_id=asr_eos_id,
        target_sample_rate=int(audio_cfg.get("target_sample_rate", 16000)),
        chunk_seconds=float(audio_cfg.get("chunk_seconds", 0.16)),
        lookback_seconds=float(audio_cfg.get("lookback_seconds", 0.96)),
        lookahead_seconds=float(audio_cfg.get("lookahead_seconds", 0.04)),
        max_sequence_length=data_cfg.get("max_sequence_length"),
    )
    train_cfg = config.get("training", {})
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg.get("eval_batch_size", 1)),
        shuffle=False,
        collate_fn=lambda samples: collate_stage2(samples, pad_id=tokenizer.pad_id),
    )
    selected_device = device_name or config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(selected_device)
    model = build_interleaved_asr_model(
        config.get("model", {}),
        audio_vocab_size=speech_tokenizer.vocab_size,
        text_vocab_size=tokenizer.vocab_size,
    ).to(device)
    state = torch.load(checkpoint / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state["model"])
    return evaluate_loader(
        model,
        loader,
        tokenizer,
        device=device,
        asr_eos_id=asr_eos_id,
        max_new_tokens_per_chunk=int(train_cfg.get("max_new_tokens_per_chunk", 16)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 2 streaming ASR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        args.checkpoint,
        args.manifest,
        limit=args.limit,
        device_name=args.device,
    )
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
