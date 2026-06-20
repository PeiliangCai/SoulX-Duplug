from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from soulx_duplug.config import load_yaml, resolve_path
from soulx_duplug.data.manifest import read_manifest
from soulx_duplug.metrics import cer, wer
from soulx_duplug.models.speech_tokenizer import build_speech_tokenizer
from soulx_duplug.models.stage1_model import build_stage1_model
from soulx_duplug.models.text_tokenizer import CharTokenizer, load_text_tokenizer
from soulx_duplug.train.stage1_asr import Stage1AsrDataset, collate_stage1, move_batch


def evaluate_checkpoint(checkpoint: Path, manifest: Path, *, limit: int | None = None) -> dict[str, float]:
    checkpoint = resolve_path(checkpoint)
    manifest = resolve_path(manifest)
    config = load_yaml(checkpoint / "config.yaml")
    tokenizer_path = checkpoint / "text_tokenizer.json"
    tokenizer = load_text_tokenizer(tokenizer_path) if tokenizer_path.exists() else CharTokenizer.load(checkpoint / "char_vocab.json")
    speech_tokenizer = build_speech_tokenizer(config.get("tokenizer", {}))
    records = list(read_manifest(manifest))
    if limit is not None:
        records = records[:limit]
    dataset = Stage1AsrDataset(
        records,
        speech_tokenizer=speech_tokenizer,
        text_tokenizer=tokenizer,
        target_sample_rate=int(config.get("audio", {}).get("target_sample_rate", 16000)),
        max_audio_tokens=config.get("data", {}).get("max_audio_tokens"),
        max_text_tokens=config.get("data", {}).get("max_text_tokens"),
    )
    loader = DataLoader(dataset, batch_size=int(config.get("training", {}).get("eval_batch_size", 2)), shuffle=False, collate_fn=lambda samples: (collate_stage1(samples, tokenizer.pad_id), samples))
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    model = build_stage1_model(
        config.get("model", {}),
        audio_vocab_size=speech_tokenizer.vocab_size,
        text_vocab_size=tokenizer.vocab_size,
    ).to(device)
    state = torch.load(checkpoint / "pytorch_model.bin", map_location=device)
    model.load_state_dict(state["model"])
    model.eval()

    zh_scores: list[float] = []
    en_scores: list[float] = []
    losses: list[float] = []
    eos_terminated = 0
    with torch.no_grad():
        for batch, samples in loader:
            batch = move_batch(batch, device)
            output = model(
                audio_tokens=batch.audio_tokens,
                audio_lengths=batch.audio_lengths,
                decoder_input_ids=batch.decoder_input_ids,
                labels=batch.labels,
            )
            if "loss" in output:
                losses.append(float(output["loss"].detach().cpu()))
            generated = model.generate(
                audio_tokens=batch.audio_tokens,
                audio_lengths=batch.audio_lengths,
                bos_id=tokenizer.bos_id,
                eos_id=tokenizer.eos_id,
                max_new_tokens=int(config.get("training", {}).get("max_new_tokens", 128)),
            )
            for ids, sample in zip(generated.cpu().tolist(), samples):
                eos_terminated += int(tokenizer.eos_id in ids[1:])
                hyp = tokenizer.decode(ids)
                if sample.lang == "en":
                    en_scores.append(wer(sample.text, hyp))
                else:
                    zh_scores.append(cer(sample.text, hyp))
    metrics = {}
    if losses:
        metrics["loss"] = sum(losses) / len(losses)
    if zh_scores:
        metrics["cer_zh"] = sum(zh_scores) / len(zh_scores)
    if en_scores:
        metrics["wer_en"] = sum(en_scores) / len(en_scores)
    decoded_samples = len(zh_scores) + len(en_scores)
    if decoded_samples:
        metrics["decode_eos_rate"] = eos_terminated / decoded_samples
        metrics["decoded_samples"] = float(decoded_samples)
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 1 ASR checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(args.checkpoint, args.manifest, limit=args.limit)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")


if __name__ == "__main__":
    main()
