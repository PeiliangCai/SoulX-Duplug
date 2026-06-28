from __future__ import annotations

import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from soulx_duplug.config import load_yaml, resolve_path, save_yaml
from soulx_duplug.data.audio import read_audio_segment, resample_linear
from soulx_duplug.data.manifest import Stage1Record, read_manifest
from soulx_duplug.distributed_utils import (
    barrier,
    cleanup_distributed,
    distributed_summary,
    make_distributed_sampler,
    maybe_no_sync,
    rank_log_path,
    reduce_mean,
    setup_distributed,
    unwrap_model,
    wrap_model_for_distributed,
)
from soulx_duplug.logging_utils import (
    cuda_memory_summary,
    log_event,
    log_exception,
    parameter_summary,
    record_summary,
    runtime_summary,
    setup_train_logger,
)
from soulx_duplug.metrics import cer, wer
from soulx_duplug.models.speech_tokenizer import SpeechTokenizerBackend, build_speech_tokenizer
from soulx_duplug.models.stage1_model import Stage1Batch, build_stage1_model
from soulx_duplug.models.text_tokenizer import CharTokenizer, HfTextTokenizer, TextTokenizer
from soulx_duplug.train.checkpoint_utils import load_training_checkpoint, resolve_resume_checkpoint
from soulx_duplug.training_curves import TrainingCurveTracker


@dataclass
class PreparedSample:
    audio_tokens: torch.LongTensor
    target_ids: list[int]
    text: str
    lang: str


class Stage1AsrDataset(Dataset[PreparedSample]):
    def __init__(
        self,
        records: list[Stage1Record],
        *,
        speech_tokenizer: SpeechTokenizerBackend,
        text_tokenizer: TextTokenizer,
        target_sample_rate: int = 16000,
        max_audio_tokens: int | None = None,
        max_text_tokens: int | None = None,
    ) -> None:
        self.records = records
        self.speech_tokenizer = speech_tokenizer
        self.text_tokenizer = text_tokenizer
        self.target_sample_rate = target_sample_rate
        self.max_audio_tokens = max_audio_tokens
        self.max_text_tokens = max_text_tokens

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> PreparedSample:
        record = self.records[idx]
        path = record.normalized_path or record.audio_path
        audio = read_audio_segment(
            path,
            start=None if record.normalized_path else record.segment_start,
            end=None if record.normalized_path else record.segment_end,
            ffmpeg_sample_rate=self.target_sample_rate,
        )
        waveform = resample_linear(audio.waveform, audio.sample_rate, self.target_sample_rate)
        audio_tokens = self.speech_tokenizer.encode(waveform, self.target_sample_rate)
        if self.max_audio_tokens is not None:
            audio_tokens = audio_tokens[: self.max_audio_tokens]
        target_ids = self.text_tokenizer.encode(record.text, add_special_tokens=True)
        if self.max_text_tokens is not None:
            target_ids = target_ids[: self.max_text_tokens]
            if target_ids[-1] != self.text_tokenizer.eos_id:
                target_ids[-1] = self.text_tokenizer.eos_id
        return PreparedSample(audio_tokens=audio_tokens, target_ids=target_ids, text=record.text, lang=record.lang)


def collate_stage1(samples: list[PreparedSample], pad_id: int) -> Stage1Batch:
    max_audio = max(sample.audio_tokens.numel() for sample in samples)
    max_target = max(len(sample.target_ids) for sample in samples)
    audio_tokens = torch.zeros((len(samples), max_audio), dtype=torch.long)
    audio_lengths = torch.zeros(len(samples), dtype=torch.long)
    decoder_input_ids = torch.full((len(samples), max_target - 1), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), max_target - 1), -100, dtype=torch.long)
    for row, sample in enumerate(samples):
        audio_len = sample.audio_tokens.numel()
        audio_tokens[row, :audio_len] = sample.audio_tokens
        audio_lengths[row] = audio_len
        target = torch.tensor(sample.target_ids, dtype=torch.long)
        decoder = target[:-1]
        label = target[1:]
        decoder_input_ids[row, : decoder.numel()] = decoder
        labels[row, : label.numel()] = label
    return Stage1Batch(
        audio_tokens=audio_tokens,
        audio_lengths=audio_lengths,
        decoder_input_ids=decoder_input_ids,
        labels=labels,
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_records(path: str | Path, limit: int | None = None) -> list[Stage1Record]:
    records = list(read_manifest(path))
    if limit is not None:
        return records[:limit]
    return records


def build_text_tokenizer(config: dict[str, Any], train_records: list[Stage1Record], vocab_path: Path | None = None) -> TextTokenizer:
    if vocab_path and vocab_path.exists():
        return CharTokenizer.load(vocab_path)
    model_cfg = config.get("model", {})
    if model_cfg.get("backend") == "qwen":
        return HfTextTokenizer(str(model_cfg["model_name_or_path"]))
    tokenizer = CharTokenizer()
    tokenizer.fit([record.text for record in train_records])
    return tokenizer


def move_batch(batch: Stage1Batch, device: torch.device) -> Stage1Batch:
    return Stage1Batch(
        audio_tokens=batch.audio_tokens.to(device),
        audio_lengths=batch.audio_lengths.to(device),
        decoder_input_ids=batch.decoder_input_ids.to(device),
        labels=batch.labels.to(device),
    )


def select_batch(batch: Stage1Batch, indices: list[int]) -> Stage1Batch:
    rows = torch.tensor(indices, dtype=torch.long, device=batch.audio_tokens.device)
    return Stage1Batch(
        audio_tokens=batch.audio_tokens.index_select(0, rows),
        audio_lengths=batch.audio_lengths.index_select(0, rows),
        decoder_input_ids=batch.decoder_input_ids.index_select(0, rows),
        labels=batch.labels.index_select(0, rows),
    )


def make_eval_loader(dataset: Stage1AsrDataset, *, batch_size: int, pad_id: int) -> DataLoader:
    def collate_with_refs(samples: list[PreparedSample]) -> tuple[Stage1Batch, list[str], list[str]]:
        return collate_stage1(samples, pad_id), [sample.text for sample in samples], [sample.lang for sample in samples]

    return DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_with_refs)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    dataloader: DataLoader,
    tokenizer: TextTokenizer,
    *,
    device: torch.device,
    max_new_tokens: int,
    max_decode_samples_per_language: int | None = None,
    logger: Any | None = None,
    step: int | None = None,
    log_examples: int = 3,
) -> dict[str, float]:
    model.eval()
    losses: list[float] = []
    zh_scores: list[float] = []
    en_scores: list[float] = []
    decoded_by_language = {"zh": 0, "en": 0}
    eos_terminated = 0
    examples_logged = 0
    for batch, refs, langs in dataloader:
        batch = move_batch(batch, device)
        output = model(
            audio_tokens=batch.audio_tokens,
            audio_lengths=batch.audio_lengths,
            decoder_input_ids=batch.decoder_input_ids,
            labels=batch.labels,
        )
        if "loss" in output:
            losses.append(float(output["loss"].detach().cpu()))
        decode_indices = []
        for index, lang in enumerate(langs):
            language = "en" if lang == "en" else "zh"
            if (
                max_decode_samples_per_language is None
                or decoded_by_language[language] < max_decode_samples_per_language
            ):
                decode_indices.append(index)
                decoded_by_language[language] += 1
        if not decode_indices:
            continue
        decode_batch = select_batch(batch, decode_indices)
        generated = model.generate(
            audio_tokens=decode_batch.audio_tokens,
            audio_lengths=decode_batch.audio_lengths,
            bos_id=tokenizer.bos_id,
            eos_id=tokenizer.eos_id,
            max_new_tokens=max_new_tokens,
        )
        selected_refs = [refs[index] for index in decode_indices]
        selected_langs = [langs[index] for index in decode_indices]
        for ids, ref, lang in zip(generated.cpu().tolist(), selected_refs, selected_langs):
            eos_terminated += int(tokenizer.eos_id in ids[1:])
            hyp = tokenizer.decode(ids)
            if logger is not None and examples_logged < log_examples:
                log_event(
                    logger,
                    "eval_prediction",
                    stage="stage1",
                    step=step,
                    lang=lang,
                    reference=ref,
                    hypothesis=hyp,
                    ended_with_eos=tokenizer.eos_id in ids[1:],
                )
                examples_logged += 1
            if lang == "en":
                en_scores.append(wer(ref, hyp))
            else:
                zh_scores.append(cer(ref, hyp))
    metrics: dict[str, float] = {}
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
        metrics["decoded_zh_samples"] = float(len(zh_scores))
        metrics["decoded_en_samples"] = float(len(en_scores))
    return metrics


def save_checkpoint(
    *,
    output_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    tokenizer: TextTokenizer,
    config: dict[str, Any],
    step: int,
    logger: Any | None = None,
) -> Path:
    checkpoint_dir = output_dir / f"step-{step}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": unwrap_model(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
        },
        checkpoint_dir / "pytorch_model.bin",
    )
    tokenizer.save(checkpoint_dir / "text_tokenizer.json")
    if isinstance(tokenizer, CharTokenizer):
        tokenizer.save(checkpoint_dir / "char_vocab.json")
    save_yaml(config, checkpoint_dir / "config.yaml")
    update_latest_checkpoint(output_dir, checkpoint_dir, logger=logger)
    return checkpoint_dir


def update_latest_checkpoint(checkpoint_root: Path, checkpoint_dir: Path, *, logger: Any | None = None) -> None:
    latest = checkpoint_root / "latest"
    tmp = checkpoint_root / ".latest.tmp"
    if tmp.exists() or tmp.is_symlink():
        tmp.unlink()
    if latest.exists() and not latest.is_symlink():
        message = f"latest checkpoint path exists and is not a symlink: {latest}"
        if logger is not None:
            log_event(logger, "checkpoint_latest_update_skipped", level=30, path=str(latest), reason=message)
        else:
            print(f"[warn] {message}")
        return
    tmp.symlink_to(checkpoint_dir.name, target_is_directory=True)
    os.replace(tmp, latest)


def train(config: dict[str, Any], log_file: str | Path | None = None) -> Path:
    context = setup_distributed(config.get("device"))
    device = context.device
    output_dir = resolve_path(config.get("output_dir", "outputs/stage1"))
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = rank_log_path(resolve_path(log_file or config.get("log_file") or output_dir / "log.txt"), context)
    logger = setup_train_logger(f"soulx_duplug.stage1.rank{context.rank}", log_path)

    try:
        seed = int(config.get("seed", 1337))
        set_seed(seed)
        log_event(
            logger,
            "train_start",
            stage="stage1",
            seed=seed,
            output_dir=str(output_dir),
            checkpoint_dir=str(checkpoint_dir),
            log_file=str(log_path),
        )
        log_event(logger, "distributed_ready", **distributed_summary(context))
        log_event(logger, "config", config=config)
        log_event(logger, "runtime", **runtime_summary(device))

        data_cfg = config["data"]
        train_manifest = resolve_path(data_cfg["train_manifest"])
        dev_manifest = resolve_path(data_cfg["dev_manifest"]) if data_cfg.get("dev_manifest") else None
        log_event(
            logger,
            "data_paths",
            train_manifest=str(train_manifest),
            dev_manifest=str(dev_manifest) if dev_manifest else None,
        )
        train_records = load_records(train_manifest, data_cfg.get("limit_train_samples"))
        dev_records = load_records(dev_manifest, data_cfg.get("limit_dev_samples")) if dev_manifest else []
        log_event(logger, "manifest_loaded", split="train", **record_summary(train_records))
        if dev_records:
            log_event(logger, "manifest_loaded", split="dev", **record_summary(dev_records))
        if not train_records:
            raise RuntimeError("train manifest has no records")

        text_tokenizer = build_text_tokenizer(config, train_records)
        speech_tokenizer = build_speech_tokenizer(config.get("tokenizer", {}))
        log_event(
            logger,
            "tokenizers_ready",
            text_tokenizer=type(text_tokenizer).__name__,
            text_vocab_size=text_tokenizer.vocab_size,
            speech_tokenizer=type(speech_tokenizer).__name__,
            speech_vocab_size=speech_tokenizer.vocab_size,
            tokenizer_backend=config.get("tokenizer", {}).get("backend", "dummy"),
        )
        model = build_stage1_model(
            config.get("model", {}),
            audio_vocab_size=speech_tokenizer.vocab_size,
            text_vocab_size=text_tokenizer.vocab_size,
        ).to(device)
        log_event(
            logger,
            "model_ready",
            model_backend=config.get("model", {}).get("backend", "dummy"),
            model_class=type(model).__name__,
            **parameter_summary(model),
            **cuda_memory_summary(device),
        )
        model = wrap_model_for_distributed(model, context)

        audio_cfg = config.get("audio", {})
        target_sample_rate = int(audio_cfg.get("target_sample_rate", 16000))
        train_dataset = Stage1AsrDataset(
            train_records,
            speech_tokenizer=speech_tokenizer,
            text_tokenizer=text_tokenizer,
            target_sample_rate=target_sample_rate,
            max_audio_tokens=data_cfg.get("max_audio_tokens"),
            max_text_tokens=data_cfg.get("max_text_tokens"),
        )
        dev_dataset = Stage1AsrDataset(
            dev_records,
            speech_tokenizer=speech_tokenizer,
            text_tokenizer=text_tokenizer,
            target_sample_rate=target_sample_rate,
            max_audio_tokens=data_cfg.get("max_audio_tokens"),
            max_text_tokens=data_cfg.get("max_text_tokens"),
        ) if dev_records else None

        train_cfg = config.get("training", {})
        batch_size = int(train_cfg.get("batch_size", 2))
        train_sampler = make_distributed_sampler(train_dataset, context, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda samples: collate_stage1(samples, text_tokenizer.pad_id),
        )
        dev_loader = make_eval_loader(
            dev_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
            pad_id=text_tokenizer.pad_id,
        ) if dev_dataset and context.is_main else None

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        )
        max_steps = int(train_cfg.get("max_steps", 1000))
        grad_accum = int(train_cfg.get("gradient_accumulation_steps", 1))
        log_every = int(train_cfg.get("log_every", 10))
        eval_every = int(train_cfg.get("eval_every", 100))
        save_every = int(train_cfg.get("save_every", 500))
        max_new_tokens = int(train_cfg.get("max_new_tokens", 128))
        eval_decode_samples_per_language_value = train_cfg.get("eval_decode_samples_per_language")
        eval_decode_samples_per_language = (
            int(eval_decode_samples_per_language_value)
            if eval_decode_samples_per_language_value is not None
            else None
        )
        eval_log_examples = max(0, int(train_cfg.get("eval_log_examples", 3)))
        plot_every = max(0, int(train_cfg.get("plot_every", 100)))
        plot_smoothing_window = max(1, int(train_cfg.get("plot_smoothing_window", 20)))
        resume_checkpoint = resolve_resume_checkpoint(
            checkpoint_dir,
            train_cfg,
            logger=logger,
            stage="stage1",
            is_main=context.is_main,
        )
        start_step = 0
        if resume_checkpoint is not None:
            start_step = load_training_checkpoint(
                checkpoint_dir=resume_checkpoint,
                model=model,
                optimizer=optimizer,
                device=device,
                logger=logger,
                stage="stage1",
                is_main=context.is_main,
            )
        barrier(context)
        curve_tracker = TrainingCurveTracker(
            output_dir=output_dir,
            stage="stage1",
            logger=logger,
            plot_every=plot_every,
            smoothing_window=plot_smoothing_window,
        ) if context.is_main else None
        log_event(
            logger,
            "training_ready",
            batch_size=batch_size,
            eval_batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
            gradient_accumulation_steps=grad_accum,
            world_size=context.world_size,
            effective_batch_size=batch_size * grad_accum * context.world_size,
            learning_rate=float(train_cfg.get("learning_rate", 1e-4)),
            weight_decay=float(train_cfg.get("weight_decay", 0.0)),
            max_steps=max_steps,
            log_every=log_every,
            eval_every=eval_every,
            save_every=save_every,
            max_new_tokens=max_new_tokens,
            eval_decode_samples_per_language=eval_decode_samples_per_language,
            eval_log_examples=eval_log_examples,
            plot_every=plot_every,
            plot_smoothing_window=plot_smoothing_window,
            resume_from_checkpoint=str(resume_checkpoint) if resume_checkpoint else None,
            start_step=start_step,
        )

        step = start_step
        epoch = 0
        last_checkpoint = resume_checkpoint or checkpoint_dir
        while step < max_steps:
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            for batch in train_loader:
                model.train()
                batch = move_batch(batch, device)
                sync_gradients = (step + 1) % grad_accum == 0
                with maybe_no_sync(model, context, sync_gradients=sync_gradients):
                    output = model(
                        audio_tokens=batch.audio_tokens,
                        audio_lengths=batch.audio_lengths,
                        decoder_input_ids=batch.decoder_input_ids,
                        labels=batch.labels,
                    )
                    raw_loss = output["loss"]
                    loss = raw_loss / grad_accum
                    loss.backward()
                if sync_gradients:
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                step += 1
                if step % log_every == 0:
                    train_loss = reduce_mean(raw_loss, context)
                    if context.is_main:
                        log_event(logger, "train_step", step=step, train_loss=train_loss, **cuda_memory_summary(device))
                        if curve_tracker is not None:
                            curve_tracker.record_train(step=step, train_loss=train_loss)
                if dev_dataset and step % eval_every == 0:
                    barrier(context)
                    if context.is_main and dev_loader is not None:
                        metrics = evaluate_loader(
                            unwrap_model(model),
                            dev_loader,
                            text_tokenizer,
                            device=device,
                            max_new_tokens=max_new_tokens,
                            max_decode_samples_per_language=eval_decode_samples_per_language,
                            logger=logger,
                            step=step,
                            log_examples=eval_log_examples,
                        )
                        log_event(logger, "eval", step=step, **metrics, **cuda_memory_summary(device))
                        if curve_tracker is not None:
                            curve_tracker.record_eval(step=step, metrics=metrics)
                            curve_tracker.plot(step=step, reason="eval")
                    barrier(context)
                elif context.is_main and curve_tracker is not None and curve_tracker.should_plot(step):
                    curve_tracker.plot(step=step, reason="periodic")
                if step % save_every == 0:
                    if context.is_main:
                        last_checkpoint = save_checkpoint(
                            output_dir=checkpoint_dir,
                            model=model,
                            optimizer=optimizer,
                            tokenizer=text_tokenizer,
                            config=config,
                            step=step,
                            logger=logger,
                        )
                        log_event(logger, "checkpoint_saved", step=step, checkpoint=str(last_checkpoint))
                    barrier(context)
                if step >= max_steps:
                    break
            epoch += 1

        final_checkpoint = checkpoint_dir / f"step-{step}"
        if context.is_main:
            if last_checkpoint != final_checkpoint:
                last_checkpoint = save_checkpoint(
                    output_dir=checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    tokenizer=text_tokenizer,
                    config=config,
                    step=step,
                    logger=logger,
                )
                log_event(logger, "checkpoint_saved", step=step, checkpoint=str(last_checkpoint))
            else:
                log_event(logger, "checkpoint_already_saved", step=step, checkpoint=str(last_checkpoint))
            if curve_tracker is not None:
                curve_tracker.plot(step=step, reason="train_complete")
            log_event(logger, "train_complete", stage="stage1", step=step, checkpoint=str(last_checkpoint), **cuda_memory_summary(device))
        barrier(context)
        return final_checkpoint
    except Exception as exc:
        log_exception(logger, "stage1", exc, device)
        raise
    finally:
        cleanup_distributed(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 1 non-streaming ASR.")
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_asr.yaml"))
    parser.add_argument("--log-file", type=Path, help="Write training logs to this file. Defaults to <output_dir>/log.txt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_yaml(args.config), log_file=args.log_file)


if __name__ == "__main__":
    main()
