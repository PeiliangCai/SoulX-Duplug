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
from soulx_duplug.data.stage2_chunks import ASR_EOS_TOKEN, Stage2Record, read_stage2_manifest
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
    stage2_chunk_summary,
)
from soulx_duplug.metrics import cer, wer
from soulx_duplug.models.speech_tokenizer import SpeechTokenizerBackend, build_speech_tokenizer
from soulx_duplug.models.streaming_asr_model import InterleavedBatch, build_interleaved_asr_model
from soulx_duplug.models.text_tokenizer import CharTokenizer, TextTokenizer, load_text_tokenizer
from soulx_duplug.train.checkpoint_utils import load_training_checkpoint, resolve_resume_checkpoint
from soulx_duplug.train.stage1_asr import build_text_tokenizer
from soulx_duplug.training_curves import TrainingCurveTracker


@dataclass
class Stage2Sample:
    audio_token_ids: torch.LongTensor
    text_input_ids: torch.LongTensor
    modality: torch.LongTensor
    labels: torch.LongTensor
    text: str
    lang: str


class Stage2StreamingAsrDataset(Dataset[Stage2Sample]):
    def __init__(
        self,
        records: list[Stage2Record],
        *,
        speech_tokenizer: SpeechTokenizerBackend,
        text_tokenizer: TextTokenizer,
        asr_eos_id: int,
        target_sample_rate: int = 16000,
        chunk_seconds: float = 0.16,
        lookback_seconds: float = 0.96,
        lookahead_seconds: float = 0.04,
        max_sequence_length: int | None = None,
    ) -> None:
        self.records = records
        self.speech_tokenizer = speech_tokenizer
        self.text_tokenizer = text_tokenizer
        self.asr_eos_id = asr_eos_id
        self.target_sample_rate = target_sample_rate
        self.chunk_seconds = chunk_seconds
        self.lookback_seconds = lookback_seconds
        self.lookahead_seconds = lookahead_seconds
        self.max_sequence_length = max_sequence_length

    def __len__(self) -> int:
        return len(self.records)

    def _chunk_audio_tokens(self, record: Stage2Record) -> list[torch.LongTensor]:
        audio = read_audio_segment(
            record.audio_path,
            start=record.segment_start,
            end=record.segment_end,
            ffmpeg_sample_rate=self.target_sample_rate,
        )
        waveform = resample_linear(audio.waveform, audio.sample_rate, self.target_sample_rate)
        chunks = self.speech_tokenizer.encode_streaming_chunks(
            waveform,
            self.target_sample_rate,
            chunk_seconds=self.chunk_seconds,
            lookback_seconds=self.lookback_seconds,
            lookahead_seconds=self.lookahead_seconds,
        )
        if len(chunks) < len(record.chunks):
            pad = torch.zeros_like(chunks[-1]) if chunks else torch.zeros(2, dtype=torch.long)
            chunks.extend([pad] * (len(record.chunks) - len(chunks)))
        return chunks[: len(record.chunks)]

    def __getitem__(self, idx: int) -> Stage2Sample:
        record = self.records[idx]
        audio_chunks = self._chunk_audio_tokens(record)
        audio_ids: list[int] = []
        text_inputs: list[int] = []
        modality: list[int] = []
        labels: list[int] = []
        included_text: list[str] = []
        for chunk, chunk_audio_tokens in zip(record.chunks, audio_chunks):
            chunk_audio_ids = [int(token_id) for token_id in chunk_audio_tokens.tolist()]
            target = self.text_tokenizer.encode(chunk.text, add_special_tokens=False) + [self.asr_eos_id]
            decoder_inputs = [self.text_tokenizer.bos_id] + target[:-1]
            chunk_sequence_length = len(chunk_audio_ids) + len(decoder_inputs) + 1
            if (
                self.max_sequence_length is not None
                and len(audio_ids) + chunk_sequence_length > self.max_sequence_length
            ):
                if audio_ids:
                    break
                raise RuntimeError(
                    f"Stage 2 chunk exceeds max_sequence_length: "
                    f"utt_id={record.utt_id} chunk={chunk.index} "
                    f"length={chunk_sequence_length} limit={self.max_sequence_length}"
                )
            for token_id in chunk_audio_ids:
                audio_ids.append(token_id)
                text_inputs.append(self.text_tokenizer.pad_id)
                modality.append(1)
                labels.append(-100)
            for decoder_id, label_id in zip(decoder_inputs, target):
                audio_ids.append(0)
                text_inputs.append(int(decoder_id))
                modality.append(0)
                labels.append(int(label_id))
            audio_ids.append(0)
            text_inputs.append(self.asr_eos_id)
            modality.append(0)
            labels.append(-100)
            included_text.append(chunk.text)
        return Stage2Sample(
            audio_token_ids=torch.tensor(audio_ids, dtype=torch.long),
            text_input_ids=torch.tensor(text_inputs, dtype=torch.long),
            modality=torch.tensor(modality, dtype=torch.long),
            labels=torch.tensor(labels, dtype=torch.long),
            text="".join(included_text),
            lang=record.lang,
        )


def collate_stage2(samples: list[Stage2Sample], *, pad_id: int) -> tuple[InterleavedBatch, list[str], list[str]]:
    max_len = max(sample.labels.numel() for sample in samples)
    audio_token_ids = torch.zeros((len(samples), max_len), dtype=torch.long)
    text_input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
    modality = torch.zeros((len(samples), max_len), dtype=torch.long)
    attention_mask = torch.zeros((len(samples), max_len), dtype=torch.long)
    labels = torch.full((len(samples), max_len), -100, dtype=torch.long)
    for row, sample in enumerate(samples):
        length = sample.labels.numel()
        audio_token_ids[row, :length] = sample.audio_token_ids
        text_input_ids[row, :length] = sample.text_input_ids
        modality[row, :length] = sample.modality
        labels[row, :length] = sample.labels
        attention_mask[row, :length] = 1
    return (
        InterleavedBatch(
            audio_token_ids=audio_token_ids,
            text_input_ids=text_input_ids,
            modality=modality,
            attention_mask=attention_mask,
            labels=labels,
        ),
        [sample.text for sample in samples],
        [sample.lang for sample in samples],
    )


def move_interleaved_batch(batch: InterleavedBatch, device: torch.device) -> InterleavedBatch:
    return InterleavedBatch(
        audio_token_ids=batch.audio_token_ids.to(device),
        text_input_ids=batch.text_input_ids.to(device),
        modality=batch.modality.to(device),
        attention_mask=batch.attention_mask.to(device),
        labels=batch.labels.to(device),
    )


def select_interleaved_batch(batch: InterleavedBatch, indices: list[int]) -> InterleavedBatch:
    rows = torch.tensor(indices, dtype=torch.long, device=batch.audio_token_ids.device)
    return InterleavedBatch(
        audio_token_ids=batch.audio_token_ids.index_select(0, rows),
        text_input_ids=batch.text_input_ids.index_select(0, rows),
        modality=batch.modality.index_select(0, rows),
        attention_mask=batch.attention_mask.index_select(0, rows),
        labels=batch.labels.index_select(0, rows),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _records_as_stage1_like(records: list[Stage2Record]):
    from soulx_duplug.data.manifest import Stage1Record

    return [
        Stage1Record(
            utt_id=record.utt_id,
            dataset=record.dataset,
            split=record.split,
            lang=record.lang,
            audio_path=record.audio_path,
            text=record.text,
            speaker_id=record.speaker_id,
            duration=record.duration,
        )
        for record in records
    ]


def build_or_load_text_tokenizer(config: dict[str, Any], train_records: list[Stage2Record]) -> TextTokenizer:
    stage1_checkpoint = config.get("stage1_checkpoint")
    if stage1_checkpoint:
        checkpoint = resolve_path(stage1_checkpoint)
        tokenizer_path = checkpoint / "text_tokenizer.json"
        if tokenizer_path.exists():
            tokenizer = load_text_tokenizer(tokenizer_path)
        elif (checkpoint / "char_vocab.json").exists():
            tokenizer = CharTokenizer.load(checkpoint / "char_vocab.json")
        else:
            tokenizer = build_text_tokenizer(config, _records_as_stage1_like(train_records))
    else:
        tokenizer = build_text_tokenizer(config, _records_as_stage1_like(train_records))
    tokenizer.ensure_token(ASR_EOS_TOKEN)
    return tokenizer


def load_stage1_weights_if_available(
    model: torch.nn.Module,
    checkpoint: str | None,
    device: torch.device,
    *,
    logger: Any | None = None,
) -> None:
    if not checkpoint:
        return
    state_path = resolve_path(checkpoint) / "pytorch_model.bin"
    if not state_path.exists():
        message = f"Stage 1 checkpoint missing: {state_path}"
        if logger is not None:
            log_event(logger, "stage1_checkpoint_missing", level=30, checkpoint=str(state_path), message=message)
        else:
            print(f"[warn] {message}")
        return
    state = torch.load(state_path, map_location=device)
    source_state = state.get("model", state)
    target_state = model.state_dict()
    compatible_state: dict[str, torch.Tensor] = {}
    resized_keys: list[str] = []
    skipped_keys: list[str] = []
    unexpected_source_keys = 0
    for key, source_value in source_state.items():
        target_value = target_state.get(key)
        if target_value is None:
            unexpected_source_keys += 1
            continue
        if source_value.shape == target_value.shape:
            compatible_state[key] = source_value
            continue
        if (
            source_value.ndim == target_value.ndim
            and source_value.shape[1:] == target_value.shape[1:]
            and source_value.shape[0] < target_value.shape[0]
        ):
            resized_value = target_value.clone()
            resized_value[: source_value.shape[0]].copy_(
                source_value.to(device=resized_value.device, dtype=resized_value.dtype)
            )
            compatible_state[key] = resized_value
            resized_keys.append(key)
            continue
        skipped_keys.append(
            f"{key}:source={tuple(source_value.shape)} target={tuple(target_value.shape)}"
        )
    missing, unexpected = model.load_state_dict(compatible_state, strict=False)
    if logger is not None:
        log_event(
            logger,
            "stage1_checkpoint_loaded",
            checkpoint=str(state_path),
            missing=len(missing),
            unexpected=len(unexpected) + unexpected_source_keys,
            resized_keys=resized_keys,
            skipped_keys=skipped_keys,
        )
    else:
        print(
            f"[init] loaded Stage 1 checkpoint; missing={len(missing)} "
            f"unexpected={len(unexpected) + unexpected_source_keys} "
            f"resized={len(resized_keys)} skipped={len(skipped_keys)}"
        )


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    tokenizer: TextTokenizer,
    *,
    device: torch.device,
    asr_eos_id: int,
    max_new_tokens_per_chunk: int,
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
    decoded_chunks = 0
    eos_chunks = 0
    truncated_chunks = 0
    examples_logged = 0
    for batch, refs, langs in loader:
        batch = move_interleaved_batch(batch, device)
        output = model(batch)
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
        decode_batch = select_interleaved_batch(batch, decode_indices)
        generations = model.generate_streaming(
            decode_batch,
            bos_id=tokenizer.bos_id,
            asr_eos_id=asr_eos_id,
            max_new_tokens_per_chunk=max_new_tokens_per_chunk,
        )
        selected_refs = [refs[index] for index in decode_indices]
        selected_langs = [langs[index] for index in decode_indices]
        for generation, ref, lang in zip(generations, selected_refs, selected_langs):
            decoded_chunks += generation.chunk_count
            eos_chunks += generation.eos_count
            truncated_chunks += generation.truncated_chunk_count
            hyp = tokenizer.decode(generation.token_ids)
            if logger is not None and examples_logged < log_examples:
                log_event(
                    logger,
                    "eval_prediction",
                    stage="stage2",
                    step=step,
                    lang=lang,
                    reference=ref,
                    hypothesis=hyp,
                    chunks=generation.chunk_count,
                    eos_chunks=generation.eos_count,
                    truncated_chunks=generation.truncated_chunk_count,
                )
                examples_logged += 1
            if lang == "en":
                en_scores.append(wer(ref, hyp))
            else:
                zh_scores.append(cer(ref, hyp))
    metrics = {"loss": sum(losses) / max(1, len(losses))}
    if zh_scores:
        metrics["cer_zh"] = sum(zh_scores) / len(zh_scores)
    if en_scores:
        metrics["wer_en"] = sum(en_scores) / len(en_scores)
    decoded_samples = len(zh_scores) + len(en_scores)
    if decoded_samples:
        metrics["decoded_samples"] = float(decoded_samples)
        metrics["decoded_zh_samples"] = float(len(zh_scores))
        metrics["decoded_en_samples"] = float(len(en_scores))
    if decoded_chunks:
        metrics["decode_eos_rate"] = eos_chunks / decoded_chunks
        metrics["decode_truncated_chunk_rate"] = truncated_chunks / decoded_chunks
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
    torch.save({"model": unwrap_model(model).state_dict(), "optimizer": optimizer.state_dict(), "step": step}, checkpoint_dir / "pytorch_model.bin")
    tokenizer.save(checkpoint_dir / "text_tokenizer.json")
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
    run_output_dir = resolve_path(config.get("output_dir", "outputs/stage2"))
    checkpoint_dir = run_output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = rank_log_path(resolve_path(log_file or config.get("log_file") or run_output_dir / "log.txt"), context)
    logger = setup_train_logger(f"soulx_duplug.stage2.rank{context.rank}", log_path)

    try:
        seed = int(config.get("seed", 1337))
        set_seed(seed)
        log_event(
            logger,
            "train_start",
            stage="stage2",
            seed=seed,
            output_dir=str(run_output_dir),
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
            stage1_checkpoint=str(resolve_path(config["stage1_checkpoint"])) if config.get("stage1_checkpoint") else None,
        )
        train_records = list(read_stage2_manifest(train_manifest))
        if data_cfg.get("limit_train_samples") is not None:
            train_records = train_records[: int(data_cfg["limit_train_samples"])]
        dev_records = list(read_stage2_manifest(dev_manifest)) if dev_manifest else []
        if data_cfg.get("limit_dev_samples") is not None:
            dev_records = dev_records[: int(data_cfg["limit_dev_samples"])]
        log_event(logger, "manifest_loaded", split="train", **record_summary(train_records), **stage2_chunk_summary(train_records))
        if dev_records:
            log_event(logger, "manifest_loaded", split="dev", **record_summary(dev_records), **stage2_chunk_summary(dev_records))
        if not train_records:
            raise RuntimeError("Stage 2 train manifest has no records")

        text_tokenizer = build_or_load_text_tokenizer(config, train_records)
        asr_eos_id = text_tokenizer.ensure_token(ASR_EOS_TOKEN)
        speech_tokenizer = build_speech_tokenizer(config.get("tokenizer", {}))
        log_event(
            logger,
            "tokenizers_ready",
            text_tokenizer=type(text_tokenizer).__name__,
            text_vocab_size=text_tokenizer.vocab_size,
            asr_eos_id=asr_eos_id,
            speech_tokenizer=type(speech_tokenizer).__name__,
            speech_vocab_size=speech_tokenizer.vocab_size,
            tokenizer_backend=config.get("tokenizer", {}).get("backend", "dummy"),
        )
        model = build_interleaved_asr_model(
            config.get("model", {}),
            audio_vocab_size=speech_tokenizer.vocab_size,
            text_vocab_size=text_tokenizer.vocab_size,
        ).to(device)
        load_stage1_weights_if_available(model, config.get("stage1_checkpoint"), device, logger=logger)
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
        dataset_kwargs = {
            "speech_tokenizer": speech_tokenizer,
            "text_tokenizer": text_tokenizer,
            "asr_eos_id": asr_eos_id,
            "target_sample_rate": int(audio_cfg.get("target_sample_rate", 16000)),
            "chunk_seconds": float(audio_cfg.get("chunk_seconds", 0.16)),
            "lookback_seconds": float(audio_cfg.get("lookback_seconds", 0.96)),
            "lookahead_seconds": float(audio_cfg.get("lookahead_seconds", 0.04)),
            "max_sequence_length": data_cfg.get("max_sequence_length"),
        }
        log_event(
            logger,
            "stage2_audio_config",
            target_sample_rate=dataset_kwargs["target_sample_rate"],
            chunk_seconds=dataset_kwargs["chunk_seconds"],
            lookback_seconds=dataset_kwargs["lookback_seconds"],
            lookahead_seconds=dataset_kwargs["lookahead_seconds"],
            max_sequence_length=dataset_kwargs["max_sequence_length"],
        )
        train_dataset = Stage2StreamingAsrDataset(train_records, **dataset_kwargs)
        dev_dataset = Stage2StreamingAsrDataset(dev_records, **dataset_kwargs) if dev_records else None

        train_cfg = config.get("training", {})
        batch_size = int(train_cfg.get("batch_size", 1))
        train_sampler = make_distributed_sampler(train_dataset, context, shuffle=True)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_sampler is None,
            sampler=train_sampler,
            collate_fn=lambda samples: collate_stage2(samples, pad_id=text_tokenizer.pad_id),
        )
        dev_loader = DataLoader(
            dev_dataset,
            batch_size=int(train_cfg.get("eval_batch_size", batch_size)),
            shuffle=False,
            collate_fn=lambda samples: collate_stage2(samples, pad_id=text_tokenizer.pad_id),
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
        max_new_tokens_per_chunk = int(train_cfg.get("max_new_tokens_per_chunk", 16))
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
            stage="stage2",
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
                stage="stage2",
                is_main=context.is_main,
            )
        barrier(context)
        curve_tracker = TrainingCurveTracker(
            output_dir=run_output_dir,
            stage="stage2",
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
            max_new_tokens_per_chunk=max_new_tokens_per_chunk,
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
            for batch, _, _ in train_loader:
                model.train()
                batch = move_interleaved_batch(batch, device)
                sync_gradients = (step + 1) % grad_accum == 0
                with maybe_no_sync(model, context, sync_gradients=sync_gradients):
                    output = model(batch)
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
                            asr_eos_id=asr_eos_id,
                            max_new_tokens_per_chunk=max_new_tokens_per_chunk,
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
            log_event(logger, "train_complete", stage="stage2", step=step, checkpoint=str(last_checkpoint), **cuda_memory_summary(device))
        barrier(context)
        return final_checkpoint
    except Exception as exc:
        log_exception(logger, "stage2", exc, device)
        raise
    finally:
        cleanup_distributed(context)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 2 chunk-based streaming ASR.")
    parser.add_argument("--config", type=Path, default=Path("configs/stage2_streaming_asr.yaml"))
    parser.add_argument("--log-file", type=Path, help="Write training logs to this file. Defaults to <output_dir>/log.txt.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_yaml(args.config), log_file=args.log_file)


if __name__ == "__main__":
    main()
