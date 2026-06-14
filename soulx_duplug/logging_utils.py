from __future__ import annotations

import json
import logging
import platform
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import torch


def setup_train_logger(name: str, log_file: str | Path) -> logging.Logger:
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    for handler in (stream_handler, file_handler):
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    log_event(logger, "logger_ready", log_file=str(log_path))
    return logger


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields: Any) -> None:
    payload = {"event": event, **fields}
    logger.log(level, json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str))


def runtime_summary(device: torch.device) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
    }
    summary.update(cuda_memory_summary(device))
    return summary


def cuda_memory_summary(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {}
    index = device.index if device.index is not None else torch.cuda.current_device()
    props = torch.cuda.get_device_properties(index)
    gib = 1024**3
    return {
        "cuda_device_index": index,
        "cuda_device_name": props.name,
        "cuda_total_memory_gib": round(props.total_memory / gib, 3),
        "cuda_memory_allocated_gib": round(torch.cuda.memory_allocated(index) / gib, 3),
        "cuda_memory_reserved_gib": round(torch.cuda.memory_reserved(index) / gib, 3),
        "cuda_max_memory_allocated_gib": round(torch.cuda.max_memory_allocated(index) / gib, 3),
        "cuda_max_memory_reserved_gib": round(torch.cuda.max_memory_reserved(index) / gib, 3),
    }


def parameter_summary(model: torch.nn.Module) -> dict[str, Any]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return {
        "total_parameters": total,
        "trainable_parameters": trainable,
        "frozen_parameters": total - trainable,
        "trainable_ratio": round(trainable / total, 6) if total else 0.0,
    }


def record_summary(records: Iterable[Any]) -> dict[str, Any]:
    items = list(records)
    by_dataset = Counter(str(getattr(item, "dataset", "") or "unknown") for item in items)
    by_lang = Counter(str(getattr(item, "lang", "") or "unknown") for item in items)
    by_split = Counter(str(getattr(item, "split", "") or "unknown") for item in items)
    duration_seconds = 0.0
    records_with_duration = 0
    for item in items:
        duration = getattr(item, "duration", None)
        if duration is None:
            continue
        try:
            duration_seconds += float(duration)
            records_with_duration += 1
        except (TypeError, ValueError):
            continue
    return {
        "records": len(items),
        "hours": round(duration_seconds / 3600.0, 4),
        "records_with_duration": records_with_duration,
        "by_dataset": dict(sorted(by_dataset.items())),
        "by_lang": dict(sorted(by_lang.items())),
        "by_split": dict(sorted(by_split.items())),
    }


def stage2_chunk_summary(records: Iterable[Any]) -> dict[str, Any]:
    items = list(records)
    chunk_counts = [len(getattr(item, "chunks", []) or []) for item in items]
    total_chunks = sum(chunk_counts)
    return {
        "total_chunks": total_chunks,
        "avg_chunks_per_record": round(total_chunks / len(items), 3) if items else 0.0,
        "max_chunks_per_record": max(chunk_counts) if chunk_counts else 0,
    }


def log_exception(logger: logging.Logger, stage: str, exc: BaseException, device: torch.device) -> None:
    fields: dict[str, Any] = {
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
    }
    if isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower():
        fields["probable_oom"] = True
        fields.update(cuda_memory_summary(device))
    log_event(logger, "train_failed", logging.ERROR, **fields)
    logger.exception("uncaught exception during %s", stage)
