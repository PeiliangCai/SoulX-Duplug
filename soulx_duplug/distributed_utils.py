from __future__ import annotations

import os
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    distributed: bool
    backend: str
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _requested_device(config_device: Any | None) -> str:
    if config_device is None:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return str(config_device)


def setup_distributed(config_device: Any | None = None) -> DistributedContext:
    requested_device = _requested_device(config_device)
    world_size = _env_int("WORLD_SIZE", 1)
    rank = _env_int("RANK", 0)
    local_rank = _env_int("LOCAL_RANK", 0)
    distributed = world_size > 1

    if requested_device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is false")
        if distributed:
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device(requested_device)
            backend = "none"
    else:
        device = torch.device("cpu")
        backend = "gloo" if distributed else "none"

    if distributed and not dist.is_initialized():
        dist.init_process_group(backend=backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    return DistributedContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        distributed=distributed,
        backend=backend,
        device=device,
    )


def cleanup_distributed(context: DistributedContext) -> None:
    if context.distributed and dist.is_initialized():
        dist.destroy_process_group()


def rank_log_path(log_path: str | Path, context: DistributedContext) -> Path:
    path = Path(log_path)
    if context.is_main:
        return path
    suffix = path.suffix
    stem = path.stem if suffix else path.name
    rank_name = f"{stem}.rank{context.rank}{suffix}" if suffix else f"{stem}.rank{context.rank}"
    return path.with_name(rank_name)


def distributed_summary(context: DistributedContext) -> dict[str, Any]:
    return {
        "distributed": context.distributed,
        "backend": context.backend,
        "rank": context.rank,
        "local_rank": context.local_rank,
        "world_size": context.world_size,
        "is_main": context.is_main,
        "device": str(context.device),
    }


def make_distributed_sampler(
    dataset: Dataset[Any],
    context: DistributedContext,
    *,
    shuffle: bool,
) -> DistributedSampler[Any] | None:
    if not context.distributed:
        return None
    return DistributedSampler(
        dataset,
        num_replicas=context.world_size,
        rank=context.rank,
        shuffle=shuffle,
        drop_last=False,
    )


def wrap_model_for_distributed(model: torch.nn.Module, context: DistributedContext) -> torch.nn.Module:
    if not context.distributed:
        return model
    if context.device.type == "cuda":
        return DistributedDataParallel(
            model,
            device_ids=[context.local_rank],
            output_device=context.local_rank,
        )
    return DistributedDataParallel(model)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    if isinstance(model, DistributedDataParallel):
        return model.module
    return model


def maybe_no_sync(model: torch.nn.Module, context: DistributedContext, *, sync_gradients: bool) -> ContextManager[Any]:
    if context.distributed and not sync_gradients and isinstance(model, DistributedDataParallel):
        return model.no_sync()
    return nullcontext()


def reduce_mean(value: torch.Tensor | float, context: DistributedContext) -> float:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().float()
        if tensor.numel() != 1:
            tensor = tensor.mean()
        tensor = tensor.to(context.device)
    else:
        tensor = torch.tensor(float(value), dtype=torch.float32, device=context.device)
    if context.distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= context.world_size
    return float(tensor.detach().cpu())


def barrier(context: DistributedContext) -> None:
    if context.distributed:
        dist.barrier()
