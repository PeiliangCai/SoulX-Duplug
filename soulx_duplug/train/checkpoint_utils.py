from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

from soulx_duplug.config import resolve_path
from soulx_duplug.distributed_utils import unwrap_model
from soulx_duplug.logging_utils import log_event


def _resume_disabled(value: Any) -> bool:
    if value is None:
        return False
    if value is False:
        return True
    if isinstance(value, str):
        return value.strip().lower() in {"0", "false", "no", "none", "off", "disable", "disabled"}
    return False


def resolve_resume_checkpoint(
    checkpoint_root: Path,
    train_cfg: dict[str, Any],
    *,
    logger: logging.Logger,
    stage: str,
    is_main: bool,
) -> Path | None:
    value = train_cfg.get("resume_from_checkpoint", "latest")
    if _resume_disabled(value):
        if is_main:
            log_event(logger, "checkpoint_resume_disabled", stage=stage)
        return None

    if value is True or str(value).strip().lower() in {"", "1", "true", "yes", "auto", "latest"}:
        checkpoint_dir = checkpoint_root / "latest"
        explicit = False
    else:
        checkpoint_dir = resolve_path(str(value))
        explicit = True

    state_path = checkpoint_dir / "pytorch_model.bin"
    if not state_path.exists():
        if explicit:
            raise FileNotFoundError(f"resume checkpoint not found: {state_path}")
        if is_main:
            log_event(
                logger,
                "checkpoint_resume_skipped",
                stage=stage,
                checkpoint=str(checkpoint_dir),
                reason="latest checkpoint is missing",
            )
        return None
    return checkpoint_dir.resolve()


def _move_optimizer_state_to_device(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def load_training_checkpoint(
    *,
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    logger: logging.Logger,
    stage: str,
    is_main: bool,
) -> int:
    state_path = checkpoint_dir / "pytorch_model.bin"
    payload = torch.load(state_path, map_location=device)
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid checkpoint payload: {state_path}")

    model_state = payload.get("model", payload)
    unwrap_model(model).load_state_dict(model_state, strict=True)

    optimizer_loaded = False
    if "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
        _move_optimizer_state_to_device(optimizer, device)
        optimizer_loaded = True

    step = int(payload.get("step", 0))
    if is_main:
        log_event(
            logger,
            "checkpoint_resumed",
            stage=stage,
            checkpoint=str(checkpoint_dir),
            state_file=str(state_path),
            step=step,
            optimizer_loaded=optimizer_loaded,
        )
    return step
