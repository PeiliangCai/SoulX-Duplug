from __future__ import annotations

import json
import logging
import os
import uuid
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from soulx_duplug.logging_utils import log_event


class TrainingCurveTracker:
    def __init__(
        self,
        *,
        output_dir: str | Path,
        stage: str,
        logger: logging.Logger,
        plot_every: int = 100,
        smoothing_window: int = 20,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.stage = stage
        self.logger = logger
        self.plot_every = max(0, int(plot_every))
        self.smoothing_window = max(1, int(smoothing_window))
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_id = f"{stage}-{timestamp}-{os.getpid()}-{uuid.uuid4().hex[:8]}"
        self.metrics_path = self.output_dir / "training_metrics.jsonl"
        self.plot_path = self.output_dir / "training_curves.png"
        self.train_points: list[tuple[int, float]] = []
        self.eval_points: list[dict[str, float]] = []

        log_event(
            self.logger,
            "training_curves_ready",
            run_id=self.run_id,
            metrics_file=str(self.metrics_path),
            plot_file=str(self.plot_path),
            plot_every=self.plot_every,
            smoothing_window=self.smoothing_window,
        )

    def _append_metric(self, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "run_id": self.run_id,
            "stage": self.stage,
            **payload,
        }
        with self.metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def record_train(self, *, step: int, train_loss: float) -> None:
        loss = float(train_loss)
        self.train_points.append((int(step), loss))
        self._append_metric({"event": "train", "step": int(step), "train_loss": loss})

    def record_eval(self, *, step: int, metrics: dict[str, float]) -> None:
        point = {"step": float(step)}
        payload: dict[str, Any] = {"event": "eval", "step": int(step)}
        for name in ("loss", "cer_zh", "wer_en"):
            if name in metrics:
                value = float(metrics[name])
                point[name] = value
                payload[name] = value
        self.eval_points.append(point)
        self._append_metric(payload)

    def should_plot(self, step: int) -> bool:
        return self.plot_every > 0 and step > 0 and step % self.plot_every == 0

    def plot(self, *, step: int, reason: str) -> bool:
        if self.plot_every == 0:
            return False
        if not self.train_points and not self.eval_points:
            return False
        try:
            self._plot()
        except Exception as exc:
            log_event(
                self.logger,
                "training_curves_failed",
                level=logging.WARNING,
                step=step,
                reason=reason,
                exception_type=type(exc).__name__,
                message=str(exc),
            )
            return False
        log_event(
            self.logger,
            "training_curves_updated",
            step=step,
            reason=reason,
            plot_file=str(self.plot_path),
            metrics_file=str(self.metrics_path),
            train_points=len(self.train_points),
            eval_points=len(self.eval_points),
        )
        return True

    def _plot(self) -> None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, (loss_axis, error_axis) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        try:
            if self.train_points:
                train_steps = [step for step, _ in self.train_points]
                train_losses = [loss for _, loss in self.train_points]
                loss_axis.plot(train_steps, train_losses, color="#9ecae1", linewidth=1, alpha=0.65, label="train loss")
                smooth_losses = self._moving_average(train_losses)
                loss_axis.plot(
                    train_steps,
                    smooth_losses,
                    color="#08519c",
                    linewidth=2,
                    label=f"train loss (moving avg {self.smoothing_window})",
                )

            dev_loss_points = [
                (int(point["step"]), point["loss"])
                for point in self.eval_points
                if "loss" in point
            ]
            if dev_loss_points:
                loss_axis.plot(
                    [step for step, _ in dev_loss_points],
                    [value for _, value in dev_loss_points],
                    color="#e6550d",
                    marker="o",
                    linewidth=1.5,
                    label="dev loss",
                )
            loss_axis.set_ylabel("Loss")
            loss_axis.set_title(f"{self.stage.upper()} training progress")
            loss_axis.grid(True, alpha=0.25)
            if loss_axis.lines:
                loss_axis.legend()

            error_series = (
                ("cer_zh", "CER (zh)", "#31a354"),
                ("wer_en", "WER (en)", "#756bb1"),
            )
            for key, label, color in error_series:
                points = [
                    (int(point["step"]), point[key])
                    for point in self.eval_points
                    if key in point
                ]
                if points:
                    error_axis.plot(
                        [step for step, _ in points],
                        [value for _, value in points],
                        color=color,
                        marker="o",
                        linewidth=1.5,
                        label=label,
                    )
            error_axis.set_xlabel("Training step")
            error_axis.set_ylabel("Error rate")
            error_axis.grid(True, alpha=0.25)
            if error_axis.lines:
                error_axis.legend()
            else:
                error_axis.text(
                    0.5,
                    0.5,
                    "CER/WER is not available yet",
                    ha="center",
                    va="center",
                    transform=error_axis.transAxes,
                    color="#666666",
                )

            figure.tight_layout()
            temporary_path = self.plot_path.with_name(f".{self.plot_path.stem}.tmp{self.plot_path.suffix}")
            figure.savefig(temporary_path, dpi=150)
        finally:
            plt.close(figure)
        os.replace(temporary_path, self.plot_path)

    def _moving_average(self, values: list[float]) -> list[float]:
        window: deque[float] = deque()
        total = 0.0
        smoothed: list[float] = []
        for value in values:
            window.append(value)
            total += value
            if len(window) > self.smoothing_window:
                total -= window.popleft()
            smoothed.append(total / len(window))
        return smoothed
