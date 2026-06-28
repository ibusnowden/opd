"""Thin Weights & Biases wrapper — drop-in for how upstream uses `mlrunx`.

Upstream `policy_gradients/train.py` does roughly:
    run = mlrunx.init(project_id=..., name=...); mlrunx.log_params(...); mlrunx.log(metrics, step=...)
This module mirrors that surface with W&B and degrades gracefully:
  * `wandb` not installed            -> no-op logger (prints once).
  * `WANDB_MODE=offline` / `disabled` -> respected (W&B handles it).
  * not the main DDP rank             -> no-op (pass `is_main=False`).

Usage:
    from harness.wandb_logging import Logger
    log = Logger.init(project="distill-harness", name="opd_seed42", config=cfg.model_dump(),
                      is_main=dist_env.is_main_process)
    log.log_params({"alpha": cfg.alpha, "lam": cfg.lam})
    log.log({"loss": 0.31, "kl_to_teacher": 0.02}, step=step)
    log.finish()
"""

from __future__ import annotations

import os
from typing import Any


class _NoOpLogger:
    """Used when wandb is unavailable, disabled, or on a non-main rank."""

    def __init__(self, reason: str = "") -> None:
        self._reason = reason

    def log_params(self, params: dict[str, Any]) -> None:  # noqa: D102
        pass

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:  # noqa: D102
        pass

    def finish(self) -> None:  # noqa: D102
        pass


class Logger:
    """W&B-backed run logger."""

    def __init__(self, run) -> None:  # `run` is a wandb.sdk.wandb_run.Run
        self._run = run

    # -- construction ---------------------------------------------------------
    @classmethod
    def init(
        cls,
        *,
        project: str | None = None,
        name: str | None = None,
        config: dict[str, Any] | None = None,
        group: str | None = None,
        tags: list[str] | None = None,
        is_main: bool = True,
    ) -> "Logger | _NoOpLogger":
        if not is_main:
            return _NoOpLogger("non-main rank")
        if os.environ.get("WANDB_MODE") in {"disabled"}:
            return _NoOpLogger("WANDB_MODE=disabled")
        try:
            import wandb  # noqa: PLC0415
        except ImportError:
            print("[harness.wandb_logging] `wandb` not installed — metrics will not be logged. "
                  "`pip install wandb` (and `wandb login`) to enable.")
            return _NoOpLogger("wandb not installed")

        run = wandb.init(
            project=project or os.environ.get("WANDB_PROJECT", "distill-harness"),
            name=name,
            config=config or {},
            group=group,
            tags=tags,
        )
        return cls(run)

    # -- logging --------------------------------------------------------------
    def log_params(self, params: dict[str, Any]) -> None:
        # W&B merges into the run config.
        self._run.config.update(params, allow_val_change=True)

    def log(self, metrics: dict[str, float], step: int | None = None) -> None:
        self._run.log(metrics, step=step)

    def finish(self) -> None:
        self._run.finish()
