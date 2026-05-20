"""
Trainer callbacks for GRPO: training-loss / reward CSV + PDF.

Mirrors ``sft/callbacks.py`` cadence: dense ``training_loss_history.csv`` every
``--loss-csv-every`` logged steps and sparse ``metrics_history.csv`` +
``statistics.pdf`` every ``--plot-every`` steps.
"""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from sft.progress_callbacks import TRAINING_LOSS_HISTORY_CSV


METRICS_CSV_NAME = "metrics_history.csv"
STATISTICS_PDF_NAME = "statistics.pdf"

METRICS_CSV_FIELDNAMES = ("global_step", "train_loss", "reward")

GRPO_HISTORY_CSV_FIELDS = (
    "global_step",
    "train_loss",
    "learning_rate",
    "reward",
    "kl",
    "entropy",
)

_log = logging.getLogger(__name__)


def _distributed_barrier() -> None:
    if not torch.distributed.is_initialized():
        return
    try:
        nccl_cuda = (
            torch.distributed.get_backend() == "nccl"
            and torch.cuda.is_available()
        )
    except RuntimeError:
        nccl_cuda = False
    if nccl_cuda:
        torch.distributed.barrier(device_ids=[torch.cuda.current_device()])
    else:
        torch.distributed.barrier()


def _latest_logged_metric(state: TrainerState, key: str) -> float | None:
    for entry in reversed(state.log_history):
        if key in entry and entry.get("step") == state.global_step:
            return float(entry[key])
    for entry in reversed(state.log_history):
        if key in entry:
            return float(entry[key])
    return None


def _load_training_loss_history_rows(path: Path) -> tuple[list[int], list[float]]:
    if not path.is_file():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps_out: list[int] = []
    loss_out: list[float] = []
    for r in rows:
        try:
            gs_raw = (r.get("global_step") or "").strip()
            lv_raw = (r.get("train_loss") or "").strip()
            if not gs_raw or not lv_raw:
                continue
            steps_out.append(int(gs_raw))
            loss_out.append(float(lv_raw))
        except (TypeError, ValueError):
            continue
    return steps_out, loss_out


class GrpoTrainHistoryCallback(TrainerCallback):
    """Append ``training_loss_history.csv`` from GRPO ``on_log`` metrics."""

    def __init__(self, output_dir: Path, every: int) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every = max(1, int(every))
        self.csv_path = self.output_dir / TRAINING_LOSS_HISTORY_CSV

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        del args, kwargs
        if not state.is_world_process_zero:
            return control
        if logs is None or "loss" not in logs:
            return control
        step = int(state.global_step)
        if step <= 0 or step % self.every != 0:
            return control

        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(GRPO_HISTORY_CSV_FIELDS),
                extrasaction="ignore",
            )
            if write_header:
                w.writeheader()

            def _fmt(key: str) -> str:
                val = logs.get(key)
                return "" if val is None else str(round(float(val), 8))

            lr = logs.get("learning_rate")
            w.writerow(
                {
                    "global_step": step,
                    "train_loss": round(float(logs["loss"]), 8),
                    "learning_rate": "" if lr is None else round(float(lr), 12),
                    "reward": _fmt("reward"),
                    "kl": _fmt("kl"),
                    "entropy": _fmt("entropy"),
                }
            )
        return control


class GrpoTrainingPlotCallback(TrainerCallback):
    """
    Sparse ``metrics_history.csv`` + ``statistics.pdf`` every ``plot_every`` steps.

    Panel 1: training loss snapshots; panel 2: reward snapshots; panel 3: dense
    loss from ``training_loss_history.csv``.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        plot_every: int,
        metrics_csv_name: str = METRICS_CSV_NAME,
        pdf_name: str = STATISTICS_PDF_NAME,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.plot_every = max(1, int(plot_every))
        self.metrics_csv_path = self.output_dir / metrics_csv_name
        self.pdf_path = self.output_dir / pdf_name
        self.trainer = None
        self._last_metrics_global_step: int | None = None
        self._pdf_executor: ThreadPoolExecutor | None = None

    @staticmethod
    def _ddp_sync_needed(trainer) -> bool:
        if not torch.distributed.is_available():
            return False
        if not torch.distributed.is_initialized():
            return False
        return int(getattr(trainer.args, "world_size", 1) or 1) > 1

    def _run_plot_synchronized(self, trainer, state: TrainerState, *, triggered_from: str) -> None:
        del triggered_from
        if not self._ddp_sync_needed(trainer):
            self._maybe_plot(trainer, state)
            return
        _distributed_barrier()
        try:
            if trainer.is_world_process_zero():
                self._maybe_plot(trainer, state)
        finally:
            _distributed_barrier()

    @staticmethod
    def _checkpoint_save_this_step(args, step: int) -> bool:
        strategy = getattr(args, "save_strategy", None)
        if strategy is None:
            return False
        name = getattr(strategy, "value", strategy)
        if name != "steps":
            return False
        ss = int(getattr(args, "save_steps", 0) or 0)
        return ss > 0 and step % ss == 0

    def _append_csv_row(self, row: dict[str, object]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(METRICS_CSV_FIELDNAMES), extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _maybe_plot(self, trainer, state: TrainerState) -> None:
        step = int(state.global_step)
        if step <= 0:
            return
        if self._last_metrics_global_step == step:
            return
        if not trainer.is_world_process_zero():
            return

        train_loss = _latest_logged_metric(state, "loss")
        reward = _latest_logged_metric(state, "reward")
        self._append_csv_row(
            {
                "global_step": step,
                "train_loss": "" if train_loss is None else round(train_loss, 6),
                "reward": "" if reward is None else round(reward, 6),
            }
        )
        self._schedule_pdf_render()
        self._last_metrics_global_step = step

    def _schedule_pdf_render(self) -> None:
        def run() -> None:
            try:
                self._render_pdf()
            except Exception:
                _log.debug("statistics.pdf render failed", exc_info=True)

        if self._pdf_executor is None:
            self._pdf_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="grpo-loss-pdf",
            )
        self._pdf_executor.submit(run)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        del args, kwargs
        if state.is_world_process_zero() and self._pdf_executor is not None:
            self._pdf_executor.shutdown(wait=True)
            self._pdf_executor = None
        return control

    def _render_pdf(self) -> None:
        rows: list[dict[str, str]] = []
        if self.metrics_csv_path.is_file():
            with open(self.metrics_csv_path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        hist_path = self.output_dir / TRAINING_LOSS_HISTORY_CSV
        hist_steps, hist_loss = _load_training_loss_history_rows(hist_path)

        if not rows and not hist_steps:
            return

        steps = [int(r["global_step"]) for r in rows] if rows else []

        def col(name: str) -> list[float | None]:
            out: list[float | None] = []
            for r in rows:
                v = r.get(name, "").strip()
                out.append(float(v) if v != "" else None)
            return out

        loss_y = col("train_loss")
        reward_y = col("reward")

        fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
        ax0, ax1, ax2 = axes

        if any(y is not None for y in loss_y):
            ax0.plot(steps, loss_y, marker="o", ms=3)
        ax0.set_title("Training loss (plot-every snapshots)")
        ax0.set_ylabel("loss")
        ax0.grid(True, alpha=0.3)

        if any(y is not None for y in reward_y):
            ax1.plot(steps, reward_y, marker="o", ms=3, color="C1")
        ax1.set_title("Mean reward (Judger outcome + format bonus)")
        ax1.set_ylabel("reward")
        ax1.grid(True, alpha=0.3)

        if hist_steps:
            ax2.plot(hist_steps, hist_loss, color="C2", linewidth=1.0, alpha=0.9)
        ax2.set_title("Training loss (logged, training_loss_history.csv)")
        ax2.set_xlabel("global_step")
        ax2.set_ylabel("loss")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        del kwargs
        trainer = self.trainer
        if trainer is None:
            return control
        if logs is None or "loss" not in logs:
            return control
        step = int(state.global_step)
        if step <= 0 or step % self.plot_every != 0:
            return control
        if self._checkpoint_save_this_step(args, step):
            return control
        self._run_plot_synchronized(trainer, state, triggered_from="on_log")
        return control

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        del kwargs
        trainer = self.trainer
        if trainer is None:
            return control
        step = int(state.global_step)
        if step <= 0:
            return control
        if not self._checkpoint_save_this_step(args, step):
            return control
        if step % self.plot_every != 0:
            return control
        self._run_plot_synchronized(trainer, state, triggered_from="on_save")
        return control


def latest_checkpoint_dir(output_dir: Path) -> str | None:
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith(PREFIX_CHECKPOINT_DIR)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None
