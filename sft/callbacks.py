"""
Trainer callbacks for SFT: training-loss CSV/PDF.

Stratified ``trainer.evaluate()`` during SFT was removed (DDP / NCCL fragile); use ``infer.py`` +
``evaluate.py`` on the public set for accuracy.
"""

from __future__ import annotations

import csv
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from .progress_callbacks import (
    CHECKPOINT_LATEST,
    LatestCheckpointSymlinkCallback,
    TRAINING_LOSS_HISTORY_CSV,
    TrainLossHistoryCallback,
    install_checkpoint_chunk_progress_bar,
    resolve_checkpoint_latest_path,
)


METRICS_CSV_NAME = "metrics_history.csv"
STATISTICS_PDF_NAME = "statistics.pdf"

# Sparse snapshots every ``plot_every`` optimizer steps (same cadence as statistics.pdf).
METRICS_CSV_FIELDNAMES = ("global_step", "train_loss", "mean_token_accuracy")

_log = logging.getLogger(__name__)

def _distributed_barrier() -> None:
    """Synchronize all ranks; NCCL uses explicit device to avoid implicit-device UserWarnings."""
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


def _latest_train_loss(state: TrainerState) -> float | None:
    for entry in reversed(state.log_history):
        if "loss" in entry and entry.get("step") == state.global_step:
            return float(entry["loss"])
    for entry in reversed(state.log_history):
        if "loss" in entry:
            return float(entry["loss"])
    return None


def _latest_mean_token_accuracy(state: TrainerState) -> float | None:
    """TRL ``SFTTrainer`` averages masked next-token match into ``mean_token_accuracy`` (training batches)."""
    for entry in reversed(state.log_history):
        if "mean_token_accuracy" in entry and entry.get("step") == state.global_step:
            return float(entry["mean_token_accuracy"])
    for entry in reversed(state.log_history):
        if "mean_token_accuracy" in entry:
            return float(entry["mean_token_accuracy"])
    return None


def _accuracy_ylim(values: list[float | None]) -> tuple[float, float] | None:
    """Tight y-range around observed train token accuracy (still clipped to [0, 1])."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    lo, hi = min(vals), max(vals)
    span = hi - lo
    pad = max(0.02, span * 0.15) if span > 1e-9 else 0.05
    ymin = max(0.0, lo - pad)
    ymax = min(1.0, hi + pad)
    if ymax <= ymin:
        ymax = min(1.0, ymin + 0.08)
    return ymin, ymax


def _load_training_loss_history_rows(path: Path) -> tuple[list[int], list[float]]:
    """Parse ``training_loss_history.csv`` for ``global_step`` / ``train_loss``."""
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


class TrainingLossPlotCallback(TrainerCallback):
    """
    Append ``metrics_history.csv`` + ``statistics.pdf`` on a schedule: **training loss** and
    **mean_token_accuracy** from TRL ``SFTTrainer`` logs (masked next-token match on labels — not
    public-set Judge accuracy).

    Does **not** call ``trainer.evaluate()`` during training (DDP / NCCL fragile at scale). For
    leaderboard scores use ``inference/infer.py`` + ``inference/evaluate.py``.

    Refreshes every ``plot_every`` global steps (``--plot-every``). PDF: sparse metrics (loss +
    token accuracy), dense loss from ``training_loss_history.csv`` when present.

    When a plot step coincides with ``save_steps``, CSV update runs in ``on_save`` (after checkpoint).
    Under DDP, non-zero ranks barrier only around rank 0 appending ``metrics_history.csv``; PDF rendering
    runs in a background thread on rank 0 so tqdm / the training loop are not blocked by matplotlib.
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
        """All ranks barrier around rank 0 appending CSV (PDF runs later on a worker thread)."""
        if not self._ddp_sync_needed(trainer):
            self._maybe_plot(trainer, state, triggered_from=triggered_from)
            return
        _distributed_barrier()
        try:
            if trainer.is_world_process_zero():
                self._maybe_plot(trainer, state, triggered_from=triggered_from)
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

    def _maybe_plot(self, trainer, state: TrainerState, *, triggered_from: str) -> None:
        step = int(state.global_step)
        if step <= 0:
            return
        if self._last_metrics_global_step == step:
            return
        if not trainer.is_world_process_zero():
            return

        train_loss = _latest_train_loss(state)
        mta = _latest_mean_token_accuracy(state)
        self._append_csv_row(
            {
                "global_step": step,
                "train_loss": "" if train_loss is None else round(train_loss, 6),
                "mean_token_accuracy": "" if mta is None else round(mta, 6),
            }
        )
        self._schedule_pdf_render()
        self._last_metrics_global_step = step

    def _schedule_pdf_render(self) -> None:
        """Matplotlib work off the training thread so tqdm is not stalled."""

        def run() -> None:
            try:
                self._render_pdf()
            except Exception:
                _log.debug("statistics.pdf render failed", exc_info=True)

        if self._pdf_executor is None:
            self._pdf_executor = ThreadPoolExecutor(
                max_workers=1,
                thread_name_prefix="sft-loss-pdf",
            )
        self._pdf_executor.submit(run)

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and self._pdf_executor is not None:
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
        tok_y = col("mean_token_accuracy")

        fig, axes = plt.subplots(3, 1, figsize=(8, 9.5), sharex=True)
        ax0, ax1, ax2 = axes

        if any(y is not None for y in loss_y):
            ax0.plot(steps, loss_y, marker="o", ms=3)
        ax0.set_title("Training loss (plot-every snapshots)")
        ax0.set_ylabel("loss")
        ax0.grid(True, alpha=0.3)

        if any(y is not None for y in tok_y):
            ax1.plot(steps, tok_y, marker="o", ms=3, color="C1")
        ax1.set_title("Train token accuracy (TRL, masked LM)")
        ax1.set_ylabel("accuracy")
        ylim_acc = _accuracy_ylim(tok_y)
        if ylim_acc is not None:
            ax1.set_ylim(ylim_acc)
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

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = self.trainer
        if trainer is None:
            return control
        step = int(state.global_step)
        if step <= 0:
            return control

        if step % self.plot_every == 0:
            if self._checkpoint_save_this_step(args, step):
                return control
            self._run_plot_synchronized(trainer, state, triggered_from="on_step_end")
        return control

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
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


# Backwards-compatible name for older docs / forks.
StatisticsPlotCallback = TrainingLossPlotCallback


def latest_checkpoint_dir(output_dir: Path) -> str | None:
    """Return path string to newest checkpoint-* under output_dir, or None."""
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return None
    candidates = sorted(
        (p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith(PREFIX_CHECKPOINT_DIR)),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(candidates[0]) if candidates else None
