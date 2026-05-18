"""
Trainer callbacks for SFT: metrics CSV + statistics.pdf and optional train-set reload.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR


METRICS_CSV_NAME = "metrics_history.csv"
STATISTICS_PDF_NAME = "statistics.pdf"
"""Symlink name updated on every checkpoint save (directory symlink → ``checkpoint-{step}``)."""
CHECKPOINT_LATEST = "checkpoint-latest"


class CheckpointChunkProgressCallback(TrainerCallback):
    """
    tqdm segments of ``save_steps`` optimizer steps (checkpoint cadence), restarting the bar each segment.
    Last segment may be shorter. Loss / token accuracy only as postfix (no dict spam).
    Total optimizer steps are logged once at train start; the bar title stays short (segment id + global step range).
    """

    def __init__(self) -> None:
        super().__init__()
        self.bar = None
        self._cur_seg_start: int | None = None
        self._printed_step_banner = False

    @staticmethod
    def _num_chunks(max_steps: int, chunk: int) -> int:
        chunk = max(1, chunk)
        ms = max(0, max_steps)
        return max(1, (ms + chunk - 1) // chunk)

    @staticmethod
    def _chunk_index_one_based(gs: int, chunk: int) -> int:
        chunk = max(1, chunk)
        return (max(gs, 1) - 1) // chunk + 1

    def on_train_begin(self, args, state, control, **kwargs):
        self.bar = None
        self._cur_seg_start = None
        self._printed_step_banner = False
        if state.is_world_process_zero:
            chunk = int(args.save_steps) if getattr(args, "save_steps", None) else 500
            chunk = max(1, chunk)
            max_steps = int(state.max_steps or 0)
            if max_steps > 0:
                n_chunks = self._num_chunks(max_steps, chunk)
                print(
                    f"[progress] total optimizer steps={max_steps}; "
                    f"{n_chunks} tqdm segments (up to {chunk} steps each, aligned to checkpoint cadence)",
                )
                self._printed_step_banner = True
        return control

    def _segment_bounds(self, gs: int, max_steps: int, chunk: int) -> tuple[int, int, int]:
        chunk = max(1, chunk)
        seg_start = ((max(gs, 1) - 1) // chunk) * chunk + 1
        seg_end = min(seg_start + chunk - 1, max_steps)
        seg_total = seg_end - seg_start + 1
        return seg_start, seg_end, seg_total

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return control
        gs = int(state.global_step)
        max_steps = int(state.max_steps)
        chunk = int(args.save_steps) if getattr(args, "save_steps", None) else 500
        chunk = max(1, chunk)
        seg_start, seg_end, seg_total = self._segment_bounds(gs, max_steps, chunk)
        pos_done = min(gs - seg_start + 1, seg_total)
        total_chunks = self._num_chunks(max_steps, chunk)
        chunk_i = self._chunk_index_one_based(gs, chunk)

        if not self._printed_step_banner and max_steps > 0:
            print(
                f"[progress] total optimizer steps={max_steps}; "
                f"{total_chunks} tqdm segments (up to {chunk} steps each, aligned to checkpoint cadence)",
            )
            self._printed_step_banner = True

        if self.bar is None or seg_start != self._cur_seg_start:
            if self.bar is not None:
                self.bar.close()
            self._cur_seg_start = seg_start
            self.bar = tqdm(
                total=seg_total,
                initial=min(pos_done, seg_total),
                dynamic_ncols=True,
                unit="step",
                desc=f"Training [{chunk_i}/{total_chunks}] {seg_start}-{seg_end}",
            )
        else:
            self.bar.update(1)

        return control

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if (
            not state.is_world_process_zero
            or self.bar is None
            or logs is None
        ):
            return control
        loss = logs.get("loss")
        mta = logs.get("mean_token_accuracy")
        parts: list[str] = []
        if loss is not None:
            parts.append(f"loss={float(loss):.4g}")
        if mta is not None:
            parts.append(f"mean_tok_acc={float(mta):.4g}")
        if parts:
            self.bar.set_postfix_str(", ".join(parts))
        return control

    def on_train_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero and self.bar is not None:
            self.bar.close()
            self.bar = None
            self._cur_seg_start = None
        return control


class LatestCheckpointSymlinkCallback(TrainerCallback):
    """Keeps ``checkpoint-latest`` → newest numeric ``checkpoint-{step}`` (rank 0 only)."""

    def on_save(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if not state.is_world_process_zero:
            return control
        run_dir = Path(args.output_dir)
        step = int(state.global_step)
        target_name = f"{PREFIX_CHECKPOINT_DIR}-{step}"
        target = run_dir / target_name
        link = run_dir / CHECKPOINT_LATEST
        if not target.is_dir():
            return control
        try:
            link.unlink(missing_ok=True)
            link.symlink_to(target_name, target_is_directory=True)
        except OSError:
            pass
        return control


def install_checkpoint_chunk_progress_bar(trainer) -> None:
    """Swap HF's tqdm callback for checkpoint-aligned chunk progress."""
    trainer.pop_callback(ProgressCallback)
    try:
        from transformers.utils.notebook import NotebookProgressCallback

        trainer.pop_callback(NotebookProgressCallback)
    except ImportError:
        pass
    trainer.add_callback(CheckpointChunkProgressCallback())


def _latest_train_loss(state: TrainerState) -> float | None:
    for entry in reversed(state.log_history):
        if "loss" in entry and entry.get("step") == state.global_step:
            return float(entry["loss"])
    for entry in reversed(state.log_history):
        if "loss" in entry:
            return float(entry["loss"])
    return None


class StatisticsPlotCallback(TrainerCallback):
    """
    Every ``plot_every`` steps: append metrics_history.csv and regenerate statistics.pdf.

    Train loss comes from trainer.state.log_history. Stratified accuracies come from
    ``trainer.evaluate`` on MCQ / FRQ / overall subsets (masked token accuracy); those subsets must be
    tokenized via ``SFTTrainer(eval_dataset={...})``.
    """

    def __init__(
        self,
        *,
        output_dir: Path,
        plot_every: int,
        skip_acc_eval: bool,
        metrics_csv_name: str = METRICS_CSV_NAME,
        pdf_name: str = STATISTICS_PDF_NAME,
    ) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.plot_every = max(1, int(plot_every))
        self.skip_acc_eval = skip_acc_eval
        self.metrics_csv_path = self.output_dir / metrics_csv_name
        self.pdf_path = self.output_dir / pdf_name
        # Populated by train.py after SFTTrainer is constructed (HF omits trainer from kwargs).
        self.trainer = None

    def _append_csv_row(self, row: dict[str, object]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        write_header = not self.metrics_csv_path.exists()
        with open(self.metrics_csv_path, "a", newline="") as f:
            fieldnames = [
                "global_step",
                "train_loss",
                "eval_accuracy_mcq",
                "eval_accuracy_frq",
                "eval_accuracy_overall",
            ]
            w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            if write_header:
                w.writeheader()
            w.writerow(row)

    def _run_eval_accuracy(self, trainer, key: str) -> float | None:
        eds = getattr(trainer, "eval_dataset", None)
        if not isinstance(eds, dict) or key not in eds:
            return None
        ds = eds[key]
        if ds is None or len(ds) == 0:
            return None
        metrics = trainer.evaluate(eval_dataset=ds, metric_key_prefix=f"{key}_")
        # Trainer prefixes metric keys; search for accuracy
        for k, v in metrics.items():
            lk = k.lower()
            if "accuracy" in lk and "runtime" not in lk:
                return float(v)
        return None

    def _maybe_plot(self, trainer, state: TrainerState) -> None:
        step = int(state.global_step)
        if step <= 0 or step % self.plot_every != 0:
            return

        # Avoid duplicate full evals on every DDP rank (expensive and can deadlock I/O).
        if not trainer.is_world_process_zero():
            return

        acc_mcq = acc_frq = acc_overall = None
        if not self.skip_acc_eval:
            acc_mcq = self._run_eval_accuracy(trainer, "mcq")
            acc_frq = self._run_eval_accuracy(trainer, "frq")
            acc_overall = self._run_eval_accuracy(trainer, "overall")

        train_loss = _latest_train_loss(state)
        self._append_csv_row(
            {
                "global_step": step,
                "train_loss": "" if train_loss is None else round(train_loss, 6),
                "eval_accuracy_mcq": "" if acc_mcq is None else round(acc_mcq, 6),
                "eval_accuracy_frq": "" if acc_frq is None else round(acc_frq, 6),
                "eval_accuracy_overall": "" if acc_overall is None else round(acc_overall, 6),
            }
        )
        self._render_pdf()

    def _render_pdf(self) -> None:
        if not self.metrics_csv_path.exists():
            return
        rows: list[dict[str, str]] = []
        with open(self.metrics_csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            return

        steps = [int(r["global_step"]) for r in rows]

        def col(name: str) -> list[float | None]:
            out: list[float | None] = []
            for r in rows:
                v = r.get(name, "").strip()
                out.append(float(v) if v != "" else None)
            return out

        loss_y = col("train_loss")
        mcq_y = col("eval_accuracy_mcq")
        frq_y = col("eval_accuracy_frq")
        ov_y = col("eval_accuracy_overall")

        if self.skip_acc_eval:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(steps, loss_y, marker="o", ms=3)
            ax.set_title("Training loss (--skip-acc-eval: accuracy panels omitted)")
            ax.set_xlabel("global_step")
            ax.set_ylabel("loss")
            ax.grid(True, alpha=0.3)
            fig.savefig(self.pdf_path, format="pdf", bbox_inches="tight")
            plt.close(fig)
            return

        fig, axes = plt.subplots(4, 1, figsize=(8, 11), sharex=True)
        titles = [
            "Training loss",
            "Eval token accuracy — MCQ (teacher targets)",
            "Eval token accuracy — FRQ (teacher targets)",
            "Eval token accuracy — overall (teacher targets)",
        ]
        series = [loss_y, mcq_y, frq_y, ov_y]
        ylabels = ["loss", "accuracy", "accuracy", "accuracy"]
        for ax, title, ys, ylab in zip(axes, titles, series, ylabels):
            if any(y is not None for y in ys):
                ax.plot(steps, ys, marker="o", ms=3)
            ax.set_title(title)
            ax.set_ylabel(ylab)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("global_step")
        fig.text(
            0.5,
            0.01,
            "Accuracy = masked token match on assistant completion (not Judge / leaderboard accuracy).",
            ha="center",
            fontsize=8,
            style="italic",
        )
        plt.tight_layout(rect=[0, 0.03, 1, 1])
        fig.savefig(self.pdf_path, format="pdf", bbox_inches="tight")
        plt.close(fig)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = self.trainer
        if trainer is None:
            return control
        self._maybe_plot(trainer, state)
        return control


class ReloadTrainDatasetCallback(TrainerCallback):
    """Replace ``trainer.train_dataset`` from disk every ``reload_every`` steps."""

    def __init__(
        self,
        *,
        reload_every: int,
        rebuild_train_fn: Callable[[int], object],
    ) -> None:
        super().__init__()
        self.reload_every = int(reload_every)
        self.rebuild_train_fn = rebuild_train_fn
        self.trainer = None

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        if self.reload_every <= 0:
            return control
        step = int(state.global_step)
        if step <= 0 or step % self.reload_every != 0:
            return control
        trainer = self.trainer
        if trainer is None:
            return control
        new_ds = self.rebuild_train_fn(step)
        fmt = getattr(trainer, "_sft_formatting_func", None)
        if hasattr(trainer, "_prepare_dataset"):
            new_ds = trainer._prepare_dataset(
                new_ds,
                trainer.processing_class,
                trainer.args,
                trainer.args.packing,
                fmt,
                "train",
            )
        trainer.train_dataset = new_ds
        trainer._train_dataloader = None  # noqa: SLF001 — HF Trainer cache invalidation
        if getattr(trainer, "sampler", None) is not None:
            trainer.sampler = None
        return control


class ReloadTrainDatasetEachEpochCallback(TrainerCallback):
    """Reload train dataset at the start of each epoch (fallback if step-wise reload is brittle)."""

    def __init__(self, rebuild_train_fn: Callable[[int], object]) -> None:
        super().__init__()
        self.rebuild_train_fn = rebuild_train_fn
        self.trainer = None

    def on_epoch_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        trainer = self.trainer
        if trainer is None:
            return control
        new_ds = self.rebuild_train_fn(int(state.global_step))
        fmt = getattr(trainer, "_sft_formatting_func", None)
        if hasattr(trainer, "_prepare_dataset"):
            new_ds = trainer._prepare_dataset(
                new_ds,
                trainer.processing_class,
                trainer.args,
                trainer.args.packing,
                fmt,
                "train",
            )
        trainer.train_dataset = new_ds
        trainer._train_dataloader = None  # noqa: SLF001
        return control


def save_eval_state(path: Path, seed: int, eval_n: int, eval_hashes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"seed": seed, "eval_num_examples": eval_n, "eval_hashes": eval_hashes}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_eval_state(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


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
