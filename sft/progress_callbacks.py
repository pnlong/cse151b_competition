"""
Shared training progress + ``checkpoint-latest`` helpers (no matplotlib).

Used by ``sft/train.py`` and ``rl/train.py`` so RL does not import SFT plot callbacks.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

from tqdm.auto import tqdm
from transformers import TrainerCallback, TrainerControl, TrainerState
from transformers.trainer_callback import ProgressCallback
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from inference.utils import is_huggingface_hub_id, normalize_model_ref

CHECKPOINT_LATEST = "checkpoint-latest"
TRAINING_LOSS_HISTORY_CSV = "training_loss_history.csv"


def resolve_checkpoint_latest_path(path: Path | str) -> Path:
    """Resolve ``…/checkpoint-latest`` to the real ``checkpoint-{step}/`` folder.

    Training defaults to a **pointer file** (one line: relative dir name) because symlinks to
    directories are unreliable on some shared filesystems (``ls`` looks fine but traversal fails).

    Order: existing real directory with this name → symlink (``readlink``) → plain pointer file.
    If resolution fails, returns *path* unchanged.
    """
    p = Path(path).expanduser()
    if p.name != CHECKPOINT_LATEST:
        return p
    if not (p.exists() or p.is_symlink()):
        return p
    parent = p.parent
    if p.is_dir() and not p.is_symlink():
        return p
    if p.is_symlink():
        try:
            tgt = os.readlink(p)
        except OSError:
            return p
        cand = Path(tgt) if Path(tgt).is_absolute() else (parent / tgt)
        if cand.is_dir():
            return cand
        return p
    if p.is_file():
        try:
            name = p.read_text(encoding="utf-8").strip().splitlines()[0].strip()
        except OSError:
            return p
        if name:
            cand = parent / name
            if cand.is_dir():
                return cand
    return p


def resolve_base_and_adapter(model_path: Path | str) -> tuple[str, str | None]:
    """If *model_path* is a PEFT adapter dir, return (base_model, adapter_dir).

    Otherwise return (hub id or local path, None).
    """
    raw = str(model_path).strip()
    if is_huggingface_hub_id(raw):
        return raw, None

    p = Path(raw).expanduser().resolve()
    adapter_cfg = p / "adapter_config.json"
    if adapter_cfg.is_file():
        meta = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base = meta.get("base_model_name_or_path")
        if not base:
            raise ValueError(
                f"adapter_config.json missing base_model_name_or_path: {adapter_cfg}"
            )
        return str(base), str(p)
    return normalize_model_ref(model_path), None


def adapter_lora_rank(adapter_dir: Path | str) -> int:
    """Read LoRA rank ``r`` from a PEFT adapter directory."""
    adapter_cfg = Path(adapter_dir) / "adapter_config.json"
    meta = json.loads(adapter_cfg.read_text(encoding="utf-8"))
    return int(meta.get("r", 16))


_TRAIN_LOSS_CSV_FIELDS = ("global_step", "train_loss", "learning_rate", "mean_token_accuracy")


class TrainLossHistoryCallback(TrainerCallback):
    """
    Cheap training curves on disk: append CSV rows from ``on_log`` (same metrics HF logs).

    Includes ``mean_token_accuracy`` when TRL ``SFTTrainer`` provides it (masked next-token match on
    labels — not public-set Judge accuracy). Tune ``--logging-steps`` and ``every`` for cadence.

    No ``trainer.evaluate()`` during training.
    """

    def __init__(self, output_dir: Path, every: int) -> None:
        super().__init__()
        self.output_dir = Path(output_dir)
        self.every = max(1, int(every))
        self.csv_path = self.output_dir / TRAINING_LOSS_HISTORY_CSV

    def _migrate_csv_schema_if_needed(self) -> None:
        """Rewrite legacy 3-column CSVs once so ``mean_token_accuracy`` column aligns."""
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames or ()
            if "mean_token_accuracy" in fieldnames:
                return
            rows = list(reader)
        new_rows: list[dict[str, object]] = []
        for r in rows:
            new_rows.append(
                {
                    "global_step": r.get("global_step", ""),
                    "train_loss": r.get("train_loss", ""),
                    "learning_rate": r.get("learning_rate", ""),
                    "mean_token_accuracy": "",
                }
            )
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(_TRAIN_LOSS_CSV_FIELDS), extrasaction="ignore")
            w.writeheader()
            w.writerows(new_rows)
        print(
            f"[progress][train-loss-csv] Migrated {self.csv_path.name} (added mean_token_accuracy column).",
            flush=True,
        )

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return control
        if logs is None or "loss" not in logs:
            return control
        step = int(state.global_step)
        if step <= 0 or step % self.every != 0:
            return control

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._migrate_csv_schema_if_needed()
        write_header = not self.csv_path.exists()
        with open(self.csv_path, "a", newline="") as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(_TRAIN_LOSS_CSV_FIELDS),
                extrasaction="ignore",
            )
            if write_header:
                w.writeheader()
            lr = logs.get("learning_rate")
            mta = logs.get("mean_token_accuracy")
            w.writerow(
                {
                    "global_step": step,
                    "train_loss": round(float(logs["loss"]), 8),
                    "learning_rate": "" if lr is None else round(float(lr), 12),
                    "mean_token_accuracy": "" if mta is None else round(float(mta), 8),
                }
            )
        return control


class CheckpointChunkProgressCallback(TrainerCallback):
    """
    tqdm segments of ``save_steps`` optimizer steps (checkpoint cadence), restarting the bar each segment.
    Postfix shows loss / token accuracy / reward / entropy / kl when present in logs.
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
                    flush=True,
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
                flush=True,
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
        reward = logs.get("reward")
        entropy = logs.get("entropy")
        kl = logs.get("kl")
        parts: list[str] = []
        if loss is not None:
            parts.append(f"loss={float(loss):.4g}")
        if mta is not None:
            parts.append(f"mean_tok_acc={float(mta):.4g}")
        if reward is not None:
            parts.append(f"reward={float(reward):.4g}")
        if entropy is not None:
            parts.append(f"entropy={float(entropy):.4g}")
        if kl is not None:
            parts.append(f"kl={float(kl):.4g}")
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
    """Updates ``checkpoint-latest`` after each save (rank 0).

    Default: **pointer file** (one line: ``checkpoint-{step}``) — works on NFS-like mounts where
    directory symlinks break. Pass ``use_symlink=True`` for a real symlink on local disks.
    """

    def __init__(self, *, use_symlink: bool = False) -> None:
        super().__init__()
        self.use_symlink = bool(use_symlink)

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
            if self.use_symlink:
                link.symlink_to(target_name, target_is_directory=True)
                kind = "symlink"
            else:
                link.write_text(f"{target_name}\n", encoding="utf-8")
                kind = "pointer"
            print(f"[progress] checkpoint-latest ({kind}) → {target_name}/", flush=True)
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
