#!/usr/bin/env python3
"""
Two-panel horizontal training loss figure for the final report (SFT + GRPO).

Reads ``training_loss_history.csv`` from SFT and RL checkpoint directories and
writes a single PDF suitable for ``\includegraphics`` at the top of the Experiments
section.

Usage (from ``cse151b/final``):

    cd cse151b/final
    micromamba run -n cse151b_competition python analysis/plot_training_losses.py
    micromamba run -n cse151b_competition python analysis/plot_training_losses.py \\
        --output scratchpaper/figs/training_losses.pdf
    micromamba run -n cse151b_competition python analysis/plot_training_losses.py \\
        --sft-csv /path/to/sft/training_loss_history.csv \\
        --rl-csv /path/to/rl/training_loss_history.csv \\
        --output scratchpaper/figs/training_losses.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt

from config import CHECKPOINTS_DIR

DEFAULT_SFT_CSV = CHECKPOINTS_DIR / "sft" / "training_loss_history.csv"
DEFAULT_RL_CSV = CHECKPOINTS_DIR / "rl" / "training_loss_history.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "scratchpaper" / "figs" / "training_losses.pdf"


def load_training_loss_history(path: Path) -> tuple[list[int], list[float]]:
    """Parse ``global_step`` / ``train_loss`` rows from a training history CSV."""
    if not path.is_file():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps: list[int] = []
    losses: list[float] = []
    for row in rows:
        try:
            step_raw = (row.get("global_step") or "").strip()
            loss_raw = (row.get("train_loss") or "").strip()
            if not step_raw or not loss_raw:
                continue
            steps.append(int(step_raw))
            losses.append(float(loss_raw))
        except (TypeError, ValueError):
            continue
    return steps, losses


def plot_loss_panel(
    ax,
    steps: list[int],
    losses: list[float],
    *,
    title: str,
    pending_label: str = "Training pending",
) -> None:
    if steps and losses:
        ax.plot(steps, losses, color="C0", linewidth=1.0, alpha=0.9)
        ax.set_xlabel("global step")
        ax.set_ylabel("train loss")
    else:
        ax.text(
            0.5,
            0.5,
            pending_label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=11,
            color="0.45",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SFT and GRPO training loss curves.")
    parser.add_argument("--sft-csv", type=Path, default=DEFAULT_SFT_CSV)
    parser.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    sft_steps, sft_losses = load_training_loss_history(args.sft_csv)
    rl_steps, rl_losses = load_training_loss_history(args.rl_csv)

    fig, (ax_sft, ax_rl) = plt.subplots(1, 2, figsize=(10, 3.5))
    plot_loss_panel(ax_sft, sft_steps, sft_losses, title="SFT (QLoRA)")
    plot_loss_panel(ax_rl, rl_steps, rl_losses, title="GRPO")

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(f"  SFT: {len(sft_steps)} points from {args.sft_csv}")
    print(f"  RL:  {len(rl_steps)} points from {args.rl_csv}")


if __name__ == "__main__":
    main()
