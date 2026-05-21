#!/usr/bin/env python3
"""
Two-panel SFT + GRPO training figure for the final report.

Left: SFT ``train_loss`` from ``training_loss_history.csv``.
Right: GRPO ``reward`` (Judger outcome + format bonus) from the RL history CSV.

Usage (from ``cse151b/final``):

    cd cse151b/final
    micromamba run -n cse151b_competition python analysis/plot_sft_grpo_training.py
    micromamba run -n cse151b_competition python analysis/plot_sft_grpo_training.py \\
        --output scratchpaper/figs/sft_grpo_training.pdf
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

import matplotlib.pyplot as plt
import seaborn as sns

from analysis.plot_style import (
    AXIS_FS,
    TICK_FS,
    TITLE_FS,
    TRAINING_CURVES_FIG_HEIGHT,
    two_panel_figsize,
)
from config import CHECKPOINTS_DIR

DEFAULT_SFT_CSV = CHECKPOINTS_DIR / "sft" / "training_loss_history.csv"
DEFAULT_RL_CSV = CHECKPOINTS_DIR / "rl" / "training_loss_history.csv"
DEFAULT_OUTPUT = _REPO_ROOT / "scratchpaper" / "figs" / "sft_grpo_training.pdf"


def load_training_history(path: Path, value_column: str) -> tuple[list[int], list[float]]:
    """Parse ``global_step`` and *value_column* rows from a training history CSV."""
    if not path.is_file():
        return [], []
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    steps: list[int] = []
    values: list[float] = []
    for row in rows:
        try:
            step_raw = (row.get("global_step") or "").strip()
            value_raw = (row.get(value_column) or "").strip()
            if not step_raw or not value_raw:
                continue
            steps.append(int(step_raw))
            values.append(float(value_raw))
        except (TypeError, ValueError):
            continue
    return steps, values


def plot_metric_panel(
    ax,
    steps: list[int],
    values: list[float],
    *,
    title: str,
    ylabel: str,
    color: str,
    ylim: tuple[float, float] | None = None,
    pending_label: str = "Training pending",
) -> None:
    if steps and values:
        ax.plot(steps, values, color=color, linewidth=1.0, alpha=0.9)
        ax.set_xlabel("Global Step", fontsize=AXIS_FS)
        ax.set_ylabel(ylabel, fontsize=AXIS_FS)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        if ylim is not None:
            ax.set_ylim(*ylim)
    else:
        ax.text(
            0.5,
            0.5,
            pending_label,
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=AXIS_FS,
            color="0.45",
        )
        ax.set_xticks([])
        ax.set_yticks([])
    ax.set_title(title, fontsize=TITLE_FS, fontweight="bold", pad=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot SFT training loss and GRPO mean reward curves.",
    )
    parser.add_argument("--sft-csv", type=Path, default=DEFAULT_SFT_CSV)
    parser.add_argument("--rl-csv", type=Path, default=DEFAULT_RL_CSV)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    sft_steps, sft_losses = load_training_history(args.sft_csv, "train_loss")
    rl_steps, rl_rewards = load_training_history(args.rl_csv, "reward")

    # Match plot_dataset_breakdown.py: seaborn palette only (no set_theme).
    sft_color, rl_color = sns.color_palette("muted", n_colors=2)

    fig, (ax_sft, ax_rl) = plt.subplots(
        1, 2, figsize=two_panel_figsize(fig_height=TRAINING_CURVES_FIG_HEIGHT)
    )
    plot_metric_panel(
        ax_sft,
        sft_steps,
        sft_losses,
        title="SFT (QLoRA)",
        ylabel="Train Loss",
        color=sft_color,
    )
    plot_metric_panel(
        ax_rl,
        rl_steps,
        rl_rewards,
        title="GRPO",
        ylabel="Mean Reward",
        color=rl_color,
        ylim=(0.0, 1.05),
    )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, format="pdf", bbox_inches="tight", dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {args.output}")
    print(f"  SFT loss:  {len(sft_steps)} points from {args.sft_csv}")
    print(f"  GRPO reward: {len(rl_steps)} points from {args.rl_csv}")


if __name__ == "__main__":
    main()
