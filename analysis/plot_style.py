"""
Shared figure sizing and typography for report analysis plots.

Two-panel figures use a 3:1 width-to-height ratio (see ``plot_dataset_breakdown.py``).
"""

from __future__ import annotations

FIG_HEIGHT_MIN = 2.8
FIG_HEIGHT_PER_BAR = 0.21
FIG_HEIGHT_MAX = 8.2
FIG_WIDTH_ASPECT = 3.0

TITLE_FS = 12
AXIS_FS = 11
TICK_FS = 10

# SFT + GRPO training figure: 1.5× default height, same 3:1 aspect → (12.6, 4.2) inches
TRAINING_CURVES_FIG_HEIGHT = 3.8

# Backward-compatible alias (deprecated)
TRAINING_LOSS_FIG_HEIGHT = TRAINING_CURVES_FIG_HEIGHT


def two_panel_figsize(
    *,
    n_bars: int | None = None,
    fig_height: float | None = None,
) -> tuple[float, float]:
    """
    Return ``(width, height)`` for a 1×2 subplot figure.

    When *fig_height* is set, use it directly (width = 3 × height).
    When *n_bars* is given, height follows the dataset-breakdown formula
    (``max(2.8, min(n_bars * 0.21, 8.2))``). Otherwise use the minimum height (2.8).
    """
    if fig_height is not None:
        fig_h = fig_height
    elif n_bars is None:
        fig_h = FIG_HEIGHT_MIN
    else:
        fig_h = max(FIG_HEIGHT_MIN, min(n_bars * FIG_HEIGHT_PER_BAR, FIG_HEIGHT_MAX))
    return (FIG_WIDTH_ASPECT * fig_h, fig_h)
