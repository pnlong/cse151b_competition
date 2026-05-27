"""Shared LaTeX tabular formatting for analysis table scripts."""

from __future__ import annotations

import math


def latex_int(value: int | None) -> str:
    if value is None:
        return "---"
    return f"{value:,}".replace(",", "{,}")


def latex_pct(value: float | None, *, bold: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    text = f"{float(value):.2f}"
    return f"\\textbf{{{text}}}" if bold else text


def latex_kss(value: float | None, *, bold: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "---"
    text = f"{float(value):.3f}"
    return f"\\textbf{{{text}}}" if bold else text


def is_max(value: float | None, best: float | None, tol: float = 1e-9) -> bool:
    if value is None or best is None:
        return False
    if isinstance(value, float) and math.isnan(value):
        return False
    return abs(float(value) - float(best)) <= tol


def column_maxima(values: list[float | None]) -> float | None:
    nums = [
        float(v) for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    return max(nums) if nums else None


def pad_description(description: str, width: int) -> str:
    if len(description) >= width:
        return description
    return description + " " * (width - len(description))
