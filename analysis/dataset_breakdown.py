#!/usr/bin/env python3
"""
Dataset breakdown — topic charts from the inference router or from topic_classifications.csv.

Two modes (--source):

  router — Uses `RuleBasedRouter` secondary keyword tags (five narrow topics + “None”).
           Topics can overlap per problem. Matches inference-time prompt routing.

  csv    — Reads `data/topic_classifications.csv` (from `analysis/classify_topics.py`).
           Twenty mutually exclusive keyword-scored topics per problem.

In both modes, **format** breakdown (MCQ vs free-form single vs multi-[ANS]) is computed
from each problem’s JSON fields via `primary_route` (same primary routing as inference).

Usage (from the repo root):
    python analysis/dataset_breakdown.py
    python analysis/dataset_breakdown.py --source router --output analysis/breakdown_router.pdf
    python analysis/dataset_breakdown.py --source csv --plot-top 10 --output analysis/breakdown_topics.pdf
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

# ── Repo root on sys.path so project modules are importable ────────────────────
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
_ANALYSIS_DIR = REPO_ROOT / "analysis"
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from classify_topics import CANONICAL_TOPIC_ORDER
from config import PUBLIC_DATA, PRIVATE_DATA
from inference.router import RuleBasedRouter, primary_route

# ── Label mappings ─────────────────────────────────────────────────────────────

_PRIMARY_LABELS: dict[str, str] = {
    "mcq_single": "MCQ",
    "fr_single":  "Free-form (single answer)",
    "fr_multi":   "Free-form (multi-answer)",
}

# Router secondary labels (display order for bars)
_ROUTER_TOPIC_ORDER: list[str] = [
    "Linear Algebra",
    "Stats – Descriptive",
    "Stats – Inference",
    "Calculus",
    "Geometry",
    "None",
]

_SECONDARY_TO_TOPIC: dict[str, str] = {
    "geometry":          "Geometry",
    "calculus":          "Calculus",
    "stats_inference":   "Stats – Inference",
    "stats_descriptive": "Stats – Descriptive",
    "linear_algebra":    "Linear Algebra",
}

DEFAULT_CLASSIFICATIONS_CSV = REPO_ROOT / "data" / "topic_classifications.csv"

# ── I/O helpers ───────────────────────────────────────────────────────────────

def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _load_topic_lookup(csv_path: Path) -> dict[tuple[str, int], str]:
    """Map (set_name, problem_id) → topic label from classifications CSV."""
    lookup: dict[tuple[str, int], str] = {}
    with open(csv_path, newline="") as fh:
        for row in csv.DictReader(fh):
            split = row["set"].strip().lower()
            pid = int(row["id"])
            lookup[(split, pid)] = row["topic"].strip()
    return lookup


# ── Analysis ──────────────────────────────────────────────────────────────────

class AnalysisResult:
    """Per-split counters for topics and answer formats."""

    def __init__(self, topic_counts: Counter, format_counts: Counter, n_total: int) -> None:
        self.topic_counts  = topic_counts
        self.format_counts = format_counts
        self.n_total       = n_total


def _format_counts_for_items(items: list[dict]) -> Counter:
    fmt = Counter()
    for item in items:
        p = primary_route(item["question"], item.get("options"))
        fmt[_PRIMARY_LABELS[p]] += 1
    return fmt


def analyze_router(items: list[dict]) -> AnalysisResult:
    """Secondary-keyword topics from RuleBasedRouter (may overlap); 'None' if no tag."""
    router = RuleBasedRouter(enable_secondary_keywords=True)
    topic_counts: Counter = Counter()
    format_counts: Counter = Counter()

    for item in items:
        dec = router.route_one(item["question"], item.get("options"))
        format_counts[_PRIMARY_LABELS.get(dec.primary, dec.primary)] += 1

        if dec.secondary:
            for tag in dec.secondary:
                mapped = _SECONDARY_TO_TOPIC.get(tag)
                if mapped:
                    topic_counts[mapped] += 1
        else:
            topic_counts["None"] += 1

    return AnalysisResult(topic_counts, format_counts, len(items))


def analyze_csv(
    items: list[dict],
    split_name: str,
    lookup: dict[tuple[str, int], str],
) -> AnalysisResult:
    """One topic per row from CSV; format from primary_route on JSONL."""
    topic_counts: Counter = Counter()
    format_counts = _format_counts_for_items(items)
    missing = 0

    sn = split_name.lower()
    for item in items:
        pid = item["id"]
        topic = lookup.get((sn, pid))
        if topic is None:
            missing += 1
            topic = "__MISSING__"
        topic_counts[topic] += 1

    if missing:
        print(
            f"  Warning: {missing} problem(s) in '{split_name}' split "
            f"had no matching row in the classifications CSV.",
            file=sys.stderr,
        )

    return AnalysisResult(topic_counts, format_counts, len(items))


# ── Text summary ───────────────────────────────────────────────────────────────

def _print_format_summary(result: AnalysisResult, split_name: str) -> None:
    n = result.n_total
    print(f"\n  {split_name}  (n = {n:,})")
    print(f"    {'Format':<30}  {'Count':>6}  {'%':>6}")
    print(f"    {'-'*30}  {'-'*6}  {'-'*6}")
    for label in _PRIMARY_LABELS.values():
        cnt = result.format_counts.get(label, 0)
        pct = 100.0 * cnt / n if n else 0.0
        print(f"    {label:<30}  {cnt:>6,}  {pct:>5.1f}%")


def _print_topic_table(
    result: AnalysisResult,
    topic_order: list[str],
    *,
    split_title: str,
) -> None:
    print(f"\n  {split_title}  (n = {result.n_total:,})")
    for topic in topic_order:
        cnt = result.topic_counts.get(topic, 0)
        pct = 100.0 * cnt / result.n_total if result.n_total else 0.0
        print(f"    {topic:<30}  {cnt:>5,}  ({pct:.1f}%)")


def _select_plot_topics(
    active: list[str],
    private: AnalysisResult,
    plot_top: int,
) -> tuple[list[str], list[str]]:
    """
    Choose which topics appear in the figure.

    When ``plot_top`` > 0, keep only the ``plot_top`` topics with the largest
    private-set counts (ties broken by topic name). Otherwise keep all ``active``.

    Returns ``(display_order, excluded)`` where ``display_order`` is sorted for
    ``barh`` (ascending private count → rarest at bottom, most common at top).
    """
    ranked = sorted(active, key=lambda t: (-private.topic_counts.get(t, 0), t))
    if plot_top <= 0 or plot_top >= len(ranked):
        chosen = ranked
        excluded: list[str] = []
    else:
        chosen = ranked[:plot_top]
        excluded = ranked[plot_top:]

    display_order = sorted(chosen, key=lambda t: (private.topic_counts.get(t, 0), t))
    return display_order, excluded


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot(
    public: AnalysisResult,
    private: AnalysisResult,
    topic_order_full: list[str],
    output: Path | None,
    *,
    plot_top: int = 0,
) -> list[str]:
    """
    Horizontal bar chart; only topics with nonzero total count across both splits.

    Categories are ordered by **private-set** frequency (ascending among plotted rows):
    least common near the bottom, **most common at the top**.

    If ``plot_top`` > 0, only the ``plot_top`` topics with the largest private-set
    counts are shown (figure only). Stdout tables are unaffected; omitted topics are
    listed on stdout after the tables for use in captions.

    Returns topics omitted from the figure when ``plot_top`` limits apply.
    """
    combined_total = Counter()
    combined_total.update(public.topic_counts)
    combined_total.update(private.topic_counts)

    active = [t for t in topic_order_full if combined_total.get(t, 0) > 0]
    extras = [
        k for k in combined_total
        if k not in topic_order_full and combined_total[k] > 0
    ]
    active.extend(extras)

    plot_topics, excluded_from_plot = _select_plot_topics(active, private, plot_top)

    if not plot_topics:
        print("Nothing to plot (no topic counts).", file=sys.stderr)
        return excluded_from_plot

    palette = sns.color_palette("muted", n_colors=len(plot_topics))

    n_bars = len(plot_topics)
    # 3:1 aspect ratio (width = 3 × height)
    fig_h = max(2.8, min(n_bars * 0.21, 8.2))
    fig_w = 3.0 * fig_h

    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        sharey=True,
    )

    _title_fs = 12
    _axis_fs = 11
    _tick_fs = 10
    _ann_fs = 10

    for ax, (result, split_name) in zip(
        axes,
        [(public, "Public Set"), (private, "Private Set")],
    ):
        n = result.n_total
        values = [result.topic_counts.get(t, 0) for t in plot_topics]
        max_val = max(values) if any(v > 0 for v in values) else 1

        ax.set_xlim(0, max_val * 1.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True, nbins=5))
        ax.set_axisbelow(True)
        ax.grid(
            axis="x",
            which="major",
            linestyle="-",
            linewidth=0.55,
            color="0.72",
            alpha=0.65,
            zorder=0,
        )

        bars = ax.barh(
            plot_topics, values,
            color=palette,
            edgecolor="white",
            linewidth=0.6,
            zorder=2,
        )

        for bar, val in zip(bars, values):
            if val == 0:
                continue
            pct = 100.0 * val / n
            ax.text(
                bar.get_width() + max_val * 0.015,
                bar.get_y() + bar.get_height() / 2.0,
                f"{val:,}  ({pct:.1f}%)",
                va="center", ha="left", fontsize=_ann_fs,
            )

        ax.set_title(f"{split_name}  (n = {n:,})", fontsize=_title_fs, fontweight="bold", pad=8)
        ax.set_xlabel("Number of Problems", fontsize=_axis_fs)
        ax.tick_params(axis="both", labelsize=_tick_fs)

    axes[0].set_ylabel("Topic", fontsize=_axis_fs)

    fig.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, bbox_inches="tight", dpi=150)
        print(f"\nSaved → {output}")
    else:
        plt.show()

    return excluded_from_plot


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot topic distribution and print format summary for both dataset splits.",
    )
    parser.add_argument(
        "--source",
        choices=("router", "csv"),
        default="router",
        help="router: inference keyword secondary tags (default). "
             "csv: topic_classifications.csv from classify_topics.py.",
    )
    parser.add_argument(
        "--classifications",
        type=Path,
        default=DEFAULT_CLASSIFICATIONS_CSV,
        metavar="PATH",
        help=f"CSV path when --source csv (default: {DEFAULT_CLASSIFICATIONS_CSV}).",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        metavar="PATH",
        help="Save the figure to this path (e.g. analysis/breakdown.pdf). "
             "Displayed interactively if omitted.",
    )
    parser.add_argument(
        "--plot-top",
        type=int,
        default=0,
        metavar="N",
        help="Figure only: show only the N topics with the largest counts on the "
             "**private** split (0 = show all). Stdout tables are unchanged.",
    )
    args = parser.parse_args()
    if args.plot_top < 0:
        parser.error("--plot-top must be >= 0")

    print("Loading datasets …")
    public_items  = _load_jsonl(PUBLIC_DATA)
    private_items = _load_jsonl(PRIVATE_DATA)
    print(f"  Public : {len(public_items):,} problems")
    print(f"  Private: {len(private_items):,} problems")

    if args.source == "router":
        print("Analyzing (inference router secondary keywords) …")
        public_result  = analyze_router(public_items)
        private_result = analyze_router(private_items)
        topic_order_full = list(_ROUTER_TOPIC_ORDER)
        router_mode = True
        topic_header = (
            "── Topic breakdown (router secondary keywords; "
            "may overlap — counts need not sum to n) ──"
        )
    else:
        if not args.classifications.is_file():
            parser.error(
                f"classifications file not found: {args.classifications}\n"
                "  Generate it with:  python analysis/classify_topics.py",
            )
        print(f"Analyzing (CSV topics from {args.classifications}) …")
        lookup = _load_topic_lookup(args.classifications)
        public_result  = analyze_csv(public_items, "public", lookup)
        private_result = analyze_csv(private_items, "private", lookup)
        topic_order_full = list(CANONICAL_TOPIC_ORDER)
        router_mode = False
        topic_header = (
            "── Topic breakdown (topic_classifications.csv; "
            "one topic per problem) ──"
        )

    print("\n── Format breakdown (primary_route on JSONL) ─────────────────────")
    _print_format_summary(public_result,  "Public Set")
    _print_format_summary(private_result, "Private Set")

    print(f"\n{topic_header}")
    if router_mode:
        print("  (Router topics may overlap; bar totals need not sum to n per split.)")
    else:
        print("  (CSV assigns exactly one topic per problem; counts sum to n per split.)")

    _print_topic_table(public_result,  topic_order_full, split_title="Public Set")
    _print_topic_table(private_result, topic_order_full, split_title="Private Set")

    excluded = plot(
        public_result,
        private_result,
        topic_order_full,
        args.output,
        plot_top=args.plot_top,
    )

    if args.plot_top > 0 and excluded:
        topics_str = ", ".join(f'"{t}"' for t in sorted(excluded))
        print(
            f"\n── Figure note (--plot-top {args.plot_top}) ─────────────────────────────\n"
            f"  Topics omitted from the plot (not among the top {args.plot_top} by "
            f"**private-set** count): {topics_str}",
        )


if __name__ == "__main__":
    main()
