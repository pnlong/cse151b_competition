#!/usr/bin/env python3
"""
Recompute milestone report Table (tab:sft-dataset): SFT training corpus counts.

Counts judger-verified public traces and majority-vote private pseudo-labels per
teacher by scanning ``DISTILL_DIR/*/public_traces.jsonl`` and
``DISTILL_DIR/*/private_traces.jsonl`` — the same sources ``distill/merge.py``
uses before writing ``sft_data.jsonl``.

Public counts change when traces are re-collected after a ``judger`` update;
private counts are one row per attempted private question per teacher.

Usage (from ``cse151b/final``):

    micromamba run -n cse151b_competition python analysis/table_sft_dataset.py
    micromamba run -n cse151b_competition python analysis/table_sft_dataset.py --no-private
    micromamba run -n cse151b_competition python analysis/table_sft_dataset.py --latex
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.latex_format import latex_int
from config import DISTILL_DIR
from constants import INCLUDE_PRIVATE_IN_SFT
from distill.utils import load_jsonl

# Display order / labels for teachers in the milestone report table.
TEACHER_ORDER: list[tuple[str, str]] = [
    ("qwen3-8b", "Qwen3-8B"),
    ("deepseek-r1-distill-qwen-14b", "DeepSeek-R1-Distill-Qwen-14B"),
    ("deepseek-r1-distill-qwen-7b", "DeepSeek-R1-Distill-Qwen-7B"),
    ("claude-sonnet-4-6", "Claude Sonnet 4.6"),
]


def count_traces(distill_dir: Path, *, include_private: bool) -> pd.DataFrame:
    """Return teacher × split counts from trace JSONLs under *distill_dir*."""
    counts: dict[str, dict[str, int]] = {}

    for path in sorted(distill_dir.glob("*/public_traces.jsonl")):
        slug = path.parent.name
        counts.setdefault(slug, {"public": 0, "private": 0})
        counts[slug]["public"] += len(load_jsonl(path))

    if include_private:
        for path in sorted(distill_dir.glob("*/private_traces.jsonl")):
            slug = path.parent.name
            counts.setdefault(slug, {"public": 0, "private": 0})
            counts[slug]["private"] += len(load_jsonl(path))

    rows: list[dict] = []
    seen_slugs = set()

    for slug, teacher in TEACHER_ORDER:
        c = counts.get(slug, {"public": 0, "private": 0})
        total = c["public"] + (c["private"] if include_private else 0)
        rows.append({
            "teacher_slug": slug,
            "teacher": teacher,
            "public": c["public"],
            "private": c["private"] if include_private else None,
            "total": total,
        })
        seen_slugs.add(slug)

    for slug in sorted(counts):
        if slug in seen_slugs:
            continue
        c = counts[slug]
        total = c["public"] + (c["private"] if include_private else 0)
        rows.append({
            "teacher_slug": slug,
            "teacher": slug,
            "public": c["public"],
            "private": c["private"] if include_private else None,
            "total": total,
        })

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    combined = {
        "teacher_slug": "combined",
        "teacher": "Combined",
        "public": int(df["public"].sum()),
        "private": int(df["private"].sum()) if include_private else None,
        "total": int(df["total"].sum()),
    }
    return pd.concat([df, pd.DataFrame([combined])], ignore_index=True)


def paper_view_sft_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Columns matching ``tab:sft-dataset`` in milestone_report.tex."""
    teachers = df[df["teacher_slug"] != "combined"].copy()
    out = teachers[["teacher", "public", "private", "total"]].rename(columns={
        "teacher": "Teacher",
        "public": "Public",
        "private": "Private",
        "total": "Total",
    })
    combined = df[df["teacher_slug"] == "combined"].iloc[0]
    out = pd.concat([
        out,
        pd.DataFrame([{
            "Teacher": "Combined",
            "Public": combined["public"],
            "Private": combined["private"],
            "Total": combined["total"],
        }]),
    ], ignore_index=True)
    return out


def format_sft_dataset_latex(df: pd.DataFrame, *, include_private: bool) -> str:
    """Return a copy-paste-ready ``tabular`` block for tab:sft-dataset."""
    paper = paper_view_sft_dataset(df)
    teacher_rows = paper.iloc[:-1]
    combined = paper.iloc[-1]

    lines = [
        r"\begin{tabular}{lrrr}",
        r"    \toprule",
        r"    \textbf{Teacher} & \textbf{Public} & \textbf{Private} & \textbf{Total} \\",
        r"    \midrule",
    ]

    for _, row in teacher_rows.iterrows():
        priv_val = row["Private"]
        priv = latex_int(int(priv_val)) if include_private and pd.notna(priv_val) else "---"
        lines.append(
            f"    {row['Teacher']} & {latex_int(int(row['Public']))} & "
            f"{priv} & {latex_int(int(row['Total']))} \\\\"
        )

    priv_val = combined["Private"]
    priv = latex_int(int(priv_val)) if include_private and pd.notna(priv_val) else "---"
    lines.extend([
        r"    \midrule",
        f"    \\textbf{{Combined}} & {latex_int(int(combined['Public']))} & "
        f"{priv} & \\textbf{{{latex_int(int(combined['Total']))}}} \\\\",
        r"    \bottomrule",
        r"\end{tabular}",
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build SFT dataset breakdown table (tab:sft-dataset).",
    )
    parser.add_argument(
        "--distill-dir",
        type=Path,
        default=DISTILL_DIR,
        help=f"Root distillation directory (default: {DISTILL_DIR})",
    )
    parser.add_argument(
        "--no-private",
        action="store_true",
        help="Exclude private_traces.jsonl counts (overrides INCLUDE_PRIVATE_IN_SFT)",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a copy-paste-ready LaTeX tabular block (tab:sft-dataset)",
    )
    args = parser.parse_args()

    include_private = INCLUDE_PRIVATE_IN_SFT and not args.no_private
    df = count_traces(args.distill_dir.resolve(), include_private=include_private)

    if df.empty:
        print(f"No trace files found under {args.distill_dir}/")
        print("Run distill/collect.py first.")
        sys.exit(1)

    if args.latex:
        print(format_sft_dataset_latex(df, include_private=include_private))
        return

    print(f"Distillation dir : {args.distill_dir}")
    print(f"Private included : {include_private}")
    print()
    print(paper_view_sft_dataset(df).to_string(index=False))


if __name__ == "__main__":
    main()
