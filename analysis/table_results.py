#!/usr/bin/env python3
"""
Recompute milestone report Table (tab:results): public-set accuracy by experiment.

Scores each experiment's public inference CSV through the competition ``judger``
(via ``inference.evaluate.evaluate_submission``), so rerunning after ``judger``
changes refreshes MCQ / free-form / overall accuracy.

Kaggle submission scores (KSS) are **not** computable locally — private labels are
withheld. Pass ``--kss-csv`` pointing at a two-column ``exp,kss`` file (see
``analysis/kaggle_scores.csv``) to attach leaderboard subsample scores from
Kaggle after each private submission.

Usage (from ``cse151b/final``):

    micromamba run -n cse151b_competition python analysis/table_results.py
    micromamba run -n cse151b_competition python analysis/table_results.py \\
        --kss-csv analysis/kaggle_scores.csv
    micromamba run -n cse151b_competition python analysis/table_results.py --latex
    micromamba run -n cse151b_competition python analysis/table_results.py --workers 8
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from analysis.latex_format import (
    column_maxima,
    is_max,
    latex_kss,
    latex_pct,
    pad_description,
)
from config import PUBLIC_DATA, RESULTS_DIR
from inference.evaluate import aggregate_records, score_single_question
from inference.utils import load_jsonl

DEFAULT_KSS_CSV = REPO_ROOT / "analysis" / "kaggle_scores.csv"

# Maps milestone report rows → public result CSV under RESULTS_DIR.
EXPERIMENTS: list[dict] = [
    {
        "exp": "1a",
        "description": "Starter code (N = 1)",
        "description_latex": r"Starter code ($N = 1$)",
        "results_csv": "public_starter_baseline.csv",
        "private_results_csv": "private_starter_baseline.csv",
    },
    {
        "exp": "1b",
        "description": "Our prompts, N = 1",
        "description_latex": r"Our prompts, $N = 1$",
        "results_csv": "public_baseline_n1.csv",
        "private_results_csv": "private_baseline_n1.csv",
    },
    {
        "exp": "1c",
        "description": "Our prompts, N = 4 self-consistency",
        "description_latex": r"Our prompts, $N = 4$ self-consistency",
        "results_csv": "public_baseline_n4.csv",
        "private_results_csv": "private_baseline_n4.csv",
    },
    {
        "exp": "1d",
        "description": "Prompt routing, N = 4",
        "description_latex": r"Prompt routing, $N = 4$",
        "results_csv": "public_router_n4.csv",
        "private_results_csv": "private_router_n4.csv",
    },
    {
        "exp": "1e",
        "description": "Thinking mode off, N = 4",
        "description_latex": r"Thinking mode off, $N = 4$",
        "results_csv": "public_nothinking_n4.csv",
        "private_results_csv": "private_nothinking_n4.csv",
    },
    {
        "exp": "2a",
        "description": "SFT distilled, router, N = 4",
        "description_latex": r"SFT distilled, router, $N = 4$",
        "results_csv": "public_sft_n4.csv",
        "private_results_csv": "private_sft_n4.csv",
    },
    {
        "exp": "2b",
        "description": "SFT distilled, no router, N = 4",
        "description_latex": r"SFT distilled, no router, $N = 4$",
        "results_csv": "public_sft_n4_no_router.csv",
        "private_results_csv": "private_sft_n4_no_router.csv",
    },
    {
        "exp": "3a",
        "description": "GRPO, best ckpt., router, N = 4",
        "description_latex": r"GRPO, best ckpt., router, $N = 4$",
        "results_csv": "public_grpo_n4_bestreward.csv",
        "private_results_csv": "private_grpo_n4_bestreward.csv",
    },
    {
        "exp": "3b",
        "description": "GRPO, latest ckpt., router, N = 4",
        "description_latex": r"GRPO, latest ckpt., router, $N = 4$",
        "results_csv": "public_grpo_n4_latest.csv",
        "private_results_csv": "private_grpo_n4_latest.csv",
    },
    {
        "exp": "3c",
        "description": "GRPO, best ckpt., router, N = 8",
        "description_latex": r"GRPO, best ckpt., router, $N = 8$",
        "results_csv": "public_grpo_n8_bestreward.csv",
        "private_results_csv": "private_grpo_n8_bestreward.csv",
    },
]

# Insert a \midrule after these experiment ids (matches milestone_report.tex).
RESULTS_LATEX_MIDRULE_AFTER = {"1e", "2b"}


def load_kss_lookup(path: Path | None) -> dict[str, float]:
    if path is None or not path.is_file():
        return {}
    lookup: dict[str, float] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            exp = row.get("exp", "").strip()
            kss_raw = row.get("kss", "").strip()
            if exp and kss_raw:
                lookup[exp] = float(kss_raw)
    return lookup


def _load_predictions(csv_path: Path) -> dict[str, str]:
    preds: dict[str, str] = {}
    with open(csv_path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            preds[str(row["id"])] = row["response"]
    return preds


def _score_tagged_question(task: tuple) -> tuple[str, dict]:
    """Pickle-safe worker: ``(exp, qid, item, response, strict_extract)``."""
    exp, qid, item, response, strict = task
    rec = score_single_question((qid, item, response, strict))
    return exp, rec


def build_results_table(
    results_dir: Path,
    *,
    public_data: Path = PUBLIC_DATA,
    kss_lookup: dict[str, float] | None = None,
    workers: int = 1,
    show_progress: bool = True,
) -> pd.DataFrame:
    kss_lookup = kss_lookup or {}
    public_data = public_data.resolve()
    results_dir = results_dir.resolve()

    gold_by_id = {str(item["id"]): item for item in load_jsonl(public_data)}
    n_questions = len(gold_by_id)

    # Reserve output rows in experiment order; score in one batched pass.
    row_by_exp: dict[str, dict] = {}
    tasks: list[tuple] = []
    task_exp: list[str] = []

    for spec in EXPERIMENTS:
        csv_path = results_dir / spec["results_csv"]
        row: dict = {
            "exp": spec["exp"],
            "description": spec["description"],
            "description_latex": spec["description_latex"],
            "results_csv": str(csv_path),
            "kss": kss_lookup.get(spec["exp"]),
        }
        if not csv_path.is_file():
            row.update({
                "mcq_acc": None,
                "ff_acc": None,
                "overall_acc": None,
                "missing": None,
                "error": f"missing file: {csv_path}",
            })
            row_by_exp[spec["exp"]] = row
            continue

        preds = _load_predictions(csv_path)
        row_by_exp[spec["exp"]] = row
        for qid, item in gold_by_id.items():
            tasks.append((qid, item, preds.get(qid), False))
            task_exp.append(spec["exp"])

    n_exps = sum(
        1 for spec in EXPERIMENTS
        if (results_dir / spec["results_csv"]).is_file()
    )
    records_by_exp: dict[str, list[dict]] = defaultdict(list)

    if tasks:
        progress_desc = f"Judging {n_exps} exps × {n_questions} q"
        tagged_tasks = [
            (exp, qid, item, response, strict)
            for exp, (qid, item, response, strict) in zip(task_exp, tasks)
        ]
        chunksize = max(1, len(tagged_tasks) // max(workers * 4, 1))

        if workers <= 1:
            iterator = tagged_tasks
            if show_progress:
                from tqdm import tqdm
                iterator = tqdm(tagged_tasks, desc=progress_desc, unit="q")
            for task in iterator:
                exp, rec = _score_tagged_question(task)
                records_by_exp[exp].append(rec)
        else:
            from concurrent.futures import ProcessPoolExecutor
            from tqdm import tqdm

            with ProcessPoolExecutor(max_workers=workers) as pool:
                mapped = pool.map(_score_tagged_question, tagged_tasks, chunksize=chunksize)
                if show_progress:
                    mapped = tqdm(
                        mapped,
                        total=len(tagged_tasks),
                        desc=progress_desc,
                        unit="q",
                    )
                for exp, rec in mapped:
                    records_by_exp[exp].append(rec)

    rows: list[dict] = []
    for spec in EXPERIMENTS:
        row = row_by_exp[spec["exp"]]
        if "error" not in row:
            metrics = aggregate_records(records_by_exp[spec["exp"]])
            row.update({
                "mcq_acc": metrics["mcq_acc"],
                "ff_acc": metrics["free_acc"],
                "overall_acc": metrics["overall_acc"],
                "missing": metrics["missing"],
                "error": None,
            })
        rows.append(row)

    return pd.DataFrame(rows)


def paper_view_results(df: pd.DataFrame) -> pd.DataFrame:
    """Columns matching ``tab:results`` in milestone_report.tex."""
    return df[["exp", "description_latex", "mcq_acc", "ff_acc", "overall_acc", "kss"]].rename(columns={
        "exp": "Exp",
        "description_latex": "Description",
        "mcq_acc": "MCQ (%)",
        "ff_acc": "FF (%)",
        "overall_acc": "Overall (%)",
        "kss": "KSS",
    })


def format_results_latex(df: pd.DataFrame) -> str:
    """Return a copy-paste-ready ``tabular`` block for tab:results."""
    paper = paper_view_results(df)
    desc_width = max(len(str(row["Description"])) for _, row in paper.iterrows())

    best_mcq = column_maxima(paper["MCQ (%)"].tolist())
    best_ff = column_maxima(paper["FF (%)"].tolist())
    best_overall = column_maxima(paper["Overall (%)"].tolist())
    best_kss = column_maxima(paper["KSS"].tolist()) if paper["KSS"].notna().any() else None

    lines = [
        r"\begin{tabular}{ll|ccc|c}",
        r"    \toprule",
        r"    \textbf{Exp} & \textbf{Description} & \textbf{MCQ} (\%) "
        r"& \textbf{FF} (\%) & \textbf{Overall} (\%) & \textbf{KSS} \\",
        r"    \midrule",
    ]

    for _, row in paper.iterrows():
        mcq = row["MCQ (%)"]
        ff = row["FF (%)"]
        overall = row["Overall (%)"]
        kss = row["KSS"]

        desc = pad_description(str(row["Description"]), desc_width)
        lines.append(
            f"    {row['Exp']} & {desc} & "
            f"{latex_pct(mcq, bold=is_max(mcq, best_mcq))} & "
            f"{latex_pct(ff, bold=is_max(ff, best_ff))} & "
            f"{latex_pct(overall, bold=is_max(overall, best_overall))} & "
            f"{latex_kss(kss, bold=is_max(kss, best_kss))} \\\\"
        )
        if row["Exp"] in RESULTS_LATEX_MIDRULE_AFTER:
            lines.append(r"    \midrule")

    lines.extend([
        r"    \bottomrule",
        r"\end{tabular}",
    ])
    return "\n".join(lines)


def print_kaggle_submission_guide(
    results_dir: Path,
    kss_csv: Path | None,
    *,
    no_kss: bool,
    file=None,
) -> None:
    """
    Remind the user that KSS is manual and list private submission CSV paths.
    """
    results_dir = results_dir.resolve()
    print(file=file)
    print("=" * 72, file=file)
    print("KAGGLE SUBMISSION SCORES (KSS)", file=file)
    print("=" * 72, file=file)

    if no_kss:
        print(
            "KSS column omitted (--no-kss). Private submission paths are still listed below.",
            file=file,
        )
    elif kss_csv is not None and kss_csv.is_file():
        print(
            f"KSS values in the table above come from: {kss_csv.resolve()}",
            file=file,
        )
    else:
        print(
            f"KSS values not loaded (expected CSV: {DEFAULT_KSS_CSV}).",
            file=file,
        )

    print(file=file)
    print(
        "These KSS numbers are NOT computed locally — they come from Kaggle's "
        "random ~30% private subsample after you upload each CSV.",
        file=file,
    )
    print(
        "Re-upload and check Kaggle after judger or inference changes, then update "
        f"{DEFAULT_KSS_CSV} with the new scores.",
        file=file,
    )
    print(file=file)
    print("Private submission CSVs (upload these to Kaggle):", file=file)
    print(file=file)

    for spec in EXPERIMENTS:
        private_path = results_dir / spec["private_results_csv"]
        if private_path.is_file():
            status = "ok"
            path_str = str(private_path.resolve())
        else:
            status = "MISSING"
            path_str = str(private_path.resolve())
        print(f"  Exp {spec['exp']}  [{status}]", file=file)
        print(f"    {path_str}", file=file)

    print(file=file)
    print(
        "After each Kaggle upload, edit analysis/kaggle_scores.csv "
        "(columns: exp,kss) and re-run this script.",
        file=file,
    )
    print("=" * 72, file=file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build public evaluation results table (tab:results).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help=f"Directory with public_*.csv inference outputs (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=PUBLIC_DATA,
        help=f"Ground-truth JSONL (default: {PUBLIC_DATA})",
    )
    parser.add_argument(
        "--kss-csv",
        type=Path,
        default=DEFAULT_KSS_CSV,
        help=f"Optional exp,kss CSV from Kaggle submissions (default: {DEFAULT_KSS_CSV})",
    )
    parser.add_argument(
        "--no-kss",
        action="store_true",
        help="Skip loading KSS scores even if kaggle_scores.csv exists",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print a copy-paste-ready LaTeX tabular block (tab:results)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        metavar="N",
        help="Parallel worker processes for judger scoring (default: 1 = sequential)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable the progress bar",
    )
    args = parser.parse_args()

    if args.workers < 1:
        parser.error("--workers must be >= 1")

    kss_path = None if args.no_kss else args.kss_csv
    kss_lookup = load_kss_lookup(kss_path)

    df = build_results_table(
        args.results_dir.resolve(),
        public_data=args.data.resolve(),
        kss_lookup=kss_lookup,
        workers=args.workers,
        show_progress=not args.quiet,
    )

    if args.latex:
        print_kaggle_submission_guide(
            args.results_dir,
            kss_path,
            no_kss=args.no_kss,
            file=sys.stderr,
        )
        print(format_results_latex(df))
        return

    print(f"Results dir : {args.results_dir}")
    print(f"Public data : {args.data}")
    if kss_lookup:
        print(f"KSS source  : {kss_path}")
    else:
        print("KSS source  : (none — use --kss-csv or update analysis/kaggle_scores.csv)")
    print()
    print(paper_view_results(df).to_string(index=False))

    print_kaggle_submission_guide(
        args.results_dir,
        kss_path,
        no_kss=args.no_kss,
    )

    missing = df[df["error"].notna()]
    if not missing.empty:
        print("\nWarnings:", file=sys.stderr)
        for _, r in missing.iterrows():
            print(f"  {r['exp']}: {r['error']}", file=sys.stderr)


if __name__ == "__main__":
    main()
