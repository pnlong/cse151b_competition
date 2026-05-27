#!/usr/bin/env python3
"""
Score a submission CSV against public.jsonl ground truth.

Usage:
    python inference/evaluate.py --results results/public.csv
    python inference/evaluate.py --results results/public.csv --verbose
    python inference/evaluate.py --results /tmp/test.csv --data data/public.jsonl
    python inference/evaluate.py --results results/public.csv --save results/public_eval.jsonl

Log stats to the comparison CSV (appends one row):
    python inference/evaluate.py --results results/public.csv \\
        --model "Qwen3-4B baseline" --n-samples 16 --notes "no fine-tuning"
"""

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Iterator

# ── Project imports ────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PUBLIC_DATA, RESULTS_DIR
from inference.utils import load_jsonl, save_results_jsonl, score_mcq

from judger import Judger

# Default path for the running comparison log
DEFAULT_LOG_CSV = RESULTS_DIR / "eval_log.csv"

LOG_FIELDNAMES = [
    "timestamp", "model", "n_samples", "checkpoint",
    "mcq_correct", "mcq_total", "mcq_acc",
    "free_correct", "free_total", "free_acc",
    "total_correct", "total_total", "overall_acc",
    "missing", "results_file", "notes",
]


# ── CSV logging ────────────────────────────────────────────────────────────────

def append_log_row(log_path: Path, row: dict) -> None:
    """
    Append one row to the comparison log CSV.
    Creates the file with a header if it does not yet exist.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not log_path.exists()
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDNAMES)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Score submission CSV on public set")
    p.add_argument("--results",    required=True,
                   help="Submission CSV with columns id, response")
    p.add_argument("--data",       default=str(PUBLIC_DATA),
                   help=f"Ground-truth JSONL  (default: {PUBLIC_DATA})")
    p.add_argument("--save",       default=None,
                   help="Write full per-question results to this JSONL path "
                        "(fields: id, is_mcq, gold, response, correct)")
    p.add_argument("--verbose",    action="store_true",
                   help="Print details for every wrong answer")

    # ── Comparison log ─────────────────────────────────────────────────────────
    p.add_argument("--log-csv",    default=str(DEFAULT_LOG_CSV),
                   help=f"Append one stats row to this CSV "
                        f"(default: {DEFAULT_LOG_CSV})")
    p.add_argument("--no-log",     action="store_true",
                   help="Skip writing to the log CSV")
    p.add_argument("--model",      default="",
                   help="Label for the model / run (e.g. 'Qwen3-4B baseline')")
    p.add_argument("--n-samples",  type=int, default=None,
                   help="Number of self-consistency samples used (for the log)")
    p.add_argument("--checkpoint", default="",
                   help="Checkpoint stage label (e.g. 'base', 'sft', 'rl')")
    p.add_argument("--notes",      default="",
                   help="Free-text notes appended to the log row")
    return p.parse_args()


def pct(num: int, den: int) -> float:
    return round(100 * num / den, 2) if den else 0.0


def score_single_question(task: tuple) -> dict:
    """
    Score one public-set question. Pickle-safe worker for multiprocessing.

    *task* is ``(qid, item, response, strict_extract)`` where *response* is
    ``None`` when the submission CSV lacks that id.
    """
    qid, item, response, strict_extract = task
    gold = item["answer"]
    is_mcq = bool(item.get("options"))

    if response is None:
        return {
            "id": item["id"],
            "is_mcq": is_mcq,
            "gold": gold,
            "response": None,
            "correct": False,
            "missing": True,
        }

    if is_mcq:
        correct = score_mcq(response, str(gold))
    else:
        gold_list = gold if isinstance(gold, list) else [gold]
        try:
            judger = Judger(strict_extract=strict_extract)
            correct = judger.auto_judge(
                pred=response,
                gold=gold_list,
                options=[[]] * len(gold_list),
            )
        except Exception:
            correct = False

    return {
        "id": item["id"],
        "is_mcq": is_mcq,
        "gold": gold,
        "response": response,
        "correct": correct,
        "missing": False,
    }


def aggregate_records(records: list[dict]) -> dict:
    """Summarize per-question *records* into milestone/eval_log metrics."""
    mcq_correct = mcq_total = 0
    free_correct = free_total = 0
    missing = 0

    for rec in records:
        if rec["missing"]:
            missing += 1
            continue
        if rec["is_mcq"]:
            mcq_total += 1
            mcq_correct += int(rec["correct"])
        else:
            free_total += 1
            free_correct += int(rec["correct"])

    total = mcq_total + free_total
    correct = mcq_correct + free_correct

    return {
        "mcq_correct": mcq_correct,
        "mcq_total": mcq_total,
        "mcq_acc": pct(mcq_correct, mcq_total),
        "free_correct": free_correct,
        "free_total": free_total,
        "free_acc": pct(free_correct, free_total),
        "total_correct": correct,
        "total_total": total,
        "overall_acc": pct(correct, total),
        "missing": missing,
    }


def run_scoring_tasks(
    tasks: list[tuple],
    *,
    workers: int = 1,
    show_progress: bool = False,
    progress_desc: str = "Scoring",
) -> list[dict]:
    """Run ``score_single_question`` over *tasks*, optionally in parallel."""
    if not tasks:
        return []

    if workers <= 1:
        iterator: Iterator[tuple] = tasks
        if show_progress:
            from tqdm import tqdm
            iterator = tqdm(tasks, desc=progress_desc, unit="q")
        return [score_single_question(task) for task in iterator]

    chunksize = max(1, len(tasks) // (workers * 4))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        mapped = pool.map(score_single_question, tasks, chunksize=chunksize)
        if show_progress:
            from tqdm import tqdm
            mapped = tqdm(mapped, total=len(tasks), desc=progress_desc, unit="q")
        return list(mapped)


def evaluate_submission(
    results_path: Path | str,
    data_path: Path | str | None = None,
    *,
    strict_extract: bool = False,
    workers: int = 1,
    show_progress: bool = False,
    progress_desc: str | None = None,
) -> tuple[dict, list[dict]]:
    """
    Score a submission CSV against a ground-truth JSONL.

    Returns ``(metrics, records)`` where *metrics* has count/accuracy fields and
    *records* holds one dict per scored question (same shape as ``--save`` JSONL).

    Set ``workers`` > 1 to judge free-form questions in parallel processes.
    """
    data_path = Path(data_path or PUBLIC_DATA)
    results_path = Path(results_path)

    gold_by_id = {str(item["id"]): item for item in load_jsonl(data_path)}

    preds_by_id: dict[str, str] = {}
    with open(results_path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            preds_by_id[str(row["id"])] = row["response"]

    tasks = [
        (qid, item, preds_by_id.get(qid), strict_extract)
        for qid, item in gold_by_id.items()
    ]
    desc = progress_desc or f"Scoring {results_path.name}"
    records = run_scoring_tasks(
        tasks,
        workers=workers,
        show_progress=show_progress,
        progress_desc=desc,
    )
    return aggregate_records(records), records


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    metrics, records = evaluate_submission(args.results, args.data)

    mcq_correct = metrics["mcq_correct"]
    mcq_total = metrics["mcq_total"]
    free_correct = metrics["free_correct"]
    free_total = metrics["free_total"]
    correct = metrics["total_correct"]
    total = metrics["total_total"]
    missing = metrics["missing"]

    if args.verbose:
        for rec in records:
            if not rec["correct"]:
                qid = rec["id"]
                print(f"\n── WRONG  id={qid}  MCQ={rec['is_mcq']} ──")
                print(f"  gold     : {rec['gold']}")
                print(f"  response : {rec['response'][:300]}")

    # ── Print summary ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("EVALUATION RESULTS")
    print("=" * 55)
    print(f"  MCQ        : {mcq_correct:4d} / {mcq_total:4d}  ({pct(mcq_correct, mcq_total):.2f}%)")
    print(f"  Free-form  : {free_correct:4d} / {free_total:4d}  ({pct(free_correct, free_total):.2f}%)")
    print(f"  Overall    : {correct:4d} / {total:4d}  ({pct(correct, total):.2f}%)")
    if missing:
        print(f"  Missing IDs: {missing}")
    print("=" * 55)

    # ── Append to comparison log ───────────────────────────────────────────────
    if not args.no_log:
        log_path = Path(args.log_csv)
        row = {
            "timestamp":    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model":        args.model,
            "n_samples":    args.n_samples if args.n_samples is not None else "",
            "checkpoint":   args.checkpoint,
            "mcq_correct":  mcq_correct,
            "mcq_total":    mcq_total,
            "mcq_acc":      pct(mcq_correct, mcq_total),
            "free_correct": free_correct,
            "free_total":   free_total,
            "free_acc":     pct(free_correct, free_total),
            "total_correct": correct,
            "total_total":  total,
            "overall_acc":  pct(correct, total),
            "missing":      missing,
            "results_file": args.results,
            "notes":        args.notes,
        }
        append_log_row(log_path, row)
        print(f"Logged  →  {log_path}")

    # ── Optionally save full per-question results ──────────────────────────────
    if args.save:
        save_results_jsonl(records, Path(args.save))
        print(f"Saved {len(records)} records  →  {args.save}")


if __name__ == "__main__":
    main()
