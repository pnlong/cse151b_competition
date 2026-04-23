#!/usr/bin/env python3
"""
Combine all teacher trace files into a single SFT-ready JSONL dataset.

Scans DISTILL_DIR for all {model}/public_traces.jsonl and (optionally)
{model}/private_traces.jsonl, converts each trace to the standard chat
format used by trl / transformers SFT trainers, shuffles, and saves.

Include private traces (default, controlled by INCLUDE_PRIVATE_IN_SFT):
    python distill/merge.py

Exclude private traces (override the constant):
    python distill/merge.py --no-private

Custom output path:
    python distill/merge.py --output /path/to/sft_data.jsonl
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from constants import (
    INCLUDE_PRIVATE_IN_SFT,
    DISTILL_SYSTEM_MATH,
    DISTILL_SYSTEM_MCQ,
    MULTI_ANS_NOTE,
)
from config import DISTILL_DIR
from distill.utils import load_jsonl, save_jsonl, build_prompt, count_ans_slots

random.seed(42)


# ── SFT format conversion ─────────────────────────────────────────────────────

def to_sft_record(trace: dict) -> dict:
    """
    Convert a raw trace record to the standard chat format expected by trl's
    SFTTrainer / DataCollatorForSeq2Seq.

    The assistant content is the full teacher response (including any
    <think>...</think> tokens), which is exactly what we want the student
    to learn to reproduce.
    """
    question = trace["question"]
    options  = trace.get("options")

    system, user = build_prompt(
        question, options,
        DISTILL_SYSTEM_MATH, DISTILL_SYSTEM_MCQ, MULTI_ANS_NOTE,
    )

    return {
        "messages": [
            {"role": "system",    "content": system},
            {"role": "user",      "content": user},
            {"role": "assistant", "content": trace["response"]},
        ]
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Merge distillation traces into SFT dataset")
    p.add_argument("--no-private", action="store_true",
                   help="Exclude private.jsonl pseudo-labeled traces "
                        "(overrides INCLUDE_PRIVATE_IN_SFT constant)")
    p.add_argument("--output",     default=str(DISTILL_DIR / "sft_data.jsonl"),
                   help="Output JSONL path (default: DISTILL_DIR/sft_data.jsonl)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    include_private = INCLUDE_PRIVATE_IN_SFT and not args.no_private

    # ── Collect trace files ───────────────────────────────────────────────────
    public_files  = sorted(DISTILL_DIR.glob("*/public_traces.jsonl"))
    private_files = sorted(DISTILL_DIR.glob("*/private_traces.jsonl")) if include_private else []

    if not public_files and not private_files:
        print(f"No trace files found under {DISTILL_DIR}/")
        print("Run distill/collect.py first.")
        return

    # ── Load and convert ──────────────────────────────────────────────────────
    sft_records = []
    summary: dict[str, dict[str, int]] = {}   # model_slug → {public, private}

    for f in public_files:
        slug = f.parent.name
        records = load_jsonl(f)
        sft_records.extend(to_sft_record(r) for r in records)
        summary.setdefault(slug, {"public": 0, "private": 0})
        summary[slug]["public"] += len(records)

    for f in private_files:
        slug = f.parent.name
        records = load_jsonl(f)
        sft_records.extend(to_sft_record(r) for r in records)
        summary.setdefault(slug, {"public": 0, "private": 0})
        summary[slug]["private"] += len(records)

    # ── Shuffle and save ──────────────────────────────────────────────────────
    random.shuffle(sft_records)
    out_path = Path(args.output)
    save_jsonl(sft_records, out_path)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 60)
    print("MERGE SUMMARY")
    print("=" * 60)
    print(f"  {'Model':<42} {'Public':>7} {'Private':>8}")
    print(f"  {'-'*42} {'-'*7} {'-'*8}")
    for slug, counts in sorted(summary.items()):
        priv_str = str(counts["private"]) if include_private else "—"
        print(f"  {slug:<42} {counts['public']:>7} {priv_str:>8}")
    print(f"  {'-'*42} {'-'*7} {'-'*8}")
    total_pub  = sum(c["public"]  for c in summary.values())
    total_priv = sum(c["private"] for c in summary.values()) if include_private else 0
    priv_total_str = str(total_priv) if include_private else "—"
    print(f"  {'TOTAL':<42} {total_pub:>7} {priv_total_str:>8}")
    print(f"\n  Grand total : {len(sft_records)} SFT records")
    print(f"  Private data: {'included' if include_private else 'EXCLUDED (--no-private)'}")
    print(f"\nSaved → {out_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
