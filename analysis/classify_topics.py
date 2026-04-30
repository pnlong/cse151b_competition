#!/usr/bin/env python3
"""
Topic classifier CLI for the public and private math problem datasets.

Scoring rules live in ``topic_taxonomy`` at the repo root; this script only
handles I/O and invokes ``classify_problem`` (question + optional options text)
so labels match inference-time routing.

Output CSV columns:
  set      — "public" or "private"
  id       — problem id from the JSONL
  topic    — one of the 20 topic labels (see topic_taxonomy.CANONICAL_TOPIC_ORDER)

Usage (from the repo root):
    python analysis/classify_topics.py
    python analysis/classify_topics.py --output data/my_classifications.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config import PUBLIC_DATA, PRIVATE_DATA
from topic_taxonomy import classify_problem

DEFAULT_OUTPUT = REPO_ROOT / "data" / "topic_classifications.csv"


def _load_jsonl(path: Path) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify every problem into one of 20 topic categories.",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        metavar="PATH",
        help=f"Output CSV path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    rows: list[dict] = []

    for path, split_name in [(PUBLIC_DATA, "public"), (PRIVATE_DATA, "private")]:
        print(f"Classifying {split_name} set …")
        items = _load_jsonl(path)
        for item in items:
            topic = classify_problem(item["question"], item.get("options"))
            rows.append({"set": split_name, "id": item["id"], "topic": topic})
        print(f"  Done — {len(items):,} problems classified.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["set", "id", "topic"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nWrote {len(rows):,} rows → {args.output}")

    print("\n── Topic distribution ────────────────────────────────────────────")
    for split_name in ("public", "private"):
        split_rows = [r for r in rows if r["set"] == split_name]
        counts = Counter(r["topic"] for r in split_rows)
        n = len(split_rows)
        print(f"\n  {split_name.capitalize()} Set  (n = {n:,})")
        for topic, cnt in sorted(counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * cnt / n
            print(f"    {topic:<28}  {cnt:>5,}  ({pct:.1f}%)")


if __name__ == "__main__":
    main()
