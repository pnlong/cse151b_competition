#!/usr/bin/env python3
"""
Remap ``id`` fields in private distillation artifacts after ``private.jsonl`` is
reordered or extended.

Matches rows by exact ``question`` + ``options`` (not by old integer id), then
writes the id from the current ``private.jsonl``. Also rewrites
``private_traces.attempted.txt`` sidecars used by ``distill/collect.py`` resume.

Uses ``data/private.jsonl`` (current ids) and ``data/private.old.jsonl`` (old id
layout) by default.

Also remaps inference submission CSVs (``id,response``) so ``infer.py`` can
resume and fill only the new questions.

Dry-run (default — no files written):
    python distill/remap_private_ids.py

Remap traces + ``results/private_router_n4.csv``:
    python distill/remap_private_ids.py --inference-csv

Apply in place (``.bak`` backup per file):
    python distill/remap_private_ids.py --apply --inference-csv

Then resume inference for the 50 new private questions (no ``--reset``):
    CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize --use-router \\
        --output /deepfreeze/pnlong/school/cse151b/final/results/private_router_n4.csv

One model directory:
    python distill/remap_private_ids.py --apply \\
        --traces /deepfreeze/.../distillation/deepseek-r1-distill-qwen-14b/private_traces.jsonl
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import DISTILL_DIR, PRIVATE_DATA, PRIVATE_OLD_DATA, RESULTS_DIR
from distill.utils import load_jsonl, save_jsonl

DEFAULT_INFERENCE_CSV = RESULTS_DIR / "private_router_n4.csv"


def _normalize_options(options) -> tuple[str, ...] | None:
    if not options:
        return None
    if isinstance(options, list):
        return tuple(str(x) for x in options)
    return tuple(str(x) for x in options)


def record_key(record: dict) -> tuple[str, tuple[str, ...] | None]:
    return (record["question"], _normalize_options(record.get("options")))


def load_private_rows(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(f"private JSONL not found: {path}")
    return load_jsonl(path)


def load_old_private(path: Path) -> list[dict]:
    return load_private_rows(path)


def build_key_to_id(rows: list[dict]) -> dict[tuple[str, tuple[str, ...] | None], int]:
    mapping: dict[tuple[str, tuple[str, ...] | None], int] = {}
    for row in rows:
        key = record_key(row)
        if key in mapping:
            raise ValueError(
                f"Duplicate question key in {row.get('id')!r} and id {mapping[key]!r}"
            )
        mapping[key] = int(row["id"])
    return mapping


def build_old_id_to_new_id(
    old_rows: list[dict],
    new_key_to_id: dict[tuple[str, tuple[str, ...] | None], int],
) -> dict[str, int]:
    out: dict[str, int] = {}
    missing: list[str] = []
    for row in old_rows:
        old_id = str(row["id"])
        key = record_key(row)
        new_id = new_key_to_id.get(key)
        if new_id is None:
            missing.append(old_id)
            continue
        out[old_id] = new_id
    if missing:
        print(
            f"  warning: {len(missing)} old private ids have no match in new private.jsonl "
            f"(first few: {', '.join(missing[:5])})"
        )
    return out


def remap_traces(
    traces_path: Path,
    new_key_to_id: dict[tuple[str, tuple[str, ...] | None], int],
) -> tuple[list[dict], int, list[str]]:
    rows = load_jsonl(traces_path)
    remapped: list[dict] = []
    changed = 0
    unmatched: list[str] = []
    for row in rows:
        key = record_key(row)
        new_id = new_key_to_id.get(key)
        if new_id is None:
            unmatched.append(str(row.get("id")))
            remapped.append(row)
            continue
        old_id = row.get("id")
        if int(old_id) != int(new_id):
            row = dict(row)
            row["id"] = new_id
            changed += 1
        remapped.append(row)
    return remapped, changed, unmatched


def remap_attempted(
    attempted_path: Path,
    old_id_to_new_id: dict[str, int],
) -> tuple[list[str], int, list[str]]:
    if not attempted_path.is_file():
        return [], 0, []
    lines = [line.strip() for line in attempted_path.read_text().splitlines() if line.strip()]
    out: list[str] = []
    changed = 0
    missing: list[str] = []
    for old_id in lines:
        new_id = old_id_to_new_id.get(old_id)
        if new_id is None:
            missing.append(old_id)
            out.append(old_id)
            continue
        new_s = str(new_id)
        if new_s != old_id:
            changed += 1
        out.append(new_s)
    return out, changed, missing


def remap_submission_csv(
    csv_path: Path,
    old_id_to_new_id: dict[str, int],
) -> tuple[list[dict[str, str]], int, list[str], list[str], set[str]]:
    """Remap ``id`` column in an infer.py submission CSV."""
    with open(csv_path, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return [], 0, [], [], set()
    if "id" not in rows[0] or "response" not in rows[0]:
        raise ValueError(f"{csv_path} must have id,response columns")

    remapped: list[dict[str, str]] = []
    changed = 0
    unmapped: list[str] = []
    seen_new: set[str] = set()
    duplicate_new: list[str] = []
    covered_new: set[str] = set()
    for row in rows:
        old_id = str(row["id"])
        new_id = old_id_to_new_id.get(old_id)
        if new_id is None:
            unmapped.append(old_id)
            remapped.append({"id": old_id, "response": row["response"]})
            continue
        new_s = str(new_id)
        covered_new.add(new_s)
        if new_s in seen_new:
            duplicate_new.append(new_s)
        seen_new.add(new_s)
        if new_s != old_id:
            changed += 1
        remapped.append({"id": new_s, "response": row["response"]})

    remapped.sort(key=lambda r: int(r["id"]))
    return remapped, changed, unmapped, duplicate_new, covered_new


def write_submission_csv(rows: list[dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows(rows)


def missing_private_ids(
    new_rows: list[dict],
    covered_ids: set[str],
) -> list[str]:
    all_new = {str(row["id"]) for row in new_rows}
    return sorted(all_new - covered_ids, key=int)


def discover_trace_files(distill_dir: Path, explicit: list[Path] | None) -> list[Path]:
    if explicit:
        return explicit
    return sorted(distill_dir.glob("*/private_traces.jsonl"))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Remap private trace ids after private.jsonl reorder/extend"
    )
    p.add_argument(
        "--new-private",
        type=str,
        default=str(PRIVATE_DATA),
        help=f"Current private.jsonl (default: {PRIVATE_DATA})",
    )
    p.add_argument(
        "--old-private",
        type=str,
        default=str(PRIVATE_OLD_DATA),
        help=f"Previous private.jsonl (default: {PRIVATE_OLD_DATA})",
    )
    p.add_argument(
        "--distill-dir",
        type=str,
        default=str(DISTILL_DIR),
        help=f"Scan for */private_traces.jsonl under this dir (default: {DISTILL_DIR})",
    )
    p.add_argument(
        "--traces",
        type=str,
        action="append",
        default=None,
        help="Explicit private_traces.jsonl path (repeatable; skips --distill-dir scan)",
    )
    p.add_argument(
        "--csv",
        type=str,
        action="append",
        default=None,
        help="Submission CSV to remap (repeatable; id,response columns)",
    )
    p.add_argument(
        "--inference-csv",
        action="store_true",
        help=f"Shorthand for --csv {DEFAULT_INFERENCE_CSV}",
    )
    p.add_argument(
        "--skip-traces",
        action="store_true",
        help="Only remap --csv / --inference-csv targets (skip distillation traces)",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Write remapped files (default: dry-run summary only)",
    )
    p.add_argument(
        "--no-backup",
        action="store_true",
        help="With --apply, skip creating .bak copies before overwrite",
    )
    return p.parse_args()


def write_with_backup(path: Path, write_fn, *, backup: bool) -> None:
    if backup and path.exists():
        bak = path.with_suffix(path.suffix + ".bak")
        shutil.copy2(path, bak)
        print(f"    backup → {bak}")
    write_fn()


def main() -> None:
    args = parse_args()
    new_private = Path(args.new_private).resolve()
    old_private = Path(args.old_private).resolve()
    distill_dir = Path(args.distill_dir).resolve()

    csv_paths: list[Path] = [Path(p).resolve() for p in (args.csv or [])]
    if args.inference_csv:
        csv_paths.append(DEFAULT_INFERENCE_CSV.resolve())
    csv_paths = list(dict.fromkeys(csv_paths))

    if args.skip_traces:
        trace_paths: list[Path] = []
    elif args.traces:
        trace_paths = [Path(t).resolve() for t in args.traces]
    else:
        trace_paths = discover_trace_files(distill_dir, None)

    if not trace_paths and not csv_paths:
        raise SystemExit(
            "Nothing to remap. Pass --inference-csv / --csv, or ensure trace files exist "
            f"under {distill_dir}"
        )

    new_rows = load_private_rows(new_private)
    old_rows = load_old_private(old_private)
    new_key_to_id = build_key_to_id(new_rows)
    old_id_to_new_id = build_old_id_to_new_id(old_rows, new_key_to_id)

    print(f"New private: {new_private} ({len(new_rows)} rows)")
    print(f"Old private: {old_private} ({len(old_rows)} rows)")
    print(f"Trace files : {len(trace_paths)}")
    print(f"CSV files   : {len(csv_paths)}")
    print(f"Mode        : {'apply' if args.apply else 'dry-run'}")
    print()

    total_trace_changes = 0
    total_attempted_changes = 0
    total_csv_changes = 0
    covered_after_csv: set[str] = set()

    for traces_path in trace_paths:
        print(f"→ {traces_path}")
        remapped, n_changed, unmatched = remap_traces(traces_path, new_key_to_id)
        attempted_path = traces_path.with_suffix(".attempted.txt")
        attempted_out, n_attempted, missing_attempted = remap_attempted(
            attempted_path, old_id_to_new_id
        )
        total_trace_changes += n_changed
        total_attempted_changes += n_attempted

        print(f"    traces: {len(remapped)} rows, {n_changed} id updates")
        if unmatched:
            print(f"    traces unmatched by question: {len(unmatched)}")
        if attempted_path.is_file():
            print(
                f"    attempted sidecar: {len(attempted_out)} ids, "
                f"{n_attempted} remapped"
            )
            if missing_attempted:
                print(
                    f"    attempted ids missing from new private: "
                    f"{len(missing_attempted)}"
                )
        else:
            print("    attempted sidecar: (none)")

        if args.apply:
            backup = not args.no_backup

            def _write_traces() -> None:
                save_jsonl(remapped, traces_path)

            write_with_backup(traces_path, _write_traces, backup=backup)

            if attempted_path.is_file():

                def _write_attempted() -> None:
                    attempted_path.write_text(
                        "\n".join(attempted_out) + ("\n" if attempted_out else "")
                    )

                write_with_backup(attempted_path, _write_attempted, backup=backup)
        print()

    for csv_path in csv_paths:
        print(f"→ {csv_path}")
        if not csv_path.is_file():
            print("    missing — skipped")
            print()
            continue
        remapped, n_changed, unmapped, duplicate_new, covered_new = remap_submission_csv(
            csv_path, old_id_to_new_id
        )
        total_csv_changes += n_changed
        covered_after_csv.update(covered_new)

        print(f"    rows: {len(remapped)}, {n_changed} id updates")
        if unmapped:
            print(f"    unmapped old ids: {len(unmapped)}")
        if duplicate_new:
            print(f"    warning: duplicate new ids after remap: {len(duplicate_new)}")

        missing = missing_private_ids(new_rows, covered_after_csv)
        print(f"    still missing vs new private.jsonl: {len(missing)}")
        if missing:
            preview = ", ".join(missing[:12])
            suffix = "..." if len(missing) > 12 else ""
            print(f"      ids: {preview}{suffix}")

        if args.apply:

            def _write_csv() -> None:
                write_submission_csv(remapped, csv_path)

            write_with_backup(csv_path, _write_csv, backup=not args.no_backup)
        print()

    print(
        f"Done. {total_trace_changes} trace id updates, "
        f"{total_attempted_changes} attempted-id updates, "
        f"{total_csv_changes} csv id updates."
    )
    if csv_paths and covered_after_csv:
        still_missing = missing_private_ids(new_rows, covered_after_csv)
        if still_missing:
            csv_out = next((p for p in csv_paths if p.is_file()), csv_paths[0])
            print()
            print(
                f"Resume inference for {len(still_missing)} missing questions "
                f"(omit --reset; infer.py skips ids already in the CSV):"
            )
            print(
                f"  CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize "
                f"--use-router --output {csv_out}"
            )
    if not args.apply:
        print("Re-run with --apply to write changes (.bak backup unless --no-backup).")


if __name__ == "__main__":
    main()
