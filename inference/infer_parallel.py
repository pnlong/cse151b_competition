#!/usr/bin/env python3
"""
Data-parallel inference: run inference/infer.py once per visible GPU, then merge CSVs.

GPU count is inferred from CUDA_VISIBLE_DEVICES (comma-separated physical IDs). If that
variable is unset, every CUDA device reported by PyTorch is used (device indices 0..N-1).

Multi-GPU behaviour:
  - Before sharding, collects completed question ids from the merged ``--output`` CSV and
    any existing ``*.shardK.csv`` files, then splits **only the remaining** questions
    across workers (round-robin). Each worker gets a small ``*.shardK.todo.jsonl`` subset.
  - Each shard writes full logs to ``<output-stem>.shard<K>.log`` (next to the CSV shards).
  - This driver prints a single periodic summary line on the terminal (CSV rows completed
    per shard vs quota), avoiding interleaved tqdm from multiple workers.
  - Shard ``*.shardK.csv`` and ``*.shardK.log`` files are **not** deleted after merge
    (kept for resume and debugging); only the merged ``--output`` CSV is written/updated.

Examples:
    CUDA_VISIBLE_DEVICES=0,1,2 python inference/infer_parallel.py --gpu
    python inference/infer_parallel.py --gpu --data data/public.jsonl --output results/out.csv

Any arguments are forwarded to infer.py except --tp / --num-shards / --shard-id / --output
/ --data, which this driver sets per worker. The merged file is written to the path implied
by --output (default: same as infer.py).
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
from pathlib import Path

# Repo root on sys.path (for config + load_jsonl only — no vLLM import here)
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from config import PRIVATE_DATA, RESULTS_DIR  # noqa: E402
from inference.utils import load_jsonl, save_jsonl  # noqa: E402

_INFER_SCRIPT = Path(__file__).resolve().parent / "infer.py"

# Interval between consolidated progress lines on the parent terminal (shard CSV row counts).
_PROGRESS_INTERVAL_SEC = 30 * 60  # 30 minutes

# Flags whose values this driver strips from the forwarded argv and rewrites per worker.
_STRIP_VALUE_FLAGS = frozenset({"--tp", "--num-shards", "--shard-id", "--output", "--data"})


def _get_flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        prefix = name + "="
        if a.startswith(prefix):
            return a[len(prefix) :]
    return None


def _has_flag(argv: list[str], name: str) -> bool:
    return name in argv or any(a.startswith(f"{name}=") for a in argv)


def _strip_value_flags(argv: list[str], flags: frozenset[str]) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in flags:
            if i + 1 < len(argv) and not argv[i + 1].startswith("-"):
                i += 2
            else:
                i += 1
            continue
        if any(a.startswith(f"{f}=") for f in flags):
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def _visible_cuda_device_ids() -> list[str]:
    """Physical GPU IDs to pass as CUDA_VISIBLE_DEVICES for one-GPU workers."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is not None:
        s = raw.strip()
        if not s:
            return []
        return [p.strip() for p in s.split(",") if p.strip()]
    try:
        import torch

        n = torch.cuda.device_count()
    except Exception as exc:
        raise RuntimeError(
            "Could not detect CUDA devices (set CUDA_VISIBLE_DEVICES or install PyTorch with CUDA)."
        ) from exc
    if n <= 0:
        return []
    return [str(i) for i in range(n)]


def _shard_path(final: Path, shard_id: int) -> Path:
    return final.parent / f"{final.stem}.shard{shard_id}{final.suffix}"


def _shard_todo_path(final: Path, shard_id: int) -> Path:
    return final.parent / f"{final.stem}.shard{shard_id}.todo.jsonl"


def _shard_log_path(final: Path, shard_id: int) -> Path:
    """Plain-text log for one worker (same basename as shard CSV, ``.log`` suffix)."""
    return final.parent / f"{final.stem}.shard{shard_id}.log"


def _csv_body_row_count(path: Path) -> int:
    """Number of logical data rows in a submission CSV (excluding header).

    Must use the CSV parser: ``response`` cells often contain newlines, so counting
    raw text lines massively over-counts (e.g. ``18097/447``).
    """
    if not path.exists():
        return 0
    with open(path, encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            next(reader)  # header: id,response
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def _load_csv_responses(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {str(row["id"]): row["response"] for row in csv.DictReader(f)}


def _load_dataset_rows(data_path: Path, limit: int | None) -> list[dict]:
    rows = load_jsonl(data_path)
    if limit is not None:
        rows = rows[:limit]
    return rows


def _ordered_question_ids(data_path: Path, limit: int | None) -> list[str]:
    return [str(r["id"]) for r in _load_dataset_rows(data_path, limit)]


def _collect_done_responses(
    final_csv: Path,
    shard_paths: list[Path],
    *,
    reset: bool,
) -> dict[str, str]:
    """Union completed rows from merged output and shard CSVs (unless --reset)."""
    if reset:
        return {}
    done = _load_csv_responses(final_csv)
    for sp in shard_paths:
        done.update(_load_csv_responses(sp))
    return done


def _split_todo_rows(
    rows: list[dict],
    done_ids: set[str],
    num_shards: int,
) -> tuple[list[list[dict]], list[int]]:
    """Round-robin split of dataset rows not yet in *done_ids*."""
    todo = [row for row in rows if str(row["id"]) not in done_ids]
    shard_rows: list[list[dict]] = [[] for _ in range(num_shards)]
    for i, row in enumerate(todo):
        shard_rows[i % num_shards].append(row)
    quotas = [len(s) for s in shard_rows]
    return shard_rows, quotas


def _write_final_csv(final_csv: Path, ordered_ids: list[str], responses: dict[str, str]) -> None:
    missing = [qid for qid in ordered_ids if qid not in responses]
    if missing:
        raise SystemExit(
            f"Merge failed: {len(missing)} question ids missing "
            f"(showing up to 5): {missing[:5]}"
        )
    final_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(final_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "response"])
        w.writeheader()
        for qid in ordered_ids:
            w.writerow({"id": qid, "response": responses[qid]})


def _merge_results(
    final_csv: Path,
    shard_paths: list[Path],
    ordered_ids: list[str],
    preload: dict[str, str],
) -> None:
    merged = dict(preload)
    for sp in shard_paths:
        merged.update(_load_csv_responses(sp))

    extras = set(merged.keys()) - set(ordered_ids)
    if extras:
        preview = sorted(extras)[:5]
        print(
            f"[infer_parallel] warning: ignoring {len(extras)} id(s) not in dataset "
            f"(showing up to 5): {preview}"
        )

    _write_final_csv(final_csv, ordered_ids, merged)


def _ensure_gpu_flag(forward: list[str]) -> list[str]:
    if "--gpu" in forward:
        return forward
    return ["--gpu", *forward]


def _spawn_workers(
    *,
    devices: list[str],
    forward_argv: list[str],
    final_output: Path,
    shard_todos: list[list[dict]],
    quotas: list[int],
) -> list[int]:
    n = len(devices)
    procs: list[subprocess.Popen] = []
    log_files: list[object] = []

    print(f"[infer_parallel] {n} workers — full logs:")
    for shard_id, phys in enumerate(devices):
        print(f"    shard {shard_id} (CUDA_VISIBLE_DEVICES={phys}): {_shard_log_path(final_output, shard_id)}")
    print(
        f"[infer_parallel] consolidated progress every {_PROGRESS_INTERVAL_SEC / 60:.0f} min "
        f"(CSV rows done / questions in shard)",
        flush=True,
    )

    stop_evt = threading.Event()

    def _progress_loop() -> None:
        while True:
            parts = []
            for k in range(n):
                done = _csv_body_row_count(_shard_path(final_output, k))
                parts.append(f"s{k}:{done}/{quotas[k]}")
            print(f"[infer_parallel] {' | '.join(parts)}", flush=True)
            if stop_evt.wait(_PROGRESS_INTERVAL_SEC):
                break

    prog_thread = threading.Thread(target=_progress_loop, daemon=True)
    prog_thread.start()

    try:
        for shard_id, phys in enumerate(devices):
            todo_path = _shard_todo_path(final_output, shard_id)
            save_jsonl(shard_todos[shard_id], todo_path)

            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = phys
            env["PYTHONUNBUFFERED"] = "1"
            env["INFER_PARALLEL_WORKER"] = "1"
            out_shard = _shard_path(final_output, shard_id)
            log_fp = open(_shard_log_path(final_output, shard_id), "w", encoding="utf-8")
            log_files.append(log_fp)
            child_argv = [
                *forward_argv,
                "--data",
                str(todo_path),
                "--tp",
                "1",
                "--num-shards",
                "1",
                "--shard-id",
                "0",
                "--output",
                str(out_shard),
            ]
            procs.append(
                subprocess.Popen(
                    [sys.executable, str(_INFER_SCRIPT), *child_argv],
                    cwd=str(_REPO_ROOT),
                    env=env,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                )
            )
        codes = [p.wait() for p in procs]
        parts = []
        for k in range(n):
            done = _csv_body_row_count(_shard_path(final_output, k))
            parts.append(f"s{k}:{done}/{quotas[k]}")
        print(f"[infer_parallel] final {' | '.join(parts)}", flush=True)
        return codes
    finally:
        stop_evt.set()
        prog_thread.join(timeout=_PROGRESS_INTERVAL_SEC + 2.0)
        for fp in log_files:
            try:
                fp.close()
            except Exception:
                pass


def main() -> None:
    argv = sys.argv[1:]
    if any(a in ("-h", "--help") for a in argv):
        subprocess.run([sys.executable, str(_INFER_SCRIPT), "--help"], cwd=str(_REPO_ROOT))
        return
    if not argv:
        subprocess.run([sys.executable, str(_INFER_SCRIPT), "--help"], cwd=str(_REPO_ROOT))
        return

    devices = _visible_cuda_device_ids()
    if not devices:
        raise SystemExit(
            "No CUDA devices found. Set CUDA_VISIBLE_DEVICES to a comma-separated list "
            "(e.g. 0,1) or ensure PyTorch sees at least one GPU."
        )

    out_s = _get_flag_value(argv, "--output")
    final_out = Path(out_s) if out_s else (RESULTS_DIR / "submission.csv")
    data_s = _get_flag_value(argv, "--data")
    data_path = Path(data_s) if data_s else PRIVATE_DATA
    lim_s = _get_flag_value(argv, "--limit")
    limit = int(lim_s) if lim_s is not None else None
    reset = _has_flag(argv, "--reset")

    forward_base = _strip_value_flags(argv, _STRIP_VALUE_FLAGS)

    if len(devices) == 1:
        subprocess.run(
            [sys.executable, str(_INFER_SCRIPT), *forward_base],
            cwd=str(_REPO_ROOT),
            check=True,
        )
        return

    forward_base = _ensure_gpu_flag(forward_base)

    n = len(devices)
    shard_paths = [_shard_path(final_out, k) for k in range(n)]
    dataset_rows = _load_dataset_rows(data_path, limit)
    ordered_ids = [str(r["id"]) for r in dataset_rows]
    preload = _collect_done_responses(final_out, shard_paths, reset=reset)
    done_ids = set(preload)
    shard_todos, quotas = _split_todo_rows(dataset_rows, done_ids, n)
    remaining = sum(quotas)

    print(f"[infer_parallel] {n} GPUs → data-parallel shards; merge → {final_out}")
    print(
        f"[infer_parallel] progress: {len(done_ids)}/{len(ordered_ids)} already done; "
        f"{remaining} remaining"
        + (" (--reset)" if reset else " (merged CSV + shard CSVs)")
    )
    if remaining:
        print(f"[infer_parallel] shard quotas: {quotas}")

    if remaining == 0:
        print("[infer_parallel] nothing to generate — writing merged CSV from existing results")
        _merge_results(final_out, shard_paths, ordered_ids, preload)
        print(f"[infer_parallel] Merged {len(ordered_ids)} rows → {final_out}")
        return

    codes = _spawn_workers(
        devices=devices,
        forward_argv=forward_base,
        final_output=final_out,
        shard_todos=shard_todos,
        quotas=quotas,
    )
    if any(c != 0 for c in codes):
        raise SystemExit(f"One or more shard workers failed (exit codes={codes}).")

    _merge_results(final_out, shard_paths, ordered_ids, preload)
    print(f"[infer_parallel] Merged {len(ordered_ids)} rows → {final_out}")


if __name__ == "__main__":
    main()
