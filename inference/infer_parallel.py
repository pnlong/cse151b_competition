#!/usr/bin/env python3
"""
Data-parallel inference: run inference/infer.py once per visible GPU, then merge CSVs.

GPU count is inferred from CUDA_VISIBLE_DEVICES (comma-separated physical IDs). If that
variable is unset, every CUDA device reported by PyTorch is used (device indices 0..N-1).

Multi-GPU behaviour:
  - Each shard writes full logs to ``<output-stem>.shard<K>.log`` (next to the CSV shards).
  - This driver prints a single periodic summary line on the terminal (CSV rows completed
    per shard vs quota), avoiding interleaved tqdm from multiple workers.

Examples:
    CUDA_VISIBLE_DEVICES=0,1,2 python inference/infer_parallel.py --gpu
    python inference/infer_parallel.py --gpu --data data/public.jsonl --output results/out.csv

Any arguments are forwarded to infer.py except --tp / --num-shards / --shard-id / --output,
which this driver sets per worker. The merged file is written to the path implied by
--output (default: same as infer.py).
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
from inference.utils import load_jsonl  # noqa: E402

_INFER_SCRIPT = Path(__file__).resolve().parent / "infer.py"

# Seconds between consolidated progress lines on the parent terminal (shard CSV row counts).
_PROGRESS_INTERVAL_SEC = 12.0

# Flags whose values this driver strips from the forwarded argv and rewrites per worker.
_STRIP_VALUE_FLAGS = frozenset({"--tp", "--num-shards", "--shard-id", "--output"})


def _get_flag_value(argv: list[str], name: str) -> str | None:
    for i, a in enumerate(argv):
        if a == name and i + 1 < len(argv):
            return argv[i + 1]
        prefix = name + "="
        if a.startswith(prefix):
            return a[len(prefix) :]
    return None


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


def _shard_log_path(final: Path, shard_id: int) -> Path:
    """Plain-text log for one worker (same basename as shard CSV, ``.log`` suffix)."""
    return final.parent / f"{final.stem}.shard{shard_id}.log"


def _csv_body_row_count(path: Path) -> int:
    """Number of data rows in a submission CSV (excluding header)."""
    if not path.exists():
        return 0
    with open(path, encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def _per_shard_quotas(data_path: Path, limit: int | None, num_shards: int) -> list[int]:
    rows = load_jsonl(data_path)
    if limit is not None:
        rows = rows[:limit]
    counts = [0] * num_shards
    for i in range(len(rows)):
        counts[i % num_shards] += 1
    return counts


def _ordered_question_ids(data_path: Path, limit: int | None) -> list[str]:
    rows = load_jsonl(data_path)
    if limit is not None:
        rows = rows[:limit]
    return [str(r["id"]) for r in rows]


def _merge_shards(final_csv: Path, shard_paths: list[Path], ordered_ids: list[str]) -> None:
    merged: dict[str, str] = {}
    for sp in shard_paths:
        if not sp.exists():
            continue
        with open(sp, newline="") as f:
            for row in csv.DictReader(f):
                merged[str(row["id"])] = row["response"]

    missing = [qid for qid in ordered_ids if qid not in merged]
    if missing:
        raise SystemExit(
            f"Merge failed: {len(missing)} question ids missing from shard CSVs "
            f"(showing up to 5): {missing[:5]}"
        )

    extras = set(merged.keys()) - set(ordered_ids)
    if extras:
        raise SystemExit(
            f"Merge failed: shard CSVs contain unexpected ids (showing up to 5): "
            f"{sorted(extras)[:5]}"
        )

    final_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(final_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "response"])
        w.writeheader()
        for qid in ordered_ids:
            w.writerow({"id": qid, "response": merged[qid]})


def _ensure_gpu_flag(forward: list[str]) -> list[str]:
    if "--gpu" in forward:
        return forward
    return ["--gpu", *forward]


def _spawn_workers(
    *,
    devices: list[str],
    forward_argv: list[str],
    final_output: Path,
    data_path: Path,
    limit: int | None,
) -> list[int]:
    n = len(devices)
    quotas = _per_shard_quotas(data_path, limit, n)
    procs: list[subprocess.Popen] = []
    log_files: list[object] = []

    print(f"[infer_parallel] {n} workers — full logs:")
    for shard_id, phys in enumerate(devices):
        print(f"    shard {shard_id} (CUDA_VISIBLE_DEVICES={phys}): {_shard_log_path(final_output, shard_id)}")
    print(
        f"[infer_parallel] consolidated progress every {_PROGRESS_INTERVAL_SEC:.0f}s "
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
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = phys
            env["PYTHONUNBUFFERED"] = "1"
            env["INFER_PARALLEL_WORKER"] = "1"
            out_shard = _shard_path(final_output, shard_id)
            log_fp = open(_shard_log_path(final_output, shard_id), "w", encoding="utf-8")
            log_files.append(log_fp)
            child_argv = [
                *forward_argv,
                "--tp",
                "1",
                "--num-shards",
                str(n),
                "--shard-id",
                str(shard_id),
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

    forward_base = _strip_value_flags(argv, _STRIP_VALUE_FLAGS)

    if len(devices) == 1:
        subprocess.run(
            [sys.executable, str(_INFER_SCRIPT), *forward_base],
            cwd=str(_REPO_ROOT),
            check=True,
        )
        return

    forward_base = _ensure_gpu_flag(forward_base)

    print(f"[infer_parallel] {len(devices)} GPUs → data-parallel shards; merge → {final_out}")
    codes = _spawn_workers(
        devices=devices,
        forward_argv=forward_base,
        final_output=final_out,
        data_path=data_path,
        limit=limit,
    )
    if any(c != 0 for c in codes):
        raise SystemExit(f"One or more shard workers failed (exit codes={codes}).")

    ordered_ids = _ordered_question_ids(data_path, limit)
    shard_paths = [_shard_path(final_out, k) for k in range(len(devices))]
    _merge_shards(final_out, shard_paths, ordered_ids)
    print(f"[infer_parallel] Merged {len(ordered_ids)} rows → {final_out}")


if __name__ == "__main__":
    main()
