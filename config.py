"""
Project configuration — paths and environment.

Loads .env from the repo root, then exposes ROOT_DIR, STORAGE_DIR,
and every derived sub-path used across the project.

Usage:
    from config import ROOT_DIR, PRIVATE_DATA, RESULTS_DIR
"""

import os
from pathlib import Path

# ── Load .env ──────────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent   # repo root

try:
    from dotenv import load_dotenv
    load_dotenv(_HERE / ".env")
except ImportError:
    # python-dotenv not installed — fall back to reading the file manually
    _env_file = _HERE / ".env"
    if _env_file.exists():
        for _line in _env_file.read_text().splitlines():
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _, _v = _line.partition("=")
                os.environ.setdefault(_k.strip(), _v.strip())

# ── HuggingFace token (optional but recommended for faster downloads) ──────────

HF_TOKEN = os.environ.get("HF_TOKEN", None)

# Xet staging cache — xet does atomic chunk reconstruction and needs a local
# POSIX filesystem. Point this to local disk (/data3 or /tmp) if STORAGE_DIR
# is on NFS; otherwise xet downloads silently produce empty/corrupt files.
# Override via HF_XET_CACHE in .env, or leave unset to use the HF default.
HF_XET_CACHE = os.environ.get("HF_XET_CACHE", None)

# ── Root directories ───────────────────────────────────────────────────────────

ROOT_DIR    = Path(os.environ.get("ROOT_DIR",    _HERE))
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", _HERE / "storage"))

# ── Data (always inside the repo root) ────────────────────────────────────────

DATA_DIR      = ROOT_DIR / "data"
PRIVATE_DATA  = DATA_DIR / "private.jsonl"   # private test set (no answers)
PUBLIC_DATA   = DATA_DIR / "public.jsonl"    # public set (with answers)

# ── Storage sub-directories ────────────────────────────────────────────────────
# These live on the storage node and are created on first access.

RESULTS_DIR      = STORAGE_DIR / "results"        # inference outputs / CSVs
DISTILL_DIR      = STORAGE_DIR / "distillation"   # teacher reasoning traces
CHECKPOINTS_DIR  = STORAGE_DIR / "checkpoints"    # SFT / RL model checkpoints
# HF_CACHE_DIR defaults to STORAGE_DIR/cache but can be overridden in .env
# to point at a local disk when STORAGE_DIR is on NFS (xet downloads need POSIX fs).
HF_CACHE_DIR     = Path(os.environ.get("HF_CACHE_DIR", STORAGE_DIR / "cache"))

def ensure_storage_dirs() -> None:
    """Create all storage sub-directories if they don't exist."""
    for d in (RESULTS_DIR, DISTILL_DIR, CHECKPOINTS_DIR, HF_CACHE_DIR):
        d.mkdir(parents=True, exist_ok=True)
