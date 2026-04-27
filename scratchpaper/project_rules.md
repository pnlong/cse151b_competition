# Project Rules & Conventions

Working norms for the CSE 151B competition project. All code additions should follow these guidelines.

---

## Environment

- **Micromamba environment**: `cse151b_competition`
  - Create and populate once with: `bash setup.sh` (from the repo root)
  - Activate before running any scripts: `micromamba activate cse151b_competition`
  - All dependencies (vLLM, transformers, trl, etc.) should be installed into this environment
  - Do not use `.venv` or system Python for this project
  - If adding a new dependency, add it to `setup.sh` as well

## GPU Usage

- Scripts accept a `--gpu` **boolean flag** to signal that GPU inference is intended
- **Device selection is always done externally** via `CUDA_VISIBLE_DEVICES`, never inside scripts:
  ```bash
  # Single GPU
  CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu

  # Two-GPU tensor parallel
  CUDA_VISIBLE_DEVICES=0,1 python inference/infer.py --gpu --tp 2
  ```
- Scripts must **not** assign `os.environ["CUDA_VISIBLE_DEVICES"]` internally
- Omitting `--gpu` should produce a CPU-compatible run (useful for smoke-tests and import checks)

---

## Directory Structure

The repo root should stay **clean and minimal**. Only project-wide infrastructure lives there:

```
final/                        ← repo root
├── constants.py              OK — project-wide constants
├── config.py                 OK — env/path loading
├── utils.py                  OK — math eval helpers (shared with judger.py)
├── judger.py                 OK — competition-provided, do not modify
├── .env / .env.example       OK — environment config
├── .gitignore
├── README.md
├── data/                     OK — raw datasets only
│   ├── public.jsonl
│   └── private.jsonl
│
├── inference/                ← baseline inference pipeline
├── distill/                  ← distillation data collection (to be built)
├── sft/                      ← supervised fine-tuning (to be built)
├── rl/                       ← reinforcement learning (to be built)
└── scratchpaper/             ← markdown notes, git-ignored
```

**Rules:**
- Each pipeline stage gets its **own subdirectory** (`inference/`, `distill/`, `sft/`, `rl/`)
- Each subdirectory has its own `utils.py` for stage-specific helpers and a `README.md`
- Stage-specific `utils.py` files should **import from the root** where applicable (e.g., `from utils import last_boxed_only_string`) rather than duplicating logic
- Do **not** add scripts directly to the repo root — they belong in a subdirectory
- Large artifacts (model weights, generated traces, results CSVs, checkpoints) go in `STORAGE_DIR`, not the repo

---

## Storage Layout

Defined by `STORAGE_DIR` in `.env`. The `config.py` module exposes all sub-paths.

```
$STORAGE_DIR/
├── results/          inference outputs, submission CSVs
├── distillation/     teacher reasoning traces for SFT
├── checkpoints/      SFT and RL model checkpoints
└── cache/            HuggingFace model cache (set via HF_HOME)
```

Call `from config import ensure_storage_dirs; ensure_storage_dirs()` once to initialize the layout.

---

## Constants & Configuration

- **All** numerical, boolean, and string constants belong in `constants.py` — never hardcode values inline in scripts
- File paths and environment-dependent settings belong in `config.py`, loaded from `.env`
- `.env` is **git-ignored**; `.env.example` is committed and kept up to date with all required variables
- API keys go in `.env` only — never commit them

---

## README Maintenance

- **`README.md` must always reflect what is actually in the repo** — not what is planned
- Any time a new subdirectory, script, or root-level module is added or significantly changed, update `README.md` immediately:
  - Add/update the entry in the repository layout tree
  - Add/update the description in the relevant section
- The layout tree in `README.md` shows only committed, working code — do not list planned future directories
- `scratchpaper/` docs (`game_plan.md`, etc.) are the right place for forward-looking plans; `README.md` is a live map of what exists now

---

## Documentation

Every file and function should be documented:

- **Each subdirectory** must have a `README.md` explaining:
  - What the subdirectory does conceptually
  - How each script works at a high level
  - Usage examples with concrete CLI invocations
  - A table of all public functions/utilities in `utils.py`

- **Each script** must have a module-level docstring with:
  - One-line summary
  - Concrete usage examples

- **Each non-trivial function** must have a docstring explaining what it does, its arguments, and its return value. Short utility functions with obvious names are exempt.

- **Constants** in `constants.py` must have inline comments explaining units or intent for anything non-obvious (e.g., `# fraction of GPU VRAM reserved for vLLM`)

---

## External Services

- No API keys are used in this project — all model access is either local (HuggingFace via vLLM) or manual (Claude/ChatGPT/Gemini web UI)
- `.env` contains only directory paths (`ROOT_DIR`, `STORAGE_DIR`) — do not add API keys to it

---

## Code Style

- Python 3.10+ — use `match`, `|` union types, etc. where they improve clarity
- Imports: stdlib → third-party → project-local, separated by blank lines; project imports always come after `sys.path.insert(0, ...)` for the repo root
- CLI scripts use `argparse`; defaults always reference constants from `constants.py`
- Prefer explicit over implicit: name the fields when constructing dicts for CSV/JSONL output
- No bare `except:` — catch at minimum `Exception` and log or re-raise

---

## Git

- `scratchpaper/` is git-ignored — use it freely for notes, experiments, planning
- `results/` and `$STORAGE_DIR` are not committed — only code and config templates
- Commit messages should describe *why*, not just *what*
