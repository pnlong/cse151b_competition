# CSE 151B Spring 2026 — Math Reasoning Competition

Maximize the mathematical reasoning accuracy of **Qwen3-4B** on 893 private math problems spanning high school to graduate level. See `scratchpaper/directions.md` for the full competition spec and submission format.

**Environment**: `micromamba activate cse151b_competition`  (create once with `bash setup.sh`)

---

## Strategy

1. **Prompt engineering + self-consistency** ✓ built
2. **Knowledge distillation** ✓ built — collect teacher traces → **SFT** ← *current stage*
3. **Reinforcement learning (GRPO)** starting from the SFT checkpoint

---

## Experiments

See [`EXPERIMENTS.md`](EXPERIMENTS.md) for every experimental condition, the files it involves, exact reproduction commands, and results logged to `STORAGE_DIR/results/eval_log.csv`.

| Experiment | Description | Status |
|------------|-------------|--------|
| 1a | Starter code (notebook baseline) | 🔲 to run |
| 1b | N=1 greedy, no self-consistency | 🔲 to run |
| 1c | Single system prompt, N=4 self-consistency | 🔲 to run |
| 1d | Prompt routing, N=4 self-consistency | 🔲 to run |
| 1e | Thinking mode off, N=4 self-consistency | 🔲 to run |
| 2 | Knowledge distillation + SFT | ⚙ pipeline built, training not started |
| 3 | Reinforcement learning (GRPO) | 🔲 not started |

---

## Repository Layout

```
final/
├── EXPERIMENTS.md                All experimental conditions, run commands, and results log
├── constants.py                  Project-wide constants (model ID, sampling params, prompts)
├── config.py                     Loads .env → ROOT_DIR, STORAGE_DIR, all derived paths
├── utils.py                      Math answer evaluation helpers (used by judger.py)
├── judger.py                     Competition-provided answer judging logic — do not modify
│
├── data/
│   ├── public.jsonl              1116 problems with ground-truth answers (local eval)
│   └── private.jsonl             893 problems without answers (leaderboard submission)
│
├── inference/                    Baseline inference pipeline → README inside
│   ├── starter.py                Starter-code baseline (Exp 1a) — faithful port of the notebook
│   ├── infer.py                  Run Qwen3-4B with self-consistency voting → submission CSV
│   ├── evaluate.py               Score a CSV against public.jsonl using Judger
│   ├── utils.py                  Inference utilities (extraction, voting, prompt building, I/O)
│   ├── router.py                 Optional prompt router (format-first + topic refinements)
│   └── README.md
│
├── prompts/                      System prompts and instruction templates → README inside
│   ├── routing/
│   │   └── prompts.py            Router-oriented prompt library (primary + secondary prompts)
│   └── distillation/
│       ├── teacher.md            Orchestrator prompt for Claude-based distillation
│       └── solver.md             Sub-agent prompt for batch problem solving
│
├── distill/                      Knowledge distillation pipeline → README inside
│   ├── collect.py                Run a teacher model, save verified/pseudo-labeled traces
│   ├── merge.py                  Combine all models' traces into one SFT JSONL dataset
│   ├── utils.py                  Distillation utilities (re-exports inference/utils + extras)
│   └── README.md
│
├── scratchpaper/                 Git-ignored notes and planning documents
│   ├── directions.md             Official competition spec (preserved from original README)
│   ├── game_plan.md              Strategy overview and next steps
│   ├── pipeline.md               End-to-end pipeline description for all stages
│   └── project_rules.md          Coding conventions and project norms
│
├── setup.sh                      One-time environment setup (micromamba + pip deps)
├── .env                          Local directory paths (git-ignored)
├── .env.example                  Committed template — fill in ROOT_DIR and STORAGE_DIR
└── starter_code_cse151b_comp.ipynb  Original starter notebook (reference only)
```

Large artifacts (model weights, generated traces, results CSVs, checkpoints) live in `STORAGE_DIR` defined in `.env` — not in this repo.

---

## Quick Start

```bash
# 1. Set up paths
cp .env.example .env    # fill in ROOT_DIR and STORAGE_DIR

# 2. Create environment (first time only)
bash setup.sh

# 3. Activate environment
micromamba activate cse151b_competition

# 3. Run inference on the private test set (submission)
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu

# 4. Run on public set and evaluate locally
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu \
    --data data/public.jsonl --output results/public.csv
python inference/evaluate.py --results results/public.csv
```

See `inference/README.md` for full options including multi-GPU, quantization, and smoke-testing.

---

## Root-Level Modules

### `constants.py`
All numerical, boolean, and string constants — model ID, sampling parameters, vLLM settings, system prompts. Every script imports defaults from here rather than hardcoding values.

### `prompts/routing/prompts.py`
Router-oriented prompt library. Defines:
- Primary, format-driven system prompts (`fr_single`, `fr_multi`, `mcq_single`)
- Optional secondary refinement snippets (stats/geometry/calculus/linear algebra)
- A lightweight router-classifier prompt that outputs strict JSON (used only if you enable LLM-based secondary routing)

### `config.py`
Loads `.env` and exposes `ROOT_DIR`, `STORAGE_DIR`, and every derived sub-path used across the project (`PRIVATE_DATA`, `PUBLIC_DATA`, `RESULTS_DIR`, `DISTILL_DIR`, `CHECKPOINTS_DIR`, `HF_CACHE_DIR`). Call `ensure_storage_dirs()` once to initialize the storage layout.

### `utils.py`
Math answer parsing and normalization helpers shared with `judger.py`: `last_boxed_only_string`, `remove_boxed`, `fix_sqrt`, `fix_fracs`, and related utilities. **Do not modify** — imported by the competition-provided `judger.py`.

### `judger.py`
Competition-provided answer judging logic. Handles symbolic equivalence, numeric approximation, unit stripping, ordered/unordered list matching, and MCQ scoring. **Do not modify.**
