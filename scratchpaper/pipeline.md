# Full Pipeline: Baseline → Distillation → SFT → RL → Submission

End-to-end description of every step from raw data to final `submission.csv`.

---

## Overview

```
data/public.jsonl          data/private.jsonl
     │                           │
     ▼                           ▼
┌─────────────────────────────────────────────┐
│  STAGE 1: Baseline Inference               │
│  inference/infer.py  (Qwen3-4B, N=4 vote)  │
└─────────────────────────────────────────────┘
     │                           │
     ▼                           │
 baseline accuracy               │
 (evaluate.py)                   ▼
                         submission.csv  ← submit this as baseline
                         
     ┌───────────────────────────────────────────────────┐
     │  STAGE 2: Knowledge Distillation                  │
     │  distill/collect.py  (one run per teacher model)  │
     └───────────────────────────────────────────────────┘
          │ public_traces.jsonl    │ private_traces.jsonl
          │ (Judger-verified)      │ (majority-vote pseudo-label)
          ▼                        ▼
     ┌─────────────────────────────────────────┐
     │  distill/merge.py                       │
     │  → DISTILL_DIR/sft_data.jsonl           │
     └─────────────────────────────────────────┘
                        │
                        ▼
     ┌─────────────────────────────────────────┐
     │  STAGE 3: Supervised Fine-Tuning (SFT)  │
     │  sft/train.py  (LoRA on Qwen3-4B)       │
     └─────────────────────────────────────────┘
                        │
                        ▼ SFT checkpoint
     ┌─────────────────────────────────────────┐
     │  STAGE 4: Reinforcement Learning (GRPO) │
     │  rl/train.py                            │
     └─────────────────────────────────────────┘
                        │
                        ▼ RL checkpoint
     ┌─────────────────────────────────────────┐
     │  STAGE 5: Final Inference               │
     │  inference/infer.py  (N=4 vote)        │
     └─────────────────────────────────────────┘
                        │
                        ▼
                 submission.csv  ← final leaderboard submission
```

---

## Stage 1: Baseline Inference

**Goal**: establish a score to beat; gives us a lower bound on the current model without any fine-tuning.

**Scripts**: `inference/infer.py`, `inference/evaluate.py`

**Workflow**: iterate exclusively on `public.jsonl` (ground truth available → can score locally with `evaluate.py`) until we have our best solution, then run `private.jsonl` exactly once for the final submission.

**What happens**:
1. Load `data/public.jsonl` (or `private.jsonl` for the final submission run)
2. For each question, build a prompt using `SYSTEM_MATH` / `SYSTEM_MCQ` from `constants.py`
   - Free-form: Qwen3-4B thinking mode, answer in `\boxed{}`
   - Multi-`[ANS]`: comma-separated answers in a single `\boxed{answer_1, answer_2, ...}`
   - MCQ: output only the letter in `\boxed{}`
3. Generate N=4 responses per question via vLLM (Qwen3-4B-Thinking-2507), `max_tokens=8192`
4. Self-consistency vote: extract `\boxed{}` answer from each response, take plurality, submit the winning trace
5. Write results to CSV incrementally every `chunk_size=10` questions; re-running resumes from where it left off (use `--reset` to start over)

**Key inference settings** (in `constants.py`):
- `DEFAULT_N_SAMPLES = 4` — lower than ideal but necessary for a single 10 GB GPU
- `DEFAULT_MAX_TOKENS = DEFAULT_MAX_SEQ_LEN = 8192` — enough for math reasoning; 32K leaves only ~1 concurrent request in the KV cache
- `DEFAULT_QUANTIZE_GPU_UTIL = 0.50` — always use `--quantize` for INT8 on a 10 GB GPU
- `--tp 2` splits the model across both GPUs (tensor parallelism, not data parallelism — required when a model doesn't fit on one GPU)

**Commands**:
```bash
# Score locally on public set (development loop — run this repeatedly)
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --data data/public.jsonl --output /deepfreeze/pnlong/school/cse151b/final/results/public_baseline.csv
python inference/evaluate.py --results /deepfreeze/pnlong/school/cse151b/final/results/public_baseline.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 4 \
    --save /deepfreeze/pnlong/school/cse151b/final/results/public_baseline_eval.jsonl

# Final submission — run once when best checkpoint is ready
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize
```

**Evaluation logging**: `evaluate.py` appends one stats row per run to `RESULTS_DIR/eval_log.csv` (auto-created). This lets you compare baseline vs SFT vs RL side-by-side in a spreadsheet.

```
timestamp | model | n_samples | checkpoint | mcq_acc | free_acc | overall_acc | missing | results_file | notes
```

Use `--no-log` to skip logging, `--log-csv <path>` to override the default location. Use `--notes` for free-text labels:

```bash
python inference/evaluate.py --results /deepfreeze/pnlong/school/cse151b/final/results/public_sft.csv \
    --model "Qwen3-4B" --checkpoint sft --n-samples 4 \
    --notes "distilled from DeepSeek-R1-32B + Qwen3-32B"
```

---

## Stage 2: Knowledge Distillation

**Goal**: collect a large set of high-quality reasoning traces from stronger teacher models to use as SFT training data. Two sources:
- **Public split** (1116 questions, ground truth known) → keep only correct traces
- **Private split** (893 questions, no ground truth) → pseudo-label via majority vote

**Scripts**: `distill/collect.py`, `distill/merge.py`

### Recommended teacher models

Run in priority order — higher rows give more training signal per GPU-hour. See `distill/README.md` for the full catalogue.

GPU memory guide: INT8 quantization uses ~1 GB per 1B params. A 10 GB GPU fits ≤7B models. A 24 GB GPU fits ≤14B. Two 24 GB GPUs (48 GB with `--tp 2`) fit up to ~40B comfortably. `--tp` is **tensor parallelism** — it splits the model itself across GPUs, which is what allows a model that doesn't fit on one GPU to run at all (unlike data parallelism which just splits batches).

| Priority | Model | HuggingFace ID | VRAM (INT8) | GPUs needed | Access |
|----------|-------|---------------|-------------|-------------|--------|
| 1 | DeepSeek-R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | ~32 GB | 2 (`--tp 2`) | Free |
| 2 | Qwen3-32B | `Qwen/Qwen3-32B` | ~32 GB | 2 (`--tp 2`) | Free |
| 3 | Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` | ~70 GB | 2+ (`--tp 2`) | Request (Meta) |
| 4 | Qwen2.5-Math-72B-Instruct | `Qwen/Qwen2.5-Math-72B-Instruct` | ~72 GB | 2+ (`--tp 2`) | Free |
| 5 | Phi-4 | `microsoft/phi-4` | ~14 GB | 2 (`--tp 2`) | Click-through license |
| 6 | DeepSeek-R1-Distill-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | ~14 GB | 2 (`--tp 2`) | Free |
| 7 | DeepSeek-R1-Distill-Qwen-7B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B` | ~7 GB | 1 | Free |
| 8 | Gemma-2-27B-it | `google/gemma-2-27b-it` | ~27 GB | 2 (`--tp 2`) | Click-through license |
| 9 | Llama-2-70B-chat | `meta-llama/Llama-2-70b-chat-hf` | ~70 GB | 2+ (`--tp 2`) | Request (Meta, already granted) |

> **Click-through license**: visit the model page on HuggingFace, click "Agree and access repository" — immediately available, no review needed.
> **Request (Meta)**: submit a short form on the model page; approved within hours for research use. Llama-2 access already granted implies Llama-3 access is easy to get.

### 2a. Collect traces (one run per teacher model)

```bash
# DeepSeek-R1-Distill-Qwen-7B — fits on single 10 GB GPU ✅
CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu --gpu-util 0.90 --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B

# Qwen3-8B ✅
CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu --gpu-util 0.90 --model Qwen/Qwen3-8B

# DeepSeek-R1-Distill-Qwen-14B — fits on single 10 GB GPU
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B

# DeepSeek-R1-Distill-Qwen-32B — best reasoning traces, needs both GPUs (~32 GB INT8)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

# Qwen3-32B — same family as student, strong reasoning, needs both GPUs
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model Qwen/Qwen3-32B

# Phi-4 — 14B, needs both GPUs on 10 GB cards (~14 GB INT8)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model microsoft/phi-4

# Llama-3.3-70B — diverse trace style, needs both GPUs (~70 GB INT8)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model meta-llama/Llama-3.3-70B-Instruct

# Qwen2.5-Math-72B — math specialist, needs both GPUs
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model Qwen/Qwen2.5-Math-72B-Instruct
```

Each run saves to `DISTILL_DIR/{model-slug}/`:
- `public_traces.jsonl` — verified-correct traces: `{id, question, options, answer, response}`
- `private_traces.jsonl` — pseudo-labeled traces: `{id, question, options, response}`

After each split completes, `collect.py` prints a **sanity check**: a randomly sampled question, its gold answer (public only), and the collected response. Use this to quickly verify the model is producing reasonable output before committing to a full run.

Runs are **append-safe**: re-running skips question IDs already saved. If a run is interrupted, just restart the same command.

### Verifying collection quality

**Public set — filter rate** (automatic, printed by `collect.py`):
```
Public : 743 correct traces from 8928 responses (1116 questions)
```
This is the most direct quality signal per teacher. `743/8928 ≈ 8.3%` per-response accuracy, meaning ~66% of questions got at least one correct trace. Higher is better; expect strong math models (DeepSeek-R1, Qwen2.5-Math) to hit 60–80% of questions covered.

**Public set — full accuracy score** (optional, run after collection):
```bash
# Quick 1-sample pass with the teacher model on public.jsonl
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu \
    --model <teacher-model-id> --quantize \
    --n-samples 1 --data data/public.jsonl \
    --output /tmp/teacher_check.csv
python inference/evaluate.py --results /tmp/teacher_check.csv
```
Gives per-category accuracy (MCQ vs free-form) and an overall score. Useful for comparing teachers before committing to a full N=8 run.

**Private set**: no direct evaluation is possible — no ground truth exists. The public filter rate is the best available proxy for private trace quality. If a teacher scores 70% on public, expect roughly similar quality on private.

### 2b. Merge into one SFT dataset

```bash
python distill/merge.py              # includes private traces
python distill/merge.py --no-private # exclude private traces if needed
```

Output: `DISTILL_DIR/sft_data.jsonl` — chat-format records:
```json
{"messages": [
  {"role": "system",    "content": "...DISTILL_SYSTEM_MATH..."},
  {"role": "user",      "content": "question text"},
  {"role": "assistant", "content": "full trace ending in \\boxed{answer}"}
]}
```

---

## Stage 3: Supervised Fine-Tuning (SFT)

**Goal**: fine-tune Qwen3-4B on the distilled traces so it learns to produce similarly thorough, structured reasoning.

**Script**: `sft/train.py` *(to be built)*

**What happens**:
1. Load `DISTILL_DIR/sft_data.jsonl`
2. LoRA fine-tune `Qwen/Qwen3-4B-Thinking-2507` using `trl.SFTTrainer`
   - LoRA rank r=16–64, target modules: q_proj, v_proj (and possibly k_proj, o_proj)
   - QLoRA (4-bit base + LoRA adapters) to fit on 2×24GB
3. Save checkpoint to `CHECKPOINTS_DIR/sft/`

**Eval after SFT**:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu \
    --model CHECKPOINTS_DIR/sft \
    --data data/public.jsonl --output /deepfreeze/pnlong/school/cse151b/final/results/public_sft.csv
python inference/evaluate.py --results /deepfreeze/pnlong/school/cse151b/final/results/public_sft.csv \
    --model "Qwen3-4B" --checkpoint sft --n-samples 4
```

Compare accuracy to the Stage 1 baseline in `RESULTS_DIR/eval_log.csv` to confirm SFT helped.

---

## Stage 4: Reinforcement Learning (GRPO)

**Goal**: further improve the SFT checkpoint using outcome-based reward on the public set ground truth. No labeled data needed beyond what we already have.

**Script**: `rl/train.py` *(to be built)*

**What happens**:
1. Start from the SFT checkpoint
2. GRPO training loop on `public.jsonl`:
   - Generate K responses per question
   - Score each response with `Judger` (exact match = reward 1, wrong = 0)
   - Partial credit for multi-`[ANS]`: reward = fraction of sub-answers correct
   - Format bonus: small reward for correctly producing `\boxed{}`
   - Update model via GRPO policy gradient
3. Save checkpoint to `CHECKPOINTS_DIR/rl/`

**Key design note on answer equivalence**: The GRPO reward must use `Judger.auto_judge()` (not simple string match) so that equivalent forms like `-0.65` and `-13/20` both score as correct.

**Eval after RL**:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu \
    --model CHECKPOINTS_DIR/rl \
    --data data/public.jsonl --output /deepfreeze/pnlong/school/cse151b/final/results/public_rl.csv
python inference/evaluate.py --results /deepfreeze/pnlong/school/cse151b/final/results/public_rl.csv \
    --model "Qwen3-4B" --checkpoint rl --n-samples 4
```

---

## Stage 5: Final Inference & Submission

**Goal**: run the best available checkpoint on `private.jsonl` and produce the final submission.

**Script**: `inference/infer.py`

```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu \
    --model CHECKPOINTS_DIR/rl \   # or /sft if RL didn't help
    --n-samples 4 \
    --output /deepfreeze/pnlong/school/cse151b/final/results/final_submission.csv
```

Submit `/deepfreeze/pnlong/school/cse151b/final/results/final_submission.csv` to the leaderboard.

**Checkpoint selection**: always compare Stage 1 (base), Stage 3 (SFT), and Stage 4 (RL) on `public.jsonl` before deciding which to use for the final submission.

---

## Storage Layout

All large artifacts live under `STORAGE_DIR` (`/deepfreeze/pnlong/school/cse151b/final`):

```
STORAGE_DIR/
├── distillation/
│   ├── qwen3-32b/
│   │   ├── public_traces.jsonl
│   │   └── private_traces.jsonl
│   ├── deepseek-r1-distill-qwen-32b/
│   │   └── ...
│   └── sft_data.jsonl
├── checkpoints/
│   ├── sft/                    SFT LoRA checkpoint
│   └── rl/                     GRPO RL checkpoint
├── results/
│   ├── public_baseline.csv
│   ├── public_sft.csv
│   ├── public_rl.csv
│   └── final_submission.csv
└── cache/                      HuggingFace model weights
```

---

## Progress Tracker

| Stage | Status | Notes |
|-------|--------|-------|
| Baseline inference pipeline | ✅ Built | `inference/infer.py`, `inference/evaluate.py` |
| Distillation pipeline | ✅ Built | `distill/collect.py`, `distill/merge.py` |
| SFT | 🔲 Not started | Need to build `sft/train.py` |
| RL (GRPO) | 🔲 Not started | Need to build `rl/train.py` |
| Final submission | 🔲 Not started | Pending SFT + RL |
