# Experiments

All experiments evaluate on `data/public.jsonl` (1126 problems with ground-truth answers) using `inference/evaluate.py`, which scores each prediction through the competition `Judger` and reports **MCQ accuracy**, **free-form accuracy**, and **overall accuracy**. Final leaderboard submissions run on `data/private.jsonl` (893 problems, no ground truth).

Results CSVs and the running comparison log all land in `STORAGE_DIR/results/` (`/deepfreeze/pnlong/school/cse151b/final/results/`). The log file `eval_log.csv` grows one row per `evaluate.py` call and records timestamp, model, checkpoint stage, n\_samples, MCQ/free-form/overall accuracy, and any free-text notes — making side-by-side comparison across all experiments straightforward.

---

## Experiment 1 — Baselines (prompt engineering only)

**Goal**: establish accuracy upper and lower bounds using only prompt design and self-consistency, with no fine-tuning. All conditions use the base `Qwen/Qwen3-4B-Thinking-2507` weights.

### 1a. Starter code

The original competition-provided notebook, converted to a runnable script. Establishes the floor — uses the notebook's simpler system prompts, N=1 (no self-consistency), `max_model_len=16384`, and `enable_prefix_caching=False`. No routing, no structured multi-answer handling beyond what the notebook ships with.

**Relevant files**:
- `inference/starter.py` — runnable port of the notebook
- `starter_code_cse151b_comp.ipynb` — original reference

**How to run**:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/starter.py --gpu \
    --output /deepfreeze/pnlong/school/cse151b/final/results/starter_baseline.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/starter_baseline.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 1 \
    --notes "starter code notebook baseline"
```

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy.

---

### 1b. N=1, no self-consistency

Single-sample inference with the standard system prompts (`SYSTEM_MATH` / `SYSTEM_MCQ` from `constants.py`) and Qwen3's thinking mode enabled. No voting — the one response is the submission. Isolates how much self-consistency contributes on top of the base model.

**Relevant files**:
- `inference/infer.py`
- `inference/evaluate.py`
- `constants.py` — `SYSTEM_MATH`, `SYSTEM_MCQ`

**How to run**:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --data data/public.jsonl \
    --n-samples 1 \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_baseline_n1.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_baseline_n1.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 1 \
    --notes "single system prompt, no self-consistency, thinking on"
```

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy.

---

### 1c. Single system prompt, N=4 self-consistency

The main baseline: four samples per question with majority-vote self-consistency, using the standard `SYSTEM_MATH` / `SYSTEM_MCQ` prompts. No routing.

**Relevant files**:
- `inference/infer.py`
- `inference/evaluate.py`
- `constants.py` — `SYSTEM_MATH`, `SYSTEM_MCQ`, `DEFAULT_N_SAMPLES`

**How to run**:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --data data/public.jsonl \
    --n-samples 4 \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_baseline_n4.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_baseline_n4.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 4 \
    --notes "single system prompt, self-consistency N=4, thinking on"
```

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy. Compare to 1b to quantify the self-consistency gain.

---

### 1d. Prompt routing, N=4 self-consistency

Replaces the single system prompt pair with a format-first router that selects among `fr_single`, `fr_multi`, and `mcq_single` prompts based on question format, and optionally appends topic-specific refinement snippets. Topics are one of 20 curriculum labels from `topic_taxonomy.classify_problem` (same weighted-regex scoring as `analysis/classify_topics.py`).

**Relevant files**:
- `inference/infer.py` — `--use-router` flag
- `inference/router.py` — deterministic primary routing + `classify_problem` (or optional LLM topic with fallback)
- `topic_taxonomy.py` — shared `TOPICS` weights and `classify` / `classify_problem`
- `prompts/routing/prompts.py` — `PRIMARY_PROMPTS`, `TOPIC_REFINEMENTS`
- `constants.py` — fallback prompts

**How to run**:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --data data/public.jsonl \
    --n-samples 4 \
    --use-router \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_router_n4.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_router_n4.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 4 \
    --notes "prompt routing (topic_taxonomy + refinements), self-consistency N=4, thinking on"
```

To test routing in isolation without loading the main model:
```bash
python inference/test_router.py
```

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy. Compare to 1c to measure the routing gain.

---

### 1e. Thinking mode off, N=4 self-consistency

Disables Qwen3's native chain-of-thought (`<think>...</think>`) by passing `enable_thinking=False` in `inference/utils.py:apply_chat_template_safe`. Everything else identical to 1c. Quantifies how much of the accuracy comes from CoT reasoning alone.

**Relevant files**:
- `inference/utils.py` — `apply_chat_template_safe` (change `enable_thinking=True` → `False`)
- `inference/infer.py`
- `inference/evaluate.py`
- `constants.py` — `SYSTEM_MATH`, `SYSTEM_MCQ`

**How to run**: edit `inference/utils.py` to set `enable_thinking=False`, then:
```bash
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --data data/public.jsonl \
    --n-samples 4 \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_nothinking_n4.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_nothinking_n4.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 4 \
    --notes "single system prompt, self-consistency N=4, thinking OFF"
```

Restore `enable_thinking=True` after running.

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy. Compare to 1c to quantify the thinking-mode contribution.

---

## Experiment 2 — Knowledge Distillation + Supervised Fine-Tuning (SFT)

**Goal**: improve Qwen3-4B by fine-tuning on high-quality reasoning traces distilled from larger teacher models. Two data sources:
- **Public split** (1126 problems, ground truth available) — keep only teacher traces the `Judger` marks correct.
- **Private split** (893 problems, no ground truth) — pseudo-label via majority vote across N=8 teacher samples.

### 2a. Trace collection

Run one or more teacher models on both dataset splits. Each run saves verified traces (public) and pseudo-labeled traces (private) to `STORAGE_DIR/distillation/{model-slug}/`.

**Relevant files**:
- `distill/collect.py` — main collection script
- `distill/utils.py` — `model_slug`, `traces_dir`, `verify_trace`
- `constants.py` — `DISTILL_SYSTEM_MATH`, `DISTILL_SYSTEM_MCQ`, `DEFAULT_DISTILL_N_SAMPLES`
- `judger.py` — used to filter public traces

**Recommended teacher models** (priority order):

| Priority | Model | HuggingFace ID | GPUs needed |
|----------|-------|---------------|-------------|
| 1 | DeepSeek-R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | 2 (`--tp 2`) |
| 2 | Qwen3-32B | `Qwen/Qwen3-32B` | 2 (`--tp 2`) |
| 3 | DeepSeek-R1-Distill-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | 2 (`--tp 2`) |
| 4 | Qwen2.5-Math-72B-Instruct | `Qwen/Qwen2.5-Math-72B-Instruct` | 2+ (`--tp 2`) |
| 5 | Phi-4 | `microsoft/phi-4` | 2 (`--tp 2`) |

See `distill/README.md` for the full catalogue.

**How to run** (one command per teacher model — runs are append-safe and can be resumed):
```bash
# DeepSeek-R1-Distill-Qwen-32B (best reasoning traces)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

# Qwen3-32B (same family as student)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 --quantize \
    --model Qwen/Qwen3-32B
```

**What to report**: per-teacher filter rate on the public split (correct traces / total responses), total training examples collected.

---

### 2b. Merge into SFT dataset

Combines all collected teacher traces into a single shuffled SFT-ready JSONL in the standard `trl` chat format.

**Relevant files**:
- `distill/merge.py`
- `distill/utils.py`
- `constants.py` — `INCLUDE_PRIVATE_IN_SFT`

**How to run**:
```bash
# Include private pseudo-labeled traces (default)
python distill/merge.py

# Exclude private traces if needed
python distill/merge.py --no-private
```

Output: `STORAGE_DIR/distillation/sft_data.jsonl`

**What to report**: total training examples in `sft_data.jsonl`, breakdown by teacher model and split.

---

### 2c. LoRA SFT training

Fine-tune Qwen3-4B on `sft_data.jsonl` using LoRA (QLoRA: 4-bit base + LoRA adapters). Trainer checkpoints live under `STORAGE_DIR/checkpoints/sft/checkpoint-{step}/` with `checkpoint-latest` (pointer file by default; optional symlink) for inference and GRPO.

**Relevant files**:
- `sft/train.py` — `trl.SFTTrainer`, resume; training loss CSV/PDF (no in-training `evaluate()`)
- `sft/callbacks.py` — metrics CSV / plots
- `constants.py` — default student model id
- `config.py` — `DISTILL_DIR`, `CHECKPOINTS_DIR`

**Single GPU vs multi-GPU**

| Mode | Command sketch | Notes |
|------|------------------|--------|
| **Single GPU** | `CUDA_VISIBLE_DEVICES=0 python sft/train.py ...` | Optional `--single-gpu` pins the full model on `cuda:0` if multiple GPUs are visible. |
| **Multi-GPU (DDP)** | `CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 sft/train.py ...` | One full replica per GPU; global batch scales with world size. Prefer this over `python` with two GPUs visible (that uses `device_map="auto"` layer sharding and often poor SM balance). |

Under `torchrun`, `train.py` defaults to **`--dataloader-workers 4`** (prefetch on CPU). If one GPU still shows lower utilization than the other, raise workers (e.g. `--dataloader-workers 8`) and/or **`--batch-size 2`** when VRAM allows — short dips also happen when rank 0 saves checkpoints (`--save-every`) or runs periodic eval for plots.

**How to run** (adjust paths to your machine):

```bash
# Single GPU
CUDA_VISIBLE_DEVICES=0 python sft/train.py \
    --data /deepfreeze/pnlong/school/cse151b/final/distillation/sft_data.jsonl \
    --output-dir /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/

# Two GPUs (DDP)
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 sft/train.py \
    --data /deepfreeze/pnlong/school/cse151b/final/distillation/sft_data.jsonl \
    --output-dir /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/
```

**What to report**: training loss + optional TRL `mean_token_accuracy` (`training_loss_history.csv`, `statistics.pdf` / `metrics_history.csv`; token accuracy is masked LM match on labels, not Judge scores), checkpoint step, and public-set scores from `infer.py` + `evaluate.py`; note single-GPU vs `torchrun` configuration.

---

### 2d. SFT inference + evaluation

Run the SFT checkpoint on the public set and compare accuracy to the Experiment 1 baselines.

Experiment 2 in the report (**Exp 2**) uses **`--use-router`** with $N = 4$: same inference recipe as **Exp 1d** (format + topic routing, self-consistency). Omitting **`--use-router`** is closer to **Exp 1c** ($N = 4$ without routing)—use separate output CSVs so the two setups are not conflated.

**Relevant files**:
- `inference/infer.py`
- `inference/infer_parallel.py` — data-parallel throughput (recommended when multiple GPUs are available)
- `inference/evaluate.py`
- `config.py` — `CHECKPOINTS_DIR`

**How to run** (adjust `CUDA_VISIBLE_DEVICES` and checkpoint paths):

**With router (matches Exp 2 / Exp 1d recipe)** — data-parallel example:

```bash
CUDA_VISIBLE_DEVICES=0,3 python inference/infer_parallel.py --gpu --quantize \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/checkpoint-latest \
    --n-samples 4 \
    --data data/public.jsonl \
    --use-router \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_sft_n4.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_sft_n4.csv \
    --model "Qwen3-4B" --checkpoint sft --n-samples 4 \
    --notes "SFT on distilled traces; router (Exp 2)"
```

**Without router ($N = 4$ only; ablation / closer to Exp 1c)**

```bash
CUDA_VISIBLE_DEVICES=0,3 python inference/infer_parallel.py --gpu --quantize \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/checkpoint-latest \
    --n-samples 4 \
    --data data/public.jsonl \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_sft_n4_no_router.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_sft_n4_no_router.csv \
    --model "Qwen3-4B" --checkpoint sft --n-samples 4 \
    --notes "SFT on distilled traces; no router (ablation)"
```

**Single GPU (either recipe)** — use `infer.py`; add **`--use-router`** for Exp 2, omit it for the no-router ablation.

```bash
# With router → same outputs as infer_parallel snippet above:
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/checkpoint-latest \
    --data data/public.jsonl \
    --n-samples 4 \
    --use-router \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_sft_n4.csv

# Without router → align output path with the no_router CSV name:
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --quantize \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/checkpoint-latest \
    --data data/public.jsonl \
    --n-samples 4 \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_sft_n4_no_router.csv
```

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy for the **router** run as Experiment 2. Compare to best Experiment 1 result to quantify the SFT gain.

---

## Experiment 3 — Reinforcement Learning (GRPO)

**Goal**: further improve the SFT checkpoint using outcome-based reward signals derived directly from the competition `Judger`. No new labeled data is needed — the public split ground-truth answers serve as the reward signal.

### 3a. GRPO training

Start from the SFT checkpoint. For each question, generate K responses and score them with `Judger.auto_judge()` (reward = 1 if correct, 0 if wrong; partial credit for multi-`[ANS]` questions based on fraction of sub-answers correct). A small format bonus rewards correctly producing `\boxed{}`. Update the policy via GRPO.

**Relevant files**:
- `rl/train.py` — CLI, `GRPOTrainer`, dataset building, best-reward callback
- `rl/rewards.py` — Judger / `score_mcq` outcome reward + optional `\boxed{}` bonus
- `judger.py` — reward function
- `utils.py` — answer parsing helpers used by reward function
- `constants.py` — model ID, answer format constants (`BOXED_CMD`, `ANS_PLACEHOLDER`)
- `config.py` — `CHECKPOINTS_DIR`

**Dependencies**: use a recent **`trl`** build with `GRPOTrainer` (e.g. `trl>=0.14`; this repo is tested with TRL 1.x). TRL pulls in `transformers`; if import errors mention `AutoProcessor` / PIL, upgrade **`pillow`** (e.g. `pip install -U 'pillow>=10'`) so `PIL.Image.Resampling` exists. The judging stack needs **`antlr4-python3-runtime==4.11.1`** (already in `setup.sh`).

**How to run**:
```bash
# Single GPU (QLoRA continues SFT 4-bit adapter by default; add --no-qlora for bf16 LoRA)
CUDA_VISIBLE_DEVICES=0 python rl/train.py \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/checkpoint-latest \
    --data data/public.jsonl \
    --output-dir /deepfreeze/pnlong/school/cse151b/final/checkpoints/rl/

# Multi-GPU data parallel (recommended over one process + device_map=auto)
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 rl/train.py \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/sft/checkpoint-latest \
    --data data/public.jsonl \
    --output-dir /deepfreeze/pnlong/school/cse151b/final/checkpoints/rl/
```

Defaults mirror ``sft/train.py`` where applicable (epochs, batch/grad-accum, LR, warmup, ``--save-every``, ``--save-total-limit``, segmented tqdm, ``checkpoint-latest``, ``--resume``). Useful overrides: ``--num-generations``, ``--max-completion-length``, ``--format-bonus``, ``--no-qlora``, ``--single-gpu``. Best logged **`reward`** → ``checkpoint-best-reward/``.

**What to report**: reward curve over training steps, checkpoint saved at peak reward.

---

### 3b. RL inference + evaluation

Run the RL (GRPO) adapter on the public set with the **same inference recipe as Experiment 2d (router + $N = 4$)** and compare to Experiment 1 and SFT.

**Checkpoints**:

- **`checkpoints/rl/checkpoint-best-reward/`** — snapshot when Trainer’s logged **`reward`** improved (**default for Experiment 3**).
- **`checkpoints/rl/checkpoint-latest`** — pointer/symlink to the last saved step (e.g. after an early stop). Use if you deliberately want weights from the final step regardless of logged reward peaks.

Inference expects a directory that contains a PEFT **`adapter_config.json`** (typically `checkpoint-best-reward/` or resolved `checkpoint-latest`), not bare `checkpoints/rl/` by itself unless that folder is wired as the adapter root.

**Relevant files**:
- `inference/infer.py`
- `inference/infer_parallel.py`
- `inference/evaluate.py`
- `config.py` — `CHECKPOINTS_DIR`

**How to run** — **`checkpoint-best-reward`** + router (matches Exp 3), data-parallel:

```bash
CUDA_VISIBLE_DEVICES=0,3 python inference/infer_parallel.py --gpu --quantize \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/rl/checkpoint-best-reward \
    --n-samples 4 \
    --data data/public.jsonl \
    --use-router \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_grpo_n4_bestreward.csv

python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/public_grpo_n4_bestreward.csv \
    --model "Qwen3-4B" --checkpoint rl --n-samples 4 \
    --notes "GRPO from SFT; checkpoint-best-reward; router"
```

**Optional** — final-step adapter instead of best-reward snapshot:

```bash
CUDA_VISIBLE_DEVICES=0,3 python inference/infer_parallel.py --gpu --quantize \
    --model /deepfreeze/pnlong/school/cse151b/final/checkpoints/rl/checkpoint-latest \
    --n-samples 4 \
    --data data/public.jsonl \
    --use-router \
    --output /deepfreeze/pnlong/school/cse151b/final/results/public_grpo_n4_latest.csv
```

**Single GPU** — same flags on `inference/infer.py` with one visible device.

**What to report**: MCQ accuracy, free-form accuracy, overall accuracy. Compare to Experiment 2d (SFT with router) and best Experiment 1 baseline.

---

## Results Summary

All runs append one row to `STORAGE_DIR/results/eval_log.csv`. The columns are:

```
timestamp | model | n_samples | checkpoint | mcq_acc | free_acc | overall_acc | missing | results_file | notes
```

Open this file in any spreadsheet to compare all experiments side-by-side. For the final leaderboard submission, run `inference/infer.py` on `data/private.jsonl` using whichever checkpoint (base / sft / rl) achieved the highest `overall_acc` on the public set.
