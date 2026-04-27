# Inference Pipeline

Scripts for running Qwen3-4B on the competition dataset and scoring results locally.

## Quick start

```bash
# 1. Copy .env.example → .env and fill in your paths
cp .env.example .env   # edit ROOT_DIR and STORAGE_DIR

# 2. Install dependencies (into the cse151b_competition mamba env)
pip install vllm transformers tqdm python-dotenv

# 3. Smoke-test (1 sample, 20 questions — CPU-only, no --gpu needed)
python inference/infer.py --n-samples 1 --limit 20 --output /tmp/test.csv

# 4. Score against public set (single GPU)
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --data data/public.jsonl --output /tmp/public.csv
python inference/evaluate.py --results /tmp/public.csv

# 5. Full private-set submission (default settings, single GPU)
CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu

# 6. Two-GPU tensor parallel
CUDA_VISIBLE_DEVICES=0,1 python inference/infer.py --gpu --tp 2
```

> **GPU convention**: device selection is always done via `CUDA_VISIBLE_DEVICES` set in the shell before running. The `--gpu` flag simply tells the script that GPU inference is intended; omit it for CPU-only runs (useful for import checks or dry runs).

---

## `infer.py` — generate a submission CSV

**What it does conceptually:**

1. **Loads** a JSONL dataset (default: `data/private.jsonl`).
2. **Builds prompts** for each question using either:
   - the baseline two-prompt scheme from `constants.py` (free-form vs. MCQ), or
   - the optional router (`inference/router.py`) that selects a format-first prompt (`fr_single`, `fr_multi`, `mcq_single`) and can append small topic refinements (stats/geometry/calculus/linear algebra).
   For free-form questions with multiple `[ANS]` slots, a per-question note is injected into the user message instructing the model to produce a single `\boxed{answer_1, answer_2, ...}`.
3. **Enables thinking mode** via `apply_chat_template(..., enable_thinking=True)` — Qwen3's native chain-of-thought mode, which produces a `<think>...</think>` block before the final answer. The full trace (thinking + answer) is kept as the submission response, since the evaluator extracts `\boxed{}` from anywhere in the text.
4. **Generates N responses per question** (default N=16) by repeating each prompt N times in one big vLLM batch. vLLM's prefix cache means the shared prompt tokens are only computed once per unique question.
5. **Votes** using self-consistency: extracts the `\boxed{}` answer from each of the N responses, normalizes lightly for comparison, and picks the plurality answer. The winning response (the full trace) is kept as the final output.
6. **Writes** a CSV with columns `id` and `response`.

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/private.jsonl` | Input JSONL |
| `--output` | `$STORAGE_DIR/results/submission.csv` | Output CSV |
| `--n-samples` | `16` | Responses per question |
| `--gpu` | off | Enable GPU inference (device set via `CUDA_VISIBLE_DEVICES` externally) |
| `--tp` | `1` | Tensor-parallel degree (must match number of visible GPUs) |
| `--quantize` | off | INT8 via bitsandbytes — halves VRAM, ~1.5× slower |
| `--limit` | off | Process only the first N questions (smoke-testing) |
| `--use-router` | off | Enable prompt routing (format-first prompts + optional topic refinements) |
| `--router-secondary-llm` | off | Use a tiny LLM to pick *secondary* topic tags (primary routing stays deterministic) |
| `--router-model` | `Qwen/Qwen2.5-0.5B-Instruct` | Router model used only when `--router-secondary-llm` is enabled |
| `--router-device` | `cpu` | Router device: `cpu` (safe) or `auto` |

---

## `evaluate.py` — score a CSV against public.jsonl

**What it does conceptually:**

Loads a submission CSV and the public dataset (which has ground-truth answers), then scores each prediction using the competition's `Judger`:

- **MCQ**: extracts the letter from `\boxed{}` and compares to the gold letter.
- **Free-form**: passes the full response to `Judger.auto_judge()`, which handles symbolic equivalence, numeric approximation, unit stripping, and ordered/unordered list matching. This mirrors what the competition evaluator does.

Prints accuracy broken down by MCQ vs. free-form, and appends one stats row to a running comparison log CSV (`RESULTS_DIR/eval_log.csv` by default) so different runs can be compared in a table.

**Options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--results` | *(required)* | Submission CSV to score |
| `--data` | `data/public.jsonl` | Ground-truth JSONL |
| `--save` | off | Write full per-question JSONL (id, gold, response, correct) |
| `--verbose` | off | Print details for every wrong answer |
| `--log-csv` | `RESULTS_DIR/eval_log.csv` | Path to the comparison log CSV |
| `--no-log` | off | Skip writing to the log CSV |
| `--model` | `""` | Label for the model / run |
| `--n-samples` | `""` | Number of self-consistency samples (for the log) |
| `--checkpoint` | `""` | Stage label: `base`, `sft`, `rl` |
| `--notes` | `""` | Free-text notes appended to the log row |

**Example — logging a labelled run:**
```bash
python inference/evaluate.py --results results/public_sft.csv \
    --model "Qwen3-4B" --checkpoint sft --n-samples 16 \
    --notes "distilled from DeepSeek-R1 + Qwen3-32B"
```

The log CSV grows one row per call and can be opened in any spreadsheet for side-by-side comparison of baseline vs SFT vs RL runs.

---

## `utils.py` — shared inference utilities

Extends the repo-root `utils.py` (math answer helpers) with inference-specific functions. Both scripts import from here rather than duplicating logic.

| Function | Purpose |
|----------|---------|
| `extract_last_boxed(text)` | Pulls inner content of last `\boxed{}` (wraps root `last_boxed_only_string` + `remove_boxed`) |
| `norm_for_vote(s)` | Lightweight normalization for vote-counting (strips whitespace, LaTeX spacing) |
| `split_top_level_commas(s)` | Splits `a, b, c` respecting nested braces, for multi-answer parsing |
| `answer_key(response, n_slots, is_mcq)` | Canonical voting key from one response |
| `majority_vote(responses, n_slots, is_mcq)` | Returns the winning response from a group |
| `build_prompt(question, options, …)` | Constructs (system, user) pair from constants |
| `apply_chat_template_safe(tokenizer, messages)` | Applies chat template with `enable_thinking` fallback |
| `load_jsonl(path)` | Read JSONL → list of dicts |
| `save_jsonl(records, path)` | Write list of dicts → JSONL |
| `save_submission_csv(rows, path)` | Write `[{id, response}]` → competition CSV |

---

## `starter.py` — starter-code baseline (Experiment 1a)

Faithful port of `starter_code_cse151b_comp.ipynb` to a runnable script. Uses the original notebook's system prompts, model settings (`max_model_len=16384`, `enable_prefix_caching=False`), and N=1 generation (no self-consistency). Outputs a CSV compatible with `evaluate.py`.

```bash
# Run on public set (single GPU, quantized — matches notebook defaults)
CUDA_VISIBLE_DEVICES=0 python inference/starter.py --gpu

# Smoke-test (10 questions, no GPU)
python inference/starter.py --limit 10 --output /tmp/starter_test.csv

# Score
python inference/evaluate.py \
    --results /deepfreeze/pnlong/school/cse151b/final/results/starter_baseline.csv \
    --model "Qwen3-4B" --checkpoint base --n-samples 1 \
    --notes "starter code notebook baseline"
```

| Flag | Default | Description |
|------|---------|-------------|
| `--data` | `data/public.jsonl` | Input JSONL |
| `--output` | `$STORAGE_DIR/results/starter_baseline.csv` | Output CSV |
| `--gpu` | off | Enable GPU inference |
| `--limit` | off | First N questions only |
| `--no-quantize` | off | Disable INT8 (notebook uses quantization by default) |

---

## `router.py` — optional prompt router

The router is designed to be **safe for scoring**:
- **Primary routing is deterministic** (based on `options` and the number of `[ANS]` slots) to avoid format mistakes.
- **Secondary tags are optional**:
  - default: conservative keyword matching
  - optional: `--router-secondary-llm` uses a small instruct model to emit a strict JSON decision, then appends the corresponding refinement snippets.

In all cases, the generated system prompts are written to preserve `judger.py` extraction behavior by ensuring the model outputs exactly one final `\boxed{...}`.

## Project-wide modules (repo root)

| File | Purpose |
|------|---------|
| `constants.py` | All string / numeric / bool constants (model name, sampling params, system prompts) |
| `config.py` | Loads `.env`, exposes `ROOT_DIR`, `STORAGE_DIR`, and all derived paths |
| `.env` | Local environment (git-ignored) — set `ROOT_DIR`, `STORAGE_DIR`, `ANTHROPIC_API_KEY` |
| `.env.example` | Committed template showing all required variables |
| `utils.py` | Math answer evaluation helpers (used by `judger.py`) |
| `judger.py` | Competition-provided answer judging logic |
