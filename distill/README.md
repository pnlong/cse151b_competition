# Distillation Pipeline

Generates high-quality reasoning traces from large open-source teacher models on HuggingFace, filters/pseudo-labels them, and packages everything into a single SFT-ready JSONL dataset.

## Concept

```
public.jsonl  ──►  teacher model  ──►  N traces/question  ──►  Judger filter  ──►  public_traces.jsonl
                                                                (keep correct)
private.jsonl ──►  teacher model  ──►  N traces/question  ──►  majority vote  ──►  private_traces.jsonl
                                                                (pseudo-label)

public_traces.jsonl  ─┐
                       ├──►  merge.py  ──►  sft_data.jsonl  (chat-format, shuffled)
private_traces.jsonl  ─┘  (--no-private to exclude)
```

**Public split**: ground-truth answers available → we keep every trace the teacher got right (all N, not just one — more training data, more diverse).

**Private split**: no ground truth → we run N samples and pick the majority-vote response as a pseudo-labeled training target. The hope is that the student memorizes these answers.

---

## Quick start

```bash
# 1. Smoke-test (5 questions, 2 samples, no GPU needed)
python distill/collect.py --model Qwen/Qwen3-32B \
    --public-only --limit 5 --n-samples 2

# 1b. Debug one teacher on 1–2 public questions (verbose, no trace files)
CUDA_VISIBLE_DEVICES=0,1 python distill/debug_collect.py --gpu --tp 2 --quantize \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

# 2. Full run — Qwen3-32B, both splits, single GPU
CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu \
    --model Qwen/Qwen3-32B --quantize

# 3. Merge all collected traces into one SFT dataset
python distill/merge.py

# 4. Exclude private traces (if needed)
python distill/merge.py --no-private
```

---

## `collect.py` — run one teacher model

Generates N responses per question, filters public traces by correctness, and pseudo-labels private traces via majority vote. **Append-safe**: re-running skips question IDs already in the output file, so interrupted runs can be resumed.

For **DeepSeek-R1 / R1-Distill** HuggingFace ids, vLLM receives the rendered chat **string** (not client-side `prompt_token_ids`) so tokenization matches the model’s special AddedTokens, and **`enforce_eager=True`** is set when constructing the engine (mitigates known vLLM CUDA-graph / compile bugs that can yield degenerate repetition on this family). If output is still garbage, try **without** `--quantize` (bitsandbytes INT8 can be unstable with some vLLM builds).

**Key options:**

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `Qwen/Qwen3-4B-Thinking-2507` | Teacher model HuggingFace ID |
| `--n-samples` | `8` | Responses per question |
| `--gpu` | off | Enable GPU (device via `CUDA_VISIBLE_DEVICES`) |
| `--tp` | `1` | Tensor-parallel degree |
| `--quantize` | off | INT8 bitsandbytes (halves VRAM) |
| `--public-only` | off | Skip private.jsonl |
| `--private-only` | off | Skip public.jsonl |
| `--limit` | off | First N questions per split (smoke-testing) |

**Output files** under `DISTILL_DIR/{model-slug}/`:
- `public_traces.jsonl` — `{id, question, options, answer, response}` (correct only)
- `private_traces.jsonl` — `{id, question, options, response}` (pseudo-labeled)

---

## `debug_collect.py` — inspect 1–2 questions before a full run

Same tokenizer, vLLM settings, prompts, and `verify_trace` logic as `collect.py`, but prints diagnostics to the terminal and **does not write** any trace JSONL. Use this when a teacher is producing “0 correct traces” or you want to see raw outputs (including chain-of-thought) before committing GPU time to a full collection.

**Defaults:** prints the **entire** question text and the **entire** raw model output (thinking block + final answer). Pass `--question-chars` / `--response-chars` with a positive limit if you want truncation.

**Example** (matches a two-GPU quantized DeepSeek teacher):

```bash
CUDA_VISIBLE_DEVICES=0,1 python distill/debug_collect.py --gpu --tp 2 --quantize \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B
```

By default it runs **two rows from `public.jsonl`** so gold labels exist and `verify_trace` is meaningful.

**Useful flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--num-questions` | `2` | How many dataset rows to run |
| `--n-samples` | `1` | Responses per question |
| `--start-index` | `0` | Skip the first N public (or private) rows |
| `--split` | `public` | `public` (has gold) or `private` (no verify) |
| `--question-chars` | `0` | Clip printed question (`0` = full question) |
| `--response-chars` | `0` | Clip printed model output and `Judger.extract_ans` (`0` = full) |

---

## `merge.py` — combine traces into SFT dataset

Scans all `DISTILL_DIR/*/public_traces.jsonl` (and optionally `private_traces.jsonl`), converts each trace to the standard trl chat format, shuffles, and saves.

**Output format** (`sft_data.jsonl`):
```json
{
  "messages": [
    {"role": "system",    "content": "DISTILL_SYSTEM_MATH or DISTILL_SYSTEM_MCQ"},
    {"role": "user",      "content": "question text"},
    {"role": "assistant", "content": "full trace ending in \\boxed{answer}"}
  ]
}
```

**Excluding private data** (three layers of control):
1. `python distill/merge.py --no-private` — CLI override, one-shot
2. `INCLUDE_PRIVATE_IN_SFT = False` in `constants.py` — persistent default
3. Files are always kept separate (`public_traces.jsonl` vs `private_traces.jsonl`) — never mixed until `merge.py`

---

## `utils.py` — shared utilities

Re-exports everything from `inference/utils.py` and adds:

| Function | Description |
|----------|-------------|
| `model_slug(model_id)` | `"Qwen/Qwen3-32B"` → `"qwen3-32b"` (safe directory name) |
| `traces_dir(model_id)` | `DISTILL_DIR / model_slug(model_id)` (creates if needed) |
| `verify_trace(response, gold, is_mcq, judger)` | Checks correctness via `score_mcq` or `Judger.auto_judge` |

---

## Recommended teacher models

Models are grouped by family. Run as many as feasible — more diverse teachers means more and better-distributed training data.

### Qwen family

| Model | HuggingFace ID | VRAM (4-bit) | Setup | Notes |
|-------|---------------|-------------|-------|-------|
| Qwen3-32B | `Qwen/Qwen3-32B` | ~18 GB | 1 GPU, `--quantize` | Same family as student; traces naturally match style |
| Qwen2.5-Math-72B-Instruct | `Qwen/Qwen2.5-Math-72B-Instruct` | ~38 GB | 2 GPU, `--tp 2 --quantize` | Math-specialist; best domain accuracy in the Qwen line |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` | ~38 GB | 2 GPU, `--tp 2 --quantize` | Strong general reasoning; good complement to Math variant |

### DeepSeek family

| Model | HuggingFace ID | VRAM (4-bit) | Setup | Notes |
|-------|---------------|-------------|-------|-------|
| DeepSeek-R1-Distill-Qwen-32B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | ~18 GB | 1 GPU, `--quantize` | Explicit chain-of-thought; best for teaching step-by-step reasoning |
| DeepSeek-R1-Distill-Qwen-14B | `deepseek-ai/DeepSeek-R1-Distill-Qwen-14B` | ~9 GB | 1 GPU, `--quantize` | Fast; good quality-to-speed ratio |
| DeepSeek-R1-Distill-Llama-70B | `deepseek-ai/DeepSeek-R1-Distill-Llama-70B` | ~38 GB | 2 GPU, `--tp 2 --quantize` | Llama backbone; diverse trace style |

### Meta Llama family

> Requires a HuggingFace account with access granted at meta-llama repos.

| Model | HuggingFace ID | VRAM (4-bit) | Setup | Notes |
|-------|---------------|-------------|-------|-------|
| Llama-3.3-70B-Instruct | `meta-llama/Llama-3.3-70B-Instruct` | ~38 GB | 2 GPU, `--tp 2 --quantize` | Best Llama model for reasoning; strongly preferred over Llama 2 |
| Llama-3.1-70B-Instruct | `meta-llama/Meta-Llama-3.1-70B-Instruct` | ~38 GB | 2 GPU, `--tp 2 --quantize` | Slightly older; still strong |
| Llama-3.1-8B-Instruct | `meta-llama/Meta-Llama-3.1-8B-Instruct` | ~5 GB | 1 GPU, `--quantize` | Fast; lower quality but good for high-volume sampling |
| Llama-2-70B-chat-hf | `meta-llama/Llama-2-70b-chat-hf` | ~38 GB | 2 GPU, `--tp 2 --quantize` | Older; math reasoning significantly weaker than Llama 3 — use only if Llama 3 unavailable |
| Llama-2-13B-chat-hf | `meta-llama/Llama-2-13b-chat-hf` | ~8 GB | 1 GPU, `--quantize` | Fast but weak at math; low expected filter rate |

### Other strong options

| Model | HuggingFace ID | VRAM (4-bit) | Setup | Notes |
|-------|---------------|-------------|-------|-------|
| Phi-4 | `microsoft/phi-4` | ~8 GB | 1 GPU, `--quantize` | 14B model; surprisingly strong at math for its size |
| Gemma-2-27B-it | `google/gemma-2-27b-it` | ~15 GB | 1 GPU, `--quantize` | Strong general reasoning; diverse trace style |

---

### Recommended run order

Run the highest-quality models first (they give the most training signal). Lower-quality models are worth running if you have time — even a 40% filter rate on 1126 questions yields ~450 extra correct traces.

```bash
# 1. DeepSeek-R1-Distill-Qwen-32B (best reasoning traces, 1 GPU)
CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --quantize

# 2. Qwen3-32B (same family as student, 1 GPU)
CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu \
    --model Qwen/Qwen3-32B --quantize

# 3. Llama-3.3-70B-Instruct (diverse style, needs 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 \
    --model meta-llama/Llama-3.3-70B-Instruct --quantize

# 4. Qwen2.5-Math-72B-Instruct (math specialist, needs 2 GPUs)
CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 \
    --model Qwen/Qwen2.5-Math-72B-Instruct --quantize

# 5. Phi-4 (fast, 1 GPU — good for boosting sample count)
CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu \
    --model microsoft/phi-4 --quantize

# Merge all collected traces
python distill/merge.py
```

---

## Adding a new teacher model

1. Run `distill/collect.py --model <HF_ID>` — traces land in `DISTILL_DIR/{model-slug}/`
2. Run `distill/merge.py` — the new model's traces are automatically picked up
3. No code changes required

---

## Teacher system prompts

Defined in `constants.py` as `DISTILL_SYSTEM_MATH` and `DISTILL_SYSTEM_MCQ`. These are richer than the inference-time prompts — they include explicit 5-step structure (Understand → Plan → Execute → Verify → Conclude) to encourage thorough, well-labeled reasoning traces that make effective SFT targets.
