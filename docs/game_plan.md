# CSE 151B Spring 2026 — Competition Game Plan

## Goal

Maximize mathematical reasoning accuracy of **Qwen3-4B** on a private test set of **893 math problems** spanning high school to graduate level.

- **Free-form** questions: one or more `[ANS]` placeholders; model fills them in order, answers wrapped in `\boxed{}`
- **Multiple-choice** questions: model selects a letter (A–J) from a provided `options` list, answer wrapped in `\boxed{}`

Submission format: CSV with columns `id` and `response` (full reasoning trace; evaluator extracts the last `\boxed{}`).

**Constraints**: No external API calls at inference time. Only model-intrinsic methods: prompt engineering, SFT, RL.

---

## Strategy (priority order)

### 1. Prompt Engineering + Self-Consistency  ← *current focus*

- Enable Qwen3-4B's native **thinking mode** at inference time (`enable_thinking=True` in chat template)
- Carefully crafted system prompts:
  - Free-form: enforce `\boxed{}`, step-by-step reasoning
  - MCQ: output only the letter in `\boxed{}`
  - Multi-`[ANS]`: inject per-question note with exact count, enforce `\boxed{answer_1, answer_2, ..., answer_n}` (comma-separated in a single box)
- **Self-consistency voting** (N=16 default): sample N responses per question, extract `\boxed{}` answer from each, take plurality vote, submit the winning trace

### 2. Supervised Fine-Tuning (SFT)

Collect high-quality reasoning traces from teacher models, then fine-tune Qwen3-4B (LoRA/QLoRA) on them.

**Data sources:**
- Public math datasets: NuminaMath, MATH, AIME/AMC, OlympiadBench
- Knowledge distillation: generate traces using large teacher models on similar problems, filter to only keep traces where extracted answer matches ground truth

**Teacher model options (for distillation):**
- **Claude / ChatGPT / Gemini via web UI** — manually prompt with a well-crafted system prompt, collect traces, verify against ground truth; no API key required
- **Local HuggingFace models** (Qwen2.5-Math-72B-Instruct, DeepSeek-R1-Distill-Qwen-32B) via vLLM on 2×24GB GPUs with 4-bit quantization — fully automated pipeline

**Training data strategy:**
- Domain-weight toward complex analysis, statistics, combinatorics
- Special attention to OEIS/sequence problems (few-shot prompting or targeted data)
- Training format: `(system + question, full reasoning trace ending in \boxed{})`

### 3. Reinforcement Learning (GRPO)

Start from the SFT checkpoint.

- **Reward signal**: exact-match on public set ground truth answers
- **Partial credit**: per sub-answer for multi-`[ANS]` questions
- **Format reward**: bonus for always producing `\boxed{}`
- **Answer equivalence**: handle `-0.65` vs `-13/20`, symbolic vs decimal forms (using `Judger`)

---

## What Has Been Built

### `inference/` (baseline pipeline)

| File | Purpose |
|---|---|
| `infer.py` | Loads JSONL, builds prompts, runs Qwen3-4B via vLLM, self-consistency votes, outputs submission CSV |
| `evaluate.py` | Scores a CSV against `public.jsonl` using the competition `Judger`; optionally saves full per-question JSONL |
| `utils.py` | Shared inference utilities (answer extraction, voting, prompt building, I/O helpers); re-exports from root `utils.py` |
| `README.md` | Conceptual docs and usage guide |

### Root-level infrastructure

| File | Purpose |
|---|---|
| `constants.py` | All string/numeric/bool constants (model ID, sampling params, system prompts) |
| `config.py` | Loads `.env` → `ROOT_DIR`, `STORAGE_DIR`, and all derived paths |
| `.env` / `.env.example` | Local directory paths (`.env` is git-ignored) |

---

## Next Steps

### Immediate
- [ ] Run `inference/infer.py` on `public.jsonl` to establish a baseline accuracy number
- [ ] Compare `--n-samples 1` vs `--n-samples 16` to quantify the voting gain
- [ ] Experiment with system prompt variants (few-shot examples, stronger format constraints)

### Short term — Distillation pipeline
- [ ] Create `distill/` subdirectory
- [ ] Build `distill/collect.py`: sends problems to teacher model, verifies answer, saves traces
- [ ] Decide on teacher: Claude API vs local Qwen2.5-Math-72B / DeepSeek-R1-Distill-Qwen-32B
- [ ] Process public.jsonl (1116 problems with known answers) as primary distillation source
- [ ] Explore NuminaMath / MATH / AIME datasets for additional SFT data

### Medium term — SFT
- [ ] Set up LoRA/QLoRA fine-tuning on the SFT checkpoint (likely using `trl` or `LLaMA-Factory`)
- [ ] Establish eval loop: SFT checkpoint → `inference/infer.py` → `inference/evaluate.py`

### Longer term — RL
- [ ] GRPO training starting from the SFT checkpoint
- [ ] Implement answer equivalence logic for reward computation using `Judger`
- [ ] Partial-credit reward for multi-`[ANS]` questions

---

## Open Questions

- **Distillation teacher**: Claude API (fast, high quality) vs local model (free, slower)? Use both?
- **Multi-`[ANS]` voting**: currently votes on the tuple of comma-separated answers — might need refinement if models produce varied orderings
- **OEIS/sequence problems**: need to investigate if few-shot examples significantly help or if targeted SFT data is required
- **RL compute**: GRPO on 4B model is feasible on 2×24GB; need to budget GPU time
