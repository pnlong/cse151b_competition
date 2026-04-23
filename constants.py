"""
Project-wide constants.

All numerical, boolean, and string constants live here.
Import from this module rather than hardcoding values in scripts.
"""

# ── Model ──────────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"

# ── Inference / sampling ───────────────────────────────────────────────────────

DEFAULT_N_SAMPLES    = 4       # responses per question for self-consistency voting
DEFAULT_MAX_TOKENS   = 8192    # max new tokens per response (thinking traces are long)
DEFAULT_MAX_SEQ_LEN  = 8192    # vLLM max_model_len (prompt + generation); lower = more KV cache concurrency
DEFAULT_TEMPERATURE  = 0.6
DEFAULT_TOP_P        = 0.95
DEFAULT_TOP_K        = 20
DEFAULT_MIN_P              = 0.0
DEFAULT_PRESENCE_PENALTY   = 0.0
DEFAULT_REPETITION_PENALTY = 1.0
DEFAULT_GPU_UTIL           = 0.90   # fraction of GPU VRAM (non-quantized)
DEFAULT_QUANTIZE_GPU_UTIL  = 0.50   # fraction of GPU VRAM (bitsandbytes INT8)
DEFAULT_MAX_NUM_SEQS       = 256    # vLLM max concurrent sequences
DEFAULT_MAX_NUM_BATCHED_TOKENS = 32768   # vLLM max tokens across one scheduler step

# ── Answer format ──────────────────────────────────────────────────────────────

BOXED_CMD      = "\\boxed"     # LaTeX command used to wrap final answers
ANS_PLACEHOLDER = "[ANS]"      # placeholder token in free-form questions

# ── System prompts ─────────────────────────────────────────────────────────────
# Used by both inference and distillation pipelines so that teacher and student
# see the same formatting instructions.

SYSTEM_MATH = (
    "You are an expert mathematician. Solve the problem step by step, "
    "showing your complete reasoning. "
    "Wrap your final answer inside \\boxed{}. "
    "Be precise and double-check your answer before writing it."
)

SYSTEM_MCQ = (
    "You are an expert mathematician. "
    "Carefully analyze the problem and each answer choice. "
    "Work through the problem step by step to determine the correct answer. "
    "At the end, output ONLY the letter of the correct choice inside \\boxed{}, "
    "e.g. \\boxed{C}. Do not put anything else inside \\boxed{}."
)

# Appended to the user message for multi-[ANS] free-form questions.
# {n} is replaced with the number of [ANS] slots at prompt-build time.
MULTI_ANS_NOTE = (
    "\n\nNote: This problem has {n} blanks marked [ANS] to fill in, "
    "in the order they appear. "
    "Put all {n} answers in a single \\boxed{{}}, separated by commas: "
    "\\boxed{{answer_1, answer_2, ..., answer_{n}}}"
)

# ── Distillation ───────────────────────────────────────────────────────────────

DEFAULT_DISTILL_N_SAMPLES   = 8       # samples per question (more → more correct traces)
DEFAULT_DISTILL_MAX_TOKENS  = 16384   # teacher traces are long; needs to fit within max_seq_len
DEFAULT_DISTILL_MAX_SEQ_LEN = 20480   # prompt (~1-2K) + trace (~16K) headroom
DEFAULT_DISTILL_TEMPERATURE = 0.7     # higher than inference for trace diversity

# Set to False to exclude private.jsonl pseudo-labeled traces from SFT data.
# Can also be overridden at merge time with --no-private.
INCLUDE_PRIVATE_IN_SFT = True

# Richer teacher system prompts with explicit step labeling.
# These are intentionally more detailed than the inference prompts — the traces
# produced by teacher models become training targets for the student, so we want
# them to be thorough and well-structured.

DISTILL_SYSTEM_MATH = (
    "You are a world-class mathematician and educator. "
    "Solve the problem by explicitly working through these steps:\n"
    "Step 1 — Understand: State what the problem is asking and what information is given.\n"
    "Step 2 — Plan: Identify the mathematical concepts, theorems, or techniques that apply.\n"
    "Step 3 — Execute: Carry out each calculation in full detail, showing every algebraic step.\n"
    "Step 4 — Verify: Check your answer by substituting back or using an alternative method.\n"
    "Step 5 — Conclude: State the final answer inside \\boxed{}.\n\n"
    "Be thorough and explicit at every step."
)

DISTILL_SYSTEM_MCQ = (
    "You are a world-class mathematician and educator. "
    "Work through the problem by explicitly following these steps:\n"
    "Step 1 — Understand: State what the problem is asking.\n"
    "Step 2 — Analyze options: Briefly explain why each answer choice is plausible or not.\n"
    "Step 3 — Solve: Work through the problem in full detail to determine the correct answer.\n"
    "Step 4 — Verify: Confirm your answer is consistent with the problem conditions.\n"
    "Step 5 — Conclude: Output ONLY the letter of the correct answer inside \\boxed{}, "
    "e.g. \\boxed{C}.\n\n"
    "Be thorough and explicit at every step."
)
