#!/usr/bin/env python3
"""
Starter-code baseline — faithful port of starter_code_cse151b_comp.ipynb.

This is Experiment 1a: the original competition starter code with no
modifications to prompts, model settings, or generation logic. N=1 (no
self-consistency). Outputs a CSV that inference/evaluate.py can score.

Usage:
    # Score on public set (default) — single GPU with quantization
    CUDA_VISIBLE_DEVICES=0 python inference/starter.py --gpu

    # Smoke-test (first 10 questions, no GPU needed)
    python inference/starter.py --limit 10 --output /tmp/starter_test.csv

    # Then score:
    python inference/evaluate.py \\
        --results /deepfreeze/pnlong/school/cse151b/final/results/starter_baseline.csv \\
        --model "Qwen3-4B" --checkpoint base --n-samples 1 \\
        --notes "starter code notebook baseline"
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PUBLIC_DATA, RESULTS_DIR

# ── Prompts — verbatim from starter_code_cse151b_comp.ipynb ──────────────────

SYSTEM_PROMPT_MATH = (
    "You are an expert mathematician. Solve the problem step-by-step. "
    "Put your final answer inside \\boxed{}. "
    "If the problem has multiple sub-answers, separate them by commas inside a single \\boxed{}, "
    "e.g. \\boxed{3, 7}."
)

SYSTEM_PROMPT_MCQ = (
    "You are an expert mathematician. "
    "Read the problem and the answer choices below, then select the single best answer. "
    "Output ONLY the letter of your chosen option inside \\boxed{}, e.g. \\boxed{C}."
)

MODEL_ID   = "Qwen/Qwen3-4B-Thinking-2507"
MAX_TOKENS = 32768


def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) — verbatim from notebook."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Starter-code baseline (Experiment 1a)"
    )
    p.add_argument("--data",   default=str(PUBLIC_DATA),
                   help=f"Input JSONL (default: {PUBLIC_DATA})")
    p.add_argument("--output", default=str(RESULTS_DIR / "starter_baseline.csv"),
                   help="Output CSV (id, response) for evaluate.py")
    p.add_argument("--gpu",    action="store_true",
                   help="Enable GPU inference (device set via CUDA_VISIBLE_DEVICES)")
    p.add_argument("--limit",  type=int, default=None,
                   help="Process only the first N questions (smoke-test)")
    p.add_argument("--no-quantize", action="store_true",
                   help="Disable INT8 quantization (notebook uses quantization by default)")
    p.add_argument("--gpu-util",  type=float, default=0.85,
                   help="Fraction of GPU VRAM for model + KV cache (default: 0.85)")
    p.add_argument("--max-len",   type=int, default=16384,
                   help="vLLM max_model_len, prompt + generation (default: 16384, notebook value; "
                        "try 8192 to halve KV cache footprint if still OOM)")
    p.add_argument("--max-seqs",  type=int, default=64,
                   help="vLLM max_num_seqs, concurrent sequences (default: 64)")
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Load dataset ───────────────────────────────────────────────────────────
    data = [json.loads(line) for line in open(args.data)]
    if args.limit:
        data = data[:args.limit]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"Loaded {len(data)} questions  ({n_mcq} MCQ, {n_free} free-form)")

    # ── Load model ─────────────────────────────────────────────────────────────
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm_kwargs = dict(
        model=MODEL_ID,
        enable_prefix_caching=False,           # notebook default
        gpu_memory_utilization=args.gpu_util,
        max_model_len=args.max_len,            # notebook used 16384; lower to shrink KV cache if OOM
        trust_remote_code=True,
        max_num_seqs=args.max_seqs,
        max_num_batched_tokens=32768,
    )
    if not args.gpu:
        llm_kwargs["device"] = "cpu"
    if not args.no_quantize:
        llm_kwargs["quantization"]  = "bitsandbytes"
        llm_kwargs["load_format"]   = "bitsandbytes"

    llm = LLM(**llm_kwargs)

    # Sampling params — verbatim from notebook
    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )
    print("Model loaded.")

    # ── Build prompts ──────────────────────────────────────────────────────────
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        # apply_chat_template without enable_thinking — matches notebook exactly
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # ── Generate (single sample, N=1 — no self-consistency) ───────────────────
    print(f"Generating responses for {len(prompts)} questions...")
    outputs = llm.generate(prompts, sampling_params=sampling_params)
    responses = [out.outputs[0].text.strip() for out in outputs]
    print("Generation complete.")

    # ── Save CSV ───────────────────────────────────────────────────────────────
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "response"])
        for item, response in zip(data, responses):
            writer.writerow([item["id"], response])

    print(f"Saved {len(responses)} responses  →  {out_path}")
    print(f"\nScore with:")
    print(f"  python inference/evaluate.py --results {out_path} \\")
    print(f"      --model 'Qwen3-4B' --checkpoint base --n-samples 1 \\")
    print(f"      --notes 'starter code notebook baseline'")


if __name__ == "__main__":
    main()
