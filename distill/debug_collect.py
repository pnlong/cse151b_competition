#!/usr/bin/env python3
"""
Run the same teacher stack as distill/collect.py on 1–2 questions and print
what the model returned and whether verify_trace() accepts it.

By default prints the **full** question and the **full** raw model output
(chain-of-thought + final answer). Use --question-chars / --response-chars to
truncate if needed.

Does not write public_traces.jsonl / private_traces.jsonl.

Example (match pipeline two-GPU teacher):
    CUDA_VISIBLE_DEVICES=0,1 python distill/debug_collect.py --gpu --tp 2 \\
        --quantize --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B

CPU smoke (slow / may OOM on large models):
    python distill/debug_collect.py --model Qwen/Qwen3-4B-Thinking-2507 \\
        --num-questions 1 --n-samples 1
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# ── HF_HOME before huggingface / vllm imports (same as collect.py) ────────────
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_CACHE_DIR, HF_TOKEN, HF_XET_CACHE

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
if HF_XET_CACHE:
    os.environ.setdefault("HF_XET_CACHE", HF_XET_CACHE)

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

from constants import (
    DEFAULT_MODEL,
    DEFAULT_MIN_P,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GPU_UTIL,
    DEFAULT_QUANTIZE_GPU_UTIL,
    DEFAULT_MAX_NUM_SEQS,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_DISTILL_MAX_TOKENS,
    DEFAULT_DISTILL_MAX_SEQ_LEN,
    DEFAULT_DISTILL_TEMPERATURE,
    DISTILL_SYSTEM_MATH,
    DISTILL_SYSTEM_MCQ,
    MULTI_ANS_NOTE,
)
from config import PUBLIC_DATA, PRIVATE_DATA
from distill.collect import build_vllm_request_dicts, generate_chunk, get_model_max_seq_len
from distill.utils import (
    extract_last_boxed,
    extract_letter,
    load_jsonl,
    verify_trace,
)
from inference.utils import (
    final_answer_segment,
    is_deepseek_r1_vllm_special_case,
    tokenizer_chat_template_debug,
)
from judger import Judger


def _clip(s: str, max_chars: int) -> str:
    s = s.replace("\r", "")
    if max_chars <= 0 or len(s) <= max_chars:
        return s
    half = max_chars // 2
    return s[:half] + "\n\n... [middle omitted for display] ...\n\n" + s[-half:]


def parse_args():
    p = argparse.ArgumentParser(description="Debug teacher generation + verify_trace (no trace files)")
    p.add_argument("--model", default=DEFAULT_MODEL, help="HuggingFace model ID")
    p.add_argument("--num-questions", type=int, default=2, metavar="N",
                   help="How many dataset rows to run (default: 2)")
    p.add_argument("--n-samples", type=int, default=1,
                   help="Responses per question (default: 1)")
    p.add_argument("--gpu", action="store_true", help="GPU via CUDA_VISIBLE_DEVICES")
    p.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    p.add_argument("--quantize", action="store_true", help="bitsandbytes INT8")
    p.add_argument("--max-tokens", type=int, default=DEFAULT_DISTILL_MAX_TOKENS)
    p.add_argument("--max-seq-len", type=int, default=DEFAULT_DISTILL_MAX_SEQ_LEN)
    p.add_argument("--temperature", type=float, default=DEFAULT_DISTILL_TEMPERATURE)
    p.add_argument("--gpu-util", type=float, default=None)
    p.add_argument("--split", choices=("public", "private"), default="public",
                   help="Which JSONL to draw questions from (default: public — has gold labels)")
    p.add_argument("--start-index", type=int, default=0,
                   help="0-based offset into the chosen split (default: 0)")
    p.add_argument("--question-chars", type=int, default=0,
                   help="Max characters of question text to print (0 = full, default)")
    p.add_argument("--response-chars", type=int, default=0,
                   help="Max characters of model output / extract_ans to print (0 = full, default)")
    return p.parse_args()


def main():
    args = parse_args()
    data_path = PUBLIC_DATA if args.split == "public" else PRIVATE_DATA
    rows = load_jsonl(data_path)
    start = max(0, args.start_index)
    end = start + args.num_questions
    if start >= len(rows):
        print(f"No rows at start_index={start} (file has {len(rows)} lines).")
        sys.exit(1)
    items = rows[start:end]

    max_seq_len = get_model_max_seq_len(args.model, args.max_seq_len)
    max_tokens = min(args.max_tokens, max_seq_len - 512)

    gpu_util = args.gpu_util if args.gpu_util is not None else (
        DEFAULT_QUANTIZE_GPU_UTIL if args.quantize else DEFAULT_GPU_UTIL
    )

    print(f"[debug] split={args.split} path={data_path}")
    print(f"[debug] rows [{start}:{end})  |  model={args.model}")
    print(f"[debug] max_seq_len={max_seq_len} max_tokens={max_tokens} "
          f"temperature={args.temperature} n_samples={args.n_samples}\n")

    print(f"[1/2] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[debug] {tokenizer_chat_template_debug(tokenizer)}")
    ds = is_deepseek_r1_vllm_special_case(tokenizer, args.model)
    print(f"[debug] DeepSeek-R1 vLLM mitigations active: {ds}  "
          f"(string prompt + enforce_eager when loading LLM)")

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        max_model_len=max_seq_len,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    if args.gpu:
        llm_kwargs["tensor_parallel_size"] = args.tp
        llm_kwargs["gpu_memory_utilization"] = gpu_util
        llm_kwargs["enable_prefix_caching"] = not args.quantize
        if args.quantize:
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
    else:
        llm_kwargs["device"] = "cpu"

    if is_deepseek_r1_vllm_special_case(tokenizer, args.model):
        llm_kwargs["enforce_eager"] = True

    print(f"[2/2] Loading model (gpu={args.gpu}, tp={args.tp}, quantize={args.quantize})...")
    llm = LLM(**llm_kwargs)

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=DEFAULT_MIN_P,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
    )

    judger = Judger(strict_extract=False)
    requests = build_vllm_request_dicts(items, tokenizer, args.model)
    groups = generate_chunk(requests, llm, sampling_params, args.n_samples)

    bar = "=" * 72
    for item, responses in zip(items, groups):
        qid = item["id"]
        gold = item.get("answer")
        is_mcq = bool(item.get("options"))
        qtext = item["question"]

        print(bar)
        print(f"id={qid}  MCQ={is_mcq}")
        if args.split == "public":
            print(f"gold={gold!r}")
        else:
            print("gold=(none — private split; verify_trace not meaningful)")
        print(f"question ({len(qtext)} chars):\n{_clip(qtext, args.question_chars)}\n")

        for si, resp in enumerate(responses):
            ok = verify_trace(resp, gold, is_mcq, judger) if args.split == "public" else None
            seg = final_answer_segment(resp)
            boxed_seg = extract_last_boxed(seg)
            boxed_full = extract_last_boxed(resp)

            print(f"--- sample {si + 1}/{len(responses)}  len={len(resp)} chars ---")
            if is_mcq:
                letter = extract_letter(resp)
                print(f"extract_letter={letter!r}  (gold letter {gold!r})")
            else:
                extracted = ""
                try:
                    extracted = judger.extract_ans(resp)
                except Exception as e:
                    extracted = f"<Judger.extract_ans error: {e}>"
                print(f"Judger.extract_ans={_clip(str(extracted), args.response_chars)!r}")
                print(f"last \\boxed{{}} in post-thinking segment: {boxed_seg!r}")
                print(f"last \\boxed{{}} in full response:        {boxed_full!r}")

            if args.split == "public":
                print(f"verify_trace -> {ok}")
            print(f"thought + response (raw model output):\n{_clip(resp, args.response_chars)}\n")

    print(bar)
    print("Done (no files written). For full collection: distill/collect.py ...")


if __name__ == "__main__":
    main()
