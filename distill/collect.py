#!/usr/bin/env python3
"""
Run a teacher model on public.jsonl and/or private.jsonl to collect reasoning
traces for SFT knowledge distillation.

GPU device selection is via CUDA_VISIBLE_DEVICES (set externally). Pass --gpu
to enable GPU inference.

Collect from both splits (default):
    CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu \\
        --model Qwen/Qwen3-32B --quantize

Public only:
    CUDA_VISIBLE_DEVICES=0 python distill/collect.py --gpu \\
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B --quantize --public-only

Two-GPU tensor parallel:
    CUDA_VISIBLE_DEVICES=0,1 python distill/collect.py --gpu --tp 2 \\
        --model Qwen/Qwen2.5-Math-72B-Instruct --quantize

Smoke-test (5 public questions, 2 samples, no GPU):
    python distill/collect.py --model Qwen/Qwen3-32B \\
        --public-only --limit 5 --n-samples 2

Output files (per model, under DISTILL_DIR/{model-slug}/):
  public_traces.jsonl   {id, question, options, answer, response}  — verified correct
  private_traces.jsonl  {id, question, options, response}          — pseudo-labeled
"""

import argparse
import os
import random
import sys
from pathlib import Path

# ── Set HF_HOME before ANY huggingface/transformers/vllm imports ──────────────
# HuggingFace locks in the cache directory at import time; setting os.environ
# after the import is too late and the default ~/.cache/huggingface is used.
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import HF_CACHE_DIR, HF_TOKEN, HF_XET_CACHE
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")   # use HTTP download, not xet storage
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
if HF_XET_CACHE:
    os.environ.setdefault("HF_XET_CACHE", HF_XET_CACHE)  # local-disk path for xet staging

from tqdm import tqdm
from transformers import AutoTokenizer, AutoConfig
from vllm import LLM, SamplingParams

from constants import (
    DEFAULT_MODEL,
    DEFAULT_DISTILL_MAX_SEQ_LEN,
    DEFAULT_MIN_P,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GPU_UTIL,
    DEFAULT_QUANTIZE_GPU_UTIL,
    DEFAULT_MAX_NUM_SEQS,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_DISTILL_N_SAMPLES,
    DEFAULT_DISTILL_MAX_TOKENS,
    DEFAULT_DISTILL_MAX_SEQ_LEN,
    DEFAULT_DISTILL_TEMPERATURE,
    DISTILL_SYSTEM_MATH,
    DISTILL_SYSTEM_MCQ,
    MULTI_ANS_NOTE,
)
from config import PUBLIC_DATA, PRIVATE_DATA, HF_CACHE_DIR
from distill.utils import (
    load_jsonl,
    save_jsonl,
    build_prompt,
    apply_chat_template_safe,
    count_ans_slots,
    majority_vote,
    model_slug,
    traces_dir,
    verify_trace,
)
from judger import Judger


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Collect distillation traces from a teacher model")
    p.add_argument("--model",        default=DEFAULT_MODEL,
                   help="HuggingFace model ID of the teacher")
    p.add_argument("--n-samples",    type=int,   default=DEFAULT_DISTILL_N_SAMPLES,
                   help=f"Samples per question (default: {DEFAULT_DISTILL_N_SAMPLES})")
    p.add_argument("--gpu",          action="store_true",
                   help="Enable GPU inference (device via CUDA_VISIBLE_DEVICES)")
    p.add_argument("--tp",           type=int,   default=1,
                   help="Tensor-parallel degree (must match number of visible GPUs)")
    p.add_argument("--max-tokens",   type=int,   default=DEFAULT_DISTILL_MAX_TOKENS)
    p.add_argument("--max-seq-len",  type=int,   default=DEFAULT_DISTILL_MAX_SEQ_LEN)
    p.add_argument("--temperature",  type=float, default=DEFAULT_DISTILL_TEMPERATURE)
    p.add_argument("--gpu-util",     type=float, default=None,
                   help=f"GPU VRAM fraction (default: {DEFAULT_QUANTIZE_GPU_UTIL} with "
                        f"--quantize, {DEFAULT_GPU_UTIL} otherwise)")
    p.add_argument("--quantize",     action="store_true",
                   help="INT8 bitsandbytes quantization (saves VRAM, disables prefix cache)")
    p.add_argument("--public-only",  action="store_true",
                   help="Only process public.jsonl")
    p.add_argument("--private-only", action="store_true",
                   help="Only process private.jsonl")
    p.add_argument("--limit",        type=int,   default=None,
                   help="Process only the first N questions per split (smoke-testing)")
    p.add_argument("--chunk-size",   type=int,   default=10,
                   help="Questions per generation batch; results written after each chunk (default: 10)")
    p.add_argument("--reset",        action="store_true",
                   help="Ignore existing output files and reprocess all questions from scratch")
    return p.parse_args()


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_existing_ids(path: Path) -> set:
    """Return the set of question IDs already written to an output file."""
    if not path.exists():
        return set()
    return {str(r["id"]) for r in load_jsonl(path)}


def get_model_max_seq_len(model_id: str, requested: int) -> int:
    """
    Cap max_seq_len at the model's actual context limit.
    Reads the model config from HF cache (no download needed if already cached).
    Returns min(requested, model_max_position_embeddings).
    """
    try:
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_max = getattr(cfg, "max_position_embeddings", None)
        if model_max is not None and requested > model_max:
            print(f"  [max_seq_len] model max_position_embeddings={model_max} < requested {requested}; "
                  f"capping to {model_max}")
            return model_max
    except Exception as e:
        print(f"  [max_seq_len] could not read model config ({e}); using requested {requested}")
    return requested


def build_token_ids(items: list[dict], tokenizer) -> list[list[int]]:
    """
    Pre-tokenize one prompt per question and return token ID lists.
    Pre-tokenising avoids vLLM re-tokenising every duplicate when N > 1.
    """
    ids_list = []
    for item in items:
        system, user = build_prompt(
            item["question"], item.get("options"),
            DISTILL_SYSTEM_MATH, DISTILL_SYSTEM_MCQ, MULTI_ANS_NOTE,
        )
        text = apply_chat_template_safe(
            tokenizer,
            [{"role": "system", "content": system},
             {"role": "user",   "content": user}],
        )
        ids_list.append(tokenizer.encode(text, add_special_tokens=False))
    return ids_list


def generate_chunk(prompt_ids: list[list[int]], llm: LLM,
                   sampling_params: SamplingParams, n: int) -> list[list[str]]:
    """
    Generate N responses per question for a chunk of questions.
    Returns a list of length len(prompt_ids), each element a list of N strings.
    """
    repeated = [{"prompt_token_ids": ids} for ids in prompt_ids for _ in range(n)]
    outputs  = llm.generate(repeated, sampling_params=sampling_params)
    flat     = [out.outputs[0].text.strip() for out in outputs]
    return [flat[i * n : (i + 1) * n] for i in range(len(prompt_ids))]


# ── Sanity check ──────────────────────────────────────────────────────────────

def sanity_check(out_path: Path, split: str) -> None:
    """Print a randomly sampled question + response for a quick visual check."""
    if not out_path.exists():
        return
    records = load_jsonl(out_path)
    if not records:
        print(f"\n  [sanity check — {split}] no records found in {out_path}")
        return

    rec = random.choice(records)
    q   = rec["question"]
    r   = rec["response"]
    q_display = q if len(q) <= 300 else q[:300] + " …"
    r_display = r if len(r) <= 600 else r[:600] + " …"

    print(f"\n{'─' * 60}")
    print(f"  SANITY CHECK — {split}  (id={rec['id']}, "
          f"{'MCQ' if rec.get('options') else 'free-form'})")
    print(f"{'─' * 60}")
    if rec.get("answer") is not None:
        print(f"  gold   : {rec['answer']}")
    print(f"  question:\n    {q_display}")
    print(f"\n  response:\n    {r_display}")
    print(f"{'─' * 60}\n")


# ── Per-split processing ───────────────────────────────────────────────────────

def process_public_chunked(data: list[dict], out_path: Path,
                            llm: LLM, sampling_params: SamplingParams,
                            tokenizer, n: int, chunk_size: int,
                            judger: Judger, reset: bool) -> None:
    """
    Chunked, append-safe public trace collection.
    For each question, keeps every response that Judger verifies as correct.
    Writes results after every chunk_size questions so progress is preserved.
    """
    existing_ids = set() if reset else load_existing_ids(out_path)
    todo = [item for item in data if str(item["id"]) not in existing_ids]

    if existing_ids:
        print(f"  [resume] {len(existing_ids)} public questions already done, "
              f"{len(todo)} remaining")
    if not todo:
        print("  Public: nothing to do — all questions already processed.")
        return

    total_correct = total_responses = 0
    pbar = tqdm(total=len(todo), desc="Public traces", unit="q")

    for chunk_start in range(0, len(todo), chunk_size):
        chunk      = todo[chunk_start : chunk_start + chunk_size]
        prompt_ids = build_token_ids(chunk, tokenizer)
        groups     = generate_chunk(prompt_ids, llm, sampling_params, n)

        new_records = []
        for item, responses in zip(chunk, groups):
            gold   = item["answer"]
            is_mcq = bool(item.get("options"))
            total_responses += len(responses)
            for resp in responses:
                if verify_trace(resp, gold, is_mcq, judger):
                    new_records.append({
                        "id":       item["id"],
                        "question": item["question"],
                        "options":  item.get("options"),
                        "answer":   gold,
                        "response": resp,
                    })
                    total_correct += 1

        # Append to existing file
        existing_records = load_jsonl(out_path) if out_path.exists() else []
        save_jsonl(existing_records + new_records, out_path)

        pbar.update(len(chunk))
        tqdm.write(f"  wrote {len(new_records)} correct traces from chunk "
                   f"({chunk_start + len(chunk)}/{len(todo)})  →  {out_path}")

    pbar.close()
    print(f"  Public: {total_correct} correct traces from {total_responses} responses "
          f"({len(data)} questions total)")


def process_private_chunked(data: list[dict], out_path: Path,
                             llm: LLM, sampling_params: SamplingParams,
                             tokenizer, n: int, chunk_size: int,
                             reset: bool) -> None:
    """
    Chunked, append-safe private trace collection.
    For each question, picks the majority-vote response as a pseudo-labeled trace.
    Writes results after every chunk_size questions so progress is preserved.
    """
    existing_ids = set() if reset else load_existing_ids(out_path)
    todo = [item for item in data if str(item["id"]) not in existing_ids]

    if existing_ids:
        print(f"  [resume] {len(existing_ids)} private questions already done, "
              f"{len(todo)} remaining")
    if not todo:
        print("  Private: nothing to do — all questions already processed.")
        return

    total_new = 0
    pbar = tqdm(total=len(todo), desc="Private traces", unit="q")

    for chunk_start in range(0, len(todo), chunk_size):
        chunk      = todo[chunk_start : chunk_start + chunk_size]
        prompt_ids = build_token_ids(chunk, tokenizer)
        groups     = generate_chunk(prompt_ids, llm, sampling_params, n)

        new_records = []
        for item, responses in zip(chunk, groups):
            n_slots = count_ans_slots(item["question"])
            is_mcq  = bool(item.get("options"))
            winner  = majority_vote(responses, n_slots, is_mcq)
            new_records.append({
                "id":       item["id"],
                "question": item["question"],
                "options":  item.get("options"),
                "response": winner,
            })

        existing_records = load_jsonl(out_path) if out_path.exists() else []
        save_jsonl(existing_records + new_records, out_path)
        total_new += len(new_records)

        pbar.update(len(chunk))
        tqdm.write(f"  wrote {len(new_records)} pseudo-labeled traces from chunk "
                   f"({chunk_start + len(chunk)}/{len(todo)})  →  {out_path}")

    pbar.close()
    print(f"  Private: {total_new} new pseudo-labeled traces ({len(data)} questions total)")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Debug: show exactly where HF will download model weights ──────────────
    import shutil
    hf_home = Path(os.environ["HF_HOME"])
    print(f"\n[cache] HF_HOME      : {hf_home}")
    print(f"[cache] HF_HOME exists: {hf_home.exists()}")
    total, used, free = shutil.disk_usage(hf_home if hf_home.exists() else hf_home.parent)
    print(f"[cache] Disk usage on that partition: "
          f"{used / 1e9:.1f} GB used / {total / 1e9:.1f} GB total  "
          f"({free / 1e9:.1f} GB free)")
    # Also check for any env vars that could redirect HF downloads
    for var in ("HUGGINGFACE_HUB_CACHE", "HF_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE"):
        val = os.environ.get(var)
        if val:
            print(f"[cache] {var} = {val}  ← may override HF_HOME")

    slug    = model_slug(args.model)
    out_dir = traces_dir(args.model)
    public_out  = out_dir / "public_traces.jsonl"
    private_out = out_dir / "private_traces.jsonl"

    do_public  = not args.private_only
    do_private = not args.public_only

    # ── Load data ──────────────────────────────────────────────────────────────
    pub_data  = load_jsonl(PUBLIC_DATA)  if do_public  else []
    priv_data = load_jsonl(PRIVATE_DATA) if do_private else []
    if args.limit:
        pub_data  = pub_data[: args.limit]
        priv_data = priv_data[: args.limit]

    all_data = pub_data + priv_data
    if not all_data:
        print("Nothing to process.")
        return

    # ── Resolve max_seq_len (cap to model's actual limit) ─────────────────────
    print(f"Checking model config for max_position_embeddings: {args.model}")
    max_seq_len = get_model_max_seq_len(args.model, args.max_seq_len)
    max_tokens  = min(args.max_tokens, max_seq_len - 512)   # keep 512 tokens for the prompt

    print(f"\n{'='*55}")
    print(f"  Model  : {args.model}  (slug: {slug})")
    print(f"  Public : {len(pub_data)} questions  →  {public_out}")
    print(f"  Private: {len(priv_data)} questions  →  {private_out}")
    print(f"  Samples: {args.n_samples}  |  max_tokens: {max_tokens}  "
          f"|  temperature: {args.temperature}")
    print(f"  max_seq_len: {max_seq_len}  |  chunk_size: {args.chunk_size}")
    print(f"{'='*55}\n")

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    print(f"[1/3] Loading tokenizer ({args.model})...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"      Tokenizer ready.")

    # ── Load model ────────────────────────────────────────────────────────────
    gpu_util = args.gpu_util if args.gpu_util is not None else (
        DEFAULT_QUANTIZE_GPU_UTIL if args.quantize else DEFAULT_GPU_UTIL
    )
    print(f"[2/3] Loading model weights into {'GPU' if args.gpu else 'CPU'} memory...")
    print(f"      (Note: first run may take several extra minutes for torch compilation)")

    llm_kwargs = dict(
        model=args.model,
        trust_remote_code=True,
        max_model_len=max_seq_len,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    if args.gpu:
        llm_kwargs["tensor_parallel_size"]   = args.tp
        llm_kwargs["gpu_memory_utilization"] = gpu_util
        llm_kwargs["enable_prefix_caching"]  = not args.quantize
        if args.quantize:
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"]  = "bitsandbytes"
    else:
        llm_kwargs["device"] = "cpu"

    llm = LLM(**llm_kwargs)
    print(f"      Model ready.")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=args.temperature,
        top_p=0.95,
        top_k=20,
        min_p=DEFAULT_MIN_P,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
    )

    # ── Generate ──────────────────────────────────────────────────────────────
    judger = Judger(strict_extract=False)
    N      = args.n_samples

    print(f"[3/3] Generating and writing traces (chunk_size={args.chunk_size}, "
          f"writing to disk every {args.chunk_size} questions)...")

    if do_public and pub_data:
        process_public_chunked(pub_data, public_out, llm, sampling_params,
                               tokenizer, N, args.chunk_size, judger, args.reset)
        sanity_check(public_out, "public")

    if do_private and priv_data:
        process_private_chunked(priv_data, private_out, llm, sampling_params,
                                tokenizer, N, args.chunk_size, args.reset)
        sanity_check(private_out, "private")

    print(f"\nDone. Traces saved to {out_dir}/")


if __name__ == "__main__":
    main()
