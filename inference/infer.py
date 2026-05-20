#!/usr/bin/env python3
"""
Qwen3-4B inference with self-consistency voting → submission CSV.

GPU device selection is controlled entirely via CUDA_VISIBLE_DEVICES (set
externally before running). Pass --gpu to enable GPU inference; omit it to
run on CPU (not recommended for production, but useful for import checks).

When launched by ``inference/infer_parallel.py``, workers get the environment
variable ``INFER_PARALLEL_WORKER=1`` so tqdm is disabled and verbose logs can be
captured cleanly in per-shard ``*.shardK.log`` files.

Run on private test set (submission):
    CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu

Run on public set (local eval):
    CUDA_VISIBLE_DEVICES=0 python inference/infer.py --gpu --data data/public.jsonl --output results/public.csv

Quick smoke-test (1 sample, 20 questions, CPU-only):
    python inference/infer.py --n-samples 1 --limit 20 --output /tmp/test.csv

Multi-GPU tensor parallel (2 GPUs):
    CUDA_VISIBLE_DEVICES=0,1 python inference/infer.py --gpu --tp 2

Multi-GPU data parallel (throughput — one full model copy per GPU):
    python inference/infer_parallel.py --gpu
    CUDA_VISIBLE_DEVICES=0,1 python inference/infer_parallel.py --gpu --output results/sub.csv
"""

import argparse
import csv
import os
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
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from constants import (
    DEFAULT_MODEL,
    DEFAULT_N_SAMPLES,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    DEFAULT_TOP_K,
    DEFAULT_MIN_P,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_GPU_UTIL,
    DEFAULT_QUANTIZE_GPU_UTIL,
    DEFAULT_MAX_NUM_SEQS,
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    SYSTEM_MATH,
    SYSTEM_MCQ,
    MULTI_ANS_NOTE,
)
from config import PRIVATE_DATA, RESULTS_DIR, HF_CACHE_DIR
from sft.progress_callbacks import (
    adapter_lora_rank,
    resolve_base_and_adapter,
    resolve_checkpoint_latest_path,
)
from inference.utils import (
    load_jsonl,
    build_prompt,
    apply_chat_template_safe,
    count_ans_slots,
    majority_vote,
    is_deepseek_r1_vllm_special_case,
    normalize_model_ref,
)


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Qwen3-4B self-consistency inference")
    p.add_argument("--data",        default=str(PRIVATE_DATA),
                   help=f"Input JSONL  (default: {PRIVATE_DATA})")
    p.add_argument("--output",      default=str(RESULTS_DIR / "submission.csv"),
                   help="Output CSV  (default: $STORAGE_DIR/results/submission.csv)")
    p.add_argument("--model",       default=DEFAULT_MODEL)
    p.add_argument("--n-samples",   type=int,   default=DEFAULT_N_SAMPLES,
                   help=f"Samples per question for self-consistency (default: {DEFAULT_N_SAMPLES})")
    p.add_argument("--gpu",         action="store_true",
                   help="Enable GPU inference. Device selection is via CUDA_VISIBLE_DEVICES "
                        "(set externally, e.g. CUDA_VISIBLE_DEVICES=0,1 before running).")
    p.add_argument("--tp",          type=int,   default=1,
                   help="Tensor-parallel degree — must match the number of visible GPUs (default: 1)")
    p.add_argument("--max-tokens",  type=int,   default=DEFAULT_MAX_TOKENS)
    p.add_argument("--max-seq-len", type=int,   default=DEFAULT_MAX_SEQ_LEN)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p",       type=float, default=DEFAULT_TOP_P)
    p.add_argument("--top-k",       type=int,   default=DEFAULT_TOP_K)
    p.add_argument("--gpu-util",    type=float, default=None,
                   help=f"GPU VRAM fraction for vLLM "
                        f"(default: {DEFAULT_QUANTIZE_GPU_UTIL} with --quantize, "
                        f"{DEFAULT_GPU_UTIL} otherwise)")
    p.add_argument("--quantize",    action="store_true",
                   help="INT8 bitsandbytes quantization (saves VRAM, disables prefix cache)")
    p.add_argument("--chunk-size",  type=int,   default=10,
                   help="Questions per generation batch; results written after each chunk (default: 10)")
    p.add_argument("--reset",       action="store_true",
                   help="Ignore existing output and reprocess all questions from scratch")
    p.add_argument("--limit",       type=int,   default=None,
                   help="Process only the first N questions (smoke-testing)")
    p.add_argument("--num-shards", type=int, default=1,
                   help="Split the (possibly --limit-truncated) dataset into this many "
                        "index-based shards; only rows with index %% num-shards == shard-id "
                        "are processed (default: 1 = full dataset).")
    p.add_argument("--shard-id",   type=int, default=0,
                   help="Which shard to run [0, num-shards); used with --num-shards (default: 0).")

    # ── Prompt routing ────────────────────────────────────────────────────────
    p.add_argument("--use-router", action="store_true",
                   help="Enable prompt routing (format + optional topic refinements).")
    p.add_argument("--router-secondary-llm", action="store_true",
                   help="Use a lightweight LLM to suggest a topic label; invalid output "
                        "falls back to topic_taxonomy.classify_problem (primary route stays deterministic).")
    p.add_argument("--router-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Router model for --router-secondary-llm (default: Qwen2.5-0.5B-Instruct).")
    p.add_argument("--router-device", default="cpu", choices=["cpu", "auto"],
                   help="Device for router model: cpu (safe) or auto (uses available accelerators).")
    args = p.parse_args()
    if args.num_shards < 1:
        p.error("--num-shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        p.error("--shard-id must satisfy 0 <= shard-id < num-shards")
    if args.num_shards > 1 and args.tp != 1:
        p.error("With --num-shards > 1, use --tp 1 (one GPU per shard process). "
                "Use inference/infer_parallel.py for multi-GPU throughput.")
    return args


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_done_ids(out_path: Path) -> set:
    """Return the set of question IDs already written to the output CSV."""
    if not out_path.exists():
        return set()
    with open(out_path, newline="") as f:
        return {str(row["id"]) for row in csv.DictReader(f)}


def append_rows(rows: list[dict], out_path: Path, write_header: bool) -> None:
    """Append voted rows to the output CSV, writing the header when needed."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    model_path = resolve_checkpoint_latest_path(Path(args.model))
    base_id, adapter_dir = resolve_base_and_adapter(model_path)
    vllm_model = base_id if adapter_dir else normalize_model_ref(model_path)
    lora_request: LoRARequest | None = None
    if adapter_dir:
        lora_request = LoRARequest("sft_adapter", 1, adapter_dir)
    model_label = (
        f"{base_id} + LoRA ({Path(adapter_dir).name})"
        if adapter_dir
        else vllm_model
    )
    # When infer_parallel.py runs workers, it sets INFER_PARALLEL_WORKER=1 so tqdm/log
    # output is plain print (full logs go to per-shard *.log files).
    _parallel_worker = os.environ.get("INFER_PARALLEL_WORKER", "").lower() in (
        "1", "true", "yes",
    )
    # When --gpu IS set, CUDA_VISIBLE_DEVICES must be set externally by the caller.

    gpu_util = args.gpu_util if args.gpu_util is not None else (
        DEFAULT_QUANTIZE_GPU_UTIL if args.quantize else DEFAULT_GPU_UTIL
    )
    out_path = Path(args.output)
    N        = args.n_samples

    # ── Load data & resume ────────────────────────────────────────────────────
    data = load_jsonl(Path(args.data))
    if args.limit:
        data = data[: args.limit]
    if args.num_shards > 1:
        data = [
            item for idx, item in enumerate(data)
            if idx % args.num_shards == args.shard_id
        ]

    done_ids = set() if args.reset else load_done_ids(out_path)
    todo     = [item for item in data if str(item["id"]) not in done_ids]

    n_mcq  = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"\n{'='*55}")
    print(f"  Data   : {args.data}")
    print(f"  Output : {out_path}")
    print(f"  Model  : {model_label}")
    device_str = f"GPU (tp={args.tp}, gpu_util={gpu_util:.0%}" + (", INT8 quantized" if args.quantize else "") + ")" if args.gpu else "CPU"
    print(f"  Device : {device_str}")
    if args.num_shards > 1:
        print(f"  Shard    : {args.shard_id + 1}/{args.num_shards} "
              f"(rows where dataset_index % {args.num_shards} == {args.shard_id})")
    print(f"  Questions: {len(data)} total  ({n_mcq} MCQ, {n_free} free-form)")
    print(f"  Samples  : {N} per question  →  {len(data) * N} total generations")
    print(f"{'='*55}\n")

    if done_ids:
        print(f"[resume] {len(done_ids)} questions already done, {len(todo)} remaining")
    if not todo:
        print("Nothing to do — all questions already processed. Use --reset to rerun.")
        return

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tok_source = adapter_dir or vllm_model
    print(f"[1/3] Loading tokenizer ({tok_source})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(vllm_model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"      Tokenizer ready.")

    # ── Router (optional) ─────────────────────────────────────────────────────
    router = None
    if args.use_router:
        from inference.router import RuleBasedRouter, LLMSecondaryRouter, build_routed_prompts

        if args.router_secondary_llm:
            # NOTE: loading a second vLLM instance is typically heavy; we use a tiny HF model.
            # This is usually only worthwhile if you think topic refinements help measurably.
            router = LLMSecondaryRouter(
                model=args.router_model,
                device=("cpu" if args.router_device == "cpu" else "auto"),
            )
            print(f"[router] enabled (topic via LLM with taxonomy fallback: {args.router_model}, device={args.router_device})")
        else:
            router = RuleBasedRouter(enable_topic_refinements=True)
            print(f"[router] enabled (topic_taxonomy + optional refinements)")

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"[2/3] Loading model weights into {'GPU' if args.gpu else 'CPU'} memory...")
    print(f"      (Note: first run may take several extra minutes for torch compilation)")
    llm_kwargs = dict(
        model=vllm_model,
        trust_remote_code=True,
        max_model_len=args.max_seq_len,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    if adapter_dir:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = adapter_lora_rank(adapter_dir)
        print(f"      LoRA adapter: {adapter_dir} (rank={llm_kwargs['max_lora_rank']})")
    if args.gpu:
        llm_kwargs["tensor_parallel_size"]   = args.tp
        llm_kwargs["gpu_memory_utilization"] = gpu_util
        llm_kwargs["enable_prefix_caching"]  = not args.quantize
        if args.quantize:
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"]  = "bitsandbytes"
    else:
        llm_kwargs["device"] = "cpu"

    if is_deepseek_r1_vllm_special_case(tokenizer, vllm_model):
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    print(f"      Model ready.")
    if is_deepseek_r1_vllm_special_case(tokenizer, vllm_model):
        print("      [vllm] DeepSeek-R1: string prompts + enforce_eager=True "
              "(vLLM CUDA-graph / compile glitches on this family).")

    sampling_params = SamplingParams(
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=DEFAULT_MIN_P,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
    )

    # ── Chunked generate + write ───────────────────────────────────────────────
    # Process args.chunk_size questions at a time; vote and flush to disk after
    # each chunk so progress is preserved if the run is interrupted.
    print(f"[3/3] Generating responses and voting  "
          f"(chunk_size={args.chunk_size}, writing to disk every {args.chunk_size} questions)...")
    need_header   = args.reset or not out_path.exists()
    total_written = len(done_ids)

    pbar   = tqdm(total=len(todo), desc="Questions", unit="q", initial=0, disable=_parallel_worker)
    chunks = range(0, len(todo), args.chunk_size)
    for chunk_start in chunks:
        chunk = todo[chunk_start : chunk_start + args.chunk_size]

        # Pre-tokenize once per unique prompt (avoids vLLM re-tokenising N copies),
        # except DeepSeek-R1 / Distill: use string prompts so vLLM tokenization matches.
        use_str = is_deepseek_r1_vllm_special_case(tokenizer, vllm_model)
        chunk_inputs: list[dict] = []
        if router is None:
            for item in chunk:
                system, user = build_prompt(
                    item["question"], item.get("options"),
                    SYSTEM_MATH, SYSTEM_MCQ, MULTI_ANS_NOTE,
                )
                text = apply_chat_template_safe(
                    tokenizer,
                    [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                )
                if use_str:
                    chunk_inputs.append({"prompt": text})
                else:
                    chunk_inputs.append({
                        "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False),
                    })
        else:
            routed = build_routed_prompts(router, chunk)
            for system, user in routed:
                text = apply_chat_template_safe(
                    tokenizer,
                    [{"role": "system", "content": system},
                     {"role": "user",   "content": user}],
                )
                if use_str:
                    chunk_inputs.append({"prompt": text})
                else:
                    chunk_inputs.append({
                        "prompt_token_ids": tokenizer.encode(text, add_special_tokens=False),
                    })

        all_prompts = [dict(inp) for inp in chunk_inputs for _ in range(N)]
        outputs     = llm.generate(
            all_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        flat_resps  = [out.outputs[0].text.strip() for out in outputs]

        rows = []
        for i, item in enumerate(chunk):
            group   = flat_resps[i * N : (i + 1) * N]
            n_slots = count_ans_slots(item["question"])
            is_mcq  = bool(item.get("options"))
            winner  = majority_vote(group, n_slots, is_mcq)
            rows.append({"id": item["id"], "response": winner})

        append_rows(rows, out_path, write_header=need_header)
        need_header    = False
        total_written += len(rows)
        pbar.update(len(rows))
        chunk_msg = (
            f"  wrote {len(rows)} rows  (total {total_written}/{len(data)})  →  {out_path}"
        )
        if _parallel_worker:
            print(chunk_msg, flush=True)
        else:
            tqdm.write(chunk_msg)

    pbar.close()
    print(f"\nDone. {total_written}/{len(data)} questions written to {out_path}")


if __name__ == "__main__":
    main()
