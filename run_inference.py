#!/usr/bin/env python3
"""
Course final submission entry point.

Duplicated from ``inference/infer.py`` (``main()``) per project plan — keep in sync
when changing the CLI pipeline. Canonical reproduction for Gradescope:

    micromamba activate cse151b_competition
    CUDA_VISIBLE_DEVICES=0 python run_inference.py

Override adapter location (e.g. after HuggingFace upload)::

    export SUBMISSION_MODEL=p1long/cse151b_competition
    CUDA_VISIBLE_DEVICES=0 python run_inference.py

Default Experiment 3c: GRPO ``checkpoint-best-reward``, router on, N=8, **no**
``--quantize`` (full precision).
"""

from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

# ── Repo root + HF_HOME before ANY huggingface/transformers/vllm imports ────────
_REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_REPO_ROOT))
from config import (  # noqa: E402
    CHECKPOINTS_DIR,
    HF_CACHE_DIR,
    HF_TOKEN,
    HF_XET_CACHE,
    PRIVATE_DATA,
    RESULTS_DIR,
)
os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
if HF_XET_CACHE:
    os.environ.setdefault("HF_XET_CACHE", HF_XET_CACHE)

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

from constants import (  # noqa: E402
    DEFAULT_MAX_NUM_BATCHED_TOKENS,
    DEFAULT_MAX_NUM_SEQS,
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MIN_P,
    DEFAULT_N_SAMPLES,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_QUANTIZE_GPU_UTIL,
    DEFAULT_GPU_UTIL,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    MULTI_ANS_NOTE,
    SYSTEM_MATH,
    SYSTEM_MCQ,
)
from sft.progress_callbacks import (
    adapter_lora_rank,
    resolve_base_and_adapter,
    resolve_checkpoint_latest_path,
)
from inference.utils import (
    apply_chat_template_safe,
    build_prompt,
    count_ans_slots,
    is_deepseek_r1_vllm_special_case,
    load_jsonl,
    majority_vote,
    normalize_model_ref,
)

#
# ── Verification defaults (Experiment 3c) ─────────────────────────────────────
# Prefer Hugging Face Hub path after uploading ``checkpoint-best-reward``::
#     export SUBMISSION_MODEL=p1long/cse151b_competition
# Local layout (GRPO Trainer output): ``$CHECKPOINTS_DIR/rl/checkpoint-best-reward``
#
_DEFAULT_RL_ADAPTER = CHECKPOINTS_DIR / "rl" / "checkpoint-best-reward"


def _default_model_path() -> str:
    return os.environ.get("SUBMISSION_MODEL", str(_DEFAULT_RL_ADAPTER))


# ── Helpers (duplicated from inference/infer.py) ──────────────────────────────


def load_done_ids(out_path: Path) -> set[str]:
    if not out_path.exists():
        return set()
    with open(out_path, newline="") as f:
        return {str(row["id"]) for row in csv.DictReader(f)}


def append_rows(rows: list[dict], out_path: Path, write_header: bool) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "w" if write_header else "a"
    with open(out_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


# ── Public API ────────────────────────────────────────────────────────────────


def run_inference(
    *,
    data_path: Path | str = PRIVATE_DATA,
    output_path: Path | str = RESULTS_DIR / "submission.csv",
    model: str | None = None,
    gpu: bool = True,
    reset: bool = True,
    n_samples: int = DEFAULT_N_SAMPLES,
    tp: int = 1,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_seq_len: int = DEFAULT_MAX_SEQ_LEN,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    gpu_util: float | None = None,
    quantize: bool = False,
    chunk_size: int = 10,
    limit: int | None = None,
    num_shards: int = 1,
    shard_id: int = 0,
    use_router: bool = True,
    router_secondary_llm: bool = False,
    router_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    router_device: str = "cpu",
) -> Path:
    """
    Full private-set pipeline: load GRPO adapter → generate → self-consistency vote → CSV.

    Defaults match **Experiment 3c** (``checkpoint-best-reward``, router, N=8, no INT8).
    """
    if model is None:
        model = _default_model_path()
    if num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not (0 <= shard_id < num_shards):
        raise ValueError("shard_id must satisfy 0 <= shard_id < num_shards")
    if num_shards > 1 and tp != 1:
        raise ValueError(
            "With num_shards > 1, use tp=1 (one GPU per shard). "
            "Use inference/infer_parallel.py for multi-GPU throughput.",
        )

    model_path = resolve_checkpoint_latest_path(Path(model))
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
    _parallel_worker = os.environ.get("INFER_PARALLEL_WORKER", "").lower() in (
        "1",
        "true",
        "yes",
    )

    effective_gpu_util = (
        gpu_util
        if gpu_util is not None
        else (DEFAULT_QUANTIZE_GPU_UTIL if quantize else DEFAULT_GPU_UTIL)
    )
    out_path = Path(output_path)
    n_gen = n_samples

    data = load_jsonl(Path(data_path))
    if limit:
        data = data[:limit]
    if num_shards > 1:
        data = [
            item for idx, item in enumerate(data) if idx % num_shards == shard_id
        ]

    done_ids = set() if reset else load_done_ids(out_path)
    todo = [item for item in data if str(item["id"]) not in done_ids]

    n_mcq = sum(bool(d.get("options")) for d in data)
    n_free = len(data) - n_mcq
    print(f"\n{'=' * 55}")
    print(f"  Data   : {data_path}")
    print(f"  Output : {out_path}")
    print(f"  Model  : {model_label}")
    device_str = (
        f"GPU (tp={tp}, gpu_util={effective_gpu_util:.0%}"
        + (", INT8 quantized" if quantize else "")
        + ")"
        if gpu
        else "CPU"
    )
    print(f"  Device : {device_str}")
    if num_shards > 1:
        print(
            f"  Shard    : {shard_id + 1}/{num_shards} "
            f"(rows where dataset_index % {num_shards} == {shard_id})",
        )
    print(
        f"  Questions: {len(data)} total  ({n_mcq} MCQ, {n_free} free-form)",
    )
    print(
        f"  Samples  : {n_gen} per question  →  {len(data) * n_gen} total generations",
    )
    print(f"{'=' * 55}\n")

    if done_ids:
        print(f"[resume] {len(done_ids)} questions already done, {len(todo)} remaining")
    if not todo:
        print(
            "Nothing to do — all questions already processed. "
            "Use reset=True to rerun.",
        )
        return out_path

    tok_source = adapter_dir or vllm_model
    print(f"[1/3] Loading tokenizer ({tok_source})...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tok_source,
            trust_remote_code=True,
        )
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(
            vllm_model,
            trust_remote_code=True,
        )
    tokenizer.pad_token = tokenizer.eos_token
    print("      Tokenizer ready.")

    router = None
    if use_router:
        from inference.router import (
            LLMSecondaryRouter,
            RuleBasedRouter,
            build_routed_prompts,
        )

        if router_secondary_llm:
            router = LLMSecondaryRouter(
                model=router_model,
                device=("cpu" if router_device == "cpu" else "auto"),
            )
            print(
                f"[router] enabled (topic via LLM with taxonomy fallback: "
                f"{router_model}, device={router_device})",
            )
        else:
            router = RuleBasedRouter(enable_topic_refinements=True)
            print("[router] enabled (topic_taxonomy + optional refinements)")

    print(
        f"[2/3] Loading model weights into {'GPU' if gpu else 'CPU'} memory...\n"
        f"      (Note: first run may take several extra minutes for compilation)",
    )
    llm_kwargs: dict = dict(
        model=vllm_model,
        trust_remote_code=True,
        max_model_len=max_seq_len,
        max_num_seqs=DEFAULT_MAX_NUM_SEQS,
        max_num_batched_tokens=DEFAULT_MAX_NUM_BATCHED_TOKENS,
    )
    if adapter_dir:
        llm_kwargs["enable_lora"] = True
        llm_kwargs["max_lora_rank"] = adapter_lora_rank(adapter_dir)
        print(f"      LoRA adapter: {adapter_dir} (rank={llm_kwargs['max_lora_rank']})")
    if gpu:
        llm_kwargs["tensor_parallel_size"] = tp
        llm_kwargs["gpu_memory_utilization"] = effective_gpu_util
        llm_kwargs["enable_prefix_caching"] = not quantize
        if quantize:
            llm_kwargs["quantization"] = "bitsandbytes"
            llm_kwargs["load_format"] = "bitsandbytes"
    else:
        llm_kwargs["device"] = "cpu"

    if is_deepseek_r1_vllm_special_case(tokenizer, vllm_model):
        llm_kwargs["enforce_eager"] = True

    llm = LLM(**llm_kwargs)
    print("      Model ready.")
    if is_deepseek_r1_vllm_special_case(tokenizer, vllm_model):
        print(
            "      [vllm] DeepSeek-R1: string prompts + enforce_eager=True "
            "(vLLM CUDA-graph / compile glitches on this family).",
        )

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        min_p=DEFAULT_MIN_P,
        presence_penalty=DEFAULT_PRESENCE_PENALTY,
        repetition_penalty=DEFAULT_REPETITION_PENALTY,
    )

    print(
        f"[3/3] Generating responses and voting  "
        f"(chunk_size={chunk_size}, writing every {chunk_size} questions)...",
    )
    need_header = reset or not out_path.exists()
    total_written = len(done_ids)

    pbar = tqdm(
        total=len(todo),
        desc="Questions",
        unit="q",
        initial=0,
        disable=_parallel_worker,
    )
    for chunk_start in range(0, len(todo), chunk_size):
        chunk = todo[chunk_start : chunk_start + chunk_size]

        use_str = is_deepseek_r1_vllm_special_case(tokenizer, vllm_model)
        chunk_inputs: list[dict] = []
        if router is None:
            for item in chunk:
                system, user = build_prompt(
                    item["question"],
                    item.get("options"),
                    SYSTEM_MATH,
                    SYSTEM_MCQ,
                    MULTI_ANS_NOTE,
                )
                text = apply_chat_template_safe(
                    tokenizer,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                if use_str:
                    chunk_inputs.append({"prompt": text})
                else:
                    chunk_inputs.append(
                        {
                            "prompt_token_ids": tokenizer.encode(
                                text,
                                add_special_tokens=False,
                            ),
                        },
                    )
        else:
            routed = build_routed_prompts(router, chunk)
            for system, user in routed:
                text = apply_chat_template_safe(
                    tokenizer,
                    [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                if use_str:
                    chunk_inputs.append({"prompt": text})
                else:
                    chunk_inputs.append(
                        {
                            "prompt_token_ids": tokenizer.encode(
                                text,
                                add_special_tokens=False,
                            ),
                        },
                    )

        all_prompts = [dict(inp) for inp in chunk_inputs for _ in range(n_gen)]
        outputs = llm.generate(
            all_prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
        flat_resps = [out.outputs[0].text.strip() for out in outputs]

        rows = []
        for i, item in enumerate(chunk):
            group = flat_resps[i * n_gen : (i + 1) * n_gen]
            n_slots = count_ans_slots(item["question"])
            is_mcq = bool(item.get("options"))
            winner = majority_vote(group, n_slots, is_mcq)
            rows.append({"id": item["id"], "response": winner})

        append_rows(rows, out_path, write_header=need_header)
        need_header = False
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
    return out_path


if __name__ == "__main__":
    run_inference()
