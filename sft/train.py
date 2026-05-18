#!/usr/bin/env python3
"""
LoRA / QLoRA supervised fine-tuning on distilled chat JSONL (trl.SFTTrainer).

Features:
  - Resume from checkpoint (--resume / --resume-from)
  - metrics_history.csv + statistics.pdf (loss + MCQ/FRQ/overall token accuracy on eval shard)
  - Optional periodic reload of train JSONL from disk (--reload-data-every / --reload-data-each-epoch)
  - Multi-GPU: prefer ``torchrun --nproc_per_node=N ...`` (DDP, full replica per GPU). A single
    process with ``device_map="auto"`` splits layers across GPUs and often shows poor SM balance.
    Use ``--single-gpu`` to pin the whole model to the first visible GPU when one GPU is enough.

HF cache: mirrors inference scripts — set HF_HOME before heavy imports.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import warnings
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import HF_CACHE_DIR, HF_TOKEN, HF_XET_CACHE  # noqa: E402

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
if HF_TOKEN:
    os.environ.setdefault("HF_TOKEN", HF_TOKEN)
if HF_XET_CACHE:
    os.environ.setdefault("HF_XET_CACHE", HF_XET_CACHE)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from constants import DEFAULT_MODEL  # noqa: E402
from config import CHECKPOINTS_DIR, DISTILL_DIR, ensure_storage_dirs  # noqa: E402
from sft.callbacks import (  # noqa: E402
    ReloadTrainDatasetCallback,
    ReloadTrainDatasetEachEpochCallback,
    StatisticsPlotCallback,
    install_labeled_progress_bar,
    load_eval_state,
    save_eval_state,
)


EVAL_STATE_NAME = "sft_eval_state.json"
DEFAULT_SFT_DATA = DISTILL_DIR / "sft_data.jsonl"
DEFAULT_OUT = CHECKPOINTS_DIR / "sft"


def messages_hash(messages: object) -> str:
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def example_hash(example: dict) -> str:
    return messages_hash(example["messages"])


def load_jsonl_dataset(path: Path) -> Dataset:
    return load_dataset("json", data_files=str(path), split="train")


def initial_train_eval_split(ds: Dataset, seed: int, eval_n: int) -> tuple[Dataset, Dataset, list[str]]:
    n = len(ds)
    if n < 2:
        raise ValueError(f"Dataset too small ({n} rows); need at least 2 examples.")
    ne = min(eval_n, n - 1)
    indices = list(range(n))
    rng = __import__("random").Random(seed)
    rng.shuffle(indices)
    eval_indices = indices[:ne]
    train_indices = indices[ne:]
    eval_ds = ds.select(eval_indices)
    train_ds = ds.select(train_indices)
    hashes = [example_hash(ds[i]) for i in eval_indices]
    return train_ds, eval_ds, hashes


def rebuild_eval_from_hashes(path: Path, hashes: list[str]) -> Dataset:
    ds = load_jsonl_dataset(path)
    hset = set(hashes)

    def filt(ex: dict) -> bool:
        return example_hash(ex) in hset

    return ds.filter(filt)


def rebuild_train_excluding_hashes(path: Path, exclude_hashes: set[str], shuffle_seed: int) -> Dataset:
    ds = load_jsonl_dataset(path)

    def filt(ex: dict) -> bool:
        return example_hash(ex) not in exclude_hashes

    return ds.filter(filt).shuffle(seed=shuffle_seed)


def preprocess_logits_for_metrics(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    del labels
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)


def compute_metrics(eval_pred):
    preds = eval_pred.predictions
    labels = eval_pred.label_ids
    mask = labels != -100
    correct = np.sum((preds == labels) * mask)
    total = np.sum(mask)
    return {"accuracy": float(correct) / max(float(total), 1.0)}


def estimated_optimizer_steps(
    n_train: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    max_steps: int,
) -> int:
    """Rough upper bound on global optimizer steps (single-process estimate)."""
    if max_steps > 0:
        return max_steps
    micro_batch = max(1, batch_size * grad_accum)
    steps_per_epoch = max(1, math.ceil(n_train / micro_batch))
    return max(1, math.ceil(steps_per_epoch * epochs))


def silence_known_third_party_warnings() -> None:
    """bitsandbytes emits PyTorch FutureWarnings we cannot fix in this repo."""
    warnings.filterwarnings(
        "ignore",
        message=r".*_check_is_size.*",
        category=FutureWarning,
    )


def parse_args():
    p = argparse.ArgumentParser(description="LoRA SFT on distilled chat JSONL")
    p.add_argument("--data", type=str, default=str(DEFAULT_SFT_DATA), help="Merged SFT JSONL path")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_OUT), help="Checkpoints + plots")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--eval-num-examples", type=int, default=256,
                   help="Examples held out for stratified eval plots (stable across reloads via hashes)")
    p.add_argument("--plot-every", type=int, default=500,
                   help="Steps between metrics CSV append + statistics.pdf refresh")
    p.add_argument("--save-every", type=int, default=500, help="Checkpoint save interval (steps)")
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument(
        "--progress-desc",
        type=str,
        default=None,
        help="Label on the HF/tqdm training bar (default: SFT · <output-dir name> · optimizer steps)",
    )
    p.add_argument("--skip-acc-eval", action="store_true",
                   help="Loss-only statistics.pdf (no trainer.evaluate — faster)")

    p.add_argument("--reload-data-every", type=int, default=0,
                   help="Reload train JSONL from disk every N steps (0 = off)")
    p.add_argument("--reload-data-each-epoch", action="store_true",
                   help="Reload train JSONL at each epoch start (fallback)")

    p.add_argument("--resume", action="store_true",
                   help="Resume from latest checkpoint in output-dir")
    p.add_argument("--resume-from", type=str, default=None,
                   help="Explicit checkpoint folder (overrides --resume)")

    p.add_argument("--no-qlora", dest="qlora", action="store_false",
                   help="bf16/fp16 LoRA without 4-bit quantization (default is QLoRA / 4-bit)")
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=None,
                   help="Default: 2 * lora-r")
    p.add_argument("--lora-dropout", type=float, default=0.05)

    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max-steps", type=int, default=-1,
                   help="If > 0, overrides epoch-based training length")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup in optimizer steps (default: derive from --warmup-ratio × estimated steps)",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Used only when --warmup-steps is unset: fraction of estimated training steps",
    )
    p.add_argument("--max-seq-length", type=int, default=4096,
                   help="Max sequence length passed to TRL as max_length (default: 4096)")
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help="Load the full model on cuda:0 (first visible GPU) instead of device_map=\"auto\" "
        "across all visible GPUs — avoids idle/high-VRAM imbalance from pipeline-style sharding.",
    )
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="HF DataLoader worker processes (-1 = 4 under torchrun/LOCAL_RANK set, else 0). "
        "Higher values prefetch batches on CPU so GPUs wait less (helps uneven DDP util).",
    )

    p.set_defaults(qlora=True)
    return p.parse_args()


def main():
    args = parse_args()
    silence_known_third_party_warnings()
    ensure_storage_dirs()

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    data_path = Path(args.data).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not data_path.is_file():
        raise FileNotFoundError(f"Training data not found: {data_path}")

    eval_state_path = output_dir / EVAL_STATE_NAME

    resume_ckpt: str | None = None
    if args.resume_from:
        resume_ckpt = str(Path(args.resume_from).resolve())
    elif args.resume:
        resume_ckpt = get_last_checkpoint(str(output_dir))
        if not resume_ckpt:
            raise ValueError(f"No checkpoint-* folder found under {output_dir}; cannot --resume.")

    exclude_hashes: set[str]

    if resume_ckpt:
        state = load_eval_state(eval_state_path)
        if not state:
            raise ValueError(
                f"Resume requires {eval_state_path} (eval split hashes). "
                "Train once without --resume to create it."
            )
        exclude_hashes = set(state["eval_hashes"])
        eval_ds = rebuild_eval_from_hashes(data_path, state["eval_hashes"])
        train_ds = rebuild_train_excluding_hashes(data_path, exclude_hashes, args.seed)
        print(f"[resume] checkpoint={resume_ckpt}")
        print(f"[resume] eval examples recovered: {len(eval_ds)} (expected {len(exclude_hashes)})")
    else:
        ds = load_jsonl_dataset(data_path)
        train_ds, eval_ds, eval_hashes_list = initial_train_eval_split(
            ds, args.seed, args.eval_num_examples,
        )
        exclude_hashes = set(eval_hashes_list)
        if local_rank <= 0:
            save_eval_state(eval_state_path, args.seed, len(eval_hashes_list), eval_hashes_list)
            print(f"[init] train={len(train_ds)} eval={len(eval_ds)} (hashes saved → {eval_state_path})")
        else:
            print(f"[init] train={len(train_ds)} eval={len(eval_ds)} (DDP rank {local_rank}; eval hashes from rank 0)")

    if len(eval_ds) == 0:
        raise RuntimeError("Eval dataset is empty — cannot compute plot metrics.")

    # Stratified eval subsets (masked token accuracy on teacher targets)
    if "is_mcq" in eval_ds.column_names:
        eval_mcq = eval_ds.filter(lambda x: bool(x["is_mcq"]))
        eval_frq = eval_ds.filter(lambda x: not bool(x["is_mcq"]))
    else:
        eval_mcq = eval_ds.select([])
        eval_frq = eval_ds.select([])
        print("[warn] JSONL rows lack 'is_mcq'; MCQ/FRQ panels will be empty (merge.py adds this field).")

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    est_steps = estimated_optimizer_steps(
        len(train_ds),
        args.batch_size,
        args.grad_accum,
        args.epochs,
        args.max_steps,
    )
    if args.warmup_steps is not None:
        warmup_steps = max(0, args.warmup_steps)
    else:
        warmup_steps = max(0, int(round(est_steps * args.warmup_ratio)))
    print(f"[train] estimated optimizer steps ≈ {est_steps}; warmup_steps={warmup_steps}")

    dataloader_workers = int(args.dataloader_workers)
    if dataloader_workers < 0:
        dataloader_workers = 4 if local_rank >= 0 else 0
    print(f"[train] dataloader_num_workers={dataloader_workers}")

    if local_rank < 0 and torch.cuda.device_count() > 1 and not args.single_gpu:
        print(
            "[train] Multiple GPUs visible and device_map=\"auto\" will shard layers across them "
            "(pipeline-style). That often shows ~one GPU at high SM util and others idle or holding weights only. "
            "For balanced utilization use DDP: CUDA_VISIBLE_DEVICES=0,3 torchrun --standalone --nproc_per_node=2 "
            "sft/train.py ...   or pin one GPU: CUDA_VISIBLE_DEVICES=3 python sft/train.py --single-gpu ..."
        )

    device_map: dict[str, int] | str
    if local_rank >= 0:
        device_map = {"": local_rank}
    elif args.single_gpu:
        device_map = {"": 0}
    else:
        device_map = "auto"

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "right"

    def formatting_func(example: dict) -> str:
        return tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )

    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        optim = "paged_adamw_8bit"
    else:
        dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        optim = "adamw_torch"

    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(model, peft_config)

    use_cuda = torch.cuda.is_available()
    sft_kw: dict = dict(
        output_dir=str(output_dir),
        seed=args.seed,
        max_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=float(warmup_steps),
        logging_steps=args.logging_steps,
        save_steps=args.save_every,
        save_strategy="steps",
        eval_strategy="no",
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=True,
        optim=optim,
        remove_unused_columns=False,
        report_to="none",
        save_total_limit=5,
        load_best_model_at_end=False,
        metric_for_best_model=None,
        dataloader_num_workers=dataloader_workers,
        dataloader_persistent_workers=dataloader_workers > 0,
        dataloader_pin_memory=use_cuda,
        # LoRA forward uses all trainable params; True adds a full backward-graph scan each step (DDP warning).
        ddp_find_unused_parameters=False,
    )
    if dataloader_workers > 0:
        sft_kw["dataloader_prefetch_factor"] = 4
    training_args = SFTConfig(**sft_kw)

    eval_builders = {
        "mcq": lambda: eval_mcq,
        "frq": lambda: eval_frq,
        "overall": lambda: eval_ds,
    }

    callbacks = [
        StatisticsPlotCallback(
            output_dir=output_dir,
            plot_every=args.plot_every,
            skip_acc_eval=args.skip_acc_eval,
            eval_builders=eval_builders,
        ),
    ]

    base_seed = args.seed

    def rebuild_train(step: int):
        return rebuild_train_excluding_hashes(data_path, exclude_hashes, base_seed + step)

    if args.reload_data_every > 0:
        callbacks.append(
            ReloadTrainDatasetCallback(
                reload_every=args.reload_data_every,
                rebuild_train_fn=rebuild_train,
            )
        )
    elif args.reload_data_each_epoch:
        callbacks.append(
            ReloadTrainDatasetEachEpochCallback(rebuild_train_fn=rebuild_train),
        )

    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        formatting_func=formatting_func,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        callbacks=callbacks,
    )
    try:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_kw)
    except TypeError:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_kw)

    # HF CallbackHandler passes model/optimizer into callbacks but not ``trainer`` (see
    # transformers CallbackHandler.call_event); our callbacks need the Trainer instance.
    for cb in callbacks:
        cb.trainer = trainer

    progress_desc = args.progress_desc or f"SFT · {output_dir.name} · optimizer steps"
    install_labeled_progress_bar(trainer, progress_desc)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Done. Adapter + tokenizer saved under {output_dir}")


if __name__ == "__main__":
    main()
