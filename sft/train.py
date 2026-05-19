#!/usr/bin/env python3
"""
LoRA / QLoRA supervised fine-tuning on distilled chat JSONL (trl.SFTTrainer).

Features:
  - Resume from checkpoint (--resume / --resume-from)
  - Training curves: ``training_loss_history.csv`` (loss + optional TRL ``mean_token_accuracy`` via
    ``--loss-csv-every``) + ``metrics_history.csv`` / ``statistics.pdf`` every ``--plot-every`` steps
    (no in-loop ``evaluate()``; ``mean_token_accuracy`` is masked next-token match on labels, not
    public-set Judge scores — use ``infer.py`` + ``evaluate.py`` for those).
  - Multi-GPU: prefer ``torchrun --nproc_per_node=N ...`` (DDP, full replica per GPU). A single
    process with ``device_map="auto"`` splits layers across GPUs and often shows poor SM balance.
    Use ``--single-gpu`` to pin the whole model to the first visible GPU when one GPU is enough.
  - Output: ``checkpoint-{step}/`` Trainer checkpoints (adapter + tokenizer + resume state); file
    ``checkpoint-latest`` points at the newest step (pointer text by default; ``--checkpoint-latest-symlink``
    for a symlink on local disks). Pass ``…/checkpoint-latest`` to infer / RL — it resolves to the real folder.

HF cache: mirrors inference scripts — set HF_HOME before heavy imports.
"""

from __future__ import annotations

import argparse
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

import torch  # noqa: E402
from datasets import Dataset, load_dataset  # noqa: E402
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

from constants import DEFAULT_MAX_SEQ_LEN, DEFAULT_MODEL  # noqa: E402
from config import CHECKPOINTS_DIR, DISTILL_DIR, ensure_storage_dirs  # noqa: E402
from sft.callbacks import (  # noqa: E402
    CHECKPOINT_LATEST,
    LatestCheckpointSymlinkCallback,
    TrainingLossPlotCallback,
    TrainLossHistoryCallback,
    install_checkpoint_chunk_progress_bar,
    resolve_checkpoint_latest_path,
)


DEFAULT_SFT_DATA = DISTILL_DIR / "sft_data.jsonl"
DEFAULT_OUT = CHECKPOINTS_DIR / "sft"


def load_jsonl_dataset(path: Path) -> Dataset:
    return load_dataset("json", data_files=str(path), split="train")


def estimated_optimizer_steps(
    n_train: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    max_steps: int,
    *,
    world_size: int = 1,
) -> int:
    """Rough optimizer-step count for the scheduled run (matches HF tqdm total under DDP)."""
    if max_steps > 0:
        return max_steps
    ws = max(1, int(world_size))
    # Per global optimizer step, all ranks each process (batch_size * grad_accum) examples from disjoint shards.
    examples_per_global_step = max(1, batch_size * grad_accum * ws)
    steps_per_epoch = max(1, math.ceil(n_train / examples_per_global_step))
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

    p.add_argument(
        "--plot-every",
        type=int,
        default=250,
        help="Steps between metrics_history.csv + statistics.pdf refresh (training loss + TRL mean_token_accuracy)",
    )
    p.add_argument("--save-every", type=int, default=250, help="Checkpoint save interval (steps)")
    p.add_argument(
        "--checkpoint-latest-symlink",
        action="store_true",
        help="Write checkpoint-latest as a symlink (default: one-line pointer file — reliable on NFS/deepfreeze)",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=10,
        help="Trainer log interval; should divide --plot-every so PDF refresh runs on a logged step. "
        "Also align --loss-csv-every (e.g. divide --plot-every) so training_loss_history.csv has a row "
        "at each plot boundary (statistics.pdf panel 3).",
    )
    p.add_argument(
        "--loss-csv-every",
        type=int,
        default=10,
        help="Append training_loss_history.csv every N global steps when Trainer logs loss "
        "(includes mean_token_accuracy when TRL logs it; no eval). "
        "Use a small N for a smooth curve; frequency also follows --logging-steps. Set to 0 to disable.",
    )

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

    p.add_argument("--epochs", type=float, default=5.0)
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
    p.add_argument(
        "--max-seq-length",
        type=int,
        default=DEFAULT_MAX_SEQ_LEN,
        help=f"Max sequence length passed to TRL as max_length (default: constants.DEFAULT_MAX_SEQ_LEN = {DEFAULT_MAX_SEQ_LEN})",
    )
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

    resume_ckpt: str | None = None
    if args.resume_from:
        resume_ckpt = str(Path(args.resume_from).resolve())
    elif args.resume:
        resume_ckpt = get_last_checkpoint(str(output_dir))
        if not resume_ckpt:
            raise ValueError(f"No checkpoint-* folder found under {output_dir}; cannot --resume.")

    train_ds = load_jsonl_dataset(data_path)
    if len(train_ds) == 0:
        raise ValueError(f"Training data is empty: {data_path}")
    train_ds = train_ds.shuffle(seed=args.seed)

    if resume_ckpt:
        print(f"[resume] checkpoint={resume_ckpt}")
    print(f"[train] examples={len(train_ds)} (full JSONL, seed={args.seed})")

    lora_alpha = args.lora_alpha if args.lora_alpha is not None else 2 * args.lora_r

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if local_rank < 0:
        world_size = 1

    est_steps = estimated_optimizer_steps(
        len(train_ds),
        args.batch_size,
        args.grad_accum,
        args.epochs,
        args.max_steps,
        world_size=world_size,
    )
    if args.warmup_steps is not None:
        warmup_steps = max(0, args.warmup_steps)
    else:
        warmup_steps = max(0, int(round(est_steps * args.warmup_ratio)))
    ex_per_step = max(1, args.batch_size * args.grad_accum * max(1, world_size))
    print(
        f"[train] ≈{ex_per_step} examples per global optimizer step "
        f"(world_size={max(1, world_size)} × batch × grad_accum) | "
        f"estimated optimizer steps ≈ {est_steps}; warmup_steps={warmup_steps}",
    )

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

    callbacks = [
        TrainingLossPlotCallback(
            output_dir=output_dir,
            plot_every=args.plot_every,
        ),
        LatestCheckpointSymlinkCallback(use_symlink=args.checkpoint_latest_symlink),
    ]
    if int(args.loss_csv_every) > 0:
        callbacks.insert(
            0,
            TrainLossHistoryCallback(output_dir=output_dir, every=int(args.loss_csv_every)),
        )

    trainer_kw = dict(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        formatting_func=formatting_func,
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

    trainer._sft_formatting_func = formatting_func  # noqa: SLF001

    install_checkpoint_chunk_progress_bar(trainer)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    link_path = output_dir / CHECKPOINT_LATEST
    resolved = resolve_checkpoint_latest_path(link_path)
    if resolved.is_dir():
        print(f"Done. Primary checkpoint for inference / RL: {resolved.resolve()}")
    else:
        fallback = get_last_checkpoint(str(output_dir))
        if fallback:
            print(
                f"Done. Latest folder: {fallback}. "
                f"Expected {link_path.name} missing or unresolved — use this folder or re-save.",
            )
        else:
            print(f"Done. Warning: no checkpoint-* saved under {output_dir}")


if __name__ == "__main__":
    main()
