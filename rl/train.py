#!/usr/bin/env python3
"""
GRPO reinforcement learning on ``public.jsonl`` using TRL's ``GRPOTrainer``.

Starts from an SFT LoRA directory (or a full HF model id). Prompts match
``inference/infer.py`` via ``build_prompt`` + ``apply_chat_template_safe``.
"""

from __future__ import annotations

import argparse
import json
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
from datasets import Dataset  # noqa: E402
from peft import PeftModel, prepare_model_for_kbit_training  # noqa: E402
from transformers import (  # noqa: E402
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from constants import DEFAULT_MODEL, MULTI_ANS_NOTE, SYSTEM_MATH, SYSTEM_MCQ  # noqa: E402
from config import CHECKPOINTS_DIR, PUBLIC_DATA, ensure_storage_dirs  # noqa: E402
from inference.utils import apply_chat_template_safe, build_prompt, load_jsonl  # noqa: E402
from rl.rewards import JudgerOutcomeReward  # noqa: E402


DEFAULT_RL_OUT = CHECKPOINTS_DIR / "rl"


def silence_known_third_party_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*_check_is_size.*",
        category=FutureWarning,
    )


def resolve_base_and_adapter(model_path: Path) -> tuple[str, str | None]:
    """
    If *model_path* contains ``adapter_config.json``, return
    (base_model_name_or_path, adapter_dir). Otherwise return (path_str, None).
    """
    adapter_cfg = model_path / "adapter_config.json"
    if adapter_cfg.is_file():
        meta = json.loads(adapter_cfg.read_text(encoding="utf-8"))
        base = meta.get("base_model_name_or_path")
        if not base:
            raise ValueError(f"adapter_config.json missing base_model_name_or_path: {adapter_cfg}")
        return str(base), str(model_path.resolve())
    return str(model_path.resolve()), None


def build_grpo_dataset(tokenizer, jsonl_path: Path) -> Dataset:
    """
    Rows: ``prompt`` (string), ``is_mcq``, ``gold`` (raw answer field), ``id``.
    """
    records: list[dict] = []
    for item in load_jsonl(jsonl_path):
        system, user = build_prompt(
            item["question"],
            item.get("options"),
            SYSTEM_MATH,
            SYSTEM_MCQ,
            MULTI_ANS_NOTE,
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        prompt = apply_chat_template_safe(tokenizer, messages)
        records.append(
            {
                "prompt": prompt,
                "is_mcq": bool(item.get("options")),
                "gold": item["answer"],
                "id": item.get("id"),
            }
        )
    return Dataset.from_list(records)


class BestRewardCallback(TrainerCallback):
    """Save ``checkpoint-best-reward/`` when the logged ``reward`` improves (rank 0 only)."""

    def __init__(self, metric_key: str = "reward"):
        self.metric_key = metric_key
        self.best: float | None = None
        self.trainer: GRPOTrainer | None = None

    def on_log(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        logs: dict[str, float] | None = None,
        **kwargs,
    ):
        del args, state, control, kwargs
        if logs is None or self.trainer is None:
            return
        if self.metric_key not in logs:
            return
        if hasattr(self.trainer, "is_world_process_zero") and not self.trainer.is_world_process_zero():
            return
        val = float(logs[self.metric_key])
        if self.best is None or val > self.best:
            self.best = val
            out = Path(self.trainer.args.output_dir) / "checkpoint-best-reward"
            out.mkdir(parents=True, exist_ok=True)
            self.trainer.save_model(str(out))
            proc = getattr(self.trainer, "processing_class", None)
            if proc is not None:
                proc.save_pretrained(str(out))
            print(f"[best-reward] {self.metric_key}={val:.6f} → saved {out}", flush=True)


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training on public.jsonl with Judger rewards")
    p.add_argument("--model", type=str, default=DEFAULT_MODEL, help="HF model id or SFT adapter directory")
    p.add_argument("--data", type=str, default=str(PUBLIC_DATA), help="JSONL with question, options, answer, id")
    p.add_argument("--output", type=str, default=str(DEFAULT_RL_OUT), help="Checkpoints + best-reward snapshot")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--num-generations", type=int, default=4, help="K completions per prompt (GRPO group size)")
    p.add_argument("--max-completion-length", type=int, default=2048, help="Max new tokens per rollout")
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--format-bonus", type=float, default=0.02, help="Extra reward if output contains \\\\boxed")

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--logging-steps", type=int, default=10)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--save-total-limit", type=int, default=3)
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 = off, TRL default)")
    p.add_argument("--scale-rewards", type=str, default="group", help="e.g. group, batch, false — see GRPOConfig")
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="DataLoader workers (-1: 4 if LOCAL_RANK set else 0)",
    )

    p.add_argument("--no-qlora", dest="qlora", action="store_false", help="bf16/fp16 weights instead of 4-bit")
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help="Pin the full model on cuda:0 instead of device_map=\"auto\" with multiple GPUs visible",
    )

    p.set_defaults(qlora=True)
    return p.parse_args()


def _parse_scale_rewards(s: str) -> str | bool:
    sl = s.strip().lower()
    if sl in ("false", "none", "no", "0"):
        return False
    return s


def main():
    args = parse_args()
    silence_known_third_party_warnings()
    ensure_storage_dirs()

    data_path = Path(args.data).resolve()
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data JSONL not found: {data_path}")

    model_path = Path(args.model)
    base_id, adapter_dir = resolve_base_and_adapter(model_path)

    local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
    if local_rank >= 0:
        torch.cuda.set_device(local_rank)

    if local_rank >= 0:
        device_map: dict[str, int] | str = {"": local_rank}
    elif args.single_gpu:
        device_map = {"": 0}
    else:
        device_map = "auto"

    tok_source = adapter_dir if adapter_dir else str(model_path.resolve())
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_ds = build_grpo_dataset(tokenizer, data_path)
    print(f"[data] {len(train_ds)} prompts from {data_path}")

    use_cuda = torch.cuda.is_available()
    if args.qlora:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        optim = "paged_adamw_8bit"
    else:
        dtype = torch.bfloat16 if use_cuda else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            base_id,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        optim = "adamw_torch"

    if adapter_dir:
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)

    scale_rw = _parse_scale_rewards(args.scale_rewards)

    dl_workers = int(args.dataloader_workers)
    if dl_workers < 0:
        dl_workers = 4 if local_rank >= 0 else 0

    grpo_kw: dict = dict(
        output_dir=str(output_dir),
        seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        optim=optim,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        scale_rewards=scale_rw,
        remove_unused_columns=False,
        dataloader_num_workers=dl_workers,
        dataloader_pin_memory=use_cuda,
        dataloader_persistent_workers=dl_workers > 0,
    )
    if dl_workers > 0:
        grpo_kw["dataloader_prefetch_factor"] = 4

    training_args = GRPOConfig(**grpo_kw)

    reward_fn = JudgerOutcomeReward(format_bonus=args.format_bonus)
    best_cb = BestRewardCallback(metric_key="reward")

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=[best_cb],
    )
    best_cb.trainer = trainer

    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Done. Final weights under {output_dir}; best reward snapshot: {output_dir / 'checkpoint-best-reward'}")


if __name__ == "__main__":
    main()
