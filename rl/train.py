#!/usr/bin/env python3
"""
GRPO reinforcement learning on ``public.jsonl`` using TRL's ``GRPOTrainer``.

Starts from an SFT LoRA directory (or a full HF model id). Prompts match
``inference/infer.py`` via ``build_prompt`` + ``apply_chat_template_safe``.

Progress + checkpoint cadence mirror ``sft/train.py``: segmented tqdm (aligned to
``--save-every``), ``checkpoint-latest`` (pointer file or symlink), resume, training
curves (``training_loss_history.csv`` every ``--loss-csv-every`` steps;
``metrics_history.csv`` + ``statistics.pdf`` every ``--plot-every`` steps), and
matching LR / batch / epoch / warmup / save defaults unless overridden.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r"TRL currently supports vLLM versions.*",
    category=UserWarning,
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from config import HF_CACHE_DIR, HF_TOKEN, HF_XET_CACHE  # noqa: E402

os.environ["HF_HOME"] = str(HF_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
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
from tqdm import tqdm  # noqa: E402
from transformers.trainer_utils import get_last_checkpoint  # noqa: E402
from trl import GRPOConfig, GRPOTrainer  # noqa: E402

from constants import (  # noqa: E402
    DEFAULT_MAX_SEQ_LEN,
    DEFAULT_MODEL,
    DEFAULT_RL_MAX_COMPLETION_LENGTH,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    MULTI_ANS_NOTE,
    SYSTEM_MATH,
    SYSTEM_MCQ,
)
from config import CHECKPOINTS_DIR, PUBLIC_DATA, ensure_storage_dirs  # noqa: E402
from inference.utils import apply_chat_template_safe, build_prompt, load_jsonl  # noqa: E402
from rl.callbacks import GrpoTrainingPlotCallback, GrpoTrainHistoryCallback  # noqa: E402
from rl.rewards import JudgerOutcomeReward, normalize_gold_answer  # noqa: E402
from sft.progress_callbacks import (
    LatestCheckpointSymlinkCallback,
    install_checkpoint_chunk_progress_bar,
    resolve_checkpoint_latest_path,
)


DEFAULT_RL_OUT = CHECKPOINTS_DIR / "rl"


def silence_known_third_party_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r".*_check_is_size.*",
        category=FutureWarning,
    )


def resolve_grpo_steps_per_generation(
    *,
    batch_size: int,
    world_size: int,
    steps_per_generation: int,
    num_generations: int,
) -> int:
    """Bump steps_per_generation so TRL's generation_batch_size divides num_generations."""
    micro = batch_size * max(1, world_size)
    unit = num_generations // math.gcd(micro, num_generations)
    resolved = max(steps_per_generation, unit)
    if resolved % unit != 0:
        resolved = math.ceil(resolved / unit) * unit
    return resolved


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
    Rows: ``prompt`` (string), ``is_mcq``, ``gold`` (list of answer strings), ``id``.
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
                "gold": normalize_gold_answer(item["answer"]),
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
            tqdm.write(f"[best-reward] {self.metric_key}={val:.6f} → saved {out}")


def estimated_optimizer_steps(
    n_train: int,
    batch_size: int,
    grad_accum: int,
    epochs: float,
    max_steps: int,
    *,
    world_size: int = 1,
) -> int:
    """Matches ``sft/train.py`` — HF tqdm total under DDP for epoch schedules."""
    if max_steps > 0:
        return max_steps
    ws = max(1, int(world_size))
    examples_per_global_step = max(1, batch_size * grad_accum * ws)
    steps_per_epoch = max(1, math.ceil(n_train / examples_per_global_step))
    return max(1, math.ceil(steps_per_epoch * epochs))


def parse_args():
    p = argparse.ArgumentParser(description="GRPO training on public.jsonl with Judger rewards")
    p.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HF model id or local adapter dir (use .../sft/checkpoint-latest after SFT in this repo)",
    )
    p.add_argument("--data", type=str, default=str(PUBLIC_DATA), help="JSONL with question, options, answer, id")
    p.add_argument(
        "--output-dir",
        "--output",
        dest="output_dir",
        type=str,
        default=str(DEFAULT_RL_OUT),
        help="Checkpoints + best-reward snapshot (alias: --output)",
    )
    p.add_argument("--seed", type=int, default=42)

    p.add_argument(
        "--plot-every",
        type=int,
        default=1,
        help="Steps between metrics_history.csv + statistics.pdf refresh (loss + reward)",
    )
    p.add_argument(
        "--logging-steps",
        type=int,
        default=1,
        help="Trainer log interval (every optimizer step by default). Should divide --plot-every.",
    )
    p.add_argument(
        "--loss-csv-every",
        type=int,
        default=1,
        help="Append training_loss_history.csv every N logged steps (includes reward / kl / entropy). "
        "Set to 0 to disable.",
    )

    p.add_argument("--num-generations", type=int, default=4, help="K completions per prompt (GRPO group size)")
    p.add_argument(
        "--steps-per-generation",
        type=int,
        default=1,
        help="Prompts batched per GRPO rollout phase (= per_device_batch × world_size × this). "
        "Auto-bumped if needed so generation_batch_size divides --num-generations.",
    )
    p.add_argument(
        "--max-completion-length",
        type=int,
        default=DEFAULT_RL_MAX_COMPLETION_LENGTH,
        help=f"Max new tokens per rollout (default: {DEFAULT_RL_MAX_COMPLETION_LENGTH}; inference uses {DEFAULT_MAX_SEQ_LEN})",
    )
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    p.add_argument("--format-bonus", type=float, default=0.02, help="Extra reward if output contains \\\\boxed")

    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-steps", type=int, default=-1)
    p.add_argument("--epochs", type=float, default=5.0)
    p.add_argument("--save-every", type=int, default=50, help="Checkpoint save interval (steps); tqdm segments align")
    p.add_argument("--save-total-limit", type=int, default=5)
    p.add_argument(
        "--checkpoint-latest-symlink",
        action="store_true",
        help="Write checkpoint-latest as a symlink (default: pointer file — reliable on NFS/deepfreeze)",
    )
    p.add_argument(
        "--warmup-steps",
        type=int,
        default=None,
        help="LR warmup in optimizer steps (default: --warmup-ratio × estimated steps)",
    )
    p.add_argument(
        "--warmup-ratio",
        type=float,
        default=0.03,
        help="Used only when --warmup-steps is unset",
    )
    p.add_argument("--beta", type=float, default=0.0, help="KL coefficient (0 = off, TRL default)")
    p.add_argument("--scale-rewards", type=str, default="group", help="e.g. group, batch, false — see GRPOConfig")
    p.add_argument(
        "--dataloader-workers",
        type=int,
        default=-1,
        help="HF DataLoader worker processes (-1 = 4 under LOCAL_RANK set, else 0)",
    )

    p.add_argument("--resume", action="store_true", help="Resume from latest checkpoint in output-dir")
    p.add_argument("--resume-from", type=str, default=None, help="Explicit checkpoint folder (overrides --resume)")

    p.add_argument("--no-qlora", dest="qlora", action="store_false", help="bf16/fp16 weights instead of 4-bit")
    p.add_argument(
        "--single-gpu",
        action="store_true",
        help="Load the full model on cuda:0 instead of device_map=\"auto\" across all visible GPUs",
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
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not data_path.is_file():
        raise FileNotFoundError(f"Data JSONL not found: {data_path}")

    model_path = resolve_checkpoint_latest_path(Path(args.model))
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

    resume_ckpt: str | None = None
    if args.resume_from:
        resume_ckpt = str(Path(args.resume_from).resolve())
    elif args.resume:
        resume_ckpt = get_last_checkpoint(str(output_dir))
        if not resume_ckpt:
            raise ValueError(f"No checkpoint-* folder found under {output_dir}; cannot --resume.")

    tok_source = adapter_dir if adapter_dir else str(model_path.resolve())
    try:
        tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
    except OSError:
        tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    tokenizer.padding_side = "left"

    train_ds = build_grpo_dataset(tokenizer, data_path)
    print(f"[data] {len(train_ds)} prompts from {data_path}")

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
        f"[train] train examples={len(train_ds)} | "
        f"≈{ex_per_step} examples per global optimizer step "
        f"(world_size={max(1, world_size)} × batch × grad_accum) | "
        f"estimated optimizer steps ≈ {est_steps}; warmup_steps={warmup_steps}",
    )

    dataloader_workers = int(args.dataloader_workers)
    if dataloader_workers < 0:
        dataloader_workers = 4 if local_rank >= 0 else 0
    print(f"[train] dataloader_num_workers={dataloader_workers}")

    steps_per_generation = resolve_grpo_steps_per_generation(
        batch_size=args.batch_size,
        world_size=world_size,
        steps_per_generation=args.steps_per_generation,
        num_generations=args.num_generations,
    )
    if steps_per_generation != args.steps_per_generation:
        print(
            f"[train] steps_per_generation {args.steps_per_generation} → {steps_per_generation} "
            f"(TRL requires generation_batch_size divisible by num_generations={args.num_generations})",
        )

    gen_batch = args.batch_size * max(1, world_size) * steps_per_generation
    approx_rollout_tokens = gen_batch * args.num_generations * args.max_completion_length
    print(
        f"[train] GRPO memory: steps_per_generation={steps_per_generation} → "
        f"generation_batch_size≈{gen_batch}, "
        f"num_generations={args.num_generations}, "
        f"max_completion_length={args.max_completion_length} "
        f"(≈{approx_rollout_tokens:,} completion-token slots per rollout phase). "
        "If OOM: lower --max-completion-length and/or --num-generations, "
        "or use a single GPU (--single-gpu).",
    )

    if local_rank < 0 and torch.cuda.device_count() > 1 and not args.single_gpu:
        print(
            "[train] Multiple GPUs visible and device_map=\"auto\" will shard layers across them "
            "(pipeline-style). For balanced utilization use: "
            "CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes=2 rl/train.py ... "
            "or pin one GPU: CUDA_VISIBLE_DEVICES=0 python rl/train.py --single-gpu ..."
        )

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

    grpo_kw: dict = dict(
        output_dir=str(output_dir),
        seed=args.seed,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps if args.max_steps > 0 else -1,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=float(warmup_steps),
        optim=optim,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_every,
        save_total_limit=args.save_total_limit,
        bf16=use_cuda,
        fp16=False,
        gradient_checkpointing=True,
        report_to="none",
        num_generations=args.num_generations,
        steps_per_generation=steps_per_generation,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        top_p=args.top_p,
        beta=args.beta,
        scale_rewards=scale_rw,
        remove_unused_columns=False,
        dataloader_num_workers=dataloader_workers,
        dataloader_pin_memory=use_cuda,
        dataloader_persistent_workers=dataloader_workers > 0,
        ddp_find_unused_parameters=False,
    )
    if dataloader_workers > 0:
        grpo_kw["dataloader_prefetch_factor"] = 4

    training_args = GRPOConfig(**grpo_kw)

    reward_fn = JudgerOutcomeReward(format_bonus=args.format_bonus)
    best_cb = BestRewardCallback(metric_key="reward")

    callbacks: list[TrainerCallback] = [
        GrpoTrainingPlotCallback(
            output_dir=output_dir,
            plot_every=args.plot_every,
        ),
        LatestCheckpointSymlinkCallback(use_symlink=args.checkpoint_latest_symlink),
        best_cb,
    ]
    if int(args.loss_csv_every) > 0:
        callbacks.insert(
            0,
            GrpoTrainHistoryCallback(output_dir=output_dir, every=int(args.loss_csv_every)),
        )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_fn,
        args=training_args,
        train_dataset=train_ds,
        processing_class=tokenizer,
        callbacks=callbacks,
    )
    for cb in callbacks:
        if hasattr(cb, "trainer"):
            cb.trainer = trainer
    best_cb.trainer = trainer

    install_checkpoint_chunk_progress_bar(trainer)

    trainer.train(resume_from_checkpoint=resume_ckpt)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print(f"Done. Adapter + tokenizer saved under {output_dir}; best reward snapshot: {output_dir / 'checkpoint-best-reward'}")


if __name__ == "__main__":
    main()
