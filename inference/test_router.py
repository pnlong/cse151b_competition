#!/usr/bin/env python3
"""
Standalone router sanity test.

This script does NOT load vLLM or the main Qwen3-4B model.
It only exercises the routing logic:
- primary route (deterministic: options / [ANS] count)
- optional secondary tags (keyword-based or tiny LLM classifier)

Examples:
  # Rule-based router only (no model downloads)
  python inference/test_router.py --limit 10

  # Run on public set
  python inference/test_router.py --data data/public.jsonl --limit 20

  # Enable tiny LLM to choose secondary tags (will download router model once)
  python inference/test_router.py --limit 5 --secondary-llm --router-device cpu
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from inference.utils import load_jsonl
from inference.router import (
    RuleBasedRouter,
    LLMSecondaryRouter,
    build_routed_prompts,
)


def parse_args():
    p = argparse.ArgumentParser(description="Standalone prompt-router sanity test")
    p.add_argument("--data", default="data/public.jsonl", help="Input JSONL (default: data/public.jsonl)")
    p.add_argument("--limit", type=int, default=10, help="Number of questions to route (default: 10)")
    p.add_argument("--secondary-llm", action="store_true",
                   help="Use a tiny LLM to choose secondary tags (downloads model once).")
    p.add_argument("--router-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="Router model for --secondary-llm (default: Qwen2.5-0.5B-Instruct).")
    p.add_argument("--router-device", default="cpu", choices=["cpu", "auto"],
                   help="Device for router model: cpu (safe) or auto.")
    p.add_argument("--print-prompts", action="store_true",
                   help="Also print the resulting system+user prompts (truncated).")
    p.add_argument("--out", default="", help="Optional path to write decisions as JSONL.")
    return p.parse_args()


def truncate(s: str, n: int = 500) -> str:
    s = s.replace("\r\n", "\n")
    return s if len(s) <= n else s[: n] + "\n...<truncated>..."


def main():
    args = parse_args()
    data = load_jsonl(Path(args.data))
    data = data[: args.limit]

    if args.secondary_llm:
        router = LLMSecondaryRouter(
            model=args.router_model,
            device=("cpu" if args.router_device == "cpu" else "auto"),
        )
    else:
        router = RuleBasedRouter(enable_secondary_keywords=True)

    decisions = router.route_batch(data)
    routed_prompts = build_routed_prompts(router, data)

    out_f = open(args.out, "w", encoding="utf-8") if args.out else None
    try:
        for item, dec, (sys_p, user_p) in zip(data, decisions, routed_prompts):
            rec: dict[str, Any] = {
                "id": item.get("id"),
                "primary": dec.primary,
                "secondary": dec.secondary,
                "n_ans": dec.n_ans,
                "has_options": dec.has_options,
            }
            print(json.dumps(rec, ensure_ascii=False))
            if out_f:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if args.print_prompts:
                print("SYSTEM_PROMPT:\n" + truncate(sys_p, 800))
                print("USER_PROMPT:\n" + truncate(user_p, 800))
                print("-" * 60)
    finally:
        if out_f:
            out_f.close()


if __name__ == "__main__":
    main()

