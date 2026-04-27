"""
Prompt routing for inference.

Design goals:
- Keep primary routing deterministic and cheap (based on `options` and `[ANS]` count).
- Allow an optional lightweight LLM to add *secondary* topic tags (stats/geometry/...)
  without risking format mistakes.
- Produce prompts that match `judger.py` extraction expectations: a single final \\boxed{...}.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from constants import ANS_PLACEHOLDER, MULTI_ANS_NOTE
from prompts.routing.prompts import (
    PRIMARY_PROMPTS,
    SECONDARY_KEYWORDS,
    SECONDARY_REFINEMENTS,
    ROUTER_SYSTEM,
    ROUTER_USER_TEMPLATE,
)


def count_ans_slots(question: str) -> int:
    return question.count(ANS_PLACEHOLDER)


def primary_route(question: str, options: Optional[list]) -> str:
    if options:
        return "mcq_single"
    n = count_ans_slots(question)
    if n >= 2:
        return "fr_multi"
    return "fr_single"


def keyword_secondary_tags(question: str, options: Optional[list]) -> list[str]:
    """Conservative keyword router for secondary tags."""
    text = (question or "").lower()
    if options:
        # include options text lightly; it can contain keywords like "determinant"
        try:
            text += "\n" + "\n".join(str(o) for o in options).lower()
        except Exception:
            pass

    tags: list[str] = []
    for tag, kws in SECONDARY_KEYWORDS.items():
        for kw in kws:
            if kw.lower() in text:
                tags.append(tag)
                break
    # stable order, unique
    seen = set()
    out = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def compose_system(primary: str, secondary: Iterable[str]) -> str:
    sys = PRIMARY_PROMPTS[primary]
    for tag in secondary:
        ref = SECONDARY_REFINEMENTS.get(tag)
        if ref:
            sys += ref
    return sys


def build_user_prompt(question: str, options: Optional[list]) -> str:
    if options:
        labels = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {str(opt).strip()}" for lbl, opt in zip(labels, options))
        return f"{question}\n\nAnswer choices:\n{opts_text}"

    n_slots = count_ans_slots(question)
    if n_slots > 1:
        note = MULTI_ANS_NOTE.format(n=n_slots)
        return question + note
    return question


@dataclass(frozen=True)
class RouteDecision:
    primary: str
    secondary: list[str]
    n_ans: int
    has_options: bool


class BaseRouter:
    def route_one(self, question: str, options: Optional[list]) -> RouteDecision:
        raise NotImplementedError

    def route_batch(self, items: list[dict[str, Any]]) -> list[RouteDecision]:
        return [self.route_one(it["question"], it.get("options")) for it in items]


class RuleBasedRouter(BaseRouter):
    def __init__(self, enable_secondary_keywords: bool = True):
        self.enable_secondary_keywords = enable_secondary_keywords

    def route_one(self, question: str, options: Optional[list]) -> RouteDecision:
        p = primary_route(question, options)
        sec = keyword_secondary_tags(question, options) if self.enable_secondary_keywords else []
        return RouteDecision(
            primary=p,
            secondary=sec,
            n_ans=count_ans_slots(question),
            has_options=bool(options),
        )


class LLMSecondaryRouter(BaseRouter):
    """
    Uses a lightweight LLM to choose *secondary* tags.

    Primary routing remains deterministic (format-based) to avoid LLM mistakes that
    would break the answer format.
    """

    def __init__(self, model: str, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            device_map="auto" if device != "cpu" else None,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()

        # Conservative generation defaults: deterministic JSON
        self.max_new_tokens = 128

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        # Try direct parse
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to find first {...} block
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _secondary_from_llm(self, question: str, options: Optional[list]) -> list[str]:
        user = ROUTER_USER_TEMPLATE.format(question=question, options=options)
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cpu":
            inputs = {k: v.to("cpu") for k, v in inputs.items()}
        else:
            # Let HF handle device placement if using device_map="auto"
            pass

        out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        obj = self._extract_json(decoded)
        if not isinstance(obj, dict):
            return []
        sec = obj.get("secondary", [])
        if not isinstance(sec, list):
            return []
        # Filter to allowed
        allowed = set(SECONDARY_REFINEMENTS.keys())
        out_sec: list[str] = []
        for t in sec:
            if isinstance(t, str) and t in allowed and t not in out_sec:
                out_sec.append(t)
        return out_sec

    def route_one(self, question: str, options: Optional[list]) -> RouteDecision:
        p = primary_route(question, options)
        sec = self._secondary_from_llm(question, options)
        # Fallback: if LLM returns nothing, use conservative keyword tags
        if not sec:
            sec = keyword_secondary_tags(question, options)
        return RouteDecision(
            primary=p,
            secondary=sec,
            n_ans=count_ans_slots(question),
            has_options=bool(options),
        )


def build_routed_prompts(router: BaseRouter, items: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """
    Returns list of (system_prompt, user_prompt) for each item in `items`.
    """
    decisions = router.route_batch(items)
    out: list[tuple[str, str]] = []
    for it, dec in zip(items, decisions):
        system = compose_system(dec.primary, dec.secondary)
        user = build_user_prompt(it["question"], it.get("options"))
        out.append((system, user))
    return out

