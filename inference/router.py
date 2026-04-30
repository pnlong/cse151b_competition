"""
Prompt routing for inference.

Design goals:
- Keep primary routing deterministic and cheap (based on `options` and `[ANS]` count).
- Attach optional topic refinements using the same 20-way ``topic_taxonomy`` scoring
  as offline ``classify_topics.py`` (question + option text).
- Optional lightweight LLM may override the topic label when confident; invalid
  outputs fall back to the taxonomy classifier.
- Produce prompts that match `judger.py` extraction expectations: a single final \\boxed{...}.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from constants import ANS_PLACEHOLDER, MULTI_ANS_NOTE
from prompts.routing.prompts import (
    PRIMARY_PROMPTS,
    ROUTER_SYSTEM,
    ROUTER_USER_TEMPLATE,
    TOPIC_REFINEMENTS,
)
from topic_taxonomy import CANONICAL_TOPIC_ORDER, classify_problem


def count_ans_slots(question: str) -> int:
    return question.count(ANS_PLACEHOLDER)


def primary_route(question: str, options: Optional[list]) -> str:
    if options:
        return "mcq_single"
    n = count_ans_slots(question)
    if n >= 2:
        return "fr_multi"
    return "fr_single"


def compose_system(
    primary: str,
    topic: str,
    *,
    enable_topic_refinements: bool = True,
) -> str:
    sys = PRIMARY_PROMPTS[primary]
    if enable_topic_refinements:
        sys += TOPIC_REFINEMENTS.get(topic, "")
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


_ALLOWED_TOPICS = frozenset(CANONICAL_TOPIC_ORDER)


@dataclass(frozen=True)
class RouteDecision:
    primary: str
    topic: str
    n_ans: int
    has_options: bool


class BaseRouter:
    enable_topic_refinements: bool = True

    def route_one(self, question: str, options: Optional[list]) -> RouteDecision:
        raise NotImplementedError

    def route_batch(self, items: list[dict[str, Any]]) -> list[RouteDecision]:
        return [self.route_one(it["question"], it.get("options")) for it in items]


class RuleBasedRouter(BaseRouter):
    """Deterministic format routing + taxonomy topic label (``classify_problem``)."""

    def __init__(self, enable_topic_refinements: bool = True):
        self.enable_topic_refinements = enable_topic_refinements

    def route_one(self, question: str, options: Optional[list]) -> RouteDecision:
        p = primary_route(question, options)
        topic = classify_problem(question, options)
        return RouteDecision(
            primary=p,
            topic=topic,
            n_ans=count_ans_slots(question),
            has_options=bool(options),
        )


class LLMTopicRouter(BaseRouter):
    """
    Primary routing stays deterministic. A lightweight LLM may suggest a topic;
    invalid or empty suggestions fall back to ``classify_problem``.
    """

    def __init__(self, model: str, device: str = "cpu", enable_topic_refinements: bool = True):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_name = model
        self.device = device
        self.enable_topic_refinements = enable_topic_refinements
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            trust_remote_code=True,
            device_map="auto" if device != "cpu" else None,
        )
        if device == "cpu":
            self.model.to("cpu")
        self.model.eval()
        self.max_new_tokens = 256

    def _extract_json(self, text: str) -> dict[str, Any] | None:
        text = text.strip()
        try:
            return json.loads(text)
        except Exception:
            pass
        m = re.search(r"\{[\s\S]*\}", text)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    def _topic_from_llm(self, question: str, options: Optional[list]) -> str | None:
        user = ROUTER_USER_TEMPLATE.format(question=question, options=options)
        messages = [
            {"role": "system", "content": ROUTER_SYSTEM},
            {"role": "user", "content": user},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(prompt, return_tensors="pt")
        if self.device == "cpu":
            inputs = {k: v.to("cpu") for k, v in inputs.items()}

        out = self.model.generate(
            **inputs,
            do_sample=False,
            max_new_tokens=self.max_new_tokens,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        decoded = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        obj = self._extract_json(decoded)
        if not isinstance(obj, dict):
            return None
        raw = obj.get("topic", "")
        if raw is None:
            return None
        if not isinstance(raw, str):
            return None
        raw = raw.strip()
        if not raw:
            return None
        if raw in _ALLOWED_TOPICS:
            return raw
        return None

    def route_one(self, question: str, options: Optional[list]) -> RouteDecision:
        p = primary_route(question, options)
        topic = self._topic_from_llm(question, options)
        if topic is None:
            topic = classify_problem(question, options)
        return RouteDecision(
            primary=p,
            topic=topic,
            n_ans=count_ans_slots(question),
            has_options=bool(options),
        )


# Backwards-compatible alias
LLMSecondaryRouter = LLMTopicRouter


def build_routed_prompts(router: BaseRouter, items: list[dict[str, Any]]) -> list[tuple[str, str]]:
    """Returns list of (system_prompt, user_prompt) for each item in `items`."""
    decisions = router.route_batch(items)
    out: list[tuple[str, str]] = []
    enable = getattr(router, "enable_topic_refinements", True)
    for it, dec in zip(items, decisions):
        system = compose_system(
            dec.primary,
            dec.topic,
            enable_topic_refinements=enable,
        )
        user = build_user_prompt(it["question"], it.get("options"))
        out.append((system, user))
    return out
