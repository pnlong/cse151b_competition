"""
Outcome rewards for GRPO using the competition Judger (and MCQ scoring).

Aligned with ``inference/evaluate.py``: MCQ → ``score_mcq``; free-form →
``Judger.auto_judge`` for single-blank / full-credit; multi-``[ANS]`` uses
fraction of sub-answers judged equal via ``Judger.is_equal``.
"""

from __future__ import annotations

from typing import Any

from judger import Judger

from constants import BOXED_CMD
from inference.utils import score_mcq


def normalize_gold_answer(answer: Any) -> list:
    """Normalize public-set answers to a list for Arrow / HuggingFace datasets.

    ``public.jsonl`` mixes MCQ strings (``\"F\"``) and multi-blank lists (``[\"a\", \"b\"]``).
    PyArrow rejects a single column with both scalars and lists; always store a list.
    """
    if isinstance(answer, list):
        return answer
    return [answer]


def _as_gold_list(gold: Any) -> list:
    if isinstance(gold, list):
        return gold
    return [gold]


def _mcq_gold_letter(gold: Any) -> str:
    """Extract the gold MCQ letter whether *gold* is stored as str or single-item list."""
    items = _as_gold_list(gold)
    if not items:
        return ""
    return str(items[0])


def _norm_gold_list(judger: Judger, gold: Any) -> list[str]:
    return [judger.norm_ans_str(str(item)) for item in _as_gold_list(gold)]


def _freeform_reward(judger: Judger, completion: str, gold: Any) -> float:
    """Return outcome reward in [0, 1]. Multi-blank uses mean sub-answer correctness."""
    gold_norm = _norm_gold_list(judger, gold)
    n = len(gold_norm)
    if n == 1:
        options = [[]] * n
        return 1.0 if judger.auto_judge(completion, _as_gold_list(gold), options) else 0.0

    extracted = judger.extract_ans(completion)
    if not extracted:
        return 0.0
    extracted_pred = judger.split_by_comma(extracted)
    extracted_pred = [judger.norm_ans_str(item) for item in extracted_pred]
    if len(extracted_pred) != len(gold_norm):
        return 0.0
    correct = sum(
        1 for pred_i, gold_i in zip(extracted_pred, gold_norm) if judger.is_equal(pred_i, gold_i)
    )
    return correct / float(n)


class JudgerOutcomeReward:
    """
    TRL-compatible reward callable: ``(prompts, completions, is_mcq=..., gold=..., **kwargs)``.

    ``is_mcq`` and ``gold`` must be HuggingFace batch columns parallel to ``completions``.
    """

    __name__ = "judger_outcome"

    def __init__(self, format_bonus: float = 0.02):
        self.format_bonus = float(format_bonus)
        self._judger = Judger(strict_extract=False)

    def __call__(
        self,
        prompts: list[str],
        completions: list[str],
        is_mcq: list[bool],
        gold: list[Any],
        **kwargs: Any,
    ) -> list[float]:
        del prompts, kwargs
        rewards: list[float] = []
        boxed_needle = BOXED_CMD if "\\" in BOXED_CMD else f"\\{BOXED_CMD}"
        for completion, mcq, g in zip(completions, is_mcq, gold):
            try:
                if bool(mcq):
                    r = 1.0 if score_mcq(completion, _mcq_gold_letter(g)) else 0.0
                else:
                    r = _freeform_reward(self._judger, completion, g)
            except Exception:
                r = 0.0
            if self.format_bonus > 0.0 and (
                boxed_needle in completion
                or "\\boxed{" in completion
            ):
                r += self.format_bonus
            rewards.append(float(r))
        return rewards


def make_judger_outcome_reward(format_bonus: float = 0.02) -> JudgerOutcomeReward:
    """Factory for backward compatibility; prefer ``JudgerOutcomeReward`` directly."""
    return JudgerOutcomeReward(format_bonus=format_bonus)
