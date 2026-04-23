"""
Inference-specific utilities.

Builds on the root utils.py (math answer helpers) and adds:
  - Prompt construction
  - \\boxed{} extraction (wrapping root utils)
  - Answer normalization and voting
  - JSONL / CSV I/O helpers
"""

import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

# ── Re-export relevant helpers from the root utils.py ─────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from utils import (           # noqa: E402  (import after sys.path manipulation)
    last_boxed_only_string,   # finds last \boxed{...} substring
    remove_boxed,             # strips the \boxed{ } wrapper, returns inner text
    norm_str2bool,
    fix_sqrt,
    fix_fracs,
)

# Re-export so callers can do `from inference.utils import last_boxed_only_string`
__all__ = [
    "last_boxed_only_string", "remove_boxed", "norm_str2bool", "fix_sqrt", "fix_fracs",
    "extract_last_boxed", "split_top_level_commas", "norm_for_vote",
    "count_ans_slots", "answer_key", "majority_vote",
    "extract_letter", "score_mcq",
    "build_prompt", "apply_chat_template_safe",
    "load_jsonl", "save_jsonl", "save_submission_csv", "save_results_jsonl",
]


# ── \\boxed{} extraction ───────────────────────────────────────────────────────

def extract_last_boxed(text: str) -> Optional[str]:
    """
    Return the inner content of the last \\boxed{} in *text*, or None.

    Wraps root utils' last_boxed_only_string + remove_boxed so callers
    don't need to chain two calls.
    """
    return remove_boxed(last_boxed_only_string(text))


# ── Answer normalization ───────────────────────────────────────────────────────

def norm_for_vote(s: str) -> str:
    """
    Lightweight normalization used only for vote-counting (not for correctness
    judgement — the Judger handles that).  Strips spacing and common LaTeX
    spacing macros so that e.g. "3/4" and "3 / 4" count as the same vote.
    """
    s = s.strip().lower()
    s = re.sub(r"\\[,;:! ]+", "", s)          # LaTeX thin/thick spaces
    s = re.sub(r"\\(?:left|right)", "", s)     # \left, \right
    s = re.sub(r"\s+", "", s)                  # all whitespace
    s = s.replace("\u2212", "-").replace("\u2013", "-")   # unicode minus
    return s


def split_top_level_commas(s: str) -> list[str]:
    """
    Split *s* by commas that are not inside {}, [], or ().
    Used to separate multi-answer \\boxed{a, b, c} into ["a", "b", "c"].
    """
    parts: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in s:
        if ch in "({[":
            depth += 1
            buf.append(ch)
        elif ch in ")}]":
            depth -= 1
            buf.append(ch)
        elif ch == "," and depth == 0:
            parts.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    if buf:
        parts.append("".join(buf).strip())
    return parts


# ── Per-question voting key ────────────────────────────────────────────────────

def count_ans_slots(question: str) -> int:
    """Count the number of [ANS] placeholders in a question string."""
    return question.count("[ANS]")


def answer_key(response: str, n_slots: int, is_mcq: bool) -> str:
    """
    Derive a canonical string key from one model response, used for vote-counting.

    - MCQ        : the letter inside \\boxed{} (e.g. "C")
    - 1-slot     : norm_for_vote of the last \\boxed{} content
    - N-slot     : norm_for_vote of each comma part, joined with "|"
    Returns "" when no parseable answer is found.
    """
    raw = extract_last_boxed(response)
    if raw is None:
        return ""

    if is_mcq:
        m = re.match(r"[A-Ja-j]", raw.strip())
        return m.group(0).upper() if m else norm_for_vote(raw)

    if n_slots <= 1:
        return norm_for_vote(raw)

    parts = split_top_level_commas(raw)
    if len(parts) >= n_slots:
        return "|".join(norm_for_vote(p) for p in parts[:n_slots])
    return norm_for_vote(raw)  # fallback: treat as opaque blob


# ── Self-consistency voting ────────────────────────────────────────────────────

def majority_vote(responses: list[str], n_slots: int, is_mcq: bool) -> str:
    """
    Given *N* responses for a single question, return the response whose
    extracted answer received the most votes (plurality).

    Ties are broken by order of first occurrence.
    Falls back to responses[0] if nothing is parseable.
    """
    keyed = [(answer_key(r, n_slots, is_mcq), r) for r in responses]
    valid = [(k, r) for k, r in keyed if k]
    if not valid:
        return responses[0]
    best_key = Counter(k for k, _ in valid).most_common(1)[0][0]
    for k, r in valid:
        if k == best_key:
            return r
    return responses[0]


# ── Prompt construction ────────────────────────────────────────────────────────

def build_prompt(question: str, options: Optional[list],
                 system_math: str, system_mcq: str,
                 multi_ans_note: str) -> tuple[str, str]:
    """
    Return *(system_prompt, user_prompt)* for one dataset item.

    Args:
        question       : raw question string (may contain [ANS] slots)
        options        : list of option strings for MCQ, or None
        system_math    : system prompt for free-form questions  (from constants)
        system_mcq     : system prompt for MCQ questions        (from constants)
        multi_ans_note : format string with {n} for multi-slot note (from constants)
    """
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return system_mcq, f"{question}\n\nAnswer choices:\n{opts_text}"

    n_slots = count_ans_slots(question)
    if n_slots > 1:
        note = multi_ans_note.format(n=n_slots)
        return system_math, question + note

    return system_math, question


def apply_chat_template_safe(tokenizer, messages: list[dict]) -> str:
    """
    Apply the tokenizer's chat template with thinking mode enabled if supported,
    falling back gracefully if the tokenizer does not accept enable_thinking.
    """
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=True, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


# ── MCQ letter extraction ─────────────────────────────────────────────────────

def extract_letter(text: str) -> str:
    """
    Extract the answer letter from an MCQ response.

    Primary:  looks for \\boxed{A} (single letter inside \\boxed{}).
    Fallback: takes the last standalone uppercase letter A–J in the text.
              A–J covers all possible option labels (options list has ≤ 10 items).
    """
    m = re.search(r"\\boxed\{([A-Za-z])\}", text)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([A-J])\b", text.upper())
    return matches[-1] if matches else ""


def score_mcq(response: str, gold_letter: str) -> bool:
    """Return True if the extracted letter matches the gold letter."""
    return extract_letter(response) == gold_letter.strip().upper()


# ── I/O helpers ───────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    """Load a newline-delimited JSON file into a list of dicts."""
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def save_jsonl(records: list[dict], path: Path) -> None:
    """Write a list of dicts to a newline-delimited JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def save_results_jsonl(records: list[dict], path: Path) -> None:
    """
    Save full evaluation records as JSONL.
    Each record should contain: {id, is_mcq, gold, response, correct}.
    Mirrors the format used in the starter notebook for local analysis.
    """
    save_jsonl(records, path)


def save_submission_csv(rows: list[dict], path: Path) -> None:
    """
    Write the final submission CSV with columns [id, response].
    *rows* must be a list of dicts with at least those two keys.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "response"])
        writer.writeheader()
        writer.writerows({"id": r["id"], "response": r["response"]} for r in rows)
