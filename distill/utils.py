"""
Distillation-specific utilities.

Imports all helpers from inference/utils.py and adds:
  - model_slug()     — safe directory name from a HuggingFace model ID
  - traces_dir()     — storage path for one model's trace files
  - verify_trace()   — check a teacher response against ground truth
"""

import sys
from pathlib import Path

# ── Re-export everything from inference/utils.py ──────────────────────────────
_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from inference.utils import (          # noqa: E402
    last_boxed_only_string,
    remove_boxed,
    norm_str2bool,
    fix_sqrt,
    fix_fracs,
    extract_last_boxed,
    split_top_level_commas,
    norm_for_vote,
    count_ans_slots,
    answer_key,
    majority_vote,
    extract_letter,
    score_mcq,
    build_prompt,
    apply_chat_template_safe,
    load_jsonl,
    save_jsonl,
    save_results_jsonl,
    save_submission_csv,
)

__all__ = [
    # re-exported
    "last_boxed_only_string", "remove_boxed", "norm_str2bool", "fix_sqrt", "fix_fracs",
    "extract_last_boxed", "split_top_level_commas", "norm_for_vote",
    "count_ans_slots", "answer_key", "majority_vote",
    "extract_letter", "score_mcq",
    "build_prompt", "apply_chat_template_safe",
    "load_jsonl", "save_jsonl", "save_results_jsonl", "save_submission_csv",
    # distill-specific
    "model_slug", "traces_dir", "verify_trace",
]


# ── Distill-specific helpers ───────────────────────────────────────────────────

def model_slug(model_id: str) -> str:
    """
    Convert a HuggingFace model ID to a safe directory name.

    Examples:
        "Qwen/Qwen3-32B"                        → "qwen3-32b"
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" → "deepseek-r1-distill-qwen-32b"
    """
    name = model_id.split("/")[-1]   # drop the org prefix
    return name.lower().replace("_", "-")


def traces_dir(model_id: str) -> Path:
    """
    Return the storage directory for one teacher model's trace files.
    Creates the directory if it does not exist.
    """
    from config import DISTILL_DIR  # imported here to avoid circular import at module level
    d = DISTILL_DIR / model_slug(model_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def verify_trace(response: str, gold, is_mcq: bool, judger) -> bool:
    """
    Check whether a teacher response is correct against the ground truth.

    For MCQ  : compares the extracted letter to the gold letter.
    For free-form: uses Judger.auto_judge for symbolic / numeric equivalence.

    Args:
        response : full model response string (may contain thinking tokens)
        gold     : ground-truth answer — letter string for MCQ, string or
                   list[str] for free-form
        is_mcq   : True if this is a multiple-choice question
        judger   : an instantiated Judger object (from judger.py)

    Returns:
        True if the response is correct, False otherwise.
        Never raises — exceptions from Judger are caught and treated as False.
    """
    if is_mcq:
        return score_mcq(response, str(gold))

    gold_list = gold if isinstance(gold, list) else [gold]
    try:
        return judger.auto_judge(
            pred=response,
            gold=gold_list,
            options=[[]] * len(gold_list),
        )
    except Exception:
        return False
