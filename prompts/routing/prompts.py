"""
System prompts used by the prompt router.

Primary routing is **answer format** (MCQ vs single vs multi free-response).
Optional **topic refinements** append short, topic-specific guardrails; labels
match ``topic_taxonomy.CANONICAL_TOPIC_ORDER`` (same 20-way scoring as offline
``classify_topics.py``).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root so ``topic_taxonomy`` imports when this package is loaded
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from topic_taxonomy import CANONICAL_TOPIC_ORDER

# ── Core output-format prompts (primary routing) ────────────────────────────────

SYSTEM_FR_SINGLE = (
    "You are an expert mathematician. Solve the problem carefully.\n"
    "- Work step by step, but keep the solution focused.\n"
    "- Your final output MUST contain exactly one final answer wrapped in \\boxed{...}.\n"
    "- Put ONLY the final answer inside \\boxed{...} (no words).\n"
    "- Do not include any other \\boxed{} earlier in the response.\n"
    "- Use exact values (fractions, radicals, \\pi) when appropriate; otherwise a decimal is fine.\n"
    "- Once you have written \\boxed{...}, stop immediately. Do not revise, second-guess, or add any text after the boxed answer.\n"
)

SYSTEM_FR_MULTI = (
    "You are an expert mathematician. Solve the problem carefully.\n"
    "- This problem has multiple blanks marked [ANS]. Fill them IN ORDER.\n"
    "- Your final output MUST contain exactly one \\boxed{...}.\n"
    "- Put ALL answers inside ONE \\boxed{...}, separated by commas, in the order the blanks appear.\n"
    "  Example: \\boxed{answer_1, answer_2, answer_3}\n"
    "- Put ONLY the answers inside \\boxed{...} (no labels like (a) or units unless required).\n"
    "- Do not include any other \\boxed{} earlier in the response.\n"
    "- Once you have written \\boxed{...}, stop immediately. Do not revise, second-guess, or add any text after the boxed answer.\n"
)

SYSTEM_MCQ_SINGLE = (
    "You are an expert mathematician. Carefully solve the problem and choose the correct option.\n"
    "- Your final output MUST be exactly one letter inside \\boxed{...}.\n"
    "- Put ONLY the letter (A, B, C, ...) inside \\boxed{...}. Example: \\boxed{C}\n"
    "- Do not include any other \\boxed{} earlier in the response.\n"
    "- Do NOT output the full option text; only output the letter.\n"
    "- Once you have written \\boxed{...}, stop immediately. Do not revisit your choice or add any text after the boxed answer.\n"
)

# ── Optional topic refinements (one label per problem from topic_taxonomy) ─────

_TOPIC_REFINE_BODY: dict[str, str] = {
    "Arithmetic": (
        "\nExtra rules for arithmetic / numeric reasoning:\n"
        "- Track units and signs; reduce fractions and simplify radicals when asked.\n"
        "- For ratios, percentages, and remainders, restate the interpretation before computing.\n"
    ),
    "Algebra": (
        "\nExtra rules for algebra and functions:\n"
        "- State domains where relevant; watch extraneous solutions after squaring or clearing denominators.\n"
        "- For systems, check consistency; for quadratics, consider discriminant and factoring first.\n"
    ),
    "Trigonometry": (
        "\nExtra rules for trigonometry:\n"
        "- Prefer exact values (sin, cos, tan of standard angles) when possible; mind radians vs degrees if specified.\n"
        "- Use identities to simplify before inverting trig functions; check quadrant for inverse-trig answers.\n"
    ),
    "Geometry": (
        "\nExtra rules for geometry:\n"
        "- Sketch relationships mentally; mark known lengths/angles.\n"
        "- Use exact forms (radicals, \\pi) when natural; sanity-check units and scale.\n"
    ),
    "Number Theory": (
        "\nExtra rules for number theory:\n"
        "- State divisibility, gcd/lcm, and modular steps clearly; keep congruences consistent modulo m.\n"
        "- For counting primes or factors, avoid off-by-one errors.\n"
    ),
    "Combinatorics": (
        "\nExtra rules for combinatorics:\n"
        "- Decide whether order matters and whether replacement is allowed before using nCr/nPr.\n"
        "- For inclusion–exclusion or casework, ensure cases are disjoint or adjust overlaps.\n"
    ),
    "Probability": (
        "\nExtra rules for probability:\n"
        "- Identify sample space and whether events are independent; match conditional vs joint probability.\n"
        "- For expectation/variance, use linearity where applicable and check boundaries.\n"
    ),
    "Statistics — Descriptive": (
        "\nExtra rules for descriptive statistics (quartiles/IQR/boxplot/regression summaries):\n"
        "- Use the standard classroom 'median-of-halves (Tukey)' convention unless the problem states otherwise.\n"
        "- Keep track of ordering and whether the median is included in halves when n is odd.\n"
    ),
    "Statistics — Inference": (
        "\nExtra rules for hypothesis tests / inference:\n"
        "- Identify H0 and H1 and whether the test is left-, right-, or two-tailed.\n"
        "- Use the tail that matches H1 for critical value and p-value; conclusion must match reject vs fail to reject H0.\n"
    ),
    "Sequences & Recurrences": (
        "\nExtra rules for sequences and recurrences:\n"
        "- Index shifts (n vs n+1); verify the first few terms against the recurrence.\n"
        "- For closed forms, substitute back into the recurrence when feasible.\n"
    ),
    "Series & Convergence": (
        "\nExtra rules for series:\n"
        "- Name the test you use (geometric, ratio, root, comparison, alternating) and check hypotheses.\n"
        "- For power series, state interval/radius carefully, including endpoint checks when required.\n"
    ),
    "Differential Calculus": (
        "\nExtra rules for differential calculus:\n"
        "- Mind domain, chain/product/quotient rules, and implicit differentiation constraints.\n"
        "- For optimization, justify extrema with derivative tests or closed-interval endpoints.\n"
    ),
    "Integral Calculus": (
        "\nExtra rules for integral calculus:\n"
        "- Watch limits of integration, odd/even symmetry, and substitution Jacobian where relevant.\n"
        "- For improper integrals, split at singularities and take limits separately.\n"
    ),
    "Differential Equations": (
        "\nExtra rules for differential equations:\n"
        "- Identify order, linearity, and method (separable, integrating factor, etc.); use initial/boundary data to fix constants.\n"
        "- Substitute the solution back when quick verification is possible.\n"
    ),
    "Linear Algebra": (
        "\nExtra rules for linear algebra:\n"
        "- For determinants/eigenvalues, use row/column operations or invariants; row-rank vs column-rank for consistency.\n"
        "- Check dimensions when multiplying matrices or interpreting span/basis.\n"
    ),
    "Complex Analysis": (
        "\nExtra rules for complex numbers:\n"
        "- Use Re/Im, modulus, and argument consistently; mind branch cuts if stated.\n"
        "- For roots of unity or conjugates, exploit symmetry before expanding.\n"
    ),
    "Real Analysis": (
        "\nExtra rules for analysis-style problems:\n"
        "- State epsilons/deltas or bounds explicitly; monotone and bounded arguments when applicable.\n"
        "- For sup/inf, relate to least upper bound property and avoid confusing strict vs non-strict inequalities.\n"
    ),
    "Discrete Mathematics": (
        "\nExtra rules for discrete math (graphs, algorithms, logic):\n"
        "- For graphs, track vertices vs edges and whether the graph is directed; induction bases cover smallest cases.\n"
        "- For counting on structures, avoid double counting.\n"
    ),
    "Applied Mathematics": (
        "\nExtra rules for applied / modeling problems:\n"
        "- Translate words to equations carefully; note units and proportionality constants.\n"
        "- Sanity-check magnitudes (growth/decay, finance, mixture) against the scenario.\n"
    ),
    "Abstract Algebra": (
        "\nExtra rules for abstract algebra:\n"
        "- State the structure (group/ring/field) and axioms you use; homomorphism kernels/images precisely.\n"
        "- For cosets/quotients, keep representatives consistent.\n"
    ),
}

TOPIC_REFINEMENTS: dict[str, str] = {
    topic: _TOPIC_REFINE_BODY.get(topic, "") for topic in CANONICAL_TOPIC_ORDER
}

if set(TOPIC_REFINEMENTS) != set(CANONICAL_TOPIC_ORDER):
    raise RuntimeError("TOPIC_REFINEMENTS keys must match CANONICAL_TOPIC_ORDER")

# ── Router-facing structures ───────────────────────────────────────────────────

PRIMARY_PROMPTS = {
    "fr_single": SYSTEM_FR_SINGLE,
    "fr_multi": SYSTEM_FR_MULTI,
    "mcq_single": SYSTEM_MCQ_SINGLE,
}

_TOPIC_CHOICES_JSON = json.dumps(list(CANONICAL_TOPIC_ORDER), ensure_ascii=False)

ROUTER_SYSTEM = (
    "You are a routing assistant for math competition problems.\n"
    "Your job: output JSON describing the answer format and (optionally) the curriculum topic.\n"
    "Use ONLY the provided fields: question (string), options (null or list of strings).\n\n"
    "You MUST output ONLY valid JSON (no markdown, no commentary).\n\n"
    "### Primary route (required)\n"
    "Choose exactly ONE of:\n"
    '- "mcq_single": options is a non-empty list\n'
    '- "fr_multi": options is null/empty AND question contains 2+ occurrences of "[ANS]"\n'
    '- "fr_single": otherwise\n\n'
    "### Topic (optional)\n"
    'Set "topic" to exactly ONE string from the following JSON array, or "" if unsure:\n'
    f"{_TOPIC_CHOICES_JSON}\n"
    "The string must match an element exactly (including punctuation and dashes).\n\n"
    "### Output schema\n"
    "{\n"
    '  "primary": <string>,\n'
    '  "topic": <string>,\n'
    '  "n_ans": <integer count of "[ANS]" in question>,\n'
    '  "has_options": <true|false>\n'
    "}\n\n"
    "Hard rules:\n"
    "- Count [ANS] literally.\n"
    "- If options is present and non-empty, primary MUST be mcq_single.\n"
    '- If unsure about topic, set "topic" to "".\n'
)

ROUTER_USER_TEMPLATE = (
    "question:\n"
    "{question}\n\n"
    "options:\n"
    "{options}\n"
)
