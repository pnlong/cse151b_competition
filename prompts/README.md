# Prompts

System prompts and instruction templates used across the project.
Organized by purpose.

## Structure

```
prompts/
├── routing/
│   └── prompts.py      Router-oriented prompt library (primary + topic refinements)
└── distillation/
    ├── teacher.md      System prompt for knowledge distillation via Claude
    └── solver.md       Sub-agent prompt for batch problem solving
```

## Subdirectories

### `routing/`

Prompt definitions and routing structures used by `inference/router.py` to select the
appropriate system prompt for each problem at inference time.

| Symbol | Type | Purpose |
|--------|------|---------|
| `PRIMARY_PROMPTS` | `dict` | Maps route keys (`fr_single`, `fr_multi`, `mcq_single`) to system prompt strings |
| `TOPIC_REFINEMENTS` | `dict` | Optional addenda keyed by the 20 labels in `topic_taxonomy.CANONICAL_TOPIC_ORDER` (values may be empty) |
| `ROUTER_SYSTEM` | `str` | System prompt for the optional lightweight LLM topic classifier (strict JSON) |
| `ROUTER_USER_TEMPLATE` | `str` | User message template for the LLM router (formatted with `question` and `options`) |

Topic labels and weighted-regex **scoring** live in repo-root `topic_taxonomy.py` (shared with `analysis/classify_topics.py`); this file only holds **prompt** text keyed by those labels.

#### Routing logic

The router selects a primary prompt based purely on answer format:
- `options` present → `mcq_single` (letter inside `\boxed{}`)
- 2+ `[ANS]` slots → `fr_multi` (all answers comma-separated in one `\boxed{}`)
- otherwise → `fr_single` (single value in `\boxed{}`)

A single topic label is chosen via `topic_taxonomy.classify_problem` (question + option text).
Optional `TOPIC_REFINEMENTS[topic]` text is appended when enabled. An optional tiny LLM may
suggest a topic; invalid suggestions fall back to the same classifier.

### `distillation/`

Prompts for collecting reasoning traces from teacher models (used to build
the SFT training set for the student model).

| File | Role | Purpose |
|------|------|---------|
| `teacher.md` | Orchestrator | Loaded into the Claude Code session; reads data files, spawns solver agents in parallel, verifies answers, writes output |
| `solver.md` | Sub-agent | Passed to each spawned agent; instructs it to solve a batch of problems and return JSONL traces |

#### Usage

```bash
cat prompts/distillation/teacher.md | claude
```

Run from the repo root. The session is fully autonomous — it reads both data
splits, fans out solver agents across batches in parallel, and writes results
to the distillation output directory. No further input needed.

Environment: `mamba activate cse151b_competition`

See [`distillation/teacher.md`](distillation/teacher.md) and
[`distillation/solver.md`](distillation/solver.md) for full prompt text.
