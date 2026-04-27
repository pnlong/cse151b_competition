# Prompts

System prompts and instruction templates used across the project.
Organized by purpose.

## Structure

```
prompts/
├── routing/
│   └── prompts.py      Router-oriented prompt library (primary + secondary system prompts)
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
| `SECONDARY_REFINEMENTS` | `dict` | Optional topic-specific addenda (stats, geometry, calculus, linear algebra) |
| `SECONDARY_KEYWORDS` | `dict` | Conservative keyword lists used by the rule-based secondary router |
| `ROUTER_SYSTEM` | `str` | System prompt for the lightweight LLM-based secondary classifier |
| `ROUTER_USER_TEMPLATE` | `str` | User message template for the LLM router (formatted with `question` and `options`) |

#### Routing logic

The router selects a primary prompt based purely on answer format:
- `options` present → `mcq_single` (letter inside `\boxed{}`)
- 2+ `[ANS]` slots → `fr_multi` (all answers comma-separated in one `\boxed{}`)
- otherwise → `fr_single` (single value in `\boxed{}`)

Secondary refinement tags (`stats_inference`, `stats_descriptive`, `geometry`,
`calculus`, `linear_algebra`) are optionally appended when a keyword match or a
lightweight LLM classifier is confident.

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
