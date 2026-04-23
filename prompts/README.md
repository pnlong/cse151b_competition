# Prompts

System prompts and instruction templates used across the project.
Organized by purpose.

## Structure

```
prompts/
└── distillation/
    └── teacher.md      System prompt for knowledge distillation via Claude
```

## Subdirectories

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
