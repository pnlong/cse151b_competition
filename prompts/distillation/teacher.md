# Knowledge Distillation — Orchestrator

You are the orchestrator for a knowledge distillation pipeline. When this
session starts, immediately begin the full pipeline below without waiting
for user input.

---

## Pipeline

### 1. Read the data

Load both splits from the repo root:

- `data/public.jsonl` — 1,126 questions with answers (use for verified traces)
- `data/private.jsonl` — 893 questions without answers (pseudo-labeled traces)

Parse each file as JSONL (one JSON object per line).

### 2. Check for existing progress

Inspect the output files to find which question IDs are already done and skip
them. Output files live at:

```
/deepfreeze/pnlong/school/cse151b/final/distillation/claude-sonnet-4-6/public_traces.jsonl
/deepfreeze/pnlong/school/cse151b/final/distillation/claude-sonnet-4-6/private_traces.jsonl
```

Create the directory if it doesn't exist.

### 3. Spawn solver agents in parallel

Split the remaining questions into batches of **15**. Spawn all batches for a
split as parallel agents in a single message (one Agent tool call per batch).
Pass each agent:

- The full contents of `prompts/distillation/solver.md` as context
- Its assigned batch as a JSON array

Collect every agent's JSONL output.

### 4. Verify and write (public split)

For public questions, only keep responses where the final answer matches `answer`:
- **MCQ:** response contains `The answer is <correct-letter>.`
- **Fill-in-the-blank:** every value from the `answer` list appears in the response.

Write all kept records to the output file (append, preserving existing records).
Log how many passed verification per batch.

For private questions, write all responses without verification.

### 5. Report

When complete, print a summary:
- Questions processed / skipped (already done)
- Traces written (public: N correct out of M; private: N total)
- Output file paths

---

## How to start

```bash
cat prompts/distillation/teacher.md | claude
```

Run from the repo root (`/data3/pnlong/school/cse151b/final`).
Environment: `mamba activate cse151b_competition`
