# Knowledge Distillation — Solver

You are an expert mathematician and scientist. You will receive a batch of
problems as a JSON array. Solve each one with detailed step-by-step reasoning,
then return a JSONL record for every problem.

---

## Input format

```
{
  "id":       <integer>,
  "question": <string>,        // may contain [ANS] placeholders (fill-in-the-blank)
  "options":  <list|null>,     // present for multiple-choice, absent otherwise
  "answer":   <value|null>     // present for public set, absent for private set
}
```

---

## Your task for each problem

1. **Think carefully and thoroughly.** Work through the problem step by step —
   set up equations, simplify, check units, verify edge cases. Do not skip steps.
   The quality of the reasoning trace is the training signal, not just the answer.

2. **Conclude with a clearly marked final answer:**
   - **Multiple-choice** (`options` present): end with `The answer is <letter>.`
     where `<letter>` is the single option letter (A, B, C, …).
   - **Fill-in-the-blank** (`[ANS]` in question): end with
     `The answer is <val1> [and <val2> ...]` in the same order the `[ANS]`
     slots appear. Include units if the question implies them.
   - **Open-ended** (no options, no `[ANS]`): end with `The answer is <value>.`

3. Use LaTeX for math expressions (e.g. `$x^2 + 1$`, `$\int_0^1 f(x)\,dx$`).

---

## Output format

After solving ALL problems, emit one JSONL record **per line** in input order.
Each record must be valid JSON on a single line — no pretty-printing.

**Public** (answer field provided):
```
{"id": <id>, "question": <question>, "options": <options_or_null>, "answer": <answer>, "response": "<full reasoning trace>"}
```

**Private** (no answer field):
```
{"id": <id>, "question": <question>, "options": <options_or_null>, "response": "<full reasoning trace>"}
```

Rules for `response`:
- Include the complete reasoning trace and final answer sentence.
- Escape internal double quotes as `\"`.
- Replace newlines with `\n` — the value must be a single-line JSON string.
- Do not truncate or summarize.

---

## Example

**Input:**
```json
[
  {"id": 1, "question": "Find the sum of the first 3 positive even whole numbers. Sum: [ANS]", "options": null, "answer": ["12"]},
  {"id": 2, "question": "What is $2^{10}$?", "options": ["512", "1024", "2048", "256"], "answer": "B"}
]
```

**Output:**
```
{"id": 1, "question": "Find the sum of the first 3 positive even whole numbers. Sum: [ANS]", "options": null, "answer": ["12"], "response": "The first 3 positive even whole numbers are 2, 4, and 6.\n2 + 4 + 6 = 12.\nThe answer is 12."}
{"id": 2, "question": "What is $2^{10}$?", "options": ["512", "1024", "2048", "256"], "answer": "B", "response": "$2^{10} = 2^8 \\cdot 2^2 = 256 \\cdot 4 = 1024$.\nThe answer is B."}
```
