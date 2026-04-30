# Data

Competition datasets for the CSE 151B math reasoning task.

## Files

| File | Split | Records | Has answers |
|------|-------|---------|-------------|
| `public.jsonl` | Public (train/eval) | 1,126 | Yes |
| `private.jsonl` | Private (test) | 893 | No |

## Schema

Every record is a JSON object on a single line.

**Public** (`public.jsonl`):
```json
{
  "id":       0,
  "question": "Find the sum ... [ANS]",
  "options":  ["A", "B", "C", "D"],
  "answer":   ["42"]
}
```

**Private** (`private.jsonl`):
```json
{
  "id":       0,
  "question": "Suppose that $M$ is the function ..."
}
```

Private records never have `answer` or `options` — those must be predicted.

## Question types

**Multiple-choice** — `options` list is present. `answer` is a single letter
(`"A"`, `"B"`, …) indexing into `options`.

**Fill-in-the-blank** — `[ANS]` placeholder(s) appear in the question text.
`answer` is a list of strings in the same order as the placeholders.
Questions can have more than one `[ANS]` slot.

**Open-ended** — No `options`, no `[ANS]`. `answer` is a list with one value.

## Notes

- Math is typeset in LaTeX notation (e.g. `$x^2$`, `$\int_0^1 f(x)\,dx$`).
- `id` values are unique within each split but the two splits share the same
  id namespace (i.e. both start at 0).
- Answer values for fill-in-the-blank questions may be symbolic expressions
  (e.g. `"325*(1+325)"`) rather than evaluated numerals — both forms are
  accepted during evaluation.
