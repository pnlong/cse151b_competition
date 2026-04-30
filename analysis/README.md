# Analysis scripts

Utilities for exploring the competition datasets (`data/public.jsonl`, `data/private.jsonl`). Run everything from the **repository root** with the project environment activated:

```bash
micromamba activate cse151b_competition
```

Paths to the JSONL files come from [`config.py`](../config.py) (via `.env` → `ROOT_DIR`).

---

## Scripts

| Script | Purpose |
|--------|---------|
| [`dataset_breakdown.py`](dataset_breakdown.py) | Summarizes **answer format** (MCQ vs free-form single vs multi-`[ANS]`) using `primary_route` on the JSONL fields. **Topic** bars come from either (**default**) inference **secondary keywords** via [`RuleBasedRouter`](../inference/router.py) — sparse 5-topic hints — or from **`--source csv`** reading [`data/topic_classifications.csv`](../data/topic_classifications.csv) (20 mutually exclusive topics from `classify_topics.py`). Prints tables to stdout and optionally saves a **two-panel horizontal bar chart**. |
| [`classify_topics.py`](classify_topics.py) | Offline **20-topic** taxonomy via weighted keyword scoring (not used at inference). Writes [`data/topic_classifications.csv`](../data/topic_classifications.csv) with columns `set`, `id`, `topic`. |

### `dataset_breakdown.py`

- **`--source router`** (default): topic bars = inference router secondary keywords (may overlap per problem; many rows end up as “None”).
- **`--source csv`**: topic bars = labels from `topic_classifications.csv` (exactly one topic per problem). Requires the CSV to exist — run `classify_topics.py` first.
- **`--classifications PATH`**: CSV path when using `--source csv` (default: `data/topic_classifications.csv`).
- **Plot**: optional `--output path.pdf` (or `.png`). Without `--output`, opens an interactive matplotlib window.
- **Format breakdown** is always computed from the JSONL (`primary_route`), independent of `--source`.

```bash
python analysis/dataset_breakdown.py
python analysis/dataset_breakdown.py --source router --output analysis/breakdown_router.pdf
python analysis/dataset_breakdown.py --source csv --output analysis/breakdown_topics.pdf
python analysis/dataset_breakdown.py --source csv --classifications data/topic_classifications.csv
```

### `classify_topics.py`

- Regenerates the topic CSV from scratch each run (overwrites by default).

```bash
python analysis/classify_topics.py
python analysis/classify_topics.py --output data/topic_classifications.csv
```

---

## Generated artifacts

| Artifact | Produced by | Notes |
|----------|-------------|--------|
| `analysis/breakdown.pdf` (or similar) | `dataset_breakdown.py --output …` | Git-ignored if you add patterns in `.gitignore`; optional to commit. |
| `data/topic_classifications.csv` | `classify_topics.py` | One row per problem: `public`/`private`, numeric `id`, single `topic` label. |

---

## Dependencies

- **Shared**: `matplotlib`, `seaborn`, project imports (`config`, `inference.router`).
- **`dataset_breakdown.py`**: `--source csv` imports [`classify_topics.py`](classify_topics.py) only for `CANONICAL_TOPIC_ORDER` (topic label ordering).
- **`classify_topics.py`**: standard library only beyond project modules.

If plotting fails, ensure matplotlib has a backend (e.g. use `--output` for non-interactive PDF/PNG on headless machines).
