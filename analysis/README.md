# Analysis scripts

Utilities for exploring the competition datasets (`data/public.jsonl`, `data/private.jsonl`) and generating report figures. Run everything from **`cse151b/final`** with the project environment:

```bash
cd cse151b/final
micromamba activate cse151b_competition
```

Paths to the JSONL files come from [`config.py`](../config.py) (via `.env` → `ROOT_DIR` / `STORAGE_DIR`).

---

## Scripts

| Script | Purpose |
|--------|---------|
| [`table_sft_dataset.py`](table_sft_dataset.py) | **Table (tab:sft-dataset)** — per-teacher public/private trace counts from `DISTILL_DIR` (same inputs as `distill/merge.py`). Public counts reflect current `judger`-verified traces. |
| [`table_results.py`](table_results.py) | **Table (tab:results)** — MCQ / free-form / overall public accuracy for all milestone experiments by re-scoring `RESULTS_DIR/public_*.csv` through `judger`. KSS comes from optional [`kaggle_scores.csv`](kaggle_scores.csv) (not computable locally). |
| [`plot_style.py`](plot_style.py) | Shared matplotlib figure sizes / typography (imported by the other plotting scripts). |
| [`plot_dataset_breakdown.py`](plot_dataset_breakdown.py) | Summarizes **answer format** (MCQ vs free-form single vs multi-`[ANS]`) using `primary_route` on the JSONL fields. **Topic** bars come from either (**default**) the inference router’s `topic_taxonomy.classify_problem` label (same 20-way scoring as the CSV pipeline) via [`RuleBasedRouter`](../inference/router.py), or from **`--source csv`** reading [`data/topic_classifications.csv`](../data/topic_classifications.csv) (offline run of `classify_topics.py`). Prints tables to stdout and optionally saves a **two-panel horizontal bar chart**. |
| [`plot_sft_grpo_training.py`](plot_sft_grpo_training.py) | Two-panel **SFT loss + GRPO mean reward** figure from `training_loss_history.csv` under `STORAGE_DIR/checkpoints/` (defaults to `scratchpaper/figs/sft_grpo_training.pdf`). GRPO panel shows a placeholder until RL reward logs exist. |
| [`classify_topics.py`](classify_topics.py) | Offline CLI: writes [`data/topic_classifications.csv`](../data/topic_classifications.csv) with columns `set`, `id`, `topic`. Scoring rules live in repo-root [`topic_taxonomy.py`](../topic_taxonomy.py). |

### `table_sft_dataset.py` / `table_results.py`

Recompute milestone report tables after a `judger` change or new inference CSVs. Pass **`--latex`** to print a copy-paste-ready `\begin{tabular}...\end{tabular}` block (caption/label stay in the `.tex` source).

```bash
cd cse151b/final
micromamba run -n cse151b_competition python analysis/table_sft_dataset.py
micromamba run -n cse151b_competition python analysis/table_sft_dataset.py --latex
micromamba run -n cse151b_competition python analysis/table_results.py
micromamba run -n cse151b_competition python analysis/table_results.py --latex
micromamba run -n cse151b_competition python analysis/table_results.py --workers 8
```

- **`--workers N`**: parallel judger processes (default `1`). One tqdm bar tracks all experiments × 1126 public questions.
- **`--quiet`**: disable the progress bar (e.g. when piping output).
- After the table, prints KSS sourcing notes and absolute paths to each experiment's **private** submission CSV for Kaggle upload (`private_*.csv` under `RESULTS_DIR`).

Update [`kaggle_scores.csv`](kaggle_scores.csv) when you get new Kaggle subsample scores; pass `--no-kss` to omit that column.

### `plot_dataset_breakdown.py`

- **`--source router`** (default): topic bars = one label per problem from `RuleBasedRouter` / `topic_taxonomy` (counts sum to *n* per split, same as CSV mode).
- **`--source csv`**: topic bars = labels from `topic_classifications.csv`. Requires the CSV — run `classify_topics.py` first.
- **`--classifications PATH`**: CSV path when using `--source csv` (default: `data/topic_classifications.csv`).
- **`--plot-top N`**: figure only — show the **`N` topics with the largest counts on the private split** (ties broken alphabetically). `0` = show every topic that appears in either split. Stdout tables still list all topics; omitted topics are summarized after the tables when `N > 0`.
- **Format breakdown** is always computed from the JSONL (`primary_route`), independent of `--source`.

```bash
cd cse151b/final
micromamba run -n cse151b_competition python analysis/plot_dataset_breakdown.py
micromamba run -n cse151b_competition python analysis/plot_dataset_breakdown.py \
    --source router --plot-top 10 --output scratchpaper/figs/breakdown.pdf
micromamba run -n cse151b_competition python analysis/plot_dataset_breakdown.py \
    --source csv --plot-top 10 --output analysis/breakdown_topics.pdf
micromamba run -n cse151b_competition python analysis/plot_dataset_breakdown.py \
    --source csv --classifications data/topic_classifications.csv
```

### `plot_sft_grpo_training.py`

- **`--sft-csv` / `--rl-csv`**: override checkpoint history CSV paths (defaults: `STORAGE_DIR/checkpoints/sft|rl/training_loss_history.csv`).
- **`--output`**: PDF path (default: `scratchpaper/figs/sft_grpo_training.pdf`).

```bash
cd cse151b/final
micromamba run -n cse151b_competition python analysis/plot_sft_grpo_training.py
micromamba run -n cse151b_competition python analysis/plot_sft_grpo_training.py \
    --output scratchpaper/figs/sft_grpo_training.pdf
```

Re-run after GRPO training progresses so the right panel includes reward data.

### `classify_topics.py`

- Regenerates the topic CSV from scratch each run (overwrites by default). Uses `topic_taxonomy.classify_problem` so question + option text match inference routing.

```bash
cd cse151b/final
micromamba run -n cse151b_competition python analysis/classify_topics.py
micromamba run -n cse151b_competition python analysis/classify_topics.py \
    --output data/topic_classifications.csv
```

---

## Generated artifacts

| Artifact | Produced by | Notes |
|----------|-------------|-------|
| `scratchpaper/figs/breakdown.pdf` | `plot_dataset_breakdown.py --output …` | Report topic/format figure. |
| `scratchpaper/figs/sft_grpo_training.pdf` | `plot_sft_grpo_training.py --output …` | Report SFT loss + GRPO reward figure. |
| `analysis/breakdown_topics.pdf` (or similar) | `plot_dataset_breakdown.py --source csv --output …` | Optional CSV-mode breakdown. |
| `data/topic_classifications.csv` | `classify_topics.py` | One row per problem: `public`/`private`, numeric `id`, single `topic` label. |

---

## Dependencies

- **Shared**: `matplotlib`, `seaborn`, project imports (`config`, `inference.router`, `topic_taxonomy`).
- **`plot_dataset_breakdown.py`**: imports `CANONICAL_TOPIC_ORDER` from `topic_taxonomy` for topic label ordering.
- **`plot_sft_grpo_training.py`**: `matplotlib`, `seaborn`, `config.CHECKPOINTS_DIR`.
- **`classify_topics.py`**: standard library + `topic_taxonomy`, `config`.

If plotting fails, ensure matplotlib has a backend (e.g. use `--output` for non-interactive PDF/PNG on headless machines).
