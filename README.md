# CSE 151B Spring 2026 — Math Reasoning Competition

Fine-tuned **Qwen3-4B** (GRPO **`checkpoint-best-reward`**, router, **`N=8`**, no quantization). Full spec: [`docs/directions.md`](docs/directions.md).

---

## Reproduce our submission (staff)

**Adapter on Hugging Face:** **`p1long/cse151b_competition`** — https://huggingface.co/p1long/cse151b_competition  

`run_inference` reads **`adapter_config.json`**, pulls the adapter from Hugging Face, loads the **base** named there from Hub, and applies our LoRA. Sampling defaults are in [`constants.py`](constants.py).

**Weights:** **`SUBMISSION_MODEL`** must reference our Hub repo (step 4 below). Adapters and base weights cache under **`HF_CACHE_DIR`** (`config` / `.env`).

```bash
# 1. Clone this GitHub repo and cd into the root (where setup.sh and run_inference.py live).

# 2. Paths + token (gated models need HF_TOKEN — use a read token if unsure)
cp .env.example .env
# Edit .env: ROOT_DIR = absolute path to this clone; STORAGE_DIR = fast disk for cache + outputs + CSV

# 3. Environment (micromamba + PyTorch cu124 + vLLM + deps)
bash setup.sh
micromamba activate cse151b_competition

# 4. Our weights (fixed id — no guesswork)
export SUBMISSION_MODEL=p1long/cse151b_competition

# 5. Inference → ${STORAGE_DIR}/results/submission.csv
CUDA_VISIBLE_DEVICES=0 python run_inference.py
```

**Call `run_inference()` from Python** (same pipeline and defaults as the CLI):

```python
from run_inference import run_inference

run_inference()  # uses SUBMISSION_MODEL from the bash block above
# or equivalently:
# run_inference(model="p1long/cse151b_competition")
```

Default **input / output**: private set **`config.PRIVATE_DATA`** → **`config.RESULTS_DIR / "submission.csv"`** (i.e. under **`${STORAGE_DIR}`** from `.env`).

Optional: **`export SUBMISSION_MODEL_REVISION=<git_sha>`** to pin a Hub commit.

| | |
|--|--|
| **GPU we used** | NVIDIA GeForce RTX 3090 |
| **Wall-clock** | ~12 h on full **`data/private.jsonl`** |

**Gradescope:** public **GitHub** URL + all teammates on the roster (no weight upload there).

---

## Notes

- **`run_inference.py`** is the required entry point; it matches **`inference/infer.py`** logic. Recipe **§3c**: [`docs/experiments.md`](docs/experiments.md).
- **`HF_HOME`** / cache: **`HF_CACHE_DIR`** in **`.env`** (see [`config.py`](config.py)).
- **`infer_parallel.py`** + flags: [`inference/README.md`](inference/README.md).

```
├── run_inference.py   # submission pipeline
├── setup.sh           # micromamba env cse151b_competition
├── config.py, constants.py, inference/, rl/, sft/, docs/, data/
```
