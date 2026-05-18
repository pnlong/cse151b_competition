#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — create and populate the cse151b_competition micromamba environment
#
# Usage (from the repo root):
#   bash setup.sh
#
# What it does:
#   1. Creates cse151b_competition (Python 3.11) only if it does not already exist —
#      never removes or recreates an existing env.
#   2. Installs PyTorch 2.x compiled for CUDA 12.4
#      (the driver supports CUDA 13.1, which is backwards-compatible)
#   3. Installs vLLM and all project dependencies
#   4. Initialises the storage directory layout defined in .env
#
# All pip installs below target ONLY this env via:
#     micromamba run -n cse151b_competition pip install ...
# which is equivalent to:
#     micromamba activate cse151b_competition
#     pip install ...
#
# After running, activate with:
#   micromamba activate cse151b_competition
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="cse151b_competition"
PYTHON_VERSION="3.11"
CUDA_TAG="cu124"   # PyTorch CUDA 12.4 build — compatible with driver CUDA 13.1

if micromamba run -n "${ENV_NAME}" python --version &>/dev/null; then
    echo "==> Micromamba env '${ENV_NAME}' already exists — skipping create (pip installs still run)."
else
    echo "==> Creating micromamba environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
    micromamba create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi

echo ""
echo "==> Installing PyTorch (${CUDA_TAG})..."
micromamba run -n "${ENV_NAME}" pip install \
    torch torchvision torchaudio \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

echo ""
echo "==> Installing vLLM..."
micromamba run -n "${ENV_NAME}" pip install vllm

echo ""
echo "==> Installing project dependencies..."
micromamba run -n "${ENV_NAME}" pip install \
    transformers \
    tqdm \
    python-dotenv \
    sympy \
    numpy \
    "antlr4-python3-runtime==4.11.1" \
    bitsandbytes \
    accelerate \
    trl \
    peft \
    datasets \
    matplotlib

echo ""
echo "==> Initialising storage directories..."
micromamba run -n "${ENV_NAME}" python3 -c "
import sys; sys.path.insert(0, '.')
from config import ensure_storage_dirs
ensure_storage_dirs()
print('Storage dirs OK')
"

echo ""
echo "Done. Activate with:"
echo "  micromamba activate ${ENV_NAME}"
echo ""
echo "If you add or upgrade SFT packages later, install into THIS env only:"
echo "  micromamba activate ${ENV_NAME}"
echo "  pip install trl peft datasets matplotlib"
