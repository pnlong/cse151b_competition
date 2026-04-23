#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# setup.sh — create and populate the cse151b_competition micromamba environment
#
# Usage (from the repo root):
#   bash setup.sh
#
# What it does:
#   1. Creates a micromamba env named cse151b_competition (Python 3.11)
#   2. Installs PyTorch 2.x compiled for CUDA 12.4
#      (the driver supports CUDA 13.1, which is backwards-compatible)
#   3. Installs vLLM and all project dependencies
#   4. Initialises the storage directory layout defined in .env
#
# After running, activate with:
#   micromamba activate cse151b_competition
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

ENV_NAME="cse151b_competition"
PYTHON_VERSION="3.11"
CUDA_TAG="cu124"   # PyTorch CUDA 12.4 build — compatible with driver CUDA 13.1

echo "==> Creating micromamba environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
micromamba create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y

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
    accelerate

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
