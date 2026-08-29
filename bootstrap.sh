#!/usr/bin/env bash
# Bootstrap ACE-Step serverless worker on a public base image (no private GHCR needed).
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export ACESTEP_PROJECT_ROOT="${ACESTEP_PROJECT_ROOT:-/app/acestep-repo}"
export ACESTEP_MODELS_DIR="${ACESTEP_MODELS_DIR:-/runpod-volume/models/acestep}"
export ACESTEP_CHECKPOINT_DIR="${ACESTEP_CHECKPOINT_DIR:-/runpod-volume/models/acestep/checkpoints}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-0.6B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export ACESTEP_OFFLOAD="${ACESTEP_OFFLOAD:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"

MARKER=/app/.ace_bootstrap_ok
mkdir -p /app /runpod-volume/models/acestep/checkpoints

if [ ! -f "$MARKER" ]; then
  apt-get update
  apt-get install -y --no-install-recommends git ffmpeg curl ca-certificates
  rm -rf /var/lib/apt/lists/*

  if [ ! -d "$ACESTEP_PROJECT_ROOT/.git" ]; then
    git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git "$ACESTEP_PROJECT_ROOT"
  fi

  pip install --no-cache-dir --upgrade pip
  pip install --no-cache-dir -e "$ACESTEP_PROJECT_ROOT/acestep/third_parts/nano-vllm"
  pip install --no-cache-dir -e "$ACESTEP_PROJECT_ROOT" --no-deps
  pip install --no-cache-dir \
    "transformers>=4.51.0,<4.58.0" \
    diffusers \
    "matplotlib>=3.7.5" \
    "scipy>=1.10.1" \
    "soundfile>=0.13.1" \
    "loguru>=0.7.3" \
    "einops>=0.8.1" \
    "accelerate>=1.12.0" \
    diskcache \
    "numba>=0.63.1" \
    "vector-quantize-pytorch>=1.27.15" \
    torchao \
    toml \
    modelscope \
    "peft>=0.18.0" \
    xxhash \
    "typer-slim>=0.21.1" \
    "runpod>=1.7.0" \
    "huggingface_hub>=0.25.0" \
    "hf_transfer>=0.1.0"

  curl -fsSL -o /app/handler.py \
    "https://raw.githubusercontent.com/leidige/ace-step-serverless-worker/master/handler.py"
  touch "$MARKER"
fi

exec python -u /app/handler.py
