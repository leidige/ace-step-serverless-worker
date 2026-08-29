#!/usr/bin/env bash
# Bootstrap ACE-Step serverless worker on a public base image (no private GHCR needed).
# Installs + marker live on the network volume so cold starts after the first are fast.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
export ACESTEP_PROJECT_ROOT="${ACESTEP_PROJECT_ROOT:-/runpod-volume/app/acestep-repo}"
export ACESTEP_VENV="${ACESTEP_VENV:-/runpod-volume/app/venv}"
export ACESTEP_MODELS_DIR="${ACESTEP_MODELS_DIR:-/runpod-volume/models/acestep}"
export ACESTEP_CHECKPOINT_DIR="${ACESTEP_CHECKPOINT_DIR:-/runpod-volume/models/acestep/checkpoints}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export ACESTEP_OFFLOAD="${ACESTEP_OFFLOAD:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export PATH="${ACESTEP_VENV}/bin:${PATH}"

MARKER=/runpod-volume/app/.ace_bootstrap_ok
HANDLER_DST=/runpod-volume/app/handler.py
mkdir -p /runpod-volume/app /runpod-volume/models/acestep/checkpoints /app

log() { echo "[ace-bootstrap] $*" >&2; }

need_apt=0
command -v git >/dev/null 2>&1 || need_apt=1
command -v ffmpeg >/dev/null 2>&1 || need_apt=1
command -v curl >/dev/null 2>&1 || need_apt=1
if [ "$need_apt" = "1" ]; then
  log "Installing system packages (git/ffmpeg/curl)..."
  apt-get update
  apt-get install -y --no-install-recommends git ffmpeg curl ca-certificates
  rm -rf /var/lib/apt/lists/*
fi

if [ ! -f "$MARKER" ] || [ ! -x "${ACESTEP_VENV}/bin/python" ]; then
  log "Cold bootstrap on network volume (first time or incomplete)..."
  if [ ! -d "$ACESTEP_PROJECT_ROOT/.git" ]; then
    log "Cloning ACE-Step-1.5..."
    git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git "$ACESTEP_PROJECT_ROOT"
  fi

  if [ ! -x "${ACESTEP_VENV}/bin/python" ]; then
    log "Creating venv at ${ACESTEP_VENV}..."
    python3 -m venv --system-site-packages "$ACESTEP_VENV"
  fi
  # shellcheck disable=SC1091
  source "${ACESTEP_VENV}/bin/activate"

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

  curl -fsSL -o "$HANDLER_DST" \
    "https://raw.githubusercontent.com/leidige/ace-step-serverless-worker/master/handler.py"
  cp -f "$HANDLER_DST" /app/handler.py
  touch "$MARKER"
  log "Bootstrap complete."
else
  log "Reusing volume bootstrap (marker present)."
  # shellcheck disable=SC1091
  source "${ACESTEP_VENV}/bin/activate" || true
  # Always refresh handler from GitHub so hotfixes apply without wiping the venv
  curl -fsSL -o /app/handler.py \
    "https://raw.githubusercontent.com/leidige/ace-step-serverless-worker/master/handler.py"
  cp -f /app/handler.py "$HANDLER_DST" || true
fi

export PYTHONPATH="${ACESTEP_PROJECT_ROOT}:${PYTHONPATH:-}"
exec "${ACESTEP_VENV}/bin/python" -u /app/handler.py
