#!/usr/bin/env bash
# Bootstrap ACE-Step serverless worker on a public base image (no private GHCR needed).
# Installs + marker live on the network volume so cold starts after the first are fast.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1

log() { echo "[ace-bootstrap] $*" >&2; }

# Detect network volume mount: serverless often /runpod-volume; some templates use /workspace.
detect_root() {
  for d in /runpod-volume /workspace; do
    if [ -d "$d" ] && touch "$d/.ace_vol_probe" 2>/dev/null; then
      rm -f "$d/.ace_vol_probe"
      echo "$d"
      return 0
    fi
  done
  mkdir -p /runpod-volume
  echo /runpod-volume
}

VOL_ROOT="$(detect_root)"
export ACESTEP_VOLUME_ROOT="$VOL_ROOT"
log "Using volume root: $VOL_ROOT"

export ACESTEP_PROJECT_ROOT="${ACESTEP_PROJECT_ROOT:-$VOL_ROOT/app/acestep-repo}"
export ACESTEP_VENV="${ACESTEP_VENV:-$VOL_ROOT/app/venv}"
export ACESTEP_MODELS_DIR="${ACESTEP_MODELS_DIR:-$VOL_ROOT/models/acestep}"
export ACESTEP_CHECKPOINT_DIR="${ACESTEP_CHECKPOINT_DIR:-$VOL_ROOT/models/acestep/checkpoints}"
export ACESTEP_CONFIG_PATH="${ACESTEP_CONFIG_PATH:-acestep-v15-turbo}"
export ACESTEP_LM_MODEL_PATH="${ACESTEP_LM_MODEL_PATH:-acestep-5Hz-lm-1.7B}"
export ACESTEP_DEVICE="${ACESTEP_DEVICE:-cuda}"
export ACESTEP_OFFLOAD="${ACESTEP_OFFLOAD:-0}"
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-$VOL_ROOT/.cache/huggingface}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export PATH="${ACESTEP_VENV}/bin:${PATH}"

# v6: torchao pin 0.5.0 (newer needs torch.int1 / torch 2.5+) + diffusers pin for torch 2.4
BOOT_REV="v6-torchao05-diffusers"
MARKER="$VOL_ROOT/app/.ace_bootstrap_ok"
HANDLER_DST="$VOL_ROOT/app/handler.py"
mkdir -p "$VOL_ROOT/app" "$VOL_ROOT/models/acestep/checkpoints" /app "$HF_HOME"

# Keep /runpod-volume path usable even when mount is /workspace
if [ "$VOL_ROOT" = "/workspace" ]; then
  mkdir -p /runpod-volume
  ln -sfn /workspace/app /runpod-volume/app 2>/dev/null || true
  ln -sfn /workspace/models /runpod-volume/models 2>/dev/null || true
  ln -sfn /workspace/.cache /runpod-volume/.cache 2>/dev/null || true
fi

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

NEED_FULL=0
if [ ! -f "$MARKER" ] || [ ! -x "${ACESTEP_VENV}/bin/python" ]; then
  NEED_FULL=1
elif [ "$(cat "$MARKER" 2>/dev/null || true)" != "$BOOT_REV" ]; then
  log "Bootstrap rev mismatch — will repair deps without full wipe."
fi

if [ "$NEED_FULL" = "1" ]; then
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
    "diffusers>=0.32.0,<0.35.0" \
    "matplotlib>=3.7.5" \
    "scipy>=1.10.1" \
    "soundfile>=0.13.1" \
    "loguru>=0.7.3" \
    "einops>=0.8.1" \
    "accelerate>=1.12.0" \
    diskcache \
    "numba>=0.63.1" \
    "vector-quantize-pytorch>=1.27.15" \
    "torchao==0.5.0" \
    toml \
    modelscope \
    "peft>=0.18.0" \
    xxhash \
    "typer-slim>=0.21.1" \
    "runpod>=1.7.0" \
    "huggingface_hub>=0.25.0" \
    "hf_transfer>=0.1.0"
else
  log "Reusing volume bootstrap (marker present)."
  # shellcheck disable=SC1091
  source "${ACESTEP_VENV}/bin/activate" || true
fi

# Always refresh handler from GitHub
curl -fsSL -o /app/handler.py \
  "https://raw.githubusercontent.com/leidige/ace-step-serverless-worker/master/handler.py"
cp -f /app/handler.py "$HANDLER_DST" || true

# shellcheck disable=SC1091
source "${ACESTEP_VENV}/bin/activate" || true

# Fix broken torchaudio that shadows the base image CUDA build:
# "Could not load .../torchaudio/lib/libtorchaudio.so"
fix_torchaudio() {
  if python - <<'PY' >/dev/null 2>&1
import torchaudio
print("ok", torchaudio.__version__)
PY
  then
    log "torchaudio OK"
    return 0
  fi
  log "Repairing torchaudio..."
  pip uninstall -y torchaudio 2>/dev/null || true
  python - <<'PY'
import glob, os, shutil, site
for sp in site.getsitepackages():
    for p in glob.glob(os.path.join(sp, "torchaudio*")):
        print("rm", p, flush=True)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass
PY
  TORCH_VER="$(python -c "import torch; print(torch.__version__.split('+')[0])" 2>/dev/null || echo "2.4.0")"
  pip install --no-cache-dir "torchaudio==${TORCH_VER}" \
    --index-url https://download.pytorch.org/whl/cu124 \
    || pip install --no-cache-dir "torchaudio==2.4.1" --index-url https://download.pytorch.org/whl/cu124 \
    || true
  if python - <<'PY' >/dev/null 2>&1
import torchaudio
print(torchaudio.__version__)
PY
  then
    log "torchaudio repaired"
  else
    log "WARNING: torchaudio still broken after repair"
  fi
}

# transformers eagerly imports torchao; torchao>=0.7 needs torch.int1 (torch>=2.5).
# Base image is torch 2.4.1 — pin 0.5.0 so DiT load does not crash.
fix_torchao() {
  if python - <<'PY' >/dev/null 2>&1
import torch
import torchao
from torchao.quantization import quant_primitives  # noqa: F401
print("ok", getattr(torchao, "__version__", "?"))
PY
  then
    log "torchao OK"
    return 0
  fi
  log "Repairing torchao → 0.5.0 (torch 2.4 compat)..."
  pip uninstall -y torchao 2>/dev/null || true
  python - <<'PY'
import glob, os, shutil, site
for sp in site.getsitepackages():
    for p in glob.glob(os.path.join(sp, "torchao*")):
        print("rm", p, flush=True)
        if os.path.isdir(p):
            shutil.rmtree(p, ignore_errors=True)
        else:
            try:
                os.remove(p)
            except OSError:
                pass
PY
  pip install --no-cache-dir "torchao==0.5.0" || true
  if python - <<'PY' >/dev/null 2>&1
import torchao
from torchao.quantization import quant_primitives  # noqa: F401
print(torchao.__version__)
PY
  then
    log "torchao repaired"
  else
    log "WARNING: torchao still broken after repair"
  fi
}

fix_diffusers() {
  # Newest diffusers registers FA3 custom_ops with from __future__ annotations;
  # torch 2.4 infer_schema rejects that → VAE AutoencoderOobleck import dies.
  if python - <<'PY' >/dev/null 2>&1
from diffusers.models import AutoencoderOobleck  # noqa: F401
print("ok")
PY
  then
    log "diffusers AutoencoderOobleck OK"
    return 0
  fi
  log "Repairing diffusers for torch 2.4..."
  python - <<'PY'
import sys
for k in list(sys.modules):
    if k == "diffusers" or k.startswith("diffusers."):
        del sys.modules[k]
print("cleared diffusers modules", flush=True)
PY
  pip uninstall -y diffusers 2>/dev/null || true
  pip install --no-cache-dir "diffusers==0.34.0" || pip install --no-cache-dir "diffusers==0.33.1" || true
  python - <<'PY'
import sys
for k in list(sys.modules):
    if k == "diffusers" or k.startswith("diffusers."):
        del sys.modules[k]
PY
  if python - <<'PY' >/dev/null 2>&1
from diffusers.models import AutoencoderOobleck  # noqa: F401
print("ok")
PY
  then
    log "diffusers repaired"
  else
    log "WARNING: diffusers still broken after repair — handler will try custom_op patch"
  fi
}
fix_torchaudio
fix_torchao
fix_diffusers
echo "$BOOT_REV" > "$MARKER"

export PYTHONPATH="${ACESTEP_PROJECT_ROOT}:${PYTHONPATH:-}"
exec "${ACESTEP_VENV}/bin/python" -u /app/handler.py
