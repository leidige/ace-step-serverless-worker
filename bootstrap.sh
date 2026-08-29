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

# v7-hf-hub: diffusers 0.32.2 needs huggingface_hub>=0.24 (volume often stuck on 0.23.0).
# Prior v6 pinned 0.32.2 but repair only reinstalled diffusers → Oobleck import still FATAL.
BOOT_REV="v7-hf-hub"
export ACESTEP_BOOT_REV="$BOOT_REV"
MARKER="$VOL_ROOT/app/.ace_bootstrap_ok"
DIFFUSERS_FAIL="$VOL_ROOT/app/.ace_diffusers_fail"
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
    "diffusers==0.32.2" \
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
    "huggingface_hub>=0.24.0,<1.0" \
    "hf_transfer>=0.1.0"
else
  log "Reusing volume bootstrap (marker present)."
  # shellcheck disable=SC1091
  source "${ACESTEP_VENV}/bin/activate" || true
fi

# Always refresh handler from GitHub (cache-bust query so CDN cannot serve stale handler.py)
HANDLER_URL="https://raw.githubusercontent.com/leidige/ace-step-serverless-worker/master/handler.py?t=${BOOT_REV}"
log "Fetching handler: $HANDLER_URL"
curl -fsSL -H "Cache-Control: no-cache" -H "Pragma: no-cache" -o /app/handler.py "$HANDLER_URL"
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

_clear_diffusers_modules() {
  python - <<'PY'
import sys
prefixes = (
    "diffusers",
    "huggingface_hub",
    "transformers",
    "tokenizers",
)
for k in list(sys.modules):
    for p in prefixes:
        if k == p or k.startswith(p + "."):
            del sys.modules[k]
            break
print("cleared diffusers/hf_hub/transformers modules", flush=True)
PY
}

_verify_diffusers_oobleck() {
  python - <<'PY'
import importlib.util
import sys

import huggingface_hub
hf_ver = getattr(huggingface_hub, "__version__", "?")
print("huggingface_hub", hf_ver, flush=True)
# diffusers 0.32.x requires huggingface-hub>=0.24.0,<1.0
parts = []
for x in str(hf_ver).split("."):
    if x.isdigit():
        parts.append(int(x))
    else:
        break
maj = parts[0] if parts else -1
minr = parts[1] if len(parts) > 1 else 0
if maj != 0 or minr < 24:
    raise SystemExit(f"huggingface_hub too old for diffusers 0.32.2: {hf_ver}")

# Prove pin: 0.32.2 has AutoencoderOobleck; must NOT pull FA3 ace_step custom_op path.
import diffusers
ver = getattr(diffusers, "__version__", "?")
print("diffusers", ver, flush=True)
if ver != "0.32.2":
    raise SystemExit(f"expected diffusers==0.32.2 got {ver}")

# Hard fail if ace_step_transformer is importable as a side-effect of models.__init__
# (0.34+ registers FA3 @_custom_op that breaks torch 2.4 infer_schema).
from diffusers.models import AutoencoderOobleck  # noqa: F401
print("AutoencoderOobleck OK", flush=True)

spec = importlib.util.find_spec("diffusers.models.transformers.transformer_ace_step")
if spec is not None:
    # Module file may exist in newer trees; importing it is the crash. Ensure we did not.
    if "diffusers.models.transformers.transformer_ace_step" in sys.modules:
        raise SystemExit("ace_step transformer already imported — wrong diffusers pin")
print("ace_step_transformer not loaded — OK", flush=True)
PY
}

fix_diffusers() {
  # Newest diffusers (0.34+) registers FA3 custom_ops with from __future__ annotations;
  # torch 2.4 infer_schema rejects that → VAE AutoencoderOobleck import dies.
  # ALWAYS force 0.32.2 + huggingface_hub>=0.24 (volume often has hub 0.23.0).
  rm -f "$DIFFUSERS_FAIL" 2>/dev/null || true
  CUR="$(python -c "import diffusers; print(diffusers.__version__)" 2>/dev/null || echo none)"
  HF_CUR="$(python -c "import huggingface_hub; print(huggingface_hub.__version__)" 2>/dev/null || echo none)"
  if [ "$CUR" = "0.32.2" ] && _verify_diffusers_oobleck >/dev/null 2>&1; then
    log "diffusers $CUR + huggingface_hub $HF_CUR AutoencoderOobleck OK"
    return 0
  fi
  log "Force-pinning diffusers==0.32.2 + huggingface_hub>=0.24 + transformers (have diffusers=$CUR hub=$HF_CUR)..."
  _clear_diffusers_modules || true
  pip uninstall -y diffusers 2>/dev/null || true
  pip install --no-cache-dir --upgrade \
    "huggingface_hub>=0.24.0,<1.0" \
    "transformers>=4.51.0,<4.58.0"
  pip install --force-reinstall --no-cache-dir "diffusers==0.32.2"
  # Re-assert hub/transformers after force-reinstall (pip may have pulled a dep conflict)
  pip install --no-cache-dir --upgrade \
    "huggingface_hub>=0.24.0,<1.0" \
    "transformers>=4.51.0,<4.58.0"
  _clear_diffusers_modules || true
  if _verify_diffusers_oobleck; then
    log "diffusers repaired → 0.32.2 (hub/transformers pinned)"
    return 0
  fi
  # Do NOT exit 1 — crash-loop burns GPU $. Start handler so health can report clearly.
  ERR_MSG="$( _verify_diffusers_oobleck 2>&1 || true )"
  log "WARNING: diffusers repair incomplete — starting handler for health error (not FATAL exit)"
  printf '%s\n' "$ERR_MSG" > "$DIFFUSERS_FAIL" || true
  return 1
}

fix_torchaudio
fix_torchao
if fix_diffusers; then
  echo "$BOOT_REV" > "$MARKER"
  log "BOOT_REV=$BOOT_REV written to marker"
else
  rm -f "$MARKER" 2>/dev/null || true
  log "BOOT marker NOT written — next cold start will re-repair"
fi

export PYTHONPATH="${ACESTEP_PROJECT_ROOT}:${PYTHONPATH:-}"
export ACESTEP_BOOT_REV="$BOOT_REV"
exec "${ACESTEP_VENV}/bin/python" -u /app/handler.py
