# Lightweight ACE-Step Serverless worker — models live on Network Volume (/runpod-volume)
# NOTE: ACE-Step's pyproject uses uv path sources for nano-vllm; plain `pip install -e .` fails.
# We install nano-vllm from path first, then ace-step --no-deps, then inference deps.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    ACESTEP_PROJECT_ROOT=/app/acestep-repo \
    ACESTEP_MODELS_DIR=/runpod-volume/models/acestep \
    ACESTEP_CHECKPOINT_DIR=/runpod-volume/models/acestep/checkpoints \
    ACESTEP_CONFIG_PATH=acestep-v15-turbo \
    ACESTEP_LM_MODEL_PATH=acestep-5Hz-lm-0.6B \
    ACESTEP_DEVICE=cuda \
    ACESTEP_OFFLOAD=0 \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends git ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# ACE-Step code only (no model weights in image)
RUN git clone --depth 1 https://github.com/ACE-Step/ACE-Step-1.5.git /app/acestep-repo \
    && pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -e /app/acestep-repo/acestep/third_parts/nano-vllm \
    && pip install --no-cache-dir -e /app/acestep-repo --no-deps \
    && pip install --no-cache-dir \
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

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
