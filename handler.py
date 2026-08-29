#!/usr/bin/env python3
"""RunPod Serverless handler for ACE-Step 1.5 (models on Network Volume)."""

from __future__ import annotations

import base64
import os
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import runpod

MODELS_ROOT = Path(os.environ.get("ACESTEP_MODELS_DIR", "/runpod-volume/models/acestep"))
HF_REPO = os.environ.get("ACESTEP_HF_REPO", "ACE-Step/Ace-Step1.5")
DIT_NAME = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
# HF repo ACE-Step/Ace-Step1.5 currently ships 1.7B LM (no 0.6B folder).
LM_NAME = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")

_dit_handler = None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _real_weight_files(dit_dir: Path) -> list[Path]:
    """Ignore tiny HF LFS pointer stubs; turbo weights are multi-GB."""
    if not dit_dir.is_dir():
        return []
    out: list[Path] = []
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        for p in dit_dir.rglob(pat):
            try:
                if p.is_file() and p.stat().st_size > 1_000_000:
                    out.append(p)
            except OSError:
                continue
    return out


def ensure_volume_layout() -> Path:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    cache = Path("/runpod-volume/.cache/huggingface")
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    # Point ACE checkpoints dir at the network volume
    ckpt = MODELS_ROOT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("ACESTEP_CHECKPOINT_DIR", str(ckpt))
    return ckpt


def download_models(force: bool = False) -> dict[str, Any]:
    """Download turbo + vae + embedding + LM into network volume."""
    from huggingface_hub import snapshot_download

    ckpt = ensure_volume_layout()
    patterns = [
        f"{DIT_NAME}/*",
        "vae/*",
        "Qwen3-Embedding-0.6B/*",
        f"{LM_NAME}/*",
        "config.json",
    ]
    marker = ckpt / DIT_NAME / ".download_ok"
    if marker.exists() and not force:
        return {"status": "already_present", "models_dir": str(ckpt), "dit": DIT_NAME, "lm": LM_NAME}

    log(f"Downloading {HF_REPO} patterns={patterns} -> {ckpt}")
    t0 = time.time()
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(ckpt),
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )
    dit_dir = ckpt / DIT_NAME
    if not dit_dir.exists():
        for child in ckpt.rglob(DIT_NAME):
            if child.is_dir():
                log(f"Found nested {DIT_NAME} at {child}")
                break
    # Require real weights (>1MB), not HF LFS pointer stubs
    weight_hits = _real_weight_files(dit_dir)
    if not dit_dir.is_dir() or not weight_hits:
        sizes = {
            str(p): p.stat().st_size
            for p in dit_dir.rglob("*")
            if p.is_file()
        } if dit_dir.is_dir() else {}
        raise RuntimeError(
            f"Download incomplete: {dit_dir} missing real weights after snapshot_download "
            f"(patterns={patterns}, files={sizes})"
        )
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(str(time.time()), encoding="utf-8")
    elapsed = round(time.time() - t0, 1)
    total_gb = round(sum(p.stat().st_size for p in weight_hits) / (1024**3), 2)
    log(f"Download done in {elapsed}s files={len(weight_hits)} total_gb={total_gb}")
    return {
        "status": "downloaded",
        "models_dir": str(ckpt),
        "seconds": elapsed,
        "dit": DIT_NAME,
        "lm": LM_NAME,
        "weight_files": len(weight_hits),
        "weight_gb": total_gb,
    }


def get_dit_handler():
    global _dit_handler
    if _dit_handler is not None:
        return _dit_handler

    ensure_volume_layout()
    # Prefer volume checkpoints; fall back to baked /app paths
    project_root = Path(os.environ.get("ACESTEP_PROJECT_ROOT", "/runpod-volume/app/acestep-repo"))
    volume_ckpt = Path(os.environ["ACESTEP_CHECKPOINT_DIR"])
    if (volume_ckpt / DIT_NAME).exists():
        # Symlink into expected checkpoints location if package expects ./checkpoints
        app_ckpt = project_root / "checkpoints"
        if not app_ckpt.exists():
            try:
                app_ckpt.symlink_to(volume_ckpt, target_is_directory=True)
            except OSError:
                os.environ["ACESTEP_CHECKPOINT_DIR"] = str(volume_ckpt)

    log("Loading DiT...")
    t0 = time.time()
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    handler.initialize_service(
        project_root=str(project_root),
        config_path=DIT_NAME,
        device=os.environ.get("ACESTEP_DEVICE", "cuda"),
        offload_to_cpu=os.environ.get("ACESTEP_OFFLOAD", "0") == "1",
        offload_dit_to_cpu=os.environ.get("ACESTEP_OFFLOAD", "0") == "1",
        quantization=None,
    )
    _dit_handler = handler
    log(f"DiT ready in {time.time() - t0:.1f}s")
    return _dit_handler


def save_temp_audio(audio_base64: str, suffix: str = ".mp3") -> str:
    raw = base64.b64decode(audio_base64)
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.write(raw)
    tmp.close()
    return tmp.name


def run_cover(input_data: dict) -> dict:
    from acestep.inference import GenerationParams, GenerationConfig, generate_music

    dit = get_dit_handler()
    ref_b64 = input_data.get("reference_audio_base64") or input_data.get("src_audio_base64")
    if not ref_b64:
        return {"error": "reference_audio_base64 (or src_audio_base64) required for cover"}

    ref_path = save_temp_audio(ref_b64)
    prompt = input_data.get("prompt") or input_data.get("caption") or (
        "jazz trio cover arrangement, warm upright bass, soft brushed drums, "
        "intimate piano, clear natural vocals, same melody and song structure"
    )
    duration = float(input_data.get("audio_duration", input_data.get("duration", 60)))
    steps = int(input_data.get("inference_steps", 8))
    seed = input_data.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    audio_format = input_data.get("audio_format", "mp3")

    params = GenerationParams(
        task_type="cover",
        caption=prompt,
        lyrics=input_data.get("lyrics", "[Instrumental]") if input_data.get("instrumental") else input_data.get("lyrics", ""),
        duration=duration,
        inference_steps=steps,
        seed=int(seed),
        vocal_language=input_data.get("vocal_language", "unknown"),
        reference_audio=ref_path,
        audio_cover_strength=float(input_data.get("audio_cover_strength", 0.7)),
    )
    if input_data.get("cover_noise_strength") is not None:
        try:
            params.cover_noise_strength = float(input_data["cover_noise_strength"])
        except Exception:
            pass

    config = GenerationConfig(
        batch_size=1,
        audio_format=audio_format,
        seeds=[int(seed)],
        use_random_seed=False,
    )
    save_dir = tempfile.mkdtemp(prefix="acestep_")
    t0 = time.time()
    result = generate_music(
        dit_handler=dit,
        llm_handler=None,
        params=params,
        config=config,
        save_dir=save_dir,
    )
    elapsed_ms = int((time.time() - t0) * 1000)

    try:
        os.unlink(ref_path)
    except OSError:
        pass

    if not result or not getattr(result, "success", False) or not getattr(result, "audios", None):
        err = getattr(result, "error", None) if result else "No audio generated"
        return {"error": str(err)}

    out_path = result.audios[0].get("path", "")
    if not out_path or not Path(out_path).exists():
        return {"error": "No output audio file produced"}

    with open(out_path, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode("utf-8")

    # Persist a copy on volume for debugging
    out_dir = Path("/runpod-volume/outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / Path(out_path).name
    try:
        dest.write_bytes(Path(out_path).read_bytes())
    except OSError:
        dest = None

    return {
        "success": True,
        "task_type": "cover",
        "seed": int(seed),
        "inference_time_ms": elapsed_ms,
        "audio_base64": audio_b64,
        "audio_format": audio_format,
        "volume_output": str(dest) if dest else None,
    }


def handler(event: dict) -> dict:
    try:
        data = event.get("input") or {}
        action = (data.get("action") or data.get("task_type") or "cover").lower()

        if action in ("setup", "download", "health"):
            if action == "health":
                ensure_volume_layout()
                dit_dir = Path(os.environ["ACESTEP_CHECKPOINT_DIR"]) / DIT_NAME
                weight_hits = _real_weight_files(dit_dir)
                rp = Path("/runpod-volume")
                ws = Path("/workspace")
                return {
                    "status": "ok",
                    "volume": str(MODELS_ROOT),
                    "checkpoint_dir": os.environ.get("ACESTEP_CHECKPOINT_DIR"),
                    "dit_present": bool(weight_hits),
                    "dit_weight_files": len(weight_hits),
                    "dit_weight_gb": round(sum(p.stat().st_size for p in weight_hits) / (1024**3), 2)
                    if weight_hits
                    else 0,
                    "runpod_volume_entries": sorted(p.name for p in rp.iterdir())[:20]
                    if rp.is_dir()
                    else [],
                    "workspace_entries": sorted(p.name for p in ws.iterdir())[:20]
                    if ws.is_dir()
                    else [],
                    "cuda": _cuda_ok(),
                }
            info = download_models(force=bool(data.get("force")))
            return {"success": True, **info}

        if action == "cover":
            # Auto-download if missing
            ckpt = ensure_volume_layout()
            if not (ckpt / DIT_NAME).exists() and not (ckpt / DIT_NAME / ".download_ok").exists():
                download_models()
            return run_cover(data)

        if action == "text2music":
            data = {**data, "task_type": "text2music"}
            # Reuse cover path structure with GenerationParams text2music
            return {"error": "text2music not enabled in this worker yet; use action=cover"}

        return {"error": f"Unknown action/task_type: {action}. Use setup|health|cover"}
    except Exception as e:
        log(f"Handler error: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {"error": str(e)}


def _cuda_ok() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


if __name__ == "__main__":
    ensure_volume_layout()
    log("ACE-Step Serverless worker starting (volume-first models)")
    runpod.serverless.start({"handler": handler})
