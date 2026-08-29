#!/usr/bin/env python3
"""RunPod Serverless handler for ACE-Step 1.5 (models on Network Volume)."""

from __future__ import annotations

import base64
import os
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import runpod

MODELS_ROOT = Path(os.environ.get("ACESTEP_MODELS_DIR", "") or "")
HF_REPO = os.environ.get("ACESTEP_HF_REPO", "ACE-Step/Ace-Step1.5")
DIT_NAME = os.environ.get("ACESTEP_CONFIG_PATH", "acestep-v15-turbo")
# HF repo ACE-Step/Ace-Step1.5 currently ships 1.7B LM (no 0.6B folder).
LM_NAME = os.environ.get("ACESTEP_LM_MODEL_PATH", "acestep-5Hz-lm-1.7B")
EMBED_NAME = "Qwen3-Embedding-0.6B"
VAE_NAME = "vae"

# Must match acestep.model_downloader.MAIN_MODEL_COMPONENTS
MAIN_COMPONENTS = [DIT_NAME, VAE_NAME, EMBED_NAME, LM_NAME]

_dit_handler = None


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def detect_volume_root() -> Path:
    """Serverless usually /runpod-volume; some templates mount network volume at /workspace."""
    preferred = os.environ.get("ACESTEP_VOLUME_ROOT", "").strip()
    candidates: list[Path] = []
    if preferred:
        candidates.append(Path(preferred))
    candidates.extend([Path("/runpod-volume"), Path("/workspace")])
    seen: set[str] = set()
    for root in candidates:
        key = str(root)
        if key in seen:
            continue
        seen.add(key)
        if not root.is_dir():
            continue
        probe = root / ".ace_vol_probe"
        try:
            probe.write_text("ok", encoding="utf-8")
            probe.unlink(missing_ok=True)
            return root
        except OSError:
            continue
    return Path("/runpod-volume")


def bind_volume_paths() -> Path:
    global MODELS_ROOT
    root = detect_volume_root()
    if not os.environ.get("ACESTEP_MODELS_DIR"):
        MODELS_ROOT = root / "models" / "acestep"
    elif not MODELS_ROOT.parts:
        MODELS_ROOT = Path(os.environ["ACESTEP_MODELS_DIR"])
    os.environ.setdefault("ACESTEP_VOLUME_ROOT", str(root))
    os.environ.setdefault("ACESTEP_MODELS_DIR", str(MODELS_ROOT))
    return root


def _real_weight_files(component_dir: Path) -> list[Path]:
    """Ignore tiny HF LFS pointer stubs; real weights are multi-MB+."""
    if not component_dir.is_dir():
        return []
    out: list[Path] = []
    for pat in ("*.safetensors", "*.bin", "*.pt"):
        for p in component_dir.rglob(pat):
            try:
                if p.is_file() and p.stat().st_size > 1_000_000:
                    out.append(p)
            except OSError:
                continue
    return out


def _component_report(ckpt: Path, name: str) -> dict[str, Any]:
    d = ckpt / name
    weights = _real_weight_files(d)
    total = sum(p.stat().st_size for p in weights) if weights else 0
    return {
        "present": bool(weights),
        "dir_exists": d.is_dir(),
        "weight_files": len(weights),
        "weight_gb": round(total / (1024**3), 3) if total else 0,
        "path": str(d),
    }


def ensure_volume_layout() -> Path:
    root = bind_volume_paths()
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    cache = root / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(cache / "hub")
    os.environ["HF_HUB_CACHE"] = str(cache / "hub")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "transformers")
    ckpt = MODELS_ROOT / "checkpoints"
    ckpt.mkdir(parents=True, exist_ok=True)
    os.environ["ACESTEP_CHECKPOINT_DIR"] = str(ckpt)
    return ckpt


def resolve_ace_package_root() -> Path | None:
    """Where AceStepHandler._get_project_root() will resolve (editable install root)."""
    try:
        import acestep

        # acestep/__init__.py -> parent = package dir -> parent = project root
        return Path(acestep.__file__).resolve().parent.parent
    except Exception as e:
        log(f"resolve_ace_package_root failed: {e}")
        return None


def ensure_ace_checkpoints_link(volume_ckpt: Path) -> dict[str, Any]:
    """
    AceStepHandler.initialize_service IGNORES passed project_root and uses
    self._get_project_root()/checkpoints. Bind that path to our volume checkpoints.
    """
    info: dict[str, Any] = {
        "volume_ckpt": str(volume_ckpt),
        "ace_root": None,
        "link": None,
        "ok": False,
    }
    ace_root = resolve_ace_package_root()
    if ace_root is None:
        env_root = os.environ.get("ACESTEP_PROJECT_ROOT", "").strip()
        ace_root = Path(env_root) if env_root else None
    if ace_root is None:
        info["error"] = "cannot resolve ACE package root"
        return info

    info["ace_root"] = str(ace_root)
    app_ckpt = ace_root / "checkpoints"
    vol_resolved = volume_ckpt.resolve()

    try:
        if app_ckpt.is_symlink() or app_ckpt.exists():
            try:
                if app_ckpt.resolve() == vol_resolved:
                    info["link"] = "already_bound"
                    info["ok"] = True
                    return info
            except OSError:
                pass
            if app_ckpt.is_symlink() or app_ckpt.is_file():
                app_ckpt.unlink()
            elif app_ckpt.is_dir():
                # Empty or stale local dir — replace with symlink to volume
                try:
                    if not any(app_ckpt.iterdir()):
                        app_ckpt.rmdir()
                    else:
                        # Move aside so we don't destroy accidental downloads forever
                        bak = ace_root / f"checkpoints.bak.{int(time.time())}"
                        shutil.move(str(app_ckpt), str(bak))
                        info["moved_aside"] = str(bak)
                except OSError as e:
                    info["error"] = f"cannot clear {app_ckpt}: {e}"
                    return info

        app_ckpt.symlink_to(vol_resolved, target_is_directory=True)
        info["link"] = f"symlink -> {vol_resolved}"
        info["ok"] = True
        log(f"Bound ACE checkpoints: {app_ckpt} -> {vol_resolved}")
    except OSError as e:
        # Fallback: copy-tree is too heavy; monkeypatch will still redirect
        info["error"] = f"symlink failed: {e}"
        log(f"symlink failed ({e}); will monkeypatch _get_project_root")
    return info


def patch_ace_project_root(volume_ckpt: Path) -> None:
    """
    Force AceStepHandler to treat MODELS_ROOT as project root so
    join(root, 'checkpoints') == volume_ckpt.
    """
    models_root = volume_ckpt.parent  # .../models/acestep
    root_str = str(models_root)

    def _forced_root(self) -> str:  # noqa: ARG001
        return root_str

    try:
        from acestep.core.generation.handler.progress import ProgressMixin
        from acestep.handler import AceStepHandler

        ProgressMixin._get_project_root = _forced_root  # type: ignore[method-assign]
        AceStepHandler._get_project_root = _forced_root  # type: ignore[method-assign]
        log(f"patched AceStepHandler._get_project_root -> {root_str}")
    except Exception as e:
        log(f"patch_ace_project_root failed: {e}")
        raise


def missing_components(ckpt: Path) -> list[str]:
    return [name for name in MAIN_COMPONENTS if not _real_weight_files(ckpt / name)]


def download_models(force: bool = False) -> dict[str, Any]:
    """Download turbo + vae + embedding + LM into network volume."""
    from huggingface_hub import snapshot_download

    ckpt = ensure_volume_layout()
    patterns = [
        f"{DIT_NAME}/*",
        f"{VAE_NAME}/*",
        f"{EMBED_NAME}/*",
        f"{LM_NAME}/*",
        "config.json",
    ]
    missing = missing_components(ckpt)
    marker = ckpt / ".download_ok_all"
    if marker.exists() and not force and not missing:
        return {
            "status": "already_present",
            "models_dir": str(ckpt),
            "components": {n: _component_report(ckpt, n) for n in MAIN_COMPONENTS},
            "dit": DIT_NAME,
            "lm": LM_NAME,
        }

    if missing and not force:
        log(f"Missing components on volume: {missing} — downloading")
    else:
        log(f"Downloading {HF_REPO} patterns={patterns} -> {ckpt}")

    t0 = time.time()
    snapshot_download(
        repo_id=HF_REPO,
        local_dir=str(ckpt),
        local_dir_use_symlinks=False,
        allow_patterns=patterns,
    )

    still_missing = missing_components(ckpt)
    reports = {n: _component_report(ckpt, n) for n in MAIN_COMPONENTS}
    if still_missing:
        raise RuntimeError(
            f"Download incomplete: missing real weights for {still_missing}. "
            f"components={reports}"
        )

    marker.write_text(str(time.time()), encoding="utf-8")
    # Keep legacy per-DiT marker for older health probes
    dit_marker = ckpt / DIT_NAME / ".download_ok"
    dit_marker.parent.mkdir(parents=True, exist_ok=True)
    dit_marker.write_text(str(time.time()), encoding="utf-8")

    elapsed = round(time.time() - t0, 1)
    total_gb = round(
        sum(r["weight_gb"] for r in reports.values()),
        2,
    )
    log(f"Download done in {elapsed}s total_gb~={total_gb}")
    return {
        "status": "downloaded",
        "models_dir": str(ckpt),
        "seconds": elapsed,
        "dit": DIT_NAME,
        "lm": LM_NAME,
        "components": reports,
        "weight_gb": total_gb,
    }


def _assert_handler_ready(handler: Any) -> None:
    missing = []
    for attr in ("model", "vae", "text_tokenizer", "text_encoder"):
        if getattr(handler, attr, None) is None:
            missing.append(attr)
    if missing:
        raise RuntimeError(
            f"ACE handler init incomplete: {missing} still None "
            f"(checkpoint_dir={os.environ.get('ACESTEP_CHECKPOINT_DIR')})"
        )


def patch_torch_int1_compat() -> None:
    """torchao import needs torch.int1 (PyTorch 2.5+); stub on 2.4 so transformers can load."""
    import torch

    if hasattr(torch, "int1"):
        return
    # Import-only stub; Cover does not use torchao quantization.
    torch.int1 = torch.int8  # type: ignore[attr-defined]
    log("patched torch.int1 stub (torch 2.4 compat for torchao)")


def patch_torch_custom_op_compat() -> None:
    """
    Newer diffusers FA3 custom_ops use postponed annotations; torch 2.4
    infer_schema then raises ValueError and breaks AutoencoderOobleck import.
    Skip failing registrations — SDPA path still works.
    """
    import torch

    if getattr(torch.library, "_ace_custom_op_patched", False):
        return
    orig = torch.library.custom_op

    def _wrapped(*args, **kwargs):
        deco = orig(*args, **kwargs)

        def _inner(fn):
            try:
                return deco(fn)
            except (ValueError, TypeError) as e:
                log(f"skip torch.library.custom_op (torch compat): {e}")
                return fn

        return _inner

    torch.library.custom_op = _wrapped  # type: ignore[method-assign]
    torch.library._ace_custom_op_patched = True  # type: ignore[attr-defined]
    log("patched torch.library.custom_op for diffusers/torch 2.4")


def get_dit_handler():
    global _dit_handler
    if _dit_handler is not None:
        return _dit_handler

    volume_ckpt = ensure_volume_layout()
    missing = missing_components(volume_ckpt)
    if missing:
        raise RuntimeError(
            f"Volume missing model components: {missing}. "
            f"Run action=setup first. checkpoint_dir={volume_ckpt}"
        )

    link_info = ensure_ace_checkpoints_link(volume_ckpt)
    patch_ace_project_root(volume_ckpt)
    patch_torch_int1_compat()
    patch_torch_custom_op_compat()
    log(f"checkpoint bind: {link_info}")

    log("Loading DiT...")
    t0 = time.time()
    from acestep.handler import AceStepHandler

    handler = AceStepHandler()
    # project_root arg is ignored by ACE; still pass volume models root for clarity
    status_msg, ok = handler.initialize_service(
        project_root=str(volume_ckpt.parent),
        config_path=DIT_NAME,
        device=os.environ.get("ACESTEP_DEVICE", "cuda"),
        offload_to_cpu=os.environ.get("ACESTEP_OFFLOAD", "0") == "1",
        offload_dit_to_cpu=os.environ.get("ACESTEP_OFFLOAD", "0") == "1",
        quantization=None,
    )
    if not ok:
        raise RuntimeError(f"initialize_service failed: {status_msg}")

    _assert_handler_ready(handler)
    _dit_handler = handler
    log(f"DiT ready in {time.time() - t0:.1f}s ({status_msg})")
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
        lyrics=input_data.get("lyrics", "[Instrumental]")
        if input_data.get("instrumental")
        else input_data.get("lyrics", ""),
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

    out_dir = MODELS_ROOT.parent.parent / "outputs"
    if not out_dir.parent.exists():
        out_dir = Path(os.environ.get("ACESTEP_VOLUME_ROOT", "/runpod-volume")) / "outputs"
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


def _health_payload() -> dict[str, Any]:
    root = bind_volume_paths()
    ckpt = ensure_volume_layout()
    components = {n: _component_report(ckpt, n) for n in MAIN_COMPONENTS}
    missing = [n for n, r in components.items() if not r["present"]]
    ace_root = resolve_ace_package_root()
    ace_ckpt = (ace_root / "checkpoints") if ace_root else None
    link_state = None
    if ace_ckpt is not None:
        try:
            if ace_ckpt.is_symlink():
                link_state = {"symlink": True, "target": str(ace_ckpt.resolve())}
            elif ace_ckpt.exists():
                link_state = {"symlink": False, "exists": True}
            else:
                link_state = {"symlink": False, "exists": False}
        except OSError as e:
            link_state = {"error": str(e)}

    rp = Path("/runpod-volume")
    ws = Path("/workspace")
    return {
        "status": "ok" if not missing else "incomplete",
        "volume_root": str(root),
        "volume": str(MODELS_ROOT),
        "checkpoint_dir": os.environ.get("ACESTEP_CHECKPOINT_DIR"),
        "ace_package_root": str(ace_root) if ace_root else None,
        "ace_checkpoints_link": link_state,
        "components": components,
        "missing_components": missing,
        "all_main_present": not missing,
        # backward-compat fields
        "dit_present": components[DIT_NAME]["present"],
        "dit_weight_files": components[DIT_NAME]["weight_files"],
        "dit_weight_gb": components[DIT_NAME]["weight_gb"],
        "vae_present": components[VAE_NAME]["present"],
        "embed_present": components[EMBED_NAME]["present"],
        "lm_present": components[LM_NAME]["present"],
        "runpod_volume_entries": sorted(p.name for p in rp.iterdir())[:20]
        if rp.is_dir()
        else [],
        "workspace_entries": sorted(p.name for p in ws.iterdir())[:20]
        if ws.is_dir()
        else [],
        **_cuda_info(),
    }


def handler(event: dict) -> dict:
    try:
        data = event.get("input") or {}
        action = (data.get("action") or data.get("task_type") or "cover").lower()

        if action in ("setup", "download", "health"):
            if action == "health":
                return _health_payload()
            info = download_models(force=bool(data.get("force")))
            return {"success": True, **info}

        if action == "cover":
            ckpt = ensure_volume_layout()
            if missing_components(ckpt):
                download_models()
            return run_cover(data)

        if action == "init_check":
            # Load models and verify components without generating audio
            dit = get_dit_handler()
            return {
                "success": True,
                "model": dit.model is not None,
                "vae": dit.vae is not None,
                "text_tokenizer": dit.text_tokenizer is not None,
                "text_encoder": dit.text_encoder is not None,
                "checkpoint_dir": os.environ.get("ACESTEP_CHECKPOINT_DIR"),
            }

        if action == "text2music":
            return {"error": "text2music not enabled in this worker yet; use action=cover"}

        return {"error": f"Unknown action/task_type: {action}. Use setup|health|cover|init_check"}
    except Exception as e:
        log(f"Handler error: {e}")
        import traceback

        traceback.print_exc(file=sys.stderr)
        return {"error": str(e)}


def _cuda_info() -> dict[str, Any]:
    info: dict[str, Any] = {"cuda": False}
    try:
        import torch

        info["torch"] = getattr(torch, "__version__", None)
        info["torch_cuda_build"] = getattr(getattr(torch, "version", None), "cuda", None)
        info["device_count"] = int(torch.cuda.device_count())
        info["cuda"] = bool(torch.cuda.is_available())
        if info["cuda"]:
            info["device0"] = torch.cuda.get_device_name(0)
    except Exception as e:
        info["error"] = str(e)
    return info


if __name__ == "__main__":
    ensure_volume_layout()
    log("ACE-Step Serverless worker starting (volume-first models)")
    runpod.serverless.start({"handler": handler})
