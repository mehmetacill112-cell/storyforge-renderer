"""RunPod serverless entry point.

Receives:
  event["input"] = {
    "prompt": "Zeus on Mount Olympus, golden lightning",
    "char_lora": "mythology__zeus",          # optional, one of volume.known_loras()
    "style_lora": "pixar_3d_canopus",        # optional, default pixar
    "width": 1280, "height": 720,            # optional
    "duration_sec": 5.0, "fps": 25,          # optional
    "seed": 42,
    "char_lora_weight": 0.8,
    "style_lora_weight": 0.75,
    "flux_steps": 8, "flux_guidance": 5.0,
    "ic_lora_hdr": 0.7, "ic_lora_motion": 0.5, "ic_lora_union": 0.4,
    "negative_prompt": "...",
  }

Returns:
  {
    "first_frame_b64": "<png base64>",
    "video_b64": "<mp4 base64>",
    "mime": "video/mp4",
    "width": ..., "height": ...,
    "length": ..., "duration_sec": ..., "fps": ...
  }

Error → { "error": "<msg>", "type": "<ExceptionClass>" }
"""
from __future__ import annotations

import logging
import os
import traceback

import runpod

import render
import volume

# Logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
log = logging.getLogger("handler")


def _ensure_runtime_dirs() -> None:
    """Create writable scratch + HF cache dirs on volume before first job runs.

    RunPod containers ship a small / quota; redirecting TMPDIR + HF_*_CACHE to
    volume avoids 'Disk quota exceeded' during multi-GB pulls. The Python libs
    do NOT auto-create these dirs from env vars — must mkdir on startup.
    """
    paths = [
        os.environ.get("TMPDIR", "/runpod-volume/tmp"),
        os.environ.get("HF_HOME", "/runpod-volume/hf-cache"),
        os.environ.get("HF_HUB_CACHE", "/runpod-volume/hf-cache/hub"),
        os.environ.get("HF_XET_CACHE", "/runpod-volume/hf-cache/xet"),
        os.environ.get("HF_ASSETS_CACHE", "/runpod-volume/hf-cache/assets"),
    ]
    for p in paths:
        try:
            os.makedirs(p, exist_ok=True)
        except Exception as e:
            log.warning("mkdir %s failed: %s", p, e)


def handler(event: dict) -> dict:
    try:
        payload = (event or {}).get("input") or {}
        log.info("job start: prompt=%r char_lora=%s style_lora=%s w=%s h=%s dur=%s",
                 (payload.get("prompt") or "")[:80],
                 payload.get("char_lora"), payload.get("style_lora"),
                 payload.get("width"), payload.get("height"), payload.get("duration_sec"))

        # Special op: "list_loras" returns the known LoRA roster (no model load).
        if payload.get("op") == "list_loras":
            return {"loras": volume.known_loras()}
        if payload.get("op") == "ping":
            return {"ok": True}
        if payload.get("op") == "disk":
            # Volume usage + HF cache size — for diagnosing "Disk quota exceeded".
            import shutil, subprocess
            usage = {}
            for p in ["/runpod-volume", "/runpod-volume/models", "/runpod-volume/hf-cache", "/tmp", "/"]:
                try:
                    total, used, free = shutil.disk_usage(p)
                    usage[p] = {"total_gb": round(total/1e9, 1), "used_gb": round(used/1e9, 1), "free_gb": round(free/1e9, 1)}
                except Exception as e:
                    usage[p] = f"err: {e}"
            try:
                du_out = subprocess.run(["du","-sh","/runpod-volume/hf-cache","/runpod-volume/models"], capture_output=True, text=True, timeout=30).stdout
            except Exception as e:
                du_out = f"err: {e}"
            return {"usage": usage, "du": du_out}
        if payload.get("op") == "ltx_meta":
            # Read LTX checkpoint metadata to identify exact config expected
            from safetensors import safe_open
            from pathlib import Path
            ckpt = Path("/runpod-volume/models/checkpoints/ltx-2.3-22b-distilled-1.1.safetensors")
            if not ckpt.exists():
                return {"err": f"ckpt missing: {ckpt}"}
            try:
                with safe_open(str(ckpt), framework="pt") as f:
                    meta = f.metadata() or {}
                # Truncate config to readable size + extract top-level keys
                import json
                cfg = meta.get("config", "")
                try:
                    cfg_obj = json.loads(cfg) if cfg else {}
                    cfg_keys = list(cfg_obj.keys())[:30]
                    vae_keys = list(cfg_obj.get("vae", {}).keys())[:20] if isinstance(cfg_obj.get("vae"), dict) else "n/a"
                except Exception:
                    cfg_obj = {}
                    cfg_keys = []
                    vae_keys = "parse-err"
                return {
                    "meta_keys": list(meta.keys()),
                    "cfg_top_keys": cfg_keys,
                    "vae_keys": vae_keys,
                    "cfg_snippet": cfg[:1500],
                    "config_full_len": len(cfg),
                }
            except Exception as e:
                return {"err": str(e)}
        if payload.get("op") == "audit_volume":
            # Deep recursive listing — find unused/large files for cleanup decisions.
            import subprocess
            results = {}
            for cmd, label in [
                (["du","-sh","/runpod-volume/models/unet","/runpod-volume/models/clip","/runpod-volume/models/vae","/runpod-volume/models/checkpoints","/runpod-volume/models/text_encoders","/runpod-volume/models/upscale_models","/runpod-volume/models/ipadapter-flux","/runpod-volume/models/clip_vision","/runpod-volume/models/loras"], "by_subdir"),
                (["bash","-c","ls -lhS /runpod-volume/models/loras/ 2>&1 | head -40"], "loras_by_size"),
                (["bash","-c","ls -lh /runpod-volume/ 2>&1"], "root_listing"),
                (["bash","-c","find /runpod-volume/models -maxdepth 2 -type d 2>&1 | head -30"], "model_dirs"),
                (["bash","-c","du -sh /runpod-volume/wav2lip /runpod-volume/xtts /runpod-volume/tmp 2>&1 || true"], "non_model_dirs"),
            ]:
                try:
                    r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                    results[label] = r.stdout + (("\nERR:"+r.stderr) if r.stderr else "")
                except Exception as e:
                    results[label] = f"err: {e}"
            return results
        if payload.get("op") == "delete_path":
            # Whitelisted file removal — only under /runpod-volume/ for safety.
            from pathlib import Path
            target = payload.get("path", "")
            p = Path(target)
            if not str(p).startswith("/runpod-volume/"):
                return {"err": "path must be under /runpod-volume/"}
            if not p.exists():
                return {"err": f"path not found: {p}"}
            size_before = p.stat().st_size if p.is_file() else 0
            try:
                if p.is_file():
                    p.unlink()
                else:
                    import shutil
                    shutil.rmtree(p)
                return {"deleted": str(p), "freed_bytes": size_before}
            except Exception as e:
                return {"err": str(e)}
        if payload.get("op") == "clean_hf_cache":
            # Wipe HF_HOME contents — forces fresh download next cold start.
            import shutil
            from pathlib import Path
            cache = Path("/runpod-volume/hf-cache")
            removed = []
            if cache.exists():
                for child in cache.iterdir():
                    try:
                        if child.is_dir():
                            shutil.rmtree(child)
                        else:
                            child.unlink()
                        removed.append(str(child))
                    except Exception as e:
                        removed.append(f"FAIL {child}: {e}")
            return {"removed": removed, "count": len(removed)}

        result = render.render(payload)
        log.info("job done: frames=%d", result.get("length"))
        return result

    except Exception as e:
        tb = traceback.format_exc()
        log.error("job failed: %s\n%s", e, tb)
        return {
            "error": str(e),
            "type": type(e).__name__,
            "traceback": tb,
        }


if __name__ == "__main__":
    _ensure_runtime_dirs()
    log.info("Storyforge Renderer starting | volume models = %s", volume.MODELS)
    runpod.serverless.start({"handler": handler})
