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
    log.info("Storyforge Renderer starting | volume models = %s", volume.MODELS)
    runpod.serverless.start({"handler": handler})
