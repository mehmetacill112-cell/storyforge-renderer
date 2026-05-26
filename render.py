"""Render pipeline: FLUX first_frame → LTX 2.3 I2V → MP4 via ffmpeg."""
from __future__ import annotations

import base64
import io
import logging
import subprocess
import tempfile
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image

import models
import volume

log = logging.getLogger("render")


def _frames_for(duration_sec: float, fps: int = 25) -> int:
    """LTX requires (8n+1) frame counts."""
    raw = int(round(duration_sec * fps))
    n = max(1, round((raw - 1) / 8))
    return 8 * n + 1


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _frames_to_mp4(frames: list[np.ndarray], fps: int = 25, crf: int = 19) -> bytes:
    """Encode a list of HxWxC uint8 RGB frames to an MP4 byte string."""
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
        out_path = Path(tf.name)
    try:
        iio.imwrite(
            out_path, np.stack(frames, axis=0),
            extension=".mp4", fps=fps, codec="libx264",
            output_params=["-pix_fmt", "yuv420p", "-crf", str(crf)],
            macro_block_size=1,
        )
        return out_path.read_bytes()
    finally:
        out_path.unlink(missing_ok=True)


def generate_first_frame(
    prompt: str,
    *,
    width: int = 1280,
    height: int = 720,
    seed: int = 42,
    char_lora: str | None = None,
    char_lora_weight: float = 0.8,
    style_lora: str = "pixar_3d_canopus",
    style_lora_weight: float = 0.75,
    use_turbo_alpha: bool = True,
    guidance: float = 5.0,
    steps: int = 8,
    negative_prompt: str = "text, watermark, morph, deformed, lowres, blurry, nsfw",
) -> Image.Image:
    """FLUX first-frame generation with Turbo Alpha + style LoRA + optional char LoRA."""
    pipe = models.load_flux_pipe()
    models.unload_loras(pipe)

    # Stack LoRAs: Turbo Alpha + Style + Char
    adapters = []
    weights = []
    if use_turbo_alpha:
        models.apply_lora(pipe, volume.TURBO_ALPHA_LORA, weight=1.0, adapter_name="turbo")
        adapters.append("turbo"); weights.append(1.0)
    if style_lora:
        models.apply_lora(pipe, style_lora, weight=style_lora_weight, adapter_name="style")
        adapters.append("style"); weights.append(style_lora_weight)
    if char_lora:
        models.apply_lora(pipe, char_lora, weight=char_lora_weight, adapter_name="char")
        adapters.append("char"); weights.append(char_lora_weight)
    if adapters:
        pipe.set_adapters(adapters, adapter_weights=weights)

    gen = torch.Generator(device=models.DEVICE).manual_seed(int(seed))
    log.info("FLUX render: %dx%d steps=%d cfg=%.1f loras=%s",
             width, height, steps, guidance, adapters)

    result = pipe(
        prompt=prompt,
        width=width,
        height=height,
        num_inference_steps=steps,
        guidance_scale=guidance,
        generator=gen,
    )
    return result.images[0]


def generate_video(
    *,
    prompt: str,
    first_frame: Image.Image,
    width: int,
    height: int,
    duration_sec: float = 5.0,
    fps: int = 25,
    seed: int = 42,
    ic_lora_hdr: float = 0.7,
    ic_lora_motion: float = 0.5,
    ic_lora_union: float = 0.4,
) -> bytes:
    """LTX 2.3 I2V: convert first_frame to video. Returns MP4 bytes."""
    pipe = models.load_ltx_pipe()
    length = _frames_for(duration_sec, fps)

    # LTX2Pipeline (diffusers main, LTX 2.3 22B distilled native support).
    # NOTE: IC-LoRAs (hdr/motion_track/union_control) were a diffusers
    # LTXImageToVideoPipeline (0.9.x/13B) feature — LTX2Pipeline doesn't
    # expose .set_adapters yet; parking those LoRAs until library catches up.
    gen = torch.Generator(device=models.DEVICE).manual_seed(int(seed))
    log.info("LTX 2.3 22B I2V: %dx%d length=%d fps=%d", width, height, length, fps)
    result = pipe(
        prompt=prompt,
        image=first_frame,
        width=width, height=height,
        num_frames=length, frame_rate=fps,
        num_inference_steps=10,
        generator=gen,
    )
    # result.frames is List[List[PIL]] when output_type defaults to "pil"
    frames = [np.asarray(f) for f in result.frames[0]]
    return _frames_to_mp4(frames, fps=fps)


def render(payload: dict) -> dict:
    """Top-level render entry.

    payload schema:
      prompt: str  (required)
      char_lora: str | None  (e.g. "mythology__zeus")
      style_lora: str        (default "pixar_3d_canopus")
      width: int             (default 1280)
      height: int            (default 720)
      duration_sec: float    (default 5.0)
      seed: int              (default random)
      fps: int               (default 25)
      char_lora_weight, style_lora_weight, etc.

    Returns:
      { first_frame_b64: str, video_b64: str, mime: "video/mp4",
        width, height, length, duration_sec, fps }
    """
    prompt = payload.get("prompt") or ""
    if not prompt:
        raise ValueError("prompt is required")

    width = int(payload.get("width", 1280))
    height = int(payload.get("height", 720))
    duration_sec = float(payload.get("duration_sec", 5.0))
    fps = int(payload.get("fps", 25))
    seed = int(payload.get("seed", 42))
    char_lora = payload.get("char_lora")
    style_lora = payload.get("style_lora", "pixar_3d_canopus")

    if char_lora and char_lora not in volume.known_loras():
        raise ValueError(f"unknown char_lora '{char_lora}' (known: {volume.known_loras()[:5]}...)")

    # 1) FLUX first frame
    first_frame = generate_first_frame(
        prompt=prompt,
        width=width, height=height, seed=seed,
        char_lora=char_lora,
        char_lora_weight=float(payload.get("char_lora_weight", 0.8)),
        style_lora=style_lora,
        style_lora_weight=float(payload.get("style_lora_weight", 0.75)),
        use_turbo_alpha=bool(payload.get("use_turbo_alpha", True)),
        steps=int(payload.get("flux_steps", 8)),
        guidance=float(payload.get("flux_guidance", 5.0)),
        negative_prompt=payload.get("negative_prompt", ""),
    )
    first_frame_b64 = base64.b64encode(_png_bytes(first_frame)).decode()

    # 2) LTX 2.3 I2V
    mp4_bytes = generate_video(
        prompt=prompt,
        first_frame=first_frame,
        width=width, height=height,
        duration_sec=duration_sec, fps=fps, seed=seed,
        ic_lora_hdr=float(payload.get("ic_lora_hdr", 0.7)),
        ic_lora_motion=float(payload.get("ic_lora_motion", 0.5)),
        ic_lora_union=float(payload.get("ic_lora_union", 0.4)),
    )
    video_b64 = base64.b64encode(mp4_bytes).decode()

    models.cleanup()

    return {
        "first_frame_b64": first_frame_b64,
        "video_b64": video_b64,
        "mime": "video/mp4",
        "width": width,
        "height": height,
        "length": _frames_for(duration_sec, fps),
        "duration_sec": duration_sec,
        "fps": fps,
    }
