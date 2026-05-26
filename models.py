"""Model loaders — cold-start init, then cached in-process for warm calls."""
from __future__ import annotations

import gc
import logging
import os
import threading
from pathlib import Path

import torch

import volume

log = logging.getLogger("models")
_lock = threading.Lock()

# Global handles, populated lazily
_flux_pipe = None
_ltx_pipe = None
_loaded_loras: dict[str, str] = {}  # lora_path → adapter_name

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _ensure_volume() -> None:
    if not volume.MODELS.exists():
        raise RuntimeError(
            f"Volume not mounted at {volume.MODELS}. RunPod endpoint must bind "
            f"networkVolume containing /models/{{unet,clip,vae,checkpoints,loras,...}}"
        )


def load_flux_pipe():
    """FLUX dev pipeline — UNet from volume, CLIP/T5/VAE wired manually."""
    global _flux_pipe
    if _flux_pipe is not None:
        return _flux_pipe

    with _lock:
        if _flux_pipe is not None:
            return _flux_pipe

        _ensure_volume()
        log.info("loading FLUX dev pipeline...")

        from diffusers import FluxPipeline
        # Diffusers expects a Hugging Face repo layout. The volume only stores
        # raw safetensors files — load them via Diffusers' single-file utilities
        # for each component, then assemble a pipeline.
        from diffusers import AutoencoderKL, FluxTransformer2DModel
        # (removed unused FromOriginalModelMixin import — moved/renamed in diffusers 0.32+)
        from transformers import (
            CLIPTextModel, CLIPTokenizer,
            T5EncoderModel, T5TokenizerFast,
        )

        # Sub-modules from volume.
        # NOTE: from_single_file() infers component config from the safetensors
        # state-dict by default. For FLUX VAE (16-ch latent → 32-ch encoder.conv_out)
        # diffusers falls back to the SD default (4-ch latent → 8-ch conv_out) and
        # raises a shape mismatch. Pin the config explicitly to the FLUX.1-dev repo.
        FLUX_REPO = "black-forest-labs/FLUX.1-dev"
        log.info("  loading FLUX transformer from %s", volume.FLUX_UNET)
        transformer = FluxTransformer2DModel.from_single_file(
            str(volume.FLUX_UNET),
            config=FLUX_REPO, subfolder="transformer",
            torch_dtype=DTYPE,
        ).to(DEVICE)

        log.info("  loading FLUX VAE from %s", volume.FLUX_VAE)
        vae = AutoencoderKL.from_single_file(
            str(volume.FLUX_VAE),
            config=FLUX_REPO, subfolder="vae",
            torch_dtype=DTYPE,
        ).to(DEVICE)

        # CLIP-L: load from public ungated openai repo (standard CLIP weights).
        # T5XXL: build from FLUX config + manual state_dict load from volume
        # safetensors (avoids the gated FLUX.1-dev/text_encoder_2 download path
        # which fails resolution for sharded safetensors via from_pretrained).
        from transformers import T5Config
        from safetensors.torch import load_file as load_safetensors

        log.info("  loading CLIP-L from openai/clip-vit-large-patch14")
        text_encoder = CLIPTextModel.from_pretrained(
            "openai/clip-vit-large-patch14", torch_dtype=DTYPE,
        ).to(DEVICE)
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        log.info("  building T5XXL with hardcoded FLUX.1-dev dims + volume safetensors %s",
                 volume.FLUX_T5XXL)
        # T5Config.from_pretrained(..., subfolder=...) silently returned T5-base
        # defaults (d_model=512) instead of reading FLUX.1-dev/text_encoder_2/config.json
        # which has d_model=4096. Hardcode the FLUX T5XXL dims to avoid the subfolder
        # resolution quirk entirely.
        t5_config = T5Config(
            d_model=4096, d_ff=10240, d_kv=64,
            num_heads=64, num_layers=24,
            vocab_size=32128,
            feed_forward_proj="gated-gelu",
            tie_word_embeddings=False,
            is_encoder_decoder=False,
            layer_norm_epsilon=1e-6,
            relative_attention_num_buckets=32,
            dropout_rate=0.1,
            initializer_factor=1.0,
        )
        text_encoder_2 = T5EncoderModel(t5_config).to(DTYPE)
        t5_state = load_safetensors(str(volume.FLUX_T5XXL))
        missing, unexpected = text_encoder_2.load_state_dict(t5_state, strict=False)
        if missing:
            log.warning("  T5 missing keys (%d): %s", len(missing), missing[:5])
        if unexpected:
            log.warning("  T5 unexpected keys (%d): %s", len(unexpected), unexpected[:5])
        text_encoder_2 = text_encoder_2.to(DEVICE)
        # Tokenizer from FLUX_REPO subfolder — tiny files, gated but auth'd via HF_TOKEN.
        # google/t5-v1_1-xxl has been intermittently failing for fresh workers (HF rate
        # limit or transient unavailability), so prefer the already-authenticated repo.
        tokenizer_2 = T5TokenizerFast.from_pretrained(
            FLUX_REPO, subfolder="tokenizer_2",
        )

        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler()

        _flux_pipe = FluxPipeline(
            transformer=transformer,
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            scheduler=scheduler,
        )
        _flux_pipe.set_progress_bar_config(disable=True)

        # Memory hygiene
        torch.cuda.empty_cache()
        log.info("FLUX pipeline ready on %s dtype=%s", DEVICE, DTYPE)
    return _flux_pipe


def load_ltx_pipe():
    """LTX 2.3 I2V pipeline — checkpoint + Gemma encoder + spatial upscaler."""
    global _ltx_pipe
    if _ltx_pipe is not None:
        return _ltx_pipe

    with _lock:
        if _ltx_pipe is not None:
            return _ltx_pipe

        _ensure_volume()
        log.info("loading LTX 2.3 I2V pipeline...")

        # Try Diffusers LTXImageToVideoPipeline first (0.32+)
        try:
            from diffusers import LTXImageToVideoPipeline
            log.info("  using diffusers.LTXImageToVideoPipeline")
            # Diffusers expects a repo-style folder. For LTX 2.3 raw safetensors we use
            # single_file_load (if supported). If not, fall through to ltx_video SDK.
            _ltx_pipe = LTXImageToVideoPipeline.from_single_file(
                str(volume.LTX_CKPT), torch_dtype=DTYPE,
            ).to(DEVICE)
            _ltx_pipe.set_progress_bar_config(disable=True)
            log.info("LTX 2.3 pipeline ready via diffusers")
            return _ltx_pipe
        except Exception as e:
            log.warning("diffusers LTX path failed (%s) — trying ltx_video SDK", e)

        try:
            from ltx_video.inference import infer  # type: ignore  # noqa
            log.info("  ltx_video SDK available — handler will use it directly")
            _ltx_pipe = ("ltx_video_sdk", str(volume.LTX_CKPT), str(volume.LTX_GEMMA_DIR))
            return _ltx_pipe
        except ImportError:
            raise RuntimeError(
                "Neither diffusers LTX nor ltx_video SDK available. "
                "Add `pip install git+https://github.com/Lightricks/LTX-Video` to Dockerfile."
            )


def apply_lora(pipeline, lora_name: str, weight: float = 0.8, adapter_name: str | None = None):
    """Load a LoRA from volume and attach to the pipeline's transformer."""
    if not lora_name:
        return
    p = volume.lora_path(lora_name)
    if not p.exists():
        raise FileNotFoundError(f"LoRA not found in volume: {p}")
    if adapter_name is None:
        adapter_name = p.stem
    if str(p) in _loaded_loras:
        log.info("LoRA already loaded: %s (adapter=%s)", p.name, adapter_name)
    else:
        pipeline.load_lora_weights(str(p.parent), weight_name=p.name, adapter_name=adapter_name)
        _loaded_loras[str(p)] = adapter_name
        log.info("LoRA loaded: %s weight=%.2f", p.name, weight)
    pipeline.set_adapters([adapter_name], adapter_weights=[weight])


def unload_loras(pipeline):
    """Detach all LoRAs (between renders to avoid weight accumulation)."""
    try:
        pipeline.unload_lora_weights()
        _loaded_loras.clear()
    except Exception as e:
        log.warning("unload_loras failed: %s", e)


def cleanup():
    """Free CUDA memory between renders."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
