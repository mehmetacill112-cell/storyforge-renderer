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
    volume.assert_flux_files_present()


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
            T5EncoderModel, T5Tokenizer,
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
        # Slow T5Tokenizer (pure Python) avoids T5TokenizerFast's runtime
        # SentencePiece→fast conversion which needs sentencepiece+protobuf
        # packages not present in the base image. Speed delta is negligible
        # for short prompts (microseconds vs nanoseconds).
        tokenizer_2 = T5Tokenizer.from_pretrained(
            FLUX_REPO, subfolder="tokenizer_2",
        )

        from diffusers import FlowMatchEulerDiscreteScheduler
        scheduler = FlowMatchEulerDiscreteScheduler()

        # Diagnostic — log type of each component before instantiation. Smoke #8
        # hit 'bool' object has no attribute '__module__' in register_modules;
        # find which kwarg is bool.
        for name, obj in [
            ("transformer", transformer), ("vae", vae),
            ("text_encoder", text_encoder), ("tokenizer", tokenizer),
            ("text_encoder_2", text_encoder_2), ("tokenizer_2", tokenizer_2),
            ("scheduler", scheduler),
        ]:
            log.info("  FluxPipe arg %s: type=%s class=%s module=%s",
                     name, type(obj).__name__,
                     getattr(obj, '__class__', '?'),
                     getattr(type(obj), '__module__', '?'))

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
        # FLUX (12B transformer + T5XXL 9.3GB + CLIP-L) + LTX 2.3 22B (38GB
        # transformer + Gemma 12B) cannot both stay resident in an 80GB A100.
        # model_cpu_offload swaps each pipeline's modules onto GPU only while
        # actively in use, keeping the bulk in CPU RAM (host has ~150GB+).
        # Inference is ~10% slower per pass but OOM is eliminated.
        _flux_pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        log.info("FLUX pipeline ready on %s dtype=%s (cpu_offload enabled)", DEVICE, DTYPE)
    return _flux_pipe


def load_ltx_pipe():
    """LTX 2.3 22B distilled I2V pipeline — via diffusers LTX2ImageToVideoPipeline."""
    global _ltx_pipe
    if _ltx_pipe is not None:
        return _ltx_pipe

    with _lock:
        if _ltx_pipe is not None:
            return _ltx_pipe

        _ensure_volume()
        log.info("loading LTX 2.3 distilled pipeline (from_pretrained diffusers/LTX-2.3-Distilled-Diffusers)...")

        # LTX2ImageToVideoPipeline requires 8 wired components (scheduler, vae,
        # audio_vae, text_encoder, tokenizer, connectors, transformer, vocoder).
        # from_single_file on the local .safetensors fails because the file only
        # carries transformer+vae; the other components live in the reference HF
        # repo. The canonical loader is from_pretrained against the official
        # diffusers-org distilled repo (verified 2026-05-26: all 9 subfolders +
        # model_index.json + README present, transformer 38GB sharded x8).
        # First cold start pulls ~50GB into HF_HOME=/runpod-volume/hf-cache;
        # subsequent cold starts reuse the cache.
        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)

        from diffusers.pipelines.ltx2 import LTX2ImageToVideoPipeline
        # NOTE: do NOT pre-move to DEVICE — enable_model_cpu_offload moves
        # components as needed. Calling .to(DEVICE) here would defeat the
        # offload and re-trigger the OOM observed in render v6 (FLUX 12B +
        # T5XXL + LTX 22B + Gemma 12B = ~85GB > A100 80GB).
        _ltx_pipe = LTX2ImageToVideoPipeline.from_pretrained(
            "diffusers/LTX-2.3-Distilled-Diffusers",
            torch_dtype=DTYPE,
        )
        _ltx_pipe.set_progress_bar_config(disable=True)
        _ltx_pipe.enable_model_cpu_offload()
        torch.cuda.empty_cache()
        log.info("LTX 2.3 pipeline ready (cpu_offload enabled)")
        return _ltx_pipe


_ip_adapter_loaded = False


def attach_ip_adapter(pipeline, image, weight: float = 0.7) -> None:
    """Attach FLUX IP-Adapter with the given reference image.

    The IP-Adapter weights live on the volume at
    /runpod-volume/models/ipadapter-flux/ip-adapter.bin and the SigLIP CLIP
    vision encoder at /runpod-volume/models/clip_vision/siglip-so400m-patch14-384/.
    They load lazily the first time chain-mode is used (5GB + 3.3GB).

    `image` is a PIL Image — the character anchor frame.
    """
    global _ip_adapter_loaded
    if not _ip_adapter_loaded:
        try:
            ipa_path = str(volume.IPADAPTER_FLUX)  # /runpod-volume/models/ipadapter-flux/ip-adapter.bin
            siglip_dir = str(volume.CLIP_VISION_SIGLIP)  # siglip-so400m-patch14-384
            log.info("loading FLUX IP-Adapter from %s (siglip=%s)", ipa_path, siglip_dir)
            # Diffusers FluxPipeline.load_ip_adapter signature accepts a local
            # path + subfolder or a (state_dict, image_encoder) pair. We pass
            # the local repo path of the IPA weights plus the siglip encoder.
            pipeline.load_ip_adapter(
                str(Path(ipa_path).parent),
                weight_name=Path(ipa_path).name,
                image_encoder_pretrained_model_name_or_path=siglip_dir,
            )
            _ip_adapter_loaded = True
        except Exception as e:
            log.warning("ip_adapter_load_failed: %s — chain anchor disabled", e)
            return
    try:
        pipeline.set_ip_adapter_scale(float(weight))
        # The image is passed at __call__ time via ip_adapter_image kwarg, but
        # to avoid changing render.py's call site signature heavily, we stash it
        # on the pipeline; render.py's `pipe(**kwargs)` will pick it up.
        pipeline._sf_ip_adapter_image = image  # type: ignore[attr-defined]
        log.info("IP-Adapter attached weight=%.2f", weight)
    except Exception as e:
        log.warning("ip_adapter_attach_failed: %s", e)


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
    """Detach all LoRAs (between renders to avoid weight accumulation).

    Clears the in-process registry even if the underlying pipeline call
    raises — adapter accumulation across renders is worse than a noisy log,
    and the registry tracks our intent regardless of pipeline state.
    """
    try:
        pipeline.unload_lora_weights()
    except Exception as e:
        log.warning("pipeline.unload_lora_weights raised %s: %s — clearing registry anyway",
                    type(e).__name__, e)
    _loaded_loras.clear()


def unload_flux() -> None:
    """Free FLUX pipeline + components from VRAM before loading LTX.

    cpu_offload was insufficient for our manually-constructed FluxPipeline
    (components were .to(DEVICE) before pipeline init, so the offload hook
    couldn't track them). Explicit del + gc + empty_cache reclaims the ~22GB
    that FLUX holds (12B transformer bf16 + 9GB T5XXL + 250MB CLIP-L + LoRAs +
    activations), letting LTX 22B + Gemma 12B + audio_vae fit on A100 80GB.
    """
    global _flux_pipe
    if _flux_pipe is None:
        return
    try:
        unload_loras(_flux_pipe)
    except Exception:
        pass
    del _flux_pipe
    _flux_pipe = None
    _loaded_loras.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("FLUX unloaded — VRAM reclaimed before LTX load")


def unload_ltx() -> None:
    """Symmetric LTX unload (e.g., if next request needs FLUX-only path)."""
    global _ltx_pipe
    if _ltx_pipe is None:
        return
    del _ltx_pipe
    _ltx_pipe = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    log.info("LTX unloaded — VRAM reclaimed")


def cleanup():
    """Free CUDA memory between renders."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
