"""Volume path resolver. RunPod network volumes mount at /runpod-volume."""
from pathlib import Path

VOLUME_ROOT = Path("/runpod-volume")
MODELS = VOLUME_ROOT / "models"

# Model file paths (verified from volume audit)
FLUX_UNET = MODELS / "unet" / "flux1-dev.safetensors"            # 22.7 GB
FLUX_CLIP_L = MODELS / "clip" / "clip_l.safetensors"             # 234 MB
FLUX_T5XXL = MODELS / "clip" / "t5xxl_fp16.safetensors"          # 9.3 GB
FLUX_VAE = MODELS / "vae" / "ae.safetensors"                     # 320 MB

LTX_CKPT = MODELS / "checkpoints" / "ltx-2.3-22b-distilled-1.1.safetensors"  # 44 GB
LTX_GEMMA_DIR = MODELS / "text_encoders" / "gemma-3-12b-it-qat-q4_0-unquantized"
LTX_SPATIAL_UPSCALER = MODELS / "upscale_models" / "ltxv-spatial-upscaler-0.9.8.safetensors"

LORAS = MODELS / "loras"
IPADAPTER_FLUX = MODELS / "ipadapter-flux" / "ip-adapter.bin"
CLIP_VISION_SIGLIP = MODELS / "clip_vision" / "siglip-so400m-patch14-384"


def lora_path(name: str) -> Path:
    """Resolve a LoRA name (with or without .safetensors) to a volume path."""
    if not name.endswith(".safetensors"):
        name = name + ".safetensors"
    return LORAS / name


# Trained LoRA roster (audit 2026-05-25)
MYTHOLOGY_LORAS = [
    "mythology__zeus", "mythology__thor", "mythology__odin", "mythology__loki",
    "mythology__hades", "mythology__athena", "mythology__ra", "mythology__anubis",
    "mythology__hera", "mythology__apollo", "mythology__artemis",
    "mythology__poseidon", "mythology__hermes", "mythology__bastet",
    "mythology__sage_the_storyteller_owl",
]

BIBLE_LORAS = [
    "bible_stories_for_kids__abraham", "bible_stories_for_kids__daniel",
    "bible_stories_for_kids__esther", "bible_stories_for_kids__jonah",
    "bible_stories_for_kids__lyra_the_storybook_friend",
    "bible_stories_for_kids__moses", "bible_stories_for_kids__noah",
    "bible_stories_for_kids__young_david", "bible_stories_for_kids__young_joseph",
    "bible_stories_for_kids__young_mary",
]

STYLE_LORAS = {
    "pixar_3d": "pixar_3d_canopus",
    "ghibli_warm": "ghibli_warm_openfree",
    "pixar_lh": "lh_pixar_3d_style",
}

TURBO_ALPHA_LORA = "flux_turbo_alpha_8step"

IC_LORAS_LTX = {
    "hdr": "ltx23_iclora_hdr",
    "motion_track": "ltx23_iclora_motion_track",
    "union_control": "ltx23_iclora_union_control",
    "lipdub": "ltx23_iclora_lipdub",
}


def known_loras() -> list[str]:
    return MYTHOLOGY_LORAS + BIBLE_LORAS + list(STYLE_LORAS.values()) + [TURBO_ALPHA_LORA] + list(IC_LORAS_LTX.values())


def assert_flux_files_present() -> None:
    """Per-file presence check — surfaces broken volume mount before the slow
    pipeline init that would otherwise fail mid-load with a confusing trace."""
    required = {
        "FLUX_UNET": FLUX_UNET,
        "FLUX_CLIP_L": FLUX_CLIP_L,
        "FLUX_T5XXL": FLUX_T5XXL,
        "FLUX_VAE": FLUX_VAE,
    }
    missing = [name for name, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"volume mounted but FLUX files missing: {missing}. "
            f"Check /runpod-volume/models/{{unet,clip,vae}} on network volume."
        )
