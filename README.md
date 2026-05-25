# storyforge-renderer

Custom RunPod serverless endpoint — FLUX dev (first frame) + LTX 2.3 22B distilled-1.1 (image-to-video) + 25 trained character LoRAs.

Direct Python handler, **no ComfyUI**. All models served from RunPod network volume `z0jl8dbtgo` (EU-RO-1).

## Architecture

```
event["input"] = { prompt, char_lora?, style_lora?, width, height, duration_sec, seed, ... }
                          │
                          ▼
  ┌───────────── handler.py ─────────────┐
  │                                       │
  │  render.render(payload)               │
  │   ├─ FLUX dev first-frame             │
  │   │    + Turbo Alpha 8-step LoRA      │
  │   │    + style LoRA (pixar / ghibli)  │
  │   │    + optional char LoRA           │
  │   ├─ LTX 2.3 22B I2V                  │
  │   │    + IC-LoRA (hdr / motion /      │
  │   │      union)                       │
  │   └─ ffmpeg h264 yuv420p              │
  │                                       │
  └───────────────────────────────────────┘
                          │
                          ▼
  { first_frame_b64, video_b64, mime, width, height, length, duration_sec, fps }
```

## API

### Render
```json
POST /run
{
  "input": {
    "prompt": "Zeus on Mount Olympus, golden lightning, majestic 3D Pixar",
    "char_lora": "mythology__zeus",
    "style_lora": "pixar_3d_canopus",
    "width": 1280,
    "height": 720,
    "duration_sec": 5.0,
    "fps": 25,
    "seed": 42,
    "char_lora_weight": 0.8,
    "style_lora_weight": 0.75,
    "ic_lora_hdr": 0.7,
    "ic_lora_motion": 0.5,
    "ic_lora_union": 0.4,
    "flux_steps": 8,
    "flux_guidance": 5.0,
    "negative_prompt": "text, watermark, morph, blurry, lowres"
  }
}
```

### Ops
- `{"input": {"op": "ping"}}` → `{"ok": true}` — handler alive check
- `{"input": {"op": "list_loras"}}` → `{"loras": [...]}` — known LoRA roster

## Volume Layout

```
/runpod-volume/models/
├── unet/flux1-dev.safetensors                              # 22.7 GB
├── clip/{clip_l, t5xxl_fp16}.safetensors
├── vae/ae.safetensors
├── checkpoints/ltx-2.3-22b-distilled-1.1.safetensors       # 44 GB
├── text_encoders/gemma-3-12b-it-qat-q4_0-unquantized/      # LTX 2.3 Gemma encoder
├── upscale_models/ltxv-spatial-upscaler-0.9.8.safetensors
├── loras/
│   ├── mythology__{zeus,thor,odin,loki,hades,athena,ra,anubis,...}.safetensors  # 15 chars
│   ├── bible_stories_for_kids__{abraham,daniel,esther,...}.safetensors          # 10 chars
│   ├── pixar_3d_canopus.safetensors        # style
│   ├── ghibli_warm_openfree.safetensors    # style
│   ├── flux_turbo_alpha_8step.safetensors  # 8-step accelerator
│   ├── ltx23_iclora_hdr.safetensors
│   ├── ltx23_iclora_motion_track.safetensors
│   ├── ltx23_iclora_union_control.safetensors
│   └── ltx23_iclora_lipdub.safetensors
└── ipadapter-flux/ip-adapter.bin
```

## Deployment

1. Push to `main` → GitHub Action builds + pushes `ghcr.io/mehmetacill112-cell/storyforge-renderer:latest`.
2. Create RunPod template referencing this image + network volume.
3. Create endpoint with `minCudaVersion=13.0`, GPU pool: A100 / L40S / L40 / RTX 6000 Ada / RTX A6000 (48GB+).
4. Update `storyforge` `.env` `RUNPOD_ENDPOINT_ID` to point at the new endpoint.

## Hardware

- **GPU**: 48GB+ VRAM (LTX 2.3 22B BF16 ≈ 44GB resident; Ada/Ampere/Hopper).
- **Network volume**: 200GB at EU-RO-1, mounted at `/runpod-volume`.
- **CUDA**: 13.0, PyTorch 2.9.1+cu130, Python 3.12, Ubuntu 24.04.

## Implementation status

- [x] Repo scaffold
- [x] handler.py + render.py + models.py + volume.py
- [x] Dockerfile (cu130/torch291/ubuntu2404 base)
- [x] GitHub Actions build-and-push
- [ ] FLUX text encoder offline cache wired (currently relies on online download — TODO bundle into image)
- [ ] LTX 2.3 22B Diffusers single-file load verified
- [ ] Smoke render success
- [ ] storyforge `_submit` integration
