# Custom RunPod serverless image for FLUX + LTX 2.3 rendering.
# Base: PyTorch 2.9.1 + CUDA 13.0 + Python 3.12 + Ubuntu 24.04 (Ada/Ampere/Hopper)
FROM runpod/pytorch:1.0.3-dev-fix-image-vulnerabilities-cu1300-torch291-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive
# Online-with-cache strategy: HF_HUB_OFFLINE removed. Workers fetch tokenizer
# configs + (on first cold start) FLUX/T5/CLIP repos to HF_HOME, then reuse the
# cache on subsequent boots. HF_HOME is set on the endpoint to point into the
# RunPod network volume so the cache survives worker rotation.

# System deps: ffmpeg for video encoding
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        git \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python deps
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --break-system-packages -r /app/requirements.txt

# Lightricks LTX-Video repo for LTX 2.3 inference (Gemma loader + tiled VAE decode).
# CRITICAL: ltx-video pulls in older transformers/hub/tokenizers that downgrade our base
# AND can prune unrelated packages (smoke #11 lost `av` despite explicit install above).
# Re-pin av + transformers AFTER ltx-video install to guarantee runtime presence.
RUN pip install --no-cache-dir --break-system-packages \
    "git+https://github.com/Lightricks/LTX-Video.git@main#egg=ltx-video" \
    && pip install --no-cache-dir --break-system-packages --force-reinstall \
        "av" "transformers>=4.45.0" "huggingface_hub>=0.27.0" "tokenizers" \
    && python3 -c "import av; print('av ok', av.__version__)" \
    && python3 -c "from ltx_video.inference import infer; print('ltx_video.inference ok')"

# App code
COPY handler.py models.py render.py volume.py /app/

# Volume models mount: /runpod-volume/models/{checkpoints,unet,clip,vae,loras,text_encoders,upscale_models,ipadapter-flux}
# Models loaded lazily on first handler call (warm-cache).

CMD ["python3", "-u", "/app/handler.py"]
