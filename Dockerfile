# Custom RunPod serverless image for FLUX + LTX 2.3 rendering.
# Base: PyTorch 2.9.1 + CUDA 13.0 + Python 3.12 + Ubuntu 24.04 (Ada/Ampere/Hopper)
FROM runpod/pytorch:1.0.3-dev-fix-image-vulnerabilities-cu1300-torch291-ubuntu2404

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    DEBIAN_FRONTEND=noninteractive

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

# Lightricks LTX-Video repo for LTX 2.3 inference (Gemma loader + tiled VAE decode)
RUN pip install --no-cache-dir --break-system-packages \
    "git+https://github.com/Lightricks/LTX-Video.git@main#egg=ltx-video" || \
    echo "LTX-Video pip install failed — handler will fallback to diffusers"

# App code
COPY handler.py models.py render.py volume.py /app/

# Volume models mount: /runpod-volume/models/{checkpoints,unet,clip,vae,loras,text_encoders,upscale_models,ipadapter-flux}
# Models loaded lazily on first handler call (warm-cache).

CMD ["python3", "-u", "/app/handler.py"]
