FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System deps for audio processing + Python 3.12
# The base image ships Python 3.13; omnilingual-asr==0.2.0 requires <=3.12.
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
 && add-apt-repository ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    ffmpeg \
    libsndfile1 \
    libtbb-dev \
 && rm -rf /var/lib/apt/lists/*

# Bootstrap pip for Python 3.12 and make python3.12 the default interpreter
RUN python3.12 -m ensurepip --upgrade \
 && python3.12 -m pip install --upgrade pip \
 && update-alternatives --install /usr/local/bin/python python /usr/bin/python3.12 10 \
 && update-alternatives --install /usr/local/bin/python3 python3 /usr/bin/python3.12 10

WORKDIR /app

# Install Python deps
# torch/torchaudio already present in the base image; skip re-installing them
# to keep layer size down.  We still list them in requirements.txt for local
# dev — Docker build ignores lines already satisfied by the base image.
COPY requirements.txt .
RUN python3.12 -m pip install --no-cache-dir runpod \
 && python3.12 -m pip install --no-cache-dir \
       --extra-index-url https://download.pytorch.org/whl/cu128 \
       pydub==0.25.1 \
       omnilingual-asr==0.2.0

# Pre-download the default model weights into the image so cold starts are fast.
# ASRInferencePipeline downloads from HuggingFace Hub on first init.
# We use device='cpu' here (no GPU at build time); at runtime the handler
# will re-init with device='cuda', which re-uses the cached weights.
RUN python3.12 - <<'EOF'
import os
os.environ.setdefault("FAIRSEQ2_CACHE_DIR", "/opt/fairseq2_cache")
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
print("Downloading omniASR_LLM_3B_v2 weights ...")
ASRInferencePipeline(model_card="omniASR_LLM_3B_v2", device="cpu")
print("Done.")
EOF

# Copy handler
COPY handler.py .

# Point fairseq2 at the pre-downloaded cache at runtime too
ENV FAIRSEQ2_CACHE_DIR=/opt/fairseq2_cache

CMD ["python3.12", "-u", "handler.py"]
