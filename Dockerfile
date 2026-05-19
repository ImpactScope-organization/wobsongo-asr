FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

# System deps for audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libtbb-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
# torch/torchaudio already present in the base image; skip re-installing them
# to keep layer size down.  We still list them in requirements.txt for local
# dev — Docker build ignores lines already satisfied by the base image.
COPY requirements.txt .
RUN pip install --no-cache-dir runpod \
 && pip install --no-cache-dir \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      pydub==0.25.1 \
      omnilingual-asr==0.2.0

# Pre-download the default model weights into the image so cold starts are fast.
# ASRInferencePipeline downloads from HuggingFace Hub on first init.
# We use device='cpu' here (no GPU at build time); at runtime the handler
# will re-init with device='cuda', which re-uses the cached weights.
RUN python - <<'EOF'
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

CMD ["python", "-u", "handler.py"]
