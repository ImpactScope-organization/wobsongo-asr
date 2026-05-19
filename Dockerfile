FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install pydub omnilingual-asr runpod

RUN python - <<'EOF'
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
print("Downloading omniASR_LLM_3B_v2 weights ...")
ASRInferencePipeline(model_card="omniASR_LLM_3B_v2", device="cpu")
print("Done.")
EOF

COPY handler.py .
CMD ["python", "-u", "handler.py"]
