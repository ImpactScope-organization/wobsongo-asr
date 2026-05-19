FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install pydub omnilingual-asr runpod 
COPY handler.py .
CMD ["python", "-u", "handler.py"]
