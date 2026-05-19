FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install pydub omnilingual-asr runpod

# Pre-download model weights directly into fairseq2's cache so cold starts
# don't need to download at runtime.  Paths are derived from:
#   sha1(url).hexdigest()[:24] / filename
# (see fairseq2 StandardAssetDownloadManager._get_uri_hash)
RUN mkdir -p \
      /root/.cache/fairseq2/assets/8e3cc13350b150509589afd0 \
      /root/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce \
 && wget -q -O /root/.cache/fairseq2/assets/8e3cc13350b150509589afd0/omniASR-LLM-3B-v2.pt \
      https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B-v2.pt \
 && wget -q -O /root/.cache/fairseq2/assets/e7be1a6acb8f76fdbca19dce/omniASR_tokenizer_written_v2.model \
      https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer_written_v2.model

COPY handler.py .
CMD ["python", "-u", "handler.py"]
