FROM runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404

RUN pip install pydub omnilingual-asr runpod

# Pre-download model weights directly into fairseq2's cache so cold starts
# don't need to download at runtime.  Paths are derived from:
#   sha1(url).hexdigest()[:24] / filename
# (see fairseq2 StandardAssetDownloadManager._get_uri_hash)
RUN mkdir -p \
      /root/.cache/fairseq2/assets/01fd052e87486e6e4d742fdf \
      /root/.cache/fairseq2/assets/b86047ffa9089216c2972a21 \
 && wget -q -O /root/.cache/fairseq2/assets/01fd052e87486e6e4d742fdf/omniASR-LLM-3B.pt \
      https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt \
 && wget -q -O /root/.cache/fairseq2/assets/b86047ffa9089216c2972a21/omniASR_tokenizer.model \
      https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model

COPY handler.py .
CMD ["python", "-u", "handler.py"]
