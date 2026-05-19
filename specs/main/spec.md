# wobsongo-asr — RunPod Serverless Worker Spec

## Problem

Package the `omnilingual-asr` inference pipeline as a scalable, GPU-backed HTTP endpoint
for transcribing audio in Moore, Dioula, French, and English.

### Constraints

- `omnilingual-asr==0.1.0` requires `fairseq2[arrow]>=0.5.2,<=0.6`
- `fairseq2n==0.5.2` hard-pins `torch==2.8.0`
- `torch==2.8.0` is only published as `+cu128` (CUDA 12.8) wheels
- Cannot use Cog/Replicate: Cog's cuDNN table doesn't include CUDA 12.8 reliably
- Dev machine is ARM macOS — cannot build x86+CUDA images locally

## Solution: RunPod Serverless

Use RunPod Serverless with a custom Docker image.

- **Base image**: `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
  (ships torch 2.8.0+cu128, CUDA 12.8.1, Ubuntu 24.04)
- **Handler**: `handler.py` using the `runpod` Python SDK
- **Registry**: `ghcr.io/impactscope/wobsongo-asr`
- **Model baked into image**: `omniASR_LLM_3B_v2` (default); others download at runtime

## Input Schema

```json
{
  "audio_url": "https://...",          // Public URL to audio file (preferred)
  "audio_base64": "...",               // OR base64-encoded audio bytes
  "audio_format": "wav",               // hint for pydub (default: auto-detect)
  "model": "omniASR_LLM_3B_v2",        // optional, default omniASR_LLM_3B_v2
  "language": "auto"                   // "auto"|"moore"|"dioula"|"french"|"english"
}
```

## Output Schema

```json
{
  "transcript": "...",
  "language_detected": "fra_Latn",
  "chunks": [{"text": "...", "language": "fra_Latn"}, ...]
}
```

## Language Enum

| Key       | BCP-47 code |
|-----------|-------------|
| `auto`    | tries all 4, picks longest result |
| `moore`   | `mos_Latn` |
| `dioula`  | `dyu_Latn` |
| `french`  | `fra_Latn` |
| `english` | `eng_Latn` |

## Model Enum (LLM-conditioned only, ≤3B)

- `omniASR_LLM_300M`
- `omniASR_LLM_1B`
- `omniASR_LLM_3B`
- `omniASR_LLM_300M_v2`
- `omniASR_LLM_1B_v2`
- `omniASR_LLM_3B_v2` ← **default, baked into image**
- `omniASR_LLM_Unlimited_300M_v2`
- `omniASR_LLM_Unlimited_1B_v2`
- `omniASR_LLM_Unlimited_3B_v2`

## Chunking Strategy

- Non-Unlimited models: split audio into 30s chunks with 1.5s overlap
- Unlimited models: pass full audio directly
- Chunk transcripts merged with overlap-deduplication on word boundary

## CI/CD

GitHub Actions workflow (`.github/workflows/deploy.yml`):
1. Trigger: push to `main`, `workflow_dispatch`
2. Build Docker image for `linux/amd64`
3. Push to `ghcr.io/impactscope/wobsongo-asr:latest` (and `:sha-<SHA>`)
4. Secrets needed: `GHCR_TOKEN` (GitHub PAT with `write:packages`)

## TODO

- [x] Write `predict.py` (Cog — legacy, kept for reference)
- [x] Write `cog.yaml` (legacy)
- [ ] Write `handler.py` (RunPod Serverless)
- [ ] Write `Dockerfile`
- [ ] Update `deploy.yml` for GHCR
- [ ] Test handler locally with `python handler.py --test_input '{"input": {...}}'`
- [ ] Create RunPod endpoint pointing to image
