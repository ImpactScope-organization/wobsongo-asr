"""
RunPod Serverless handler for omnilingual-asr.

Input:
  audio_url     (str)  — public URL to download audio from
  audio_base64  (str)  — base64-encoded audio bytes (alternative to audio_url)
  audio_format  (str)  — file extension hint for pydub, e.g. "wav", "mp3" (default: "wav")
  model         (str)  — model card name (default: omniASR_LLM_3B_v2)
  language      (str)  — "auto"|"moore"|"dioula"|"french"|"english" (default: "auto")

Output:
  transcript        (str)
  language_detected (str)   — BCP-47 code of dominant detected language
  chunks            (list)  — per-chunk {"text": str, "language": str}
"""

import base64
import json
import os
import tempfile
import urllib.request

import runpod
from pydub import AudioSegment
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHUNK_DURATION_MS = 30 * 1000
OVERLAP_MS = 1_500
DEFAULT_MODEL = "omniASR_LLM_3B_v2"

LLM_MODELS = {
    "omniASR_LLM_300M",
    "omniASR_LLM_1B",
    "omniASR_LLM_3B",
    # "omniASR_LLM_300M_v2",
    # "omniASR_LLM_1B_v2",
    # "omniASR_LLM_3B_v2",
    # "omniASR_LLM_Unlimited_300M_v2",
    # "omniASR_LLM_Unlimited_1B_v2",
    # "omniASR_LLM_Unlimited_3B_v2",
}

UNLIMITED_MODELS = {m for m in LLM_MODELS if "Unlimited" in m}

LANGUAGE_CODES = {
    "auto": None,
    "moore": "mos_Latn",
    "dioula": "dyu_Latn",
    "french": "fra_Latn",
    "english": "eng_Latn",
}
ALL_LANG_CODES = ["mos_Latn", "dyu_Latn", "fra_Latn", "eng_Latn"]

# ---------------------------------------------------------------------------
# Model cache — re-used across warm invocations
# ---------------------------------------------------------------------------

_current_model_card: str | None = None
_pipeline: ASRInferencePipeline | None = None


def get_pipeline(model_card: str) -> ASRInferencePipeline:
    global _current_model_card, _pipeline
    if _pipeline is None or model_card != _current_model_card:
        print(f"Loading model: {model_card}")
        _pipeline = ASRInferencePipeline(model_card=model_card, device="cuda")
        _current_model_card = model_card
        print("Model ready.")
    return _pipeline


# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------


def extract_text(result) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return str(first)
    return str(result) if result else ""


def score_text(text: str) -> int:
    return -100 if not text or not text.strip() else len(text)


def remove_duplicate_tail(prev: str, curr: str) -> str:
    pw, cw = prev.split(), curr.split()
    limit = min(len(pw), len(cw), 10)
    for i in range(limit, 0, -1):
        if pw[-i:] == cw[:i]:
            return " ".join(cw[i:])
    return curr


def split_audio(file_path: str) -> list[str]:
    tmp_dir = tempfile.mkdtemp(prefix="asr_chunks_")
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    chunks, start, i = [], 0, 0
    base = os.path.basename(file_path)
    while start < len(audio):
        chunk = audio[start : start + CHUNK_DURATION_MS]
        path = os.path.join(tmp_dir, f"{base}_{i}.wav")
        chunk.export(path, format="wav")
        chunks.append(path)
        start += CHUNK_DURATION_MS - OVERLAP_MS
        i += 1

    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def transcribe_chunk(
    pipeline: ASRInferencePipeline, chunk_path: str, candidate_langs: list[str]
) -> dict:
    best_text, best_score, best_lang = "", -999, candidate_langs[0]
    for lang in candidate_langs:
        try:
            result = pipeline.transcribe([chunk_path], lang=[lang])
            text = extract_text(result).strip()
            s = score_text(text)
            print(f"  [{lang}] score={s} text={text[:60]!r}")
            if s > best_score:
                best_score, best_text, best_lang = s, text, lang
        except Exception as exc:
            print(f"  [{lang}] ERROR: {exc}")
    return {"text": best_text or "[EMPTY]", "language": best_lang}


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------


def handler(job: dict) -> dict:
    inp = job.get("input", {})

    # --- resolve audio to a temp file ---
    audio_url: str | None = inp.get("audio_url")
    audio_b64: str | None = inp.get("audio_base64")
    audio_fmt: str = inp.get("audio_format", "wav")

    if not audio_url and not audio_b64:
        return {"error": "Provide either 'audio_url' or 'audio_base64'"}

    tmp_audio = tempfile.NamedTemporaryFile(
        suffix=f".{audio_fmt}", delete=False, mode="wb"
    )
    try:
        if audio_url:
            print(f"Downloading audio from {audio_url}")
            urllib.request.urlretrieve(audio_url, tmp_audio.name)
        else:
            print("Decoding base64 audio")
            assert audio_b64 is not None
            tmp_audio.write(base64.b64decode(audio_b64))
        tmp_audio.flush()
        tmp_audio.close()

        # --- resolve model / language ---
        model_card = inp.get("model", DEFAULT_MODEL)
        if model_card not in LLM_MODELS:
            return {"error": f"Unknown model '{model_card}'. Valid: {sorted(LLM_MODELS)}"}

        language = inp.get("language", "auto")
        if language not in LANGUAGE_CODES:
            return {"error": f"Unknown language '{language}'. Valid: {list(LANGUAGE_CODES)}"}

        lang_code = LANGUAGE_CODES[language]
        candidate_langs = [lang_code] if lang_code else ALL_LANG_CODES

        pipeline = get_pipeline(model_card)
        is_unlimited = model_card in UNLIMITED_MODELS

        # --- transcribe ---
        chunk_results: list[dict] = []

        if is_unlimited:
            print("Unlimited model — transcribing full audio")
            chunk_results.append(transcribe_chunk(pipeline, tmp_audio.name, candidate_langs))
        else:
            audio_seg = AudioSegment.from_file(tmp_audio.name)
            duration_ms = len(audio_seg)
            print(f"Duration: {duration_ms / 1000:.1f}s")

            if duration_ms <= CHUNK_DURATION_MS:
                print("Short audio — single inference")
                chunk_results.append(
                    transcribe_chunk(pipeline, tmp_audio.name, candidate_langs)
                )
            else:
                print("Long audio — chunking")
                chunks = split_audio(tmp_audio.name)
                try:
                    for idx, cp in enumerate(chunks):
                        print(f"Chunk {idx + 1}/{len(chunks)}")
                        chunk_results.append(transcribe_chunk(pipeline, cp, candidate_langs))
                finally:
                    for cp in chunks:
                        if os.path.exists(cp):
                            os.remove(cp)

        # --- merge ---
        final_text = ""
        for i, c in enumerate(chunk_results):
            text = c["text"]
            final_text = text if i == 0 else final_text + " " + remove_duplicate_tail(final_text, text)
        final_text = final_text.strip()

        lang_counts: dict[str, int] = {}
        for c in chunk_results:
            lang_counts[c["language"]] = lang_counts.get(c["language"], 0) + 1
        dominant_lang = max(lang_counts, key=lang_counts.__getitem__)

        return {
            "transcript": final_text,
            "language_detected": dominant_lang,
            "chunks": chunk_results,
        }

    finally:
        if os.path.exists(tmp_audio.name):
            os.remove(tmp_audio.name)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Pre-warm default model on container start
    print(f"Pre-warming model {DEFAULT_MODEL} ...")
    get_pipeline(DEFAULT_MODEL)
    print("Starting RunPod serverless worker")
    runpod.serverless.start({"handler": handler})
