import json
import os
import time
import tempfile
from typing import Optional

from cog import BasePredictor, Input, Path
from pydub import AudioSegment
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CHUNK_DURATION_MS = 30 * 1000
OVERLAP_MS = 1500
TEMP_CHUNK_DIR = "/tmp/asr_chunks"

LLM_MODELS = [
    "omniASR_LLM_300M",
    "omniASR_LLM_1B",
    "omniASR_LLM_3B",
    "omniASR_LLM_300M_v2",
    "omniASR_LLM_1B_v2",
    "omniASR_LLM_3B_v2",
    "omniASR_LLM_Unlimited_300M_v2",
    "omniASR_LLM_Unlimited_1B_v2",
    "omniASR_LLM_Unlimited_3B_v2",
]

UNLIMITED_MODELS = {m for m in LLM_MODELS if "Unlimited" in m}

LANGUAGE_CODES = {
    "auto": None,
    "moore": "mos_Latn",
    "dioula": "dyu_Latn",
    "french": "fra_Latn",
    "english": "eng_Latn",
}

ALL_LANG_CODES = ["mos_Latn", "dyu_Latn", "fra_Latn", "eng_Latn"]

DEFAULT_MODEL = "omniASR_LLM_3B_v2"


def score_text(text: str) -> int:
    if not text or text.strip() == "":
        return -100
    return len(text)


def extract_text(result) -> str:
    if isinstance(result, list) and len(result) > 0:
        first = result[0]
        if isinstance(first, dict):
            return first.get("text", "")
        elif isinstance(first, str):
            return first
        else:
            return str(first)
    elif isinstance(result, str):
        return result
    return str(result)


def remove_duplicate_tail(prev_text: str, curr_text: str) -> str:
    prev_words = prev_text.split()
    curr_words = curr_text.split()
    max_overlap = min(len(prev_words), len(curr_words), 10)
    for i in range(max_overlap, 0, -1):
        if prev_words[-i:] == curr_words[:i]:
            return " ".join(curr_words[i:])
    return curr_text


def split_audio(file_path: str) -> list[str]:
    os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    chunks = []
    start = 0
    i = 0
    base = os.path.basename(file_path)

    while start < len(audio):
        end = start + CHUNK_DURATION_MS
        chunk = audio[start:end]
        chunk_path = os.path.join(TEMP_CHUNK_DIR, f"{base}_{i}.wav")
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
        start += CHUNK_DURATION_MS - OVERLAP_MS
        i += 1

    print(f"Split into {len(chunks)} chunk(s)")
    return chunks


def transcribe_chunk(
    pipeline: ASRInferencePipeline, chunk_path: str, candidate_langs: list[str]
) -> dict:
    """
    Transcribe a single chunk. If multiple candidate_langs, pick best by score_text.
    Returns {"text": str, "language": str}
    """
    best_text = ""
    best_score = -999
    best_lang = candidate_langs[0]

    for lang in candidate_langs:
        try:
            result = pipeline.transcribe([chunk_path], lang=[lang])
            text = extract_text(result).strip()
            s = score_text(text)
            print(f"  [{lang}] score={s} text={text[:40]!r}")
            if s > best_score:
                best_score = s
                best_text = text
                best_lang = lang
        except Exception as e:
            print(f"  [{lang}] ERROR: {e}")

    return {"text": best_text or "[EMPTY]", "language": best_lang}


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Pre-load the default model to warm cold starts."""
        print(f"Loading default model {DEFAULT_MODEL} ...")
        self._current_model_card = DEFAULT_MODEL
        self._pipeline = ASRInferencePipeline(
            model_card=DEFAULT_MODEL,
            device="cuda",
        )
        print("Model loaded.")

    def _get_pipeline(self, model_card: str) -> ASRInferencePipeline:
        if model_card != self._current_model_card:
            print(f"Switching model: {self._current_model_card} → {model_card}")
            self._pipeline = ASRInferencePipeline(
                model_card=model_card,
                device="cuda",
            )
            self._current_model_card = model_card
        return self._pipeline

    def predict(
        self,
        audio: Path = Input(
            description="Input audio file (mp3, wav, ogg, flac, m4a, etc.)"
        ),
        model: str = Input(
            description="LLM-conditioned model card",
            default=DEFAULT_MODEL,
            choices=LLM_MODELS,
        ),
        language: str = Input(
            description=(
                "Language to transcribe. "
                "'auto' tries all supported languages and picks the best result."
            ),
            default="auto",
            choices=list(LANGUAGE_CODES.keys()),
        ),
        human_transcription: str | None = Input(
            description="Human-provided transcription for evaluation (optional)",
            default=None,
        ),
    ) -> str:
        pipeline = self._get_pipeline(model)
        is_unlimited = model in UNLIMITED_MODELS

        lang_code = LANGUAGE_CODES[language]
        candidate_langs = [lang_code] if lang_code else ALL_LANG_CODES

        file_path = str(audio)
        chunk_results = []

        if is_unlimited:
            print("Unlimited model — transcribing full audio without chunking")
            result = transcribe_chunk(pipeline, file_path, candidate_langs)
            chunk_results.append(result)
        else:
            audio_seg = AudioSegment.from_file(file_path)
            duration_ms = len(audio_seg)
            print(f"Audio duration: {duration_ms / 1000:.1f}s")

            if duration_ms <= CHUNK_DURATION_MS:
                print("Short audio — single inference")
                result = transcribe_chunk(pipeline, file_path, candidate_langs)
                chunk_results.append(result)
            else:
                print("Long audio — chunking...")
                chunks = split_audio(file_path)
                try:
                    for i, chunk_path in enumerate(chunks):
                        print(f"Chunk {i + 1}/{len(chunks)}: {chunk_path}")
                        result = transcribe_chunk(pipeline, chunk_path, candidate_langs)
                        chunk_results.append(result)
                finally:
                    for c in chunks:
                        if os.path.exists(c):
                            os.remove(c)

        # Merge chunks
        final_text = ""
        for i, chunk in enumerate(chunk_results):
            text = chunk["text"]
            if i == 0:
                final_text = text
            else:
                cleaned = remove_duplicate_tail(final_text, text)
                final_text += " " + cleaned
        final_text = final_text.strip()

        # Determine dominant language (most frequent among chunks)
        lang_counts: dict[str, int] = {}
        for chunk in chunk_results:
            lang_counts[chunk["language"]] = lang_counts.get(chunk["language"], 0) + 1
        dominant_lang = max(lang_counts, key=lang_counts.__getitem__)

        return json.dumps(
            {
                "transcript": final_text,
                "language_selected": dominant_lang,
                "chunks": chunk_results,
                "final_text": final_text,
            },
            ensure_ascii=False,
            indent=2,
        )
