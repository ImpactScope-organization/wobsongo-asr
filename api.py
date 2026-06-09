import base64
import json
import os
import tempfile
import urllib.request
import modal
from typing import Dict, Any
import time
from modules.audio_utils import split_audio, get_audio_duration_ms

image = (modal.Image.from_dockerfile("modal.Dockerfile")
         .add_local_python_source("modules")
)

app = modal.App("wobsongo-asr")
volume_model = modal.Volume.from_name("wobsongo-model-asr", create_if_missing=False)


CHUNK_DURATION_MS = 30 * 1000
OVERLAP_MS = 1_500
DEFAULT_MODEL = "omniASR_LLM_3B"

LLM_MODELS = {"omniASR_LLM_3B"}
UNLIMITED_MODELS = set()

LANGUAGE_CODES = {
    "auto": None,
    "moore": "mos_Latn",
    "dioula": "dyu_Latn",
    "french": "fra_Latn",
    "english": "eng_Latn",
}
ALL_LANG_CODES = ["mos_Latn", "dyu_Latn", "fra_Latn", "eng_Latn"]


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


@app.cls(
    image=image,
    gpu="A10G", 
    memory=32768,
    volumes={"/root/.cache/fairseq2": volume_model},
    timeout=1800,
    scaledown_window=300
)
class ASREndpoint:
    
    @modal.enter()
    def load_model(self):
        print("\nStart profiling cold start")
        start_total = time.perf_counter()

        t0 = time.perf_counter()
        os.environ["FAIRSEQ2_CACHE_DIR"] ="/root/.cache/fairseq2"
        os.environ["XDG_CACHE_HOME"] = "/root/.cache"

        t1 = time.perf_counter()
        print(f"[PROFILING] Environment setup: {t1 - t0:.2f} seconds")

        print(f"Loading model {DEFAULT_MODEL} into VRAM...")

        print("[PROFILING] Import library ASRInferencePipeline...")
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

        
        t2 = time.perf_counter()
        print(f"[PROFILING] Import library done: {t2 - t1:.2f} seconds")

        print(f"[PROFILING] Load model {DEFAULT_MODEL} from volume to VRAM GPU...")
        self.pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL, device="cuda")
        t3 = time.perf_counter()
        print(f"[PROFILING] Finish loading to GPU {t3 - t2:.2f} seconds")

        print(f"[PROFILING] Total cold start time (load_model): {t3 - start_total:.2f} seconds")
        print("Profiling done...\n")

        print("Model ready!")

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, inp: Dict[str, Any]) -> dict:
        start_exec = time.perf_counter()

        print("\nStart Profiling Execution")

        audio_url = inp.get("audio_url")
        audio_b64 = inp.get("audio_base64")
        audio_fmt = inp.get("audio_format", "wav")

        if not audio_url and not audio_b64:
            return {"error": "Provide either 'audio_url' or 'audio_base64'"}

        tmp_audio = tempfile.NamedTemporaryFile(suffix=f".{audio_fmt}", delete=False, mode="wb")
        
        try:
            if audio_url:
                urllib.request.urlretrieve(audio_url, tmp_audio.name)
            else:
                tmp_audio.write(base64.b64decode(audio_b64))
            tmp_audio.flush()
            tmp_audio.close()

            t1 = time.perf_counter()
            print(f"[EXEC PROFILING] Decode Audio: {t1 - start_exec:.2f} s")

            language = inp.get("source_lang", "auto")
            if language not in LANGUAGE_CODES:
                return {"error": f"Unknown language '{language}'. Valid: {list(LANGUAGE_CODES)}"}

            lang_code = LANGUAGE_CODES[language]
            candidate_langs = [lang_code] if lang_code else ALL_LANG_CODES

            duration_ms = get_audio_duration_ms(tmp_audio.name)
            print(f"Audio Duration: {duration_ms / 1000:.1f}s")

            t2 = time.perf_counter()
            if duration_ms <= CHUNK_DURATION_MS:
                print("Short audio — single inference")
                chunks = [tmp_audio.name]
                is_temporary_chunks = False
                t2_chunking_done = time.perf_counter()
            else:
                print("Long audio — chunking")
                chunks = split_audio(tmp_audio.name)
                is_temporary_chunks = False
                t2_chunking_done = time.perf_counter()
                print(f"[EXEC PROFILING] Chunking Process: {t2_chunking_done - t1:.2f} s")
            
            best_results_per_chunk = [{"text": "[EMPTY]", "score": -999, "language": candidate_langs[0]} for _ in chunks]

            try:
                for lang in candidate_langs:
                    try:
                        print(f"Processing batch inference for language: {lang}")
                        batch_results = self.pipeline.transcribe(chunks, lang=[lang] * len(chunks))

                        for idx, result in enumerate(batch_results):
                            text = extract_text([result] if not isinstance(result, list) else result).strip()
                            score = score_text(text)

                            if score > best_results_per_chunk[idx]["score"]:
                                best_results_per_chunk[idx] = {
                                    "text": text or "[EMPTY]",
                                    "score": score,
                                    "language": lang
                                }
                    except Exception as exc:
                        print(f"  [{lang}] BATCH INFERENCE ERROR: {exc}")

            finally:
                if is_temporary_chunks:
                    for cp in chunks:
                        if os.path.exists(cp):
                            os.remove(cp)

            t3 = time.perf_counter()
            print(f"[EXEC PROFILING] Total Inference {len(candidate_langs)} Lang: {t3 - t2_chunking_done if duration_ms > CHUNK_DURATION_MS else t3 - t2:.2f} s")

            # Merge results
            final_text = ""
            chunk_results_response = []
            for i, res in enumerate(best_results_per_chunk):
                text = res["text"]
                final_text = text if i == 0 else final_text + " " + remove_duplicate_tail(final_text, text)
                chunk_results_response.append({"text": text, "language": res["language"]})
            
            lang_counts = {}
            for res in best_results_per_chunk:
                lang_counts[res["language"]] = lang_counts.get(res["language"], 0) + 1
            dominant_lang = max(lang_counts, key=lang_counts.__getitem__)

            t4 = time.perf_counter()
            print(f"[EXEC PROFILING] Merging & Result: {t4 - t3:.2f} s")
            print(f"[EXEC PROFILING] Total Execution time: {t4 - start_exec:.2f} s")
            print("Profiling execution time done\n")

            return {
                "transcript": final_text.strip(),
                "language_detected": dominant_lang,
                "chunks": chunk_results_response,
                "modal_execution_time": t4 - start_exec
            }

        finally:
            if os.path.exists(tmp_audio.name):
                os.remove(tmp_audio.name)