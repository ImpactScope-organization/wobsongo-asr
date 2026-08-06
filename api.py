import base64
import os
import tempfile
import urllib.request
import modal
from typing import Dict, Any
import time
import gc
import shutil
from modules.audio_utils import split_audio, get_audio_duration_ms, normalize_audio

image = (modal.Image.from_dockerfile("modal.Dockerfile")
         .apt_install("ffmpeg")
         .pip_install("transformers", "accelerate", "peft")
         .add_local_python_source("modules")
)

app = modal.App("wobsongo-asr")

volume_omni = modal.Volume.from_name("wobsongo-model-asr", create_if_missing=False)
volume_whisper = modal.Volume.from_name("finetuned-model", create_if_missing=False)

CHUNK_DURATION_MS = 30 * 1000
OVERLAP_MS = 1_500
DEFAULT_MODEL = "omniASR_LLM_3B"

LANGUAGE_CODES = {
    "auto": None,
    "moore": "mos_Latn",
    "dioula": "dyu_Latn",
    "french": "fra_Latn",
    "english": "eng_Latn",
}
ALL_LANG_CODES = ["mos_Latn", "dyu_Latn", "fra_Latn", "eng_Latn"]

# Helper functions for processing ASR results
def extract_text(result) -> str:
    if isinstance(result, list) and result:
        first = result[0]
        if isinstance(first, dict):
            return first.get("text", "")
        return str(first)
    return str(result) if result else ""

def score_text(text: str) -> int:
    return -100 if not text or not text.strip() else len(text)

# Remove duplicate tail words between two strings
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
    volumes={
        "/omni_data": volume_omni,
        "/whisper_data": volume_whisper
    },
    secrets=[modal.Secret.from_name("apify-token")],
    timeout=1800,
    scaledown_window=300
)
class ASREndpoint:
    
    @modal.enter()
    def setup(self):
        print("\n[PROFILING] Endpoint container initializing...")
        
        os.environ["FAIRSEQ2_CACHE_DIR"] = "/omni_data"
        os.environ["XDG_CACHE_HOME"] = "/root/.cache"
        
        self.current_model_name = None
        self.omni_pipeline = None
        self.whisper_pipeline = None
        print("Endpoint ready for dynamic routing.\n")

    # Switch model based on the requested model name
    def _switch_model(self, requested_model: str):
        if self.current_model_name == requested_model:
            return
            
        print(f"\nSwitching AI Model to '{requested_model}'...")
        start_load = time.perf_counter()
        
        import torch
        
        if self.omni_pipeline is None:
            print("[PROFILING] Loading Omnilingual ASR (Kept in memory)...")
            from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
            self.omni_pipeline = ASRInferencePipeline(model_card=DEFAULT_MODEL, device="cuda")

        if requested_model != "Omnilingual ASR":
            
            if self.whisper_pipeline is not None:
                del self.whisper_pipeline
                self.whisper_pipeline = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()

            MODEL_PATHS = {
                "Whisper Small (Untrained)": "/whisper_data/whisper-small-original",
                "Whisper Large-V3 (Untrained)": "/whisper_data/whisper-large-v3-original",
                "Whisper Large-V3 (Augmentation)": "/whisper_data/whisper-large-v3-dioula-split8020-final",
                "Whisper Small (Augmentation)": "/whisper_data/whisper-small-dioula-split8020-final",
                "Whisper Large-V3": "/whisper_data/whisper-large-v3-NO-AUG-final",
                "Whisper Small": "/whisper_data/whisper-small-NO-AUG-final"
            }
            
            model_path = MODEL_PATHS.get(requested_model)
            if not model_path:
                raise ValueError(f"Unknown model mapping for: {requested_model}")
                
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"The Whisper folder was not found at path: {model_path}. Make sure the folder name in the Volume matches.")
            
            from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline
            
            if "Small" in requested_model:
                base_model_path = "/whisper_data/whisper-small-original"
            else:
                base_model_path = "/whisper_data/whisper-large-v3-original"

            print(f"[PROFILING] Preparing processor from base model {base_model_path}...")
            processor = WhisperProcessor.from_pretrained(base_model_path)

            if "Untrained" in requested_model:
                print(f"[PROFILING] Loading Untrained Base Model into pipeline...")
                self.whisper_pipeline = pipeline(
                    "automatic-speech-recognition", 
                    model=base_model_path, 
                    device="cuda",
                    chunk_length_s=10,
                    stride_length_s=[2, 2],
                    torch_dtype=torch.float16
                )
            else:
                from peft import PeftModel
                print(f"[PROFILING] Loading Base Model weights to GPU...")
                base_model = WhisperForConditionalGeneration.from_pretrained(
                    base_model_path, 
                    torch_dtype=torch.float16,
                    device_map="cuda"
                )
                
                print(f"[PROFILING] Attaching LoRA adapters from {model_path}...")
                model = PeftModel.from_pretrained(base_model, model_path)
                
                print(f"[PROFILING] Merging LoRA weights for inference speed...")
                model = model.merge_and_unload()
                
                print(f"[PROFILING] Injecting merged model into HF Pipeline...")
                self.whisper_pipeline = pipeline(
                    "automatic-speech-recognition",
                    model=model,
                    tokenizer=processor.tokenizer,
                    feature_extractor=processor.feature_extractor,
                    device="cuda",
                    chunk_length_s=10,
                    stride_length_s=[2, 2],
                    torch_dtype=torch.float16
                )
            
        end_load = time.perf_counter()
        self.current_model_name = requested_model
        print(f"[PROFILING] Model switch completed in {end_load - start_load:.2f} seconds")
        print(f"Model {requested_model} ready!\n")

    @modal.fastapi_endpoint(method="POST")
    def transcribe(self, inp: Dict[str, Any]) -> dict:
        print("\nStart Profiling Execution")
        start_exec = time.perf_counter()
        
        requested_model = inp.get("model", "Omnilingual ASR")
        audio_url = inp.get("audio_url")
        audio_b64 = inp.get("audio_base64")
        audio_fmt = inp.get("audio_format", "wav")

        if not audio_url and not audio_b64:
            return {"error": "Provide either 'audio_url' or 'audio_base64'"}

        try:
            self._switch_model(requested_model)
        except Exception as e:
            return {"error": f"Failed to load model: {str(e)}"}

        tmp_audio = tempfile.NamedTemporaryFile(suffix=f".{audio_fmt}", delete=False, mode="wb")

        is_temporary_chunks = False
        chunks = []
        
        try:
            if audio_url:
                apify_token = os.environ.get("APIFY_TOKEN")
                if apify_token and "api.apify.com" in audio_url:
                    sep = "&" if "?" in audio_url else "?"
                    audio_url = f"{audio_url}{sep}token={apify_token}"

                req = urllib.request.Request(
                    audio_url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'}
                )
                with urllib.request.urlopen(req) as response, open(tmp_audio.name, 'wb') as out_file:
                    shutil.copyfileobj(response, out_file)
            else:
                tmp_audio.write(base64.b64decode(audio_b64))
            tmp_audio.flush()
            tmp_audio.close()

            t1 = time.perf_counter()
            print(f"[EXEC PROFILING] Decode Audio: {t1 - start_exec:.2f} s")

            language = inp.get("source_lang", "auto")
            duration_ms = get_audio_duration_ms(tmp_audio.name)
            print(f"Audio Duration: {duration_ms / 1000:.1f}s | Source Language: {language}")

            if requested_model == "Omnilingual ASR":
                lang_code = LANGUAGE_CODES.get(language)
                candidate_langs = [lang_code] if lang_code else ALL_LANG_CODES
                
                t2 = time.perf_counter()
                if duration_ms <= CHUNK_DURATION_MS:
                    print("Short audio — single inference")
                    normalized_path = normalize_audio(tmp_audio.name)
                    chunks = [normalized_path]
                    is_temporary_chunks = True
                    t2_chunking_done = time.perf_counter()
                else:
                    print("Long audio — chunking")
                    chunks = split_audio(tmp_audio.name)
                    is_temporary_chunks = True
                    t2_chunking_done = time.perf_counter()
                    print(f"[EXEC PROFILING] Chunking Process: {t2_chunking_done - t2:.2f} s")
                
                best_results_per_chunk = [{"text": "[EMPTY]", "score": -999, "language": candidate_langs[0]} for _ in chunks]

                try:
                    for lang in candidate_langs:
                        try:
                            print(f"Processing batch inference for language: {lang}")
                            batch_results = self.omni_pipeline.transcribe(chunks, lang=[lang] * len(chunks))
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
                print(f"[EXEC PROFILING] Total Inference {len(candidate_langs)} Lang (Omni): {t3 - t2_chunking_done:.2f} s")

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

            else:
                t2 = time.perf_counter()
                generate_kwargs = {
                    "repetition_penalty": 1.1,
                    "no_repeat_ngram_size": 3,
                    "max_new_tokens": 128
                }
                
                is_finetuned_dioula = "Untrained" not in requested_model
                
                if is_finetuned_dioula:
                    generate_kwargs["language"] = "french"
                else:
                    if language in ["french", "english"]:
                        generate_kwargs["language"] = language
                
                print(f"Running Whisper Inference ({requested_model})...")
                result = self.whisper_pipeline(tmp_audio.name, generate_kwargs=generate_kwargs)
                
                t3 = time.perf_counter()
                print(f"[EXEC PROFILING] Total Inference (Whisper): {t3 - t2:.2f} s")
                
                final_text = result["text"].strip()
                
                if is_finetuned_dioula:
                    dominant_lang = "dioula"
                else:
                    dominant_lang = language if language != "auto" else "auto-detected"
                    
                chunk_results_response = [{"text": final_text, "language": dominant_lang}]

            t4 = time.perf_counter()
            print(f"[EXEC PROFILING] Merging & Result Formating: {t4 - t3:.2f} s")
            print(f"[EXEC PROFILING] Total Execution time: {t4 - start_exec:.2f} s")
            print("Profiling execution time done\n")

            return {
                "transcript": final_text.strip(),
                "language_detected": dominant_lang,
                "chunks": chunk_results_response,
                "modal_execution_time": t4 - start_exec
            }

        finally:
            if is_temporary_chunks:
                for cp in chunks:
                    if os.path.exists(cp):
                        os.remove(cp)