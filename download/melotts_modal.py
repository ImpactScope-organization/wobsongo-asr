import io

import modal

APP_NAME = "melotts"
CACHE_DIR = "/cache"

volume = modal.Volume.from_name("melotts-model", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg", "libsndfile1", "build-essential")
    .pip_install("setuptools<81", "wheel")
    .pip_install("git+https://github.com/myshell-ai/MeloTTS.git", "soundfile", "fastapi[standard]")
    .pip_install("setuptools<81")
    .run_commands(
        "python -m unidic download",
        "python -c \"import nltk; nltk.download('averaged_perceptron_tagger_eng'); nltk.download('cmudict')\"",
    )
    .env(
        {
            "HF_HOME": f"{CACHE_DIR}/huggingface",
            "HF_HUB_CACHE": f"{CACHE_DIR}/huggingface/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/huggingface/hub",
        }
    )
)

app = modal.App(APP_NAME, image=image)

LANGUAGES = ["FR", "EN"]


def _load_models(device: str):
    from melo.api import TTS

    return {lang: TTS(language=lang, device=device) for lang in LANGUAGES}


@app.function(volumes={CACHE_DIR: volume}, timeout=1800)
def download_models():
    _load_models("cpu")
    volume.commit() 
    print("Model stored on Volume 'melotts-model'.")


@app.cls(
    gpu="L4",
    volumes={CACHE_DIR: volume},
    scaledown_window=300,
    timeout=600,
)
class MeloTTS:
    @modal.enter()
    def load(self):
        self.models = _load_models("cuda")

    @modal.fastapi_endpoint(method="POST")
    def tts(self, item: dict):
        import soundfile as sf
        from fastapi import HTTPException
        from fastapi.responses import Response

        text = item.get("text")
        lang = item.get("language", "FR").upper()
        speed = float(item.get("speed", 1.0))
        if not text:
            raise HTTPException(400, "field 'text' is required")
        if lang not in self.models:
            raise HTTPException(400, f"language must be one of {LANGUAGES}")

        model = self.models[lang]
        speaker_ids = model.hps.data.spk2id
        key = item.get("speaker") or next(iter(speaker_ids.keys()))
        if key not in speaker_ids:
            raise HTTPException(400, f"speaker not found. Options: {list(speaker_ids.keys())}")

        audio = model.tts_to_file(text, speaker_ids[key], None, speed=speed, quiet=True)

        buf = io.BytesIO()
        sf.write(buf, audio, model.hps.data.sampling_rate, format="WAV")
        return Response(content=buf.getvalue(), media_type="audio/wav")