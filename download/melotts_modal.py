import io
import subprocess

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


def _wav_to_ogg_opus(wav_bytes: bytes) -> bytes:
    """Compress WAV to OGG/Opus, the voice-note format for Telegram and WhatsApp."""
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-c:a", "libopus", "-b:a", "32k", "-ar", "24000", "-ac", "1",
            "-application", "voip",
            "-f", "ogg", "pipe:1",
        ],
        input=wav_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {proc.stderr.decode(errors='ignore')[:500]}")
    return proc.stdout


@app.function(volumes={CACHE_DIR: volume}, timeout=1800)
def download_models():
    _load_models("cpu")
    volume.commit()
    print("Model stored on Volume 'melotts-model'.")


@app.cls(
    gpu="L4",
    volumes={CACHE_DIR: volume},
    scaledown_window=300,
    timeout=1800,
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
        fmt = str(item.get("format", "wav")).lower()
        if not text:
            raise HTTPException(400, "field 'text' wajib diisi")
        if lang not in self.models:
            raise HTTPException(400, f"language harus salah satu dari {LANGUAGES}")
        if fmt not in ("wav", "ogg"):
            raise HTTPException(400, "format harus 'wav' atau 'ogg'")

        model = self.models[lang]
        speaker_ids = model.hps.data.spk2id
        key = item.get("speaker") or next(iter(speaker_ids.keys()))
        if key not in speaker_ids:
            raise HTTPException(400, f"speaker tidak ada. Pilihan: {list(speaker_ids.keys())}")

        audio = model.tts_to_file(text, speaker_ids[key], None, speed=speed, quiet=True)

        buf = io.BytesIO()
        sf.write(buf, audio, model.hps.data.sampling_rate, format="WAV")
        wav_bytes = buf.getvalue()

        if fmt == "ogg":
            return Response(content=_wav_to_ogg_opus(wav_bytes), media_type="audio/ogg")
        return Response(content=wav_bytes, media_type="audio/wav")