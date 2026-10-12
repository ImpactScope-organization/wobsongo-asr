import io
import os
import time
import wave
from dataclasses import dataclass

import requests

DEFAULT_URL = "https://wobsongo--melotts-melotts-tts.modal.run"

# Speaker IDs available in MeloTTS per language
SPEAKERS = {
    "FR": ["FR"],
    "EN": ["EN-Default", "EN-US", "EN-BR", "EN_INDIA", "EN-AU"],
}


@dataclass
class TTSResult:
    audio_bytes: bytes
    language: str
    speaker: str
    request_seconds: float  
    audio_seconds: float  


class ModalTTS:
    def __init__(self, url: str | None = None, timeout: int = 300):
        self.url = url or os.environ.get("MELOTTS_URL", DEFAULT_URL)
        self.timeout = timeout

    def synthesize(
        self,
        text: str,
        language: str = "FR",
        speaker: str | None = None,
        speed: float = 1.0,
    ) -> TTSResult:
        payload = {"text": text, "language": language, "speed": speed}
        if speaker:
            payload["speaker"] = speaker

        start = time.perf_counter()
        response = requests.post(self.url, json=payload, timeout=self.timeout)
        elapsed = time.perf_counter() - start

        if response.status_code != 200:
            try:
                detail = response.json().get("detail", response.text)
            except ValueError:
                detail = response.text
            raise RuntimeError(f"MeloTTS endpoint error ({response.status_code}): {detail}")

        with wave.open(io.BytesIO(response.content), "rb") as wav:
            audio_seconds = wav.getnframes() / wav.getframerate()

        return TTSResult(
            audio_bytes=response.content,
            language=language,
            speaker=speaker or SPEAKERS[language][0],
            request_seconds=elapsed,
            audio_seconds=audio_seconds,
        )