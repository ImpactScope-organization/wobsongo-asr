import os
import base64
import requests
import time
from pathlib import Path
from dotenv import load_dotenv 
from modules.transcriber import TranscriberProtocol, TranscriptionOutput, ModelType, TargetLanguage

load_dotenv()

class ModalTranscriber(TranscriberProtocol):
    def __init__(self):
        # Get API Key and Endpoint ID
        self.api_url = os.getenv("MODAL_API_URL")
        
        # Validation
        if not self.api_url:
            raise ValueError("[ERROR] MODAL_API_URL is not set inthe .env")


        self.headers = {
            "Content-Type": "application/json"
        }

    def _map_language(self, target_lang: TargetLanguage) -> str:
        if target_lang == TargetLanguage.ENGLISH:
            return "english"
        elif target_lang == TargetLanguage.FRENCH:
            return "french"
        return "auto"

    def transcribe(
        self, 
        audio: Path,
        model: ModelType,
        target_lang: TargetLanguage,
        human_transcription: str | None = None,
    ) -> TranscriptionOutput | dict[str, str]:
        
        print(f"[ModalTranscriber] Processing {audio.name} to Modal...")

        # Convert local audio files to Base64 format
        with open(audio, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Extract file formats 
        audio_format = audio.suffix.lstrip('.') 

        lang_code = self._map_language(target_lang)

        # Prepare JSON Payload according to handler.py
        payload = {
            "audio_base64": audio_b64,
            "audio_format": audio_format,
            "model": "omniASR_LLM_3B",
            "language": "auto"
        }

        # Send Request (POST)
        print("[ModalTranscriber] Sending request to Modal Endpoint...")
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=600)
        
        if resp.status_code != 200:
            error_msg = f"Modal process failed with status {resp.status_code}: {resp.text}"
            print(f"[ModalTranscriber] ERROR: {error_msg}")
            raise Exception(error_msg)
        
        output_data = resp.json()

        if "error" in output_data:
            error_msg = f"Ditolak oleh Modal API: {output_data['error']}"
            print(f"[ModalTranscriber] {error_msg}")
            raise Exception(error_msg)

        print("[ModalTranscriber] Transcription Completed!")

        return TranscriptionOutput(
            transcript=output_data.get("transcript", ""),
            language_selected=output_data.get("language_detected", ""),
            chunks=output_data.get("chunks", []),
            final_text=output_data.get("transcript", "")
        )