import os
import base64
import requests
import time
from pathlib import Path
from dotenv import load_dotenv 
from modules.audio_utils import compress_audio_to_ogg_bytes
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
        model: ModelType,
        target_lang: TargetLanguage,
        audio: Path | None = None,
        audio_url: str | None = None,
        human_transcription: str | None = None,
        source_lang: str = "auto",
    ) -> TranscriptionOutput | dict[str, str]:
        
        start_client_time = time.perf_counter()
        req_start = time.perf_counter()

        payload = {
            "model": "omniASR_LLM_3B",
            "source_lang": source_lang
        }

        # Network request to Modal Endpoint
        if audio_url:
            print(f"\n[ModalTranscriber] Using direct S3 URL: {audio_url}")
            payload["audio_url"] = audio_url

            audio_format = audio_url.split('.')[-1]
            if len(audio_format) > 4:
                audio_format = "wav" 
            payload["audio_format"] = audio_format

        elif audio:
            print(f"\n[ModalTranscriber] Starting process for {audio.name}...")
            print("[ModalTranscriber] Compressing audio to 16kHz Mono...")

            audio_bytes = compress_audio_to_ogg_bytes(audio)
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

            print(f"[ModalTranscriber] Compressed Base64 size: {len(audio_b64) / 1024:.2f} KB")

            payload["audio_base64"] = audio_b64
            payload["audio_format"] = "ogg"
        
        else:
            raise ValueError("[ERROR] Must include 'audio' (local file) or 'audio_url'.")

        print("[ModalTranscriber] Uploading to Modal Endpoint...")
        req_start = time.perf_counter()
        
        resp = requests.post(self.api_url, headers=self.headers, json=payload, timeout=600)
        
        req_end = time.perf_counter()
        total_request_time = req_end - req_start

        if resp.status_code != 200:
            error_msg = f"Modal process failed with status {resp.status_code}: {resp.text}"
            print(f"[ModalTranscriber] ERROR: {error_msg}")
            raise Exception(error_msg)
        
        output_data = resp.json()

        if "error" in output_data:
            error_msg = f"Rejected by API Modal: {output_data['error']}"
            print(f"[ModalTranscriber] {error_msg}")
            raise Exception(error_msg)

        # Calculate performance metrics
        modal_exec_time = output_data.get("modal_execution_time", 0)
        network_latency = total_request_time - modal_exec_time

        print("[PERFORMANCE LOG]")
        print(f"1. Local Compression Time : {req_start - start_client_time:.2f} s")
        print(f"2. Upload/Network Latency : {network_latency:.2f} s")
        print(f"3. Modal Server Execution : {modal_exec_time:.2f} s")

        return TranscriptionOutput(
            transcript=output_data.get("transcript", ""),
            language_selected=output_data.get("language_detected", ""),
            chunks=output_data.get("chunks", []),
            final_text=output_data.get("transcript", "")
        )