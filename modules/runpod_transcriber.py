import os
import base64
import requests
import time
from pathlib import Path
from dotenv import load_dotenv 
from modules.transcriber import TranscriberProtocol, TranscriptionOutput, ModelType, TargetLanguage

load_dotenv()

class RunPodTranscriber(TranscriberProtocol):
    def __init__(self):
        # Get API Key and Endpoint ID
        self.api_key = os.getenv("RUNPOD_API_KEY")
        self.endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID")
        
        # Validation
        if not self.api_key or not self.endpoint_id:
            raise ValueError("[ERROR] RUNPOD_API_KEY or RUNPOINT_ID is not set in the .env file")

        self.base_url = f"https://api.runpod.ai/v2/{self.endpoint_id}"

        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
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
        
        print(f"[RunPodTranscriber] Processing {audio.name} to RunPod...")

        # Convert local audio files to Base64 format
        with open(audio, "rb") as f:
            audio_bytes = f.read()
        audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')
        
        # Extract file formats 
        audio_format = audio.suffix.lstrip('.') 

        # Prepare JSON Payload according to handler.py
        payload = {
            "input": {
                "audio_base64": audio_b64,
                "audio_format": audio_format,
                "model": "omniASR_LLM_3B",
                "language": "auto"
            }
        }

        # Send Request (POST)
        print("[RunPodTranscriber] Sending request to Serverless Endpoint...")
        run_url = f"{self.base_url}/run"
        resp = requests.post(run_url, headers=self.headers, json=payload)
        resp.raise_for_status() 
        
        job_id = resp.json().get("id")
        print(f"[RunPodTranscriber] Job ID received: {job_id}. Waiting for results...")

        # Poll Status (Waiting until COMPLETED)
        status_url = f"{self.base_url}/status/{job_id}"
        
        while True:
            status_resp = requests.get(status_url, headers=self.headers)
            status_resp.raise_for_status()
            status_data = status_resp.json()
            status = status_data.get("status")

            if status == "COMPLETED":
                output_data = status_data.get("output", {})
                print("[RunPodTranscriber] Transcription Completed!")
                break
            elif status in ["FAILED", "CANCELLED", "TIMED_OUT"]:
                error_msg = f"RunPod process failed with status: {status}"
                print(f"[RunPodTranscriber] ERROR: {error_msg}")
                raise Exception(error_msg)
            
            # If the status is IN_QUEUE or IN_PROGRESS, wait 3 seconds before checking again.
            time.sleep(3) 

        # Map RunPod Responses to Application TranscriptionOutput Objects
        return TranscriptionOutput(
            transcript=output_data.get("transcript", ""),
            language_selected=output_data.get("language_detected", ""),
            chunks=output_data.get("chunks", []),
            final_text=output_data.get("transcript", "")
        )