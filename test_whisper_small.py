import modal
import os

test_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("torch", "transformers", "datasets", "peft", "soundfile")
)

app = modal.App("whisper-inference-test")
model_volume = modal.Volume.from_name("finetuned-model")

@app.function(
    image=test_image,
    gpu="A100", 
    volumes={"/output": model_volume}
)
def run_transcription_remote(audio_bytes: bytes, file_name: str):
    import torch
    import tempfile
    from transformers import WhisperProcessor, WhisperForConditionalGeneration, pipeline
    from peft import PeftModel

    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    print("Load Model...")
    base_model_path = "openai/whisper-small"
    adapter_path = "/output/whisper-small-dioula-split8020-final"

    processor = WhisperProcessor.from_pretrained(base_model_path, language="french", task="transcribe")
    
    base_model = WhisperForConditionalGeneration.from_pretrained(
        base_model_path, 
        torch_dtype=torch.float16, 
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, adapter_path)
    model = model.merge_and_unload() 

    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=0,
        chunk_length_s=30,
        generate_kwargs={"max_new_tokens": 40, "language": "french", "task": "transcribe"}
    )

    print(f"Processing transcription: {file_name} ...")
    result = asr_pipeline(temp_audio_path)
    
    os.unlink(temp_audio_path)
    
    return result["text"]


@app.local_entrypoint()
def main():
    local_audio_name = "dioula_audio.mp3" 
    
    local_audio_path = os.path.join(os.path.dirname(__file__), local_audio_name) if __file__ else local_audio_name

    if not os.path.exists(local_audio_path):
        print(f"Error: File '{local_audio_name}' not found")
        return

    print(f"Read local files: {local_audio_name} ({os.path.getsize(local_audio_path) / (1024*1024):.2f} MB)")
    with open(local_audio_path, "rb") as f:
        audio_bytes = f.read()

    print("Send data...")
    transcription_text = run_transcription_remote.remote(audio_bytes, local_audio_name)
    
    print("\n" + "="*50)
    print("Transcription Results")
    print("="*50)
    print(transcription_text)
    print("="*50 + "\n")