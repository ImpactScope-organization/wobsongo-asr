import modal

image = modal.Image.debian_slim().pip_install("transformers", "torch")
app = modal.App("download-base-whisper")

model_volume = modal.Volume.from_name("finetuned-model")

@app.function(
    image=image,
    volumes={"/output": model_volume},
    timeout=1800
)
def download_original_models():
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    print("Downloading Whisper Small Original...")
    small_path = "/output/whisper-small-original"
    model_small = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
    processor_small = WhisperProcessor.from_pretrained("openai/whisper-small")
    model_small.save_pretrained(small_path)
    processor_small.save_pretrained(small_path)

    print("Downloading Whisper Large-V3 Original...")
    large_path = "/output/whisper-large-v3-original"
    model_large = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3")
    processor_large = WhisperProcessor.from_pretrained("openai/whisper-large-v3")
    model_large.save_pretrained(large_path)
    processor_large.save_pretrained(large_path)

    model_volume.commit()
    print("All models have been saved")

@app.local_entrypoint()
def main():
    download_original_models.remote()