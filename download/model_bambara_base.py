import modal

app = modal.App("bambara-model-downloader")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "transformers",
        "peft",
        "torch",
        "huggingface_hub",
    )
)

model_volume = modal.Volume.from_name("bambara-model-base", create_if_missing=True)

hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    volumes={"/models": model_volume},
    secrets=[hf_secret],
    timeout=3600,
)
def download_model():
    import os
    from transformers import WhisperForConditionalGeneration, WhisperProcessor
    from peft import PeftModel

    hf_token = os.environ.get("HF_TOKEN")

    base_model_path = "/models/whisper-large-v2-base"
    if os.path.exists(base_model_path):
        print(f"The base model is already at {base_model_path}, skipping download.")
    else:
        print("Downloading openai/whisper-large-v2...")
        base_model = WhisperForConditionalGeneration.from_pretrained(
            "openai/whisper-large-v2", token=hf_token
        )
        processor = WhisperProcessor.from_pretrained(
            "openai/whisper-large-v2", language="french", task="transcribe", token=hf_token
        )
        base_model.save_pretrained(base_model_path)
        processor.save_pretrained(base_model_path)
        print(f"Base model saved to {base_model_path}")

    adapter_path = "/models/bambara-asr-v2-adapter"
    if os.path.exists(adapter_path):
        print(f"The Bambara adapter is already at {adapter_path}, skipping download.")
    else:
        print("Downloading adapter sudoping01/bambara-asr-v2...")
        base_for_adapter = WhisperForConditionalGeneration.from_pretrained(base_model_path)
        peft_model = PeftModel.from_pretrained(
            base_for_adapter, "sudoping01/bambara-asr-v2", token=hf_token
        )
        peft_model.save_pretrained(adapter_path)
        print(f"Adapter Bambara saved to {adapter_path}")

    model_volume.commit()
    print("Done. Model and adapter saved to the volume.")


@app.local_entrypoint()
def main():
    download_model.remote()