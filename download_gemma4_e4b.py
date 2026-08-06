import modal

download_image = (
    modal.Image.debian_slim()
    .pip_install("huggingface_hub", "hf_transfer")
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "HF_HOME": "/cache"
    })
)

app = modal.App("gemma4-e4b-downloader")
gemma_volume = modal.Volume.from_name("gemma4-e4b-vllm", create_if_missing=True)

MODEL_ID = "google/gemma-4-E4B-it"


@app.function(
    image=download_image,
    volumes={"/cache": gemma_volume},
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
def download_model():
    from huggingface_hub import snapshot_download

    print(f"Start download {MODEL_ID} to Modal...")

    snapshot_download(repo_id=MODEL_ID, cache_dir="/cache")

    gemma_volume.commit()
    print("Download is complete!")


@app.local_entrypoint()
def main():
    download_model.remote()
