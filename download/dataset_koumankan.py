import modal

app = modal.App("dioula-dataset-koumankan-downloader")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "datasets",
        "huggingface_hub",
        "soundfile",
        "librosa",
    )
)

data_volume = modal.Volume.from_name("dioula-dataset")

hf_secret = modal.Secret.from_name("huggingface-secret")


@app.function(
    image=image,
    volumes={"/data": data_volume},
    secrets=[hf_secret],
    timeout=3600,
)
def download_dataset():
    import os
    from datasets import load_dataset

    hf_token = os.environ.get("HF_TOKEN")

    dataset_path = "/data/dioula-koumankan-dataset"

    if os.path.exists(dataset_path):
        print(f"Dataset already exists at {dataset_path}, skipping download.")
        return

    print("downloading uvci/koumankan4dyula (train/validation/test)...")
    dataset_train = load_dataset("uvci/koumankan4dyula", split="train", token=hf_token)
    dataset_dev = load_dataset("uvci/koumankan4dyula", split="dev", token=hf_token)
    dataset_test = load_dataset("uvci/koumankan4dyula", split="test", token=hf_token)

    dataset_train.save_to_disk(f"{dataset_path}/train")
    dataset_dev.save_to_disk(f"{dataset_path}/dev")
    dataset_test.save_to_disk(f"{dataset_path}/test")
    print(f"Dataset saved to {dataset_path}")

    data_volume.commit()
    print("Done. The dataset is saved in Volume.")


@app.local_entrypoint()
def main():
    download_dataset.remote()