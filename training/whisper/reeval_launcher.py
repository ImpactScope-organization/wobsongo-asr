import modal

whisper_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchcodec",
        "transformers",
        "datasets",
        "peft",
        "librosa",
        "soundfile",
        "wandb",
        "evaluate",
        "jiwer",
        "accelerate",
        "pandas",
    )
    .add_local_dir("training/whisper", remote_path="/root/training/whisper")
)

app = modal.App("whisper-dioula-reeval-checkpoints")

dataset_volume = modal.Volume.from_name("dioula-dataset")
model_volume = modal.Volume.from_name("finetuned-model")
bambara_model_volume = modal.Volume.from_name("bambara-model-base")


@app.function(
    image=whisper_image,
    gpu="A100-80GB:1",
    timeout=86400,
    volumes={
        "/data": dataset_volume,
        "/output": model_volume,
        "/models": bambara_model_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def reeval_checkpoints():
    import subprocess
    subprocess.run(
        ["python", "/root/training/whisper/reeval_checkpoints_core.py"],
        check=True,
    )


@app.local_entrypoint()
def main():
    """
    CLI command to re-evaluate all existing checkpoints with the clean
    (non-augmented) collator on the dev set. Usage:
      modal run reeval_launcher.py
    """
    reeval_checkpoints.remote()