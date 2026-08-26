import modal

def download_nllb():
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
    model_name = "facebook/nllb-200-distilled-600M"
    AutoTokenizer.from_pretrained(model_name)
    AutoModelForSeq2SeqLM.from_pretrained(model_name)

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
        "sacrebleu",
        "sentencepiece"
    )
    .run_function(download_nllb)
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
def reeval_koumankan_checkpoints():
    import subprocess
    subprocess.run(
        ["python", "/root/training/whisper/reeval_koumankan.py"],
        check=True,
    )

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
def reeval_cv_checkpoints():
    import subprocess
    subprocess.run(
        ["python", "/root/training/whisper/reeval_commonvoice.py"],
        check=True,
    )


@app.local_entrypoint()
def koumankan():
    """
    CLI command to re-evaluate on Koumankan dev set.
    Usage: modal run reeval_launcher.py::koumankan
    """
    print("[INFO] Menjalankan re-evaluasi untuk dataset Koumankan...")
    reeval_koumankan_checkpoints.remote()


@app.local_entrypoint()
def commonvoice():
    """
    CLI command to re-evaluate on Common Voice test set.
    Usage: modal run reeval_launcher.py::commonvoice
    """
    print("[INFO] Menjalankan re-evaluasi untuk dataset Common Voice...")
    reeval_cv_checkpoints.remote()