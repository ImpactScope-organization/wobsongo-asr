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
        "accelerate"
    )
    .add_local_dir("training/whisper", remote_path="/root/training/whisper")
)

app = modal.App("whisper-dioula-finetune-multigpu")

dataset_volume = modal.Volume.from_name("dioula-dataset")
model_volume = modal.Volume.from_name("finetuned-model", create_if_missing=True)
bambara_model_volume = modal.Volume.from_name("bambara-model-base") 

NUM_GPUS = 8


@app.function(
    image=whisper_image,
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=86400,
    volumes={
        "/data": dataset_volume,
        "/output": model_volume,
        "/models": bambara_model_volume,
    },
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_whisper_multigpu(script_name: str):
    from torch.distributed.run import parse_args, run

    print(f"[SINGLE-NODE] Running script: {script_name} on {NUM_GPUS} GPU(s)")

    run(parse_args([
        "--nnodes=1",
        "--node-rank=0",
        "--master-addr=localhost",
        "--master-port=1234",
        f"--nproc-per-node={NUM_GPUS}",
        f"/root/training/whisper/{script_name}",
    ]))


@app.local_entrypoint()
def main(script_name: str = "train_whisper_large_core.py"):
    """
    CLI command to run the training script on Modal with multi-GPU support. Usage:
      modal run train_whisper_launcher.py --script-name train_whisper_large_core.py
      modal run train_whisper_launcher.py --script-name train_whisper_large_no_aug_core.py
      modal run train_whisper_launcher.py --script-name train_whisper_small_core.py
      modal run train_whisper_launcher.py --script-name train_whisper_small_no_aug_core.py
      modal run train_whisper_launcher.py --script-name train_whisper_bambara_base_core.py
    """
    valid_scripts = {
        "train_whisper_large_core.py",
        "train_whisper_large_no_aug_core.py",
        "train_whisper_small_core.py",
        "train_whisper_small_no_aug_core.py",
         "train_whisper_bambara_base_core.py",
    }
    assert script_name in valid_scripts, f"script_name must be one of: {valid_scripts}"
    train_whisper_multigpu.remote(script_name)