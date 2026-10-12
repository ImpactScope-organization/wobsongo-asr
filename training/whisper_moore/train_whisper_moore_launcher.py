import modal

whisper_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch",
        "torchcodec",
        "transformers",
        "datasets",
        "librosa",
        "soundfile",
        "wandb",
        "evaluate",
        "jiwer",
        "accelerate",
    )
    .add_local_dir("training/whisper_moore", remote_path="/root/training/whisper_moore")
)

app = modal.App("whisper-moore-finetune-multigpu")

dataset_volume = modal.Volume.from_name("moore-speech-data")
model_volume = modal.Volume.from_name("moore-finetuned-model", create_if_missing=True)

NUM_GPUS = 1


@app.function(
    image=whisper_image,
    gpu=f"A100-80GB:{NUM_GPUS}",
    timeout=86400,
    volumes={
        "/data": dataset_volume,
        "/output": model_volume,
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
        f"/root/training/whisper_moore/{script_name}",
    ]))


@app.local_entrypoint()
def main(script_name: str = "train_whisper_small_moore_no_aug_core.py"):
    """
    CLI command to run the Moore training script on Modal with multi-GPU support. Usage:
      modal run train_whisper_moore_launcher.py --script-name train_whisper_small_moore_no_aug_core.py
    """
    valid_scripts = {
        "train_whisper_small_moore_no_aug_core.py",
    }
    assert script_name in valid_scripts, f"script_name must be one of: {valid_scripts}"
    train_whisper_multigpu.remote(script_name)