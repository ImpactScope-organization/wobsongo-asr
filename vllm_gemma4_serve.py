import json
import modal



MODEL_NAME = "google/gemma-4-E4B-it"
MODEL_REVISION = None 
MAX_CONCURRENT_REQUESTS = 10
MAX_MODEL_LEN = 16384  
MAX_IMAGES_PER_PROMPT = 4 
N_GPU = 1
GPU_TYPE = "L40S"  
MINUTES = 60
VLLM_PORT = 8000
FAST_BOOT = False  

vllm_image = (
    modal.Image.from_registry("nvidia/cuda:13.0.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install(
        "vllm",
        pre=True,
        extra_options=(
            "--extra-index-url https://wheels.vllm.ai/nightly/cu130 "
            "--extra-index-url https://download.pytorch.org/whl/cu130 "
            "--index-strategy unsafe-best-match"
        ),
    )
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_LOG_STATS_INTERVAL": "1",
        }
    )
)

hf_cache_vol = modal.Volume.from_name("huggingface-cache", create_if_missing=True)
vllm_cache_vol = modal.Volume.from_name("vllm-cache", create_if_missing=True)

app = modal.App("wobsongo-gemma4-vllm")


@app.server(
    image=vllm_image,
    gpu=f"{GPU_TYPE}:{N_GPU}",
    scaledown_window=5 * MINUTES,
    startup_timeout=10 * MINUTES,
    volumes={
        "/root/.cache/huggingface": hf_cache_vol,
        "/root/.cache/vllm": vllm_cache_vol,
    },
    port=VLLM_PORT,
    target_concurrency=MAX_CONCURRENT_REQUESTS,
    max_containers=3,  
    secrets=[modal.Secret.from_name("vllm-gemma4-api-key")], 
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        import os
        import subprocess

        cmd = [
            "vllm",
            "serve",
            MODEL_NAME,
            "--served-model-name",
            MODEL_NAME,
            "gemma4-e-4b", 
            "--host",
            "0.0.0.0",
            "--port",
            str(VLLM_PORT),
            "--uvicorn-log-level=info",
            "--async-scheduling",
            "--max-model-len",
            str(MAX_MODEL_LEN),
            "--gpu-memory-utilization",
            "0.90",
            "--max-num-seqs",
            str(MAX_CONCURRENT_REQUESTS),
            "--limit-mm-per-prompt",
            json.dumps({"image": MAX_IMAGES_PER_PROMPT, "video": 0, "audio": 0}),
            "--mm-processor-kwargs",
            json.dumps({"max_soft_tokens": 280}),
            "--max-num-batched-tokens",
            str(MAX_MODEL_LEN),
            "--api-key",
            os.environ["VLLM_API_KEY"], 
        ]

        if MODEL_REVISION:
            cmd += ["--revision", MODEL_REVISION]

        cmd += ["--enforce-eager" if FAST_BOOT else "--no-enforce-eager"]
        cmd += ["--tensor-parallel-size", str(N_GPU)]

        print(*cmd)
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
