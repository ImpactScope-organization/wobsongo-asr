import modal

image = modal.Image.from_registry(
    "quay.io/docling-project/docling-serve-cu130:v1.18.0"
)

app = modal.App("docling-serve")

@app.function(
    image=image,
    gpu="A10G",
    timeout=600,
    env={
        "DOCLING_SERVE_ENABLE_UI": "true",
        "UVICORN_WORKERS": "1",
        "DOCLING_SERVE_MAX_SYNC_WAIT": "600",
    }
)
@modal.web_server(5001, startup_timeout=120)
def serve():
    import subprocess
    import os

    env = os.environ.copy()
    env.pop("PYTHONPATH", None)

    subprocess.Popen(
        ["/opt/app-root/bin/docling-serve", "run"], 
        cwd="/tmp",
        env=env
    )