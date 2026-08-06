import modal

image = modal.Image.from_registry(
    "quay.io/docling-project/docling-serve-cu130:v1.26.0"
).pip_install("certifi", extra_options="--upgrade")  

app = modal.App("docling-serve")

docling_volume = modal.Volume.from_name(
    "docling-model-cache", create_if_missing=True
)

VOLUME_PATH = "/vol/docling-cache"


# modal run docling_deploy.py::download_models
@app.function(
    image=image,
    timeout=1800,
    volumes={VOLUME_PATH: docling_volume},
)
def download_models():
    import subprocess
    import os

    cert_path = subprocess.run(
        ["python3", "-m", "certifi"],
        capture_output=True, text=True, check=True
    ).stdout.strip()

    env = os.environ.copy()
    env["SSL_CERT_FILE"] = cert_path
    env["REQUESTS_CA_BUNDLE"] = cert_path
    env["CURL_CA_BUNDLE"] = cert_path

    subprocess.run(
        ["docling-tools", "models", "download", "--all", "-o", VOLUME_PATH],
        check=True,
        env=env, 
    )
    docling_volume.commit()  

@app.function(
    image=image,
    gpu="A100",
    timeout=600,
    scaledown_window=300,
    volumes={
        VOLUME_PATH: docling_volume,
    },
    env={
        "DOCLING_SERVE_ENABLE_UI": "true",
        "UVICORN_WORKERS": "1",
        "DOCLING_SERVE_MAX_SYNC_WAIT": "600",
        "DOCLING_SERVE_ARTIFACTS_PATH": VOLUME_PATH,
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