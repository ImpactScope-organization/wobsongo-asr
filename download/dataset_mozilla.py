import modal

app = modal.App("download-mozilla-dioula-dataset")
volume = modal.Volume.from_name("dioula-dataset")

image = modal.Image.debian_slim().pip_install("requests")

@app.function(image=image, volumes={"/data": volume}, secrets=[modal.Secret.from_name("mdc-api-key")])
def download_cv_dioula():
    import requests, os, tarfile

    api_key = os.environ["MDC_API_KEY"]
    dataset_id = "cmqiaq3ae0083nr07rhacg5x2"

    resp = requests.post(
        f"https://mozilladatacollective.com/api/datasets/{dataset_id}/download",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
    )
    resp.raise_for_status()
    download_url = resp.json()["downloadUrl"]

    tar_path = "/data/mozilla-cv-dioula.tar.gz"
    with requests.get(download_url, stream=True) as r:
        r.raise_for_status()
        with open(tar_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    print(f"Downloaded to {tar_path}, size: {os.path.getsize(tar_path)} bytes")

    extract_dir = "/data/mozilla-dioula-dataset"
    os.makedirs(extract_dir, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(extract_dir)

    print(f"Extracted to {extract_dir}")
    print("Contents:")
    for root, dirs, files in os.walk(extract_dir):
        for name in files[:20]:
            print(os.path.join(root, name))

    os.remove(tar_path) 

    volume.commit()
    print("Done, committed to Volume.")