import modal

app = modal.App("moore-corpus-toolkit-pipeline")
volume = modal.Volume.from_name("moore-corpus-toolkit-data", create_if_missing=True)
VOLUME_PATH = "/data"

toolkit_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "ffmpeg")
    .run_commands(
        "git clone --depth 1 https://github.com/AI-KING-GROUP/moore-corpus-toolkit.git /root/moore-corpus-toolkit",
    )
    .run_commands(
        "cd /root/moore-corpus-toolkit && pip install -r requirements.txt && pip install -e ."
    )
)

export_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "datasets", "huggingface_hub", "soundfile", "pandas"
)

@app.function(image=toolkit_image, volumes={VOLUME_PATH: volume}, timeout=60 * 30)
def crawl_volume(start_url: str, volume_name: str, max_pages: int | None = None):
    import subprocess

    out_dir = f"{VOLUME_PATH}/raw/{volume_name}"
    cmd = [
        "python3", "-m", "mooreburkina_crawler", "crawl",
        start_url, out_dir,
    ]
    if max_pages:
        cmd += ["--max-pages", str(max_pages)]

    subprocess.run(cmd, cwd="/root/moore-corpus-toolkit", check=True)
    volume.commit()
    print(f"Crawl done. Results saved to volume: raw/{volume_name}/")
    return out_dir

@app.function(image=toolkit_image, volumes={VOLUME_PATH: volume}, timeout=60 * 10)
def build_pairs(volume_name: str):
    import subprocess

    in_dir = f"{VOLUME_PATH}/raw/{volume_name}"
    tsv_path = f"{VOLUME_PATH}/pairs/{volume_name}.tsv"

    subprocess.run(["mkdir", "-p", f"{VOLUME_PATH}/pairs"], check=True)
    subprocess.run(
        ["python3", "-m", "mooreburkina_crawler", "pairs", in_dir, tsv_path],
        cwd="/root/moore-corpus-toolkit",
        check=True,
    )
    volume.commit()
    print(f"TSV done: pairs/{volume_name}.tsv")
    return tsv_path

@app.function(
    image=export_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 30,
)
def export_to_hf(volume_names: list[str], repo_id: str):
    import os
    import pandas as pd
    from pathlib import Path
    from datasets import Dataset, Audio
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    rows = []
    for vol_name in volume_names:
        tsv_path = Path(f"{VOLUME_PATH}/pairs/{vol_name}.tsv")
        raw_dir = Path(f"{VOLUME_PATH}/raw/{vol_name}")
        if not tsv_path.exists():
            print(f"Skip {vol_name}: TSV not found")
            continue

        df = pd.read_csv(tsv_path, sep="\t", encoding="utf-8-sig")
        df = df.fillna("")
        matched = 0
        for _, r in df.iterrows():
            page_slug = Path(str(r["source_file"])).stem
            audio_path = raw_dir / f"{page_slug}.mp3"
            if not audio_path.exists():
                continue
            matched += 1
            rows.append({
                "audio": str(audio_path),
                "moore": str(r.get("moore", "")),
                "french": str(r.get("french", "")),
                "english": str(r.get("english", "")),
                "volume": vol_name,
            })
        print(f"{vol_name}: matched {matched}/{len(df)} rows to audio files")

    if not rows:
        raise RuntimeError(
            "No rows matched any audio file -- check that 'source_file' in "
            "the TSV corresponds to a real .mp3 stem under raw/<volume>/. "
            "Inspect one value of df['source_file'] against `ls raw/<volume>/` to confirm."
        )

    ds = Dataset.from_list(rows).cast_column("audio", Audio())
    ds.push_to_hub(repo_id, max_shard_size="300MB")
    print(f"Dataset published at: https://huggingface.co/datasets/{repo_id}")

@app.local_entrypoint()
def main(
    start_url: str = "https://media.ipsapps.org/mos/ora/p1/01-001-001.html",
    volume_name: str = "proverbes_vol1",
    max_pages: int = 0,
    hf_repo_id: str = "",
):
    print("=== 1/3 Crawling ===")
    crawl_volume.remote(start_url, volume_name, max_pages or None)

    print("=== 2/3 Building pairs TSV ===")
    build_pairs.remote(volume_name)

    if hf_repo_id:
        print("=== 3/3 Export to Hugging Face ===")
        export_to_hf.remote([volume_name], hf_repo_id)
    else:
        print("=== 3/3 Skipped (no --hf-repo-id provided) ===")

    print("\nDone.")

@app.local_entrypoint()
def export(volume_names: str, hf_repo_id: str):
    names = [v.strip() for v in volume_names.split(",") if v.strip()]
    print(f"=== Export {names} to Hugging Face ===")
    export_to_hf.remote(names, hf_repo_id)
    print("\nDone.")