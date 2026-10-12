import modal

app = modal.App("moore-speech-full-pipeline")
volume = modal.Volume.from_name("moore-speech-data", create_if_missing=True)
VOLUME_PATH = "/data"
LANG_CODE = "mos"

FIXED_SCRAPER = r'''
from pathlib import Path
from typing import List
import requests
import re

from icu_tokenizer import SentSplitter
from scrapy import Spider, Request


class BibleScraper(Spider):
    output_folder: str
    start_urls: List[str]
    code: str
    splitter = SentSplitter()

    def __init__(self, output_folder, start_urls, code, filter_nums=False, *args, **kwargs):
        super(BibleScraper, self).__init__(*args, **kwargs)
        self.output_folder = output_folder
        self.start_urls = start_urls
        self.code = code
        self.filter_nums = filter_nums

    def download_audio(self, audio_src, output_file):
        data = requests.get(audio_src)
        with open(output_file, "wb") as out_mp3:
            out_mp3.write(data.content)
            return True, Path(output_file).stem

    def get_audio(self, audio_response, output_filename):
        audio = audio_response.css("audio")
        if "src" in audio.attrib:
            audio = audio.attrib["src"]
            self.download_audio(audio, f"{output_filename}.mp3")

    def parse(self, response):
        title = response.css("h1::text")
        title = title.get()
        book, chapter, code = response.url.split("/")[-1].split(".")[-3:]

        output_folder = Path(f"{self.output_folder}")
        output_folder.mkdir(exist_ok=True, parents=True)
        output_filename = output_folder / f"{book}_{chapter}_{code}"

        with open(f"{output_folder}/{code}.books", "a+") as titles:
            titles.write(f"{book}\t{chapter}\t{title}\n")
        audio_url = "audio-bible".join(response.url.rsplit("bible", 1))
        yield Request(audio_url, cb_kwargs={"output_filename": output_filename}, callback=self.get_audio)
        with open(f"{output_filename}.txt", "w") as output_file:
            output_file.write(f"{title}\n")
            verses = []
            for span in response.css('span[class*="__content"], span[class*="__heading"]'):
                cls = span.attrib.get("class", "")
                if any(skip in cls for skip in ("__label", "__note", "__fr", "__body")):
                    continue
                for verse in span.css("*::text").getall():
                    verses.append(verse.strip())
            text = " ".join(v for v in verses if v).strip()
            if text:
                for sent in self.splitter.split(text):
                    if self.filter_nums and bool(re.search(r"\d", sent)):
                        continue
                    output_file.write(f"{sent}\n")
        next_href = response.xpath('//a[contains(., "Next Chapter")]/@href').get()
        if next_href:
            yield Request(url=f"https://www.bible.com{next_href}")
'''

# Image for crawling + preprocessing 
base_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "pkg-config", "libicu-dev", "ffmpeg", "build-essential")
    .run_commands("python -m pip install 'pip<24.1'")
    .pip_install(
        "scrapy==2.13.1",
        "twisted==24.11.0",
        "icu-tokenizer==0.0.1",
        "requests",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/anyantudre/MooreSpeechCorpora.git /root/MooreSpeechCorpora",
    )
    .run_commands(
        f"cat > /root/MooreSpeechCorpora/crawlers/bible/bible_scraper.py << 'PYEOF'\n{FIXED_SCRAPER}\nPYEOF"
    )
)

# Image for forced alignment 
align_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "build-essential", "sox", "libsox-dev", "ffmpeg")
    .run_commands("python -m pip install 'pip<24.1'")
    .pip_install(
        "torch", "torchaudio", "torchcodec", "librosa", "soundfile", "sox",
        "numpy<2", "omegaconf==2.0.6", "hydra-core==1.0.7", "bitarray",
        "sacrebleu", "regex",
    )
    .run_commands(
        "git clone --depth 1 https://github.com/facebookresearch/fairseq.git /root/fairseq",
        "pip install --editable /root/fairseq",
        (
            "sed -i "
            "'s/input_lengths = torch.tensor(emissions.shape\\[0\\]).unsqueeze(-1)/"
            "input_lengths = torch.tensor(emissions.shape[0]).unsqueeze(-1).to(DEVICE)/' "
            "/root/fairseq/examples/mms/data_prep/align_and_segment.py"
        ),
        (
            "sed -i "
            "'s/target_lengths = torch.tensor(targets.shape\\[0\\]).unsqueeze(-1)/"
            "target_lengths = torch.tensor(targets.shape[0]).unsqueeze(-1).to(DEVICE)/' "
            "/root/fairseq/examples/mms/data_prep/align_and_segment.py"
        ),
        "git clone --depth 1 https://github.com/isi-nlp/uroman.git /root/uroman",
        "test -f /root/uroman/uroman/uroman.pl",
        "ln -s /root/uroman/uroman/data /root/uroman/data",
    )
)

export_image = modal.Image.debian_slim(python_version="3.10").pip_install(
    "datasets", "huggingface_hub", "soundfile"
)


# 1. Crawling 
@app.function(image=base_image, volumes={VOLUME_PATH: volume}, timeout=60 * 30)
def crawl_bible(start_url: str, code: str):
    import subprocess

    output_folder = f"{VOLUME_PATH}/raw/Moore"
    subprocess.run(
        ["python", "crawl.py", "--output_folder", output_folder, "--url", start_url, "--code", code],
        cwd="/root/MooreSpeechCorpora/crawlers/bible",
        check=True,
    )
    volume.commit()
    print(f"Crawling complete. Results saved to volume: raw/Moore/")


# 2. Preprocessing (resample) 
@app.function(image=base_image, volumes={VOLUME_PATH: volume}, timeout=60 * 15)
def resample_all():
    import subprocess

    subprocess.run(
        [
            "bash", "preprocessing/resample.sh",
            "--input_folder", f"{VOLUME_PATH}/raw/Moore",
            "--output_folder", f"{VOLUME_PATH}/moore/bible/resampled",
        ],
        cwd="/root/MooreSpeechCorpora",
        check=True,
    )
    volume.commit()
    print("Resample complete. Results saved to volume: moore/bible/resampled/")


# 3. Forced alignment 
@app.function(
    image=align_image,
    gpu="A10G",
    volumes={VOLUME_PATH: volume},
    timeout=60 * 20,
    max_containers=8,
)
def align_one(stem: str):
    import subprocess

    audio_path = f"{VOLUME_PATH}/moore/bible/resampled/{stem}.wav"
    text_path = f"{VOLUME_PATH}/moore/bible/resampled/{stem}.txt"
    outdir = f"{VOLUME_PATH}/aligned/{stem}"
    subprocess.run(["rm", "-rf", outdir], check=False)

    cmd = [
        "python", "-m", "examples.mms.data_prep.align_and_segment",
        "--audio_filepath", audio_path,
        "--text_filepath", text_path,
        "--lang", LANG_CODE,
        "--outdir", outdir,
        "--uroman_path", "/root/uroman/uroman",
    ]
    result = subprocess.run(cmd, cwd="/root/fairseq", capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Failed to align {stem}:\n{result.stderr[-2000:]}")
    else:
        print(f"Finished aligning {stem}")
    volume.commit()
    return stem, result.returncode


# 4. Export to Hugging Face 
@app.function(
    image=export_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 3,
    cpu=16,
)
def export_to_hf(repo_id: str, min_duration: float = 1.0, max_duration: float = 15.0):
    import json
    import os
    from pathlib import Path
    from datasets import Dataset, Audio
    from huggingface_hub import login, HfApi

    login(token=os.environ["HF_TOKEN"])
    api = HfApi()

    try:
        existing_files = api.list_repo_files(repo_id, repo_type="dataset")
    except Exception as e:
        existing_files = []
        print(f"The repo might not exist yet or has just been created: {e}")

    old_files = [f for f in existing_files if f.endswith(".parquet") or f in ("README.md", "dataset_infos.json")]
    deleted, failed = 0, []
    for f in old_files:
        try:
            api.delete_file(f, repo_id=repo_id, repo_type="dataset")
            deleted += 1
        except Exception as e:
            failed.append(f)
            print(f"  Failed to delete {f}: {e}")
    print(f"Successfully deleted {deleted}/{len(old_files)} old files (parquet + schema metadata).")
    if failed:
        print(f"WARNING: {len(failed)} files FAILED to delete: {failed}")
        print("Dataset viewer might still show errors if these files have the old schema.")

    mapping_path = Path(f"{VOLUME_PATH}/speaker_groups.json")
    speaker_map = json.loads(mapping_path.read_text()) if mapping_path.exists() else {}

    aligned_root = Path(f"{VOLUME_PATH}/aligned")
    rows = []
    dropped_short, dropped_long = 0, 0
    for chapter_dir in sorted(aligned_root.iterdir()):
        manifest = chapter_dir / "manifest.json"
        if not manifest.exists():
            continue
        book = chapter_dir.name.split("_")[0] 
        speaker_group = speaker_map.get(book, "unknown")
        with open(manifest) as f:
            for line in f:
                entry = json.loads(line)
                d = entry["duration"]
                if d < min_duration:
                    dropped_short += 1
                    continue
                if d > max_duration:
                    dropped_long += 1
                    continue
                rows.append({
                    "audio": entry["audio_filepath"],
                    "text": entry["text"],
                    "duration": d,
                    "chapter": chapter_dir.name,
                    "book": book,
                    "speaker_group": speaker_group,
                })

    total_seen = len(rows) + dropped_short + dropped_long
    print(
        f"Filter duration ({min_duration}s - {max_duration}s): "
        f"{len(rows)}/{total_seen} segments passed "
        f"(dropped: {dropped_short} too short, {dropped_long} too long)."
    )

    ds = Dataset.from_list(rows).cast_column("audio", Audio())
    ds.push_to_hub(repo_id, max_shard_size="300MB", num_proc=16)
    print(f"Dataset published at: https://huggingface.co/datasets/{repo_id}")
    if not speaker_map:
        print(
            "Note: speaker_groups.json not found, all speaker_group columns "
            "will be 'unknown'. Run modal_speaker_analysis.py first "
            "if you want these columns to be filled."
        )

@app.function(
    image=export_image,
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 10,
)
def check_hf_dataset(repo_id: str):
    import os
    from collections import defaultdict
    from datasets import load_dataset
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    def fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}j {m}m {s:.0f}d"

    ds = load_dataset(repo_id, split="train")
    durations = ds["duration"]
    total = sum(durations)

    print(f"=== Dataset in HF: {repo_id} ===")
    print(f"Total rows     : {len(ds)}")
    print(f"Total duration    : {fmt(total)}  ({total / 3600:.2f} hours)")
    print(
        f"Duration per clip : min={min(durations):.1f}s, "
        f"max={max(durations):.1f}s, average={total / len(durations):.1f}s"
    )

    per_book = defaultdict(lambda: {"duration": 0.0, "segments": 0})
    for book, dur in zip(ds["book"], durations):
        per_book[book]["duration"] += dur
        per_book[book]["segments"] += 1

    print("\n=== Per book (sorted by longest duration) ===")
    for book, stats in sorted(per_book.items(), key=lambda x: -x[1]["duration"]):
        print(f"  {book:6s}: {stats['segments']:4d} segments, {fmt(stats['duration'])}")

@app.function(image=export_image, volumes={VOLUME_PATH: volume})
def dataset_stats():
    import json
    from pathlib import Path
    from collections import defaultdict

    aligned_root = Path(f"{VOLUME_PATH}/aligned")
    if not aligned_root.exists():
        print("Aligned folder missing — run forced alignment first.")
        return

    def fmt(seconds):
        h = int(seconds // 3600)
        m = int((seconds % 3600) // 60)
        s = seconds % 60
        return f"{h}j {m}m {s:.0f}d"

    total_duration = 0.0
    total_segments = 0
    per_book = defaultdict(lambda: {"duration": 0.0, "segments": 0})

    for chapter_dir in sorted(aligned_root.iterdir()):
        manifest = chapter_dir / "manifest.json"
        if not manifest.exists():
            continue
        book = chapter_dir.name.split("_")[0]
        for line in manifest.read_text().splitlines():
            entry = json.loads(line)
            d = entry["duration"]
            total_duration += d
            total_segments += 1
            per_book[book]["duration"] += d
            per_book[book]["segments"] += 1

    print("=== Total dataset ===")
    print(f"Total segments : {total_segments}")
    print(f"Total duration : {fmt(total_duration)}  ({total_duration / 3600:.2f} hours)")

    print("\n=== Per book (sorted by longest duration) ===")
    for book, stats in sorted(per_book.items(), key=lambda x: -x[1]["duration"]):
        print(f"  {book:6s}: {stats['segments']:4d} segments, {fmt(stats['duration'])}")

def peek_aligned(chapter: str = ""):
    import json
    from pathlib import Path

    aligned_root = Path(f"{VOLUME_PATH}/aligned")
    chapters = sorted(d.name for d in aligned_root.iterdir()) if aligned_root.exists() else []

    if not chapter:
        print(f"Found {len(chapters)} chapters aligned: {chapters}")
        return

    manifest = aligned_root / chapter / "manifest.json"
    if not manifest.exists():
        print(f"Manifest not found for {chapter}")
        return

    lines = manifest.read_text().splitlines()
    print(f"{chapter}: {len(lines)} segments")
    for line in lines[:3]:
        entry = json.loads(line)
        print(f"  [{entry['duration']:.1f}s] {entry['text']}")


@app.local_entrypoint()
def main(start_url: str = "https://www.bible.com/bible/3058/GEN.1.MPBU", code: str = "MPBU", hf_repo_id: str = ""):
    print("=== 1/4 Crawling ===")
    crawl_bible.remote(start_url, code)

    print("=== 2/4 Preprocessing (resample) ===")
    resample_all.remote()

    print("=== 3/4 Forced alignment (paralel di GPU) ===")
    entries = volume.listdir("moore/bible/resampled")
    stems = sorted({e.path.split("/")[-1].rsplit(".", 1)[0] for e in entries if e.path.endswith(".wav")})
    print(f"Found {len(stems)} files to align: {stems}")
    results = list(align_one.map(stems))
    for stem, code_ in results:
        print(f"  {stem}: {'OK' if code_ == 0 else 'FAILED'}")

    if hf_repo_id:
        print("=== 4/4 Export to Hugging Face ===")
        export_to_hf.remote(hf_repo_id)
    else:
        print("=== 4/4 Skipped (no --hf-repo-id provided) ===")

    print(
        "\nDone. To preview/download the results:\n"
        f"    modal volume get moore-speech-data aligned ./aligned_preview"
    )