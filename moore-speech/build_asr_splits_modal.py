import modal

app = modal.App("moore-asr-build-splits")
volume = modal.Volume.from_name("moore-speech-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "setuptools<82", "datasets", "huggingface_hub", "soundfile",
        "numpy", "torch", "torchcodec",
    )
)

BIBLE_GROUPS = {
    "G1": ["1CH", "1KI", "1SA", "2CH", "2KI", "2SA", "JDG", "JOB", "JOS"],
    "G2": ["AMO", "DAN", "DEU", "EZK", "GEN", "HAB", "HAG", "HOS", "JOL",
           "JON", "LAM", "LEV", "MIC", "OBA", "ZEP"],
    "G3": ["ECC", "EXO", "ISA", "JER", "NUM", "PRO", "PSA", "SNG"],
    "G4": ["EST", "RUT"],
    "G5": ["EZR", "MAL", "NAM", "NEH", "ZEC"],
}
BIBLE_SPLIT_MAP = {"G1": "train", "G2": "train", "G3": "train", "G4": "dev", "G5": "test"}

PROVERBS_GROUPS = {
    "G1": ["vol1", "vol7"],
    "G2": ["vol10", "vol11", "vol3"],
    "G3": ["vol2", "vol4", "vol6"],
    "G4": ["vol5"],
    "G5": ["vol8"],
}
PROVERBS_SPLIT_MAP = {"G1": "train", "G2": "train", "G3": "train", "G4": "dev", "G5": "test"}

CONTES_GROUPS = {
    "G1": ["vol2"],
    "G2": ["vol3", "vol4"],
}
CONTES_SPLIT_MAP = {"G1": "test", "G2": "train"}

MIN_DURATION = 1.0
MAX_DURATION = 15.0


def _reverse_group_map(groups, split_map):
    """unit_id -> (group_id, split)"""
    out = {}
    for gid, units in groups.items():
        for u in units:
            out[u] = (gid, split_map[gid])
    return out


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 60 * 2,
    cpu=8,
)
def build_manifests(min_duration: float = MIN_DURATION, max_duration: float = MAX_DURATION):
    import os
    import json
    import soundfile as sf
    from pathlib import Path
    from datasets import load_dataset

    all_rows = []  
    stats_dropped = {"short": 0, "long": 0}

    print("=== Processing Bible (from existing volume/aligned) ===")
    bible_unit_map = _reverse_group_map(BIBLE_GROUPS, BIBLE_SPLIT_MAP)
    aligned_root = Path(f"{VOLUME_PATH}/aligned")

    bible_count = 0
    for chapter_dir in sorted(aligned_root.iterdir()):
        manifest = chapter_dir / "manifest.json"
        if not manifest.exists():
            continue
        book = chapter_dir.name.split("_")[0]
        if book not in bible_unit_map:
            print(f"  WARNING: book '{book}' Not found in BIBLE_GROUPS; skipped.")
            continue
        group_id, split = bible_unit_map[book]

        for idx, line in enumerate(manifest.read_text().splitlines()):
            entry = json.loads(line)
            d = entry["duration"]
            if d < min_duration:
                stats_dropped["short"] += 1
                continue
            if d > max_duration:
                stats_dropped["long"] += 1
                continue
            all_rows.append({
                "id": f"bible_{chapter_dir.name}_{idx:03d}",
                "source_dataset": "bible",
                "audio_path": entry["audio_filepath"], 
                "text": entry["text"],
                "duration_sec": d,
                "unit_id": book,
                "chapter": chapter_dir.name,
                "speaker_group": f"bible_{group_id}",
                "split": split,
                "license": "mooreburkina-original",
            })
            bible_count += 1
    print(f"  Bible: {bible_count} clips loaded from existing volume.")

    goai_configs = [
        {
            "name": "goai-proverbs",
            "repo_id": "goaicorp/new-moore-speech-clean",
            "audio_col": "moore_audio",
            "text_col": "moore_text",
            "unit_map": _reverse_group_map(PROVERBS_GROUPS, PROVERBS_SPLIT_MAP),
        },
        {
            "name": "goai-contes",
            "repo_id": "goaicorp/goai-moore-speech-contes",
            "audio_col": "audio",
            "text_col": "transcription",
            "unit_map": _reverse_group_map(CONTES_GROUPS, CONTES_SPLIT_MAP),
        },
    ]

    token = os.environ["HF_TOKEN"]

    for cfg in goai_configs:
        print(f"\n=== Processing {cfg['name']} (download from HF, write to volume) ===")
        out_audio_dir = Path(f"{VOLUME_PATH}/{cfg['name']}/audio")
        out_audio_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for vol_key, (group_id, split) in sorted(cfg["unit_map"].items()):
            print(f"  Loading split '{vol_key}' ...")
            ds = load_dataset(cfg["repo_id"], split=vol_key, token=token)

            for idx, row in enumerate(ds):
                d = row[cfg["audio_col"]]["array"]
                sr = row[cfg["audio_col"]]["sampling_rate"]
                dur = len(d) / sr

                if dur < min_duration:
                    stats_dropped["short"] += 1
                    continue
                if dur > max_duration:
                    stats_dropped["long"] += 1
                    continue

                clip_id = f"{cfg['name']}_{vol_key}_{idx:04d}"
                wav_path = out_audio_dir / f"{clip_id}.wav"
                if not wav_path.exists():
                    sf.write(wav_path, d, sr)

                all_rows.append({
                    "id": clip_id,
                    "source_dataset": cfg["name"],
                    "audio_path": str(wav_path),
                    "text": row[cfg["text_col"]],
                    "duration_sec": dur,
                    "unit_id": vol_key,
                    "chapter": row.get("chapter", vol_key),
                    "speaker_group": f"{cfg['name']}_{group_id}",
                    "split": split,
                    "license": "cc-by-nc-4.0",
                })
                count += 1
            print(f"    [{vol_key}] -> split={split}, group={group_id}")
        print(f"  {cfg['name']}: {count} clips processed.")

    manifest_dir = Path(f"{VOLUME_PATH}/manifests")
    manifest_dir.mkdir(parents=True, exist_ok=True)

    by_split = {"train": [], "dev": [], "test": []}
    for row in all_rows:
        by_split[row["split"]].append(row)

    print("\n=== Final Summary ===")
    for split, rows in by_split.items():
        total_sec = sum(r["duration_sec"] for r in rows)
        out_path = manifest_dir / f"{split}.jsonl"
        with open(out_path, "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        by_source = {}
        for r in rows:
            by_source.setdefault(r["source_dataset"], {"count": 0, "sec": 0.0})
            by_source[r["source_dataset"]]["count"] += 1
            by_source[r["source_dataset"]]["sec"] += r["duration_sec"]
        print(f"  {split}: {len(rows)} clips, {total_sec/3600:.2f} h  -> saved to {out_path}")
        for src, s in by_source.items():
            print(f"      - {src}: {s['count']} clips, {s['sec']/3600:.2f} h")

    combined_path = manifest_dir / "all.jsonl"
    with open(combined_path, "w") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    grand_total_sec = sum(r["duration_sec"] for r in all_rows)
    print(f"\nGrand total: {len(all_rows)} clips, {grand_total_sec/3600:.2f} h")
    print(
        f"Dropped by duration filter ({min_duration}s-{max_duration}s): "
        f"{stats_dropped['short']} too short, {stats_dropped['long']} too long"
    )

    volume.commit()
    print(f"\nManifests saved to volume: manifests/train.jsonl, dev.jsonl, test.jsonl, all.jsonl")


@app.local_entrypoint()
def main(min_duration: float = MIN_DURATION, max_duration: float = MAX_DURATION):
    build_manifests.remote(min_duration, max_duration)