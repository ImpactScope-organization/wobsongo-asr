import modal

app = modal.App("moore-goai-speaker-analysis")
volume = modal.Volume.from_name("moore-speech-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "setuptools<82", "resemblyzer", "librosa", "numpy", "soundfile",
        "datasets", "huggingface_hub", "torchcodec",
    )
)

DATASET_CONFIGS = [
    {
        "name": "goai-proverbs",
        "repo_id": "goaicorp/new-moore-speech-clean",
        "audio_col": "moore_audio",
    },
    {
        "name": "goai-contes",
        "repo_id": "goaicorp/goai-moore-speech-contes",
        "audio_col": "audio",
    },
]

EXCLUDE_SPLITS = {"full"} 


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=60 * 30,
)
def analyze_speakers(dataset_key: str, threshold: float = 0.80):
    import os
    import json
    import numpy as np
    from pathlib import Path
    from datasets import load_dataset, get_dataset_split_names
    from resemblyzer import VoiceEncoder, preprocess_wav
    import soundfile as sf

    cfg = next(c for c in DATASET_CONFIGS if c["name"] == dataset_key)
    token = os.environ["HF_TOKEN"]

    all_splits = get_dataset_split_names(cfg["repo_id"], token=token)
    volume_splits = sorted(s for s in all_splits if s not in EXCLUDE_SPLITS)
    print(f"Splits found on HF: {all_splits}")
    print(f"Using as volume units: {volume_splits}")

    encoder = VoiceEncoder()
    embeddings = {}
    total_duration_sec = {}  
    num_clips = {} 
    tmp_dir = Path("/tmp/audio_chunks")
    tmp_dir.mkdir(exist_ok=True)

    for vol_key in volume_splits:
        print(f"\nLoading split '{vol_key}' ...")
        ds = load_dataset(cfg["repo_id"], split=vol_key, token=token)

        longest_dur, longest_arr, longest_sr = -1.0, None, None
        total_dur = 0.0

        for row in ds:
            audio = row[cfg["audio_col"]]
            dur = len(audio["array"]) / audio["sampling_rate"]
            total_dur += dur
            if dur > longest_dur:
                longest_dur, longest_arr, longest_sr = dur, audio["array"], audio["sampling_rate"]

        total_duration_sec[vol_key] = total_dur
        num_clips[vol_key] = len(ds)

        try:
            tmp_path = tmp_dir / f"{vol_key}.wav"
            sf.write(tmp_path, longest_arr, longest_sr)
            wav = preprocess_wav(str(tmp_path))
            embeddings[vol_key] = encoder.embed_utterance(wav)
            print(
                f"  [{vol_key}] {len(ds)} clips, total {total_dur/60:.1f} min, "
                f"embedding from longest clip ({longest_dur:.1f}s)"
            )
        except Exception as e:
            print(f"  [{vol_key}] failed to process audio: {e}")

    names = list(embeddings.keys())
    if len(names) < 2:
        print("Not enough volumes to compare (at least 2).")
        return

    print("\n=== Similarity matrix (cosine similarity, 0-1) ===")
    print("        " + " ".join(f"{n:>14}" for n in names))
    sims = {}
    for a in names:
        row = [a.ljust(14)]
        for b in names:
            sim = float(np.dot(embeddings[a], embeddings[b]))
            sims[(a, b)] = sim
            row.append(f"{sim:14.2f}")
        print(" ".join(row))

    print(f"\n=== Speaker group predictions (similarity > {threshold}) ===")
    groups_list = []
    for a in names:
        placed = False
        for group in groups_list:
            if all(sims.get((a, m), 0) > threshold for m in group):
                group.append(a)
                placed = True
                break
        if not placed:
            groups_list.append([a])
    groups = {i + 1: members for i, members in enumerate(groups_list)}

    grand_total_sec = sum(total_duration_sec.values())
    grand_total_clips = sum(num_clips.values())

    for gid, members in groups.items():
        group_sec = sum(total_duration_sec[m] for m in members)
        group_clips = sum(num_clips[m] for m in members)
        pct = 100 * group_sec / grand_total_sec if grand_total_sec else 0
        print(
            f"  Group {gid}: {', '.join(members)}  "
            f"(~{group_sec/3600:.2f} h, {group_clips} clips, {pct:.1f}% of total)"
        )

    print(
        f"\nGrand total: {grand_total_sec/3600:.2f} h across {grand_total_clips} clips "
        f"in {len(volume_splits)} volumes"
    )

    assigned = {vol: gid for gid, members in groups.items() for vol in members}
    mapping = {
        vol: {
            "speaker_group": f"speaker_{gid}",
            "total_duration_sec": total_duration_sec[vol],
            "num_clips": num_clips[vol],
        }
        for vol, gid in assigned.items()
    }
    mapping_path = Path(f"{VOLUME_PATH}/{dataset_key}_speaker_groups.json")
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    volume.commit()
    print(f"\nSpeaker mapping + duration saved to volume: {dataset_key}_speaker_groups.json")
    print(
        f"\nNote: threshold {threshold} is an initial estimate, not a fixed value. "
        "If the group results are too loose/tight, try running again "
        "with a different --threshold (0.7-0.9)."
    )


@app.local_entrypoint()
def main(dataset: str = "all", threshold: float = 0.80):
    """
    dataset: "goai-proverbs", "goai-contes", atau "all" (default, jalankan keduanya)
    """
    targets = (
        [c["name"] for c in DATASET_CONFIGS] if dataset == "all" else [dataset]
    )
    for name in targets:
        print(f"\n{'=' * 60}\nDataset: {name}\n{'=' * 60}")
        analyze_speakers.remote(name, threshold)