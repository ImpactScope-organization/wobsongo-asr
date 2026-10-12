import modal

app = modal.App("moore-speaker-analysis")
volume = modal.Volume.from_name("moore-speech-data", create_if_missing=True)
VOLUME_PATH = "/data"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install("setuptools<82", "resemblyzer", "librosa", "numpy", "soundfile")
)


@app.function(image=image, volumes={VOLUME_PATH: volume}, timeout=60 * 20)
def analyze_speakers(threshold: float = 0.80):
    import json
    import numpy as np
    from pathlib import Path
    from resemblyzer import VoiceEncoder, preprocess_wav

    aligned_root = Path(f"{VOLUME_PATH}/aligned")
    if not aligned_root.exists():
        print("Aligned folder missing -- run forced alignment first.")
        return
    
    chapter_dirs = sorted(d.name for d in aligned_root.iterdir() if d.is_dir())
    book_to_chapters = {}
    for name in chapter_dirs:
        parts = name.split("_")
        if len(parts) < 2:
            continue
        book = parts[0]
        book_to_chapters.setdefault(book, []).append(name)

    print(f"Found {len(book_to_chapters)} aligned books: {sorted(book_to_chapters)}")

    encoder = VoiceEncoder()
    embeddings = {}

    for book, chapters in sorted(book_to_chapters.items()):
        chapter_dir = aligned_root / sorted(chapters)[0]
        manifest = chapter_dir / "manifest.json"
        if not manifest.exists():
            print(f"  [{book}] manifest not found, skipping")
            continue

        entries = [json.loads(line) for line in manifest.read_text().splitlines()]
        if not entries:
            continue
        longest = max(entries, key=lambda e: e["duration"])
        audio_path = Path(longest["audio_filepath"])
        if not audio_path.exists():
            print(f"  [{book}] audio file {audio_path} not found, skipping")
            continue

        try:
            wav = preprocess_wav(str(audio_path))
            embeddings[book] = encoder.embed_utterance(wav)
            print(f"  [{book}] embedding OK ({longest['duration']:.1f} seconds, from {chapter_dir.name})")
        except Exception as e:
            print(f"  [{book}] failed to process audio: {e}")

    names = list(embeddings.keys())
    if len(names) < 2:
        print("Not enough books to compare (at least 2).")
        return

    print("\n=== Similarity matrix (cosine similarity, 0–1) ===")
    print("        " + " ".join(f"{n:>6}" for n in names))
    sims = {}
    for a in names:
        row = [a.ljust(8)]
        for b in names:
            sim = float(np.dot(embeddings[a], embeddings[b]))
            sims[(a, b)] = sim
            row.append(f"{sim:6.2f}")
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
    for gid, members in groups.items():
        print(f"  Group {gid}: {', '.join(members)}")
    assigned = {book: gid for gid, members in groups.items() for book in members}

    mapping = {book: f"speaker_{gid}" for book, gid in assigned.items()}
    mapping_path = Path(f"{VOLUME_PATH}/speaker_groups.json")
    mapping_path.write_text(json.dumps(mapping, indent=2, ensure_ascii=False))
    volume.commit()
    print(f"\nSpeaker mapping saved to volume: speaker_groups.json")
    print(
        "\nNote: threshold 0.80 is an initial estimate, not a fixed value. "
        "If the group results are too loose/tight, try running again "
        "with a different --threshold (0.7-0.9)."
    )


@app.local_entrypoint()
def main(threshold: float = 0.80):
    analyze_speakers.remote(threshold)