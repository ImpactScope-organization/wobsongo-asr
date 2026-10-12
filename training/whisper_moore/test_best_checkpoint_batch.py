import os
import re
import json
import glob
import unicodedata

import modal

app = modal.App("moore-whisper-realworld-probe-batch")
model_volume = modal.Volume.from_name("moore-finetuned-model", create_if_missing=True)
MODEL_PATH = "/output"
CHECKPOINT_DIR = f"{MODEL_PATH}/checkpoints_whisper_small_moore"
MODEL_ID = "openai/whisper-small"
FORCED_LANGUAGE = "french"

AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".mp4", ".ogg", ".flac"}

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio", "transformers>=4.44", "accelerate",
        "evaluate", "jiwer", "soundfile", "librosa", "numpy", "pandas",
    )
)

TONE_MARK_MAP_ITEMS = {
    "\u025b": "e", "\u0254": "o", "\u0272": "ny", "\u014b": "ng",
    "\u0190": "e", "\u0186": "o", "\u019d": "ny", "\u014a": "ng",
}


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[.,!?;:'\"\u2019\u2018\u00ab\u00bb]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_orthographic_variants(text):
    stripped = "".join(
        c for c in unicodedata.normalize("NFD", text)
        if unicodedata.category(c) != "Mn"
    )
    return stripped.translate(str.maketrans(TONE_MARK_MAP_ITEMS))


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={MODEL_PATH: model_volume},
    timeout=60 * 30,
)
def transcribe_and_score_batch(clips: list, checkpoint: str):
    """clips: list of {"clip_id": str, "audio_bytes": bytes, "audio_ext": str, "reference_text": str}"""
    import subprocess
    import tempfile

    import torch
    import evaluate
    import jiwer
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor, pipeline

    ckpt_dir = f"{CHECKPOINT_DIR}/{checkpoint}"
    print(f"Loading checkpoint dari {ckpt_dir} ...")
    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=FORCED_LANGUAGE, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=torch.float32)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=FORCED_LANGUAGE, task="transcribe"
    )
    device = 0 if torch.cuda.is_available() else -1

    asr = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        chunk_length_s=15,
        stride_length_s=5,
        device=device,
         generate_kwargs={
            "forced_decoder_ids": model.generation_config.forced_decoder_ids,
            "no_repeat_ngram_size": 3,
            "repetition_penalty": 1.3,
        },
    )

    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    per_clip_results = []
    all_hyp_norm, all_ref_norm = [], []

    for clip in clips:
        clip_id = clip["clip_id"]
        print(f"\n--- Processing: {clip_id} ---")

        with tempfile.TemporaryDirectory() as tmp:
            raw_path = os.path.join(tmp, f"input{clip['audio_ext']}")
            wav_path = os.path.join(tmp, "audio.wav")
            with open(raw_path, "wb") as f:
                f.write(clip["audio_bytes"])
            subprocess.run(
                ["ffmpeg", "-y", "-i", raw_path, "-ar", "16000", "-ac", "1", wav_path],
                check=True, capture_output=True,
            )
            audio, sr = librosa.load(wav_path, sr=16000)
            duration_sec = len(audio) / sr

        result = asr(audio)
        hyp_raw = result["text"]
        hyp_norm = normalize_text(hyp_raw)
        ref_norm = normalize_text(clip["reference_text"])

        wer = 100 * metric_wer.compute(predictions=[hyp_norm], references=[ref_norm])
        cer = 100 * metric_cer.compute(predictions=[hyp_norm], references=[ref_norm])
        hyp_relaxed = strip_orthographic_variants(hyp_norm)
        ref_relaxed = strip_orthographic_variants(ref_norm)
        cer_relaxed = 100 * metric_cer.compute(predictions=[hyp_relaxed], references=[ref_relaxed])
        breakdown = jiwer.process_words(ref_norm, hyp_norm)
        ref_words = len(ref_norm.split())
        hyp_words = len(hyp_norm.split())

        print(f"  durasi={duration_sec:.1f}s  ref_words={ref_words}  hyp_words={hyp_words}  "
              f"WER={wer:.2f}%  CER={cer:.2f}%")

        per_clip_results.append({
            "clip_id": clip_id,
            "duration_sec": duration_sec,
            "reference_raw": clip["reference_text"],
            "hypothesis_raw": hyp_raw,
            "reference_norm": ref_norm,
            "hypothesis_norm": hyp_norm,
            "ref_words": ref_words,
            "hyp_words": hyp_words,
            "length_ratio": hyp_words / ref_words if ref_words > 0 else None,
            "wer": wer,
            "cer": cer,
            "cer_relaxed": cer_relaxed,
            "substitutions": breakdown.substitutions,
            "deletions": breakdown.deletions,
            "insertions": breakdown.insertions,
        })
        all_hyp_norm.append(hyp_norm)
        all_ref_norm.append(ref_norm)

    corpus_wer = 100 * metric_wer.compute(predictions=all_hyp_norm, references=all_ref_norm)
    corpus_cer = 100 * metric_cer.compute(predictions=all_hyp_norm, references=all_ref_norm)
    mean_utt_wer = sum(r["wer"] for r in per_clip_results) / len(per_clip_results)
    median_utt_wer = sorted(r["wer"] for r in per_clip_results)[len(per_clip_results) // 2]
    total_ref_words = sum(r["ref_words"] for r in per_clip_results)
    total_duration_sec = sum(r["duration_sec"] for r in per_clip_results)

    summary = {
        "n_clips": len(per_clip_results),
        "total_duration_sec": total_duration_sec,
        "total_ref_words": total_ref_words,
        "corpus_wer": corpus_wer,
        "corpus_cer": corpus_cer,
        "mean_per_clip_wer": mean_utt_wer,
        "median_per_clip_wer": median_utt_wer,
    }

    print("\n=== AGGREGATE SUMMARY ===")
    print(f"n_clips          : {summary['n_clips']}")
    print(f"total duration   : {summary['total_duration_sec']/60:.1f} minutes")
    print(f"total ref words  : {summary['total_ref_words']}")
    print(f"corpus WER       : {summary['corpus_wer']:.2f}%   <- primary figure, word-weighted")
    print(f"corpus CER       : {summary['corpus_cer']:.2f}%")
    print(f"mean per-clip WER: {summary['mean_per_clip_wer']:.2f}%  (may differ from corpus WER if clip lengths are uneven)")
    print(f"median per-clip WER: {summary['median_per_clip_wer']:.2f}%")

    return {"per_clip": per_clip_results, "summary": summary}


@app.local_entrypoint()
def main(clip_folder: str, checkpoint: str = "checkpoint-2400"):
    import csv

    audio_files = [
        f for f in glob.glob(os.path.join(clip_folder, "*"))
        if os.path.splitext(f)[1].lower() in AUDIO_EXTS
    ]
    if not audio_files:
        raise SystemExit(f"No audio files found in {clip_folder}")
    
    clips = []
    missing_transcripts = []
    for audio_path in sorted(audio_files):
        base = os.path.splitext(os.path.basename(audio_path))[0]
        transcript_path = os.path.join(clip_folder, f"{base}.txt")
        if not os.path.exists(transcript_path):
            missing_transcripts.append(base)
            continue
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        with open(transcript_path, "r", encoding="utf-8") as f:
            reference_text = f.read()
        clips.append({
            "clip_id": base,
            "audio_bytes": audio_bytes,
            "audio_ext": os.path.splitext(audio_path)[1],
            "reference_text": reference_text,
        })

    if missing_transcripts:
        print(f"[WARNING] The following clips were skipped because no .txt transcript was found: {missing_transcripts}")
    print(f"Found {len(clips)} valid audio-transcript pairs. Processing...")

    result = transcribe_and_score_batch.remote(clips, checkpoint)

    per_clip_rows = result["per_clip"]
    if per_clip_rows:
        fieldnames = list(per_clip_rows[0].keys())
        with open("realworld_probe_per_clip.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(per_clip_rows)

    with open("realworld_probe_summary.json", "w", encoding="utf-8") as f:
        json.dump(result["summary"], f, ensure_ascii=False, indent=2)

    print(f"\nPer-clip details saved to realworld_probe_per_clip.csv")
    print(f"Summary saved to realworld_probe_summary.json")

    # modal run test_best_checkpoint_batch.py --clip-folder ./clip --checkpoint checkpoint-2400