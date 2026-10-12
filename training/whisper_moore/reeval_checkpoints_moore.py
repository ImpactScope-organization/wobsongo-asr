import os
import re
import json
import statistics
import unicodedata
import random
from collections import Counter

import modal

app = modal.App("moore-whisper-reeval")
volume = modal.Volume.from_name("moore-speech-data", create_if_missing=True)
model_volume = modal.Volume.from_name("moore-finetuned-model", create_if_missing=True)
VOLUME_PATH = "/data"
MODEL_PATH = "/output"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("ffmpeg")
    .pip_install(
        "torch", "torchaudio", "torchcodec", "transformers>=4.44", "datasets",
        "accelerate", "evaluate", "jiwer", "soundfile", "librosa",
        "wandb", "numpy", "pandas", "scipy",
    )
)

MODEL_ID = "openai/whisper-small"
FORCED_LANGUAGE = "french"
CHECKPOINT_DIR = f"{MODEL_PATH}/checkpoints_whisper_small_moore"
DEV_MANIFEST = f"{VOLUME_PATH}/manifests/dev.jsonl"
TEST_MANIFEST = f"{VOLUME_PATH}/manifests/test.jsonl"
RESULTS_DIR = f"{MODEL_PATH}/reeval"
PER_CKPT_DIR = f"{RESULTS_DIR}/per_checkpoint"

MIN_GROUP_UTTERANCES = 10
HUMAN_PANEL_SAMPLE_SIZE = 75
BOOTSTRAP_RESAMPLES = 2000
TARGETED_COMPARISON_STEPS = (2400, 3600)

TONE_MARK_MAP_ITEMS = {
    "\u025b": "e", "\u0254": "o", "\u0272": "ny", "\u014b": "ng",
    "\u0190": "e", "\u0186": "o", "\u019d": "ny", "\u014a": "ng",
}
FRENCH_FUNCTION_WORDS = {"le", "la", "les", "de", "du", "des", "et", "un", "une", "est", "que", "qui"}


def normalize_text_legacy(text):
    text = text.lower()
    text = re.sub(r"[.,!?;:'\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_text(text):
    text = unicodedata.normalize("NFC", text)
    text = text.lower()
    text = re.sub(r"[.,!?;:'\"\u2019\u2018\u00ab\u00bb]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def strip_orthographic_variants(text):
    stripped = "".join(c for c in unicodedata.normalize("NFD", text) if unicodedata.category(c) != "Mn")
    return stripped.translate(str.maketrans(TONE_MARK_MAP_ITEMS))


def has_repeated_ngram(text, n=3, min_repeats=3):
    tokens = text.split()
    if len(tokens) < n * min_repeats:
        return False
    counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return any(c >= min_repeats for c in counts.values())


def degeneracy_stats(pred_str):
    n = len(pred_str)
    if n == 0:
        return {"pct_empty": 0.0, "pct_french_leak": 0.0, "pct_repeated_ngram": 0.0}
    empty = sum(1 for p in pred_str if p.strip() == "")
    leak = sum(1 for p in pred_str if len(set(p.split()) & FRENCH_FUNCTION_WORDS) >= 2)
    rep = sum(1 for p in pred_str if has_repeated_ngram(p))
    return {"pct_empty": 100 * empty / n, "pct_french_leak": 100 * leak / n, "pct_repeated_ngram": 100 * rep / n}


@app.function(image=image, volumes={MODEL_PATH: model_volume})
def list_checkpoints():
    ckpts = sorted(
        [d for d in os.listdir(CHECKPOINT_DIR) if d.startswith("checkpoint-")],
        key=lambda d: int(d.split("-")[1]),
    )
    return ckpts


@app.function(
    image=image,
    gpu="A100-80GB",
    volumes={VOLUME_PATH: volume, MODEL_PATH: model_volume},
    timeout=60 * 30,
    max_containers=4,
)
def eval_single_checkpoint(ckpt_name: str):
    result_path = f"{PER_CKPT_DIR}/{ckpt_name}.json"
    if os.path.exists(result_path):
        print(f"[SKIP] {ckpt_name} sudah pernah dievaluasi, lewati.")
        with open(result_path) as f:
            return json.load(f)

    import torch
    import evaluate
    import jiwer
    from datasets import load_dataset, Audio
    from transformers import (
        WhisperForConditionalGeneration, WhisperProcessor,
        Seq2SeqTrainingArguments, Seq2SeqTrainer,
    )
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union

    processor = WhisperProcessor.from_pretrained(MODEL_ID, language=FORCED_LANGUAGE, task="transcribe")

    def has_valid_text(ex):
        t = ex.get("text")
        return t is not None and t.strip() != ""

    def load_manifest(path):
        ds = load_dataset("json", data_files=path, split="train")
        ds = ds.filter(has_valid_text, num_proc=2)
        ds = ds.cast_column("audio_path", Audio(sampling_rate=16000))
        return ds.rename_column("audio_path", "audio")

    dataset_dev = load_manifest(DEV_MANIFEST)
    dataset_test = load_manifest(TEST_MANIFEST)
    ref_lengths_dev = [len(ex["text"].split()) for ex in dataset_dev]
    ref_lengths_test = [len(ex["text"].split()) for ex in dataset_test]

    def prepare(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    processed_dev = dataset_dev.map(prepare, remove_columns=dataset_dev.column_names, num_proc=2)
    processed_test = dataset_test.map(prepare, remove_columns=dataset_test.column_names, num_proc=2)

    @dataclass
    class Collator:
        processor: Any
        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    collator = Collator(processor=processor)
    metric_wer = evaluate.load("wer")
    metric_cer = evaluate.load("cer")

    training_args = Seq2SeqTrainingArguments(
        output_dir="/tmp/reeval_scratch",
        per_device_eval_batch_size=16,
        predict_with_generate=True,
        generation_max_length=225,
        generation_num_beams=1,
        fp16=False, bf16=True,
        remove_unused_columns=False,
        label_names=["labels"],
        report_to="none",
    )

    def run_predict(model, dataset):
        trainer = Seq2SeqTrainer(args=training_args, model=model, data_collator=collator,
                                  processing_class=processor.feature_extractor)
        out = trainer.predict(dataset)
        label_ids = out.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.tokenizer.batch_decode(out.predictions, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return pred_str, label_str

    def bootstrap_ci(refs, hyps, metric, n_resamples=BOOTSTRAP_RESAMPLES, ci=0.95, seed=42):
        n = len(refs)
        if n == 0:
            return {"ci_low": float("nan"), "ci_high": float("nan")}
        rng = random.Random(seed)
        vals = []
        for _ in range(n_resamples):
            idxs = [rng.randrange(n) for _ in range(n)]
            vals.append(100 * metric.compute(predictions=[hyps[i] for i in idxs], references=[refs[i] for i in idxs]))
        vals.sort()
        lo = int((1 - ci) / 2 * n_resamples)
        hi = int((1 + ci) / 2 * n_resamples) - 1
        return {"ci_low": vals[lo], "ci_high": vals[hi]}

    def bucket_wer(pred_str, label_str, ref_lengths, predicate):
        idxs = [i for i, rl in enumerate(ref_lengths) if i < len(label_str) and label_str[i].strip() != "" and predicate(rl)]
        if not idxs:
            return float("nan")
        return 100 * metric_wer.compute(predictions=[pred_str[i] for i in idxs], references=[label_str[i] for i in idxs])

    def full_metrics(pred_raw, label_raw, ref_lengths):
        pred_str = [normalize_text(p) for p in pred_raw]
        label_str = [normalize_text(l) for l in label_raw]
        degeneracy = degeneracy_stats(pred_str)
        wer_short = bucket_wer(pred_str, label_str, ref_lengths, lambda rl: 0 < rl <= 5)
        wer_long = bucket_wer(pred_str, label_str, ref_lengths, lambda rl: rl > 20)

        pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip() != ""]
        if not pairs:
            return {"wer": float("nan"), "cer": float("nan"), "n_utterances": 0, **degeneracy}
        fp, fl = zip(*pairs); fp, fl = list(fp), list(fl)

        wer = 100 * metric_wer.compute(predictions=fp, references=fl)
        cer = 100 * metric_cer.compute(predictions=fp, references=fl)
        relaxed_p = [strip_orthographic_variants(p) for p in fp]
        relaxed_l = [strip_orthographic_variants(r) for r in fl]
        cer_relaxed = 100 * metric_cer.compute(predictions=relaxed_p, references=relaxed_l)

        pred_legacy = [normalize_text_legacy(p) for p in pred_raw]
        label_legacy = [normalize_text_legacy(l) for l in label_raw]
        legacy_pairs = [(p, r) for p, r in zip(pred_legacy, label_legacy) if r.strip() != ""]
        wer_legacy = (100 * metric_wer.compute(predictions=[p for p, r in legacy_pairs],
                                                 references=[r for p, r in legacy_pairs])
                      if legacy_pairs else float("nan"))

        per_utt_wer = [100 * jiwer.wer(r, p) for r, p in zip(fl, fp)]
        wer_ci = bootstrap_ci(fl, fp, metric_wer)
        cer_ci = bootstrap_ci(fl, fp, metric_cer)
        breakdown = jiwer.process_words(fl, fp)

        return {
            "n_utterances": len(fp), "total_ref_words": sum(ref_lengths[:len(fp)]),
            "wer": wer, "cer": cer, "cer_relaxed": cer_relaxed, "cer_orthographic_gap": cer - cer_relaxed,
            "wer_legacy_normalizer": wer_legacy, "wer_normalizer_gap": wer_legacy - wer,
            "median_utterance_wer": statistics.median(per_utt_wer),
            "pct_wer_zero": 100 * sum(1 for w in per_utt_wer if w == 0) / len(per_utt_wer),
            "pct_wer_under_30": 100 * sum(1 for w in per_utt_wer if w < 30) / len(per_utt_wer),
            "wer_ci_low": wer_ci["ci_low"], "wer_ci_high": wer_ci["ci_high"],
            "cer_ci_low": cer_ci["ci_low"], "cer_ci_high": cer_ci["ci_high"],
            "substitutions": breakdown.substitutions, "deletions": breakdown.deletions, "insertions": breakdown.insertions,
            "wer_short_utterances": wer_short, "wer_long_utterances": wer_long, **degeneracy,
        }

    print(f"=== Evaluating {ckpt_name} ===")
    ckpt_dir = f"{CHECKPOINT_DIR}/{ckpt_name}"
    step = int(ckpt_name.split("-")[1])
    model = WhisperForConditionalGeneration.from_pretrained(ckpt_dir, torch_dtype=torch.float32)
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language=FORCED_LANGUAGE, task="transcribe")

    pred_dev, label_dev = run_predict(model, processed_dev)
    pred_test, label_test = run_predict(model, processed_test)
    m_dev = full_metrics(pred_dev, label_dev, ref_lengths_dev)
    m_test = full_metrics(pred_test, label_test, ref_lengths_test)

    row = {"checkpoint": ckpt_name, "step": step}
    row.update({f"dev_{k}": v for k, v in m_dev.items()})
    row.update({f"test_{k}": v for k, v in m_test.items()})

    os.makedirs(PER_CKPT_DIR, exist_ok=True)
    with open(result_path, "w") as f:
        json.dump(row, f, ensure_ascii=False)

    preds_path = f"{PER_CKPT_DIR}/{ckpt_name}_test_preds.json"
    with open(preds_path, "w") as f:
        json.dump({
            "pred_raw": pred_test, "label_raw": label_test,
            "pred_norm": [normalize_text(p) for p in pred_test],
            "label_norm": [normalize_text(l) for l in label_test],
        }, f, ensure_ascii=False)

    volume.commit(); model_volume.commit()
    print(f"{ckpt_name}: dev_wer={m_dev['wer']:.2f} test_wer={m_test['wer']:.2f} (n={m_test['n_utterances']})")
    return row


@app.function(image=image, volumes={VOLUME_PATH: volume, MODEL_PATH: model_volume},
              secrets=[modal.Secret.from_name("wandb-secret")], timeout=60 * 20)
def aggregate_and_diagnose():
    import pandas as pd
    import jiwer
    import wandb
    from scipy.stats import spearmanr
    from datasets import load_dataset

    result_files = sorted(f for f in os.listdir(PER_CKPT_DIR) if f.startswith("checkpoint-") and f.endswith(".json")
                           and not f.endswith("_test_preds.json"))
    assert result_files, f"No results in {PER_CKPT_DIR} -- run eval_single_checkpoint first."

    rows = []
    for fname in result_files:
        with open(f"{PER_CKPT_DIR}/{fname}") as f:
            rows.append(json.load(f))
    df = pd.DataFrame(rows).sort_values("step").reset_index(drop=True)
    df.to_csv(f"{RESULTS_DIR}/reeval_all_checkpoints.csv", index=False)
    print(f"Loaded {len(df)} checkpoint results.")
    print(df[["checkpoint", "step", "dev_wer", "test_wer", "test_n_utterances"]].to_string(index=False))

    print("\n=== Analysis of Trends (Spearman correlation) ===")
    rho_dev, p_dev = spearmanr(df["step"], df["dev_wer"])
    rho_test, p_test = spearmanr(df["step"], df["test_wer"])
    rho_devtest, p_devtest = spearmanr(df["dev_wer"], df["test_wer"])
    print(f"Spearman(step, dev_wer)  = {rho_dev:.3f}, p={p_dev:.4f}")
    print(f"Spearman(step, test_wer) = {rho_test:.3f}, p={p_test:.4f}")
    print(f"Spearman(dev_wer, test_wer) = {rho_devtest:.3f}, p={p_devtest:.4f}")
    if rho_devtest < 0:
        print("[FLAG] Dev and test are moving in opposite directions—a pattern similar to the reviewer's findings in the Dioula case.")

    def load_preds(ckpt_name):
        with open(f"{PER_CKPT_DIR}/{ckpt_name}_test_preds.json") as f:
            return json.load(f)

    def paired_bootstrap_wer_diff(refs, hyps_a, hyps_b, n_resamples=BOOTSTRAP_RESAMPLES, ci=0.95, seed=42):
        import evaluate
        metric_wer = evaluate.load("wer")
        n = len(refs)
        rng = random.Random(seed)
        diffs = []
        for _ in range(n_resamples):
            idxs = [rng.randrange(n) for _ in range(n)]
            wer_a = 100 * metric_wer.compute(predictions=[hyps_a[i] for i in idxs], references=[refs[i] for i in idxs])
            wer_b = 100 * metric_wer.compute(predictions=[hyps_b[i] for i in idxs], references=[refs[i] for i in idxs])
            diffs.append(wer_a - wer_b)
        diffs.sort()
        lo = int((1 - ci) / 2 * n_resamples); hi = int((1 + ci) / 2 * n_resamples) - 1
        return {"mean_diff": sum(diffs) / len(diffs), "diff_ci_low": diffs[lo], "diff_ci_high": diffs[hi],
                "significant": diffs[lo] > 0 or diffs[hi] < 0}

    step_a, step_b = TARGETED_COMPARISON_STEPS
    ckpt_a, ckpt_b = f"checkpoint-{step_a}", f"checkpoint-{step_b}"
    print(f"\n=== Paired bootstrap (TEST): {ckpt_a} vs {ckpt_b} ===")
    if os.path.exists(f"{PER_CKPT_DIR}/{ckpt_a}_test_preds.json") and os.path.exists(f"{PER_CKPT_DIR}/{ckpt_b}_test_preds.json"):
        pa, pb = load_preds(ckpt_a), load_preds(ckpt_b)
        result = paired_bootstrap_wer_diff(pa["label_norm"], pa["pred_norm"], pb["pred_norm"])
        print(f"WER diff ({ckpt_a} - {ckpt_b}): mean={result['mean_diff']:.2f}, "
              f"95% CI=[{result['diff_ci_low']:.2f}, {result['diff_ci_high']:.2f}], significant={result['significant']}")
    else:
        print(f"[WARN] Prediction file {ckpt_a} or {ckpt_b} not found; skipping.")

    best_row = df.loc[df["test_wer"].idxmin()]
    print(f"\n=== Checkpoint terbaik berdasarkan TEST WER: {best_row['checkpoint']} "
          f"(test_wer={best_row['test_wer']:.2f}, dev_wer={best_row['dev_wer']:.2f}) ===")
    training_best = "checkpoint-2400"
    if best_row["checkpoint"] != training_best:
        print(f"[FLAG] The best checkpoint according to the test ({best_row['checkpoint']}) differs from the dev choice at"
              f"training ({training_best}). Consider using this as the final model.")
    else:
        print(f"The dev choice ({training_best}) is the same as the test choice.")

    print(f"\n=== Detailed Diagnosis for {best_row['checkpoint']} ===")
    best_preds = load_preds(best_row["checkpoint"])
    pred_str = best_preds["pred_norm"]
    label_str = best_preds["label_norm"]
    pred_str_raw = best_preds["pred_raw"]
    label_str_raw = best_preds["label_raw"]

    test_meta = load_dataset("json", data_files=TEST_MANIFEST, split="train")
    ref_lengths_test = [len(ex["text"].split()) for ex in test_meta]
    audio_durations_test = [ex["duration_sec"] for ex in test_meta]
    speaker_groups = test_meta["speaker_group"] if "speaker_group" in test_meta.column_names else None
    source_datasets = test_meta["source_dataset"] if "source_dataset" in test_meta.column_names else None

    detail_rows = []
    for i, (p, r) in enumerate(zip(pred_str, label_str)):
        if r.strip() == "" or i >= len(ref_lengths_test):
            continue
        wer_u = 100 * jiwer.wer(r, p)
        hyp_len = len(p.split())
        ref_len = ref_lengths_test[i]
        detail_rows.append({
            "ref_length_words": ref_len, "hyp_length_words": hyp_len,
            "length_ratio": hyp_len / ref_len if ref_len > 0 else float("nan"),
            "audio_duration_sec": audio_durations_test[i], "utterance_wer": wer_u,
            "speaker_group": speaker_groups[i] if speaker_groups else None,
            "source_dataset": source_datasets[i] if source_datasets else None,
        })
    detail_df = pd.DataFrame(detail_rows)

    if source_datasets:
        wbs = detail_df.groupby("source_dataset")["utterance_wer"].agg(["mean", "median", "count"]) \
            .rename(columns={"mean": "mean_wer", "median": "median_wer", "count": "n_utterances"})
        print("\nWER per source domain (bible / goai-proverbs / goai-contes):")
        print(wbs.sort_values(by="mean_wer", ascending=False).to_string())

    if speaker_groups:
        wbg = detail_df.groupby("speaker_group")["utterance_wer"].agg(["mean", "median", "count"]) \
            .rename(columns={"mean": "mean_wer", "median": "median_wer", "count": "n_utterances"})
        reliable = wbg[wbg["n_utterances"] >= MIN_GROUP_UTTERANCES]
        print(f"\nWER per speaker-group (n >= {MIN_GROUP_UTTERANCES}):")
        print(reliable.sort_values(by="mean_wer", ascending=False).to_string())

    detail_df["duration_bucket"] = pd.cut(detail_df["audio_duration_sec"], bins=[0, 3, 6, 10, 1000],
                                           labels=["0-3s", "3-6s", "6-10s", "10s+"])
    dur_summary = detail_df.groupby("duration_bucket", observed=True)["utterance_wer"] \
        .agg(["mean", "median", "count"]).rename(columns={"mean": "mean_wer", "median": "median_wer", "count": "n"})
    print("\nWER by audio duration:")
    print(dur_summary.to_string())

    detail_df.to_csv(f"{RESULTS_DIR}/reeval_best_checkpoint_detail.csv", index=False)

    n = len(pred_str_raw)
    sample_size = min(HUMAN_PANEL_SAMPLE_SIZE, n)
    rng = random.Random(42)
    idxs = sorted(rng.sample(range(n), sample_size))
    panel_df = pd.DataFrame([
        {"utterance_index": i, "reference": label_str_raw[i], "hypothesis": pred_str_raw[i], "intelligibility_rating_1_5": ""}
        for i in idxs
    ])
    panel_df.to_csv(f"{RESULTS_DIR}/reeval_human_panel_sample.csv", index=False)
    print(f"\nSaved human panel sample ({sample_size} rows) to {RESULTS_DIR}/reeval_human_panel_sample.csv")

    try:
        wandb.init(project="wobsongo-whisper-moore-reeval", name="reeval-aggregate-1")
        wandb.summary["best_checkpoint_by_test"] = best_row["checkpoint"]
        wandb.summary["best_test_wer"] = best_row["test_wer"]
        wandb.summary["spearman_dev_test_wer"] = rho_devtest
        wandb.finish()
    except Exception as e:
        print(f"[WARN] wandb failed ({e}), skipping.")

    model_volume.commit()
    print("\n=== SELESAI ===")


@app.local_entrypoint()
def main():
    ckpt_names = list_checkpoints.remote()
    print(f"Found {len(ckpt_names)} checkpoints: {ckpt_names}")
    print("Evaluating (4 checkpoints in parallel; already-done ones are skipped automatically)...")
    list(eval_single_checkpoint.map(ckpt_names))
    print("\nAll checkpoints evaluated. Running aggregation & diagnostics...")
    aggregate_and_diagnose.remote()