import os
import re
import glob
import torch
import wandb
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import Dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel
import evaluate
import jiwer
import unicodedata
import statistics
import random
from collections import Counter

print("[PROC] cv_reeval_checkpoints -- re-evaluation of existing checkpoints on Common Voice")

BASE_MODEL_PATH = "/models/whisper-large-v2-base"
BAMBARA_ADAPTER_PATH = "/models/bambara-asr-v2-adapter"
CV_DIOULA_PATH = "/data/mozilla-dioula-dataset/cv-corpus-26.0-2026-06-12/dyu"
CHECKPOINT_DIR = "/output/checkpoints_bambara_init_dioula"
RESULTS_CSV = "/output/cv_reeval_clean_results.csv"
DETAIL_CSV = "/output/cv_reeval_best_checkpoint_utterance_detail.csv"
HUMAN_PANEL_CSV = "/output/cv_reeval_human_panel_sample.csv"

MT_MODEL_PATH = None
MT_REFERENCE_COLUMN = "fr"
HUMAN_PANEL_SAMPLE_SIZE = 75

processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="french", task="transcribe")

def get_target_text(example):
    return example.get("dyu") or example.get("sentence") or example.get("transcription") or example.get("text") or ""

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(get_target_text(batch)).input_ids
    return batch

def has_valid_dyu_text(example):
    text = example.get("dyu")
    return text is not None and text.strip() != ""

def load_cv_split(tsv_name):
    tsv_path = f"{CV_DIOULA_PATH}/{tsv_name}"
    clips_dir = f"{CV_DIOULA_PATH}/clips"

    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["sentence"].notna() & (df["sentence"].str.strip() != "")]
    df["audio"] = df["path"].apply(lambda p: f"{clips_dir}/{p}")
    df = df.rename(columns={"sentence": "dyu", "age": "age_group"})
    
    keep_cols = ["audio", "dyu"]
    for col in ["client_id", "gender", "age_group"]:
        if col in df.columns:
            keep_cols.append(col)
            
    df = df[keep_cols].reset_index(drop=True)
    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds

print(f"Loading Common Voice test set from {CV_DIOULA_PATH} ...")
dataset_test = load_cv_split("test.tsv")
dataset_test = dataset_test.filter(has_valid_dyu_text, num_proc=4)
dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000))
processed_test = dataset_test.map(prepare_dataset, remove_columns=dataset_test.column_names, num_proc=4)
print(f"Test set (CV): {len(processed_test)} rows")

ref_lengths = [len(get_target_text(ex).split()) for ex in dataset_test]
audio_durations = [len(a["array"]) / a["sampling_rate"] for a in dataset_test["audio"]]

@dataclass
class CleanWhisperDataCollator:
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

eval_data_collator = CleanWhisperDataCollator(processor=processor)

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")

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

TONE_MARK_MAP = str.maketrans({
    "\u025b": "e", "\u0254": "o", "\u0272": "ny", "\u014b": "ng",
    "\u0190": "e", "\u0186": "o", "\u019d": "ny", "\u014a": "ng",
})

def strip_orthographic_variants(text):
    return text.translate(TONE_MARK_MAP)

FRENCH_FUNCTION_WORDS = {"le", "la", "les", "de", "du", "des", "et", "un", "une", "est", "que", "qui"}

def has_repeated_ngram(text, n=3, min_repeats=3):
    tokens = text.split()
    if len(tokens) < n * min_repeats:
        return False
    ngram_counts = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    return any(c >= min_repeats for c in ngram_counts.values())

def degeneracy_stats(pred_str):
    n = len(pred_str)
    if n == 0:
        return {"pct_empty": 0.0, "pct_french_leak": 0.0, "pct_repeated_ngram": 0.0}
    empty = sum(1 for p in pred_str if p.strip() == "")
    french_leak = sum(
        1 for p in pred_str
        if len(set(p.split()) & FRENCH_FUNCTION_WORDS) >= 2
    )
    repeated = sum(1 for p in pred_str if has_repeated_ngram(p))
    return {
        "pct_empty": 100 * empty / n,
        "pct_french_leak": 100 * french_leak / n,
        "pct_repeated_ngram": 100 * repeated / n,
    }

def bootstrap_wer_ci(refs, hyps, n_resamples=500, ci=0.95, seed=42):
    n = len(refs)
    if n == 0: return {"wer_ci_low": float("nan"), "wer_ci_high": float("nan")}
    rng = random.Random(seed)
    resampled_wers = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_refs = [refs[i] for i in idxs]
        sample_hyps = [hyps[i] for i in idxs]
        resampled_wers.append(100 * metric_wer.compute(predictions=sample_hyps, references=sample_refs))
    resampled_wers.sort()
    lo = int((1 - ci) / 2 * n_resamples)
    hi = int((1 + ci) / 2 * n_resamples) - 1
    return {"wer_ci_low": resampled_wers[lo], "wer_ci_high": resampled_wers[hi]}

def bootstrap_cer_ci(refs, hyps, n_resamples=500, ci=0.95, seed=42):
    n = len(refs)
    if n == 0: return {"cer_ci_low": float("nan"), "cer_ci_high": float("nan")}
    rng = random.Random(seed)
    resampled_cers = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_refs = [refs[i] for i in idxs]
        sample_hyps = [hyps[i] for i in idxs]
        resampled_cers.append(100 * metric_cer.compute(predictions=sample_hyps, references=sample_refs))
    resampled_cers.sort()
    lo = int((1 - ci) / 2 * n_resamples)
    hi = int((1 + ci) / 2 * n_resamples) - 1
    return {"cer_ci_low": resampled_cers[lo], "cer_ci_high": resampled_cers[hi]}

def paired_bootstrap_wer_diff(refs, hyps_a, hyps_b, n_resamples=500, ci=0.95, seed=42):
    n = len(refs)
    rng = random.Random(seed)
    diffs = []
    for _ in range(n_resamples):
        idxs = [rng.randrange(n) for _ in range(n)]
        sample_refs = [refs[i] for i in idxs]
        sample_hyps_a = [hyps_a[i] for i in idxs]
        sample_hyps_b = [hyps_b[i] for i in idxs]
        wer_a = 100 * metric_wer.compute(predictions=sample_hyps_a, references=sample_refs)
        wer_b = 100 * metric_wer.compute(predictions=sample_hyps_b, references=sample_refs)
        diffs.append(wer_a - wer_b)
    diffs.sort()
    lo = int((1 - ci) / 2 * n_resamples)
    hi = int((1 + ci) / 2 * n_resamples) - 1
    return {
        "mean_diff": sum(diffs) / len(diffs),
        "diff_ci_low": diffs[lo],
        "diff_ci_high": diffs[hi],
        "significant": diffs[lo] > 0 or diffs[hi] < 0, 
    }

def top_substitution_pairs(refs, hyps, top_k=15):
    counter = Counter()
    for ref, hyp in zip(refs, hyps):
        out = jiwer.process_words(ref, hyp)
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        for chunk in out.alignments[0]:
            if chunk.type == "substitute":
                ref_words = " ".join(ref_tokens[chunk.ref_start_idx:chunk.ref_end_idx])
                hyp_words = " ".join(hyp_tokens[chunk.hyp_start_idx:chunk.hyp_end_idx])
                counter[(ref_words, hyp_words)] += 1
    return counter.most_common(top_k)

def sample_for_human_panel(pred_str_raw, label_str_raw, dataset, sample_size, seed=42):
    n = len(pred_str_raw)
    sample_size = min(sample_size, n)
    rng = random.Random(seed)
    idxs = sorted(rng.sample(range(n), sample_size))
    rows = []
    for i in idxs:
        rows.append({
            "utterance_index": i,
            "reference": label_str_raw[i],
            "hypothesis": pred_str_raw[i],
            "intelligibility_rating_1_5": "",
        })
    panel_df = pd.DataFrame(rows)
    panel_df.to_csv(HUMAN_PANEL_CSV, index=False)
    print(f"Saved {sample_size}-utterance human panel sample to {HUMAN_PANEL_CSV}")

def bucket_wer_and_length_ratio(pred_str, label_str, ref_lengths, predicate):
    idxs = [
        i for i, rl in enumerate(ref_lengths)
        if i < len(label_str) and label_str[i].strip() != "" and predicate(rl)
    ]
    if not idxs:
        return float("nan"), float("nan")
    bucket_wer = 100 * metric_wer.compute(
        predictions=[pred_str[i] for i in idxs],
        references=[label_str[i] for i in idxs],
    )
    ratios = [len(pred_str[i].split()) / ref_lengths[i] for i in idxs if ref_lengths[i] > 0]
    length_ratio = statistics.mean(ratios) if ratios else float("nan")
    return bucket_wer, length_ratio

def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str_raw = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str_raw = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalize_text(p) for p in pred_str_raw]
    label_str = [normalize_text(l) for l in label_str_raw]

    degeneracy = degeneracy_stats(pred_str)

    wer_short_utt, _ = bucket_wer_and_length_ratio(pred_str, label_str, ref_lengths, lambda rl: 0 < rl <= 5)
    wer_long_utt, length_ratio_long_utt = bucket_wer_and_length_ratio(pred_str, label_str, ref_lengths, lambda rl: rl > 20)

    filtered_pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip() != ""]
    if len(filtered_pairs) == 0:
        return {"wer": float("nan"), "cer": float("nan"), **degeneracy}

    filtered_pred_str, filtered_label_str = zip(*filtered_pairs)
    filtered_pred_str, filtered_label_str = list(filtered_pred_str), list(filtered_label_str)

    wer = 100 * metric_wer.compute(predictions=filtered_pred_str, references=filtered_label_str)
    cer = 100 * metric_cer.compute(predictions=filtered_pred_str, references=filtered_label_str)

    relaxed_pred_str = [strip_orthographic_variants(p) for p in filtered_pred_str]
    relaxed_label_str = [strip_orthographic_variants(r) for r in filtered_label_str]
    cer_relaxed = 100 * metric_cer.compute(predictions=relaxed_pred_str, references=relaxed_label_str)

    pred_str_legacy = [normalize_text_legacy(p) for p in pred_str_raw]
    label_str_legacy = [normalize_text_legacy(l) for l in label_str_raw]
    legacy_pairs = [(p, r) for p, r in zip(pred_str_legacy, label_str_legacy) if r.strip() != ""]
    if len(legacy_pairs) > 0:
        legacy_pred, legacy_label = zip(*legacy_pairs)
        wer_legacy_normalizer = 100 * metric_wer.compute(predictions=list(legacy_pred), references=list(legacy_label))
    else:
        wer_legacy_normalizer = float("nan")

    per_utt_wer = [
        100 * jiwer.wer(r, p) for r, p in zip(filtered_label_str, filtered_pred_str)
    ]
    median_wer = statistics.median(per_utt_wer)
    pct_wer_zero = 100 * sum(1 for w in per_utt_wer if w == 0) / len(per_utt_wer)
    pct_wer_under_30 = 100 * sum(1 for w in per_utt_wer if w < 30) / len(per_utt_wer)

    wer_ci = bootstrap_wer_ci(filtered_label_str, filtered_pred_str)
    cer_ci = bootstrap_cer_ci(filtered_label_str, filtered_pred_str)

    breakdown = jiwer.process_words(filtered_label_str, filtered_pred_str)

    return {
        "wer": wer,
        "cer": cer,
        "cer_relaxed": cer_relaxed,
        "cer_orthographic_gap": cer - cer_relaxed,
        "wer_legacy_normalizer": wer_legacy_normalizer,
        "wer_normalizer_gap": wer_legacy_normalizer - wer,
        "median_utterance_wer": median_wer,
        "pct_wer_zero": pct_wer_zero,
        "pct_wer_under_30": pct_wer_under_30,
        "wer_ci_low": wer_ci["wer_ci_low"],
        "wer_ci_high": wer_ci["wer_ci_high"],
        "cer_ci_low": cer_ci["cer_ci_low"],
        "cer_ci_high": cer_ci["cer_ci_high"],
        "substitutions": breakdown.substitutions,
        "deletions": breakdown.deletions,
        "insertions": breakdown.insertions,
        "wer_short_utterances": wer_short_utt,
        "wer_long_utterances": wer_long_utt,
        "length_ratio_long_utterances": length_ratio_long_utt,
        **degeneracy,
    }

def load_eval_model(ckpt_dir):
    base_model = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL_PATH, torch_dtype=torch.float32)
    bambara_model = PeftModel.from_pretrained(base_model, BAMBARA_ADAPTER_PATH)
    merged_model = bambara_model.merge_and_unload()
    merged_model.resize_token_embeddings(len(processor.tokenizer))
    eval_model = PeftModel.from_pretrained(merged_model, ckpt_dir)
    eval_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")
    eval_model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")
    return eval_model, merged_model, bambara_model, base_model

def run_predict(model, dataset):
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        data_collator=eval_data_collator,
        processing_class=processor.feature_extractor,
    )
    output = trainer.predict(dataset)
    label_ids = output.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str_raw = processor.tokenizer.batch_decode(output.predictions, skip_special_tokens=True)
    label_str_raw = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    pred_str_norm = [normalize_text(p) for p in pred_str_raw]
    label_str_norm = [normalize_text(l) for l in label_str_raw]
    return pred_str_raw, label_str_raw, pred_str_norm, label_str_norm, trainer

checkpoint_dirs = sorted(
    glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*")),
    key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
)
assert len(checkpoint_dirs) > 0, f"No checkpoint-* folders found in {CHECKPOINT_DIR}"
print(f"Found {len(checkpoint_dirs)} checkpoints: {[os.path.basename(c) for c in checkpoint_dirs]}")

training_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/cv_reeval_scratch",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=100,
    generation_num_beams=1,
    fp16=True,
    bf16=False,
    remove_unused_columns=False,
    label_names=["labels"],
    report_to="none",
)

WANDB_ENABLED = True
try:
    wandb.init(
        project="wobsongo-whisper-bambara-dioula-reeval",
        id="whisper-large-v2-bambara-dioula-v2-cv-reeval",
        resume="allow",
        name="whisper-large-v2-bambara-dioula-v2-cv-reeval",
        settings=wandb.Settings(init_timeout=180),
        config={
            "purpose": "Re-evaluate existing checkpoints on Common Voice test set",
        },
    )
except Exception as e:
    print(f"[WARN] wandb.init failed ({e}). Continuing re-evaluation WITHOUT wandb logging.")
    WANDB_ENABLED = False

results = []

for ckpt_dir in checkpoint_dirs:
    ckpt_name = os.path.basename(ckpt_dir)
    step = int(re.search(r"checkpoint-(\d+)", ckpt_name).group(1))
    print(f"\n=== Re-evaluating {ckpt_name} on Common Voice ===")

    eval_model, merged_model, bambara_model, base_model = load_eval_model(ckpt_dir)

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=eval_model,
        data_collator=eval_data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    metrics = trainer.evaluate(eval_dataset=processed_test, metric_key_prefix="cv_test")
    row = {"checkpoint": ckpt_name, "step": step}
    for key in [
        "wer", "cer", "cer_relaxed", "cer_orthographic_gap",
        "wer_legacy_normalizer", "wer_normalizer_gap",
        "median_utterance_wer", "pct_wer_zero", "pct_wer_under_30",
        "wer_ci_low", "wer_ci_high", "cer_ci_low", "cer_ci_high",
        "substitutions", "deletions", "insertions",
        "wer_short_utterances", "wer_long_utterances", "length_ratio_long_utterances",
        "pct_empty", "pct_french_leak", "pct_repeated_ngram",
    ]:
        row[f"cv_test_{key}"] = metrics.get(f"cv_test_{key}")

    print(f"{ckpt_name}: wer={row['cv_test_wer']:.2f}  cer={row['cv_test_cer']:.2f}  "
          f"ci=[{row['cv_test_wer_ci_low']:.2f}, {row['cv_test_wer_ci_high']:.2f}]")

    if WANDB_ENABLED:
        try:
            wandb.log({"train/global_step": step, **{k: v for k, v in row.items() if k not in ("checkpoint", "step")}})
        except Exception as e:
            print(f"[WARN] wandb.log failed for {ckpt_name} ({e})")

    results.append(row)

    del eval_model, trainer, merged_model, bambara_model, base_model
    torch.cuda.empty_cache()

df = pd.DataFrame(results).sort_values("cv_test_wer").reset_index(drop=True)
print("\n=== Summary (sorted by best CV WER) ===")
print(df.to_string(index=False))

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved results to {RESULTS_CSV}")

best_row = df.iloc[0]
best_ckpt_name = best_row["checkpoint"]
best_ckpt_path = os.path.join(CHECKPOINT_DIR, best_ckpt_name)
print(f"\nBest checkpoint by CV WER: {best_ckpt_name} (WER={best_row['cv_test_wer']:.2f})")

best_model, merged_model, bambara_model, base_model = load_eval_model(best_ckpt_path)
best_pred_str_raw, best_label_str_raw, best_pred_str, best_label_str, best_trainer = run_predict(best_model, processed_test)
del best_model, best_trainer, merged_model, bambara_model, base_model
torch.cuda.empty_cache()

print("\n=== Paired bootstrap: best checkpoint vs runner-up ===")

if len(df) >= 2:
    second_row = df.iloc[1]
    second_ckpt_path = os.path.join(CHECKPOINT_DIR, second_row["checkpoint"])

    second_model, merged_model, bambara_model, base_model = load_eval_model(second_ckpt_path)
    _, _, second_pred_str, second_label_str, second_trainer = run_predict(second_model, processed_test)
    del second_model, second_trainer, merged_model, bambara_model, base_model
    torch.cuda.empty_cache()

    paired_refs, paired_best_hyp, paired_second_hyp = [], [], []
    for r_best, p_best, p_second in zip(best_label_str, best_pred_str, second_pred_str):
        if r_best.strip() != "":
            paired_refs.append(r_best)
            paired_best_hyp.append(p_best)
            paired_second_hyp.append(p_second)

    paired_result = paired_bootstrap_wer_diff(paired_refs, paired_best_hyp, paired_second_hyp)
    print(f"WER diff ({best_ckpt_name} - {second_row['checkpoint']}): "
          f"mean={paired_result['mean_diff']:.2f}, "
          f"95% CI=[{paired_result['diff_ci_low']:.2f}, {paired_result['diff_ci_high']:.2f}], "
          f"significant={paired_result['significant']}")
else:
    print("[INFO] Only 1 checkpoint, skipping paired bootstrap.")

print("\n=== Detailed diagnostic report on best checkpoint ===")

pred_str, label_str = best_pred_str, best_label_str

has_speaker_id = "client_id" in dataset_test.column_names
speaker_ids = dataset_test["client_id"] if has_speaker_id else None

has_gender = "gender" in dataset_test.column_names
genders = dataset_test["gender"] if has_gender else None
has_age_group = "age_group" in dataset_test.column_names
age_groups = dataset_test["age_group"] if has_age_group else None

detail_rows = []
for i, (p, r) in enumerate(zip(pred_str, label_str)):
    if r.strip() == "": continue
    wer_u = 100 * jiwer.wer(r, p)
    hyp_len = len(p.split())
    ref_len = ref_lengths[i]
    detail_rows.append({
        "ref_length_words": ref_len,
        "hyp_length_words": hyp_len,
        "length_ratio": hyp_len / ref_len if ref_len > 0 else float("nan"),
        "audio_duration_sec": audio_durations[i],
        "utterance_wer": wer_u,
        "speaker_id": speaker_ids[i] if has_speaker_id else None,
        "gender": genders[i] if has_gender else None,
        "age_group": age_groups[i] if has_age_group else None,
    })

detail_df = pd.DataFrame(detail_rows)

if has_speaker_id:
    wer_by_speaker = detail_df.groupby("speaker_id")["utterance_wer"].agg(["mean", "count"]).sort_values(by="mean", ascending=False)
    print("\nPer-speaker WER (Top 10 worst):")
    print(wer_by_speaker.head(10).to_string())

detail_df["duration_bucket"] = pd.cut(
    detail_df["audio_duration_sec"], bins=[0, 3, 6, 10, 1000],
    labels=["0-3s", "3-6s", "6-10s", "10s+"]
)
by_duration = detail_df.groupby("duration_bucket", observed=True).agg(
    mean_wer=("utterance_wer", "mean"),
    mean_length_ratio=("length_ratio", "mean"),
    n_utterances=("utterance_wer", "count"),
)
print("\nWER by audio duration on Common Voice:")
print(by_duration.to_string())

os.makedirs(os.path.dirname(DETAIL_CSV), exist_ok=True)
detail_df.to_csv(DETAIL_CSV, index=False)
print(f"\nSaved per-utterance detail to {DETAIL_CSV}")

sample_for_human_panel(best_pred_str_raw, best_label_str_raw, dataset_test, HUMAN_PANEL_SAMPLE_SIZE)

if WANDB_ENABLED:
    try:
        wandb.summary["best_checkpoint"] = best_ckpt_name
        wandb.summary["best_cv_test_wer"] = best_row["cv_test_wer"]
        wandb.summary["best_cv_test_cer"] = best_row["cv_test_cer"]
        wandb.summary["best_cv_test_substitutions"] = best_row["cv_test_substitutions"]
        wandb.summary["best_cv_test_deletions"] = best_row["cv_test_deletions"]
        wandb.summary["best_cv_test_insertions"] = best_row["cv_test_insertions"]
        wandb.summary["best_cv_test_pct_empty"] = best_row["cv_test_pct_empty"]
        wandb.summary["best_cv_test_pct_french_leak"] = best_row["cv_test_pct_french_leak"]
        wandb.summary["best_cv_test_cer_orthographic_gap"] = best_row["cv_test_cer_orthographic_gap"]
        wandb.summary["best_cv_test_median_utterance_wer"] = best_row["cv_test_median_utterance_wer"]
        wandb.finish()
    except Exception as e:
        print(f"[WARN] wandb summary/finish failed ({e})")