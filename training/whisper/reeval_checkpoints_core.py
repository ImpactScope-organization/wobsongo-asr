import os
import re
import glob
import torch
import wandb
import pandas as pd
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from peft import PeftModel
import evaluate

print("[PROC] reeval_checkpoints -- clean-collator re-evaluation of existing checkpoints")

BASE_MODEL_PATH = "/models/whisper-large-v2-base"
BAMBARA_ADAPTER_PATH = "/models/bambara-asr-v2-adapter"
DATASET_PATH = "/data/dioula-koumankan-dataset"
CHECKPOINT_DIR = "/output/checkpoints_bambara_init_dioula"
RESULTS_CSV = "/output/reeval_clean_results.csv"

processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="french", task="transcribe")


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    target_text = batch.get("dyu") or batch.get("sentence") or batch.get("transcription") or batch.get("text") or ""
    batch["labels"] = processor.tokenizer(target_text).input_ids
    return batch


def has_valid_dyu_text(example):
    text = example.get("dyu")
    return text is not None and text.strip() != ""


print(f"Loading dev set from {DATASET_PATH}/dev ...")
dataset_dev = load_from_disk(f"{DATASET_PATH}/dev")
dataset_dev = dataset_dev.filter(has_valid_dyu_text, num_proc=4)
dataset_dev = dataset_dev.cast_column("audio", Audio(sampling_rate=16000))
processed_dev = dataset_dev.map(prepare_dataset, remove_columns=dataset_dev.column_names, num_proc=4)
print(f"Dev set: {len(processed_dev)} rows")


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


def normalize_text(text):
    text = text.lower()
    text = re.sub(r"[.,!?;:'\"]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    pred_str = [normalize_text(p) for p in pred_str]
    label_str = [normalize_text(l) for l in label_str]

    filtered_pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip() != ""]
    if len(filtered_pairs) == 0:
        return {"wer": float("nan"), "cer": float("nan")}

    filtered_pred_str, filtered_label_str = zip(*filtered_pairs)
    wer = 100 * metric_wer.compute(predictions=filtered_pred_str, references=filtered_label_str)
    cer = 100 * metric_cer.compute(predictions=filtered_pred_str, references=filtered_label_str)
    return {"wer": wer, "cer": cer}


checkpoint_dirs = sorted(
    glob.glob(os.path.join(CHECKPOINT_DIR, "checkpoint-*")),
    key=lambda p: int(re.search(r"checkpoint-(\d+)", p).group(1)),
)
assert len(checkpoint_dirs) > 0, f"No checkpoint-* folders found in {CHECKPOINT_DIR}"
print(f"Found {len(checkpoint_dirs)} checkpoints: {[os.path.basename(c) for c in checkpoint_dirs]}")

training_args = Seq2SeqTrainingArguments(
    output_dir="/tmp/reeval_scratch",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=40,
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
        project="wobsongo-whisper-dioula",
        id="whisper-large-v2-bambara-dioula-v2-clean-reeval",
        resume="allow",
        name="whisper-large-v2-bambara-dioula-v2-clean-reeval",
        settings=wandb.Settings(init_timeout=180),
        config={
            "purpose": "Re-evaluate existing checkpoints with clean (non-augmented) collator",
            "original_run": "whisper-large-v2-bambara-dioula-v2",
        },
    )
except Exception as e:
    print(f"[WARN] wandb.init failed ({e}). Continuing re-evaluation WITHOUT wandb logging. -- "
          f"results will still be saved to {RESULTS_CSV}.")
    WANDB_ENABLED = False

results = []

for ckpt_dir in checkpoint_dirs:
    ckpt_name = os.path.basename(ckpt_dir)
    step = int(re.search(r"checkpoint-(\d+)", ckpt_name).group(1))
    print(f"\n=== Re-evaluating {ckpt_name} ===")

    base_model = WhisperForConditionalGeneration.from_pretrained(
        BASE_MODEL_PATH, torch_dtype=torch.float32
    )
    bambara_model = PeftModel.from_pretrained(base_model, BAMBARA_ADAPTER_PATH)
    merged_model = bambara_model.merge_and_unload()
    merged_model.resize_token_embeddings(len(processor.tokenizer))

    eval_model = PeftModel.from_pretrained(merged_model, ckpt_dir)
    eval_model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")
    eval_model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=eval_model,
        data_collator=eval_data_collator,
        compute_metrics=compute_metrics,
        processing_class=processor.feature_extractor,
    )

    metrics = trainer.evaluate(eval_dataset=processed_dev, metric_key_prefix="clean_dev")
    wer = metrics.get("clean_dev_wer")
    cer = metrics.get("clean_dev_cer")
    print(f"{ckpt_name}: clean_dev_wer={wer:.2f}  clean_dev_cer={cer:.2f}")

    if WANDB_ENABLED:
        try:
            wandb.log({"train/global_step": step, "clean_dev_wer": wer, "clean_dev_cer": cer})
        except Exception as e:
            print(f"[WARN] wandb.log failed for {ckpt_name} ({e}), continuing without logging.")

    results.append({"checkpoint": ckpt_name, "step": step, "clean_dev_wer": wer, "clean_dev_cer": cer})

    del eval_model, trainer, merged_model, bambara_model, base_model
    torch.cuda.empty_cache()

df = pd.DataFrame(results).sort_values("clean_dev_wer")
print("\n=== Summary (sorted by best clean WER) ===")
print(df.to_string(index=False))

os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)
df.to_csv(RESULTS_CSV, index=False)
print(f"\nSaved results to {RESULTS_CSV}")

best_row = df.iloc[0]
best_ckpt_path = os.path.join(CHECKPOINT_DIR, best_row["checkpoint"])
print(f"\nBest checkpoint by clean WER: {best_row['checkpoint']} "
      f"(WER={best_row['clean_dev_wer']:.2f}, CER={best_row['clean_dev_cer']:.2f})")
print(f"Full path on volume: {best_ckpt_path}")

if WANDB_ENABLED:
    try:
        wandb.summary["best_checkpoint"] = best_row["checkpoint"]
        wandb.summary["best_clean_dev_wer"] = best_row["clean_dev_wer"]
        wandb.summary["best_clean_dev_cer"] = best_row["clean_dev_cer"]
        wandb.finish()
    except Exception as e:
        print(f"[WARN] wandb summary/finish failed ({e}); results have been saved to CSV.")