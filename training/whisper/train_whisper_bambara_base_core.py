import os
import modal
import torch
import wandb
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_from_disk, Audio, Dataset, concatenate_datasets
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint
from peft import LoraConfig, PeftModel, get_peft_model
from accelerate import PartialState
import evaluate
import re
import pandas as pd

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
print(f"[PROC] whisper-large-v2 (bambara-init -> dioula, aug) | LOCAL_RANK={LOCAL_RANK}, "
      f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

BASE_MODEL_PATH = "/models/whisper-large-v2-base"
BAMBARA_ADAPTER_PATH = "/models/bambara-asr-v2-adapter"
DATASET_PATH = "/data/dioula-koumankan-dataset"

processor = WhisperProcessor.from_pretrained(BASE_MODEL_PATH, language="french", task="transcribe")

torch.cuda.set_device(LOCAL_RANK)


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    target_text = batch.get("dyu") or batch.get("sentence") or batch.get("transcription") or batch.get("text") or ""
    batch["labels"] = processor.tokenizer(target_text).input_ids
    return batch


CV_DIOULA_PATH = "/data/mozilla-dioula-dataset/cv-corpus-26.0-2026-06-12/dyu"

def load_cv_split(tsv_name):
    tsv_path = f"{CV_DIOULA_PATH}/{tsv_name}"
    clips_dir = f"{CV_DIOULA_PATH}/clips"

    df = pd.read_csv(tsv_path, sep="\t")
    df = df[df["sentence"].notna() & (df["sentence"].str.strip() != "")]
    df["audio"] = df["path"].apply(lambda p: f"{clips_dir}/{p}")
    df = df.rename(columns={"sentence": "dyu"})
    df = df[["audio", "dyu"]].reset_index(drop=True)

    ds = Dataset.from_pandas(df, preserve_index=False)
    return ds


with PartialState().local_main_process_first():
    print(f"Loading Koumankan4Dyula from local volume ({DATASET_PATH})...")
    dataset_train = load_from_disk(f"{DATASET_PATH}/train")
    dataset_dev = load_from_disk(f"{DATASET_PATH}/dev")

    print("Loading Common Voice Dioula (train+dev+test) as independent test set...")
    cv_train = load_cv_split("train.tsv")
    cv_dev   = load_cv_split("dev.tsv")
    cv_test  = load_cv_split("test.tsv")
    dataset_test = concatenate_datasets([cv_train, cv_dev, cv_test])
    print(f"Common Voice test pool: {len(dataset_test)} rows")

    def has_valid_dyu_text(example):
        text = example.get("dyu")
        return text is not None and text.strip() != ""

    n_train_before, n_dev_before, n_test_before = len(dataset_train), len(dataset_dev), len(dataset_test)
    dataset_train = dataset_train.filter(has_valid_dyu_text, num_proc=4)
    dataset_dev = dataset_dev.filter(has_valid_dyu_text, num_proc=4)
    dataset_test = dataset_test.filter(has_valid_dyu_text, num_proc=4)
    print(f"Filter blank rows -- train: {n_train_before}->{len(dataset_train)}, "
          f"dev: {n_dev_before}->{len(dataset_dev)}, test: {n_test_before}->{len(dataset_test)}")

    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
    dataset_dev = dataset_dev.cast_column("audio", Audio(sampling_rate=16000))
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Training Data   : {len(dataset_train)} rows (~8h)")
    print(f"Dev  : {len(dataset_dev)} rows (~1h36m)")
    print(f"Test : {len(dataset_test)} rows")

    # TODO: Challenge probe
    # dataset_challenge = load_from_disk(f"{DATASET_PATH}/challenge_probe")
    # dataset_challenge = dataset_challenge.cast_column("audio", Audio(sampling_rate=16000))

    print("Processing audio into a spectrogram matrix...")
    processed_train = dataset_train.map(
        prepare_dataset, remove_columns=dataset_train.column_names, num_proc=4
    )
    processed_dev = dataset_dev.map(
        prepare_dataset, remove_columns=dataset_dev.column_names, num_proc=4
    )
    processed_test = dataset_test.map(
        prepare_dataset, remove_columns=dataset_test.column_names, num_proc=4
    )

print("Loading base model (whisper-large-v2) from volume...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH, torch_dtype=torch.float32
)

print("Attaching the Bambara adapter and merging it with the base model....")
bambara_model = PeftModel.from_pretrained(base_model, BAMBARA_ADAPTER_PATH)
merged_model = bambara_model.merge_and_unload()  # bobot Bambara "menyatu" ke base

merged_model.resize_token_embeddings(len(processor.tokenizer))
merged_model.enable_input_require_grads()

lora_config = LoraConfig(
    r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(merged_model, lora_config)

model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="french", task="transcribe")


@dataclass
class WhisperDataCollatorWithPadding:
    processor: Any
    apply_augmentation: bool = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.apply_augmentation:
            freq_mask_param = 15
            n_mels = batch["input_features"].shape[1]
            for i in range(len(batch["input_features"])):
                if random.random() < 0.3:
                    f0 = random.randint(0, n_mels - freq_mask_param)
                    batch["input_features"][i, f0:f0 + freq_mask_param, :] = 0.0

        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


train_data_collator = WhisperDataCollatorWithPadding(processor=processor, apply_augmentation=True)
eval_data_collator = WhisperDataCollatorWithPadding(processor=processor, apply_augmentation=False)

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
        print("[compute_metrics] WARNING: All references in this batch are empty, skipping WER/CER.")
        return {"wer": float("nan"), "cer": float("nan")}

    filtered_pred_str, filtered_label_str = zip(*filtered_pairs)
    wer = 100 * metric_wer.compute(predictions=filtered_pred_str, references=filtered_label_str)
    cer = 100 * metric_cer.compute(predictions=filtered_pred_str, references=filtered_label_str)
    return {"wer": wer, "cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir="/output/checkpoints_bambara_init_dioula",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=30,
    max_steps=400,
    weight_decay=0.01,
    gradient_checkpointing=False,
    fp16=True,
    bf16=False,
    eval_strategy="steps",
    eval_steps=40,
    predict_with_generate=True,
    generation_max_length=40,
    generation_num_beams=1,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_steps=40,
    logging_steps=40,
    report_to="wandb",
    run_name="whisper-large-v2-bambara-dioula-v2",
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_train,
    eval_dataset=processed_dev,
    data_collator=train_data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

is_main_process = trainer.is_world_process_zero()

if is_main_process:
    wandb.init(
        project="wobsongo-whisper-dioula",
        id="whisper-large-v2-bambara-dioula-v2",  
        resume="allow",
        name="whisper-large-v2-bambara-dioula-v2",
        config={
            "architecture": "Whisper-Large-V2",
            "method": "Bambara-init (merged) + fresh LoRA + SpecAugment",
            "base_checkpoint": "sudoping01/bambara-asr-v2 (merged into whisper-large-v2)",
            "dataset": "UVCI-Koumankan4Dyula (official split, loaded from Volume)",
            "num_nodes": os.environ.get("WORLD_SIZE"),
        },
    )

if is_main_process:
    print("Igniting training engine...")
last_checkpoint = get_last_checkpoint("/output/checkpoints_bambara_init_dioula")

if last_checkpoint is not None:
    if is_main_process:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    if is_main_process:
        print("Starting training from scratch (Bambara-initialized)...")
    trainer.train()

if is_main_process:
    print(f"Best checkpoint used: {trainer.state.best_model_checkpoint}")
    print(f"Best eval_wer: {trainer.state.best_metric}") 

if is_main_process:
    print("Saving the final model to Volume...")
    final_output_path = "/output/whisper-large-v2-bambara-init-dioula-final"
    trainer.save_model(final_output_path)
    processor.save_pretrained(final_output_path)
    model_volume = modal.Volume.from_name("finetuned-model")
    model_volume.commit()
    print("Running final test evaluation (Koumankan4Dyula official test split)...")

trainer.data_collator = eval_data_collator
print("processed_test columns:", processed_test.column_names)
print("processed_dev columns:", processed_dev.column_names)
test_metrics = trainer.evaluate(eval_dataset=processed_test, metric_key_prefix="test")

if is_main_process:
    print(f"Final test metrics: {test_metrics}")
    wandb.log(test_metrics)
    print("Done! whisper-large-v2 (bambara-init -> dioula) training complete.")
    wandb.finish()