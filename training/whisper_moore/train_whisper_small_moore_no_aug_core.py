import os
import re
import modal
import torch
import wandb
import statistics
from dataclasses import dataclass
from typing import Any, Dict, List, Union
from datasets import load_dataset, Audio
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from transformers.trainer_utils import get_last_checkpoint
from accelerate import PartialState
import evaluate

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
print(f"[PROC] whisper-small-moore (full fine-tune, no aug) | LOCAL_RANK={LOCAL_RANK}, "
      f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

MODEL_ID = "openai/whisper-small"
FORCED_LANGUAGE = "french"

TRAIN_MANIFEST = "/data/manifests/train.jsonl"
DEV_MANIFEST = "/data/manifests/dev.jsonl"

processor = WhisperProcessor.from_pretrained(MODEL_ID, language=FORCED_LANGUAGE, task="transcribe")

torch.cuda.set_device(LOCAL_RANK)


def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    batch["labels"] = processor.tokenizer(batch["text"]).input_ids
    return batch


with PartialState().local_main_process_first():
    print("Loading manifests from Modal Volume...")
    dataset_train = load_dataset("json", data_files=TRAIN_MANIFEST, split="train")
    dataset_dev = load_dataset("json", data_files=DEV_MANIFEST, split="train")

    dataset_train = dataset_train.cast_column("audio_path", Audio(sampling_rate=16000))
    dataset_dev = dataset_dev.cast_column("audio_path", Audio(sampling_rate=16000))
    dataset_train = dataset_train.rename_column("audio_path", "audio")
    dataset_dev = dataset_dev.rename_column("audio_path", "audio")

    print(f"Train: {len(dataset_train)} clips | Dev: {len(dataset_dev)} clips")

    print("Processing audio into spectrogram features...")
    processed_train = dataset_train.map(
        prepare_dataset, remove_columns=dataset_train.column_names, num_proc=4
    )
    processed_dev = dataset_dev.map(
        prepare_dataset, remove_columns=dataset_dev.column_names, num_proc=4
    )

print(f"Loading {MODEL_ID} for full fine-tuning ...")
model = WhisperForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=FORCED_LANGUAGE, task="transcribe")
model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=FORCED_LANGUAGE, task="transcribe")

if LOCAL_RANK == 0:
    print(f"[DECODER PROMPT CHECK] forced_decoder_ids={model.generation_config.forced_decoder_ids}")


@dataclass
class WhisperDataCollatorWithPadding:
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


data_collator = WhisperDataCollatorWithPadding(processor=processor)

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

    pairs = [(p, r) for p, r in zip(pred_str, label_str) if r.strip() != ""]
    if not pairs:
        print("[compute_metrics] WARNING: All references in this batch are empty, skipping WER/CER.")
        return {"wer": float("nan"), "cer": float("nan")}

    filtered_pred, filtered_label = zip(*pairs)
    wer = 100 * metric_wer.compute(predictions=filtered_pred, references=filtered_label)
    cer = 100 * metric_cer.compute(predictions=filtered_pred, references=filtered_label)
    per_utt_wer = [100 * metric_wer.compute(predictions=[p], references=[r])
                    for p, r in zip(filtered_pred, filtered_label)]
    return {
        "wer": wer,
        "cer": cer,
        "median_utterance_wer": statistics.median(per_utt_wer),
    }

training_args = Seq2SeqTrainingArguments(
    output_dir="/output/checkpoints_whisper_small_moore",
    per_device_train_batch_size=32,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=1,
    learning_rate=1e-5,
    warmup_steps=100,
    num_train_epochs=3,
    weight_decay=0.01,
    gradient_checkpointing=False,
    bf16=True,
    fp16=False,
    eval_strategy="steps",
    eval_steps=200,
    predict_with_generate=True,
    generation_max_length=225,
    generation_num_beams=1,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_steps=200,
    save_total_limit=20,
    logging_steps=20,
    report_to="wandb",
    run_name="whisper-small-moore-fullfinetune",
    ddp_find_unused_parameters=False,
    remove_unused_columns=False,
    label_names=["labels"],
)

trainer = Seq2SeqTrainer(
    args=training_args,
    model=model,
    train_dataset=processed_train,
    eval_dataset=processed_dev,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    processing_class=processor.feature_extractor,
)

is_main_process = trainer.is_world_process_zero()

if is_main_process:
    wandb.init(
        project="wobsongo-whisper-moore",
        id="whisper-small-moore-full-finetune",
        resume="allow",
        name="whisper-small-moore-full-finetune",
        config={
            "architecture": "Whisper-Small",
            "method": "Full fine-tuning (no LoRA, no augmentation)",
            "dataset": "Bible + GoAI-Proverbs + GoAI-Contes (warm-start, Tahap 1)",
            "train_hours": 65.08,
            "dev_hours": 0.98,
            "forced_language_token": FORCED_LANGUAGE,
            "num_nodes": os.environ.get("WORLD_SIZE"),
        },
    )

if is_main_process:
    print("Igniting training engine...")
last_checkpoint = get_last_checkpoint("/output/checkpoints_whisper_small_moore")

if last_checkpoint is not None:
    if is_main_process:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    if is_main_process:
        print("Starting training from scratch...")
    trainer.train()

if is_main_process:
    print(f"Best checkpoint used: {trainer.state.best_model_checkpoint}")
    print(f"Best eval_wer: {trainer.state.best_metric}")

    print("Saving the final model to Volume...")
    final_output_path = "/output/whisper-small-moore-stage1-final"
    trainer.save_model(final_output_path)
    processor.save_pretrained(final_output_path)
    model_volume = modal.Volume.from_name("moore-finetuned-model")
    model_volume.commit()
    print("Done! whisper-small-moore (full fine-tune) training complete.")
    wandb.finish()