import os
import modal
import torch
import wandb
import random
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
from peft import LoraConfig, get_peft_model
from accelerate import PartialState
import evaluate

LOCAL_RANK = int(os.environ.get("LOCAL_RANK", 0))
print(f"[PROC] whisper-large (aug, koumankan-split) | LOCAL_RANK={LOCAL_RANK}, "
      f"RANK={os.environ.get('RANK')}, WORLD_SIZE={os.environ.get('WORLD_SIZE')}")

model_id = "openai/whisper-large-v3"
processor = WhisperProcessor.from_pretrained(model_id, language="french", task="transcribe")

torch.cuda.set_device(LOCAL_RANK)

def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_features"] = processor.feature_extractor(
        audio["array"], sampling_rate=audio["sampling_rate"]
    ).input_features[0]
    target_text = batch.get("dyu") or batch.get("sentence") or batch.get("transcription") or batch.get("text") or ""
    batch["labels"] = processor.tokenizer(target_text).input_ids
    return batch


with PartialState().local_main_process_first():
    print("Loading Koumankan4Dyula using its OFFICIAL published splits...")
    dataset_train = load_dataset("uvci/koumankan4dyula", split="train")
    dataset_dev = load_dataset("uvci/koumankan4dyula", split="validation")
    dataset_test = load_dataset("uvci/koumankan4dyula", split="test")

    dataset_train = dataset_train.cast_column("audio", Audio(sampling_rate=16000))
    dataset_dev = dataset_dev.cast_column("audio", Audio(sampling_rate=16000))
    dataset_test = dataset_test.cast_column("audio", Audio(sampling_rate=16000))

    print(f"Training Data   : {len(dataset_train)} rows (~8h)")
    print(f"Dev/Validation  : {len(dataset_dev)} rows (~1h36m)")
    print(f"Test (held-out) : {len(dataset_test)} rows (~45m)")

    # TODO: Challenge probe
    # dataset_challenge = load_dataset(...)
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

model = WhisperForConditionalGeneration.from_pretrained(model_id, torch_dtype=torch.float16)
model.resize_token_embeddings(len(processor.tokenizer))
model.enable_input_require_grads()

lora_config = LoraConfig(
    r=32, lora_alpha=64, target_modules=["q_proj", "v_proj"], lora_dropout=0.05, bias="none"
)
model = get_peft_model(model, lora_config)

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
        batch["input_features"] = batch["input_features"].to(torch.float16)
        return batch


train_data_collator = WhisperDataCollatorWithPadding(processor=processor, apply_augmentation=True)
eval_data_collator = WhisperDataCollatorWithPadding(processor=processor, apply_augmentation=False)

metric_wer = evaluate.load("wer")
metric_cer = evaluate.load("cer")


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids
    label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
    pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
    wer = 100 * metric_wer.compute(predictions=pred_str, references=label_str)
    cer = 100 * metric_cer.compute(predictions=pred_str, references=label_str)
    return {"wer": wer, "cer": cer}


training_args = Seq2SeqTrainingArguments(
    output_dir="/output/checkpoints_large_koumankan_split_aug",
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=1e-4,
    warmup_steps=62,
    max_steps=624,
    gradient_checkpointing=False,
    fp16=True,
    bf16=False,
    eval_strategy="steps",
    eval_steps=52,
    predict_with_generate=True,
    generation_max_length=40,
    generation_num_beams=1,
    metric_for_best_model="wer",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_steps=52,
    logging_steps=52,
    report_to="wandb",
    run_name="whisper-large-v3-koumankan-split-aug",
    ddp_find_unused_parameters=False,
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
trainer.data_collator = eval_data_collator if False else trainer.data_collator

is_main_process = trainer.is_world_process_zero()

if is_main_process:
    wandb.init(
        project="wobsongo-whisper-dioula",
        resume="allow",
        name="whisper-large-v3-koumankan-split-aug",
        config={
            "architecture": "Whisper-Large-V3",
            "method": "LoRA + SpecAugment + Single-node DDP",
            "dataset": "UVCI-Koumankan4Dyula (official split: train 8065 / val 1471 / test 1393)",
            "num_nodes": os.environ.get("WORLD_SIZE"),
        },
    )

if is_main_process:
    print("Igniting training engine...")
last_checkpoint = get_last_checkpoint("/output/checkpoints_large_koumankan_split_aug")

if last_checkpoint is not None:
    if is_main_process:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
    trainer.train(resume_from_checkpoint=last_checkpoint)
else:
    if is_main_process:
        print("Starting training from scratch...")
    trainer.train()

if is_main_process:
    print("Saving the final model to Volume...")
    final_output_path = "/output/whisper-large-v3-koumankan-aug-final"
    trainer.save_model(final_output_path)
    processor.save_pretrained(final_output_path)
    model_volume = modal.Volume.from_name("finetuned-model")
    model_volume.commit()

    print("Running final held-out TEST evaluation (Koumankan4Dyula official test split)...")
    trainer.data_collator = eval_data_collator 
    test_metrics = trainer.evaluate(eval_dataset=processed_test, metric_key_prefix="test")
    print(f"Final TEST metrics: {test_metrics}")
    wandb.log(test_metrics)

    # TODO: challenge probe evaluation
    # metric_key_prefix="challenge"))

    print("Done! whisper-large (aug, koumankan official split) training complete.")
    wandb.finish()