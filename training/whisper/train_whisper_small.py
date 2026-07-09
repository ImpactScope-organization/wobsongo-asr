import os
import modal

whisper_image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install(
        "torch", 
        "torchcodec",
        "transformers", 
        "datasets", 
        "peft", 
        "librosa", 
        "soundfile", 
        "wandb",
        "evaluate",
        "jiwer"
    )
)

app = modal.App("whisper-small-dioula-finetune")

dataset_volume = modal.Volume.from_name("dioula-dataset")
model_volume = modal.Volume.from_name("finetuned-model", create_if_missing=True)


@app.function(
    image=whisper_image,
    gpu="A100",
    timeout=86400,
    volumes={
        "/data": dataset_volume, 
        "/output": model_volume
    },
    secrets=[
        modal.Secret.from_name("wandb-secret")
    ]
)
def train_whisper_remote():
    import torch
    import wandb
    import random
    from dataclasses import dataclass
    from typing import Any, Dict, List, Union
    from datasets import load_from_disk, load_dataset, Audio, concatenate_datasets
    from transformers import (
        WhisperForConditionalGeneration, 
        WhisperProcessor, 
        Seq2SeqTrainingArguments, 
        Seq2SeqTrainer
    )
    from transformers.trainer_utils import get_last_checkpoint
    from peft import LoraConfig, get_peft_model
    import evaluate

    print("Initializing on Modal A100 GPU Server...")

    wandb.init(
        project="wobsongo-whisper-dioula",
        resume="allow",
        name="run-small-lora-split-8020",
        config={
            "architecture": "Whisper-Small",
            "method": "LoRA + SpecAugment",
            "dataset": "Mozilla-CV17 + UVCI-Koumankan (80/20 Split)"
        }
    )

    model_id = "openai/whisper-small"
    processor = WhisperProcessor.from_pretrained(model_id, language="french", task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, 
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.resize_token_embeddings(len(processor.tokenizer))
    model.enable_input_require_grads()
    lora_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(model, lora_config)

    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="french", task="transcribe"
    )
    model.generation_config.forced_decoder_ids = processor.get_decoder_prompt_ids(
        language="french", task="transcribe"
    )

    print("Loading datasets from Modal Volume...")
    dataset_mcv = load_dataset("csv", data_files="/data/dioula-dataset/metadata.csv", split="train")
    
    def attach_audio_path(row):
        row["audio_path"] = "/data/dioula-dataset/" + row["file_name"]
        return row
        
    dataset_mcv = dataset_mcv.map(attach_audio_path)
    dataset_mcv = dataset_mcv.cast_column("audio_path", Audio(sampling_rate=16000))
    dataset_mcv = dataset_mcv.rename_column("audio_path", "audio")
    
    dataset_uvci = load_from_disk("/data/uvci_data")
    dataset_uvci = dataset_uvci.cast_column("audio", Audio(sampling_rate=16000))

    combined_dataset = concatenate_datasets([dataset_mcv, dataset_uvci])
    
    print("Splitting dataset into 80% Training and 20% Testing...")
    split_dataset = combined_dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"Training Data : {len(split_dataset['train'])} rows")
    print(f"Testing Data  : {len(split_dataset['test'])} rows")

    @dataclass
    class WhisperDataCollatorWithPadding:
        processor: Any

        def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
            input_features = [{"input_features": feature["input_features"]} for feature in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

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

    def prepare_dataset(batch):
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"], sampling_rate=audio["sampling_rate"]
        ).input_features[0]
        
        target_text = batch.get("dyu") or batch.get("sentence") or batch.get("transcription") or batch.get("text") or ""
        batch["labels"] = processor.tokenizer(target_text).input_ids
        return batch

    print("Processing audio into a spectrogram matrix...")
    processed_dataset = split_dataset.map(
        prepare_dataset, 
        remove_columns=combined_dataset.column_names, 
        num_proc=4 
    )

    data_collator = WhisperDataCollatorWithPadding(processor=processor)

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
        output_dir="/output/checkpoints_small_split8020",
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
        learning_rate=1e-4,
        warmup_steps=500,
        max_steps=5000,
        gradient_checkpointing=True,
        fp16=True,
        bf16=False,
        
        eval_strategy="steps",
        eval_steps=500,
        predict_with_generate=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        load_best_model_at_end=True,
        
        save_steps=500,
        logging_steps=50,
        report_to="wandb",
        run_name="whisper-small-dioula-split8020"
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics, # for WER and CER
        processing_class=processor.feature_extractor,
    )

    print("Igniting training engine on A100 GPU...")
    last_checkpoint = get_last_checkpoint("/output/checkpoints_small_split8020")
    
    if last_checkpoint is not None:
        print(f"Resuming training from checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Starting training from scratch for the new 80/20 split...")
        trainer.train()

    print("Saving the final model to Volume...")
    final_output_path = "/output/whisper-small-dioula-split8020-final"
    trainer.save_model(final_output_path)
    processor.save_pretrained(final_output_path)
    model_volume.commit()
    
    print("Done! Model successfully trained with augmentation and evaluation.")
    wandb.finish()


@app.local_entrypoint()
def main():
    train_whisper_remote.remote()