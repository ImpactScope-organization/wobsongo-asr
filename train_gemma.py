import os
import modal

hf_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "torch", "torchvision", "torchaudio", 
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("transformers", "datasets", "peft", "accelerate", "wandb")
)

app = modal.App("gemma-dioula-finetune")

dataset_volume = modal.Volume.from_name("dioula-dataset-for-gemma")
model_volume = modal.Volume.from_name("gemma4-translate", create_if_missing=True)

@app.function(
    image=hf_image,
    gpu="A100-80GB", 
    timeout=86400, 
    volumes={
        "/data": dataset_volume, 
        "/gemma_model": model_volume 
    },
    secrets=[
        modal.Secret.from_name("wandb-secret") 
    ]
)
def train_gemma_remote():
    import wandb
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM, 
        AutoTokenizer, 
        TrainingArguments, 
        Trainer
    )
    from transformers.trainer_utils import get_last_checkpoint

    os.environ["HF_HOME"] = "/gemma_model"

    print("Initialize....")

    wandb.init(
        project="wobsongo-gemma-dioula",
        config={
            "architecture": "Gemma-4-12B",
            "method": "Standard Trainer LoRA + bfloat16",
            "split": "80% Train / 20% Test",
            "max_seq_length": 2048,
            "dataset": "oumankan4Dyula"
        }
    )

    max_seq_length = 2048
    model_id = "google/gemma-4-12b-it"
    
    print("Loading Base Model from Cache...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        trust_remote_code=True 
    )

    tokenizer.padding_side = "right"
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True 
    )

    print("Configuring the Standard LoRA Adapter...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading and Split Datasets (80/20)...")
    dataset = load_dataset("json", data_files="/data/gemma_dioula_new_dataset.jsonl", split="train")
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    
    print(f"Data Training: {len(split_dataset['train'])} baris")
    print(f"Data Validasi: {len(split_dataset['test'])} baris")

    print("Processing text into tokens...")
    
    def prepare_dataset(batch):
        tokenized = tokenizer(
            batch["text"], 
            truncation=True, 
            max_length=max_seq_length, 
            padding=False 
        )
        
        response_template = "<start_of_turn>model\n"
        response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)
        
        labels_list = []
        
        for input_ids in tokenized["input_ids"]:
            labels = list(input_ids) 
            
            match_idx = -1
            for i in range(len(input_ids) - len(response_token_ids)):
                if input_ids[i:i+len(response_token_ids)] == response_token_ids:
                    match_idx = i + len(response_token_ids)
                    break
            
            if match_idx != -1:
                labels[:match_idx] = [-100] * match_idx
            else:
                pass 
                
            labels_list.append(labels)
            
        tokenized["labels"] = labels_list
        return tokenized

    processed_dataset = split_dataset.map(
        prepare_dataset, 
        batched=True,
        remove_columns=["text"],
        num_proc=4 
    )

    from transformers import DataCollatorForSeq2Seq
    collator = DataCollatorForSeq2Seq(
        tokenizer, 
        pad_to_multiple_of=8, 
        return_tensors="pt", 
        padding=True
    )

    output_dir = "/gemma_model/checkpoints-new-dioula-dataset"
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=4,
        warmup_steps=20,
        max_steps=5000,
        learning_rate=2e-4,
        bf16=True,
        
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False, 
        
        logging_steps=10,
        optim="adamw_torch", 
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        report_to="wandb",
        run_name="gemma-dioula-new-dataset"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=collator, 
    )

    print("Turning on the Training Machine...")
    last_checkpoint = get_last_checkpoint(output_dir)
    
    if last_checkpoint is not None:
        print(f"Continue training from the last checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Start training from scratch...")
        trainer.train()

    print("Saving the Final LoRA Adapter to Volume gemma4-translate...")
    final_output_path = "/gemma_model/gemma-new-dataset-final"
    model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)
    
    model_volume.commit() 
    
    wandb.finish()
    print("Done!!")

@app.local_entrypoint()
def main():
    train_gemma_remote.remote()