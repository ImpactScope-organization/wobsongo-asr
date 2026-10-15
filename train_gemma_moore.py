"""Async fine-tuning pipeline for Gemma 4 (Moore -> French/English translation).

Two independent Modal functions:
    train_gemma_remote        - LoRA fine-tuning (main GPU).
    run_async_bleu_evaluation - Generation-based SacreBLEU eval (separate GPU),
                                 triggered via .spawn() on each checkpoint save.
"""

import os
import json
import re
import modal

hf_image = (
    modal.Image.from_registry("nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04", add_python="3.10")
    .apt_install("git")
    .run_commands("python -m pip install --upgrade pip")
    .pip_install(
        "torch", "torchvision", "torchaudio",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install("transformers", "datasets", "peft", "accelerate", "wandb", "evaluate", "sacrebleu")
)

app = modal.App("gemma-moore-async-pipeline")

dataset_volume = modal.Volume.from_name("moore-dataset-for-gemma")
model_volume = modal.Volume.from_name("gemma4-translate", create_if_missing=True)

MODEL_ID = "google/gemma-4-12b-it"
WANDB_PROJECT = "wobsongo-gemma-moore"
WANDB_GROUP = "gemma-moore-run-1"
RESPONSE_TEMPLATE = "<start_of_turn>model\n"

JSON_FIELD_PATTERN = re.compile(r'"Translated_Text"\s*:\s*"(.*)"', re.DOTALL)


def extract_translated_text(raw_text: str) -> tuple[str, bool]:
    """Extract the translation string from a `{"Translated_Text": "..."}` payload.

    Args:
        raw_text: Model or dataset output, optionally ending with <end_of_turn>.

    Returns:
        (text, is_valid_json). is_valid_json is False when json.loads failed
        and a regex fallback was used, or when extraction failed entirely
        (raw_text is returned unchanged in that case).
    """
    raw_text = raw_text.replace("<end_of_turn>", "").strip()

    try:
        obj = json.loads(raw_text)
        if isinstance(obj, dict) and "Translated_Text" in obj:
            return obj["Translated_Text"].strip(), True
    except (json.JSONDecodeError, ValueError):
        pass

    match = JSON_FIELD_PATTERN.search(raw_text)
    if match:
        extracted = match.group(1)
        extracted = re.sub(r'"\s*}?\s*$', '', extracted)
        extracted = extracted.replace('\\"', '"').replace("\\\\", "\\")
        return extracted.strip(), False

    return raw_text, False


def split_prompt_and_reference(text_content: str) -> tuple[str | None, str | None]:
    """Split a dataset row into (prompt, reference) at the last response template.

    Args:
        text_content: Full "text" field from the dataset.

    Returns:
        (prompt_part, reference_raw), or (None, None) if the template is missing.
    """
    idx = text_content.rfind(RESPONSE_TEMPLATE)
    if idx == -1:
        return None, None
    prompt_part = text_content[: idx + len(RESPONSE_TEMPLATE)]
    reference_raw = text_content[idx + len(RESPONSE_TEMPLATE):]
    return prompt_part, reference_raw


@app.function(
    image=hf_image,
    gpu="A100-80GB",
    timeout=3600,
    volumes={
        "/data": dataset_volume,
        "/gemma_model": model_volume
    },
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def run_async_bleu_evaluation(checkpoint_dir: str, global_step: int) -> None:
    """Generate translations from a checkpoint and log SacreBLEU to W&B.

    Args:
        checkpoint_dir: Path to the LoRA checkpoint to evaluate.
        global_step: Training step this checkpoint corresponds to.
    """
    import wandb
    import torch
    import evaluate
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    print(f"\n[EVAL] Start evaluation for step: {global_step}...")

    # Same split as training (seed must match) so eval never touches train rows.
    dataset = load_dataset("json", data_files="/data/gemma_moore_dataset.jsonl", split="train")
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)
    val_data = split_dataset["test"]

    NUM_SAMPLES = min(150, len(val_data))
    val_data = val_data.select(range(NUM_SAMPLES))

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "left"  # required for batched generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"[EVAL] Load base model and adapter from {checkpoint_dir}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_dir)
    model.eval()

    prompts, references = [], []
    skipped = 0
    for item in val_data:
        prompt_part, reference_raw = split_prompt_and_reference(item["text"])
        if prompt_part is None:
            skipped += 1
            continue
        ref_text, _ = extract_translated_text(reference_raw)
        prompts.append(prompt_part)
        references.append(ref_text)

    if skipped > 0:
        print(f"[EVAL][WARNING] {skipped} Data skipped because the template was not found.")
    print(f"[EVAL] The sample is ready for evaluation.: {len(prompts)} / {NUM_SAMPLES}")

    print(f"[EVAL] Generating translations for {len(prompts)} data...")
    gen_config = {
        "max_new_tokens": 256,
        "do_sample": False,  # greedy decoding for reproducible BLEU
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }

    predictions_raw = []
    BATCH_SIZE = 8
    with torch.no_grad():
        for i in range(0, len(prompts), BATCH_SIZE):
            batch_prompts = prompts[i:i + BATCH_SIZE]
            inputs = tokenizer(
                batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
            ).to(model.device)
            outputs = model.generate(**inputs, **gen_config)
            input_len = inputs["input_ids"].shape[1]
            for output in outputs:
                generated_tokens = output[input_len:]
                decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
                predictions_raw.append(decoded)

    predictions_clean = []
    valid_json_count = 0
    for i, pred_raw in enumerate(predictions_raw):
        pred_text, is_valid = extract_translated_text(pred_raw)
        predictions_clean.append(pred_text)
        if is_valid:
            valid_json_count += 1

        if i < 3:
            print(f"\n--- Example data number-{i + 1} ---")
            print(f"Raw Prediction  : {pred_raw[:150]}")
            print(f"Extracted    : {pred_text}")
            print(f"Reference    : {references[i]}")

    json_validity_rate = valid_json_count / len(predictions_raw) if predictions_raw else 0.0
    print(f"[EVAL] JSON validity rate: {json_validity_rate:.2%}")

    print("[EVAL] Calculating SacreBLEU Score...")
    metric = evaluate.load("sacrebleu")
    formatted_references = [[ref] for ref in references]

    bleu_result = metric.compute(predictions=predictions_clean, references=formatted_references)
    final_bleu_score = bleu_result["score"]
    print(f"[EVAL] Success! SacreBLEU Step {global_step}: {final_bleu_score:.2f} "
          f"(JSON valid: {json_validity_rate:.2%})")

    wandb.init(
        project=WANDB_PROJECT,
        name=f"eval-step-{global_step}",
        group=WANDB_GROUP,
        job_type="evaluation",
    )
    wandb.log({
        "global_step": global_step,
        "eval/sacrebleu_real_generation": final_bleu_score,
        "eval/json_validity_rate": json_validity_rate,
        "eval/num_samples": len(predictions_clean),
    })
    wandb.finish()


@app.function(
    image=hf_image,
    gpu="A100-80GB",
    timeout=86400,
    volumes={
        "/data": dataset_volume,
        "/gemma_model": model_volume
    },
    secrets=[modal.Secret.from_name("wandb-secret")]
)
def train_gemma_remote() -> None:
    """LoRA fine-tune Gemma 4 on the Moore dataset, spawning async BLEU eval per checkpoint."""
    import wandb
    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model, TaskType
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        Trainer,
        DataCollatorForSeq2Seq,
        TrainerCallback
    )
    from transformers.trainer_utils import get_last_checkpoint

    os.environ["HF_HOME"] = "/gemma_model"
    output_dir = "/gemma_model/checkpoints-moore-dataset"

    print("Initialize Training Run....")

    wandb.init(
        project=WANDB_PROJECT,
        name="gemma-moore-training-core",
        group=WANDB_GROUP,
        config={
            "architecture": "Gemma-4-12B",
            "method": "Standard Trainer LoRA + bfloat16",
            "max_seq_length": 2048,
            "dataset": "moore"
        }
    )

    max_seq_length = 2048

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)

    # seed must match run_async_bleu_evaluation() above.
    dataset = load_dataset("json", data_files="/data/gemma_moore_dataset.jsonl", split="train")
    dataset = dataset.shuffle(seed=42)
    split_dataset = dataset.train_test_split(test_size=0.2, seed=42)

    print(f"Training Data: {len(split_dataset['train'])} rows")
    print(f"Validation Data: {len(split_dataset['test'])} rows")

    def prepare_dataset(batch):
        """Tokenize and mask labels so loss is only computed on the model's answer."""
        tokenized = tokenizer(batch["text"], truncation=True, max_length=max_seq_length, padding=False)
        response_token_ids = tokenizer.encode(RESPONSE_TEMPLATE, add_special_tokens=False)

        labels_list = []
        for input_ids in tokenized["input_ids"]:
            labels = list(input_ids)
            match_idx = -1
            # Search from the end -> matches split_prompt_and_reference()'s rfind().
            for i in range(len(input_ids) - len(response_token_ids), -1, -1):
                if input_ids[i:i + len(response_token_ids)] == response_token_ids:
                    match_idx = i + len(response_token_ids)
                    break
            if match_idx != -1:
                labels[:match_idx] = [-100] * match_idx
            labels_list.append(labels)

        tokenized["labels"] = labels_list
        return tokenized

    processed_dataset = split_dataset.map(
        prepare_dataset, batched=True, remove_columns=["text"], num_proc=4
    )

    collator = DataCollatorForSeq2Seq(tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True)

    class AsyncEvaluationTriggerCallback(TrainerCallback):
        """Spawns run_async_bleu_evaluation on a separate GPU each time a checkpoint saves."""

        def on_save(self, args, state, control, **kwargs):
            current_checkpoint = f"{args.output_dir}/checkpoint-{state.global_step}"

            print(f"\n[SYSTEM] Checkpoint {state.global_step} is saved to the container's local disk.")
            print("[SYSTEM] Sync Volume (Flushing Cache to Shared Cloud)...")
            model_volume.commit()  # required so other containers can read this checkpoint

            print(f"[SYSTEM] Calling Bleu for the step {state.global_step}...")
            run_async_bleu_evaluation.spawn(current_checkpoint, state.global_step)  # non-blocking
            print("[SYSTEM] Successfully called the evaluator. Proceeding to the next step of the training....\n")
            return control

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=16,
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
        run_name="gemma-moore-training-core"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset["train"],
        eval_dataset=processed_dataset["test"],
        data_collator=collator,
        callbacks=[AsyncEvaluationTriggerCallback()]
    )

    print("Turning on the Training Machine...")
    last_checkpoint = get_last_checkpoint(output_dir)
    if last_checkpoint is not None:
        print(f"Continue training from the last checkpoint: {last_checkpoint}")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("Start training from scratch...")
        trainer.train()

    print("Saving the Final LoRA Adapter...")
    final_output_path = "/gemma_model/gemma-moore-final"
    model.save_pretrained(final_output_path)
    tokenizer.save_pretrained(final_output_path)

    model_volume.commit()
    wandb.finish()
    print("Core Training Done!!")


@app.local_entrypoint()
def main():
    train_gemma_remote.remote()