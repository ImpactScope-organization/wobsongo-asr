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
        "soundfile"
    )
)

app = modal.App("whisper-dioula-test")

dataset_volume = modal.Volume.from_name("dioula-dataset")
model_volume = modal.Volume.from_name("finetuned-model")

@app.function(
    image=whisper_image,
    gpu="T4",
    volumes={
        "/data": dataset_volume,
        "/output": model_volume
    }
)
def test_transcription():
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    from peft import PeftModel
    from datasets import load_dataset, Audio

    print("Loading Processor & Base Model Whisper...")
    model_id = "openai/whisper-large-v3"
    
    processor = WhisperProcessor.from_pretrained(model_id, language="french", task="transcribe")
    base_model = WhisperForConditionalGeneration.from_pretrained(model_id, device_map="auto")

    print("Installing LoRA from Checkpoint 3500...")
    model = PeftModel.from_pretrained(base_model, "/output/checkpoints/checkpoint-3500")

    print("Sampling audio data...")
    dataset = load_dataset("csv", data_files="/data/dioula-dataset/metadata.csv", split="train")
    
    def attach_audio_path(row):
        row["audio_path"] = "/data/dioula-dataset/" + row["file_name"]
        return row
        
    dataset = dataset.map(attach_audio_path)
    
    dataset = dataset.cast_column("audio_path", Audio(sampling_rate=16000))
    
    sample = dataset[2]
    audio = sample["audio_path"]
    
    target_text = sample.get("sentence", "Unknown")

    print("Processing audio into spectrogram...")
    input_features = processor(
        audio["array"], sampling_rate=audio["sampling_rate"], return_tensors="pt"
    ).input_features.to(device="cuda", dtype=model.dtype)

    print("Generating transcription...")
    with torch.no_grad():
        predicted_ids = model.generate(
            input_features,
            language="french",       
            task="transcribe",
            max_new_tokens=128,
            repetition_penalty=1.1
        )
    
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

    print("\n" + "="*60)
    print("Checkpoint 3500 Test Result")
    print("="*60)
    print(f"Audio Path     : {sample.get('file_name', 'Unknown path')}")
    print(f"Original Text  : {target_text}")
    print(f"Model Results  : {transcription}")
    print("="*60)

@app.local_entrypoint()
def main():
    test_transcription.remote()