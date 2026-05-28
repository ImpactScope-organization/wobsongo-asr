import modal
import urllib.request
import os

image = modal.Image.debian_slim().pip_install(
    "torch", "torchaudio", "pydub", "omnilingual-asr", "fairseq2", "fastapi[standard]"
)

app = modal.App("wobsongo-asr")
volume_model = modal.Volume.from_name("wobsongo-model-asr", create_if_missing=True)

def progress_tracker(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = (downloaded / total_size) * 100
        if int(percent) % 10 == 0 and int(percent) != getattr(progress_tracker, "last_percent", -1):
            print(f"Downloading... {int(percent)}%")
            progress_tracker.last_percent = int(percent)

@app.function(
    image=image,
    volumes={"/root/.cache/fairseq2": volume_model},
    timeout=3600
)
def setup_model():
    os.environ["FAIRSEQ2_CACHE_DIR"] = "/root/.cache/fairseq2"
    os.environ["XDG_CACHE_HOME"] = "/root/.cache"
    
    base_dir = "/root/.cache/fairseq2/assets/01fd052e87486e6e4d742fdf"
    tok_dir = "/root/.cache/fairseq2/assets/b86047ffa9089216c2972a21"
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(tok_dir, exist_ok=True)

    model_path = os.path.join(base_dir, "omniASR-LLM-3B.pt")
    tok_path = os.path.join(tok_dir, "omniASR_tokenizer.model")

    print("Start download model...")
    
    if not os.path.exists(model_path):
        print("Start download omnilingual asr 3B...")
        urllib.request.urlretrieve(
            "https://dl.fbaipublicfiles.com/mms/omniASR-LLM-3B.pt", 
            model_path,
            reporthook=progress_tracker
        )
        print("Model successfully downloaded")
    else:
        print("Model already exists in Volume. Skipping download.")
    
    if not os.path.exists(tok_path):
        print("Start downloading Tokenizer...")
        urllib.request.urlretrieve("https://dl.fbaipublicfiles.com/mms/omniASR_tokenizer.model", tok_path)
        print("Tokenizer downloaded successfully!")

    # Save to volume
    print("Save to volume...")
    volume_model.commit()
    print("Done")