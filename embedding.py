import modal
from typing import List, Dict, Any
from pydantic import BaseModel
import time

bge_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch", 
        "sentence-transformers", 
        "fastapi[standard]", 
        "pydantic"
    )
)

app = modal.App("bge-m3-serve")

bge_volume = modal.Volume.from_name("bge-m3-cache", create_if_missing=True)
CACHE_DIR = "/models_cache/huggingface"

class EmbedRequest(BaseModel):
    texts: List[str]

@app.cls(
    image=bge_image,
    gpu="A10G", 
    volumes={"/models_cache": bge_volume},
    env={"HF_HOME": CACHE_DIR},
    timeout=600,
    scaledown_window=300
)
class BGEM3Model:
    @modal.enter()
    def setup(self):
        print("Initializing BGE-M3 Model...")
        start_load = time.perf_counter()
        
        from sentence_transformers import SentenceTransformer
        
        self.model = SentenceTransformer("BAAI/bge-m3", local_files_only=True)
        
        elapsed = time.perf_counter() - start_load
        print(f"[PROFILING] Model loaded to VRAM in {elapsed:.2f} seconds\n")

    @modal.fastapi_endpoint(method="POST")
    def embed(self, request: EmbedRequest) -> Dict[str, Any]:
        start_exec = time.perf_counter()
        
        if not request.texts:
            return {"error": "Input 'texts' tidak boleh kosong."}

        embeddings = self.model.encode(request.texts).tolist()
        
        elapsed = time.perf_counter() - start_exec
        print(f"[EXEC PROFILING] Generated embeddings for {len(request.texts)} texts in {elapsed:.3f}s")
        
        return {
            "model": "BAAI/bge-m3",
            "embeddings": embeddings,
            "modal_execution_time": round(elapsed, 3)
        }