import os
import modal
import json

transformers_image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install("transformers", "torch", "accelerate", "peft")
)

app = modal.App("gemma-translator-service")
model_volume = modal.Volume.from_name("gemma4-translate")

@app.cls(
    image=transformers_image,
    gpu="A100-80GB",
    timeout=600,
    scaledown_window=300,
    volumes={"/gemma_model": model_volume},
)
class GemmaTranslator:
    @modal.enter()
    def load_model(self):
        os.environ["HF_HOME"] = "/gemma_model"
        os.environ["HF_HUB_OFFLINE"] = "1" 

        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel

        print("Loading Base Gemma 4 12B-it from Local Volume Cache...")
        model_id = "google/gemma-4-12b-it"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            local_files_only=True 
        )

        checkpoint_path = "/gemma_model/gemma-new-dataset-final"
        print(f"Attaching Fine-Tuned JSON LoRA from {checkpoint_path}...")

        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)

        self.language_map = {
            "mos_Latn": "Mooré",
            "dyu_Latn": "Dioula",
            "fra_Latn": "French",
            "eng_Latn": "English",
            "auto": "Dioula", 
            "dioula": "Dioula",
            "french": "French",
            "english": "English",
            "moore": "Mooré"
        }

    @modal.method()
    def translate(self, text: str, source_lang: str, target_lang: str) -> str:
        if not text or not str(text).strip():
            return ""

        human_source_lang = self.language_map.get(source_lang.lower(), source_lang)
        human_target_lang = self.language_map.get(target_lang.lower(), target_lang)

        instruction = (
            f"You are an expert linguist. Translate the following {human_source_lang} text to {human_target_lang}. "
            f"You must only reply with a valid JSON object containing the key 'Translated_Text'.\n\n"
            f"Text: {text}"
        )

        prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

        try:
            import torch
            inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    temperature=0.1, 
                    repetition_penalty=1.1
                )

            generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
            output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            start_idx = output_text.find('{')
            end_idx = output_text.rfind('}')
            
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                clean_json_str = output_text[start_idx:end_idx+1]
                parsed_data = json.loads(clean_json_str)
                return parsed_data.get("Translated_Text", "")
            else:
                print(f"JSON format not found. Raw Output: {output_text}")
                return ""

        except Exception as e:
            print(f"Gemma Inference Error: {e}")
            return ""