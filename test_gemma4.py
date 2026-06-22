import os
import json
import time
import modal
from dataclasses import dataclass

transformers_image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch", "accelerate", "huggingface_hub", "peft")
)

app = modal.App("wobsongo-gemma4-native-json")
gemma_volume = modal.Volume.from_name("gemma4-translate")

@dataclass
class TranslationOutput:
    original_text: str
    translated_text: str

@app.cls(
    image=transformers_image, 
    gpu="A100-80GB",           
    timeout=600,
    env={"HF_HOME": "/cache"},
    volumes={"/cache": gemma_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class LLMTranslator:
    @modal.enter()
    def load_model(self):
        print("Loading Base Gemma 4 12B-it and Tokenizer...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from peft import PeftModel
        
        model_id = "google/gemma-4-12b-it"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        
        base_model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,  
            device_map="auto"
        )

        checkpoint_path = "/cache/gemma-new-dataset-final" 
        print(f"Attaching Fine-Tuned JSON LoRA weights from {checkpoint_path}...")
        
        self.model = PeftModel.from_pretrained(base_model, checkpoint_path)

        self.language_map = {
            "mos_Latn": "Mooré",
            "dyu_Latn": "Dioula",
            "fra_Latn": "French",
            "eng_Latn": "English"
        }

    @modal.method()
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> TranslationOutput:
        """
        Translate text from the source language to the target language using Gemma Native JSON.
        """
        if not text or not str(text).strip(): 
            return TranslationOutput(original_text=text, translated_text="")
        
        human_source_lang = self.language_map.get(source_lang, source_lang)
        human_target_lang = self.language_map.get(target_lang, target_lang)

        instruction = (
            f"You are an expert linguist. Translate the following {human_source_lang} text to {human_target_lang}. "
            f"You must only reply with a valid JSON object containing the key 'Translated_Text'.\n\n"
            f"Text: {text}"
        )
        
        prompt = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n"

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                
                import torch
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=256,
                        temperature=0.1,
                        repetition_penalty=1.1
                    )
                
                generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                output_text = output_text.replace("<end_of_turn>", "").strip()
                
                parsed_data = json.loads(output_text)
                
                final_translation = parsed_data.get("Translated_Text", "")
                
                return TranslationOutput(original_text=text, translated_text=final_translation)
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"[Attempt {attempt}] Error from Gemma API / Parsing: {e}")
                print(f"DEBUG RAW OUTPUT: {output_text if 'output_text' in locals() else 'None'}")
                
                if any(err in error_msg for err in ["timeout", "json", "expecting", "unterminated string", "syntax"]):
                    print("Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    time.sleep(2)

        return TranslationOutput(original_text=text, translated_text="")

@app.local_entrypoint()
def main():
    translator = LLMTranslator()
    
    sample_text = "A yi n wele wa"
    
    print("="*60)
    print(f"Executing translation test for text: '{sample_text}'")
    
    result = translator.translate_text.remote(
        text=sample_text,
        source_lang="dyu_Latn",
        target_lang="eng_Latn"
    )
    
    print(f"Original (Dioula) : {result.original_text}")
    print(f"Result (English)  : {result.translated_text}")
    print("="*60)