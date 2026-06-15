import os
import json
import time
import modal
from dataclasses import dataclass

transformers_image = (
    modal.Image.debian_slim()
    .pip_install("transformers", "torch", "accelerate", "huggingface_hub")
)

app = modal.App("wobsongo-gemma4-tool-translator")
gemma_volume = modal.Volume.from_name("gemma4-translate")


@dataclass
class TranslationOutput:
    original_text: str
    translated_text: str


@dataclass
class FunctionMock:
    name: str
    arguments: str

@dataclass
class ToolCallMock:
    type: str
    function: FunctionMock

@dataclass
class MessageMock:
    tool_calls: list[ToolCallMock]

@dataclass
class ChoiceMock:
    message: MessageMock

@dataclass
class OpenAIResponseMock:
    choices: list[ChoiceMock]


@app.cls(
    image=transformers_image, 
    gpu="A100",           
    timeout=600,
    env={"HF_HOME": "/cache"},
    volumes={"/cache": gemma_volume},
    secrets=[modal.Secret.from_name("huggingface-secret")]
)
class GemmaLLMTranslator:
    @modal.enter()
    def load_model(self):
        print("Loading Gemma 4 12B-it and Tokenizer from Volume...")
        import torch
        from transformers import AutoTokenizer, AutoModelForCausalLM
        
        model_id = "google/gemma-4-12B-it"
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            dtype=torch.bfloat16,  
            device_map="auto"
        )

        self.language_map = {
            "mos_Latn": "Mooré",
            "dyu_Latn": "Dioula",
            "fra_Latn": "French",
            "eng_Latn": "English"
        }

    @modal.method()
    def translate_text(self, text: str, source_lang: str, target_lang: str) -> TranslationOutput:
        if not text or not str(text).strip(): 
            return TranslationOutput(original_text=text, translated_text="")
        
        human_source_lang = self.language_map.get(source_lang, source_lang)

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "provide_translation",
                    "description": f"Translate the raw {human_source_lang} transcript to {target_lang}.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "Translated_Text": {
                                "type": "string",
                                "description": f"The accurate, fluid, and grammatically correct {target_lang} translation of the provided raw transcript."
                            }
                        },
                        "required": ["Translated_Text"],
                        "additionalProperties": False 
                    }
                }
            }
        ]

        messages = [
            {"role": "system", "content": f"You are an expert linguist specializing in West African languages. Translate the following {human_source_lang} text into {target_lang} accurately using the provided tool. Maintain the original meaning accurately."},
            {"role": "user", "content": text}
        ]

        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False
        )

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                inputs = self.tokenizer(prompt, return_tensors="pt").to("cuda")
                
                import torch
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **inputs,
                        max_new_tokens=512,
                        temperature=0.3,
                        do_sample=True
                    )
                
                generated_ids = generated_ids[0][inputs.input_ids.shape[1]:]
                output_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                
                if "Translated_Text:" in output_text:
                    raw_translation = output_text.split("Translated_Text:")[1].split("}")[0]
                    clean_translation = raw_translation.strip("'\" ")
                    
                    final_json_arguments = json.dumps({"Translated_Text": clean_translation})
                else:
                    start_idx = output_text.find("{")
                    end_idx = output_text.rfind("}") + 1
                    json_str = output_text[start_idx:end_idx] if (start_idx != -1 and end_idx != 0) else output_text
                    
                    json.loads(json_str)
                    final_json_arguments = json_str

                response = OpenAIResponseMock(
                    choices=[
                        ChoiceMock(
                            message=MessageMock(
                                tool_calls=[
                                    ToolCallMock(
                                        type="function",
                                        function=FunctionMock(
                                            name="provide_translation",
                                            arguments=final_json_arguments
                                        )
                                    )
                                ]
                            )
                        )
                    ]
                )
                
                message = response.choices[0].message
                
                if message.tool_calls:
                    json_output_str = message.tool_calls[0].function.arguments
                    parsed_data = json.loads(json_output_str)
                    
                    final_translation = parsed_data.get("Translated_Text", "")
                    
                    return TranslationOutput(original_text=text, translated_text=final_translation)
                
            except Exception as e:
                print(f"[Attempt {attempt}] Error from Gemma Engine / Parsing: {e}")
                print(f"DEBUG RAW OUTPUT]: {output_text}")
                
                if any(err in str(e).lower() for err in ["timeout", "json", "expecting", "unterminated string", "syntax"]):
                    print("Retrying in 3 seconds...")
                    time.sleep(3)
                else:
                    time.sleep(2)

        return TranslationOutput(original_text=text, translated_text="")


@app.local_entrypoint()
def main():
    translator = GemmaLLMTranslator()
    sample_text_from_whisper = "A bi ji min na"
    
    print("\n" + "="*60)
    print("Executing Gemma 4 Tool Calling Emulation Test...")
    print("="*60)
    
    result = translator.translate_text.remote(
        text=sample_text_from_whisper,
        source_lang="dyu_Latn",
        target_lang="eng_Latn"
    )
    
    print(f"Original (Dioula) : {result.original_text}")
    print(f"Result (English)  : {result.translated_text}")
    print("="*60 + "\n")