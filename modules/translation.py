import os
import json
import time
from openai import OpenAI
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass
class TranslationOutput:
    original_text: str
    translated_text: str

class LLMTranslator:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)

    def translate_text(self, text: str, source_lang: str, target_lang: str) -> TranslationOutput:
        """
        Translate text from the source language to the target language using OpenAI Tool Calling.
        """
        if not text or not str(text).strip(): 
            return TranslationOutput(original_text=text, translated_text="")

        tools = [
            {
                "type": "function",
                "function": {
                    "name": "provide_translation",
                    "description": f"Translate the raw {source_lang} transcript to {target_lang}.",
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

        max_retries = 3

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini", 
                    messages=[
                        {"role": "system", "content": f"You are a professional translator. Translate the following {source_lang} text into {target_lang}. Maintain the original meaning accurately."},
                        {"role": "user", "content": text}
                    ],
                    tools=tools,
                    tool_choice={"type": "function", "function": {"name": "provide_translation"}},
                    temperature=0.2 
                )
                
                message = response.choices[0].message
                
                if message.tool_calls:
                    json_output_str = message.tool_calls[0].function.arguments
                    parsed_data = json.loads(json_output_str)
                    
                    final_translation = parsed_data.get("Translated_Text", "")
                    
                    return TranslationOutput(original_text=text, translated_text=final_translation)
                
            except Exception as e:
                error_msg = str(e).lower()
                print(f"[Attempt {attempt}] Error from OpenAI API: {e}")
                
                if "rate_limit" in error_msg or "timeout" in error_msg or "502" in error_msg:
                    time.sleep(5)
                else:
                    break 

        return TranslationOutput(original_text=text, translated_text="")