import pandas as pd
import json
import os
import time
from tqdm import tqdm
from huggingface_hub import InferenceClient 
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

if not HF_TOKEN:
    raise ValueError("HF_TOKEN environment variable is not set.")

print("Loading Gemma 4 via Hugging Face InferenceClient...")
hf_client = InferenceClient(token=HF_TOKEN, timeout=300) 

print("Client successfully loaded!\n")

# Translation Function
def translate_with_gemma_tools(text: str, lang_code: str, max_retries: int = 3):
    """
    Uses Gemma 4 Tool Calling strictly for Translation.
    Includes a retry mechanism to handle API timeouts.
    """
    if pd.isna(text) or not str(text).strip(): 
        return ""

    tools = [
        {
            "type": "function",
            "function": {
                "name": "translate_to_english",
                "description": f"Translate the raw {lang_code} transcript to English.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "English_Translation": {
                            "type": "string",
                            "description": "The accurate, fluid, and grammatically correct English translation of the provided raw transcript."
                        }
                    },
                    "required": ["English_Translation"]
                }
            }
        }
    ]

    for attempt in range(1, max_retries + 1):
        try:
            response = hf_client.chat_completion(
                model="google/gemma-4-31B-it",
                messages=[
                    {"role": "system", "content": f"You are an expert linguist specializing in West African languages. Translate the following {lang_code} text into English accurately using the provided tool. Maintain the original context and meaning."},
                    {"role": "user", "content": f"Raw Transcript: {text}"}
                ],
                tools=tools,
                tool_choice={"type": "function", "function": {"name": "translate_to_english"}},
                max_tokens=1500,
                temperature=0.2 
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                json_output_str = message.tool_calls[0].function.arguments

                json_output_str = json_output_str.strip()
                if json_output_str.startswith("```json"):
                    json_output_str = json_output_str.replace("```json", "", 1)
                if json_output_str.startswith("```"):
                    json_output_str = json_output_str.replace("```", "", 1)
                if json_output_str.endswith("```"):
                    json_output_str = json_output_str[:json_output_str.rfind("```")]
                
                json_output_str = json_output_str.strip()
                
                parsed_data = json.loads(json_output_str)
                english_translation = parsed_data.get("English_Translation", "")
                
                return english_translation
            else:
                print(f"\n[Attempt {attempt}] Model did not return any tool calls. Retrying...")
                
        except json.JSONDecodeError:
            print(f"\n[Attempt {attempt}] Error: Gemma returned invalid JSON. Retrying...")
        except Exception as e:
            error_msg = str(e)
            print(f"\n[Attempt {attempt}] Error from Hugging Face API: {error_msg}")
            
            if "504" in error_msg or "503" in error_msg:
                print("Gateway timeout hit. Waiting 5 seconds before retrying...")
                time.sleep(5)
            else:
                break 

    print("\nAll retries failed for this text block. Proceeding to next row.")
    return ""


def process_translation_file(file_path):
    print(f"Reading file: {file_path}")
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return

    try:
        if file_path.lower().endswith(('.xlsx', '.xls')):
            df = pd.read_excel(file_path)
        else:
            try:
                df = pd.read_csv(file_path, encoding='utf-8')
            except Exception:
                df = pd.read_csv(file_path, encoding='latin1', sep=None, engine='python')
    except Exception as e:
        print(f"Fatal error reading file {file_path}. Details: {e}")
        return

    total_rows = len(df)
    print(f"Successfully loaded {total_rows} rows of data. Starting Translation Process...\n")
    
    human_eng_list = []
    mms_eng_list = []

    lang_mapping = {
        "mos_Latn": "Mooré",
        "dyu_Latn": "Dioula",
        "fra_Latn": "French",
    }
    
    # Process iteration
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Translating Transcripts"):
        raw_lang_code = str(row.get('lang_code', '')).strip().lower()
        current_lang = lang_mapping.get(raw_lang_code, "African language")
        # Extract raw texts
        raw_human = str(row.get('ori_lang', ''))
        raw_mms = str(row.get('omnilingual_asr_transcrib', ''))
        
        # Translate Human Transcript
        eng_human = translate_with_gemma_tools(raw_human, lang_code=current_lang)
        
        # Translate MMS Transcript
        eng_mms = translate_with_gemma_tools(raw_mms, lang_code=current_lang)
        
        # Append to lists
        human_eng_list.append(eng_human)
        mms_eng_list.append(eng_mms)
        
    # Build final DataFrame
    df['Human: Gemma English Translation'] = human_eng_list
    df['Omnilingual: Gemma English Translation'] = mms_eng_list
    
    # Save output
    output_path = "gemma_translation_results_mos.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nDone! The resulting file is saved as: {output_path}")

if __name__ == "__main__":
    file_name = "result/result_with_score_methode.xlsx" 
    process_translation_file(file_name)