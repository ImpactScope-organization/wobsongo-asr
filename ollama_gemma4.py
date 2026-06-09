import pandas as pd
import json
import os
import time
from tqdm import tqdm
from dotenv import load_dotenv
from ollama import Client

load_dotenv()

OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")

if not OLLAMA_API_KEY:
    raise ValueError("OLLAMA_API_KEY not found.")

print("connect to gemma4...")
ollama_client = Client(
    host='https://ollama.com',
    headers={'Authorization': f'Bearer {OLLAMA_API_KEY}'}
)
print("Ollama Cloud connection successfully loaded!\n")


def translate_with_gemma_tools(text: str, lang_code: str, max_retries: int = 3):
    """
    A function for translating text using Gemma 4 Tool Calling via Ollama Cloud.
    Equipped with an automatic retry mechanism in case of a timeout.
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
            response = ollama_client.chat(
                model="gemma4:31b",
                messages=[
                    {"role": "system", "content": f"You are an expert linguist specializing. Translate the following {lang_code} text into English accurately using the provided tool. Maintain the original context and meaning."},
                    {"role": "user", "content": f"Raw Transcript: {text}"}
                ],
                tools=tools
            )
            
            message = response.get('message', {})
            tool_calls = message.get('tool_calls', [])
            
            if tool_calls:
                arguments = tool_calls[0].get('function', {}).get('arguments', {})
                english_translation = arguments.get("English_Translation", "")
                return english_translation
            else:
                print(f"\n[Model {attempt}] did not return tool function. Retrying...")
                
        except Exception as e:
            error_msg = str(e)
            print(f"\n[ Model {attempt}] Error from Ollama Cloud API: {error_msg}")
            
            if "504" in error_msg or "503" in error_msg or "timeout" in error_msg.lower():
                print("Server is busy. Waiting 5 seconds...")
                time.sleep(5)
            else:
                time.sleep(2) 

    print("\nAll attempts failed for this line of text. Proceed to the next line.")
    return ""


def process_translation_file(file_path):
    print(f"Opening file: {file_path}")
    
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
        print(f"Failed to read file {file_path}. Details: {e}")
        return

    total_rows = len(df)
    print(f"Successfully loaded {total_rows} rows of data. Starting translation process...\n")
    
    human_eng_list = []
    mms_eng_list = []

    lang_mapping = {
        "mos_Latn": "Mooré",
        "dyu_Latn": "Dioula",
        "fra_Latn": "French",
        "jav_lath":"Javanese"
    }
    
    for idx, row in tqdm(df.iterrows(), total=total_rows, desc="Menerjemahkan Transkrip"):
        raw_lang_code = str(row.get('lang_code', '')).strip().lower()
        current_lang = lang_mapping.get(raw_lang_code, "African language")
        
        raw_human = str(row.get('ori_lang', ''))
        # raw_mms = str(row.get('omnilingual_asr_transcrib', ''))
        
        eng_human = translate_with_gemma_tools(raw_human, lang_code=current_lang)
        
        # eng_mms = translate_with_gemma_tools(raw_mms, lang_code=current_lang)
        
        human_eng_list.append(eng_human)
        # mms_eng_list.append(eng_mms)
        
    df['Human: Gemma English Translation'] = human_eng_list
    # df['Omnilingual: Gemma English Translation'] = mms_eng_list
    
    output_path = "gemma_translation_results_ollama_java.xlsx"
    df.to_excel(output_path, index=False)
    print(f"\nFinished! Translation results saved to: {output_path}")

if __name__ == "__main__":
    file_name = "result_with_score_methode_java2.xlsx" 
    process_translation_file(file_name)