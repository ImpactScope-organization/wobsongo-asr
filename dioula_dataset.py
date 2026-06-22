import pandas as pd
import json
import logging
from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_koumankan4dyula():
    logging.info("Downloading and processing pure Koumankan4Dyula dataset...")
    try:
        ds = load_dataset("uvci/koumankan4dyula", split="train")
        df = ds.to_pandas()
        
        dfs = []
        
        # Dioula Extract -> French
        if 'fr' in df.columns:
            df_fr = df[['dyu', 'fr']].copy()
            df_fr = df_fr.rename(columns={'dyu': 'dioula', 'fr': 'target_text'})
            df_fr['lang'] = 'fr'
            dfs.append(df_fr)
            
        # Dioula Extract -> English
        if 'en' in df.columns:
            df_en = df[['dyu', 'en']].copy()
            df_en = df_en.rename(columns={'dyu': 'dioula', 'en': 'target_text'})
            df_en['lang'] = 'en'
            dfs.append(df_en)
            
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()
    except Exception as e:
        logging.error(f"Failed to process Koumankan4Dyula: {e}")
        return pd.DataFrame()

def apply_gemma_chat_template(row):
    target_lang = "French" if row['lang'] == "fr" else "English"
    dioula_text = str(row['dioula']).strip()
    target_text = str(row['target_text']).strip()
    
    json_output = json.dumps({"Translated_Text": target_text}, ensure_ascii=False)
    
    instruction = (
        f"You are an expert linguist. Translate the following Dioula text to {target_lang}. "
        f"You must only reply with a valid JSON object containing the key 'Translated_Text'.\n\n"
        f"Text: {dioula_text}"
    )
    
    text = f"<start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{json_output}<end_of_turn>"
    return text

def build_pipeline():
    logging.info("=== FILTER DATASET ===")
    
    master_df = process_koumankan4dyula()
    
    master_df = master_df.dropna(subset=['dioula', 'target_text'])
    master_df = master_df.drop_duplicates(subset=['dioula', 'target_text'])
    
    master_df = master_df[master_df['dioula'].astype(str).str.strip() != '']
    master_df = master_df[~master_df['dioula'].astype(str).str.lower().isin(['nan', 'none', 'null'])]
    
    master_df = master_df.copy().reset_index(drop=True)
    
    logging.info(f"Total rows of clean data ready to be trained: {len(master_df)}")
    
    master_df['text'] = master_df.apply(apply_gemma_chat_template, axis=1)
    
    output_jsonl = "gemma_dioula_8k_function_calling.jsonl"
    output_excel = "gemma_dioula_8k_function_calling.xlsx"
    
    logging.info("Saving dataset to JSONL format...")
    master_df[['text']].to_json(output_jsonl, orient="records", lines=True)
    
    logging.info("Saving dataset to Excel format...")
    master_df[['dioula', 'target_text', 'lang', 'text']].to_excel(
        output_excel, 
        index=False, 
        engine='openpyxl',
        sheet_name='Cleaned_8K_Data'
    )
    
    logging.info("=== DONE ===")
    logging.info(f"File JSONL: {output_jsonl}")
    logging.info(f"File EXCEL: {output_excel}")

if __name__ == "__main__":
    build_pipeline()