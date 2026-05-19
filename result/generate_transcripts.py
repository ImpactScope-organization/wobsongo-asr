import os
import torch
import librosa
import pandas as pd
from tqdm import tqdm
from transformers import Wav2Vec2ForCTC, AutoProcessor, AutoModelForSeq2SeqLM, AutoTokenizer

# Setup and Load Model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using devices: {device}")

print("Loading Meta MMS (ASR) model...")
mms_proc = AutoProcessor.from_pretrained("facebook/mms-1b-all")
mms_model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all").to(device)

print("Loading NLLB (Translation) models...")
nllb_tok = AutoTokenizer.from_pretrained("facebook/nllb-200-distilled-600M")
nllb_mod = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-600M").to(device)
print("All models loaded successfully!\n")

# Translate Function (Text Chunking)
def translate_chunk(text, lang_code):
    if not text.strip(): return ""
    mapping = {"mos": "mos_Latn", "fra": "fra_Latn", "dyu": "dyu_Latn"}
    src_lang = mapping.get(lang_code, "mos_Latn")
    
    nllb_tok.src_lang = src_lang
    inputs = nllb_tok(text, return_tensors="pt").to(device)
    
    # NLLB translated into English
    tokens = nllb_mod.generate(
        **inputs,
        forced_bos_token_id=nllb_tok.convert_tokens_to_ids("eng_Latn"),
        max_length=100, 
        repetition_penalty=1.2
    )
    return nllb_tok.batch_decode(tokens, skip_special_tokens=True)[0]

# Long Text Translation Function (text chunking)
def translate_long_text(long_text, lang_code):
    """Break down long text into 15 words"""
    if not long_text.strip(): return ""
    
    words = long_text.split()
    # Cut the text into small groups (15 words per group)
    chunks = [" ".join(words[i:i + 15]) for i in range(0, len(words), 15)]
    
    translated_chunks = []
    for chunk in chunks:
        # Translate per short piece
        eng_chunk = translate_chunk(chunk, lang_code)
        translated_chunks.append(eng_chunk)
        
    # Rejoin into one complete paragraph
    return " ".join(translated_chunks)

# Audio Chunking Function
def process_audio_chunked(audio_path, lang_code):
    """Cut audio by pause, then transcribe & translate per chunk"""
    try:
        # Load full audio
        speech, sr = librosa.load(audio_path, sr=16000)
        
        # Detecting Pauses
        intervals = librosa.effects.split(speech, top_db=35)
        
        mms_proc.tokenizer.set_target_lang(lang_code)
        mms_model.load_adapter(lang_code)
        
        full_transcription = []
        full_translation = []
        
        # Iterate/Loop each audio piece
        for start, end in intervals:
            chunk = speech[start:end]
            
            #  Ignore noise or breathing sounds
            if len(chunk) < 0.5 * sr:
                continue
                
            # Transcripting cuttings (MMS)
            inputs = mms_proc(chunk, sampling_rate=sr, return_tensors="pt").to(device)
            with torch.no_grad():
                logits = mms_model(**inputs).logits
            ids = torch.argmax(logits, dim=-1)[0]
            chunk_text = mms_proc.decode(ids).strip()
            
            if not chunk_text: 
                continue
                
            # Translating Cut (NLLB)
            chunk_eng = translate_chunk(chunk_text, lang_code)
            
            # Save the cut results into a list
            full_transcription.append(chunk_text)
            full_translation.append(chunk_eng)
            
        # Recombin the pieces into paragraphs
        final_transcription = " ".join(full_transcription)
        final_translation = " ".join(full_translation)
        
        return final_transcription, final_translation
        
    except Exception as e:
        print(f"\nError processing file {audio_path}: {e}")
        return "", ""

# Batch process function for all files
def run_batch_processing(csv_file_path, audio_folder_path):
    print(f"Reading raw CSV files: {csv_file_path}")
    df = pd.read_csv(csv_file_path)
    
    results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing Videos"):
        filename = str(row['filename'])
        lang_code = str(row['lang_code'])
        
        # Search for audio files within a folder
        audio_path = os.path.join(audio_folder_path, filename)
        
        if not os.path.exists(audio_path):
            print(f"\nSkipping {filename}: Audio file not found in {audio_path}")
            continue
            
        # Execute chunking function
        transcription, eng_machine = process_audio_chunked(audio_path, lang_code)
        
        # Translate human local moore 
        local_moore = str(row.get('ori_lang', ''))
        eng_human = translate_long_text(local_moore, lang_code) if local_moore else ""
        
        # Simpan ke daftar hasil
        results.append({
            "filename": filename,
            "lang_code": lang_code,
            "ori_lang": local_moore,
            "Automated audio transcription with Meta MMS": transcription,
            "Automated transcription to English": eng_machine,
            "Original language to English": eng_human
        })
        
    # Create excel file to save the result
    result_df = pd.DataFrame(results)
    output_excel = "chunking_result.xlsx"
    result_df.to_excel(output_excel, index=False)
    print(f"\nDone! The file has been successfully saved as: {output_excel}")

if __name__ == "__main__":
    FILE_NAME = "data_video.csv"
    FOLDER_AUDIO_NAME = "./" 
    
    run_batch_processing(FILE_NAME, FOLDER_AUDIO_NAME)