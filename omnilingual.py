import os
import pandas as pd
from pydub import AudioSegment
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

input_file = 'data_video.xlsx'
audio_folder = 'video_for_transcrib'
output_file = 'transcrib_omniligual_result.xlsx'

CHUNK_DURATION = 30 * 1000  
OVERLAP = 1500   
TEMP_CHUNK_DIR = "temp_chunks"

os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)

def split_audio(file_path):
    start_time = time.time()

    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    chunks = []
    start = 0
    i = 0

    while start < len(audio):
        end = start + CHUNK_DURATION
        chunk = audio[start:end]

        chunk_name = os.path.join(
            TEMP_CHUNK_DIR,
            f"{os.path.basename(file_path)}_{i}.wav"
        )

        chunk.export(chunk_name, format="wav")
        chunks.append(chunk_name)

        start += (CHUNK_DURATION - OVERLAP)
        i += 1

    end_time = time.time()
    split_time = end_time - start_time

    print(f"   Total chunks created: {len(chunks)}")
    print(f"   Split time (pydub): {split_time:.2f} sec")

    return chunks, split_time


def remove_duplicate_tail(prev_text, curr_text):
    prev_words = prev_text.split()
    curr_words = curr_text.split()

    max_overlap = min(len(prev_words), len(curr_words), 10)

    for i in range(max_overlap, 0, -1):
        if prev_words[-i:] == curr_words[:i]:
            return " ".join(curr_words[i:])

    return curr_text


def extract_text(result):
    if isinstance(result, list) and len(result) > 0:
        first = result[0]

        if isinstance(first, dict):
            return first.get("text", "")

        elif isinstance(first, str):
            return first

        else:
            return str(first)

    elif isinstance(result, str):
        return result

    return str(result)


def transcribe_with_chunking(pipeline, file_path):
    chunks, split_time = split_audio(file_path)
    final_text = ""

    for idx, chunk in enumerate(chunks):
        try:
            result = pipeline.transcribe(
                [chunk],
                lang=None  
            )

            text = extract_text(result).strip()

            if not text:
                text = "[EMPTY]"

            if idx == 0:
                final_text = text
            else:
                cleaned = remove_duplicate_tail(final_text, text)
                final_text += " " + cleaned

        except Exception as e:
            print(f"      [CHUNK ERROR] {chunk}: {e}")

    for chunk in chunks:
        if os.path.exists(chunk):
            os.remove(chunk)

    return final_text.strip()


print("Read excel file...")
df = pd.read_excel(input_file)

transcribe_result = []

print("Load model omniASR_LLM_3B ...")
try:
    pipeline = ASRInferencePipeline(
        model_card="omniASR_LLM_3B",
        device="cuda"
    )
except Exception as e:
    print(f"Failed load model: {e}")
    exit()

print("\nStart Process Transcribe...")

for index, row in df.iterrows():
    filename = str(row['filename']).strip()
    lang_code = str(row['lang_code']).strip() 

    file_path = os.path.join(audio_folder, filename)

    if os.path.exists(file_path):
        print(f"-> Process [{index+1}/{len(df)}]: {filename} (Language: {lang_code})...")

        try:
            audio = AudioSegment.from_file(file_path)

            if len(audio) > CHUNK_DURATION:
                print("   Audio > 30s, using chunking...")
                transcribed_text = transcribe_with_chunking(
                    pipeline, file_path  
                )
            else:
                result = pipeline.transcribe(
                    [file_path],
                    lang=None
                )

                transcribed_text = extract_text(result).strip()

                if not transcribed_text:
                    transcribed_text = "[EMPTY]"

            print(f"   Result: {transcribed_text[:60]}...")

        except Exception as e:
            print(f"   [ERROR] Failed to process {filename}: {e}")
            transcribed_text = "ERROR"

    else:
        print(f"   [NOT FOUND] File {filename} not found")
        transcribed_text = "FILE_NOT_FOUND"

    transcribe_result.append(transcribed_text)

print("\nSave result to Excel file...")
df['omnilingual_asr_transcrib'] = transcribe_result
df.to_excel(output_file, index=False)

print(f"Done! The results are saved in {output_file}")