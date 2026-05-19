import os
import pandas as pd
from pydub import AudioSegment
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
import time

input_file = 'data_video.xlsx'
audio_folder = 'video_for_transcrib'
output_file = 'transcrib_omniligual_result.xlsx'

CHUNK_DURATION = 30 * 1000
OVERLAP = 1500
TEMP_CHUNK_DIR = "temp_chunks"

CANDIDATE_LANGS = ["mos_Latn", "fra_Latn", "dyu_Latn"]

os.makedirs(TEMP_CHUNK_DIR, exist_ok=True)


def score_text(text):
    if not text or text.strip() == "":
        return -100

    return len(text)


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
    total_infer_time = 0

    for idx, chunk in enumerate(chunks):
        print(f"   [Chunk {idx}] Processing {chunk}")

        best_text = ""
        best_score = -999

        for lang in CANDIDATE_LANGS:
            try:
                start_infer = time.time()

                result = pipeline.transcribe(
                    [chunk],
                    lang=[lang]
                )

                end_infer = time.time()
                infer_time = end_infer - start_infer
                total_infer_time += infer_time

                text = extract_text(result).strip()
                score = score_text(text)

                print(f"      -> {lang}: score={score}")

                if score > best_score:
                    best_score = score
                    best_text = text

            except Exception as e:
                print(f"      [LANG ERROR {lang}] {e}")

        text = best_text if best_text else "[EMPTY]"

        print(f"   [Chunk {idx}] Selected: {text[:50]}...")

        if idx == 0:
            final_text = text
        else:
            cleaned = remove_duplicate_tail(final_text, text)
            final_text += " " + cleaned

    print(f"\n   Total split time: {split_time:.2f} sec")
    print(f"   Total inference time: {total_infer_time:.2f} sec")

    for chunk in chunks:
        if os.path.exists(chunk):
            os.remove(chunk)

    return final_text.strip()


print("Read excel file...")
df = pd.read_excel(input_file)

transcribe_result = []

print("Load model omniASR_LLM_3B ...")
pipeline = ASRInferencePipeline(
    model_card="omniASR_LLM_3B",
    device="cuda"
)

print("\nStart Process Transcribe...")

for index, row in df.iterrows():
    filename = str(row['filename']).strip()

    file_path = os.path.join(audio_folder, filename)

    if os.path.exists(file_path):
        print(f"-> Process [{index+1}/{len(df)}]: {filename}...")

        try:
            audio = AudioSegment.from_file(file_path)

            if len(audio) > CHUNK_DURATION:
                print("   Audio > 30s, using chunking...")
                transcribed_text = transcribe_with_chunking(
                    pipeline, file_path
                )

            else:
                result = pipeline.transcribe([file_path])
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