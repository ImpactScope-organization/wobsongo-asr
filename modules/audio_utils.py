import os
from pathlib import Path
import tempfile
from pydub import AudioSegment

def get_audio_duration_ms(file_path: str) -> int:
    """Reads the total duration of an audio file in milliseconds (ms)."""
    audio = AudioSegment.from_file(file_path)
    return len(audio)

def split_audio(file_path: str, chunk_duration_ms: int = 30000, overlap_ms: int = 1500) -> list[str]:
    """
    Splitting audio into parts using pydub. Returns list of paths to the chunks.
    """
    tmp_dir = tempfile.mkdtemp(prefix="asr_chunks_")
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)

    chunks, start, i = [], 0, 0
    base = os.path.basename(file_path)
    while start < len(audio):
        chunk = audio[start : start + chunk_duration_ms]
        path = os.path.join(tmp_dir, f"{base}_{i}.wav")
        chunk.export(path, format="wav")
        chunks.append(path)
        start += chunk_duration_ms - overlap_ms
        i += 1
    return chunks

def normalize_audio(file_path: str) -> str:
    tmp_dir = tempfile.mkdtemp(prefix="asr_normalized_")
    audio = AudioSegment.from_file(file_path)
    audio = audio.set_frame_rate(16000).set_channels(1)
 
    base = os.path.basename(file_path)
    out_path = os.path.join(tmp_dir, f"{base}_normalized.wav")
    audio.export(out_path, format="wav")
    return out_path

def compress_audio_to_ogg_bytes(file_path: str | Path) -> bytes:
    """Compresses audio to 16kHz Mono OGG and returns the byte data."""
    audio_seg = AudioSegment.from_file(file_path)
    audio_seg = audio_seg.set_frame_rate(16000).set_channels(1)
    
    with tempfile.NamedTemporaryFile(suffix=".ogg", delete=False) as tmp_audio:
        # Kualitas rendah-menengah (-q:a 0), sangat kecil tapi cukup untuk ASR
        audio_seg.export(tmp_audio.name, format="ogg", parameters=["-q:a", "0"]) 
        
        with open(tmp_audio.name, "rb") as f:
            audio_bytes = f.read()
            
    os.remove(tmp_audio.name) 
    return audio_bytes