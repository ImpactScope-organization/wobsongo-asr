import torch
import time
import numpy as np
from scipy.io.wavfile import write
from scipy.signal import butter, lfilter
from speechbrain.inference.enhancement import SpectralMaskEnhancement


# =========================
# High-pass filter function
# =========================
def highpass_filter(data, sr, cutoff=100):
    """
    Mengurangi frekuensi rendah (bass / musik)
    cutoff: semakin besar → semakin banyak bass hilang
    """
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(1, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, data)


# =========================
# Main processing function
# =========================
def process_audio(input_path, output_path):
    print("Loading model SpeechBrain...")

    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="pretrained_models/metricgan-plus-voicebank",
        run_opts={"device": "cpu"}
    )

    print(f"\nProcessing audio: {input_path}")
    start_time = time.time()

    # Enhance audio (hilangkan noise dasar)
    enhanced = enhancer.enhance_file(input_path)

    # Pindah ke CPU
    enhanced = enhanced.cpu()

    # Pastikan bentuknya benar
    if enhanced.dim() == 1:
        enhanced = enhanced.unsqueeze(0)

    # Convert ke numpy
    audio_np = enhanced.squeeze().numpy()

    # =========================
    # HIGH PASS FILTER DI SINI
    # =========================
    audio_np = highpass_filter(audio_np, 16000, cutoff=120)

    # Normalisasi biar tidak clipping
    if np.max(np.abs(audio_np)) > 0:
        audio_np = audio_np / np.max(np.abs(audio_np))

    # Convert ke int16 (format wav standar)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    # Save TANPA torchaudio (biar aman dari error)
    write(output_path, 16000, audio_int16)

    end_time = time.time()
    print(f"Selesai dalam {end_time - start_time:.2f} detik!")
    print(f"Hasil disimpan di: {output_path}")


# =========================
# RUN
# =========================
if __name__ == "__main__":
    FILE_INPUT = "test_tiktok.wav"
    FILE_OUTPUT = "hasil_vokal_bersih.wav"

    process_audio(FILE_INPUT, FILE_OUTPUT)