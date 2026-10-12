import numpy as np
import soundfile as sf
import os

def generate_tone(freq, duration, sr=22050):
    t = np.linspace(0, duration, int(sr * duration), False)
    tone = np.sin(freq * 2 * np.pi * t)
    return tone

def save_wav(filename, data, sr=22050):
    sf.write(filename, data, sr)

classes = ["Chahargah", "Homayun", "Mahur", "Nava", "Segah", "Shur"]
base_freqs = [261.63, 277.18, 293.66, 311.13, 329.63, 349.23]

for i, c in enumerate(classes):
    os.makedirs(f"Training_Data/{c}", exist_ok=True)
    for j in range(10):
        tone1 = generate_tone(base_freqs[i], 15)
        tone2 = generate_tone(base_freqs[i] * 1.5, 15)
        tone = np.concatenate([tone1, tone2])
        save_wav(f"Training_Data/{c}/{c}_{j}.wav", tone)
