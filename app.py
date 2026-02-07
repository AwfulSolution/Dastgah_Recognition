import os
import tempfile
from typing import List

import joblib
import numpy as np
import librosa
import streamlit as st

from src.dastgah.data import LABELS
from src.dastgah.features import FeatureConfig, load_audio, melspec


st.set_page_config(page_title="Dastgah Classifier", layout="centered")


def segment_starts(
    audio_len: int,
    sr: int,
    seconds: float,
    num_segments: int,
) -> List[int]:
    seg_len = int(seconds * sr)
    if audio_len <= seg_len:
        return [0]

    max_start = audio_len - seg_len
    if num_segments <= 1:
        return [max_start // 2]

    return np.linspace(0, max_start, num_segments).astype(int).tolist()


def extract_features(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mel = melspec(audio, cfg)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)

    mfcc_feat = librosa.feature.mfcc(
        y=audio,
        sr=cfg.sample_rate,
        n_mfcc=20,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mfcc_delta = librosa.feature.delta(mfcc_feat)

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    harm = librosa.effects.harmonic(y=audio)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=cfg.sample_rate)

    def stats(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=0)

    return np.concatenate(
        [
            mel_mean,
            mel_std,
            stats(mfcc_feat),
            stats(mfcc_delta),
            stats(chroma),
            stats(contrast),
            stats(tonnetz),
        ],
        axis=0,
    ).astype(np.float32)


def list_mp3s(folder: str) -> List[str]:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".mp3")
    )


st.title("Dastgah Classifier")

model_path = st.text_input(
    "Model path",
    value="/Users/taha/Code/Dastgah_Classification/runs/exp_sklearn_v2/model.joblib",
)
segment_seconds = st.slider("Segment seconds", 10, 60, 30)
num_segments = st.slider("Segments per track", 1, 12, 6)

st.markdown("### Input")
input_mode = st.radio("Choose input", ["Upload files", "Folder path"], horizontal=True)

files = []
folder_path = None

if input_mode == "Upload files":
    uploads = st.file_uploader("Upload MP3 files", type=["mp3"], accept_multiple_files=True)
    files = uploads or []
else:
    folder_path = st.text_input("Folder path with MP3s")

if st.button("Classify"):
    if not os.path.exists(model_path):
        st.error("Model not found. Check the model path.")
        st.stop()

    model = joblib.load(model_path)
    cfg = FeatureConfig()

    results = []

    if input_mode == "Folder path":
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Folder path is invalid.")
            st.stop()
        file_paths = list_mp3s(folder_path)
        if not file_paths:
            st.warning("No MP3 files found in folder.")
            st.stop()
        for path in file_paths:
            audio = load_audio(path, cfg)
            starts = segment_starts(len(audio), cfg.sample_rate, segment_seconds, num_segments)
            seg_len = int(segment_seconds * cfg.sample_rate)
            seg_features = []
            for start in starts:
                segment = audio[start : start + seg_len]
                seg_features.append(extract_features(segment, cfg))
            probs = model.predict_proba(np.vstack(seg_features)).mean(axis=0)
            top_idx = int(np.argmax(probs))
            results.append(
                {
                    "file": os.path.basename(path),
                    "label": LABELS[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probs": probs,
                }
            )
    else:
        if not files:
            st.warning("Please upload at least one MP3.")
            st.stop()
        for uploaded in files:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                tmp_path = tmp.name
            audio = load_audio(tmp_path, cfg)
            starts = segment_starts(len(audio), cfg.sample_rate, segment_seconds, num_segments)
            seg_len = int(segment_seconds * cfg.sample_rate)
            seg_features = []
            for start in starts:
                segment = audio[start : start + seg_len]
                seg_features.append(extract_features(segment, cfg))
            probs = model.predict_proba(np.vstack(seg_features)).mean(axis=0)
            top_idx = int(np.argmax(probs))
            results.append(
                {
                    "file": uploaded.name,
                    "label": LABELS[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probs": probs,
                }
            )
            os.unlink(tmp_path)

    st.markdown("### Results")
    st.dataframe(
        [{k: v for k, v in row.items() if k != "probs"} for row in results],
        use_container_width=True,
    )

    st.markdown("### Confidence Breakdown")
    for row in results:
        st.write(f"**{row['file']}**")
        st.bar_chart({label: float(prob) for label, prob in zip(LABELS, row["probs"])})
