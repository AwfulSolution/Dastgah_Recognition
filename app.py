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

st.markdown("### Model Selection")
model_choice = st.selectbox(
    "Choose model",
    ["SVM", "LR (scikit)", "Ensemble (LR + SVM)", "Compare All"],
    index=0,
)

default_paths = {
    "SVM": "/Users/taha/Code/Dastgah_Classification/runs/exp_svm/model.joblib",
    "LR (scikit)": "/Users/taha/Code/Dastgah_Classification/runs/exp_sklearn_v2/model.joblib",
    "Ensemble (LR + SVM)": "/Users/taha/Code/Dastgah_Classification/runs/exp_ensemble/meta_model.joblib",
}

default_value = default_paths.get(model_choice, default_paths["SVM"])
model_path = st.text_input("Model path", value=default_value)
segment_seconds = st.slider("Segment seconds", 10, 60, 45)
num_segments = st.slider("Segments per track", 1, 12, 10)
trim_silence = st.checkbox("Trim silence", value=True)
trim_db = st.slider("Trim dB", 10, 40, 25)

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

    if model_choice == "Compare All":
        lr_path = default_paths["LR (scikit)"]
        svm_path = default_paths["SVM"]
        ens_dir = os.path.dirname(default_paths["Ensemble (LR + SVM)"])
        ens_lr = os.path.join(ens_dir, "lr_model.joblib")
        ens_svm = os.path.join(ens_dir, "svm_model.joblib")
        ens_meta = os.path.join(ens_dir, "meta_model.joblib")
        missing = [p for p in [lr_path, svm_path, ens_lr, ens_svm, ens_meta] if not os.path.exists(p)]
        if missing:
            st.error("Missing model files: " + ", ".join(missing))
            st.stop()
        lr_model = joblib.load(lr_path)
        svm_model = joblib.load(svm_path)
        ens_lr_model = joblib.load(ens_lr)
        ens_svm_model = joblib.load(ens_svm)
        ens_meta_model = joblib.load(ens_meta)
    elif model_choice == "Ensemble (LR + SVM)":
        base_dir = os.path.dirname(model_path)
        lr_path = os.path.join(base_dir, "lr_model.joblib")
        svm_path = os.path.join(base_dir, "svm_model.joblib")
        meta_path = model_path
        if not (os.path.exists(lr_path) and os.path.exists(svm_path) and os.path.exists(meta_path)):
            st.error("Ensemble files not found. Expecting lr_model.joblib, svm_model.joblib, meta_model.joblib")
            st.stop()
        lr_model = joblib.load(lr_path)
        svm_model = joblib.load(svm_path)
        meta_model = joblib.load(meta_path)
    else:
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
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=trim_db)
            starts = segment_starts(len(audio), cfg.sample_rate, segment_seconds, num_segments)
            seg_len = int(segment_seconds * cfg.sample_rate)
            seg_features = []
            for start in starts:
                segment = audio[start : start + seg_len]
                seg_features.append(extract_features(segment, cfg))
            seg_features = np.vstack(seg_features)
            if model_choice == "Compare All":
                lr_probs = lr_model.predict_proba(seg_features).mean(axis=0)
                svm_probs = svm_model.predict_proba(seg_features).mean(axis=0)
                ens_lr_probs = ens_lr_model.predict_proba(seg_features).mean(axis=0)
                ens_svm_probs = ens_svm_model.predict_proba(seg_features).mean(axis=0)
                meta_input = np.hstack([ens_lr_probs, ens_svm_probs]).reshape(1, -1)
                ens_probs = ens_meta_model.predict_proba(meta_input)[0]
                results.append(
                    {
                        "file": os.path.basename(path),
                        "lr_label": LABELS[int(np.argmax(lr_probs))],
                        "lr_conf": float(np.max(lr_probs)),
                        "svm_label": LABELS[int(np.argmax(svm_probs))],
                        "svm_conf": float(np.max(svm_probs)),
                        "ens_label": LABELS[int(np.argmax(ens_probs))],
                        "ens_conf": float(np.max(ens_probs)),
                    }
                )
                continue
            elif model_choice == "Ensemble (LR + SVM)":
                lr_probs = lr_model.predict_proba(seg_features).mean(axis=0)
                svm_probs = svm_model.predict_proba(seg_features).mean(axis=0)
                meta_input = np.hstack([lr_probs, svm_probs]).reshape(1, -1)
                probs = meta_model.predict_proba(meta_input)[0]
            else:
                probs = model.predict_proba(seg_features).mean(axis=0)
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
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=trim_db)
            starts = segment_starts(len(audio), cfg.sample_rate, segment_seconds, num_segments)
            seg_len = int(segment_seconds * cfg.sample_rate)
            seg_features = []
            for start in starts:
                segment = audio[start : start + seg_len]
                seg_features.append(extract_features(segment, cfg))
            seg_features = np.vstack(seg_features)
            if model_choice == "Compare All":
                lr_probs = lr_model.predict_proba(seg_features).mean(axis=0)
                svm_probs = svm_model.predict_proba(seg_features).mean(axis=0)
                ens_lr_probs = ens_lr_model.predict_proba(seg_features).mean(axis=0)
                ens_svm_probs = ens_svm_model.predict_proba(seg_features).mean(axis=0)
                meta_input = np.hstack([ens_lr_probs, ens_svm_probs]).reshape(1, -1)
                ens_probs = ens_meta_model.predict_proba(meta_input)[0]
                results.append(
                    {
                        "file": uploaded.name,
                        "lr_label": LABELS[int(np.argmax(lr_probs))],
                        "lr_conf": float(np.max(lr_probs)),
                        "svm_label": LABELS[int(np.argmax(svm_probs))],
                        "svm_conf": float(np.max(svm_probs)),
                        "ens_label": LABELS[int(np.argmax(ens_probs))],
                        "ens_conf": float(np.max(ens_probs)),
                    }
                )
                continue
            elif model_choice == "Ensemble (LR + SVM)":
                lr_probs = lr_model.predict_proba(seg_features).mean(axis=0)
                svm_probs = svm_model.predict_proba(seg_features).mean(axis=0)
                meta_input = np.hstack([lr_probs, svm_probs]).reshape(1, -1)
                probs = meta_model.predict_proba(meta_input)[0]
            else:
                probs = model.predict_proba(seg_features).mean(axis=0)
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
    if model_choice == "Compare All":
        st.dataframe(results, use_container_width=True)
    else:
        st.dataframe(
            [{k: v for k, v in row.items() if k != "probs"} for row in results],
            use_container_width=True,
        )

        st.markdown("### Confidence Breakdown")
        for row in results:
            st.write(f"**{row['file']}**")
            st.bar_chart({label: float(prob) for label, prob in zip(LABELS, row["probs"])})
