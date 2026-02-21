import os
import tempfile
from typing import List, Tuple

import joblib
import librosa
import numpy as np
import streamlit as st

from src.dastgah.data import LABELS
from src.dastgah.features import FeatureConfig, load_audio
from src.dastgah.model_config import config_path_for_model, load_model_config
from src.dastgah.scikit_features import ScikitFeatureOpts, extract_features, segment_starts


st.set_page_config(page_title="Dastgah Classifier", layout="centered")
APP_DIR = os.path.dirname(os.path.abspath(__file__))


def list_mp3s(folder: str) -> List[str]:
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".mp3")
    )


def first_existing(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
        app_rel = os.path.join(APP_DIR, path)
        if os.path.exists(app_rel):
            return app_rel
    return candidates[0]


def resolve_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if os.path.exists(path):
        return path
    return os.path.join(APP_DIR, path)


def apply_low_compute(segment_seconds: int, num_segments: int, trim_db: int, mode_pitch_bins: int) -> Tuple[int, int, int, int]:
    if segment_seconds == 45:
        segment_seconds = 25
    if num_segments == 10:
        num_segments = 4
    if trim_db == 25:
        trim_db = 30
    if mode_pitch_bins == 24:
        mode_pitch_bins = 16
    return segment_seconds, num_segments, trim_db, mode_pitch_bins


def build_segment_features(
    audio: np.ndarray,
    cfg: FeatureConfig,
    opts: ScikitFeatureOpts,
    segment_seconds: int,
    num_segments: int,
) -> np.ndarray:
    starts = segment_starts(
        len(audio),
        cfg.sample_rate,
        float(segment_seconds),
        int(num_segments),
        mode="eval",
        seed=42,
    )
    seg_len = int(segment_seconds * cfg.sample_rate)
    seg_features = []
    for start in starts:
        segment = audio[start : start + seg_len]
        seg_features.append(extract_features(segment, cfg, opts))
    return np.vstack(seg_features)


st.title("Dastgah Classifier")

st.markdown("### Model Selection")
model_choice = st.selectbox(
    "Choose model",
    ["SVM", "LR (scikit)", "Ensemble (LR + SVM)", "Compare All (scikit)"],
    index=0,
)

default_paths = {
    "SVM": first_existing(
        [
            "runs/exp_svm_v3/model.joblib",
            "runs/exp_svm_mode_pca/model.joblib",
            "runs/exp_svm/model.joblib",
            "models/model.joblib",
        ]
    ),
    "LR (scikit)": first_existing(
        [
            "runs/exp_sklearn_v3/model.joblib",
            "runs/exp_sklearn_mode_pca/model.joblib",
            "runs/exp_sklearn/model.joblib",
            "runs/exp_sklearn_v2/model.joblib",
        ]
    ),
    "Ensemble (LR + SVM)": first_existing(
        [
            "runs/exp_ensemble_v3/meta_model.joblib",
            "runs/exp_ensemble_mode_pca/meta_model.joblib",
            "runs/exp_ensemble/meta_model.joblib",
        ]
    ),
}

model_path = st.text_input("Model path", value=default_paths.get(model_choice, default_paths["SVM"]))

loaded_cfg = None
resolved_for_cfg = resolve_path(model_path)
cfg_path = config_path_for_model(resolved_for_cfg)
if os.path.exists(cfg_path):
    loaded_cfg = load_model_config(resolved_for_cfg)

segment_default = int(round(float(loaded_cfg.get("segment_seconds", 45)))) if loaded_cfg else 45
num_segments_default = int(loaded_cfg.get("num_segments", 10)) if loaded_cfg else 10
trim_silence_default = bool(loaded_cfg.get("trim_silence", True)) if loaded_cfg else True
trim_db_default = int(loaded_cfg.get("trim_db", 25)) if loaded_cfg else 25
use_mode_default = bool(loaded_cfg.get("use_mode_features", False)) if loaded_cfg else False
mode_bins_default = int(loaded_cfg.get("mode_pitch_bins", 24)) if loaded_cfg else 24

segment_default = max(10, min(60, segment_default))
num_segments_default = max(1, min(12, num_segments_default))
trim_db_default = max(10, min(40, trim_db_default))
mode_bins_default = max(12, min(48, mode_bins_default))
if mode_bins_default % 2 == 1:
    mode_bins_default += 1

if loaded_cfg:
    st.caption(f"Loaded model config: {cfg_path}")

segment_seconds = st.slider("Segment seconds", 10, 60, segment_default)
num_segments = st.slider("Segments per track", 1, 12, num_segments_default)
trim_silence = st.checkbox("Trim silence", value=trim_silence_default)
trim_db = st.slider("Trim dB", 10, 40, trim_db_default)
use_mode_features = st.checkbox("Use mode features (pitch/tonic/cadence)", value=use_mode_default)
mode_pitch_bins = st.slider("Mode pitch bins", 12, 48, mode_bins_default, step=2)
low_compute = st.checkbox("Low-compute mode", value=False)

if low_compute:
    segment_seconds, num_segments, trim_db, mode_pitch_bins = apply_low_compute(
        segment_seconds,
        num_segments,
        trim_db,
        mode_pitch_bins,
    )
    st.caption(
        f"Low-compute active: segment_seconds={segment_seconds}, num_segments={num_segments}, trim_db={trim_db}, mode_pitch_bins={mode_pitch_bins}"
    )

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
    resolved_model_path = resolve_path(model_path)
    if not os.path.exists(resolved_model_path):
        st.error(
            "Model not found. Checked path: "
            + resolved_model_path
            + " (cwd: "
            + os.getcwd()
            + ", app dir: "
            + APP_DIR
            + ")."
        )
        st.stop()

    if model_choice == "Compare All (scikit)":
        lr_path = resolve_path(default_paths["LR (scikit)"])
        svm_path = resolve_path(default_paths["SVM"])
        ens_dir = os.path.dirname(resolve_path(default_paths["Ensemble (LR + SVM)"]))
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
        base_dir = os.path.dirname(resolved_model_path)
        lr_path = os.path.join(base_dir, "lr_model.joblib")
        svm_path = os.path.join(base_dir, "svm_model.joblib")
        meta_path = resolved_model_path
        if not (os.path.exists(lr_path) and os.path.exists(svm_path) and os.path.exists(meta_path)):
            st.error("Ensemble files not found. Expecting lr_model.joblib, svm_model.joblib, meta_model.joblib")
            st.stop()
        lr_model = joblib.load(lr_path)
        svm_model = joblib.load(svm_path)
        meta_model = joblib.load(meta_path)
    else:
        model = joblib.load(resolved_model_path)

    cfg = FeatureConfig()
    feat_opts = ScikitFeatureOpts(
        use_mode_features=use_mode_features,
        mode_pitch_bins=mode_pitch_bins,
    )

    results = []

    if input_mode == "Folder path":
        if not folder_path or not os.path.isdir(folder_path):
            st.error("Folder path is invalid.")
            st.stop()
        file_paths = list_mp3s(folder_path)
        if not file_paths:
            st.warning("No MP3 files found in folder.")
            st.stop()
        source_items = [(p, os.path.basename(p), False) for p in file_paths]
    else:
        if not files:
            st.warning("Please upload at least one MP3.")
            st.stop()
        source_items = []
        for uploaded in files:
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp:
                tmp.write(uploaded.getbuffer())
                source_items.append((tmp.name, uploaded.name, True))

    for path, display_name, is_temp in source_items:
        audio = load_audio(path, cfg)
        if trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=trim_db)
        seg_features = build_segment_features(audio, cfg, feat_opts, segment_seconds, num_segments)

        if model_choice == "Compare All (scikit)":
            lr_probs = lr_model.predict_proba(seg_features).mean(axis=0)
            svm_probs = svm_model.predict_proba(seg_features).mean(axis=0)
            ens_lr_probs = ens_lr_model.predict_proba(seg_features).mean(axis=0)
            ens_svm_probs = ens_svm_model.predict_proba(seg_features).mean(axis=0)
            meta_input = np.hstack([ens_lr_probs, ens_svm_probs]).reshape(1, -1)
            ens_probs = ens_meta_model.predict_proba(meta_input)[0]
            results.append(
                {
                    "file": display_name,
                    "lr_label": LABELS[int(np.argmax(lr_probs))],
                    "lr_conf": float(np.max(lr_probs)),
                    "svm_label": LABELS[int(np.argmax(svm_probs))],
                    "svm_conf": float(np.max(svm_probs)),
                    "ens_label": LABELS[int(np.argmax(ens_probs))],
                    "ens_conf": float(np.max(ens_probs)),
                }
            )
        elif model_choice == "Ensemble (LR + SVM)":
            lr_probs = lr_model.predict_proba(seg_features).mean(axis=0)
            svm_probs = svm_model.predict_proba(seg_features).mean(axis=0)
            meta_input = np.hstack([lr_probs, svm_probs]).reshape(1, -1)
            probs = meta_model.predict_proba(meta_input)[0]
            top_idx = int(np.argmax(probs))
            results.append(
                {
                    "file": display_name,
                    "label": LABELS[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probs": probs,
                }
            )
        else:
            probs = model.predict_proba(seg_features).mean(axis=0)
            top_idx = int(np.argmax(probs))
            results.append(
                {
                    "file": display_name,
                    "label": LABELS[top_idx],
                    "confidence": float(probs[top_idx]),
                    "probs": probs,
                }
            )

        if is_temp and os.path.exists(path):
            os.unlink(path)

    st.markdown("### Results")
    if model_choice == "Compare All (scikit)":
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
