import json
import os
import sys
import tempfile

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dastgah_v2.interval_features import IntervalFeatureConfig, extract_track_feature  # noqa: E402


def list_model_dirs(runs_dir: str) -> list[str]:
    if not os.path.isdir(runs_dir):
        return []
    out = []
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "model.joblib")) and os.path.exists(os.path.join(p, "model_config.json")):
            out.append(p)
    return out


def load_model_bundle(model_dir: str):
    with open(os.path.join(model_dir, "model_config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    labels = cfg["labels"]
    feat_cfg = IntervalFeatureConfig(**cfg["feature_config"])
    cache_dir = cfg.get("cache_dir", os.path.join(ROOT, "data", "cache"))
    return model, labels, feat_cfg, cache_dir


def predict_file(model, labels, feat_cfg, cache_dir: str, file_path: str):
    feat = extract_track_feature(
        track_path=file_path,
        cfg=feat_cfg,
        mode="inference",
        seed=42,
        cache_dir=cache_dir,
    ).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
    else:
        scores = model.decision_function(feat)[0]
        scores = scores - np.max(scores)
        probs = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-9)
    idx = int(np.argmax(probs))
    return labels[idx], probs


st.set_page_config(page_title="Dastgah Classifier v2", layout="centered")
st.title("Dastgah Classifier v2 (Interval-First)")
st.caption("Uses interval transitions + cadence + voiced/harmonic filtering to suppress percussive/no-pitch segments.")

runs_dir = os.path.join(ROOT, "runs")
model_dirs = list_model_dirs(runs_dir)
if not model_dirs:
    st.error("No trained model found. Train first with train_interval_model.py")
    st.stop()

selected_model = st.selectbox("Model run", model_dirs, index=len(model_dirs) - 1)
uploaded = st.file_uploader("Upload audio", type=["mp3", "wav", "flac", "m4a"])

if uploaded is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded.name)[1] or ".mp3") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    try:
        model, labels, feat_cfg, cache_dir = load_model_bundle(selected_model)
        pred_label, probs = predict_file(model, labels, feat_cfg, cache_dir, tmp_path)

        st.subheader(f"Prediction: {pred_label}")
        df = pd.DataFrame({"label": labels, "probability": probs}).sort_values("probability", ascending=False)
        st.bar_chart(df.set_index("label"))
        st.dataframe(df, use_container_width=True)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
