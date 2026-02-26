import json
import os
import sys
import tempfile
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dastgah_v2.interval_features import IntervalFeatureConfig, extract_track_feature  # noqa: E402


def list_run_model_dirs(runs_dir: str) -> List[str]:
    if not os.path.isdir(runs_dir):
        return []
    out: List[str] = []
    for name in sorted(os.listdir(runs_dir)):
        p = os.path.join(runs_dir, name)
        if not os.path.isdir(p):
            continue
        if os.path.exists(os.path.join(p, "model.joblib")) and os.path.exists(os.path.join(p, "model_config.json")):
            out.append(p)
    return out


def production_bundle(models_dir: str) -> Tuple[str, str] | None:
    model_path = os.path.join(models_dir, "model.joblib")
    cfg_path = os.path.join(models_dir, "model_config.json")
    if os.path.exists(model_path) and os.path.exists(cfg_path):
        return model_path, cfg_path
    return None


def load_bundle(model_path: str, cfg_path: str) -> Dict:
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    model = joblib.load(model_path)
    labels = cfg["labels"]
    feat_cfg = IntervalFeatureConfig(**cfg["feature_config"])
    cache_dir = cfg.get("cache_dir", os.path.join(ROOT, "data", "cache"))
    return {
        "model": model,
        "labels": labels,
        "feat_cfg": feat_cfg,
        "cache_dir": cache_dir,
        "model_path": model_path,
        "cfg_path": cfg_path,
    }


def predict_file(bundle: Dict, file_path: str) -> Tuple[str, np.ndarray]:
    feat = extract_track_feature(
        track_path=file_path,
        cfg=bundle["feat_cfg"],
        mode="inference",
        seed=42,
        cache_dir=bundle["cache_dir"],
    ).reshape(1, -1)
    model = bundle["model"]
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
    else:
        scores = model.decision_function(feat)[0]
        scores = scores - np.max(scores)
        probs = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-9)
    idx = int(np.argmax(probs))
    return bundle["labels"][idx], probs


st.set_page_config(page_title="Dastgah Classifier v2", layout="centered")
st.title("Dastgah Classifier v2")
st.caption("v2 setup with run-based models, production models/, and compare mode.")

runs_dir = os.path.join(ROOT, "runs")
models_dir = os.path.join(ROOT, "models")
run_dirs = list_run_model_dirs(runs_dir)
prod = production_bundle(models_dir)

mode = st.selectbox("Mode", ["Single model", "Compare runs"], index=0)
uploads = st.file_uploader("Upload audio files", type=["mp3", "wav", "flac", "m4a", "ogg"], accept_multiple_files=True)

if mode == "Single model":
    source = st.selectbox("Model source", ["Run directory", "Production (models/model.joblib)"], index=0)

    selected_bundle = None
    if source == "Run directory":
        if not run_dirs:
            st.error("No run models found in runs/. Train first.")
            st.stop()
        selected_run = st.selectbox("Run", run_dirs, index=len(run_dirs) - 1)
        selected_bundle = load_bundle(
            os.path.join(selected_run, "model.joblib"),
            os.path.join(selected_run, "model_config.json"),
        )
    else:
        if prod is None:
            st.error("No production model found at models/model.joblib + models/model_config.json")
            st.stop()
        selected_bundle = load_bundle(prod[0], prod[1])

    if st.button("Classify", key="single_classify"):
        if not uploads:
            st.warning("Upload at least one audio file.")
            st.stop()

        temp_files: List[Tuple[str, str]] = []
        try:
            for up in uploads:
                suffix = os.path.splitext(up.name)[1] or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(up.read())
                    temp_files.append((up.name, tmp.name))

            rows = []
            first_probs = None
            for display_name, tmp_path in temp_files:
                pred_label, probs = predict_file(selected_bundle, tmp_path)
                rows.append(
                    {
                        "file": display_name,
                        "prediction": pred_label,
                        "confidence": float(np.max(probs)),
                    }
                )
                if first_probs is None:
                    first_probs = probs

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            if first_probs is not None:
                labels = selected_bundle["labels"]
                st.bar_chart(pd.DataFrame({"label": labels, "probability": first_probs}).set_index("label"))
        finally:
            for _, tmp_path in temp_files:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

else:
    if not run_dirs:
        st.error("No run models found in runs/. Train first.")
        st.stop()

    selected_runs = st.multiselect("Runs to compare", options=run_dirs, default=run_dirs[-min(3, len(run_dirs)) :])
    if st.button("Compare", key="compare_runs"):
        if not uploads:
            st.warning("Upload at least one audio file.")
            st.stop()
        if not selected_runs:
            st.warning("Select at least one run.")
            st.stop()

        bundles = {r: load_bundle(os.path.join(r, "model.joblib"), os.path.join(r, "model_config.json")) for r in selected_runs}
        temp_files: List[Tuple[str, str]] = []
        try:
            for up in uploads:
                suffix = os.path.splitext(up.name)[1] or ".mp3"
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(up.read())
                    temp_files.append((up.name, tmp.name))

            rows = []
            for display_name, tmp_path in temp_files:
                row = {"file": display_name}
                for run_dir, bundle in bundles.items():
                    pred_label, probs = predict_file(bundle, tmp_path)
                    row[os.path.basename(run_dir)] = f"{pred_label} ({float(np.max(probs)):.3f})"
                rows.append(row)
            st.dataframe(pd.DataFrame(rows), use_container_width=True)
        finally:
            for _, tmp_path in temp_files:
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
