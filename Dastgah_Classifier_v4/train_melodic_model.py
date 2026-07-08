import argparse
import json
import os
import shutil
import sys
from typing import List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dastgah_v4 import LABELS  # noqa: E402
from dastgah_v4.data import Track, ensure_manifest_and_splits, label_to_index  # noqa: E402
from dastgah_v4.melodic_features import MelodicFeatureConfig, build_track_matrix  # noqa: E402
from dastgah_v4.modeling import MODEL_TYPES, build_model  # noqa: E402
from dastgah_v4.paths import portable_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Root data folder containing Dastgah class folders")
    p.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.json"))
    p.add_argument("--splits", default=os.path.join(ROOT, "data", "splits.json"))
    p.add_argument("--run_dir", default=os.path.join(ROOT, "runs", "exp_melodic_svm"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "cache"))
    p.add_argument("--export_production", action="store_true")
    p.add_argument("--models_dir", default=os.path.join(ROOT, "models"))
    p.add_argument("--production_model_name", default="model.joblib")

    p.add_argument("--model_type", choices=list(MODEL_TYPES), default="svm")
    p.add_argument("--use_pca", action="store_true")
    p.add_argument("--pca_variance", type=float, default=0.95)

    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--model_jobs", type=int, default=1)
    p.add_argument("--rebuild_manifest", action="store_true")
    p.add_argument("--rebuild_splits", action="store_true")

    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--hop_length", type=int, default=512)
    p.add_argument("--bins_per_octave", type=int, default=24)
    p.add_argument("--segment_seconds", type=float, default=30.0)
    p.add_argument("--num_segments", type=int, default=6)
    p.add_argument("--trim_silence", action="store_true")
    p.add_argument("--trim_db", type=int, default=25)
    p.add_argument("--voiced_ratio_threshold", type=float, default=0.25)
    p.add_argument("--min_voiced_ms", type=float, default=120.0)
    p.add_argument("--min_harmonic_ratio", type=float, default=0.50)
    p.add_argument("--min_note_ms", type=float, default=90.0)
    p.add_argument("--stable_note_ms", type=float, default=220.0)
    p.add_argument("--phrase_gap_ms", type=float, default=280.0)
    p.add_argument("--cadence_notes", type=int, default=3)
    p.add_argument("--cadence_weight", type=float, default=1.6)
    p.add_argument("--stable_weight", type=float, default=0.8)
    p.add_argument("--step_clip_bins", type=int, default=12)
    p.add_argument("--duration_bins", type=int, default=8)
    # "vote" gave the best Homayun F1 and lowest fold variance in 5-fold CV
    # at equal macro F1 (2026-06-11); pooled tonics flip on shahed-heavy tracks.
    p.add_argument("--tonic_strategy", choices=["pooled", "vote"], default="vote")
    # v4: shahed/ist note-function features. Opt-in: the one-hot form measured
    # below parity on pooled grouped CV; default (off) reproduces v3 vectors.
    p.add_argument("--function_features", action="store_true")
    # v4: Farhat interval templates (cosine alignment to theoretical scales).
    p.add_argument("--template_features", action="store_true")
    # v4: cents-level koron intonation histograms (neutral 2nd/3rd/6th regions).
    p.add_argument("--koron_features", action="store_true")
    return p.parse_args()


def pick(tracks: List[Track], idxs: List[int]) -> List[Track]:
    return [tracks[i] for i in idxs]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_split(model, X: np.ndarray, y: np.ndarray, name: str) -> tuple[dict | None, np.ndarray | None]:
    if X.shape[0] == 0:
        print(f"{name}: skipped (empty split)")
        return None, None
    pred = model.predict(X)
    metrics = metrics_dict(y, pred)
    print(f"{name}: acc={metrics['acc']:.3f} macro_f1={metrics['macro_f1']:.3f} bal_acc={metrics['balanced_acc']:.3f}")
    return metrics, pred


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    state = ensure_manifest_and_splits(
        data_root=args.data,
        manifest_path=args.manifest,
        splits_path=args.splits,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        rebuild_manifest=args.rebuild_manifest,
        rebuild_splits=args.rebuild_splits,
    )
    if state.rebuilt_manifest:
        print(f"Manifest rebuilt: {args.manifest}")
    if state.rebuilt_splits:
        print(f"Splits rebuilt: {args.splits}")

    cfg = MelodicFeatureConfig(
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        bins_per_octave=args.bins_per_octave,
        segment_seconds=args.segment_seconds,
        num_segments=args.num_segments,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
        voiced_ratio_threshold=args.voiced_ratio_threshold,
        min_voiced_ms=args.min_voiced_ms,
        min_harmonic_ratio=args.min_harmonic_ratio,
        min_note_ms=args.min_note_ms,
        stable_note_ms=args.stable_note_ms,
        phrase_gap_ms=args.phrase_gap_ms,
        cadence_notes=args.cadence_notes,
        cadence_weight=args.cadence_weight,
        stable_weight=args.stable_weight,
        step_clip_bins=args.step_clip_bins,
        duration_bins=args.duration_bins,
        tonic_strategy=args.tonic_strategy,
        function_features=args.function_features,
        template_features=args.template_features,
        koron_features=args.koron_features,
    )

    l2i = label_to_index()
    train_tracks = pick(state.tracks, state.splits["train"])
    val_tracks = pick(state.tracks, state.splits["val"])
    test_tracks = pick(state.tracks, state.splits["test"])

    X_train, y_train = build_track_matrix(train_tracks, cfg, "train", args.seed, args.cache_dir, l2i, "Features (train)", args.num_workers)
    X_val, y_val = build_track_matrix(val_tracks, cfg, "eval", args.seed, args.cache_dir, l2i, "Features (val)", args.num_workers)
    X_test, y_test = build_track_matrix(test_tracks, cfg, "eval", args.seed, args.cache_dir, l2i, "Features (test)", args.num_workers)

    model = build_model(args.model_type, seed=args.seed, use_pca=args.use_pca, pca_variance=args.pca_variance, model_jobs=args.model_jobs)
    model.fit(X_train, y_train)

    val_m, _ = evaluate_split(model, X_val, y_val, "Val")
    test_m, test_pred = evaluate_split(model, X_test, y_test, "Test")

    class_ids = list(range(len(LABELS)))
    if test_pred is None:
        cm = np.zeros((len(LABELS), len(LABELS)), dtype=np.int64)
        report = "Test split is empty; no classification report was generated.\n"
    else:
        cm = confusion_matrix(y_test, test_pred, labels=class_ids)
        report = classification_report(y_test, test_pred, labels=class_ids, target_names=LABELS, zero_division=0)

    np.save(os.path.join(args.run_dir, "confusion.npy"), cm)
    with open(os.path.join(args.run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "val": val_m,
        "test": test_m,
        "model_type": args.model_type,
        "model_jobs": args.model_jobs,
        "use_pca": args.use_pca,
        "pca_variance": args.pca_variance,
        "feature_config": cfg.__dict__,
    }
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.run_dir, "model.joblib")
    joblib.dump(model, model_path)

    model_cfg = {
        "model_type": "melodic_v3",
        "labels": LABELS,
        "feature_config": cfg.__dict__,
        "cache_dir": portable_path(args.cache_dir, ROOT),
    }
    with open(os.path.join(args.run_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=2)

    print(f"Saved model: {model_path}")

    if args.export_production:
        os.makedirs(args.models_dir, exist_ok=True)
        prod_model = os.path.join(args.models_dir, args.production_model_name)
        prod_cfg = os.path.join(args.models_dir, "model_config.json")
        shutil.copy2(model_path, prod_model)
        shutil.copy2(os.path.join(args.run_dir, "model_config.json"), prod_cfg)
        shutil.copy2(os.path.join(args.run_dir, "metrics.json"), os.path.join(args.models_dir, "metrics.json"))
        print(f"Exported production model: {prod_model}")
        print(f"Exported production config: {prod_cfg}")


if __name__ == "__main__":
    main()
