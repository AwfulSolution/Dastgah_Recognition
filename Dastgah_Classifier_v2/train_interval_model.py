import argparse
import json
import os
import sys
from typing import List

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dastgah_v2 import LABELS  # noqa: E402
from dastgah_v2.data import Track, ensure_manifest_and_splits, label_to_index  # noqa: E402
from dastgah_v2.interval_features import IntervalFeatureConfig, build_track_matrix  # noqa: E402
from dastgah_v2.modeling import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Root data folder containing Dastgah class folders")
    p.add_argument("--manifest", default=os.path.join(ROOT, "data", "manifest.json"))
    p.add_argument("--splits", default=os.path.join(ROOT, "data", "splits.json"))
    p.add_argument("--run_dir", default=os.path.join(ROOT, "runs", "exp_interval_svm"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "cache"))

    p.add_argument("--model_type", choices=["lr", "svm"], default="svm")
    p.add_argument("--use_pca", action="store_true")
    p.add_argument("--pca_variance", type=float, default=0.95)

    p.add_argument("--val_split", type=float, default=0.15)
    p.add_argument("--test_split", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--rebuild_manifest", action="store_true")
    p.add_argument("--rebuild_splits", action="store_true")

    p.add_argument("--sample_rate", type=int, default=22050)
    p.add_argument("--n_fft", type=int, default=2048)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--bins_per_octave", type=int, default=24)
    p.add_argument("--segment_seconds", type=float, default=20.0)
    p.add_argument("--num_segments", type=int, default=8)
    p.add_argument("--trim_silence", action="store_true")
    p.add_argument("--trim_db", type=int, default=25)
    p.add_argument("--voiced_ratio_threshold", type=float, default=0.30)
    p.add_argument("--min_voiced_ms", type=float, default=120.0)
    p.add_argument("--min_harmonic_ratio", type=float, default=0.55)
    p.add_argument("--cadence_fraction", type=float, default=0.15)
    p.add_argument("--step_clip_bins", type=int, default=12)
    p.add_argument("--duration_bins", type=int, default=8)
    return p.parse_args()


def pick(tracks: List[Track], idxs: List[int]) -> List[Track]:
    return [tracks[i] for i in idxs]


def metrics_dict(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
    }


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

    tracks = state.tracks
    splits = state.splits
    l2i = label_to_index()

    cfg = IntervalFeatureConfig(
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
        cadence_fraction=args.cadence_fraction,
        step_clip_bins=args.step_clip_bins,
        duration_bins=args.duration_bins,
    )

    train_tracks = pick(tracks, splits["train"])
    val_tracks = pick(tracks, splits["val"])
    test_tracks = pick(tracks, splits["test"])

    X_train, y_train = build_track_matrix(
        train_tracks,
        cfg,
        mode="train",
        seed=args.seed,
        cache_dir=args.cache_dir,
        label_to_idx=l2i,
        progress_label="Features (train)",
    )
    X_val, y_val = build_track_matrix(
        val_tracks,
        cfg,
        mode="eval",
        seed=args.seed,
        cache_dir=args.cache_dir,
        label_to_idx=l2i,
        progress_label="Features (val)",
    )
    X_test, y_test = build_track_matrix(
        test_tracks,
        cfg,
        mode="eval",
        seed=args.seed,
        cache_dir=args.cache_dir,
        label_to_idx=l2i,
        progress_label="Features (test)",
    )

    model = build_model(args.model_type, seed=args.seed, use_pca=args.use_pca, pca_variance=args.pca_variance)
    model.fit(X_train, y_train)

    val_pred = model.predict(X_val)
    val_m = metrics_dict(y_val, val_pred)
    print(f"Val: acc={val_m['acc']:.3f} macro_f1={val_m['macro_f1']:.3f} bal_acc={val_m['balanced_acc']:.3f}")

    test_pred = model.predict(X_test)
    test_m = metrics_dict(y_test, test_pred)
    print(f"Test: acc={test_m['acc']:.3f} macro_f1={test_m['macro_f1']:.3f} bal_acc={test_m['balanced_acc']:.3f}")

    class_ids = list(range(len(LABELS)))
    cm = confusion_matrix(y_test, test_pred, labels=class_ids)
    report = classification_report(y_test, test_pred, labels=class_ids, target_names=LABELS, zero_division=0)

    np.save(os.path.join(args.run_dir, "confusion.npy"), cm)
    with open(os.path.join(args.run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    metrics = {
        "val": val_m,
        "test": test_m,
        "model_type": args.model_type,
        "use_pca": args.use_pca,
        "pca_variance": args.pca_variance,
        "feature_config": cfg.__dict__,
    }
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    model_path = os.path.join(args.run_dir, "model.joblib")
    joblib.dump(model, model_path)

    model_cfg = {
        "model_type": "interval_v2",
        "labels": LABELS,
        "feature_config": cfg.__dict__,
        "cache_dir": args.cache_dir,
    }
    with open(os.path.join(args.run_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_cfg, f, indent=2)

    print(f"Saved model: {model_path}")


if __name__ == "__main__":
    main()
