import argparse
import json
import os

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.dastgah.data import LABELS, ensure_manifest_and_splits
from src.dastgah.features import FeatureConfig
from src.dastgah.model_config import save_model_config
from src.dastgah.scikit_features import ScikitFeatureOpts
from src.dastgah.scikit_train_utils import (
    build_xy,
    cross_validate_trackwise,
    predict_trackwise,
    select_tracks,
    track_metrics,
    true_labels_by_track,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to parent data folder")
    parser.add_argument("--manifest", default=os.path.join(BASE_DIR, "data", "manifest.json"))
    parser.add_argument("--splits", default=os.path.join(BASE_DIR, "data", "splits.json"))
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--cache_dir", default=os.path.join(BASE_DIR, "data", "cache"))
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--use_mode_features", action="store_true")
    parser.add_argument("--mode_pitch_bins", type=int, default=24)
    parser.add_argument("--use_pca", action="store_true")
    parser.add_argument("--pca_variance", type=float, default=0.95)
    parser.add_argument("--low_compute", action="store_true")
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", default=os.path.join(BASE_DIR, "runs", "exp_sklearn"))
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--rebuild_splits", action="store_true")
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--cv_repeats", type=int, default=1)
    return parser.parse_args()


def apply_low_compute(args: argparse.Namespace) -> None:
    if not args.low_compute:
        return
    if args.segment_seconds == 45.0:
        args.segment_seconds = 25.0
    if args.num_segments == 10:
        args.num_segments = 4
    if args.trim_db == 25:
        args.trim_db = 30
    if args.mode_pitch_bins == 24:
        args.mode_pitch_bins = 16


def build_model(args: argparse.Namespace) -> Pipeline:
    steps = [("scaler", StandardScaler())]
    if args.use_pca:
        steps.append(("pca", PCA(n_components=args.pca_variance, svd_solver="full")))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=2000,
                multi_class="multinomial",
                solver="lbfgs",
                class_weight="balanced",
                random_state=args.seed,
            ),
        )
    )
    return Pipeline(steps=steps)


def main() -> None:
    args = parse_args()
    apply_low_compute(args)
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
    tracks = state.tracks
    splits = state.splits
    if state.rebuilt_manifest:
        print(f"Manifest rebuilt: {args.manifest}")
    if state.rebuilt_splits:
        print(f"Splits rebuilt: {args.splits}")

    feature_cfg = FeatureConfig()
    feat_opts = ScikitFeatureOpts(
        use_mode_features=args.use_mode_features,
        mode_pitch_bins=args.mode_pitch_bins,
    )

    train_tracks = select_tracks(tracks, splits["train"])
    val_tracks = select_tracks(tracks, splits["val"])
    test_tracks = select_tracks(tracks, splits["test"])

    X_train, y_train, _ = build_xy(
        train_tracks,
        feature_cfg,
        feat_opts,
        args.segment_seconds,
        args.num_segments,
        args.seed,
        mode="train",
        cache_dir=args.cache_dir,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
    )
    X_val, y_val, val_track_ids = build_xy(
        val_tracks,
        feature_cfg,
        feat_opts,
        args.segment_seconds,
        args.num_segments,
        args.seed,
        mode="eval",
        cache_dir=args.cache_dir,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
    )

    if test_tracks:
        X_test, y_test, test_track_ids = build_xy(
            test_tracks,
            feature_cfg,
            feat_opts,
            args.segment_seconds,
            args.num_segments,
            args.seed,
            mode="eval",
            cache_dir=args.cache_dir,
            trim_silence=args.trim_silence,
            trim_db=args.trim_db,
        )
    else:
        X_test = None
        y_test = None
        test_track_ids = []

    model = build_model(args)
    model.fit(X_train, y_train)

    val_pred = predict_trackwise(model, X_val, val_track_ids)
    val_true = true_labels_by_track(val_track_ids, y_val)
    val_scores = track_metrics(val_true, val_pred)
    print(
        "Val metrics: "
        f"acc={val_scores.accuracy:.3f} "
        f"macro_f1={val_scores.macro_f1:.3f} "
        f"bal_acc={val_scores.balanced_accuracy:.3f}"
    )

    payload = {
        "val_acc": val_scores.accuracy,
        "val_macro_f1": val_scores.macro_f1,
        "val_balanced_acc": val_scores.balanced_accuracy,
        "segment_seconds": args.segment_seconds,
        "num_segments": args.num_segments,
        "trim_silence": args.trim_silence,
        "trim_db": args.trim_db,
        "use_mode_features": args.use_mode_features,
        "mode_pitch_bins": args.mode_pitch_bins,
        "use_pca": args.use_pca,
        "pca_variance": args.pca_variance,
        "low_compute": args.low_compute,
    }

    if X_test is not None and len(X_test) > 0:
        test_pred = predict_trackwise(model, X_test, test_track_ids)
        test_true = true_labels_by_track(test_track_ids, y_test)
        test_scores = track_metrics(test_true, test_pred)
        print(
            "Test metrics: "
            f"acc={test_scores.accuracy:.3f} "
            f"macro_f1={test_scores.macro_f1:.3f} "
            f"bal_acc={test_scores.balanced_accuracy:.3f}"
        )
        payload["test_acc"] = test_scores.accuracy
        payload["test_macro_f1"] = test_scores.macro_f1
        payload["test_balanced_acc"] = test_scores.balanced_accuracy

        cm = confusion_matrix(test_true, test_pred)
        report = classification_report(test_true, test_pred, target_names=LABELS)
        np.save(os.path.join(args.run_dir, "confusion.npy"), cm)
        with open(
            os.path.join(args.run_dir, "classification_report.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(report)

    if args.cv_folds >= 2:
        cv_summary = cross_validate_trackwise(
            tracks=train_tracks,
            cfg=feature_cfg,
            opts=feat_opts,
            segment_seconds=args.segment_seconds,
            num_segments=args.num_segments,
            cache_dir=args.cache_dir,
            trim_silence=args.trim_silence,
            trim_db=args.trim_db,
            seed=args.seed,
            n_splits=args.cv_folds,
            n_repeats=args.cv_repeats,
            model_factory=lambda: build_model(args),
        )
        payload["cv"] = cv_summary
        print(
            "CV summary (train split): "
            f"acc={cv_summary['accuracy']['mean']:.3f} "
            f"[{cv_summary['accuracy']['ci95_low']:.3f}, {cv_summary['accuracy']['ci95_high']:.3f}]"
        )

    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    joblib.dump(model, os.path.join(args.run_dir, "model.joblib"))
    save_model_config(
        args.run_dir,
        {
            "model_type": "lr_scikit",
            "segment_seconds": args.segment_seconds,
            "num_segments": args.num_segments,
            "trim_silence": args.trim_silence,
            "trim_db": args.trim_db,
            "use_mode_features": args.use_mode_features,
            "mode_pitch_bins": args.mode_pitch_bins,
            "use_pca": args.use_pca,
            "pca_variance": args.pca_variance,
        },
    )


if __name__ == "__main__":
    main()
