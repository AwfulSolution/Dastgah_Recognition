"""Grouped stratified 5-fold CV — the v4 measuring stick.

Same-album/performance tracks never span train/eval (see grouping.py), and
every headline number is a mean±std over 5 folds instead of a single ~86-track
split (those swing ±7 accuracy points between seeds).

Features use train-mode extraction (random windows) for all tracks under one
fixed seed, so the whole matrix is built once and reused across folds; train
and eval rows come from the same distribution, so there is no train/eval skew.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dastgah_v4 import LABELS  # noqa: E402
from dastgah_v4.data import Track, label_to_index  # noqa: E402
from dastgah_v4.grouping import build_groups  # noqa: E402
from dastgah_v4.melodic_features import MelodicFeatureConfig, build_track_matrix  # noqa: E402
from dastgah_v4.modeling import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(REPO, "Dastgah_Classifier_v3", "data", "manifest.json"))
    p.add_argument("--groups", default=os.path.join(ROOT, "data", "groups.json"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "cache"))
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "cv_results.json"))
    p.add_argument("--model_type", default="catboost")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model_jobs", type=int, default=4)
    p.add_argument("--use_pca", action="store_true")
    p.add_argument("--pca_variance", type=float, default=0.95)
    p.add_argument("--num_segments", type=int, default=6)
    p.add_argument("--segment_seconds", type=float, default=30.0)
    p.add_argument("--tonic_strategy", choices=["pooled", "vote"], default="vote")
    p.add_argument("--no_function_features", dest="function_features", action="store_false")
    p.add_argument("--no_trim_silence", dest="trim_silence", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.manifest) as f:
        manifest = json.load(f)

    if os.path.exists(args.groups):
        with open(args.groups) as f:
            groups = json.load(f)
        if len(groups) != len(manifest):
            print("groups.json stale (length mismatch); rebuilding")
            groups = build_groups(manifest)
            with open(args.groups, "w") as f:
                json.dump(groups, f)
    else:
        groups = build_groups(manifest)
        os.makedirs(os.path.dirname(args.groups), exist_ok=True)
        with open(args.groups, "w") as f:
            json.dump(groups, f)

    cfg = MelodicFeatureConfig(
        segment_seconds=args.segment_seconds,
        num_segments=args.num_segments,
        trim_silence=args.trim_silence,
        tonic_strategy=args.tonic_strategy,
        function_features=args.function_features,
    )
    tracks = [Track(path=m["path"], label=m["label"]) for m in manifest]
    l2i = label_to_index()
    X, y = build_track_matrix(tracks, cfg, "train", args.seed, args.cache_dir, l2i, "Features (all)", args.num_workers)
    groups_arr = np.array(groups)

    skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    fold_rows = []
    per_class_f1 = np.zeros((args.folds, len(LABELS)))
    for fold, (tr, ev) in enumerate(skf.split(X, y, groups_arr)):
        model = build_model(args.model_type, seed=args.seed, use_pca=args.use_pca,
                            pca_variance=args.pca_variance, model_jobs=args.model_jobs)
        model.fit(X[tr], y[tr])
        pred = model.predict(X[ev])
        acc = float(accuracy_score(y[ev], pred))
        mf1 = float(f1_score(y[ev], pred, average="macro", zero_division=0))
        per_class_f1[fold] = f1_score(y[ev], pred, average=None, labels=range(len(LABELS)), zero_division=0)
        eval_label_counts = {LABELS[i]: int(n) for i, n in zip(*np.unique(y[ev], return_counts=True))}
        fold_rows.append({"fold": fold, "n_eval": len(ev), "acc": acc, "macro_f1": mf1,
                          "eval_labels": eval_label_counts})
        print(f"fold {fold}: n={len(ev):3d} acc={acc:.3f} macro_f1={mf1:.3f}", flush=True)

    accs = np.array([r["acc"] for r in fold_rows])
    f1s = np.array([r["macro_f1"] for r in fold_rows])
    summary = {
        "acc_mean": round(float(accs.mean()), 4), "acc_std": round(float(accs.std()), 4),
        "macro_f1_mean": round(float(f1s.mean()), 4), "macro_f1_std": round(float(f1s.std()), 4),
        "per_class_f1_mean": {LABELS[i]: round(float(v), 3) for i, v in enumerate(per_class_f1.mean(axis=0))},
        "config": {"model_type": args.model_type, "seed": args.seed, "folds": args.folds,
                   "function_features": args.function_features, "tonic_strategy": args.tonic_strategy,
                   "num_segments": args.num_segments, "segment_seconds": args.segment_seconds},
        "folds": fold_rows,
    }
    print(f"\nCV: acc {summary['acc_mean']:.3f} ± {summary['acc_std']:.3f} | "
          f"macro F1 {summary['macro_f1_mean']:.3f} ± {summary['macro_f1_std']:.3f}")
    print("per-class F1:", summary["per_class_f1_mean"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
