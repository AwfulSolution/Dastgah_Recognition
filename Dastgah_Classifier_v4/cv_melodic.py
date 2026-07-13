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
from typing import List

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
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
    p.add_argument("--function_features", action="store_true")
    p.add_argument("--template_features", action="store_true")
    p.add_argument("--koron_features", action="store_true")
    p.add_argument("--no_trim_silence", dest="trim_silence", action="store_false")
    # "balanced": greedy size+class-balanced group assignment (default);
    # "sklearn": StratifiedGroupKFold, kept for comparison.
    p.add_argument("--splitter", choices=["balanced", "sklearn"], default="balanced")
    return p.parse_args()


def balanced_group_folds(y: np.ndarray, groups: np.ndarray, n_folds: int, seed: int):
    """Greedy group-to-fold assignment balancing fold size and class mix.

    StratifiedGroupKFold produced folds of 47-262 tracks on this dataset (a few
    performer sets hold most of the data), which makes per-fold metrics
    incomparable. Here groups are placed largest-first into the fold where the
    resulting label counts stay closest to the per-fold target, so both size
    and stratification are optimized together. Groups still never split.
    """
    rng = np.random.RandomState(seed)
    n_classes = int(y.max()) + 1
    target = np.bincount(y, minlength=n_classes).astype(np.float64) / n_folds

    by_group: dict = {}
    for i, g in enumerate(groups):
        by_group.setdefault(g, []).append(i)
    items = list(by_group.items())
    rng.shuffle(items)  # tie-break order for equal-sized groups
    items.sort(key=lambda kv: len(kv[1]), reverse=True)

    fold_counts = np.zeros((n_folds, n_classes), dtype=np.float64)
    fold_members: List[List[int]] = [[] for _ in range(n_folds)]
    for _, idxs in items:
        gvec = np.bincount(y[idxs], minlength=n_classes).astype(np.float64)
        # Marginal cost: how much this assignment increases the fold's squared
        # deviation from target. Comparing absolute post-assignment distance
        # instead makes near-full folds the cheapest home for every small
        # group and starves the empty ones.
        costs = [
            float(np.sum((fold_counts[f] + gvec - target) ** 2) - np.sum((fold_counts[f] - target) ** 2))
            for f in range(n_folds)
        ]
        best = int(np.argmin(costs))
        fold_counts[best] += gvec
        fold_members[best].extend(idxs)

    all_idx = np.arange(len(y))
    for f in range(n_folds):
        ev = np.array(sorted(fold_members[f]), dtype=np.int64)
        tr = np.setdiff1d(all_idx, ev)
        yield tr, ev


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
        template_features=args.template_features,
        koron_features=args.koron_features,
    )
    tracks = [Track(path=m["path"], label=m["label"]) for m in manifest]
    l2i = label_to_index()
    X, y = build_track_matrix(tracks, cfg, "train", args.seed, args.cache_dir, l2i, "Features (all)", args.num_workers)
    groups_arr = np.array(groups)

    if args.splitter == "balanced":
        splits = balanced_group_folds(y, groups_arr, args.folds, args.seed)
    else:
        skf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
        splits = skf.split(X, y, groups_arr)

    # Pooled evaluation: every track is predicted exactly once, by a model
    # that never saw its group. Aggregate metrics over these 570 predictions
    # are the headline; per-fold rows are kept for dispersion only.
    pooled_pred = np.full(len(y), -1, dtype=np.int64)
    fold_rows = []
    for fold, (tr, ev) in enumerate(splits):
        model = build_model(args.model_type, seed=args.seed, use_pca=args.use_pca,
                            pca_variance=args.pca_variance, model_jobs=args.model_jobs)
        model.fit(X[tr], y[tr])
        pred = np.asarray(model.predict(X[ev])).reshape(-1).astype(np.int64)
        pooled_pred[ev] = pred
        acc = float(accuracy_score(y[ev], pred))
        mf1 = float(f1_score(y[ev], pred, average="macro", zero_division=0))
        eval_label_counts = {LABELS[i]: int(n) for i, n in zip(*np.unique(y[ev], return_counts=True))}
        fold_rows.append({"fold": fold, "n_eval": len(ev), "acc": acc, "macro_f1": mf1,
                          "eval_labels": eval_label_counts})
        print(f"fold {fold}: n={len(ev):3d} acc={acc:.3f} macro_f1={mf1:.3f}", flush=True)

    assert int((pooled_pred < 0).sum()) == 0, "some tracks were never evaluated"
    pooled = {
        "acc": round(float(accuracy_score(y, pooled_pred)), 4),
        "macro_f1": round(float(f1_score(y, pooled_pred, average="macro", zero_division=0)), 4),
        "per_class_f1": {LABELS[i]: round(float(v), 3) for i, v in enumerate(
            f1_score(y, pooled_pred, average=None, labels=range(len(LABELS)), zero_division=0))},
        "confusion": confusion_matrix(y, pooled_pred, labels=range(len(LABELS))).tolist(),
    }
    accs = np.array([r["acc"] for r in fold_rows])
    summary = {
        "pooled": pooled,
        "fold_acc_mean": round(float(accs.mean()), 4), "fold_acc_std": round(float(accs.std()), 4),
        "config": {"model_type": args.model_type, "seed": args.seed, "folds": args.folds,
                   "splitter": args.splitter,
                   "function_features": args.function_features,
                   "template_features": args.template_features,
                   "koron_features": args.koron_features, "tonic_strategy": args.tonic_strategy,
                   "num_segments": args.num_segments, "segment_seconds": args.segment_seconds},
        "folds": fold_rows,
        "predictions": [LABELS[p] for p in pooled_pred],
    }
    print(f"\nPooled CV: acc {pooled['acc']:.3f} | macro F1 {pooled['macro_f1']:.3f} "
          f"(fold acc {summary['fold_acc_mean']:.3f} ± {summary['fold_acc_std']:.3f})")
    print("per-class F1:", pooled["per_class_f1"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
