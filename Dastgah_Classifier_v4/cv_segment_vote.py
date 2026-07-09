"""Grouped pooled CV for abstention segment voting.

Same leakage rules and balanced folds as cv_melodic.py, but the classifier is
trained on per-segment vectors (label = parent track's dastgah) and a track's
prediction aggregates its segments' probabilities. Segments whose top
probability falls below an abstention threshold tau are dropped before
aggregation — the mechanism that keeps modulating gushehs (Hesar, Rak) from
poisoning the track vote. All taus are evaluated from one set of probabilities,
so the sweep costs nothing beyond the base fits.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from tqdm.auto import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from cv_melodic import balanced_group_folds  # noqa: E402
from dastgah_v4 import LABELS  # noqa: E402
from dastgah_v4.data import label_to_index  # noqa: E402
from dastgah_v4.melodic_features import (  # noqa: E402
    MelodicFeatureConfig,
    extract_track_segment_features,
)
from dastgah_v4.modeling import build_model  # noqa: E402

TAUS = [0.0, 0.3, 0.4, 0.5, 0.6]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", default=os.path.join(REPO, "Dastgah_Classifier_v3", "data", "manifest.json"))
    p.add_argument("--groups", default=os.path.join(ROOT, "data", "groups.json"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "cache"))
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "cv_segvote.json"))
    p.add_argument("--model_type", default="catboost")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_jobs", type=int, default=4)
    p.add_argument("--num_segments", type=int, default=6)
    p.add_argument("--segment_seconds", type=float, default=30.0)
    p.add_argument("--tonic_strategy", choices=["pooled", "vote"], default="vote")
    p.add_argument("--function_features", action="store_true")
    p.add_argument("--template_features", action="store_true")
    p.add_argument("--koron_features", action="store_true")
    p.add_argument("--no_trim_silence", dest="trim_silence", action="store_false")
    return p.parse_args()


def aggregate(probas: np.ndarray, tau: float) -> int:
    """Track prediction from its segments' probability rows with abstention."""
    keep = probas.max(axis=1) >= tau
    votes = probas[keep] if keep.any() else probas  # all abstained: fall back to all
    return int(np.argmax(votes.mean(axis=0)))


def main() -> None:
    args = parse_args()
    with open(args.manifest) as f:
        manifest = json.load(f)
    with open(args.groups) as f:
        groups = np.array(json.load(f))

    cfg = MelodicFeatureConfig(
        segment_seconds=args.segment_seconds,
        num_segments=args.num_segments,
        trim_silence=args.trim_silence,
        tonic_strategy=args.tonic_strategy,
        function_features=args.function_features,
        template_features=args.template_features,
        koron_features=args.koron_features,
    )
    l2i = label_to_index()
    y = np.array([l2i[m["label"]] for m in manifest], dtype=np.int64)

    seg_X, seg_track = [], []
    for i, m in enumerate(tqdm(manifest, desc="Segment features")):
        mat = extract_track_segment_features(m["path"], cfg, "train", args.seed, args.cache_dir)
        if mat.shape[0]:
            seg_X.append(mat)
            seg_track.extend([i] * mat.shape[0])
    seg_X = np.vstack(seg_X)
    seg_track = np.array(seg_track, dtype=np.int64)
    seg_y = y[seg_track]
    print(f"{seg_X.shape[0]} segment rows from {len(manifest)} tracks, dim {seg_X.shape[1]}", flush=True)

    pooled_pred = {tau: np.full(len(y), -1, dtype=np.int64) for tau in TAUS}
    fold_rows = []
    for fold, (tr_tracks, ev_tracks) in enumerate(balanced_group_folds(y, groups, args.folds, args.seed)):
        tr_mask = np.isin(seg_track, tr_tracks)
        model = build_model(args.model_type, seed=args.seed, use_pca=False,
                            pca_variance=0.95, model_jobs=args.model_jobs)
        model.fit(seg_X[tr_mask], seg_y[tr_mask])

        ev_mask = np.isin(seg_track, ev_tracks)
        probas = model.predict_proba(seg_X[ev_mask])
        ev_seg_track = seg_track[ev_mask]
        no_segment_tracks = [t for t in ev_tracks if t not in set(ev_seg_track.tolist())]
        train_majority = int(np.bincount(seg_y[tr_mask]).argmax())
        for t in ev_tracks:
            rows = probas[ev_seg_track == t]
            for tau in TAUS:
                pooled_pred[tau][t] = aggregate(rows, tau) if rows.shape[0] else train_majority
        fold_acc = float(accuracy_score(y[ev_tracks], pooled_pred[0.0][ev_tracks]))
        fold_rows.append({"fold": fold, "n_eval": len(ev_tracks), "acc_tau0": fold_acc,
                          "no_segment_tracks": len(no_segment_tracks)})
        print(f"fold {fold}: n={len(ev_tracks):3d} acc(tau=0)={fold_acc:.3f}", flush=True)

    results = {}
    for tau in TAUS:
        pred = pooled_pred[tau]
        assert int((pred < 0).sum()) == 0
        results[str(tau)] = {
            "acc": round(float(accuracy_score(y, pred)), 4),
            "macro_f1": round(float(f1_score(y, pred, average="macro", zero_division=0)), 4),
            "per_class_f1": {LABELS[i]: round(float(v), 3) for i, v in enumerate(
                f1_score(y, pred, average=None, labels=range(len(LABELS)), zero_division=0))},
        }
        print(f"tau={tau}: acc {results[str(tau)]['acc']:.3f} | macro F1 {results[str(tau)]['macro_f1']:.3f}", flush=True)

    out = {
        "taus": results,
        "config": {"model_type": args.model_type, "seed": args.seed, "folds": args.folds,
                   "function_features": args.function_features,
                   "template_features": args.template_features,
                   "koron_features": args.koron_features, "tonic_strategy": args.tonic_strategy},
        "folds": fold_rows,
        "predictions_tau0": [LABELS[p] for p in pooled_pred[0.0]],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
