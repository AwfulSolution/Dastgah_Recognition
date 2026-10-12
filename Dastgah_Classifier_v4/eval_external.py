"""Evaluate a model trained on the full corpus against an external test set.

Grouped CV removes album and performer leakage *within* one collection, but it
still measures generalisation to performers drawn from the same pool. An
external corpus recorded by musicians who appear nowhere in training is a
stricter test of the same question, and the one this project cares about most:
whether the features describe the mode or the performer.

Reads a manifest of {path, label, ...} entries (see data/kdc_manifest.json),
extracts features with the same config used for training, and reports pooled
accuracy, macro F1, per-class F1 and the confusion matrix — plus per-track
predictions, so misclassifications can be inspected individually.
"""

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dastgah_v4 import LABELS  # noqa: E402
from dastgah_v4.data import Track, label_to_index  # noqa: E402
from dastgah_v4.melodic_features import MelodicFeatureConfig, build_track_matrix  # noqa: E402
from dastgah_v4.modeling import build_model  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--train_manifest", default=os.path.join(REPO, "Dastgah_Classifier_v3", "data", "manifest.json"))
    p.add_argument("--test_manifest", default=os.path.join(ROOT, "data", "kdc_manifest.json"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "cache"))
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "eval_external.json"))
    p.add_argument("--model_type", default="catboost")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model_jobs", type=int, default=4)
    p.add_argument("--num_segments", type=int, default=6)
    p.add_argument("--segment_seconds", type=float, default=30.0)
    p.add_argument("--tonic_strategy", choices=["pooled", "vote"], default="vote")
    p.add_argument("--function_features", action="store_true")
    p.add_argument("--template_features", action="store_true")
    p.add_argument("--no_koron_features", dest="koron_features", action="store_false")
    p.add_argument("--no_trim_silence", dest="trim_silence", action="store_false")
    return p.parse_args()


def main() -> None:
    args = parse_args()
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

    with open(args.train_manifest) as f:
        train_manifest = json.load(f)
    with open(args.test_manifest) as f:
        test_manifest = json.load(f)
    test_manifest = [m for m in test_manifest if os.path.exists(m["path"])]
    print(f"train: {len(train_manifest)} tracks | external test: {len(test_manifest)} tracks", flush=True)

    train_tracks = [Track(path=m["path"], label=m["label"]) for m in train_manifest]
    test_tracks = [Track(path=m["path"], label=m["label"]) for m in test_manifest]

    # "train" extraction mode for both: it is what the model was fitted on, so
    # the test features come from the same windowing distribution.
    X_tr, y_tr = build_track_matrix(train_tracks, cfg, "train", args.seed, args.cache_dir,
                                    l2i, "Features (train corpus)", args.num_workers)
    X_te, y_te = build_track_matrix(test_tracks, cfg, "train", args.seed, args.cache_dir,
                                    l2i, "Features (external)", args.num_workers)

    model = build_model(args.model_type, seed=args.seed, use_pca=False,
                        pca_variance=0.95, model_jobs=args.model_jobs)
    model.fit(X_tr, y_tr)
    pred = np.asarray(model.predict(X_te)).reshape(-1).astype(np.int64)

    acc = float(accuracy_score(y_te, pred))
    mf1 = float(f1_score(y_te, pred, average="macro", zero_division=0))
    per_class = {LABELS[i]: round(float(v), 3) for i, v in enumerate(
        f1_score(y_te, pred, average=None, labels=range(len(LABELS)), zero_division=0))}
    cm = confusion_matrix(y_te, pred, labels=range(len(LABELS)))

    print(f"\nExternal test: acc {acc:.3f} | macro F1 {mf1:.3f}  (n={len(y_te)})")
    print("per-class F1:", per_class)
    print("\nconfusion (rows = true):")
    print(" " * 11 + " ".join(f"{l[:6]:>7}" for l in LABELS))
    for l, row in zip(LABELS, cm.tolist()):
        print(f"{l:10}", " ".join(f"{v:7d}" for v in row))

    by_perf = {}
    for m, t, p in zip(test_manifest, y_te, pred):
        k = m.get("performer", "?")
        c = by_perf.setdefault(k, Counter())
        c["n"] += 1
        c["correct"] += int(t == p)
    print("\nby performer:")
    for k, c in sorted(by_perf.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {c['correct']:3d}/{c['n']:3d}  {k}")

    out = {
        "acc": round(acc, 4), "macro_f1": round(mf1, 4), "n": len(y_te),
        "per_class_f1": per_class,
        "confusion": cm.tolist(),
        "labels": list(LABELS),
        "config": {"model_type": args.model_type, "seed": args.seed,
                   "koron_features": args.koron_features,
                   "template_features": args.template_features,
                   "function_features": args.function_features},
        "by_performer": {k: dict(v) for k, v in by_perf.items()},
        "predictions": [
            {"path": os.path.basename(m["path"]), "title": m.get("title", ""),
             "performer": m.get("performer", ""), "true": LABELS[t], "pred": LABELS[p]}
            for m, t, p in zip(test_manifest, y_te, pred)
        ],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
