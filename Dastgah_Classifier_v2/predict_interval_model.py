import argparse
import json
import os
import sys
from typing import List, Tuple

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from dastgah_v2.interval_features import IntervalFeatureConfig, extract_track_feature  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True, help="Audio file path (.mp3/.wav)")
    p.add_argument("--model_dir", required=True, help="Directory containing model.joblib and model_config.json")
    p.add_argument("--cache_dir", default=None, help="Override cache dir from model_config.json")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--save_json", default=None, help="Optional output JSON path")
    return p.parse_args()


def topk(labels: List[str], probs: np.ndarray, k: int) -> List[Tuple[str, float]]:
    k = max(1, min(k, len(labels)))
    idx = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idx]


def main() -> None:
    args = parse_args()
    model_path = os.path.join(args.model_dir, "model.joblib")
    cfg_path = os.path.join(args.model_dir, "model_config.json")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Missing model: {model_path}")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        model_cfg = json.load(f)

    labels = model_cfg["labels"]
    feat_cfg = IntervalFeatureConfig(**model_cfg["feature_config"])
    cache_dir = args.cache_dir or model_cfg.get("cache_dir") or os.path.join(ROOT, "data", "cache")

    model = joblib.load(model_path)
    feat = extract_track_feature(
        track_path=os.path.abspath(args.audio),
        cfg=feat_cfg,
        mode="inference",
        seed=args.seed,
        cache_dir=cache_dir,
    )
    feat = feat.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(feat)[0]
    else:
        scores = model.decision_function(feat)[0]
        scores = scores - np.max(scores)
        probs = np.exp(scores) / (np.sum(np.exp(scores)) + 1e-9)

    pred_idx = int(np.argmax(probs))
    pred_label = labels[pred_idx]
    ranking = topk(labels, probs, args.top_k)

    print(f"Predicted Dastgah: {pred_label}")
    for i, (label, p) in enumerate(ranking, start=1):
        print(f"{i}. {label}: {p:.4f}")

    out = {
        "audio": os.path.abspath(args.audio),
        "prediction": pred_label,
        "top_k": [{"label": l, "prob": p} for l, p in ranking],
    }
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
