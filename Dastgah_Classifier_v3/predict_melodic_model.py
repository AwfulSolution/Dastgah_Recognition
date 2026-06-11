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

from dastgah_v3.melodic_features import MelodicFeatureConfig, extract_track_feature  # noqa: E402
from dastgah_v3.paths import resolve_config_path  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--model_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--save_json", default=None)
    return p.parse_args()


def topk(labels: List[str], probs: np.ndarray, k: int) -> List[Tuple[str, float]]:
    k = max(1, min(k, len(labels)))
    idx = np.argsort(-probs)[:k]
    return [(labels[i], float(probs[i])) for i in idx]


def probabilities(model, feat: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(feat)[0]
    scores = model.decision_function(feat)[0]
    scores = scores - np.max(scores)
    return np.exp(scores) / (np.sum(np.exp(scores)) + 1e-9)


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
    feat_cfg = MelodicFeatureConfig(**model_cfg["feature_config"])
    default_cache_dir = os.path.join(ROOT, "data", "cache")
    cache_dir = args.cache_dir or resolve_config_path(model_cfg.get("cache_dir"), ROOT, default_cache_dir)

    model = joblib.load(model_path)
    feat = extract_track_feature(os.path.abspath(args.audio), feat_cfg, "inference", args.seed, cache_dir).reshape(1, -1)
    probs = probabilities(model, feat)
    pred_idx = int(np.argmax(probs))
    ranking = topk(labels, probs, args.top_k)

    print(f"Predicted Dastgah: {labels[pred_idx]}")
    for i, (label, p) in enumerate(ranking, start=1):
        print(f"{i}. {label}: {p:.4f}")

    out = {
        "audio": os.path.abspath(args.audio),
        "prediction": labels[pred_idx],
        "top_k": [{"label": label, "prob": prob} for label, prob in ranking],
    }
    if args.save_json:
        with open(args.save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        print(f"Saved: {args.save_json}")


if __name__ == "__main__":
    main()
