import argparse
import json
import os
from typing import Dict, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def read_metrics(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default=os.path.join(BASE_DIR, "runs"))
    parser.add_argument("--out", default=os.path.join(BASE_DIR, "runs", "compare_models.md"))
    args = parser.parse_args()

    entries = {
        "LR (scikit)": os.path.join(args.runs, "exp_sklearn_v2", "metrics.json"),
        "SVM": os.path.join(args.runs, "exp_svm", "metrics.json"),
        "Ensemble": os.path.join(args.runs, "exp_ensemble", "metrics.json"),
    }

    lines = ["# Model Comparison", ""]
    lines.append("| Model | Val Acc | Notes |")
    lines.append("|---|---:|---|")

    for name, metric_path in entries.items():
        metrics = read_metrics(metric_path)
        val_acc = metrics.get("val_acc") if metrics else None
        val_text = f"{val_acc:.3f}" if isinstance(val_acc, (int, float)) else "N/A"
        note = "" if metrics else f"Missing: {metric_path}"
        lines.append(f"| {name} | {val_text} | {note} |")

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
