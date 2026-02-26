import argparse
import json
import os
from typing import Dict, List, Optional


ROOT = os.path.dirname(os.path.abspath(__file__))


def read_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def collect_runs(runs_dir: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.isdir(runs_dir):
        return rows
    for name in sorted(os.listdir(runs_dir)):
        run_dir = os.path.join(runs_dir, name)
        if not os.path.isdir(run_dir):
            continue
        metrics = read_json(os.path.join(run_dir, "metrics.json"))
        model_path = os.path.join(run_dir, "model.joblib")
        cfg_path = os.path.join(run_dir, "model_config.json")
        if metrics is None or (not os.path.exists(model_path)) or (not os.path.exists(cfg_path)):
            continue
        val = metrics.get("val", {})
        test = metrics.get("test", {})
        rows.append(
            {
                "run": name,
                "model_type": metrics.get("model_type", "n/a"),
                "val_acc": val.get("acc"),
                "val_macro_f1": val.get("macro_f1"),
                "val_bal_acc": val.get("balanced_acc"),
                "test_acc": test.get("acc"),
                "test_macro_f1": test.get("macro_f1"),
                "test_bal_acc": test.get("balanced_acc"),
            }
        )
    return rows


def fmt(v: Optional[float]) -> str:
    if isinstance(v, (int, float)):
        return f"{v:.3f}"
    return "N/A"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "compare_models_v2.md"))
    p.add_argument("--sort_by", choices=["test_macro_f1", "test_acc", "val_macro_f1"], default="test_macro_f1")
    args = p.parse_args()

    rows = collect_runs(args.runs)
    rows.sort(key=lambda x: (x.get(args.sort_by) is not None, x.get(args.sort_by, -1.0)), reverse=True)

    lines = ["# v2 Model Comparison", ""]
    lines.append("| Run | Model | Val Acc | Val F1 | Val Bal | Test Acc | Test F1 | Test Bal |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        lines.append(
            f"| {r['run']} | {r['model_type']} | {fmt(r['val_acc'])} | {fmt(r['val_macro_f1'])} | {fmt(r['val_bal_acc'])} | "
            f"{fmt(r['test_acc'])} | {fmt(r['test_macro_f1'])} | {fmt(r['test_bal_acc'])} |"
        )

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
