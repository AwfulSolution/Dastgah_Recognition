import argparse
import os
import shutil


ROOT = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", required=True, help="Run directory containing model.joblib + model_config.json")
    p.add_argument("--models_dir", default=os.path.join(ROOT, "models"))
    p.add_argument("--model_name", default="model.joblib")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src_model = os.path.join(args.run_dir, "model.joblib")
    src_cfg = os.path.join(args.run_dir, "model_config.json")
    src_metrics = os.path.join(args.run_dir, "metrics.json")

    if not os.path.exists(src_model):
        raise FileNotFoundError(f"Missing model file: {src_model}")
    if not os.path.exists(src_cfg):
        raise FileNotFoundError(f"Missing model config: {src_cfg}")

    os.makedirs(args.models_dir, exist_ok=True)
    dst_model = os.path.join(args.models_dir, args.model_name)
    dst_cfg = os.path.join(args.models_dir, "model_config.json")
    dst_metrics = os.path.join(args.models_dir, "metrics.json")

    shutil.copy2(src_model, dst_model)
    shutil.copy2(src_cfg, dst_cfg)
    if os.path.exists(src_metrics):
        shutil.copy2(src_metrics, dst_metrics)

    print(f"Promoted model: {dst_model}")
    print(f"Promoted config: {dst_cfg}")
    if os.path.exists(src_metrics):
        print(f"Promoted metrics: {dst_metrics}")


if __name__ == "__main__":
    main()
