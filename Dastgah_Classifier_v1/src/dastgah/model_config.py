import json
import os
from typing import Any, Dict, Optional


CONFIG_FILENAME = "model_config.json"


def config_path_for_model(model_path: str) -> str:
    return os.path.join(os.path.dirname(model_path), CONFIG_FILENAME)


def load_model_config(model_path: str) -> Optional[Dict[str, Any]]:
    config_path = config_path_for_model(model_path)
    if not os.path.exists(config_path):
        return None
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_model_config(run_dir: str, payload: Dict[str, Any]) -> str:
    os.makedirs(run_dir, exist_ok=True)
    path = os.path.join(run_dir, CONFIG_FILENAME)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path
