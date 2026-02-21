import hashlib
import os
from typing import Optional

import numpy as np


FEATURE_VERSION = "v2_interval_2"


def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{st.st_mtime_ns}-{st.st_size}"
    except FileNotFoundError:
        return "missing"


def cache_key(path: str, cfg_sig: str, suffix: str) -> str:
    payload = f"{FEATURE_VERSION}|{path}|{_file_sig(path)}|{cfg_sig}|{suffix}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def cache_path(cache_dir: str, path: str, cfg_sig: str, suffix: str) -> str:
    return os.path.join(cache_dir, f"{cache_key(path, cfg_sig, suffix)}.npz")


def load_cached_track_features(
    cache_dir: str,
    path: str,
    cfg_sig: str,
    suffix: str,
) -> Optional[np.ndarray]:
    p = cache_path(cache_dir, path, cfg_sig, suffix)
    if not os.path.exists(p):
        return None
    return np.load(p)["features"]


def save_cached_track_features(
    cache_dir: str,
    path: str,
    cfg_sig: str,
    suffix: str,
    features: np.ndarray,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path(cache_dir, path, cfg_sig, suffix)
    np.savez_compressed(p, features=features)
