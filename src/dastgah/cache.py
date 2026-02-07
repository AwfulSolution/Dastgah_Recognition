import hashlib
import os
from typing import Optional

import numpy as np


FEATURE_VERSION = "v2"


def _cache_key(
    path: str,
    segment_seconds: float,
    num_segments: int,
    seed: int,
    mode: str,
    cfg_sig: str,
) -> str:
    try:
        stat = os.stat(path)
        file_sig = f"{stat.st_mtime_ns}-{stat.st_size}"
    except FileNotFoundError:
        file_sig = "missing"
    payload = f"{FEATURE_VERSION}|{path}|{file_sig}|{segment_seconds}|{num_segments}|{seed}|{mode}|{cfg_sig}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def cache_path(
    cache_dir: str,
    path: str,
    segment_seconds: float,
    num_segments: int,
    seed: int,
    mode: str,
    cfg_sig: str,
) -> str:
    key = _cache_key(path, segment_seconds, num_segments, seed, mode, cfg_sig)
    return os.path.join(cache_dir, f"{key}.npz")


def load_cached_features(
    cache_dir: str,
    path: str,
    segment_seconds: float,
    num_segments: int,
    seed: int,
    mode: str,
    cfg_sig: str,
) -> Optional[np.ndarray]:
    cache_file = cache_path(cache_dir, path, segment_seconds, num_segments, seed, mode, cfg_sig)
    if not os.path.exists(cache_file):
        return None
    data = np.load(cache_file)
    return data["features"]


def save_cached_features(
    cache_dir: str,
    path: str,
    segment_seconds: float,
    num_segments: int,
    seed: int,
    mode: str,
    cfg_sig: str,
    features: np.ndarray,
) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = cache_path(cache_dir, path, segment_seconds, num_segments, seed, mode, cfg_sig)
    np.savez_compressed(cache_file, features=features)
