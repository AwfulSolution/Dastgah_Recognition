import hashlib
import os
from typing import Optional

import numpy as np


FEATURE_VERSION = "v3_melodic_1"


_FP_CHUNK = 65536
_fp_cache: dict = {}


def _file_sig(path: str) -> str:
    # Content fingerprint (size + md5 of first/last 64KB), NOT mtime at any
    # precision: sync/backup tooling on the training machine rewrote mtimes
    # corpus-wide twice in two days (ns truncation 2026-07-08, then wholesale
    # date changes after a power loss on 2026-07-09), orphaning every
    # mtime-keyed cache entry each time. Timestamps are metadata theater;
    # content is the identity. ~1ms per file, memoized per process.
    cached = _fp_cache.get(path)
    if cached is not None:
        return cached
    try:
        st = os.stat(path)
        h = hashlib.md5()
        with open(path, "rb") as f:
            h.update(f.read(_FP_CHUNK))
            if st.st_size > _FP_CHUNK * 2:
                f.seek(-_FP_CHUNK, os.SEEK_END)
                h.update(f.read(_FP_CHUNK))
        sig = f"{st.st_size}-{h.hexdigest()[:16]}"
    except (FileNotFoundError, OSError):
        return "missing"
    _fp_cache[path] = sig
    return sig


def cache_key(path: str, cfg_sig: str, suffix: str) -> str:
    payload = f"{FEATURE_VERSION}|{path}|{_file_sig(path)}|{cfg_sig}|{suffix}"
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def cache_path(cache_dir: str, path: str, cfg_sig: str, suffix: str) -> str:
    return os.path.join(cache_dir, f"{cache_key(path, cfg_sig, suffix)}.npz")


def load_cached_track_features(cache_dir: str, path: str, cfg_sig: str, suffix: str) -> Optional[np.ndarray]:
    p = cache_path(cache_dir, path, cfg_sig, suffix)
    if not os.path.exists(p):
        return None
    return np.load(p)["features"]


def save_cached_track_features(cache_dir: str, path: str, cfg_sig: str, suffix: str, features: np.ndarray) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path(cache_dir, path, cfg_sig, suffix)
    np.savez_compressed(p, features=features)


def load_cached_track_notes(cache_dir: str, path: str, notes_sig: str, suffix: str) -> Optional[dict]:
    """Load the intermediate note-event representation (the expensive pyin/hpss product)."""
    p = cache_path(cache_dir, path, notes_sig, suffix)
    if not os.path.exists(p):
        return None
    with np.load(p) as data:
        return {key: data[key] for key in data.files}


def save_cached_track_notes(cache_dir: str, path: str, notes_sig: str, suffix: str, arrays: dict) -> None:
    os.makedirs(cache_dir, exist_ok=True)
    p = cache_path(cache_dir, path, notes_sig, suffix)
    np.savez_compressed(p, **arrays)
