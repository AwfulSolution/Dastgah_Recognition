import hashlib
import os
from typing import Optional

import numpy as np


FEATURE_VERSION = "v3_melodic_1"


# Bump manually if corpus audio is ever actually replaced (new rips/re-encodes
# at existing paths). Adding NEW files needs no bump: new paths, fresh keys.
CORPUS_VERSION = "corpus1"


def _file_sig(path: str) -> str:
    # Identity = path + manual corpus version. Every file-derived signal was
    # tried and defeated by a background scanner that continuously rewrites
    # the corpus in place (mtimes at any precision, whole-file hashes, even
    # tag-skipping audio-region hashes — the inter-frame junk differs between
    # its file versions). Meanwhile the DECODED AUDIO provably never changes:
    # three full re-extractions reproduced identical CV results. So file
    # contents are treated as frozen per path, and cache busting is an
    # explicit human decision via CORPUS_VERSION.
    return CORPUS_VERSION


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
