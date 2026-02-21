import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split

from . import LABELS


@dataclass
class Track:
    path: str
    label: str


@dataclass
class DataState:
    tracks: List[Track]
    splits: Dict[str, List[int]]
    rebuilt_manifest: bool
    rebuilt_splits: bool


def build_manifest(data_root: str) -> List[Track]:
    tracks: List[Track] = []
    for label in LABELS:
        class_dir = os.path.join(data_root, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing folder: {class_dir}")
        for name in sorted(os.listdir(class_dir)):
            if name.lower().endswith(".mp3"):
                tracks.append(Track(path=os.path.abspath(os.path.join(class_dir, name)), label=label))
    if not tracks:
        raise RuntimeError("No .mp3 files found.")
    return tracks


def save_manifest(tracks: List[Track], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([t.__dict__ for t in tracks], f, indent=2)


def load_manifest(path: str) -> List[Track]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Track(**x) for x in data]


def build_splits(tracks: List[Track], val_split: float, test_split: float, seed: int) -> Dict[str, List[int]]:
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1")

    y = [t.label for t in tracks]
    idx = list(range(len(tracks)))

    train_idx, temp_idx = train_test_split(
        idx,
        test_size=val_split + test_split,
        stratify=y,
        random_state=seed,
    )

    if test_split == 0:
        return {"train": train_idx, "val": temp_idx, "test": []}

    temp_y = [y[i] for i in temp_idx]
    val_size = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx,
        train_size=val_size,
        stratify=temp_y,
        random_state=seed,
    )
    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(splits: Dict[str, List[int]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_index() -> Dict[str, int]:
    return {label: i for i, label in enumerate(LABELS)}


def index_to_label() -> Dict[int, str]:
    return {i: label for i, label in enumerate(LABELS)}


def _manifest_signature(tracks: List[Track]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((t.path, t.label) for t in tracks))


def _splits_valid(splits: Dict[str, List[int]], n_tracks: int) -> bool:
    if set(splits.keys()) != {"train", "val", "test"}:
        return False
    all_idx: List[int] = []
    for key in ("train", "val", "test"):
        vals = splits[key]
        if any((not isinstance(i, int)) or i < 0 or i >= n_tracks for i in vals):
            return False
        all_idx.extend(vals)
    return len(all_idx) == n_tracks and len(set(all_idx)) == n_tracks


def ensure_manifest_and_splits(
    data_root: str,
    manifest_path: str,
    splits_path: str,
    val_split: float,
    test_split: float,
    seed: int,
    rebuild_manifest: bool = False,
    rebuild_splits: bool = False,
) -> DataState:
    manifest_changed = False
    splits_changed = False

    current_tracks = build_manifest(data_root)
    if rebuild_manifest or not os.path.exists(manifest_path):
        tracks = current_tracks
        save_manifest(tracks, manifest_path)
        manifest_changed = True
    else:
        stored = load_manifest(manifest_path)
        if _manifest_signature(stored) != _manifest_signature(current_tracks):
            tracks = current_tracks
            save_manifest(tracks, manifest_path)
            manifest_changed = True
        else:
            tracks = stored

    if rebuild_splits or manifest_changed or not os.path.exists(splits_path):
        splits = build_splits(tracks, val_split, test_split, seed)
        save_splits(splits, splits_path)
        splits_changed = True
    else:
        splits = load_splits(splits_path)
        if not _splits_valid(splits, len(tracks)):
            splits = build_splits(tracks, val_split, test_split, seed)
            save_splits(splits, splits_path)
            splits_changed = True

    return DataState(
        tracks=tracks,
        splits=splits,
        rebuilt_manifest=manifest_changed,
        rebuilt_splits=splits_changed,
    )
