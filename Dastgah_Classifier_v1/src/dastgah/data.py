import json
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

from sklearn.model_selection import train_test_split


LABELS = [
    "Chahargah",
    "Homayun",
    "Mahur",
    "Nava",
    "Segah",
    "Shur",
]


@dataclass
class Track:
    path: str
    label: str


def build_manifest(data_root: str) -> List[Track]:
    tracks: List[Track] = []
    for label in LABELS:
        class_dir = os.path.join(data_root, label)
        if not os.path.isdir(class_dir):
            raise FileNotFoundError(f"Missing folder: {class_dir}")
        for name in sorted(os.listdir(class_dir)):
            if not name.lower().endswith(".mp3"):
                continue
            tracks.append(Track(path=os.path.join(class_dir, name), label=label))
    if not tracks:
        raise RuntimeError("No .mp3 files found in the labeled folders.")
    return tracks


def save_manifest(tracks: List[Track], path: str) -> None:
    payload = [track.__dict__ for track in tracks]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_manifest(path: str) -> List[Track]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Track(**item) for item in data]


def build_splits(
    tracks: List[Track],
    val_split: float,
    test_split: float,
    seed: int,
) -> Dict[str, List[int]]:
    if val_split + test_split >= 1.0:
        raise ValueError("val_split + test_split must be < 1.0")

    labels = [t.label for t in tracks]
    indices = list(range(len(tracks)))

    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        indices, labels, test_size=val_split + test_split, stratify=labels, random_state=seed
    )

    if test_split == 0:
        return {"train": train_idx, "val": temp_idx, "test": []}

    val_size = val_split / (val_split + test_split)
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=1 - val_size, stratify=temp_labels, random_state=seed
    )

    return {"train": train_idx, "val": val_idx, "test": test_idx}


def save_splits(splits: Dict[str, List[int]], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(splits, f, indent=2)


def load_splits(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def label_to_index() -> Dict[str, int]:
    return {label: i for i, label in enumerate(LABELS)}


def index_to_label() -> Dict[int, str]:
    return {i: label for i, label in enumerate(LABELS)}


@dataclass
class DataState:
    tracks: List[Track]
    splits: Dict[str, List[int]]
    rebuilt_manifest: bool
    rebuilt_splits: bool


def _manifest_signature(tracks: List[Track]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted((t.path, t.label) for t in tracks))


def _splits_are_valid(splits: Dict[str, List[int]], n_tracks: int) -> bool:
    required = {"train", "val", "test"}
    if set(splits.keys()) != required:
        return False

    all_indices: List[int] = []
    for key in ["train", "val", "test"]:
        idxs = splits[key]
        if not isinstance(idxs, list):
            return False
        if any((not isinstance(i, int)) or i < 0 or i >= n_tracks for i in idxs):
            return False
        all_indices.extend(idxs)

    unique = set(all_indices)
    if len(unique) != n_tracks:
        return False
    return len(all_indices) == n_tracks


def _ensure_parent_dir(path: str) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


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
    current_sig = _manifest_signature(current_tracks)

    if rebuild_manifest or not os.path.exists(manifest_path):
        tracks = current_tracks
        _ensure_parent_dir(manifest_path)
        save_manifest(tracks, manifest_path)
        manifest_changed = True
    else:
        stored_tracks = load_manifest(manifest_path)
        stored_sig = _manifest_signature(stored_tracks)
        if stored_sig != current_sig:
            tracks = current_tracks
            _ensure_parent_dir(manifest_path)
            save_manifest(tracks, manifest_path)
            manifest_changed = True
        else:
            tracks = stored_tracks

    must_rebuild_splits = rebuild_splits or manifest_changed or not os.path.exists(splits_path)
    if must_rebuild_splits:
        splits = build_splits(tracks, val_split, test_split, seed)
        _ensure_parent_dir(splits_path)
        save_splits(splits, splits_path)
        splits_changed = True
    else:
        splits = load_splits(splits_path)
        if not _splits_are_valid(splits, len(tracks)):
            splits = build_splits(tracks, val_split, test_split, seed)
            _ensure_parent_dir(splits_path)
            save_splits(splits, splits_path)
            splits_changed = True

    return DataState(
        tracks=tracks,
        splits=splits,
        rebuilt_manifest=manifest_changed,
        rebuilt_splits=splits_changed,
    )
