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
