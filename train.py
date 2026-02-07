import argparse
import json
import os
from typing import List

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dastgah.data import (
    LABELS,
    Track,
    build_manifest,
    build_splits,
    load_manifest,
    load_splits,
    save_manifest,
    save_splits,
)
from src.dastgah.dataset import DastgahDataset, SegmentConfig
from src.dastgah.features import FeatureConfig
from src.dastgah.model import SmallCnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to parent data folder")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_augment", action="store_true")
    parser.add_argument("--time_mask", type=int, default=16)
    parser.add_argument("--freq_mask", type=int, default=8)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--num_train_segments", type=int, default=4)
    parser.add_argument("--cache_mel", action="store_true")
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--run_dir", default="runs/exp1")
    return parser.parse_args()


def select_tracks(tracks: List[Track], indices: List[int]) -> List[Track]:
    return [tracks[i] for i in indices]


def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == targets).float().mean().item()


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    model.eval()
    total = 0.0
    count = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            total += accuracy(logits, y) * x.size(0)
            count += x.size(0)
    return total / max(count, 1)


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)

    if os.path.exists(args.manifest):
        tracks = load_manifest(args.manifest)
    else:
        tracks = build_manifest(args.data)
        os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
        save_manifest(tracks, args.manifest)

    if os.path.exists(args.splits):
        splits = load_splits(args.splits)
    else:
        splits = build_splits(tracks, args.val_split, args.test_split, args.seed)
        os.makedirs(os.path.dirname(args.splits), exist_ok=True)
        save_splits(splits, args.splits)

    feature_cfg = FeatureConfig()
    seg_cfg = SegmentConfig(
        segment_seconds=args.segment_seconds,
        use_augment=args.use_augment,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
        num_train_segments=args.num_train_segments,
        cache_mel=args.cache_mel,
        cache_dir=args.cache_dir,
    )

    train_set = DastgahDataset(select_tracks(tracks, splits["train"]), feature_cfg, seg_cfg, "train", args.seed)
    val_set = DastgahDataset(select_tracks(tracks, splits["val"]), feature_cfg, seg_cfg, "val", args.seed)
    test_set = DastgahDataset(select_tracks(tracks, splits["test"]), feature_cfg, seg_cfg, "test", args.seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallCnn(num_classes=len(LABELS)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_acc": [], "val_acc": []}

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_acc = 0.0
        count = 0
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            epoch_acc += accuracy(logits, y) * x.size(0)
            count += x.size(0)

        train_acc = epoch_acc / max(count, 1)
        val_acc = evaluate(model, val_loader, args.device)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(f"Epoch {epoch}: train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

    test_acc = evaluate(model, test_loader, args.device) if len(test_set) > 0 else 0.0
    print(f"Test accuracy: {test_acc:.3f}")

    torch.save(model.state_dict(), os.path.join(args.run_dir, "model.pt"))
    with open(os.path.join(args.run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


if __name__ == "__main__":
    main()
