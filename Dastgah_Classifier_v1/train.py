import argparse
import json
import os
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from src.dastgah.data import LABELS, Track, ensure_manifest_and_splits
from src.dastgah.dataset import DastgahDataset, SegmentConfig
from src.dastgah.features import FeatureConfig
from src.dastgah.model import SmallCnn
from src.dastgah.model_config import save_model_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to parent data folder")
    parser.add_argument("--manifest", default=os.path.join(BASE_DIR, "data", "manifest.json"))
    parser.add_argument("--splits", default=os.path.join(BASE_DIR, "data", "splits.json"))
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--use_augment", action="store_true")
    parser.add_argument("--time_mask", type=int, default=16)
    parser.add_argument("--freq_mask", type=int, default=8)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--use_mode_features", action="store_true")
    parser.add_argument("--mode_pitch_bins", type=int, default=24)
    parser.add_argument("--num_train_segments", type=int, default=4)
    parser.add_argument("--num_eval_segments", type=int, default=10)
    parser.add_argument("--cache_mel", action="store_true")
    parser.add_argument("--cache_dir", default=os.path.join(BASE_DIR, "data", "cache"))
    parser.add_argument("--early_stopping_patience", type=int, default=6)
    parser.add_argument("--early_stopping_min_delta", type=float, default=1e-4)
    parser.add_argument("--lr_patience", type=int, default=2)
    parser.add_argument("--lr_factor", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--rebuild_manifest", action="store_true")
    parser.add_argument("--rebuild_splits", action="store_true")
    parser.add_argument(
        "--device",
        default="mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"),
    )
    parser.add_argument("--run_dir", default=os.path.join(BASE_DIR, "runs", "exp_torch"))
    return parser.parse_args()


def select_tracks(tracks: List[Track], indices: List[int]) -> List[Track]:
    return [tracks[i] for i in indices]


def compute_class_weights(tracks: List[Track]) -> torch.Tensor:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for track in tracks:
        counts[label_to_idx[track.label]] += 1.0
    counts[counts == 0.0] = 1.0
    weights = counts.sum() / (len(LABELS) * counts)
    return torch.tensor(weights, dtype=torch.float32)


def make_weighted_sampler(dataset: DastgahDataset) -> WeightedRandomSampler:
    label_to_idx = dataset.label_map
    track_label_idx = [label_to_idx[track.label] for track in dataset.tracks]
    counts = np.zeros(len(LABELS), dtype=np.float64)
    for lbl in track_label_idx:
        counts[lbl] += 1.0
    counts[counts == 0.0] = 1.0

    weights = []
    for ds_idx in range(len(dataset)):
        track_idx = ds_idx // dataset.track_segments
        lbl = track_label_idx[track_idx]
        weights.append(1.0 / counts[lbl])
    weights_t = torch.tensor(weights, dtype=torch.double)
    return WeightedRandomSampler(weights=weights_t, num_samples=len(weights), replacement=True)


def trackwise_metrics_from_probs(per_track_probs: Dict[int, List[np.ndarray]], per_track_true: Dict[int, int]) -> Dict[str, float]:
    ordered = sorted(per_track_probs.keys())
    y_true = np.array([per_track_true[idx] for idx in ordered], dtype=np.int64)
    y_pred = np.array([int(np.argmax(np.mean(per_track_probs[idx], axis=0))) for idx in ordered], dtype=np.int64)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def evaluate_trackwise(model: nn.Module, loader: DataLoader, device: str) -> Dict[str, float]:
    model.eval()
    per_track_probs: Dict[int, List[np.ndarray]] = {}
    per_track_true: Dict[int, int] = {}

    with torch.no_grad():
        for x, y, track_idx in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            labels = y.numpy()
            track_ids = track_idx.numpy()

            for prob, label, tid in zip(probs, labels, track_ids):
                tid_int = int(tid)
                per_track_probs.setdefault(tid_int, []).append(prob)
                per_track_true[tid_int] = int(label)

    return trackwise_metrics_from_probs(per_track_probs, per_track_true)


def main() -> None:
    args = parse_args()
    os.makedirs(args.run_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state = ensure_manifest_and_splits(
        data_root=args.data,
        manifest_path=args.manifest,
        splits_path=args.splits,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        rebuild_manifest=args.rebuild_manifest,
        rebuild_splits=args.rebuild_splits,
    )
    tracks = state.tracks
    splits = state.splits
    if state.rebuilt_manifest:
        print(f"Manifest rebuilt: {args.manifest}")
    if state.rebuilt_splits:
        print(f"Splits rebuilt: {args.splits}")

    feature_cfg = FeatureConfig()
    seg_cfg = SegmentConfig(
        segment_seconds=args.segment_seconds,
        use_augment=args.use_augment,
        time_mask_param=args.time_mask,
        freq_mask_param=args.freq_mask,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
        use_mode_features=args.use_mode_features,
        mode_pitch_bins=args.mode_pitch_bins,
        num_train_segments=args.num_train_segments,
        num_eval_segments=args.num_eval_segments,
        cache_mel=args.cache_mel,
        cache_dir=args.cache_dir,
    )

    train_tracks = select_tracks(tracks, splits["train"])
    val_tracks = select_tracks(tracks, splits["val"])
    test_tracks = select_tracks(tracks, splits["test"])

    train_set = DastgahDataset(train_tracks, feature_cfg, seg_cfg, "train", args.seed, return_track_id=False)
    val_set = DastgahDataset(val_tracks, feature_cfg, seg_cfg, "val", args.seed, return_track_id=True)
    test_set = DastgahDataset(test_tracks, feature_cfg, seg_cfg, "test", args.seed, return_track_id=True)

    sampler = make_weighted_sampler(train_set)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = SmallCnn(num_classes=len(LABELS)).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    class_weights = compute_class_weights(train_tracks).to(args.device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=args.lr_factor,
        patience=args.lr_patience,
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_acc": [],
        "val_macro_f1": [],
        "val_balanced_acc": [],
        "lr": [],
    }

    best_val_f1 = -1.0
    best_epoch = 0
    best_state = None
    no_improve = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_count = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}"):
            x = x.to(args.device)
            y = y.to(args.device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            if args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()

            running_loss += float(loss.item()) * x.size(0)
            running_correct += int((torch.argmax(logits, dim=1) == y).sum().item())
            running_count += x.size(0)

        train_loss = running_loss / max(running_count, 1)
        train_acc = running_correct / max(running_count, 1)
        val_metrics = evaluate_trackwise(model, val_loader, args.device)
        scheduler.step(val_metrics["macro_f1"])

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_metrics["accuracy"])
        history["val_macro_f1"].append(val_metrics["macro_f1"])
        history["val_balanced_acc"].append(val_metrics["balanced_accuracy"])
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_acc={val_metrics['accuracy']:.3f} "
            f"val_macro_f1={val_metrics['macro_f1']:.3f} "
            f"val_bal_acc={val_metrics['balanced_accuracy']:.3f}"
        )

        if val_metrics["macro_f1"] > best_val_f1 + args.early_stopping_min_delta:
            best_val_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = {"accuracy": 0.0, "macro_f1": 0.0, "balanced_accuracy": 0.0}
    if len(test_set) > 0:
        test_metrics = evaluate_trackwise(model, test_loader, args.device)
        print(
            "Test metrics: "
            f"acc={test_metrics['accuracy']:.3f} "
            f"macro_f1={test_metrics['macro_f1']:.3f} "
            f"bal_acc={test_metrics['balanced_accuracy']:.3f}"
        )

    torch.save(model.state_dict(), os.path.join(args.run_dir, "model.pt"))

    metrics_payload = {
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "val_acc": history["val_acc"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "val_macro_f1": history["val_macro_f1"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "val_balanced_acc": history["val_balanced_acc"][best_epoch - 1] if best_epoch > 0 else 0.0,
        "test_acc": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "test_balanced_acc": test_metrics["balanced_accuracy"],
        "num_eval_segments": args.num_eval_segments,
        "num_train_segments": args.num_train_segments,
        "segment_seconds": args.segment_seconds,
        "trim_silence": args.trim_silence,
        "trim_db": args.trim_db,
        "use_mode_features": args.use_mode_features,
        "mode_pitch_bins": args.mode_pitch_bins,
    }

    with open(os.path.join(args.run_dir, "history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    save_model_config(
        args.run_dir,
        {
            "model_type": "torch_cnn",
            "segment_seconds": args.segment_seconds,
            "num_segments": args.num_eval_segments,
            "trim_silence": args.trim_silence,
            "trim_db": args.trim_db,
            "use_mode_features": args.use_mode_features,
            "mode_pitch_bins": args.mode_pitch_bins,
            "num_train_segments": args.num_train_segments,
            "num_eval_segments": args.num_eval_segments,
        },
    )


if __name__ == "__main__":
    main()
