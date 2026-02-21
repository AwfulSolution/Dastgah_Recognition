import argparse
import os
from typing import List

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score

from src.dastgah.data import LABELS, Track, load_manifest, load_splits
from src.dastgah.features import FeatureConfig, load_audio, melspec
from src.dastgah.model_config import config_path_for_model, load_model_config
from src.dastgah.model import SmallCnn
from src.dastgah.scikit_features import extract_mode_features

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=os.path.join(BASE_DIR, "data", "manifest.json"))
    parser.add_argument("--splits", default=os.path.join(BASE_DIR, "data", "splits.json"))
    parser.add_argument("--model", required=True, help="Path to model.pt")
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--use_mode_features", action="store_true")
    parser.add_argument("--mode_pitch_bins", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--config", default=None, help="Optional explicit path to model_config.json")
    parser.add_argument("--no_auto_config", action="store_true", help="Disable loading model_config.json")
    return parser.parse_args(), parser


def apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser, config: dict) -> None:
    for key in ["segment_seconds", "num_segments", "trim_silence", "trim_db", "use_mode_features", "mode_pitch_bins"]:
        if key in config and getattr(args, key) == parser.get_default(key):
            setattr(args, key, config[key])


def segment_starts(audio_len: int, sr: int, seconds: float, num_segments: int) -> List[int]:
    seg_len = int(seconds * sr)
    if audio_len <= seg_len:
        return [0]
    max_start = audio_len - seg_len
    if num_segments <= 1:
        return [max_start // 2]
    return np.linspace(0, max_start, num_segments).astype(int).tolist()


def predict_track(model: SmallCnn, track: Track, cfg: FeatureConfig, args: argparse.Namespace) -> int:
    audio = load_audio(track.path, cfg)
    if args.trim_silence:
        import librosa
        audio, _ = librosa.effects.trim(audio, top_db=args.trim_db)
    mode_vec = None
    if args.use_mode_features:
        mode_vec = extract_mode_features(audio, cfg, args.mode_pitch_bins).astype(np.float32)
        mode_vec = (mode_vec - mode_vec.mean()) / (mode_vec.std() + 1e-6)
    starts = segment_starts(len(audio), cfg.sample_rate, args.segment_seconds, args.num_segments)
    seg_len = int(args.segment_seconds * cfg.sample_rate)
    probs = []
    for start in starts:
        segment = audio[start : start + seg_len]
        mel = melspec(segment, cfg)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if mode_vec is not None:
            mode_map = np.repeat(mode_vec[:, None], mel.shape[1], axis=1).astype(np.float32)
            mel = np.concatenate([mel, mode_map], axis=0)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(args.device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    avg_prob = np.mean(probs, axis=0)
    return int(np.argmax(avg_prob))


def main() -> None:
    args, parser = parse_args()
    if not args.no_auto_config:
        cfg_path = args.config if args.config else config_path_for_model(args.model)
        if os.path.exists(cfg_path):
            cfg = load_model_config(args.model) if args.config is None else None
            if args.config is not None:
                import json

                with open(args.config, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
            if cfg is not None:
                apply_config_defaults(args, parser, cfg)
                print(f"Loaded config: {cfg_path}")
    tracks = load_manifest(args.manifest)
    splits = load_splits(args.splits)
    cfg = FeatureConfig()

    model = SmallCnn(num_classes=len(LABELS))
    model.load_state_dict(torch.load(args.model, map_location=args.device))
    model.to(args.device)
    model.eval()

    for split_name in ["val", "test"]:
        indices = splits.get(split_name, [])
        if not indices:
            continue
        subset = [tracks[i] for i in indices]
        preds = []
        labels = []
        label_to_idx = {label: i for i, label in enumerate(LABELS)}
        for track in subset:
            preds.append(predict_track(model, track, cfg, args))
            labels.append(label_to_idx[track.label])
        y_true = np.array(labels, dtype=np.int64)
        y_pred = np.array(preds, dtype=np.int64)
        acc = float((y_pred == y_true).mean())
        macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
        print(
            f"{split_name} metrics: "
            f"acc={acc:.3f} "
            f"macro_f1={macro_f1:.3f} "
            f"bal_acc={bal_acc:.3f}"
        )


if __name__ == "__main__":
    main()
