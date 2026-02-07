import argparse
import os
from typing import List

import numpy as np
import torch

from src.dastgah.data import LABELS, Track, load_manifest, load_splits
from src.dastgah.features import FeatureConfig, load_audio, melspec
from src.dastgah.model import SmallCnn


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--model", required=True, help="Path to model.pt")
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


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
    starts = segment_starts(len(audio), cfg.sample_rate, args.segment_seconds, args.num_segments)
    seg_len = int(args.segment_seconds * cfg.sample_rate)
    probs = []
    for start in starts:
        segment = audio[start : start + seg_len]
        mel = melspec(segment, cfg)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(args.device)
        logits = model(x)
        probs.append(torch.softmax(logits, dim=1).cpu().numpy()[0])
    avg_prob = np.mean(probs, axis=0)
    return int(np.argmax(avg_prob))


def main() -> None:
    args = parse_args()
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
        acc = (np.array(preds) == np.array(labels)).mean()
        print(f"{split_name} accuracy: {acc:.3f}")


if __name__ == "__main__":
    main()
