import argparse
import os
from typing import Dict, List

import joblib
import librosa
import numpy as np

from src.dastgah.data import LABELS
from src.dastgah.features import FeatureConfig, load_audio
from src.dastgah.model_config import config_path_for_model, load_model_config
from src.dastgah.scikit_features import (
    ScikitFeatureOpts,
    extract_features,
    file_seed,
    segment_starts,
)


def parse_args() -> tuple[argparse.Namespace, argparse.ArgumentParser]:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--input", required=True, help="Path to .mp3 or a folder of .mp3 files")
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--use_mode_features", action="store_true")
    parser.add_argument("--mode_pitch_bins", type=int, default=24)
    parser.add_argument("--low_compute", action="store_true")
    parser.add_argument("--config", default=None, help="Optional explicit path to model_config.json")
    parser.add_argument("--no_auto_config", action="store_true", help="Disable loading model_config.json")
    return parser.parse_args(), parser


def apply_low_compute(args: argparse.Namespace) -> None:
    if not args.low_compute:
        return
    if args.segment_seconds == 45.0:
        args.segment_seconds = 25.0
    if args.num_segments == 10:
        args.num_segments = 4
    if args.trim_db == 25:
        args.trim_db = 30
    if args.mode_pitch_bins == 24:
        args.mode_pitch_bins = 16


def apply_config_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser, config: Dict[str, object]) -> None:
    keys = [
        "segment_seconds",
        "num_segments",
        "trim_silence",
        "trim_db",
        "use_mode_features",
        "mode_pitch_bins",
    ]
    for key in keys:
        if key not in config:
            continue
        if getattr(args, key) == parser.get_default(key):
            setattr(args, key, config[key])


def list_inputs(path: str) -> List[str]:
    if os.path.isdir(path):
        files = [
            os.path.join(path, f)
            for f in os.listdir(path)
            if f.lower().endswith(".mp3")
        ]
        return sorted(files)
    return [path]


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

    apply_low_compute(args)

    model = joblib.load(args.model)
    cfg = FeatureConfig()
    opts = ScikitFeatureOpts(
        use_mode_features=args.use_mode_features,
        mode_pitch_bins=args.mode_pitch_bins,
    )

    for file_path in list_inputs(args.input):
        audio = load_audio(file_path, cfg)
        if args.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=args.trim_db)

        starts = segment_starts(
            len(audio),
            cfg.sample_rate,
            args.segment_seconds,
            args.num_segments,
            mode="eval",
            seed=file_seed(file_path, args.seed),
        )
        seg_len = int(args.segment_seconds * cfg.sample_rate)
        seg_features = []
        for start in starts:
            segment = audio[start : start + seg_len]
            seg_features.append(extract_features(segment, cfg, opts))
        seg_features = np.vstack(seg_features)

        probs = model.predict_proba(seg_features).mean(axis=0)
        top_idx = int(np.argmax(probs))
        label = LABELS[top_idx]
        confidence = probs[top_idx]
        print(f"{os.path.basename(file_path)}\t{label}\t{confidence:.3f}")


if __name__ == "__main__":
    main()
