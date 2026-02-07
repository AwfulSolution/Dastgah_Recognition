import argparse
import hashlib
import os
from typing import List

import joblib
import numpy as np
import librosa

from src.dastgah.data import LABELS
from src.dastgah.features import FeatureConfig, load_audio, melspec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to model.joblib")
    parser.add_argument("--input", required=True, help="Path to .mp3 or a folder of .mp3 files")
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    return parser.parse_args()


def file_seed(path: str, base_seed: int) -> int:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base_seed) % (2**31 - 1)


def segment_starts(
    audio_len: int,
    sr: int,
    seconds: float,
    num_segments: int,
    seed: int,
) -> List[int]:
    seg_len = int(seconds * sr)
    if audio_len <= seg_len:
        return [0]

    max_start = audio_len - seg_len
    if num_segments <= 1:
        return [max_start // 2]

    return np.linspace(0, max_start, num_segments).astype(int).tolist()


def extract_features(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mel = melspec(audio, cfg)
    mel_mean = np.mean(mel, axis=1)
    mel_std = np.std(mel, axis=1)

    mfcc_feat = librosa.feature.mfcc(
        y=audio,
        sr=cfg.sample_rate,
        n_mfcc=20,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    mfcc_delta = librosa.feature.delta(mfcc_feat)

    chroma = librosa.feature.chroma_stft(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    contrast = librosa.feature.spectral_contrast(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    harm = librosa.effects.harmonic(y=audio)
    tonnetz = librosa.feature.tonnetz(y=harm, sr=cfg.sample_rate)

    def stats(x: np.ndarray) -> np.ndarray:
        return np.concatenate([x.mean(axis=1), x.std(axis=1)], axis=0)

    return np.concatenate(
        [
            mel_mean,
            mel_std,
            stats(mfcc_feat),
            stats(mfcc_delta),
            stats(chroma),
            stats(contrast),
            stats(tonnetz),
        ],
        axis=0,
    ).astype(np.float32)


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
    args = parse_args()
    model = joblib.load(args.model)
    cfg = FeatureConfig()

    for file_path in list_inputs(args.input):
        audio = load_audio(file_path, cfg)
        if args.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=args.trim_db)
        starts = segment_starts(
            len(audio),
            cfg.sample_rate,
            args.segment_seconds,
            args.num_segments,
            seed=file_seed(file_path, args.seed),
        )
        seg_len = int(args.segment_seconds * cfg.sample_rate)
        seg_features = []
        for start in starts:
            segment = audio[start : start + seg_len]
            seg_features.append(extract_features(segment, cfg))
        seg_features = np.vstack(seg_features)
        probs = model.predict_proba(seg_features).mean(axis=0)
        top_idx = int(np.argmax(probs))
        label = LABELS[top_idx]
        confidence = probs[top_idx]
        print(f"{os.path.basename(file_path)}\t{label}\t{confidence:.3f}")


if __name__ == "__main__":
    main()
