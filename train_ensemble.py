import argparse
import hashlib
import json
import os
from typing import List, Tuple

import numpy as np
import librosa
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

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
from src.dastgah.features import FeatureConfig, load_audio, melspec
from src.dastgah.cache import load_cached_features, save_cached_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to parent data folder")
    parser.add_argument("--manifest", default="data/manifest.json")
    parser.add_argument("--splits", default="data/splits.json")
    parser.add_argument("--segment_seconds", type=float, default=45.0)
    parser.add_argument("--num_segments", type=int, default=10)
    parser.add_argument("--cache_dir", default="data/cache")
    parser.add_argument("--trim_silence", action="store_true")
    parser.add_argument("--trim_db", type=int, default=25)
    parser.add_argument("--val_split", type=float, default=0.15)
    parser.add_argument("--test_split", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_dir", default="runs/exp_ensemble")
    return parser.parse_args()


def select_tracks(tracks: List[Track], indices: List[int]) -> List[Track]:
    return [tracks[i] for i in indices]


def file_seed(path: str, base_seed: int) -> int:
    h = hashlib.md5(path.encode("utf-8")).hexdigest()
    return (int(h[:8], 16) + base_seed) % (2**31 - 1)


def segment_starts(
    audio_len: int,
    sr: int,
    seconds: float,
    num_segments: int,
    mode: str,
    seed: int,
) -> List[int]:
    seg_len = int(seconds * sr)
    if audio_len <= seg_len:
        return [0]

    max_start = audio_len - seg_len
    if num_segments <= 1:
        return [max_start // 2]

    if mode == "train":
        rng = np.random.RandomState(seed)
        return rng.randint(0, max_start + 1, size=num_segments).tolist()

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


def build_xy(
    tracks: List[Track],
    cfg: FeatureConfig,
    segment_seconds: float,
    num_segments: int,
    seed: int,
    mode: str,
    cache_dir: str,
    trim_silence: bool,
    trim_db: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    cfg_sig = f"{cfg.sample_rate}-{cfg.n_mels}-{cfg.n_fft}-{cfg.hop_length}-{cfg.fmin}-{cfg.fmax}"
    features = []
    targets = []
    track_ids = []
    for idx, track in enumerate(tracks):
        cache_sig = f"{cfg_sig}-trim{int(trim_silence)}-db{trim_db}"
        cached = load_cached_features(
            cache_dir,
            track.path,
            segment_seconds,
            num_segments,
            seed + idx,
            mode,
            cache_sig,
        )
        if cached is None:
            audio = load_audio(track.path, cfg)
            if trim_silence:
                audio, _ = librosa.effects.trim(audio, top_db=trim_db)
            starts = segment_starts(
                len(audio),
                cfg.sample_rate,
                segment_seconds,
                num_segments,
                mode=mode,
                seed=file_seed(track.path, seed + idx),
            )
            seg_len = int(segment_seconds * cfg.sample_rate)
            seg_features = []
            for start in starts:
                segment = audio[start : start + seg_len]
                seg_features.append(extract_features(segment, cfg))
            cached = np.vstack(seg_features)
            save_cached_features(
                cache_dir,
                track.path,
                segment_seconds,
                num_segments,
                seed + idx,
                mode,
                cache_sig,
                cached,
            )
        for _ in range(cached.shape[0]):
            targets.append(label_to_idx[track.label])
            track_ids.append(idx)
        features.append(cached)
    return (np.vstack(features), np.array(targets, dtype=np.int64), track_ids)


def probs_by_track(model: Pipeline, X: np.ndarray, track_ids: List[int]) -> np.ndarray:
    probs = model.predict_proba(X)
    track_probs = {}
    for prob, tid in zip(probs, track_ids):
        if tid not in track_probs:
            track_probs[tid] = []
        track_probs[tid].append(prob)
    out = []
    for tid in sorted(track_probs.keys()):
        out.append(np.mean(track_probs[tid], axis=0))
    return np.vstack(out)


def true_labels_by_track(track_ids: List[int], y: np.ndarray) -> np.ndarray:
    seen = {}
    for idx, tid in enumerate(track_ids):
        if tid not in seen:
            seen[tid] = y[idx]
    return np.array([seen[tid] for tid in sorted(seen.keys())], dtype=np.int64)


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

    train_tracks = select_tracks(tracks, splits["train"])
    val_tracks = select_tracks(tracks, splits["val"])
    test_tracks = select_tracks(tracks, splits["test"])

    X_train, y_train, _ = build_xy(
        train_tracks,
        feature_cfg,
        args.segment_seconds,
        args.num_segments,
        args.seed,
        mode="train",
        cache_dir=args.cache_dir,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
    )
    X_val, y_val, val_track_ids = build_xy(
        val_tracks,
        feature_cfg,
        args.segment_seconds,
        args.num_segments,
        args.seed,
        mode="eval",
        cache_dir=args.cache_dir,
        trim_silence=args.trim_silence,
        trim_db=args.trim_db,
    )
    if test_tracks:
        X_test, y_test, test_track_ids = build_xy(
            test_tracks,
            feature_cfg,
            args.segment_seconds,
            args.num_segments,
            args.seed,
            mode="eval",
            cache_dir=args.cache_dir,
            trim_silence=args.trim_silence,
            trim_db=args.trim_db,
        )
    else:
        X_test = None
        y_test = None
        test_track_ids = []

    lr_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    multi_class="multinomial",
                    solver="lbfgs",
                    class_weight="balanced",
                    random_state=args.seed,
                ),
            ),
        ]
    )

    svm_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                SVC(
                    kernel="rbf",
                    C=10.0,
                    gamma="scale",
                    class_weight="balanced",
                    probability=True,
                    random_state=args.seed,
                ),
            ),
        ]
    )

    lr_model.fit(X_train, y_train)
    svm_model.fit(X_train, y_train)

    lr_val_probs = probs_by_track(lr_model, X_val, val_track_ids)
    svm_val_probs = probs_by_track(svm_model, X_val, val_track_ids)
    meta_X_val = np.hstack([lr_val_probs, svm_val_probs])
    meta_y_val = true_labels_by_track(val_track_ids, y_val)

    meta_model = LogisticRegression(
        max_iter=1000,
        multi_class="multinomial",
        solver="lbfgs",
        class_weight="balanced",
        random_state=args.seed,
    )
    meta_model.fit(meta_X_val, meta_y_val)

    val_pred = meta_model.predict(meta_X_val)
    val_acc = accuracy_score(meta_y_val, val_pred)
    print(f"Val accuracy: {val_acc:.3f}")

    if X_test is not None and len(X_test) > 0:
        lr_test_probs = probs_by_track(lr_model, X_test, test_track_ids)
        svm_test_probs = probs_by_track(svm_model, X_test, test_track_ids)
        meta_X_test = np.hstack([lr_test_probs, svm_test_probs])
        meta_y_test = true_labels_by_track(test_track_ids, y_test)
        test_pred = meta_model.predict(meta_X_test)
        test_acc = accuracy_score(meta_y_test, test_pred)
        print(f"Test accuracy: {test_acc:.3f}")
        cm = confusion_matrix(meta_y_test, test_pred)
        report = classification_report(meta_y_test, test_pred, target_names=LABELS)
    else:
        cm = None
        report = None

    with open(os.path.join(args.run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump({"val_acc": val_acc}, f, indent=2)

    if cm is not None:
        np.save(os.path.join(args.run_dir, "confusion.npy"), cm)
    if report is not None:
        with open(os.path.join(args.run_dir, "classification_report.txt"), "w", encoding="utf-8") as f:
            f.write(report)

    import joblib

    joblib.dump(lr_model, os.path.join(args.run_dir, "lr_model.joblib"))
    joblib.dump(svm_model, os.path.join(args.run_dir, "svm_model.joblib"))
    joblib.dump(meta_model, os.path.join(args.run_dir, "meta_model.joblib"))


if __name__ == "__main__":
    main()
