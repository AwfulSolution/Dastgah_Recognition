from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import librosa
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from .cache import load_cached_features, save_cached_features
from .data import LABELS, Track
from .features import FeatureConfig, load_audio
from .scikit_features import (
    ScikitFeatureOpts,
    cache_sig,
    extract_features,
    file_seed,
    segment_starts,
)


@dataclass
class TrackMetrics:
    accuracy: float
    macro_f1: float
    balanced_accuracy: float


def select_tracks(tracks: List[Track], indices: List[int]) -> List[Track]:
    return [tracks[i] for i in indices]


def build_xy(
    tracks: List[Track],
    cfg: FeatureConfig,
    opts: ScikitFeatureOpts,
    segment_seconds: float,
    num_segments: int,
    seed: int,
    mode: str,
    cache_dir: str,
    trim_silence: bool,
    trim_db: int,
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    features = []
    targets = []
    track_ids = []
    cfg_signature = cache_sig(cfg, trim_silence, trim_db, opts)

    for idx, track in enumerate(tqdm(tracks, desc=f"Features ({mode})")):
        cached = load_cached_features(
            cache_dir,
            track.path,
            segment_seconds,
            num_segments,
            seed + idx,
            mode,
            cfg_signature,
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
                seg_features.append(extract_features(segment, cfg, opts))
            cached = np.vstack(seg_features)
            save_cached_features(
                cache_dir,
                track.path,
                segment_seconds,
                num_segments,
                seed + idx,
                mode,
                cfg_signature,
                cached,
            )

        for _ in range(cached.shape[0]):
            targets.append(label_to_idx[track.label])
            track_ids.append(idx)
        features.append(cached)

    return np.vstack(features), np.array(targets, dtype=np.int64), track_ids


def precompute_feature_bank(
    tracks: List[Track],
    cfg: FeatureConfig,
    opts: ScikitFeatureOpts,
    segment_seconds: float,
    num_segments: int,
    cache_dir: str,
    trim_silence: bool,
    trim_db: int,
    seed: int,
    cache_mode: str,
    segment_mode: str,
) -> Dict[str, np.ndarray]:
    bank: Dict[str, np.ndarray] = {}
    cfg_signature = cache_sig(cfg, trim_silence, trim_db, opts)

    for track in tqdm(tracks, desc=f"Features ({cache_mode})"):
        cache_seed = file_seed(track.path, seed)
        cached = load_cached_features(
            cache_dir,
            track.path,
            segment_seconds,
            num_segments,
            cache_seed,
            cache_mode,
            cfg_signature,
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
                mode=segment_mode,
                seed=cache_seed,
            )
            seg_len = int(segment_seconds * cfg.sample_rate)
            seg_features = []
            for start in starts:
                segment = audio[start : start + seg_len]
                seg_features.append(extract_features(segment, cfg, opts))
            cached = np.vstack(seg_features)
            save_cached_features(
                cache_dir,
                track.path,
                segment_seconds,
                num_segments,
                cache_seed,
                cache_mode,
                cfg_signature,
                cached,
            )
        bank[track.path] = cached

    return bank


def build_xy_from_bank(
    tracks: List[Track],
    bank: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, List[int]]:
    label_to_idx = {label: i for i, label in enumerate(LABELS)}
    features = []
    targets: List[int] = []
    track_ids: List[int] = []

    for idx, track in enumerate(tracks):
        feats = bank[track.path]
        features.append(feats)
        label_idx = label_to_idx[track.label]
        targets.extend([label_idx] * feats.shape[0])
        track_ids.extend([idx] * feats.shape[0])

    return np.vstack(features), np.array(targets, dtype=np.int64), track_ids


def probs_by_track(model, X: np.ndarray, track_ids: List[int]) -> np.ndarray:
    probs = model.predict_proba(X)
    per_track: Dict[int, List[np.ndarray]] = {}
    for prob, tid in zip(probs, track_ids):
        per_track.setdefault(tid, []).append(prob)
    return np.vstack([np.mean(per_track[tid], axis=0) for tid in sorted(per_track.keys())])


def predict_trackwise(model, X: np.ndarray, track_ids: List[int]) -> np.ndarray:
    track_probs = probs_by_track(model, X, track_ids)
    return np.argmax(track_probs, axis=1).astype(np.int64)


def true_labels_by_track(track_ids: List[int], y: np.ndarray) -> np.ndarray:
    seen: Dict[int, int] = {}
    for idx, tid in enumerate(track_ids):
        if tid not in seen:
            seen[tid] = int(y[idx])
    return np.array([seen[tid] for tid in sorted(seen.keys())], dtype=np.int64)


def track_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> TrackMetrics:
    return TrackMetrics(
        accuracy=float(accuracy_score(y_true, y_pred)),
        macro_f1=float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y_true, y_pred)),
    )


def mean_ci95(values: List[float]) -> Dict[str, float]:
    arr = np.array(values, dtype=np.float64)
    if arr.size == 0:
        return {"mean": 0.0, "ci95_low": 0.0, "ci95_high": 0.0, "n": 0}
    mean = float(np.mean(arr))
    if arr.size == 1:
        return {"mean": mean, "ci95_low": mean, "ci95_high": mean, "n": 1}
    se = float(np.std(arr, ddof=1) / np.sqrt(arr.size))
    half = 1.96 * se
    return {
        "mean": mean,
        "ci95_low": float(mean - half),
        "ci95_high": float(mean + half),
        "n": int(arr.size),
    }


def cross_validate_trackwise(
    tracks: List[Track],
    cfg: FeatureConfig,
    opts: ScikitFeatureOpts,
    segment_seconds: float,
    num_segments: int,
    cache_dir: str,
    trim_silence: bool,
    trim_db: int,
    seed: int,
    n_splits: int,
    n_repeats: int,
    model_factory: Callable[[], object],
) -> Dict[str, Dict[str, float]]:
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if n_repeats < 1:
        raise ValueError("n_repeats must be >= 1")

    labels = np.array([t.label for t in tracks])
    indices = np.arange(len(tracks))

    # Precompute deterministic track feature banks once, then reuse across all CV folds.
    # This avoids repeatedly decoding/extracting audio per fold and dramatically speeds up CV.
    train_bank = precompute_feature_bank(
        tracks=tracks,
        cfg=cfg,
        opts=opts,
        segment_seconds=segment_seconds,
        num_segments=num_segments,
        cache_dir=cache_dir,
        trim_silence=trim_silence,
        trim_db=trim_db,
        seed=seed,
        cache_mode="cvtrain",
        segment_mode="train",
    )
    eval_bank = precompute_feature_bank(
        tracks=tracks,
        cfg=cfg,
        opts=opts,
        segment_seconds=segment_seconds,
        num_segments=num_segments,
        cache_dir=cache_dir,
        trim_silence=trim_silence,
        trim_db=trim_db,
        seed=seed,
        cache_mode="cveval",
        segment_mode="eval",
    )
    acc_values: List[float] = []
    f1_values: List[float] = []
    bal_values: List[float] = []

    fold_counter = 0
    for repeat in range(n_repeats):
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat)
        for fold_id, (train_idx, val_idx) in enumerate(splitter.split(indices, labels), start=1):
            fold_counter += 1
            train_tracks = [tracks[i] for i in train_idx.tolist()]
            val_tracks = [tracks[i] for i in val_idx.tolist()]

            X_train, y_train, _ = build_xy_from_bank(train_tracks, train_bank)
            X_val, y_val, val_track_ids = build_xy_from_bank(val_tracks, eval_bank)
            model = model_factory()
            model.fit(X_train, y_train)
            y_pred = predict_trackwise(model, X_val, val_track_ids)
            y_true = true_labels_by_track(val_track_ids, y_val)
            metrics = track_metrics(y_true, y_pred)
            acc_values.append(metrics.accuracy)
            f1_values.append(metrics.macro_f1)
            bal_values.append(metrics.balanced_accuracy)

    return {
        "accuracy": mean_ci95(acc_values),
        "macro_f1": mean_ci95(f1_values),
        "balanced_accuracy": mean_ci95(bal_values),
        "folds_total": {"n": fold_counter, "ci95_low": 0.0, "ci95_high": 0.0, "mean": float(fold_counter)},
    }
