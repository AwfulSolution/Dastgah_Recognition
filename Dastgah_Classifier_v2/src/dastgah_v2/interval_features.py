import hashlib
from dataclasses import dataclass
from typing import Dict, List, Tuple

import librosa
import numpy as np
from tqdm.auto import tqdm

from .cache import load_cached_track_features, save_cached_track_features
from .data import Track


@dataclass
class IntervalFeatureConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    hop_length: int = 256
    bins_per_octave: int = 24
    segment_seconds: float = 20.0
    num_segments: int = 8
    trim_silence: bool = True
    trim_db: int = 25
    voiced_ratio_threshold: float = 0.30
    min_voiced_ms: float = 120.0
    min_harmonic_ratio: float = 0.55
    cadence_fraction: float = 0.15
    step_clip_bins: int = 12
    duration_bins: int = 8


def feature_dim(cfg: IntervalFeatureConfig) -> int:
    bins = cfg.bins_per_octave
    return (bins * bins) + (bins * 2) + (cfg.step_clip_bins * 2 + 1) + cfg.duration_bins + 8


def cfg_signature(cfg: IntervalFeatureConfig) -> str:
    return (
        f"sr{cfg.sample_rate}-fft{cfg.n_fft}-hop{cfg.hop_length}-bins{cfg.bins_per_octave}"
        f"-seg{cfg.segment_seconds}-n{cfg.num_segments}-trim{int(cfg.trim_silence)}-db{cfg.trim_db}"
        f"-vr{cfg.voiced_ratio_threshold}-mv{cfg.min_voiced_ms}-hr{cfg.min_harmonic_ratio}"
        f"-cad{cfg.cadence_fraction}"
        f"-sc{cfg.step_clip_bins}-dur{cfg.duration_bins}"
    )


def load_audio(path: str, cfg: IntervalFeatureConfig) -> np.ndarray:
    audio, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    if cfg.trim_silence:
        audio, _ = librosa.effects.trim(audio, top_db=cfg.trim_db)
    return audio


def _segment_starts(audio_len: int, sr: int, seconds: float, num_segments: int, mode: str, seed: int) -> List[int]:
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


def _suppress_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or mask.size == 0:
        return mask
    out = mask.copy()
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif (not val) and start is not None:
            if i - start < min_len:
                out[start:i] = False
            start = None
    if start is not None and (mask.size - start) < min_len:
        out[start:mask.size] = False
    return out


def _duration_hist(intervals: np.ndarray, n_bins: int) -> np.ndarray:
    if intervals.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    changes = np.where(np.diff(intervals) != 0)[0] + 1
    parts = np.split(intervals, changes)
    lengths = np.array([len(p) for p in parts], dtype=np.float32)
    if lengths.size == 0:
        return np.zeros(n_bins, dtype=np.float32)

    edges = np.array([1, 2, 3, 4, 6, 8, 12], dtype=np.float32)
    if n_bins != 8:
        q = np.linspace(0, 1, max(n_bins - 1, 2))[1:-1]
        edges = np.quantile(lengths, q) if lengths.size > 1 else np.array([lengths[0]], dtype=np.float32)
    idx = np.digitize(lengths, edges, right=True)
    hist = np.bincount(idx, minlength=n_bins).astype(np.float64)
    return (hist / (hist.sum() + 1e-9)).astype(np.float32)


def _extract_segment_features(segment: np.ndarray, cfg: IntervalFeatureConfig) -> Tuple[np.ndarray, Dict[str, float]]:
    bins = cfg.bins_per_octave
    step_bins = cfg.step_clip_bins * 2 + 1
    out_dim = feature_dim(cfg)
    if segment.size < cfg.n_fft:
        segment = np.pad(segment, (0, cfg.n_fft - segment.size), mode="constant")

    harmonic, percussive = librosa.effects.hpss(segment)
    h_rms = float(np.sqrt(np.mean(np.square(harmonic))) + 1e-12)
    p_rms = float(np.sqrt(np.mean(np.square(percussive))) + 1e-12)
    harmonic_ratio = h_rms / (h_rms + p_rms + 1e-9)

    f0, voiced_flag, _ = librosa.pyin(
        harmonic,
        sr=cfg.sample_rate,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    if f0 is None or len(f0) == 0:
        vec = np.zeros(out_dim, dtype=np.float32)
        vec[-7] = harmonic_ratio
        return vec, {"voiced_ratio": 0.0, "pitched": 0.0}

    energy = librosa.feature.rms(y=harmonic, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    n = min(len(f0), len(energy))
    f0 = np.asarray(f0[:n], dtype=np.float64)
    energy = np.asarray(energy[:n], dtype=np.float64)

    voiced = np.isfinite(f0)
    if voiced_flag is not None:
        voiced = np.logical_and(voiced, np.asarray(voiced_flag[:n], dtype=bool))

    if energy.size > 0:
        e_thr = np.percentile(energy, 20)
        voiced = np.logical_and(voiced, energy >= e_thr)

    min_len = max(1, int(round((cfg.min_voiced_ms / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    voiced = _suppress_short_runs(voiced, min_len)

    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
    if (
        voiced_ratio < cfg.voiced_ratio_threshold
        or int(np.sum(voiced)) < 6
        or harmonic_ratio < cfg.min_harmonic_ratio
    ):
        vec = np.zeros(out_dim, dtype=np.float32)
        vec[-8] = voiced_ratio
        vec[-7] = harmonic_ratio
        vec[-1] = 0.0
        return vec, {"voiced_ratio": voiced_ratio, "harmonic_ratio": harmonic_ratio, "pitched": 0.0}

    f0v = f0[voiced]
    midi = librosa.hz_to_midi(f0v)
    midi = midi[np.isfinite(midi)]
    if midi.size < 6:
        vec = np.zeros(out_dim, dtype=np.float32)
        vec[-8] = voiced_ratio
        vec[-7] = harmonic_ratio
        vec[-1] = 0.0
        return vec, {"voiced_ratio": voiced_ratio, "harmonic_ratio": harmonic_ratio, "pitched": 0.0}

    pitch_pc = np.mod(np.floor((midi % 12.0) * (bins / 12.0)).astype(int), bins)

    abs_hist = np.bincount(pitch_pc, minlength=bins).astype(np.float64)
    tail_n = max(1, int(np.ceil(pitch_pc.shape[0] * cfg.cadence_fraction)))
    cadence_abs = np.bincount(pitch_pc[-tail_n:], minlength=bins).astype(np.float64)
    tonic_scores = abs_hist + 0.5 * cadence_abs
    tonic = int(np.argmax(tonic_scores))

    sorted_scores = np.sort(tonic_scores)
    top1 = float(sorted_scores[-1]) if sorted_scores.size else 0.0
    top2 = float(sorted_scores[-2]) if sorted_scores.size > 1 else 0.0
    tonic_strength = (top1 - top2) / (top1 + 1e-9)

    intervals = np.mod(pitch_pc - tonic, bins).astype(int)

    hist = np.bincount(intervals, minlength=bins).astype(np.float64)
    hist /= hist.sum() + 1e-9

    trans = np.zeros((bins, bins), dtype=np.float64)
    if intervals.size > 1:
        a = intervals[:-1]
        b = intervals[1:]
        np.add.at(trans, (a, b), 1.0)
        trans /= trans.sum() + 1e-9

    cadence = np.bincount(intervals[-tail_n:], minlength=bins).astype(np.float64)
    cadence /= cadence.sum() + 1e-9

    if intervals.size > 1:
        d = intervals[1:] - intervals[:-1]
        d = ((d + (bins // 2)) % bins) - (bins // 2)
    else:
        d = np.zeros(1, dtype=np.int64)
    d = np.clip(d, -cfg.step_clip_bins, cfg.step_clip_bins)
    step_hist = np.bincount(d + cfg.step_clip_bins, minlength=step_bins).astype(np.float64)
    step_hist /= step_hist.sum() + 1e-9

    dur_hist = _duration_hist(intervals, cfg.duration_bins).astype(np.float64)

    mean_abs_step = float(np.mean(np.abs(d))) / max(cfg.step_clip_bins, 1)
    std_step = float(np.std(d)) / max(cfg.step_clip_bins, 1)
    note_density = float(intervals.size) / max(voiced.size, 1)
    cadence_to_tonic = float(cadence[0])

    vec = np.concatenate(
        [
            hist.astype(np.float32),
            trans.reshape(-1).astype(np.float32),
            cadence.astype(np.float32),
            step_hist.astype(np.float32),
            dur_hist.astype(np.float32),
            np.array(
                [
                    voiced_ratio,
                    harmonic_ratio,
                    tonic_strength,
                    cadence_to_tonic,
                    mean_abs_step,
                    std_step,
                    note_density,
                    1.0,
                ],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )
    return vec, {"voiced_ratio": voiced_ratio, "harmonic_ratio": harmonic_ratio, "pitched": 1.0}


def _stable_track_seed(track_path: str, seed: int) -> int:
    digest = hashlib.md5(track_path.encode("utf-8")).hexdigest()[:8]
    return (seed + int(digest, 16)) % (2**31 - 1)


def extract_track_feature(
    track_path: str,
    cfg: IntervalFeatureConfig,
    mode: str,
    seed: int,
    cache_dir: str,
) -> np.ndarray:
    sig = cfg_signature(cfg)
    suffix = f"track-{mode}-seed{seed}"
    cached = load_cached_track_features(cache_dir, track_path, sig, suffix)
    if cached is not None:
        return cached

    audio = load_audio(track_path, cfg)
    if audio.size == 0:
        out = np.zeros(feature_dim(cfg) + 3, dtype=np.float32)
        save_cached_track_features(cache_dir, track_path, sig, suffix, out)
        return out

    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    starts = _segment_starts(
        audio_len=len(audio),
        sr=cfg.sample_rate,
        seconds=cfg.segment_seconds,
        num_segments=cfg.num_segments,
        mode=mode,
        seed=seed,
    )

    seg_feats = []
    pitched_flags = []
    voiced_ratios = []
    harmonic_ratios = []
    for start in starts:
        seg = audio[start : start + seg_len]
        feat, meta = _extract_segment_features(seg, cfg)
        seg_feats.append(feat)
        pitched_flags.append(meta["pitched"])
        voiced_ratios.append(meta["voiced_ratio"])
        harmonic_ratios.append(meta.get("harmonic_ratio", 0.0))

    feats = np.vstack(seg_feats)
    pitched_mask = np.array(pitched_flags, dtype=np.float32) > 0.5
    if np.any(pitched_mask):
        pooled = feats[pitched_mask].mean(axis=0)
    else:
        pooled = feats.mean(axis=0)

    seg_valid_ratio = float(np.mean(pitched_mask))
    seg_voiced_mean = float(np.mean(voiced_ratios)) if voiced_ratios else 0.0
    seg_harmonic_mean = float(np.mean(harmonic_ratios)) if harmonic_ratios else 0.0
    out = np.concatenate(
        [
            pooled.astype(np.float32),
            np.array([seg_valid_ratio, seg_voiced_mean, seg_harmonic_mean], dtype=np.float32),
        ],
        axis=0,
    )

    save_cached_track_features(cache_dir, track_path, sig, suffix, out)
    return out


def build_track_matrix(
    tracks: List[Track],
    cfg: IntervalFeatureConfig,
    mode: str,
    seed: int,
    cache_dir: str,
    label_to_idx: Dict[str, int],
    progress_label: str | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    X = []
    y = []
    iterator = tqdm(tracks, desc=progress_label) if progress_label else tracks
    for t in iterator:
        track_seed = _stable_track_seed(t.path, seed)
        X.append(extract_track_feature(t.path, cfg, mode=mode, seed=track_seed, cache_dir=cache_dir))
        y.append(label_to_idx[t.label])
    return np.vstack(X), np.array(y, dtype=np.int64)
