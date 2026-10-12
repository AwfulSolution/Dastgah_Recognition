"""Log-mel spectrogram view of a track, and a small CNN over it.

This exists to be *wrong in different ways* from the melodic features. Those
reduce a performance to note events and then to tonic-relative histograms,
which discards timbre, register and articulation entirely. A convolutional
model over log-mel frames keeps exactly that discarded material and has no
notion of a note, so the two disagree on different tracks — which is the
property a stacked ensemble needs.

Segment windows are taken with the same `_segment_starts` logic as the
melodic pipeline, so both views describe the same excerpts of each track.
Spectrograms are cached per track as float16 (a full corpus is ~0.6 GB) under
the same path+CORPUS_VERSION keying used elsewhere, so the scanner that
rewrites audio metadata cannot invalidate them.
"""

from typing import List, Tuple

import librosa
import numpy as np

from .cache import cache_path, load_cached_track_features, save_cached_track_features

N_MELS = 128
HOP = 1024
FMIN = 50.0
FMAX = 8000.0


def spec_signature(cfg) -> str:
    return (f"mel1-sr{cfg.sample_rate}-fft{cfg.n_fft}-hop{HOP}-m{N_MELS}"
            f"-seg{cfg.segment_seconds}-n{cfg.num_segments}-trim{int(cfg.trim_silence)}")


def _log_mel(segment: np.ndarray, cfg) -> np.ndarray:
    m = librosa.feature.melspectrogram(
        y=segment, sr=cfg.sample_rate, n_fft=cfg.n_fft, hop_length=HOP,
        n_mels=N_MELS, fmin=FMIN, fmax=min(FMAX, cfg.sample_rate / 2),
    )
    return librosa.power_to_db(m, ref=np.max).astype(np.float32)


def extract_track_spec(track_path: str, cfg, mode: str, seed: int, cache_dir: str) -> np.ndarray:
    """(n_segments, N_MELS, frames) log-mel stack for one track, cached."""
    from .melodic_features import _load_segment, _segment_starts, _stable_track_seed

    sig = spec_signature(cfg)
    suffix = f"spec-{mode}-seed{seed}"
    cached = load_cached_track_features(cache_dir, track_path, sig, suffix)
    if cached is not None:
        return cached.astype(np.float32)

    try:
        duration = float(librosa.get_duration(path=track_path))
    except Exception:
        duration = 0.0
    total = int(duration * cfg.sample_rate)
    frames = int(round(cfg.segment_seconds * cfg.sample_rate / HOP)) + 1
    if total <= 0:
        out = np.zeros((1, N_MELS, frames), dtype=np.float32)
        save_cached_track_features(cache_dir, track_path, sig, suffix, out.astype(np.float16))
        return out

    starts = _segment_starts(total, cfg.sample_rate, cfg.segment_seconds,
                             cfg.num_segments, mode, _stable_track_seed(track_path, seed))
    specs = []
    for s in starts:
        try:
            seg = _load_segment(track_path, s, cfg)
        except Exception:
            continue
        if seg.size < cfg.n_fft:
            continue
        sp = _log_mel(seg, cfg)
        # Pad or crop to a fixed width so segments stack into one array.
        if sp.shape[1] < frames:
            sp = np.pad(sp, ((0, 0), (0, frames - sp.shape[1])), mode="edge")
        specs.append(sp[:, :frames])
    if not specs:
        specs = [np.zeros((N_MELS, frames), dtype=np.float32)]
    out = np.stack(specs).astype(np.float32)
    save_cached_track_features(cache_dir, track_path, sig, suffix, out.astype(np.float16))
    return out


TIMBRE_SIG = "timbre1"


def _timbre_vector(segment: np.ndarray, cfg) -> np.ndarray:
    """Summary statistics of the spectral envelope for one segment.

    A CNN over full spectrograms has enough capacity to memorise which
    recording a segment came from, which grouped splits then punish (it fit
    training segments to 0.70 while sitting at 0.25 on held-out groups).
    These few dozen pooled statistics describe timbre and spectral shape
    without preserving enough detail to identify a performance.
    """
    sr = cfg.sample_rate
    mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=20, n_fft=cfg.n_fft, hop_length=HOP)
    dmfcc = librosa.feature.delta(mfcc) if mfcc.shape[1] > 8 else np.zeros_like(mfcc)
    contrast = librosa.feature.spectral_contrast(y=segment, sr=sr, n_fft=cfg.n_fft, hop_length=HOP)
    centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, n_fft=cfg.n_fft, hop_length=HOP)
    rolloff = librosa.feature.spectral_rolloff(y=segment, sr=sr, n_fft=cfg.n_fft, hop_length=HOP)
    flatness = librosa.feature.spectral_flatness(y=segment, n_fft=cfg.n_fft, hop_length=HOP)
    zcr = librosa.feature.zero_crossing_rate(segment, hop_length=HOP)

    parts = []
    for arr in (mfcc, dmfcc, contrast):
        parts += [arr.mean(axis=1), arr.std(axis=1)]
    for arr in (centroid, rolloff, flatness, zcr):
        a = arr.reshape(-1)
        parts.append(np.array([a.mean(), a.std()]))
    return np.concatenate(parts).astype(np.float32)


def extract_track_timbre(track_path: str, cfg, mode: str, seed: int, cache_dir: str) -> np.ndarray:
    """One pooled timbre vector per track (mean over its segments)."""
    from .melodic_features import _load_segment, _segment_starts, _stable_track_seed

    sig = f"{TIMBRE_SIG}-sr{cfg.sample_rate}-fft{cfg.n_fft}-hop{HOP}-seg{cfg.segment_seconds}-n{cfg.num_segments}"
    suffix = f"timbre-{mode}-seed{seed}"
    cached = load_cached_track_features(cache_dir, track_path, sig, suffix)
    if cached is not None:
        return cached.astype(np.float32)

    try:
        duration = float(librosa.get_duration(path=track_path))
    except Exception:
        duration = 0.0
    total = int(duration * cfg.sample_rate)
    dim = 20 * 2 + 20 * 2 + 7 * 2 + 4 * 2
    if total <= 0:
        out = np.zeros(dim, dtype=np.float32)
        save_cached_track_features(cache_dir, track_path, sig, suffix, out)
        return out

    starts = _segment_starts(total, cfg.sample_rate, cfg.segment_seconds,
                             cfg.num_segments, mode, _stable_track_seed(track_path, seed))
    vecs = []
    for s in starts:
        try:
            seg = _load_segment(track_path, s, cfg)
        except Exception:
            continue
        if seg.size < cfg.n_fft:
            continue
        vecs.append(_timbre_vector(seg, cfg))
    out = (np.mean(vecs, axis=0) if vecs else np.zeros(dim, dtype=np.float32)).astype(np.float32)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    save_cached_track_features(cache_dir, track_path, sig, suffix, out)
    return out


def build_timbre_matrix(tracks, cfg, mode, seed, cache_dir, label_to_idx,
                        progress_label=None) -> Tuple[np.ndarray, np.ndarray]:
    from tqdm.auto import tqdm

    it = tqdm(tracks, desc=progress_label) if progress_label else tracks
    X = [extract_track_timbre(t.path, cfg, mode, seed, cache_dir) for t in it]
    y = np.array([label_to_idx[t.label] for t in tracks], dtype=np.int64)
    return np.vstack(X), y


def build_spec_dataset(tracks, cfg, mode, seed, cache_dir, label_to_idx,
                       progress_label=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (segments, segment_labels, segment_track_index)."""
    from tqdm.auto import tqdm

    specs: List[np.ndarray] = []
    labels: List[int] = []
    owner: List[int] = []
    it = tqdm(list(enumerate(tracks)), desc=progress_label) if progress_label else enumerate(tracks)
    for i, t in it:
        arr = extract_track_spec(t.path, cfg, mode, seed, cache_dir)
        for s in arr:
            specs.append(s)
            labels.append(label_to_idx[t.label])
            owner.append(i)
    return np.stack(specs), np.array(labels, dtype=np.int64), np.array(owner, dtype=np.int64)
