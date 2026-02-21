from dataclasses import dataclass
from typing import List

import librosa
import numpy as np

from .features import FeatureConfig, melspec


@dataclass
class ScikitFeatureOpts:
    use_mode_features: bool = False
    mode_pitch_bins: int = 24


def mode_feature_dim(mode_pitch_bins: int) -> int:
    bins = max(12, int(mode_pitch_bins))
    return bins * 3 + 4


def file_seed(path: str, base_seed: int) -> int:
    h = __import__("hashlib").md5(path.encode("utf-8")).hexdigest()
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


def cache_sig(
    cfg: FeatureConfig,
    trim_silence: bool,
    trim_db: int,
    opts: ScikitFeatureOpts,
) -> str:
    cfg_sig = f"{cfg.sample_rate}-{cfg.n_mels}-{cfg.n_fft}-{cfg.hop_length}-{cfg.fmin}-{cfg.fmax}"
    return (
        f"{cfg_sig}-trim{int(trim_silence)}-db{trim_db}"
        f"-mode{int(opts.use_mode_features)}-bins{opts.mode_pitch_bins}"
    )


def _mode_features(audio: np.ndarray, cfg: FeatureConfig, bins: int) -> np.ndarray:
    if bins < 12:
        bins = 12

    f0, voiced_flag, _ = librosa.pyin(
        audio,
        sr=cfg.sample_rate,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
    )

    if f0 is None:
        return np.zeros(bins * 3 + 4, dtype=np.float32)

    f0 = np.asarray(f0, dtype=np.float64)
    voiced = ~np.isnan(f0)
    if voiced_flag is not None:
        voiced = np.logical_and(voiced, np.asarray(voiced_flag, dtype=bool))

    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
    if np.sum(voiced) < 4:
        out = np.zeros(bins * 3 + 4, dtype=np.float32)
        out[-4] = voiced_ratio
        return out

    f0v = f0[voiced]
    midi = librosa.hz_to_midi(f0v)
    midi = midi[np.isfinite(midi)]
    if midi.size == 0:
        out = np.zeros(bins * 3 + 4, dtype=np.float32)
        out[-4] = voiced_ratio
        return out

    pc = np.mod(np.floor((midi % 12.0) * (bins / 12.0)).astype(int), bins)

    pitch_hist = np.bincount(pc, minlength=bins).astype(np.float64)
    pitch_hist /= np.sum(pitch_hist) + 1e-9

    tonic_bin = int(np.argmax(pitch_hist))
    interval_pc = np.mod(pc - tonic_bin, bins)
    interval_hist = np.bincount(interval_pc, minlength=bins).astype(np.float64)
    interval_hist /= np.sum(interval_hist) + 1e-9

    tail_n = max(1, int(np.ceil(pc.shape[0] * 0.1)))
    cadence_hist = np.bincount(pc[-tail_n:], minlength=bins).astype(np.float64)
    cadence_hist /= np.sum(cadence_hist) + 1e-9

    sorted_hist = np.sort(pitch_hist)
    second = float(sorted_hist[-2]) if sorted_hist.shape[0] > 1 else 0.0
    tonic_strength = float(pitch_hist[tonic_bin] - second)
    cadence_to_tonic = float(cadence_hist[tonic_bin])

    med_f0 = float(np.median(f0v))
    cents = 1200.0 * np.log2(np.maximum(f0v, 1e-9) / max(med_f0, 1e-9))
    f0_std_cents = float(np.std(cents))

    return np.concatenate(
        [
            pitch_hist.astype(np.float32),
            interval_hist.astype(np.float32),
            cadence_hist.astype(np.float32),
            np.array(
                [voiced_ratio, tonic_strength, f0_std_cents, cadence_to_tonic],
                dtype=np.float32,
            ),
        ],
        axis=0,
    )


def extract_mode_features(audio: np.ndarray, cfg: FeatureConfig, bins: int) -> np.ndarray:
    return _mode_features(audio, cfg, bins)


def extract_features(audio: np.ndarray, cfg: FeatureConfig, opts: ScikitFeatureOpts) -> np.ndarray:
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

    parts = [
        mel_mean,
        mel_std,
        stats(mfcc_feat),
        stats(mfcc_delta),
        stats(chroma),
        stats(contrast),
        stats(tonnetz),
    ]

    if opts.use_mode_features:
        parts.append(_mode_features(audio, cfg, opts.mode_pitch_bins))

    return np.concatenate(parts, axis=0).astype(np.float32)
