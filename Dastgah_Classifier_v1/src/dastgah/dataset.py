import math
import random
from dataclasses import dataclass
from typing import List

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Track, label_to_index
from .features import FeatureConfig, load_audio, melspec
from .cache import load_cached_mel, save_cached_mel
from .scikit_features import extract_mode_features


@dataclass
class SegmentConfig:
    segment_seconds: float = 30.0
    train_random_crop: bool = True
    use_augment: bool = False
    time_mask_param: int = 16
    freq_mask_param: int = 8
    trim_silence: bool = False
    trim_db: int = 25
    num_train_segments: int = 1
    num_eval_segments: int = 10
    use_mode_features: bool = False
    mode_pitch_bins: int = 24
    cache_mel: bool = False
    cache_dir: str = "data/cache"


class DastgahDataset(Dataset):
    def __init__(
        self,
        tracks: List[Track],
        cfg: FeatureConfig,
        seg_cfg: SegmentConfig,
        mode: str,
        seed: int,
        return_track_id: bool = False,
    ) -> None:
        self.tracks = tracks
        self.cfg = cfg
        self.seg_cfg = seg_cfg
        self.mode = mode
        self.seed = seed
        self.return_track_id = return_track_id
        self.label_map = label_to_index()
        self._mode_cache = {}
        self.track_segments = (
            self.seg_cfg.num_train_segments if mode == "train" else self.seg_cfg.num_eval_segments
        )

    def __len__(self) -> int:
        return len(self.tracks) * self.track_segments

    def _select_segment(self, audio: np.ndarray, track_idx: int, seg_idx: int) -> np.ndarray:
        seg_len = int(self.seg_cfg.segment_seconds * self.cfg.sample_rate)
        if len(audio) <= seg_len:
            return audio

        if self.mode == "train" and self.seg_cfg.train_random_crop:
            rng = random.Random(self.seed + (track_idx * 997) + seg_idx)
            start = rng.randint(0, len(audio) - seg_len)
        else:
            start = (len(audio) - seg_len) // 2
        return audio[start : start + seg_len]

    def __getitem__(self, idx: int):
        track_idx = idx // self.track_segments
        seg_idx = idx % self.track_segments
        track = self.tracks[track_idx]
        mel = self._get_mel(track)
        mel = self._select_mel_segment(mel, track_idx, seg_idx)
        mel = self._fix_length(mel)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        if self.seg_cfg.use_mode_features:
            mode_vec = self._get_mode_vec(track)
            mode_vec = (mode_vec - mode_vec.mean()) / (mode_vec.std() + 1e-6)
            mode_map = np.repeat(mode_vec[:, None], mel.shape[1], axis=1).astype(np.float32)
            mel = np.concatenate([mel, mode_map], axis=0)
        mel = torch.from_numpy(mel).unsqueeze(0)
        if self.mode == "train" and self.seg_cfg.use_augment:
            mel = self._spec_augment(mel)
        label = self.label_map[track.label]
        if self.return_track_id:
            return mel, label, track_idx
        return mel, label

    def _get_mode_vec(self, track: Track) -> np.ndarray:
        cached = self._mode_cache.get(track.path)
        if cached is not None:
            return cached
        audio = load_audio(track.path, self.cfg)
        if self.seg_cfg.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=self.seg_cfg.trim_db)
        mode_vec = extract_mode_features(audio, self.cfg, self.seg_cfg.mode_pitch_bins).astype(np.float32)
        self._mode_cache[track.path] = mode_vec
        return mode_vec

    def _select_mel_segment(self, mel: np.ndarray, track_idx: int, seg_idx: int) -> np.ndarray:
        seg_frames = int(self.seg_cfg.segment_seconds * self.cfg.sample_rate / self.cfg.hop_length) + 1
        total_frames = mel.shape[1]
        if total_frames <= seg_frames:
            return mel

        if self.mode == "train" and self.seg_cfg.train_random_crop:
            rng = random.Random(self.seed + (track_idx * 997) + seg_idx)
            start = rng.randint(0, total_frames - seg_frames)
        else:
            start = (total_frames - seg_frames) // 2
        return mel[:, start : start + seg_frames]

    def _get_mel(self, track: Track) -> np.ndarray:
        cfg_sig = f"{self.cfg.sample_rate}-{self.cfg.n_mels}-{self.cfg.n_fft}-{self.cfg.hop_length}-{self.cfg.fmin}-{self.cfg.fmax}"
        if self.seg_cfg.cache_mel:
            cached = load_cached_mel(self.seg_cfg.cache_dir, track.path, cfg_sig)
            if cached is not None:
                return cached

        audio = load_audio(track.path, self.cfg)
        if self.seg_cfg.trim_silence:
            audio, _ = librosa.effects.trim(audio, top_db=self.seg_cfg.trim_db)
        mel = melspec(audio, self.cfg)

        if self.seg_cfg.cache_mel:
            save_cached_mel(self.seg_cfg.cache_dir, track.path, cfg_sig, mel)
        return mel

    def _fix_length(self, mel: np.ndarray) -> np.ndarray:
        # Ensure consistent time dimension for batching
        target_frames = int(self.seg_cfg.segment_seconds * self.cfg.sample_rate / self.cfg.hop_length) + 1
        current = mel.shape[1]
        if current == target_frames:
            return mel
        if current > target_frames:
            return mel[:, :target_frames]
        pad_width = target_frames - current
        return np.pad(mel, ((0, 0), (0, pad_width)), mode="constant", constant_values=0.0)

    def _spec_augment(self, mel: torch.Tensor) -> torch.Tensor:
        # mel: [1, n_mels, time]
        n_mels = mel.size(1)
        n_steps = mel.size(2)

        if self.seg_cfg.freq_mask_param > 0:
            f = random.randint(0, self.seg_cfg.freq_mask_param)
            f0 = random.randint(0, max(n_mels - f, 1))
            mel[:, f0 : f0 + f, :] = 0.0

        if self.seg_cfg.time_mask_param > 0:
            t = random.randint(0, self.seg_cfg.time_mask_param)
            t0 = random.randint(0, max(n_steps - t, 1))
            mel[:, :, t0 : t0 + t] = 0.0

        return mel
