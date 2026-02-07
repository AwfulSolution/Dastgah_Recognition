import math
import random
from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset

from .data import Track, label_to_index
from .features import FeatureConfig, load_audio, melspec


@dataclass
class SegmentConfig:
    segment_seconds: float = 30.0
    train_random_crop: bool = True


class DastgahDataset(Dataset):
    def __init__(
        self,
        tracks: List[Track],
        cfg: FeatureConfig,
        seg_cfg: SegmentConfig,
        mode: str,
        seed: int,
    ) -> None:
        self.tracks = tracks
        self.cfg = cfg
        self.seg_cfg = seg_cfg
        self.mode = mode
        self.seed = seed
        self.label_map = label_to_index()

    def __len__(self) -> int:
        return len(self.tracks)

    def _select_segment(self, audio: np.ndarray, index: int) -> np.ndarray:
        seg_len = int(self.seg_cfg.segment_seconds * self.cfg.sample_rate)
        if len(audio) <= seg_len:
            return audio

        if self.mode == "train" and self.seg_cfg.train_random_crop:
            rng = random.Random(self.seed + index)
            start = rng.randint(0, len(audio) - seg_len)
        else:
            start = (len(audio) - seg_len) // 2
        return audio[start : start + seg_len]

    def __getitem__(self, idx: int):
        track = self.tracks[idx]
        audio = load_audio(track.path, self.cfg)
        audio = self._select_segment(audio, idx)
        mel = melspec(audio, self.cfg)
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        mel = torch.from_numpy(mel).unsqueeze(0)
        label = self.label_map[track.label]
        return mel, label
