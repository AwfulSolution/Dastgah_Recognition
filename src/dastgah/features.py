from dataclasses import dataclass
import numpy as np
import librosa


@dataclass
class FeatureConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    fmin: int = 20
    fmax: int = 8000


def load_audio(path: str, cfg: FeatureConfig) -> np.ndarray:
    audio, _ = librosa.load(path, sr=cfg.sample_rate, mono=True)
    return audio


def melspec(audio: np.ndarray, cfg: FeatureConfig) -> np.ndarray:
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_mels=cfg.n_mels,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        fmin=cfg.fmin,
        fmax=cfg.fmax,
        power=2.0,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return mel_db.astype(np.float32)
