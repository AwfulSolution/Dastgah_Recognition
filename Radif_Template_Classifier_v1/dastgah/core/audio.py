"""Turn a recording into the 24-TET pitch features the classifier consumes.

The chain is: monophonic f0 tracking, estimation of the performance's own
tuning reference, soft binning into quarter-tone pitch classes, and grouping of
stable frames into note events for the transition statistics.

Deriving features from f0 rather than from a spectrogram is deliberate. Dastgah
is a property of pitch relationships, not of timbre, so a pitch-domain
representation generalises across tar, setar, ney, santur and voice instead of
learning the instrument. It also means the tuning reference is estimated per
performance, which matters because Persian ensembles do not tune to A440.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dastgah.theory import QUARTER_TONES_PER_OCTAVE

N = QUARTER_TONES_PER_OCTAVE

#: Pitch-tracking range, wide enough for bass santur through high ney.
DEFAULT_FMIN_HZ = 70.0
DEFAULT_FMAX_HZ = 1200.0
DEFAULT_SR = 22050


@dataclass
class NoteEvent:
    """A run of frames sitting on one pitch class."""

    pitch_class: int
    start: float
    duration: float
    mean_hz: float
    cents_deviation: float  # signed offset from the exact quarter-tone
    quarter_tone: float = 0.0  # absolute pitch; pitch_class wraps, this does not


@dataclass
class PitchTrack:
    """Frame-level f0 analysis of a recording."""

    times: np.ndarray
    f0_hz: np.ndarray           # NaN where unvoiced
    voiced: np.ndarray          # bool
    confidence: np.ndarray
    sr: int
    hop_length: int
    reference_hz: float         # estimated tuning of the performance
    tuning_concentration: float  # 0-1; how firmly the contour sits on the grid
    duration: float

    @property
    def voiced_fraction(self) -> float:
        return float(self.voiced.mean()) if self.voiced.size else 0.0

    def quarter_tones(self) -> np.ndarray:
        """Fractional quarter-tone numbers, NaN where unvoiced."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return 138.0 + 24.0 * np.log2(self.f0_hz / self.reference_hz)


def load_audio(path: Path | str, sr: int = DEFAULT_SR) -> tuple[np.ndarray, int]:
    """Load a recording as mono float32 at ``sr``."""
    import librosa

    y, loaded_sr = librosa.load(str(path), sr=sr, mono=True)
    if y.size == 0:
        raise ValueError(f"no audio samples decoded from {path}")
    return y, loaded_sr


def estimate_tuning(
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    weights: np.ndarray | None = None,
) -> tuple[float, float]:
    """Find the tuning reference that aligns a performance to the 24-TET grid.

    Returns ``(reference_hz, concentration)``. Concentration runs from 0 to 1
    and says how tightly the contour sits on quarter-tone centres once aligned:
    near 1 for steady fretted playing, low for heavily ornamented or noisy
    material, in which case the reference is not worth trusting.

    Persian ensembles do not tune to A440, and a misplaced grid destroys exactly
    the koron/sori distinctions the templates depend on, so this is estimated
    per performance rather than assumed.

    The estimate is the circular mean of the fractional pitch positions. Note
    that it is only identifiable modulo one quarter-tone: a reference 20 cents
    sharp and one 30 cents flat align the grid identically, so the returned
    value should not be read as the ensemble's absolute concert pitch.
    """
    finite = voiced & np.isfinite(f0_hz)
    values = f0_hz[finite]
    if values.size == 0:
        return 440.0, 0.0

    magnitudes = np.ones(values.shape) if weights is None else np.asarray(weights)[finite]
    if magnitudes.sum() <= 0:
        magnitudes = np.ones(values.shape)

    quarter_tones = 138.0 + 24.0 * np.log2(values / 440.0)
    # Each frame becomes a unit vector whose angle is its position within the
    # quarter-tone it falls in; their weighted mean points at the grid offset.
    vectors = np.exp(2j * np.pi * quarter_tones)
    resultant = np.sum(vectors * magnitudes) / magnitudes.sum()

    offset_quarter_tones = float(np.angle(resultant) / (2 * np.pi))
    reference_hz = 440.0 * 2.0 ** (offset_quarter_tones / 24.0)
    return reference_hz, float(np.abs(resultant))


def estimate_reference_hz(
    f0_hz: np.ndarray,
    voiced: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """The tuning reference alone; see :func:`estimate_tuning`."""
    return estimate_tuning(f0_hz, voiced, weights)[0]


def track_pitch(
    y: np.ndarray,
    sr: int,
    *,
    fmin: float = DEFAULT_FMIN_HZ,
    fmax: float = DEFAULT_FMAX_HZ,
    frame_length: int = 2048,
    hop_length: int = 256,
) -> PitchTrack:
    """Extract an f0 contour with pYIN and estimate the tuning reference."""
    import librosa

    f0, voiced_flag, voiced_prob = librosa.pyin(
        y,
        fmin=fmin,
        fmax=fmax,
        sr=sr,
        frame_length=frame_length,
        hop_length=hop_length,
    )
    times = librosa.times_like(f0, sr=sr, hop_length=hop_length)
    voiced = np.asarray(voiced_flag, dtype=bool)
    confidence = np.nan_to_num(voiced_prob)
    reference, concentration = estimate_tuning(f0, voiced, confidence)

    return PitchTrack(
        times=times,
        f0_hz=f0,
        voiced=voiced,
        confidence=confidence,
        sr=sr,
        hop_length=hop_length,
        reference_hz=reference,
        tuning_concentration=concentration,
        duration=float(len(y) / sr),
    )


def pitch_histogram(
    track: PitchTrack,
    *,
    start: float | None = None,
    end: float | None = None,
    soft: bool = True,
) -> np.ndarray:
    """Duration-weighted 24-bin pitch-class histogram over a time span.

    With ``soft`` the weight of each frame is split between the two nearest
    bins in proportion to its distance from them, so vibrato and glissando
    contribute to both rather than flickering between them.
    """
    quarter = track.quarter_tones()
    mask = track.voiced & np.isfinite(quarter)
    if start is not None:
        mask &= track.times >= start
    if end is not None:
        mask &= track.times < end

    selected = quarter[mask]
    weights = track.confidence[mask]
    histogram = np.zeros(N, dtype=float)
    if selected.size == 0:
        return histogram

    if not soft:
        np.add.at(histogram, np.round(selected).astype(int) % N, weights)
        return histogram

    lower = np.floor(selected)
    fraction = selected - lower
    lower_bin = lower.astype(int) % N
    upper_bin = (lower_bin + 1) % N
    np.add.at(histogram, lower_bin, weights * (1.0 - fraction))
    np.add.at(histogram, upper_bin, weights * fraction)
    return histogram


def note_histogram(
    events: list[NoteEvent],
    *,
    start: float | None = None,
    end: float | None = None,
) -> np.ndarray:
    """Duration-weighted pitch-class histogram over note events.

    Prefer this to :func:`pitch_histogram` for classification. The templates are
    duration-weighted histograms of *notes*, and a frame-level histogram is a
    much broader distribution than that: every ornament, glissando and tahrir
    frame contributes, which on real recordings adds about 1.5 bits of entropy.
    Matched against templates built from notation, such an observation is
    broader than every template and simply selects whichever is most permissive.
    """
    histogram = np.zeros(N, dtype=float)
    for event in events:
        if start is not None and event.start < start:
            continue
        if end is not None and event.start >= end:
            continue
        histogram[event.pitch_class] += event.duration
    return histogram


def note_events(
    track: PitchTrack,
    *,
    min_duration: float = 0.06,
    tolerance: float = 0.5,
) -> list[NoteEvent]:
    """Group consecutive frames that hold one pitch class into note events.

    ``tolerance`` is in quarter-tones; a deviation beyond it starts a new event.
    Events shorter than ``min_duration`` seconds are dropped as ornament noise.
    """
    quarter = track.quarter_tones()
    frame_seconds = track.hop_length / track.sr
    events: list[NoteEvent] = []

    run_start_index: int | None = None
    run_values: list[float] = []

    def flush(end_index: int) -> None:
        if run_start_index is None or not run_values:
            return
        duration = (end_index - run_start_index) * frame_seconds
        if duration < min_duration:
            return
        mean_quarter = float(np.mean(run_values))
        rounded = int(round(mean_quarter))
        hz = track.reference_hz * (2.0 ** ((mean_quarter - 138.0) / 24.0))
        events.append(
            NoteEvent(
                pitch_class=rounded % N,
                start=float(track.times[run_start_index]),
                duration=duration,
                mean_hz=hz,
                cents_deviation=round((mean_quarter - rounded) * 50.0, 1),
                quarter_tone=mean_quarter,
            )
        )

    for index, (value, is_voiced) in enumerate(zip(quarter, track.voiced, strict=True)):
        if not is_voiced or not np.isfinite(value):
            flush(index)
            run_start_index, run_values = None, []
            continue
        if run_start_index is None:
            run_start_index, run_values = index, [value]
            continue
        if abs(value - np.mean(run_values)) <= tolerance:
            run_values.append(value)
        else:
            flush(index)
            run_start_index, run_values = index, [value]
    flush(len(quarter))

    return events


def transition_matrix(events: list[NoteEvent]) -> np.ndarray:
    """Pitch-class bigram counts over consecutive note events."""
    matrix = np.zeros((N, N), dtype=float)
    for current, following in zip(events, events[1:]):
        if current.pitch_class != following.pitch_class:
            matrix[current.pitch_class, following.pitch_class] += 1.0
    return matrix
