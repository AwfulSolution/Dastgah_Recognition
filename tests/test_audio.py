"""Audio feature tests built on synthesised tones, so they need no fixtures."""

from pathlib import Path

import numpy as np
import pytest

from dastgah.core.analyze import analyze
from dastgah.core.audio import (
    estimate_reference_hz,
    estimate_tuning,
    note_events,
    pitch_histogram,
    track_pitch,
    transition_matrix,
)
from dastgah.radif.templates import load_templates
from dastgah.theory import pitch_class

TEMPLATES = Path(__file__).resolve().parent.parent / "dastgah" / "data" / "templates.json"
SR = 22050


def tone(frequency: float, seconds: float, sr: int = SR) -> np.ndarray:
    """A decaying harmonic tone, roughly plucked-string in shape."""
    t = np.arange(int(sr * seconds)) / sr
    signal = sum(
        amplitude * np.sin(2 * np.pi * frequency * harmonic * t)
        for harmonic, amplitude in enumerate([1.0, 0.5, 0.3, 0.15], start=1)
    )
    return (signal * np.exp(-2.0 * t)).astype(np.float32)


def melody(frequencies: list[float], seconds: float = 0.6) -> np.ndarray:
    return np.concatenate([tone(f, seconds) for f in frequencies])


@pytest.fixture(scope="module")
def templates():
    return load_templates(TEMPLATES)


def test_tracks_a_steady_tone_to_the_right_pitch_class():
    track = track_pitch(tone(440.0, 2.0), SR)
    histogram = pitch_histogram(track)
    assert int(np.argmax(histogram)) == pitch_class(138)  # A440 -> quarter-tone 138


def test_estimates_a_deliberately_detuned_reference():
    """Identifiable only modulo a quarter-tone, so compare on that circle."""
    detune_cents = 20.0
    detuned = 440.0 * 2 ** (detune_cents / 1200)
    intervals = [0, 4, 8, 10, 14, 10, 8, 4]
    track = track_pitch(melody([detuned * 2 ** (i / 24) for i in intervals], 0.7), SR)

    offset = 1200 * np.log2(track.reference_hz / 440.0)
    error = (offset - detune_cents) % 50.0
    assert min(error, 50.0 - error) < 10.0


def test_reference_falls_back_to_a440_when_nothing_is_voiced():
    assert estimate_reference_hz(np.array([np.nan]), np.array([False])) == 440.0
    assert estimate_tuning(np.array([np.nan]), np.array([False])) == (440.0, 0.0)


@pytest.mark.parametrize("detune_cents", [-30, -12, 0, 7, 18, 24])
def test_tuning_estimate_is_exact_on_clean_input(detune_cents):
    """Regression: the estimate was a sweep whose objective did not vary.

    ``exp(2j*pi*(q - round(q))) == exp(2j*pi*q)``, so rounding cancelled and the
    old sweep scored every candidate identically; argmax then picked on
    floating-point noise and could land half a quarter-tone out.
    """
    degrees = np.array([0, 3, 6, 10, 14, 16, 20, 24], dtype=float)
    f0 = 261.63 * 2 ** (degrees / 24.0) * 2 ** (detune_cents / 1200.0)

    reference, concentration = estimate_tuning(f0, np.ones(f0.size, dtype=bool))
    offset = 1200 * np.log2(reference / 440.0)
    error = (offset - detune_cents) % 50.0
    assert min(error, 50.0 - error) < 1.0
    assert concentration > 0.99


def test_concentration_falls_when_pitches_sit_off_the_grid():
    """Pitches scattered across a quarter-tone cannot all be on grid centres."""
    on_grid = 261.63 * 2 ** (np.arange(8) / 24.0)
    scattered = 261.63 * 2 ** (np.arange(8) / 24.0 + np.linspace(0, 0.9, 8) / 24.0)
    _, tight = estimate_tuning(on_grid, np.ones(8, dtype=bool))
    _, loose = estimate_tuning(scattered, np.ones(8, dtype=bool))
    assert tight > loose


def test_tuning_weights_follow_frame_confidence():
    """A confident on-grid frame should outweigh an unconfident stray one."""
    f0 = np.array([440.0, 440.0 * 2 ** (25 / 1200)])
    voiced = np.ones(2, dtype=bool)
    biased, _ = estimate_tuning(f0, voiced, np.array([1.0, 0.01]))
    assert abs(1200 * np.log2(biased / 440.0)) < 5.0


def test_histogram_resolves_a_quarter_tone_apart():
    """A koron must land in its own bin, not smear into the natural beside it."""
    natural, koron = 440.0, 440.0 * 2 ** (-1 / 24)
    track = track_pitch(melody([natural, koron, natural, koron], 0.8), SR)
    histogram = pitch_histogram(track)
    top_two = set(np.argsort(histogram)[-2:])
    assert top_two == {pitch_class(138), pitch_class(137)}


def test_note_events_segment_a_melody():
    events = note_events(track_pitch(melody([220.0, 330.0, 440.0], 0.8), SR))
    assert len(events) >= 3
    assert all(event.duration > 0 for event in events)
    assert {e.pitch_class for e in events} >= {pitch_class(114), pitch_class(138)}


def test_transitions_ignore_repeated_pitches():
    events = note_events(track_pitch(melody([220.0, 220.0, 330.0], 0.8), SR))
    matrix = transition_matrix(events)
    assert np.trace(matrix) == 0


def test_windowed_histograms_partition_the_recording():
    track = track_pitch(melody([220.0] * 4 + [330.0] * 4, 0.5), SR)
    first = pitch_histogram(track, start=0.0, end=1.5)
    second = pitch_histogram(track, start=2.5, end=4.0)
    assert np.argmax(first) != np.argmax(second)


def test_analyze_rejects_silence(tmp_path, templates):
    import soundfile as sf

    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(SR * 2, dtype=np.float32), SR)
    with pytest.raises(ValueError, match="no pitched content"):
        analyze(path, templates=templates, with_segments=False)


def test_analyze_recovers_a_synthesised_chahargah(tmp_path, templates):
    """Chahargah on C: C, D-koron, E, F, G, A-koron, B."""
    import soundfile as sf

    tonic = 261.63
    degrees = [0, 3, 8, 10, 14, 17, 22, 24, 22, 17, 14, 10, 8, 3, 0]
    signal = melody([tonic * 2 ** (d / 24) for d in degrees * 3], 0.55)
    path = tmp_path / "chahargah.wav"
    sf.write(path, signal / np.abs(signal).max() * 0.9, SR)

    result = analyze(path, templates=templates, with_segments=False)
    assert result.key == "chahargah"
    assert result.tonic_name == "C"
    microtonal = {d.name for d in result.degrees if d.microtonal}
    assert {"Dk", "Ak"} <= microtonal


def test_note_histogram_weights_by_duration():
    from dastgah.core.audio import NoteEvent, note_histogram

    events = [
        NoteEvent(pitch_class=0, start=0.0, duration=3.0, mean_hz=261.6, cents_deviation=0),
        NoteEvent(pitch_class=7, start=3.0, duration=1.0, mean_hz=329.6, cents_deviation=0),
    ]
    histogram = note_histogram(events)
    assert histogram[0] == 3.0
    assert histogram[7] == 1.0
    assert int(np.argmax(histogram)) == 0


def test_note_histogram_respects_a_time_window():
    from dastgah.core.audio import NoteEvent, note_histogram

    events = [
        NoteEvent(pitch_class=0, start=0.0, duration=1.0, mean_hz=261.6, cents_deviation=0),
        NoteEvent(pitch_class=7, start=10.0, duration=1.0, mean_hz=329.6, cents_deviation=0),
    ]
    assert note_histogram(events, start=0.0, end=5.0)[7] == 0.0
    assert note_histogram(events, start=5.0, end=15.0)[0] == 0.0


def test_note_histogram_is_sharper_than_the_frame_histogram():
    """Why classification scores note events: frames are a much broader
    distribution, and against templates built from notation the broadest
    observation simply selects the most permissive template."""
    from dastgah.core.audio import note_histogram

    degrees = [0, 3, 8, 10, 14, 17, 22, 14, 10, 8]
    track = track_pitch(melody([261.63 * 2 ** (d / 24) for d in degrees * 2], 0.5), SR)

    def entropy(values):
        p = np.asarray(values, dtype=float)
        p = p[p > 0] / p.sum()
        return float(-(p * np.log2(p)).sum())

    assert entropy(note_histogram(note_events(track))) < entropy(pitch_histogram(track))
