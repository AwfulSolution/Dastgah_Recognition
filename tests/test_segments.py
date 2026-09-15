"""Tests for Viterbi smoothing and the modal timeline.

Pitch tracks are built directly from f0 arrays rather than synthesised audio, so
these stay fast and test the segmentation logic rather than pYIN.
"""

from pathlib import Path

import numpy as np
import pytest

from dastgah.core.analyze import _segments, _viterbi
from dastgah.core.audio import NoteEvent, PitchTrack
from dastgah.radif.templates import load_templates

TEMPLATES = Path(__file__).resolve().parent.parent / "dastgah" / "data" / "templates.json"
HOP = 0.01


@pytest.fixture(scope="module")
def templates():
    return load_templates(TEMPLATES)


def track_from_degrees(degrees: list[int], tonic_hz: float, seconds_per_note: float):
    """Build a PitchTrack holding each degree in turn, with matching events."""
    frames_per_note = int(seconds_per_note / HOP)
    f0, events, time = [], [], 0.0
    for degree in degrees:
        hz = tonic_hz * 2 ** (degree / 24.0)
        f0.extend([hz] * frames_per_note)
        events.append(
            NoteEvent(
                pitch_class=int(round(138 + 24 * np.log2(hz / 440.0))) % 24,
                start=time,
                duration=seconds_per_note,
                mean_hz=hz,
                cents_deviation=0.0,
            )
        )
        time += seconds_per_note

    f0 = np.array(f0)
    times = np.arange(len(f0)) * HOP
    track = PitchTrack(
        times=times,
        f0_hz=f0,
        voiced=np.ones(len(f0), dtype=bool),
        confidence=np.ones(len(f0)),
        sr=1,
        hop_length=HOP,
        reference_hz=440.0,
        tuning_concentration=1.0,
        duration=float(times[-1] + HOP),
    )
    return track, events


# --- Viterbi ---------------------------------------------------------------

def emissions_with_one_flip() -> np.ndarray:
    return np.log(
        np.array([[0.9, 0.1], [0.9, 0.1], [0.2, 0.8], [0.9, 0.1]] + [[0.1, 0.9]] * 4)
    )


def test_zero_penalty_reduces_to_the_per_window_argmax():
    emissions = emissions_with_one_flip()
    assert _viterbi(emissions, 0.0).tolist() == list(emissions.argmax(axis=1))


def test_a_modest_penalty_absorbs_an_isolated_flip():
    assert _viterbi(emissions_with_one_flip(), 1.0)[2] == 0


def test_a_large_penalty_forbids_switching_entirely():
    path = _viterbi(emissions_with_one_flip(), 100.0)
    assert len(set(path.tolist())) == 1


def test_sustained_evidence_still_moves_the_path():
    """Smoothing must not become a constant classifier."""
    path = _viterbi(emissions_with_one_flip(), 1.0)
    assert path[0] == 0 and path[-1] == 1


def test_viterbi_handles_an_empty_sequence():
    assert _viterbi(np.zeros((0, 3)), 4.0).size == 0


# --- timeline --------------------------------------------------------------

def test_short_recordings_produce_no_timeline(templates):
    from dastgah.core.classify import DEFAULT_CONFIG

    track, events = track_from_degrees([0, 4, 8], 261.63, 1.0)
    assert _segments(track, events, templates, DEFAULT_CONFIG) == []


def test_segments_tile_the_recording_without_gaps_or_overlaps(templates):
    """Regression: merging overlapping windows by their edges, rather than by
    the midpoints between window centres, produced overlapping segments."""
    from dastgah.core.classify import DEFAULT_CONFIG

    chahargah = [0, 3, 8, 10, 14, 17, 22, 14, 10, 3]
    mahur = [0, 4, 8, 10, 14, 18, 22, 14, 10, 4]
    track, events = track_from_degrees(chahargah * 8 + mahur * 8, 261.63, 0.7)

    segments = _segments(track, events, templates, DEFAULT_CONFIG)
    assert segments, "expected a timeline for a recording this long"

    assert segments[0].start == 0.0
    assert segments[-1].end == pytest.approx(track.duration, abs=1.0)
    for earlier, later in zip(segments, segments[1:]):
        assert later.start == pytest.approx(earlier.end, abs=1e-6)
        assert earlier.end > earlier.start


def test_a_steady_performance_is_not_chopped_up(templates):
    """Regression: unsmoothed windows flickered into dozens of segments."""
    from dastgah.core.classify import DEFAULT_CONFIG

    degrees = [0, 3, 8, 10, 14, 17, 22, 17, 14, 10, 8, 3]
    track, events = track_from_degrees(degrees * 14, 261.63, 0.7)

    segments = _segments(track, events, templates, DEFAULT_CONFIG)
    assert 1 <= len(segments) <= 3, f"expected a stable timeline, got {len(segments)}"


def test_every_segment_clears_the_minimum_duration(templates):
    from dastgah.core.classify import DEFAULT_CONFIG

    degrees = [0, 3, 8, 10, 14, 17, 22, 10]
    track, events = track_from_degrees(degrees * 20, 261.63, 0.7)

    segments = _segments(track, events, templates, DEFAULT_CONFIG, min_segment=15.0)
    # The first and last are stretched to cover the recording, so check the rest.
    for segment in segments[1:-1]:
        assert segment.end - segment.start >= 15.0
