"""Tests for cadence detection.

Note events are built directly rather than synthesised from audio, so these
exercise the cadence logic rather than pitch tracking.
"""

import numpy as np
import pytest

from dastgah.core.audio import NoteEvent
from dastgah.core.forud import PHRASE_GAP, find_foruds, tonic_prior


def note(quarter_tone: float, start: float, duration: float) -> NoteEvent:
    return NoteEvent(
        pitch_class=int(round(quarter_tone)) % 24,
        start=start,
        duration=duration,
        mean_hz=440.0 * 2 ** ((quarter_tone - 138) / 24),
        cents_deviation=0.0,
        quarter_tone=quarter_tone,
    )


def phrase(pitches, start=0.0, step=0.3, final_duration=1.2):
    """A run of notes ending on a held one."""
    events, time = [], start
    for pitch in pitches[:-1]:
        events.append(note(pitch, time, step))
        time += step
    events.append(note(pitches[-1], time, final_duration))
    return events


def test_a_descent_onto_a_held_note_is_a_cadence():
    events = phrase([134, 130, 126, 123, 120])
    found = find_foruds(events, total_duration=10.0)
    assert len(found) == 1
    assert found[0].resolution_pc == 120 % 24
    assert found[0].descent > 0
    assert found[0].strength > 0.5


def test_an_ascending_run_is_not_a_cadence():
    found = find_foruds(phrase([120, 123, 126, 130, 134]), total_duration=10.0)
    assert all(f.strength < 0.3 for f in found)


def test_a_descent_that_does_not_settle_scores_lower():
    """Same descent, but the final note is no longer than the rest."""
    settled = find_foruds(phrase([134, 130, 126, 123, 120]), total_duration=10.0)
    hurried = find_foruds(
        phrase([134, 130, 126, 123, 120], final_duration=0.3), total_duration=10.0
    )
    assert settled[0].strength > hurried[0].strength


def test_a_cadence_followed_by_silence_scores_higher():
    events = phrase([134, 130, 126, 123, 120])
    with_breath = find_foruds(events, total_duration=20.0)
    trailing = events + [note(126, events[-1].start + events[-1].duration + 0.01, 0.3)]
    without = find_foruds(trailing, total_duration=20.0)
    assert with_breath[0].strength > without[0].strength


def test_phrases_split_on_silence():
    first = phrase([134, 130, 126, 123, 120])
    gap_start = first[-1].start + first[-1].duration + PHRASE_GAP * 4
    second = phrase([132, 128, 124, 121, 118], start=gap_start)
    found = find_foruds(first + second, total_duration=40.0)
    assert len(found) == 2
    assert {f.resolution_pc for f in found} == {120 % 24, 118 % 24}


def test_short_phrases_are_ignored():
    assert find_foruds(phrase([130, 120]), total_duration=10.0) == []


def test_no_events_yields_nothing():
    assert find_foruds([], total_duration=0.0) == []
    assert tonic_prior([]) is None


def test_prior_points_at_the_resolution():
    found = find_foruds(phrase([134, 130, 126, 123, 120]), total_duration=10.0)
    prior = tonic_prior(found)
    assert prior is not None
    assert prior.shape == (24,)
    assert np.isclose(prior.sum(), 1.0)
    assert int(np.argmax(prior)) == 120 % 24


def test_recency_favours_the_closing_cadence():
    """A performance may modulate; only the final forud returns home."""
    early = phrase([134, 130, 126, 123, 120])
    late_start = 100.0
    late = phrase([132, 128, 124, 121, 118], start=late_start)
    found = find_foruds(early + late, total_duration=120.0)

    flat = tonic_prior(found)
    recent = tonic_prior(found, total_duration=120.0, recency_halflife=20.0)
    assert int(np.argmax(recent)) == 118 % 24
    assert recent[118 % 24] > flat[118 % 24]


def test_prior_is_rejected_by_classify_when_malformed():
    from pathlib import Path

    from dastgah.core.classify import classify
    from dastgah.radif.templates import load_templates

    templates = load_templates(
        Path(__file__).resolve().parent.parent / "dastgah" / "data" / "templates.json"
    )
    with pytest.raises(ValueError, match="tonic prior"):
        classify(np.ones(24), templates, tonic_prior=np.ones(12))


def test_an_empty_prior_falls_back_rather_than_failing():
    from pathlib import Path

    from dastgah.core.classify import classify
    from dastgah.radif.templates import load_templates

    templates = load_templates(
        Path(__file__).resolve().parent.parent / "dastgah" / "data" / "templates.json"
    )
    result = classify(np.ones(24), templates, tonic_prior=np.zeros(24))
    assert result.candidates
