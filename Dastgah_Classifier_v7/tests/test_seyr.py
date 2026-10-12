"""The progression scorer must read order, and must read nothing else."""

from __future__ import annotations

import numpy as np
import pytest

from dastgah.core.analyze import DEFAULT_GUSHEH_PATH
from dastgah.core.seyr import (
    MIN_WINDOWS,
    progression,
    progression_scores,
    window_histograms,
)
from dastgah.radif.gusheh import load_gusheh_templates
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO

N = 24


@pytest.fixture(scope="module")
def gushehs():
    return load_gusheh_templates(DEFAULT_GUSHEH_PATH)


def traversal(templates, tonic_pc: int) -> np.ndarray:
    """A synthetic performance that plays a dastgah's gushehs in radif order."""
    return np.array([np.roll(t.as_array(), tonic_pc) for t in templates])


@pytest.mark.parametrize("mode", DASTGAHS_WITH_AUDIO)
def test_a_performance_in_radif_order_scores_high(gushehs, mode):
    windows = traversal(gushehs[mode], tonic_pc=5)
    assert progression(windows, gushehs[mode], 5).score > 0.8


@pytest.mark.parametrize("mode", DASTGAHS_WITH_AUDIO)
def test_the_same_material_reversed_scores_negative(gushehs, mode):
    windows = traversal(gushehs[mode], tonic_pc=5)[::-1]
    assert progression(windows, gushehs[mode], 5).score < -0.8


@pytest.mark.parametrize("mode", DASTGAHS_WITH_AUDIO)
def test_shuffling_the_order_destroys_the_score(gushehs, mode):
    """Identical pitch content, different order: the score must collapse.

    This is the property the pitch-histogram classifier cannot have, so it is
    the one worth asserting.
    """
    windows = traversal(gushehs[mode], tonic_pc=5)
    ordered = progression(windows, gushehs[mode], 5).score

    rng = np.random.default_rng(0)
    shuffled = [
        abs(progression(windows[rng.permutation(len(windows))], gushehs[mode], 5).score)
        for _ in range(40)
    ]
    assert np.mean(shuffled) < 0.4 * ordered


def test_pitch_content_alone_cannot_produce_a_score(gushehs):
    """Summing a traversal into one window must leave no order evidence."""
    windows = traversal(gushehs["shur"], tonic_pc=0)
    pooled = np.tile(windows.sum(axis=0), (len(windows), 1))
    assert progression(pooled, gushehs["shur"], 0).score == pytest.approx(0.0, abs=1e-9)


def test_a_wrong_tonic_scores_lower_than_the_right_one(gushehs):
    windows = traversal(gushehs["mahur"], tonic_pc=9)
    right = progression(windows, gushehs["mahur"], 9).score
    wrong = [progression(windows, gushehs["mahur"], t).score for t in range(N) if t != 9]
    assert right > max(wrong)


def test_the_right_mode_wins_the_joint_search(gushehs):
    modes = list(DASTGAHS_WITH_AUDIO)
    for truth in modes:
        windows = traversal(gushehs[truth], tonic_pc=3)
        scores = progression_scores(windows, gushehs, modes)
        tonic, column = np.unravel_index(scores.argmax(), scores.shape)
        assert modes[column] == truth, f"{truth} lost to {modes[column]}"
        assert tonic == 3


def test_too_few_windows_scores_neutral(gushehs):
    windows = traversal(gushehs["segah"], tonic_pc=0)[: MIN_WINDOWS - 1]
    result = progression(windows, gushehs["segah"], 0)
    assert result.score == 0.0
    assert result.n_windows < MIN_WINDOWS


def test_empty_windows_are_dropped_not_matched(gushehs):
    windows = traversal(gushehs["shur"], tonic_pc=0)
    padded = np.vstack([np.zeros((3, N)), windows, np.zeros((3, N))])
    assert progression(padded, gushehs["shur"], 0).n_windows == len(windows)


def test_window_histograms_bin_events_by_time():
    from dastgah.core.audio import NoteEvent

    events = [
        NoteEvent(pitch_class=0, start=0.0, duration=1.0, mean_hz=220.0,
                  cents_deviation=0.0, quarter_tone=0.0),
        NoteEvent(pitch_class=7, start=25.0, duration=2.0, mean_hz=330.0,
                  cents_deviation=0.0, quarter_tone=7.0),
        NoteEvent(pitch_class=14, start=45.0, duration=3.0, mean_hz=440.0,
                  cents_deviation=0.0, quarter_tone=14.0),
    ]
    grid = window_histograms(events, seconds=20.0, duration=60.0)
    assert grid.shape == (3, N)
    assert grid[0, 0] == 1.0
    assert grid[1, 7] == 2.0
    assert grid[2, 14] == 3.0


def test_unordered_templates_are_refused(gushehs):
    import dataclasses

    broken = [dataclasses.replace(t, seyr_index=-1) for t in gushehs["shur"]]
    with pytest.raises(ValueError, match="no seyr order"):
        progression(traversal(gushehs["shur"], 0), broken, 0)
