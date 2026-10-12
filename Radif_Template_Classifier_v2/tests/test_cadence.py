"""The cadence approach: anchored on the resolution, so rotation cannot hide it."""

from __future__ import annotations

import numpy as np
import pytest

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.audio import NoteEvent
from dastgah.core.classify import DEFAULT_CONFIG, classify
import dataclasses

from dastgah.core.forud import Forud, approach_profile, cadence_agreement, tonic_prior
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

N = 24


@pytest.fixture(scope="module")
def templates():
    return load_templates(DEFAULT_TEMPLATE_PATH)


def event(pitch_class: int, start: float, duration: float = 0.5) -> NoteEvent:
    return NoteEvent(
        pitch_class=pitch_class % N, start=start, duration=duration,
        mean_hz=220.0, cents_deviation=0.0, quarter_tone=float(pitch_class),
    )


def cadence(resolution: int, approach: list[int], at: float = 0.0) -> tuple:
    """A descent through ``approach`` onto ``resolution``, and its events."""
    events = [event(pc, at + i * 0.5) for i, pc in enumerate([*approach, resolution])]
    forud = Forud(
        resolution_pc=resolution % N, start=at, end=at + 0.5 * (len(approach) + 1),
        descent=8.0, repose=1.0, silence_after=0.5, strength=0.5,
    )
    return forud, events


def test_the_approach_is_measured_from_the_resolution_not_the_tonic(templates):
    forud, events = cadence(resolution=5, approach=[15, 13, 11])
    profile = approach_profile(events, forud)
    assert profile is not None
    # 15, 13 and 11 sit 10, 8 and 6 quarter-tones above a resolution on 5.
    assert set(np.flatnonzero(profile > 0).tolist()) == {10, 8, 6}


def test_transposing_a_cadence_leaves_its_approach_unchanged(templates):
    first, first_events = cadence(resolution=5, approach=[15, 13, 11])
    second, second_events = cadence(resolution=12, approach=[22, 20, 18])
    assert np.allclose(
        approach_profile(first_events, first), approach_profile(second_events, second)
    )


def test_the_approach_distinguishes_direction(templates):
    """From above and from below are different, which a pitch profile is not."""
    above, above_events = cadence(resolution=0, approach=[10, 8, 4])
    below, below_events = cadence(resolution=0, approach=[14, 16, 20])
    from_above = approach_profile(above_events, above)
    from_below = approach_profile(below_events, below)
    norm = np.linalg.norm(from_above) * np.linalg.norm(from_below)
    assert float(np.dot(from_above, from_below) / norm) == pytest.approx(0.0, abs=1e-9)


def test_a_cadence_with_too_few_notes_has_no_approach(templates):
    forud, events = cadence(resolution=5, approach=[])
    assert approach_profile(events, forud) is None


def test_shur_and_nava_approach_a_close_differently(templates):
    """The premise of the whole feature, asserted against the notation."""
    shur, nava = templates["shur"].cadence_array(), templates["nava"].cadence_array()
    assert shur is not None and nava is not None
    direct = float(
        np.dot(shur, nava) / (np.linalg.norm(shur) * np.linalg.norm(nava))
    )
    pitch = max(
        float(
            np.dot(templates["shur"].as_array(), np.roll(templates["nava"].as_array(), s))
            / (np.linalg.norm(templates["shur"].as_array())
               * np.linalg.norm(templates["nava"].as_array()))
        )
        for s in range(N)
    )
    assert direct < pitch, "the approach must separate the pair better than pitch does"


def test_the_agreement_follows_the_direction_of_approach(templates):
    """A cadence reached from above must favour Shur over Nava at the SAME tonic.

    This is the case a normalised prior cannot express, which is why the term is
    additive: normalising over degrees cancels any per-mode weighting when every
    cadence lands on one degree.
    """
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    resolution = 7

    from_above, above_events = cadence(resolution, approach=[17, 15, 13, 11])
    above = cadence_agreement([from_above], above_events, profiles)
    assert above is not None
    above_margin = above["shur"][resolution] - above["nava"][resolution]

    from_below, below_events = cadence(resolution, approach=[1, 3, 23, 5])
    below = cadence_agreement([from_below], below_events, profiles)
    below_margin = below["shur"][resolution] - below["nava"][resolution]

    assert above_margin > below_margin


def test_modes_without_a_notated_approach_contribute_nothing(templates):
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    assert profiles["afshari"] is None, "fixture assumes afshari has too few closes"
    forud, events = cadence(resolution=3, approach=[13, 11, 9])
    agreement = cadence_agreement([forud], events, profiles)
    assert np.all(agreement["afshari"] == 0.0)


def test_a_degree_with_no_cadence_is_neutral_between_modes(templates):
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    forud, events = cadence(resolution=3, approach=[13, 11, 9])
    agreement = cadence_agreement([forud], events, profiles)
    for degree in range(N):
        if degree == 3:
            continue
        values = [agreement[k][degree] for k in profiles]
        assert all(v == 0.0 for v in values)


def test_no_cadences_means_no_agreement(templates):
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    assert cadence_agreement([], [], profiles) is None


def test_the_term_changes_the_score_only_when_weighted(templates):
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    forud, events = cadence(resolution=7, approach=[17, 15, 13, 11])
    agreement = cadence_agreement([forud], events, profiles)

    histogram = np.zeros(N)
    for e in events:
        histogram[e.pitch_class] += e.duration
    histogram += 0.01

    def scores(config, agree):
        result = classify(histogram, templates, tonic_prior=tonic_prior([forud]),
                          cadence_agreement=agree, config=config)
        return {(c.key, c.tonic_pc): c.score for c in result.candidates}

    off = scores(DEFAULT_CONFIG, agreement)
    plain = scores(DEFAULT_CONFIG, None)
    assert off == plain, "a zero weight must leave the score untouched"

    weighted = dataclasses.replace(DEFAULT_CONFIG, cadence_weight=1.0)
    on = scores(weighted, agreement)
    assert on != plain
    # and the change must be confined to the degree the cadence resolved onto
    changed = {k for k in on if on[k] != pytest.approx(plain[k])}
    assert changed and all(tonic == 7 for _, tonic in changed)


def test_the_agreement_is_centred_across_modes(templates):
    """Only relative fit may count, or a broad notated approach wins by default.

    A flat profile scores a decent cosine against any approach, which is a
    property of the template and not evidence about the recording.
    """
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    forud, events = cadence(resolution=9, approach=[19, 17, 15, 13])
    agreement = cadence_agreement([forud], events, profiles)

    described = [k for k, v in profiles.items() if v is not None]
    total = sum(agreement[k][9] for k in described)
    assert total == pytest.approx(0.0, abs=1e-12)
    # and it must still separate the modes rather than zeroing everything
    assert max(abs(agreement[k][9]) for k in described) > 1e-3


def test_centring_leaves_the_broadest_approach_no_advantage(templates):
    """The mode with the flattest notated approach must not win on every input."""
    profiles = {k: v.cadence_array() for k, v in templates.items()}
    described = [k for k, v in profiles.items() if v is not None]
    breadth = {k: float(np.exp(-np.sum(profiles[k] * np.log(profiles[k] + 1e-12))))
               for k in described}
    broadest = max(breadth, key=breadth.get)

    wins = 0
    rng = np.random.default_rng(3)
    for _ in range(25):
        resolution = int(rng.integers(0, N))
        approach = [int(rng.integers(0, N)) for _ in range(4)]
        forud, events = cadence(resolution, approach)
        agreement = cadence_agreement([forud], events, profiles)
        winner = max(described, key=lambda k: agreement[k][resolution])
        wins += winner == broadest
    assert wins < 20, f"{broadest} won {wins}/25 random approaches"
