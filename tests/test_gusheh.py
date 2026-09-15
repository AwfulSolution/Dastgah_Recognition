"""Tests for gusheh identification."""

from pathlib import Path

import numpy as np
import pytest

from dastgah.radif.gusheh import (
    GushehTemplate,
    identify_gusheh,
    load_gusheh_templates,
    match_name,
    normalise_name,
    tessitura,
)

TEMPLATES = Path(__file__).resolve().parent.parent / "dastgah" / "data" / "gushehs.json"


@pytest.fixture(scope="module")
def gushehs():
    return load_gusheh_templates(TEMPLATES)


# --- name matching ----------------------------------------------------------

@pytest.mark.parametrize(
    ("a", "b"),
    [
        ("Ouj", "owj"),
        ("Daramad", "daraamad"),
        ("Hoseyni", "hosseini"),
        ("Chaharmezrab", "chahaarmezraab"),
    ],
)
def test_romanisation_variants_normalise_together(a, b):
    assert normalise_name(a) == normalise_name(b)


def test_spellings_normalisation_cannot_unify_are_caught_by_fuzzy_matching():
    """`zirkash` vs `zirkeshe` differ by vowels no rule should collapse."""
    candidates = {normalise_name("Zirkeshe Salmak"): "Zirkeshe Salmak"}
    assert normalise_name("Zirkeshe Salmak") != normalise_name("zirkash_salmak")
    assert match_name("zirkash_salmak", candidates) == "Zirkeshe Salmak"


def test_parentheticals_and_particles_are_dropped():
    assert normalise_name("Gusheh Grayli (daramad grayli)") == normalise_name("gusheh grayli")


def test_word_order_does_not_matter():
    assert normalise_name("Salmak Zirkeshe") == normalise_name("Zirkeshe Salmak")


def test_match_name_finds_close_spellings():
    candidates = {normalise_name(n): n for n in ["Zirkeshe Salmak", "Shahnaz", "Razavi"]}
    assert match_name("zirkash_salmak", candidates) == "Zirkeshe Salmak"
    assert match_name("shahnaaz", candidates) == "Shahnaz"


def test_match_name_rejects_unrelated_names():
    candidates = {normalise_name(n): n for n in ["Shahnaz", "Razavi"]}
    assert match_name("Chahargah", candidates) is None
    assert match_name("", candidates) is None


# --- tessitura --------------------------------------------------------------

def test_tessitura_is_signed_around_the_nearest_tonic():
    """A melody just below the tonic must read negative, not almost an octave up."""
    durations = np.ones(3)
    below = tessitura(np.array([118.0, 119.0, 120.0]), durations, 120 % 24)
    assert -12 < below < 0


def test_tessitura_is_octave_invariant():
    durations = np.ones(4)
    low = np.array([120.0, 124.0, 126.0, 130.0])
    assert tessitura(low, durations, 0) == pytest.approx(
        tessitura(low + 24, durations, 0)
    )


def test_tessitura_tracks_where_the_melody_sits():
    durations = np.ones(3)
    near = tessitura(np.array([120.0, 121.0, 122.0]), durations, 0)
    high = tessitura(np.array([128.0, 129.0, 130.0]), durations, 0)
    assert high > near


def test_tessitura_of_nothing_is_zero():
    assert tessitura(np.zeros(0), np.zeros(0), 0) == 0.0


# --- identification ---------------------------------------------------------

def test_corpus_ships_templates_for_every_mode(gushehs):
    assert len(gushehs) == 13
    assert sum(len(v) for v in gushehs.values()) == 229
    for group in gushehs.values():
        for template in group:
            assert len(template.profile) == 24
            assert np.isclose(sum(template.profile), 1.0)
            assert -12.0 <= template.tessitura <= 12.0


def test_a_gusheh_identifies_itself(gushehs):
    """Each template scored against its own profile should rank first."""
    for mode, group in gushehs.items():
        if len(group) < 2:
            continue
        for template in group[:4]:
            ranked = identify_gusheh(
                template.as_array(), template.tessitura, group, tessitura_weight=1.0
            )
            assert ranked[0].name == template.name, f"{mode}/{template.name}"


def test_identification_returns_every_candidate_ranked(gushehs):
    group = gushehs["shur"]
    ranked = identify_gusheh(np.ones(24), 0.0, group)
    assert len(ranked) == len(group)
    scores = [m.score for m in ranked]
    assert scores == sorted(scores, reverse=True)


def test_no_candidates_yields_no_matches():
    assert identify_gusheh(np.ones(24), 0.0, []) == []


def test_tessitura_distance_wraps_around_the_octave():
    """Tessitura is an angle: +11.5 and -11.5 are close, not 23 apart."""
    profile = [1.0 / 24] * 24
    near = GushehTemplate("shur", "near", profile, 11.5, 1.0, 100)
    far = GushehTemplate("shur", "far", profile, 0.0, 1.0, 100)
    ranked = identify_gusheh(np.ones(24), -11.5, [near, far], tessitura_weight=4.0)
    assert ranked[0].name == "near"
