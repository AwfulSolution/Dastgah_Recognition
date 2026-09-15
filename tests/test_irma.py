"""Tests for the IRMA contour reader. Corpus-dependent tests skip if absent."""

from pathlib import Path

import numpy as np
import pytest

from dastgah.radif.irma import (
    IRMA_DASTGAH_KEYS,
    frame_durations,
    load_contours,
    parse_filename,
    to_pitch_track,
)
from dastgah.theory import MODAL_CLASSES_BY_KEY

IRMA_ROOT = Path(__file__).resolve().parent.parent / "data" / "raw" / "irma"
needs_irma = pytest.mark.skipif(not IRMA_ROOT.exists(), reason="IRMA not downloaded")


def test_every_book_maps_to_a_real_modal_class():
    assert len(IRMA_DASTGAH_KEYS) == 13
    assert set(IRMA_DASTGAH_KEYS.values()) == set(MODAL_CLASSES_BY_KEY)


def test_irma_numbering_differs_from_the_radif_corpus():
    """IRMA's D-numbers are its own; D2 is Abu'ata, not the corpus' order."""
    assert IRMA_DASTGAH_KEYS["D2"] == "abuata"
    assert IRMA_DASTGAH_KEYS["D3"] == "bayat_e_tork"  # listed as Bayat-e Zand
    assert IRMA_DASTGAH_KEYS["D9"] == "bayat_e_esfahan"


def test_parses_a_contour_filename():
    fields = parse_filename("D11_G204_M1_A1_T1_pitch_3 Chahargah_zaabol")
    assert fields["key"] == "chahargah"
    assert fields["gusheh_id"] == "204"
    assert fields["gusheh_name"] == "zaabol"
    assert fields["kind"] == "pitch"


def test_rejects_unparseable_names():
    assert parse_filename("notes.csv") is None
    assert parse_filename("D99_G1_M1_A1_T1_pitch_1 Nowhere_x") is None


def test_frame_durations_cap_silences():
    """A gap in the time column is a rest, not a very long note."""
    times = np.array([0.0, 0.006, 0.012, 5.0, 5.006])
    durations = frame_durations(times)
    assert len(durations) == len(times)
    assert durations.max() < 0.02
    assert np.isclose(durations[0], 0.006)


def test_frame_durations_handle_degenerate_input():
    assert frame_durations(np.zeros(0)).size == 0
    assert frame_durations(np.array([1.0])).size == 1


@needs_irma
def test_loads_contours_across_modes():
    contours = load_contours(IRMA_ROOT)
    assert len(contours) > 100
    assert len({c.key for c in contours}) >= 10
    assert all(len(c) > 0 for c in contours)


@needs_irma
def test_contour_converts_to_a_usable_pitch_track():
    contour = load_contours(IRMA_ROOT)[0]
    track = to_pitch_track(contour)
    assert track.voiced.any()
    assert 400.0 < track.reference_hz < 480.0
    assert 0.0 <= track.tuning_concentration <= 1.0
    assert track.duration > 0
