import pytest

from dastgah.theory import (
    MODAL_CLASSES,
    MODAL_CLASSES_BY_KEY,
    PitchParseError,
    cents_between,
    hz_to_quarter_tones,
    is_microtonal,
    modal_class_from_dirname,
    parse_pitch_label,
    pitch_class,
    pitch_class_name,
)


@pytest.mark.parametrize(
    ("label", "expected"),
    [("F", 106), ("G", 110), ("Ak", 113), ("Bb", 116), ("C", 120), ("D", 124), ("E", 128)],
)
def test_parses_the_corpus_reference_octave(label, expected):
    assert parse_pitch_label(label) == expected


def test_octave_suffixes_shift_by_a_full_octave():
    assert parse_pitch_label("G+1") - parse_pitch_label("G") == 24
    assert parse_pitch_label("C-1") - parse_pitch_label("C") == -24


def test_koron_lowers_and_sori_raises_by_a_quarter_tone():
    assert parse_pitch_label("A") - parse_pitch_label("Ak") == 1
    assert parse_pitch_label("As") - parse_pitch_label("A") == 1


def test_natural_suffix_is_a_no_op():
    assert parse_pitch_label("DN") == parse_pitch_label("D")


def test_rejects_nonsense_labels():
    with pytest.raises(PitchParseError):
        parse_pitch_label("H#")


def test_only_odd_pitch_classes_are_microtonal():
    assert is_microtonal(pitch_class(parse_pitch_label("Ak")))
    assert not is_microtonal(pitch_class(parse_pitch_label("A")))


def test_pitch_class_names_round_trip():
    for pc in range(24):
        assert pitch_class(parse_pitch_label(pitch_class_name(pc))) == pc


def test_a440_sits_at_quarter_tone_138():
    assert hz_to_quarter_tones(440.0) == pytest.approx(138.0)
    assert hz_to_quarter_tones(880.0) == pytest.approx(162.0)


def test_a_quarter_tone_is_fifty_cents():
    assert cents_between(100, 101) == pytest.approx(50.0)


def test_taxonomy_is_seven_dastgah_and_six_avaz():
    assert len(MODAL_CLASSES) == 13
    assert sum(m.kind == "dastgah" for m in MODAL_CLASSES) == 7
    assert sum(m.kind == "avaz" for m in MODAL_CLASSES) == 6


def test_every_avaz_points_at_a_real_parent_dastgah():
    for modal in MODAL_CLASSES:
        if modal.kind == "avaz":
            assert MODAL_CLASSES_BY_KEY[modal.parent].kind == "dastgah"


def test_resolves_corpus_directory_names():
    assert modal_class_from_dirname("01 - Shur").key == "shur"
    assert modal_class_from_dirname("13 - Rast - Panjgah").key == "rast_panjgah"


@pytest.mark.parametrize(
    ("written", "expected"),
    [
        ("Chahargah", "chahargah"),
        ("chahargah", "chahargah"),
        ("Shur", "shur"),
        ("Homayun", "homayun"),
        ("Homayoun", "homayun"),
        ("Mahoor", "mahur"),
        ("Rast-Panjgah", "rast_panjgah"),
        ("Rast Panjgah", "rast_panjgah"),
        ("bayat_e_tork", "bayat_e_tork"),
        ("Bayat-e Zand", "bayat_e_tork"),
        ("Esfahan", "bayat_e_esfahan"),
        ("Segāh", "segah"),
        ("Abū'atā", "abuata"),
    ],
)
def test_resolves_the_romanisations_datasets_actually_use(written, expected):
    from dastgah.theory import modal_class_from_name

    modal = modal_class_from_name(written)
    assert modal is not None, f"{written!r} did not resolve"
    assert modal.key == expected


def test_unknown_names_resolve_to_none():
    from dastgah.theory import modal_class_from_name

    assert modal_class_from_name("Hijaz") is None
    assert modal_class_from_name("") is None
