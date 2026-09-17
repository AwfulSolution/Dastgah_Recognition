from pathlib import Path

import numpy as np
import pytest

from dastgah.core.classify import DEFAULT_CONFIG, classify, rotate_to_tonic
from dastgah.radif.templates import load_templates

TEMPLATES = Path(__file__).resolve().parent.parent / "dastgah" / "data" / "templates.json"


@pytest.fixture(scope="module")
def templates():
    return load_templates(TEMPLATES)


def test_ships_all_thirteen_templates(templates):
    assert len(templates) == 13
    for template in templates.values():
        assert len(template.profile) == 24
        assert np.isclose(sum(template.profile), 1.0)


def test_transition_matrices_are_square_and_normalised(templates):
    for template in templates.values():
        matrix = template.transition_array()
        assert matrix.shape == (24, 24)
        assert np.isclose(matrix.sum(), 1.0)


def test_rotation_reindexes_relative_to_the_tonic():
    histogram = np.zeros(24)
    histogram[14] = 1.0
    assert rotate_to_tonic(histogram, 14)[0] == 1.0


def test_a_template_classifies_as_itself_at_its_own_tonic(templates):
    """Exact only at sharpen=1.0, where cross-entropy peaks at the template."""
    from dastgah.core.classify import ScoringConfig

    config = ScoringConfig(sharpen=1.0, transition_weight=0.0, tonic_prior_weight=0.0)
    for key, template in templates.items():
        histogram = np.roll(template.as_array(), template.tonic_pc)
        result = classify(histogram, templates, config=config)
        assert result.best.key == key, f"{key} did not recover itself"
        assert result.best.tonic_pc == template.tonic_pc


def test_classification_is_transposition_invariant(templates):
    """The same pitch content in any key must give the same mode."""
    template = templates["chahargah"]
    base = np.roll(template.as_array(), template.tonic_pc)
    for shift in range(24):
        result = classify(np.roll(base, shift), templates)
        assert result.best.key == "chahargah"
        assert result.best.tonic_pc == (template.tonic_pc + shift) % 24


def test_smoothing_leaves_every_distribution_normalised(templates):
    """Regression: smoothing was added without renormalising afterwards."""
    from dastgah.radif.templates import normalize

    for shape in [(24,), (24, 24)]:
        assert np.isclose(normalize(np.zeros(shape)).sum(), 1.0)
        assert np.isclose(normalize(np.ones(shape)).sum(), 1.0)
        assert np.isclose(normalize(np.random.default_rng(0).random(shape)).sum(), 1.0)


def test_marginal_probabilities_sum_to_one(templates):
    histogram = np.roll(templates["shur"].as_array(), 3)
    ranked = classify(histogram, templates).ranked_classes()
    assert np.isclose(sum(p for _, p in ranked), 1.0)
    assert len(ranked) == 13


def test_confidence_is_not_pinned_to_certainty(templates):
    """A flat histogram matches nothing well and must not report near-certainty."""
    result = classify(np.ones(24), templates)
    assert result.ranked_classes()[0][1] < 0.9


def test_rejects_empty_and_malformed_input(templates):
    with pytest.raises(ValueError, match="empty"):
        classify(np.zeros(24), templates)
    with pytest.raises(ValueError, match="24"):
        classify(np.ones(12), templates)
    with pytest.raises(ValueError, match="24x24"):
        classify(np.ones(24), templates, transitions=np.ones((12, 12)))


def test_sharpening_is_what_suppresses_permissive_templates(templates):
    """Rast-Panjgah has the broadest profile; without sharpening it over-attracts."""
    from dastgah.core.classify import ScoringConfig

    histogram = np.roll(templates["shur"].as_array(), templates["shur"].tonic_pc)
    flat = classify(histogram, templates, config=ScoringConfig(sharpen=1.0))
    sharp = classify(histogram, templates, config=DEFAULT_CONFIG)
    assert sharp.probability_of("shur") > flat.probability_of("shur")


def test_every_template_belongs_to_a_family(templates):
    for key, template in templates.items():
        assert template.family, f"{key} has no family"
        assert template.family in templates


def test_families_match_the_documented_grouping(templates):
    """Chahargah and Segah stand alone; the Shur family absorbs seven modes."""
    from dastgah.radif.templates import family_members

    members = family_members(templates)
    sizes = sorted(len(v) for v in members.values())
    assert sizes == [1, 1, 4, 7]

    by_key = {k: v for k, v in members.items()}
    solo = {k for k, v in by_key.items() if len(v) == 1}
    assert solo == {"chahargah", "segah"}

    shur_family = next(v for v in members.values() if len(v) == 7)
    assert {"shur", "nava", "dashti", "afshari"} <= set(shur_family)


def test_family_membership_is_symmetric_and_transitive(templates):
    """Families are connected components, so membership must be an equivalence."""
    from dastgah.radif.templates import family_members

    for family, keys in family_members(templates).items():
        for key in keys:
            assert templates[key].family == family


def test_aligned_similarity_finds_rotations(templates):
    """Shur and Nava share a pitch collection a fourth apart."""
    import numpy as np

    from dastgah.radif.templates import aligned_similarity

    direct = float(
        np.dot(templates["shur"].as_array(), templates["nava"].as_array())
        / (
            np.linalg.norm(templates["shur"].as_array())
            * np.linalg.norm(templates["nava"].as_array())
        )
    )
    assert aligned_similarity(templates["shur"], templates["nava"]) > 0.9
    assert aligned_similarity(templates["shur"], templates["nava"]) > direct


def test_a_template_is_maximally_similar_to_itself(templates):
    from dastgah.radif.templates import aligned_similarity

    for template in templates.values():
        assert aligned_similarity(template, template) == pytest.approx(1.0)


def test_family_probability_marginalises_over_members(templates):
    """Family probability must equal the sum over its modes, and total to one."""
    import numpy as np

    from dastgah.radif.templates import family_members

    histogram = np.roll(templates["shur"].as_array(), templates["shur"].tonic_pc)
    result = classify(histogram, templates)
    families = dict(result.ranked_families())
    classes = dict(result.ranked_classes())
    members = family_members(templates)

    assert np.isclose(sum(families.values()), 1.0)
    for family, keys in members.items():
        assert families[family] == pytest.approx(sum(classes[k] for k in keys), abs=1e-9)


def test_the_family_answer_is_never_less_confident_than_the_mode(templates):
    """A family contains its mode, so its probability is an upper bound."""
    import numpy as np

    rng = np.random.default_rng(0)
    for _ in range(20):
        result = classify(rng.random(24), templates)
        best = result.best
        families = dict(result.ranked_families())
        classes = dict(result.ranked_classes())
        assert families[best.family] >= classes[best.key] - 1e-9


def test_restricting_modes_recomputes_families(templates):
    """Over the six dastgahs with audio, only Shur and Nava stay grouped."""
    from dastgah.radif.templates import (
        DASTGAHS_WITH_AUDIO,
        family_members,
        restrict,
    )

    narrowed = restrict(templates, DASTGAHS_WITH_AUDIO)
    assert set(narrowed) == set(DASTGAHS_WITH_AUDIO)

    members = family_members(narrowed)
    sizes = sorted(len(v) for v in members.values())
    assert sizes == [1, 1, 1, 1, 2]
    pair = next(v for v in members.values() if len(v) == 2)
    assert set(pair) == {"shur", "nava"}


def test_restriction_leaves_the_original_untouched(templates):
    """Homayun groups with Mahur over all 13 and stands alone over six."""
    from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, restrict

    before = templates["homayun"].family
    narrowed = restrict(templates, DASTGAHS_WITH_AUDIO)
    assert templates["homayun"].family == before == "mahur"
    assert narrowed["homayun"].family == "homayun"


def test_restriction_does_not_change_the_answer_among_survivors(templates):
    """Scores are per-template, so narrowing cannot reorder what remains."""
    import numpy as np

    from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, restrict

    narrowed = restrict(templates, DASTGAHS_WITH_AUDIO)
    rng = np.random.default_rng(0)
    for _ in range(25):
        histogram = rng.random(24)
        full = [k for k, _ in classify(histogram, templates).ranked_classes()]
        survivor = next(k for k in full if k in DASTGAHS_WITH_AUDIO)
        assert classify(histogram, narrowed).ranked_classes()[0][0] == survivor


def test_restricting_to_nothing_known_is_an_error(templates):
    from dastgah.radif.templates import restrict

    with pytest.raises(ValueError, match="no known modes"):
        restrict(templates, ["not_a_mode"])
