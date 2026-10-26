"""The deployed second opinion: it must fire only where trained, and never
silently disagree with the model that was measured."""

from __future__ import annotations

import json

import numpy as np
import pytest

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.cascade import (
    DEFAULT_PAIR_MODEL_PATH,
    PairModel,
    load_pair_models,
    pair_features,
    second_opinion,
)
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

SPACE = list(DASTGAHS_WITH_AUDIO)


@pytest.fixture(scope="module")
def templates():
    return load_templates(DEFAULT_TEMPLATE_PATH)


@pytest.fixture(scope="module")
def models():
    return load_pair_models()


def observed(templates, key="shur", tonic=5):
    """A recording-like histogram built from a mode's own profile."""
    rng = np.random.default_rng(0)
    return np.roll(templates[key].as_array(), tonic) + 0.01 * rng.random(24)


def context(histogram, templates):
    result = classify(histogram, templates, config=DEFAULT_CONFIG)
    ranked = result.ranked_dastgahs(SPACE)
    tonics, modes = {}, {}
    for key in SPACE:
        candidate = result.best_for_dastgah(key)
        tonics[key], modes[key] = candidate.tonic_pc, candidate.key
    return ranked, tonics, modes


def test_the_shipped_models_load_and_cover_the_known_pairs(models):
    assert models, "no trained pair models; run scripts/train_pair_models.py"
    pairs = {tuple(sorted(m.pair)) for m in models.values()}
    assert ("nava", "shur") in pairs
    assert ("chahargah", "homayun") in pairs


def test_every_model_matches_the_features_it_will_be_given(models, templates):
    """The parity check: a model's width must equal what pair_features produces.

    Training and serving share one feature function, so this fails only if the
    artefact is stale -- which is exactly the failure that would otherwise be
    invisible until a deployed answer was quietly wrong.
    """
    histogram = observed(templates)
    ranked, tonics, modes = context(histogram, templates)
    for model in models.values():
        built = pair_features(
            histogram, dict(ranked), tonics, modes, model.pair, templates,
            which=model.features,
        )
        assert built.shape == model.coefficients.shape, (
            f"{model.pair}: artefact has {model.coefficients.shape[0]} "
            f"coefficients, features produce {built.shape[0]}"
        )


def test_it_fires_only_when_the_top_two_are_a_trained_pair(models, templates):
    histogram = observed(templates)
    ranked, tonics, modes = context(histogram, templates)

    out = second_opinion(ranked, histogram, tonics, modes, templates, models)
    covered = frozenset((ranked[0][0], ranked[1][0])) in models
    assert (out is not None) == covered

    # A top two with no model must return nothing rather than guessing.
    missing = [("shur", 0.6), ("segah", 0.4)]
    empty = second_opinion(missing, histogram, tonics, modes, templates, {})
    assert empty is None


def test_it_reports_agreement_against_the_shipped_answer(models, templates):
    histogram = observed(templates)
    ranked, tonics, modes = context(histogram, templates)
    out = second_opinion(ranked, histogram, tonics, modes, templates, models)
    if out is None:
        pytest.skip("this fixture's top two are not a trained pair")
    assert out["choice"] in out["pair"]
    assert out["agrees"] == (out["choice"] == ranked[0][0])
    assert 0.5 <= out["confidence"] <= 1.0


def test_a_stale_artefact_is_refused_rather_than_misread(templates):
    """A model of the wrong width must raise, not silently score garbage."""
    histogram = observed(templates)
    ranked, tonics, modes = context(histogram, templates)
    top = frozenset((ranked[0][0], ranked[1][0]))
    broken = {
        top: PairModel(
            pair=(ranked[0][0], ranked[1][0]), features="all",
            coefficients=np.ones(3), intercept=0.0,
        )
    }
    with pytest.raises(ValueError, match="features|drifted"):
        second_opinion(ranked, histogram, tonics, modes, templates, broken)


def test_the_artefact_records_what_it_was_fitted_on(models):
    """Provenance ships with the model, so a caller cannot present it as neutral."""
    payload = json.loads(DEFAULT_PAIR_MODEL_PATH.read_text())
    assert "lose on every other corpus" in payload["note"]
    for model in models.values():
        assert model.trained_on
        assert model.n_recordings > 0


def test_decide_is_a_plain_logistic_and_needs_no_sklearn(templates):
    model = PairModel(pair=("a", "b"), features="all",
                      coefficients=np.array([1.0, -1.0]), intercept=0.0)
    assert model.decide(np.array([1.0, 0.0]))[0] == "a"
    assert model.decide(np.array([0.0, 1.0]))[0] == "b"
    _, confidence = model.decide(np.array([0.0, 0.0]))
    assert confidence == pytest.approx(0.5)


def test_the_payload_is_json_serialisable(models, templates):
    """numpy scalars survive dataclass conversion and die at json.dumps."""
    histogram = observed(templates)
    ranked, tonics, modes = context(histogram, templates)
    out = second_opinion(ranked, histogram, tonics, modes, templates, models)
    if out is None:
        pytest.skip("this fixture's top two are not a trained pair")
    json.dumps(out)
    assert type(out["confidence"]) is float
