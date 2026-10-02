"""The fitted scorer must agree with the hand-built one, and its gradient."""

from __future__ import annotations

import numpy as np
import pytest

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.core.learn import (
    LEVELS,
    _objective,
    _pack,
    _mother_matrix,
    build_design,
    dastgah_probabilities,
    fit,
    initial_parameters,
    predict,
)
from dastgah.radif.templates import load_templates
from dastgah.theory import MODAL_CLASSES_BY_KEY

N = 24


@pytest.fixture(scope="module")
def templates():
    return load_templates(DEFAULT_TEMPLATE_PATH)


@pytest.fixture(scope="module")
def records(templates):
    """Synthetic recordings: each template's own profile, rotated to a tonic."""
    rng = np.random.default_rng(0)
    out = []
    for offset, (key, template) in enumerate(templates.items()):
        tonic = (5 * offset) % N
        profile = np.roll(template.as_array(), tonic)
        matrix = template.transition_array()
        bigrams = None
        if matrix is not None:
            bigrams = np.roll(np.roll(matrix, tonic, axis=0), tonic, axis=1)
        out.append(
            {
                "h": profile + 0.01 * rng.random(N),
                "B": bigrams,
                # truth is the dastgah an avaz folds into, as at inference
                "truth": MODAL_CLASSES_BY_KEY[key].parent or key,
            }
        )
    return out


def test_untrained_model_reproduces_the_hand_built_classifier(templates, records):
    """Theory's parameters must give theory's answers, or the layering is a lie."""
    design = build_design(records, templates, config=DEFAULT_CONFIG)
    parameters = initial_parameters(design, DEFAULT_CONFIG)
    probabilities, dastgahs = dastgah_probabilities(parameters, design)

    for index, record in enumerate(records):
        result = classify(
            np.asarray(record["h"], dtype=float),
            templates,
            transitions=record["B"],
            tonic_prior=None,
            config=DEFAULT_CONFIG,
        )
        expected = dict(result.ranked_dastgahs(dastgahs))
        got = dict(zip(dastgahs, probabilities[index], strict=True))
        for key in dastgahs:
            assert got[key] == pytest.approx(expected.get(key, 0.0), abs=2e-6)


@pytest.mark.parametrize("level", LEVELS)
def test_analytic_gradient_matches_finite_differences(templates, records, level):
    design = build_design(records, templates)
    _, dastgahs = _mother_matrix(design)
    target = np.array([dastgahs.index(r["truth"]) for r in records])

    rng = np.random.default_rng(1)
    start = _pack(initial_parameters(design), level)
    start = start + 0.05 * rng.standard_normal(start.shape)

    loss, gradient = _objective(start, level, design, target, 1.0)
    assert np.isfinite(loss)

    step = 1e-6
    probe = rng.choice(start.size, size=min(12, start.size), replace=False)
    for index in probe:
        shifted = start.copy()
        shifted[index] += step
        up, _ = _objective(shifted, level, design, target, 1.0)
        shifted[index] -= 2 * step
        down, _ = _objective(shifted, level, design, target, 1.0)
        numeric = (up - down) / (2 * step)
        assert gradient[index] == pytest.approx(numeric, rel=2e-4, abs=2e-7)


def test_fitting_cannot_worsen_the_training_likelihood(templates, records):
    design = build_design(records, templates)
    _, dastgahs = _mother_matrix(design)
    target = np.array([dastgahs.index(r["truth"]) for r in records])
    truth = [r["truth"] for r in records]

    before, _ = _objective(_pack(initial_parameters(design), "bias"), "bias", design, target, 1.0)
    parameters = fit(design, truth, level="bias")
    after, _ = _objective(_pack(parameters, "bias"), "bias", design, target, 1.0)
    assert after <= before + 1e-9


def test_rotating_a_recording_leaves_its_prediction_alone(templates, records):
    """Absolute pitch is summed out, so transposition cannot change the answer.

    This is the property that makes the architecture safe to train: the failure
    mode where a model learns instrument tuning is unavailable to it.
    """
    design = build_design(records, templates)
    parameters = fit(design, [r["truth"] for r in records], level="bias")
    chosen, dastgahs = predict(parameters, design)

    shifted = [
        {"h": np.roll(np.asarray(r["h"]), 7),
         "B": None if r["B"] is None else np.roll(np.roll(r["B"], 7, axis=0), 7, axis=1),
         "truth": r["truth"]}
        for r in records
    ]
    moved = build_design(shifted, templates)
    chosen_moved, _ = predict(parameters, moved)
    assert list(chosen) == list(chosen_moved)


def test_labels_outside_the_answer_space_are_refused(templates, records):
    design = build_design(records, templates)
    truth = [r["truth"] for r in records]
    with pytest.raises(ValueError, match="outside the answer space"):
        fit(design, ["not_a_dastgah"] + truth[1:], level="weights")


def test_balanced_weighting_changes_the_fit(templates, records):
    """Equal weight per dastgah must not reduce to equal weight per recording."""
    design = build_design(records, templates)
    truth = [r["truth"] for r in records]
    plain = fit(design, truth, level="bias")
    balanced = fit(design, truth, level="bias", balanced=True)
    assert not np.allclose(plain.bias, balanced.bias)


def test_progression_starts_neutral_and_can_be_learned(templates, records):
    """A zero initial weight is what makes the fitted value a measurement.

    With the term switched off at initialisation, an unfitted model is exactly
    the hand-built classifier, and whatever the fit moves the weight to is the
    contribution of order over and above pitch content.
    """
    from dastgah.core.analyze import DEFAULT_GUSHEH_PATH
    from dastgah.core.seyr import progression_scores
    from dastgah.radif.gusheh import load_gusheh_templates

    gushehs = load_gusheh_templates(DEFAULT_GUSHEH_PATH)
    assert initial_parameters(build_design(records, templates)).progression_weight == 0.0

    # Give each record a traversal of its own dastgah's gushehs in radif order.
    windowed = []
    for record in records:
        mode = next(
            k for k in templates
            if (MODAL_CLASSES_BY_KEY[k].parent or k) == record["truth"]
            and len(gushehs.get(k, [])) >= 4
        )
        windowed.append(
            {**record, "W": np.array([t.as_array() for t in gushehs[mode]])}
        )

    design = build_design(windowed, templates, gusheh_templates=gushehs)
    assert np.any(design.progression != 0.0), "progression term never populated"

    # penalty=0: the shrinkage toward theory now pulls this weight back to its
    # zero anchor, so measuring the term's own contribution needs it switched off.
    fitted = fit(design, [r["truth"] for r in windowed], level="weights", penalty=0.0)
    assert fitted.progression_weight > 0.0


def test_records_without_windows_score_zero_progression(templates, records):
    from dastgah.core.analyze import DEFAULT_GUSHEH_PATH
    from dastgah.radif.gusheh import load_gusheh_templates

    gushehs = load_gusheh_templates(DEFAULT_GUSHEH_PATH)
    design = build_design(records, templates, gusheh_templates=gushehs)
    assert np.all(design.progression == 0.0)


def test_select_carries_every_per_recording_field(templates, records):
    """A grouped split must not quietly drop a term from one side of it."""
    import dataclasses

    from dastgah.core.analyze import DEFAULT_GUSHEH_PATH
    from dastgah.radif.gusheh import load_gusheh_templates

    gushehs = load_gusheh_templates(DEFAULT_GUSHEH_PATH)
    windowed = [
        {**r, "W": np.array([t.as_array() for t in gushehs["shur"]])} for r in records
    ]
    design = build_design(windowed, templates, gusheh_templates=gushehs)

    mask = np.zeros(design.n, dtype=bool)
    mask[::2] = True
    part = design.select(mask)

    per_record = {"rotated", "transition", "log_prior", "progression"}
    assert per_record <= {f.name for f in dataclasses.fields(design)}
    for name in per_record:
        got, want = getattr(part, name), getattr(design, name)[mask]
        assert got.shape == want.shape, name
        assert np.array_equal(got, want), name
    assert part.n == mask.sum()


def test_answer_space_drops_unanswerable_modes_but_keeps_avazes(templates, records):
    """Training and inference must agree on which hypotheses exist.

    An unanswerable mode left in the design would train the fit to push
    probability away from it, work that inference discards by renormalising.
    """
    from dastgah.radif.templates import DASTGAHS_WITH_AUDIO

    full = build_design(records, templates)
    assert "rast_panjgah" in full.modes

    restricted = build_design(records, templates, answer_space=DASTGAHS_WITH_AUDIO)
    assert "rast_panjgah" not in restricted.modes
    assert set(restricted.mothers) == set(DASTGAHS_WITH_AUDIO)
    # Shur's avazes survive, since folding beats dropping.
    assert "dashti" in restricted.modes
    assert restricted.n_modes < full.n_modes

    probabilities, dastgahs = dastgah_probabilities(
        initial_parameters(restricted), restricted
    )
    assert dastgahs == sorted(DASTGAHS_WITH_AUDIO)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_an_empty_answer_space_is_refused(templates, records):
    with pytest.raises(ValueError, match="no template folds into"):
        build_design(records, templates, answer_space=["not_a_dastgah"])


def test_a_large_penalty_returns_the_hand_built_classifier(templates, records):
    """Shrinkage toward theory must actually reach theory in the limit.

    This is what makes the penalty interpretable: it interpolates between the
    notated radif and the fitted model rather than between the fitted model and
    something arbitrary.
    """
    design = build_design(records, templates)
    truth = [r["truth"] for r in records]
    theory = _pack(initial_parameters(design), "profiles")
    fitted = _pack(fit(design, truth, level="profiles", penalty=1e6), "profiles")
    assert np.allclose(fitted, theory, atol=1e-3)


@pytest.mark.parametrize("level", LEVELS)
def test_the_penalty_gradient_matches_finite_differences(templates, records, level):
    design = build_design(records, templates)
    _, dastgahs = _mother_matrix(design)
    target = np.array([dastgahs.index(r["truth"]) for r in records])

    rng = np.random.default_rng(7)
    anchor = _pack(initial_parameters(design), level)
    point = anchor + 0.1 * rng.standard_normal(anchor.shape)
    _, gradient = _objective(point, level, design, target, 0.37, None, anchor)

    step = 1e-6
    for index in rng.choice(point.size, size=min(10, point.size), replace=False):
        probe = point.copy()
        probe[index] += step
        up, _ = _objective(probe, level, design, target, 0.37, None, anchor)
        probe[index] -= 2 * step
        down, _ = _objective(probe, level, design, target, 0.37, None, anchor)
        assert gradient[index] == pytest.approx((up - down) / (2 * step), rel=2e-4, abs=2e-7)
