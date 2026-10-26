"""A second opinion on the two dastgahs a recording is most likely to be.

The classifier ranks six dastgahs. When its top two are a pair it is known to
confuse, a model trained on *only that pair* is consulted. It fires nowhere
else, so a recording whose answer is a third dastgah is untouched.

**This is not the shipped answer, and should not be.** The pair models gain 1.8
points on the corpus they were fitted to and lose on every corpus they were not:
-3.2 on KDC, -6.9 on IRMA, and -5.5/-2.6/-3.8 under leave-one-corpus-out. An
upload from an unfamiliar source is exactly the case those numbers describe. The
cascade is carried so that a listener can see *where* the two disagree --
disagreements fall almost entirely on Shur/Nava and Chahargah/Homayun, which is
where a human ear is most useful and where this project most lacks data.

Inference needs no scikit-learn. A fitted logistic regression is a dot product
and a sigmoid, so :func:`load_pair_models` reads coefficients written by
``scripts/train_pair_models.py`` and evaluates them in numpy. Training keeps
scikit-learn; serving does not, which also keeps it inside a serverless
deployment's size limit.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dastgah.radif.templates import ModalTemplate
from dastgah.theory import QUARTER_TONES_PER_OCTAVE

N = QUARTER_TONES_PER_OCTAVE

#: Where trained pair models live, relative to the package.
DEFAULT_PAIR_MODEL_PATH = Path(__file__).resolve().parent.parent / "data" / "pair_models.json"


@dataclass(frozen=True)
class PairModel:
    """One fitted decision between two dastgahs."""

    pair: tuple[str, str]
    features: str
    coefficients: np.ndarray
    intercept: float
    #: What it was fitted on, carried so a caller can report provenance rather
    #: than implying the model is corpus-neutral. It is not.
    trained_on: str = ""
    n_recordings: int = 0

    def decide(self, features: np.ndarray) -> tuple[str, float]:
        """Which of the pair, and the confidence in it."""
        z = float(np.dot(features, self.coefficients) + self.intercept)
        probability = 1.0 / (1.0 + np.exp(-z))
        return (self.pair[0], probability) if probability >= 0.5 else (
            self.pair[1], 1.0 - probability
        )


def cadence_fit(
    foruds: "list", approaches: "list", tonics: dict[str, int],
    modes: dict[str, str], pair: tuple[str, str],
    templates: dict[str, ModalTemplate],
) -> np.ndarray:
    """Agreement between how cadences were approached and the mode's notated approach."""
    out = []
    for key in pair:
        template = templates.get(modes.get(key, key))
        profile = template.cadence_array() if template is not None else None
        usable = [np.asarray(a, dtype=float) for a in (approaches or []) if a is not None]
        if profile is None or not usable:
            out.append(0.0)
            continue
        norm = float(np.linalg.norm(profile))
        similarities = [
            float(np.dot(a, profile)) / max(float(np.linalg.norm(a)) * norm, 1e-9)
            for a in usable
        ]
        out.append(float(np.mean(similarities)))
    return np.array(out)


def pair_features(
    histogram: np.ndarray,
    probabilities: dict[str, float],
    tonics: dict[str, int],
    modes: dict[str, str],
    pair: tuple[str, str],
    templates: dict[str, ModalTemplate],
    which: str = "all",
    foruds: "list | None" = None,
    approaches: "list | None" = None,
) -> np.ndarray:
    """Features for choosing between two (mode, tonic) hypotheses.

    Each hypothesis supplies the observation re-indexed against *its own* tonic,
    so the model sees the same pitch content twice, read two ways. That is the
    whole question for a rotation-confusable pair.

    Defined here rather than in the training script so that training and serving
    cannot drift apart, which is the ordinary way a deployed model quietly stops
    matching the one that was measured.
    """
    first, second = pair
    histogram = np.asarray(histogram, dtype=float)
    histogram = histogram / max(histogram.sum(), 1e-9)

    blocks = []
    if which == "all":
        for key in (first, second):
            blocks.append(np.roll(histogram, -tonics.get(key, 0)))

    if which == "cadence":
        for key in (first, second):
            tonic = tonics.get(key, 0)
            shape, weight = np.zeros(N), 0.0
            for forud, approach in zip(foruds or [], approaches or []):
                if approach is None or forud.resolution_pc != tonic:
                    continue
                shape += forud.strength * np.asarray(approach, dtype=float)
                weight += forud.strength
            blocks.append(shape / weight if weight > 0 else shape)
            blocks.append(np.array([weight, float(weight > 0)]))
        return np.concatenate([
            *blocks, cadence_fit(foruds, approaches, tonics, modes, pair, templates)
        ])

    logged = [
        np.log(probabilities.get(first, 0.0) + 1e-9),
        np.log(probabilities.get(second, 0.0) + 1e-9),
    ]
    logged.append(logged[0] - logged[1])
    return np.concatenate([
        *blocks, logged,
        cadence_fit(foruds, approaches, tonics, modes, pair, templates),
    ])


def load_pair_models(path: Path = DEFAULT_PAIR_MODEL_PATH) -> dict[frozenset, PairModel]:
    """Trained pair models, keyed by the unordered pair they decide."""
    if not path.exists():
        return {}
    payload = json.loads(path.read_text())
    out = {}
    for entry in payload["models"]:
        pair = (entry["pair"][0], entry["pair"][1])
        out[frozenset(pair)] = PairModel(
            pair=pair,
            features=entry["features"],
            coefficients=np.asarray(entry["coefficients"], dtype=float),
            intercept=float(entry["intercept"]),
            trained_on=entry.get("trained_on", ""),
            n_recordings=int(entry.get("n_recordings", 0)),
        )
    return out


def second_opinion(
    ranked: list[tuple[str, float]],
    histogram: np.ndarray,
    tonics: dict[str, int],
    modes: dict[str, str],
    templates: dict[str, ModalTemplate],
    models: dict[frozenset, PairModel],
    foruds: "list | None" = None,
    approaches: "list | None" = None,
) -> "dict | None":
    """The pair model's view, when the top two are a pair it was trained for.

    Returns ``None`` when no model covers the top two, which is the common case
    and means there is nothing to disagree about.
    """
    if len(ranked) < 2:
        return None
    top = (ranked[0][0], ranked[1][0])
    model = models.get(frozenset(top))
    if model is None:
        return None

    features = pair_features(
        histogram, dict(ranked), tonics, modes, model.pair, templates,
        which=model.features, foruds=foruds, approaches=approaches,
    )
    if features.shape != model.coefficients.shape:
        raise ValueError(
            f"pair model for {model.pair} expects {model.coefficients.shape[0]} "
            f"features, got {features.shape[0]}; retrain or the definitions drifted"
        )
    choice, confidence = model.decide(features)
    return {
        "pair": list(model.pair),
        "choice": choice,
        "confidence": round(float(confidence), 4),
        "agrees": choice == ranked[0][0],
        "trained_on": model.trained_on,
        "n_recordings": model.n_recordings,
    }
