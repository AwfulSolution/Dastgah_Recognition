"""Match observed pitch content against the modal templates.

The tonic of an unseen performance is unknown, so classification searches
jointly over the 13 modal classes and the 24 possible tonics. Each pairing is
scored and the best one yields both the mode and its tonic.

The score combines three terms:

* the log-likelihood of the pitch-class distribution under the mode's profile;
* the log-likelihood of note-to-note motion under its transition matrix, which
  is what separates an avaz from the dastgah it shares a scale with;
* a prior favouring tonics that carry sustained weight in the performance.

Profiles are raised to :attr:`ScoringConfig.sharpen` before scoring. Without
this, modes with broad templates (Rast-Panjgah and Mahur, which absorb many
modulations) attract everything, because a permissive distribution assigns
respectable likelihood to any input. Sharpening also compensates for the
observation being blurrier than the template: an f0 histogram carries ornament,
glissando and binning leakage that the notated radif does not.

Note that raising the exponent above 1.0 gives up an exactness property. At
``sharpen == 1.0`` cross-entropy is maximised when the template equals the
observation, so a template always recovers itself; above 1.0 the best match for
an observation is a slightly flatter template. That is the intended trade, but
it means self-recovery only holds exactly at 1.0.

Observed features may come from symbolic notes or from an audio f0 contour;
this module only requires 24 bins of relative weight.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from dastgah.radif.templates import ModalTemplate
from dastgah.theory import (
    MODAL_CLASSES_BY_KEY,
    QUARTER_TONES_PER_OCTAVE,
    ModalClass,
    pitch_class_name,
)

N = QUARTER_TONES_PER_OCTAVE


@dataclass(frozen=True)
class ScoringConfig:
    """Weights for the composite score.

    Defaults were selected by leave-one-out over the 229 Radif Corpus gushehs;
    because they were tuned on that same split, treat the corpus accuracy as
    optimistic rather than as a held-out estimate.
    """

    sharpen: float = 3.0
    #: Share of the tonic prior taken from detected cadences rather than
    #: sounding time. Cadence evidence is sparser but points at the ist
    #: instead of the shahed, so the two are blended rather than swapped.
    forud_prior_weight: float = 0.30
    #: Later cadences count for more; only the closing forud need return to
    #: the principal tonic.
    forud_recency_halflife: float = 60.0
    #: Weight on the note-transition term relative to pitch content.
    transition_weight: float = 0.5
    #: Weight on the tonic prior, which blends sounding time with cadences.
    tonic_prior_weight: float = 0.25
    #: Softmax temperature, calibrated so reported confidence tracks observed
    #: accuracy. Fitted for the default answer space, where each avaz folds into
    #: its mother dastgah: 0.4 gives a +0.1 point confidence gap on the archive
    #: and -2.3 on IRMA, the two sets agreeing independently on the same value.
    #: Reporting all 13 classes instead leaves the result mildly overconfident,
    #: since folding redistributes probability over fewer answers.
    temperature: float = 0.4


DEFAULT_CONFIG = ScoringConfig()


@dataclass
class Candidate:
    """One scored (mode, tonic) hypothesis."""

    key: str
    tonic_pc: int
    score: float
    probability: float = 0.0
    family: str = ""

    @property
    def modal_class(self) -> ModalClass:
        return MODAL_CLASSES_BY_KEY[self.key]

    @property
    def tonic_name(self) -> str:
        return pitch_class_name(self.tonic_pc)


@dataclass
class Classification:
    """Ranked result of matching one performance against all templates."""

    candidates: list[Candidate] = field(default_factory=list)

    @property
    def best(self) -> Candidate:
        """Highest-scoring (mode, tonic) pairing."""
        return self.candidates[0]

    def top(self, n: int = 5) -> list[Candidate]:
        return self.candidates[:n]

    def ranked_classes(self) -> list[tuple[str, float]]:
        """Modal classes with probability marginalised over tonics, best first."""
        totals: dict[str, float] = {}
        for candidate in self.candidates:
            totals[candidate.key] = totals.get(candidate.key, 0.0) + candidate.probability
        return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

    def probability_of(self, key: str) -> float:
        return dict(self.ranked_classes()).get(key, 0.0)

    def ranked_dastgahs(
        self, allowed: "Iterable[str] | None" = None
    ) -> list[tuple[str, float]]:
        """Probability per dastgah, with each avaz folded into its mother.

        An avaz is a branch of its parent dastgah rather than a rival to it, so
        evidence for Dashti is evidence for Shur. Folding the probability rather
        than dropping the avaz templates keeps that evidence: measured on
        recordings that contain no avaz at all, folding still beats leaving the
        templates out, because an avaz profile detects its parent's territory
        that the parent's own profile covers less well.

        ``allowed`` optionally restricts the answer space, in which case the
        remaining probability is renormalised over it.
        """
        totals: dict[str, float] = {}
        for candidate in self.candidates:
            modal = MODAL_CLASSES_BY_KEY[candidate.key]
            mother = modal.parent or candidate.key
            if allowed is not None and mother not in allowed:
                continue
            totals[mother] = totals.get(mother, 0.0) + candidate.probability

        total = sum(totals.values())
        if total > 0:
            totals = {k: v / total for k, v in totals.items()}
        return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)

    def best_for_dastgah(self, key: str) -> Candidate:
        """Highest-scoring hypothesis belonging to a dastgah or one of its avazes.

        Used for the tonic, which must come from whichever profile actually
        matched rather than from the mother's by default.
        """
        belonging = [
            c
            for c in self.candidates
            if (MODAL_CLASSES_BY_KEY[c.key].parent or c.key) == key
        ]
        if not belonging:
            raise KeyError(f"no candidate belongs to {key!r}")
        return max(belonging, key=lambda c: c.score)

    def ranked_families(self) -> list[tuple[str, float]]:
        """Mode families with probability marginalised over their members.

        A pitch-class profile separates families far more reliably than the
        modes inside one, so this is the answer to lead with.
        """
        totals: dict[str, float] = {}
        for candidate in self.candidates:
            family = candidate.family or candidate.key
            totals[family] = totals.get(family, 0.0) + candidate.probability
        return sorted(totals.items(), key=lambda kv: kv[1], reverse=True)


def _normalize(values: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    total = values.sum()
    if total <= 0:
        return np.full(values.shape, 1.0 / values.size)
    smoothed = values / total + epsilon
    return smoothed / smoothed.sum()


def _softmax(scores: np.ndarray, temperature: float) -> np.ndarray:
    scaled = scores / max(temperature, 1e-9)
    scaled -= scaled.max()
    exponentiated = np.exp(scaled)
    return exponentiated / exponentiated.sum()


def rotate_to_tonic(histogram: np.ndarray, tonic_pc: int) -> np.ndarray:
    """Re-index an absolute pitch-class histogram as intervals above a tonic."""
    return np.roll(histogram, -tonic_pc)


def classify(
    histogram: np.ndarray,
    templates: dict[str, ModalTemplate],
    *,
    transitions: np.ndarray | None = None,
    tonic_prior: np.ndarray | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
) -> Classification:
    """Rank every (mode, tonic) pairing for observed pitch content.

    ``histogram`` is 24 bins of absolute pitch-class weight, normalised
    internally. ``transitions`` is an optional 24x24 matrix of absolute
    pitch-class bigram counts; when omitted the transition term is skipped.

    ``tonic_prior`` optionally supplies 24 bins of belief about where the tonic
    lies, for instance from cadence detection. Without it the prior falls back to
    sounding time, which is a poor proxy: the most-sounded degree is usually the
    *shahed*, not the tonic.
    """
    observed = np.asarray(histogram, dtype=float)
    if observed.shape != (N,):
        raise ValueError(f"expected {N} histogram bins, got {observed.shape}")
    if observed.sum() <= 0:
        raise ValueError("histogram is empty; no pitch content to classify")
    observed = _normalize(observed)

    observed_transitions = None
    if transitions is not None:
        matrix = np.asarray(transitions, dtype=float)
        if matrix.shape != (N, N):
            raise ValueError(f"expected {N}x{N} transitions, got {matrix.shape}")
        if matrix.sum() > 0:
            observed_transitions = _normalize(matrix.ravel()).reshape(N, N)

    if tonic_prior is None:
        prior = observed
    else:
        prior = np.asarray(tonic_prior, dtype=float)
        if prior.shape != (N,):
            raise ValueError(f"expected {N} tonic prior bins, got {prior.shape}")
        total_prior = prior.sum()
        prior = observed if total_prior <= 0 else prior / total_prior
    log_tonic_prior = np.log(prior + 1e-9)

    candidates: list[Candidate] = []
    for key, template in templates.items():
        profile = template.as_array() ** config.sharpen
        log_profile = np.log(profile / profile.sum())

        log_transitions = None
        if observed_transitions is not None and config.transition_weight:
            template_transitions = template.transition_array()
            if template_transitions is not None:
                log_transitions = np.log(template_transitions)

        for tonic_pc in range(N):
            rotated = rotate_to_tonic(observed, tonic_pc)
            score = float(np.dot(rotated, log_profile))

            if log_transitions is not None:
                rotated_transitions = np.roll(
                    np.roll(observed_transitions, -tonic_pc, axis=0), -tonic_pc, axis=1
                )
                score += config.transition_weight * float(
                    np.sum(rotated_transitions * log_transitions)
                )

            score += config.tonic_prior_weight * float(log_tonic_prior[tonic_pc])
            candidates.append(
                Candidate(
                    key=key,
                    tonic_pc=tonic_pc,
                    score=score,
                    family=template.family or key,
                )
            )

    scores = np.array([c.score for c in candidates])
    for candidate, probability in zip(
        candidates, _softmax(scores, config.temperature), strict=True
    ):
        candidate.probability = float(probability)

    candidates.sort(key=lambda c: c.score, reverse=True)
    return Classification(candidates=candidates)
