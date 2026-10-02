"""Fit the scoring function's parameters instead of setting them by hand.

:mod:`dastgah.core.classify` scores a (mode, tonic) pairing as

    alpha * sum_d  h_rot[d]   * log pi_m[d]
  + w_T   * sum_de B_rot[d,e] * log T_m[d,e]
  + w_P   * log prior[tonic]
  + b_m

with ``pi_m`` proportional to the notated radif profile raised to ``sharpen``,
and every weight chosen by hand. The same function is written here with the
weights, the exponent and the profiles themselves as free parameters, fitted by
maximising the likelihood of the correct dastgah.

Three properties of the hand-built system are kept, and they are what separates
this from training a classifier on the features directly:

**The tonic stays a latent variable.** The score is evaluated at all 24 tonics
and the dastgah's probability is the sum over them, exactly as at inference. No
recording needs a tonic label, and the tonic falls out of the fit.

**The pitch-rotation structure is architectural, not learned.** Every parameter
is tonic-relative, so a profile means the same thing at every absolute pitch.
A model that learns on absolute pitch classes scores 39% on unseen artists by
memorising instrument tuning and player register (see docs/data-notes.md); that
failure is not available here, because absolute pitch is summed out before any
parameter sees it.

**Theory is the initialisation and the prior.** Parameters start at the notated
radif values, so an untrained fit reproduces the current classifier, and profile
deviations are penalised toward it. What is learned is how far from Talai's
notation 39 performers actually sit.

The free parameters are deliberately layered, because the useful question is
not whether training helps but how little of the theory has to be given up:

``LEVELS``
    ``"weights"``   the four global scalars (17x fewer parameters than modes)
    ``"bias"``      plus one score offset per mode, which is what corrects a
                    mode being predicted more often than it occurs
    ``"sharpen"``   plus one exponent per mode, so a broad template can be
                    sharpened harder than a narrow one
    ``"profiles"``  plus a deviation per degree per mode, pulled toward theory
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp

from dastgah.core.classify import DEFAULT_CONFIG, ScoringConfig
from dastgah.core.seyr import MIN_WINDOWS, progression_scores
from dastgah.radif.templates import ModalTemplate
from dastgah.theory import MODAL_CLASSES_BY_KEY, QUARTER_TONES_PER_OCTAVE

N = QUARTER_TONES_PER_OCTAVE

#: Which parameters each level frees, cumulatively.
LEVELS = ("weights", "bias", "sharpen", "profiles")


@dataclass
class Design:
    """Precomputed per-recording terms that no parameter changes.

    Built once; every optimiser step reuses it. ``rotated`` is the dominant
    cost and the reason this is separated out.
    """

    #: (n, 24, 24) observed profile re-indexed against each candidate tonic.
    rotated: np.ndarray
    #: (n, 24, M) transition log-likelihood, from the theory transition matrices.
    transition: np.ndarray
    #: (n, 24) log tonic prior.
    log_prior: np.ndarray
    #: (n, 24, M) progression through the seyr; zeros when no windows were cached.
    progression: np.ndarray
    #: (M, 24) log of the notated radif profile, per mode.
    log_theory: np.ndarray
    #: Mode keys, indexing axis M.
    modes: list[str]
    #: Dastgah key each mode folds into (itself, for a dastgah).
    mothers: list[str]

    @property
    def n(self) -> int:
        return self.rotated.shape[0]

    def select(self, mask: "np.ndarray") -> "Design":
        """The same design over a subset of recordings, for grouped splits.

        A method rather than a helper beside the caller: every per-recording
        array has to be carried, and a field added later must not be silently
        dropped from one copy.
        """
        return Design(
            rotated=self.rotated[mask],
            transition=self.transition[mask],
            log_prior=self.log_prior[mask],
            progression=self.progression[mask],
            log_theory=self.log_theory,
            modes=self.modes,
            mothers=self.mothers,
        )

    @property
    def n_modes(self) -> int:
        return len(self.modes)


def build_design(
    records: "list[dict]",
    templates: dict[str, ModalTemplate],
    *,
    config: ScoringConfig = DEFAULT_CONFIG,
    gusheh_templates: "dict[str, list] | None" = None,
) -> Design:
    """Assemble the fixed terms for a corpus.

    Each record needs ``h`` (24 bins of absolute pitch-class weight) and may
    carry ``B`` (24x24 absolute bigrams) and ``log_prior`` (24 bins). Missing
    transitions contribute nothing rather than failing, which matches
    :func:`dastgah.core.classify.classify`.

    Passing ``gusheh_templates`` adds the progression term for any record
    carrying ``W``, a (windows, 24) grid in time order. Records without it score
    zero there and are decided on pitch content alone, so a corpus of excerpts
    costs nothing but gains nothing.
    """
    modes = list(templates)
    mothers = [MODAL_CLASSES_BY_KEY[k].parent or k for k in modes]

    log_theory = np.empty((len(modes), N))
    log_transitions = np.zeros((len(modes), N, N))
    for index, key in enumerate(modes):
        profile = templates[key].as_array()
        log_theory[index] = np.log(profile / profile.sum())
        matrix = templates[key].transition_array()
        if matrix is not None:
            log_transitions[index] = np.log(matrix)

    n = len(records)
    rotated = np.empty((n, N, N))
    transition = np.zeros((n, N, len(modes)))
    log_prior = np.empty((n, N))
    progression = np.zeros((n, N, len(modes)))

    shifts = np.arange(N)
    for i, record in enumerate(records):
        histogram = _unit(np.asarray(record["h"], dtype=float))
        # rotated[t, d] is the weight of the degree d quarter-tones above tonic t.
        rotated[i] = histogram[(shifts[None, :] + shifts[:, None]) % N]

        matrix = record.get("B")
        if matrix is not None:
            bigrams = np.asarray(matrix, dtype=float)
            if bigrams.sum() > 0:
                bigrams = _unit(bigrams.ravel()).reshape(N, N)
                for tonic in range(N):
                    turned = np.roll(np.roll(bigrams, -tonic, axis=0), -tonic, axis=1)
                    transition[i, tonic] = np.tensordot(
                        turned, log_transitions, axes=([0, 1], [1, 2])
                    )

        prior = record.get("log_prior")
        log_prior[i] = (
            np.log(histogram + 1e-9) if prior is None else np.asarray(prior, dtype=float)
        )

        windows = record.get("W")
        if gusheh_templates is not None and windows is not None:
            grid = np.asarray(windows, dtype=float)
            if grid.ndim == 2 and grid.shape[0] >= MIN_WINDOWS:
                progression[i] = progression_scores(grid, gusheh_templates, modes)

    return Design(
        rotated=rotated,
        transition=transition,
        log_prior=log_prior,
        progression=progression,
        log_theory=log_theory,
        modes=modes,
        mothers=mothers,
    )


def _unit(values: np.ndarray, epsilon: float = 1e-4) -> np.ndarray:
    """Normalise and smooth, as the hand-built scorer does."""
    total = values.sum()
    if total <= 0:
        return np.full(values.shape, 1.0 / values.size)
    smoothed = values / total + epsilon
    return smoothed / smoothed.sum()


@dataclass
class Parameters:
    """The fitted scoring parameters, in the layers ``LEVELS`` frees."""

    alpha: float
    transition_weight: float
    prior_weight: float
    #: Weight on progression through the seyr. Starts at zero, so a fitted
    #: value is a direct measurement of how much the order of a performance
    #: adds once its pitch content has been accounted for.
    progression_weight: float
    #: Per-mode exponent on the theory profile; scalar `sharpen` broadcast at init.
    sharpen: np.ndarray
    #: Per-mode score offset.
    bias: np.ndarray
    #: Per-mode, per-degree log-profile deviation from theory.
    deviation: np.ndarray
    modes: list[str] = field(default_factory=list)

    def log_profiles(self, log_theory: np.ndarray) -> np.ndarray:
        """The (M, 24) log-profiles these parameters imply."""
        a = self.sharpen[:, None] * log_theory + self.deviation
        return a - logsumexp(a, axis=1, keepdims=True)

    def profiles(self, log_theory: np.ndarray) -> np.ndarray:
        return np.exp(self.log_profiles(log_theory))


def initial_parameters(
    design: Design, config: ScoringConfig = DEFAULT_CONFIG
) -> Parameters:
    """Theory's own values, so an unfitted model is today's classifier.

    ``alpha`` absorbs the softmax temperature: the hand-built scorer divides the
    whole score by it, and a single scale on the profile term plus free weights
    on the others spans the same family.
    """
    m = design.n_modes
    return Parameters(
        alpha=1.0 / config.temperature,
        transition_weight=config.transition_weight / config.temperature,
        prior_weight=config.tonic_prior_weight / config.temperature,
        progression_weight=0.0,
        sharpen=np.full(m, float(config.sharpen)),
        bias=np.zeros(m),
        deviation=np.zeros((m, N)),
        modes=list(design.modes),
    )


def _mother_matrix(design: Design) -> tuple[np.ndarray, list[str]]:
    """(M, D) indicator folding each mode into its dastgah."""
    dastgahs = sorted(set(design.mothers))
    fold = np.zeros((design.n_modes, len(dastgahs)))
    for index, mother in enumerate(design.mothers):
        fold[index, dastgahs.index(mother)] = 1.0
    return fold, dastgahs


def _pack(parameters: Parameters, level: str) -> np.ndarray:
    pieces = [
        np.array([parameters.alpha, parameters.transition_weight,
                  parameters.prior_weight, parameters.progression_weight]),
        np.array([float(parameters.sharpen[0])]),
    ]
    if level in ("bias", "sharpen", "profiles"):
        pieces.append(parameters.bias)
    if level in ("sharpen", "profiles"):
        pieces.append(parameters.sharpen)
    if level == "profiles":
        pieces.append(parameters.deviation.ravel())
    return np.concatenate(pieces)


def _unpack(vector: np.ndarray, level: str, design: Design) -> Parameters:
    m = design.n_modes
    alpha, w_t, w_p, w_s = vector[0:4]
    shared = vector[4]
    cursor = 5

    bias = np.zeros(m)
    if level in ("bias", "sharpen", "profiles"):
        bias = vector[cursor : cursor + m]
        cursor += m

    sharpen = np.full(m, shared)
    if level in ("sharpen", "profiles"):
        sharpen = vector[cursor : cursor + m]
        cursor += m

    deviation = np.zeros((m, N))
    if level == "profiles":
        deviation = vector[cursor : cursor + m * N].reshape(m, N)

    return Parameters(
        alpha=float(alpha),
        transition_weight=float(w_t),
        prior_weight=float(w_p),
        progression_weight=float(w_s),
        sharpen=sharpen,
        bias=bias,
        deviation=deviation,
        modes=list(design.modes),
    )


def _scores(parameters: Parameters, design: Design) -> tuple[np.ndarray, np.ndarray]:
    """(n, 24, M) scores and the (M, 24) log-profiles they used."""
    log_profiles = parameters.log_profiles(design.log_theory)
    profile_term = np.einsum("itd,md->itm", design.rotated, log_profiles)
    scores = (
        parameters.alpha * profile_term
        + parameters.transition_weight * design.transition
        + parameters.prior_weight * design.log_prior[:, :, None]
        + parameters.progression_weight * design.progression
        + parameters.bias[None, None, :]
    )
    return scores, log_profiles


def _posterior(scores: np.ndarray) -> np.ndarray:
    """Softmax over the joint (tonic, mode) hypothesis space, per recording."""
    flat = scores.reshape(scores.shape[0], -1)
    flat = flat - flat.max(axis=1, keepdims=True)
    weights = np.exp(flat)
    weights /= weights.sum(axis=1, keepdims=True)
    return weights.reshape(scores.shape)


def dastgah_probabilities(
    parameters: Parameters, design: Design
) -> tuple[np.ndarray, list[str]]:
    """(n, D) probability per dastgah, tonic marginalised and avazes folded."""
    scores, _ = _scores(parameters, design)
    posterior = _posterior(scores)
    fold, dastgahs = _mother_matrix(design)
    return np.einsum("itm,md->id", posterior, fold), dastgahs


def _objective(
    vector: np.ndarray,
    level: str,
    design: Design,
    target: np.ndarray,
    penalty: float,
    weights: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    """Weighted negative log-likelihood of the true dastgah, and its gradient.

    ``target`` indexes the dastgah columns of :func:`_mother_matrix`. ``weights``
    optionally reweights recordings; passing the inverse class frequency makes
    every dastgah count equally, which changes whether giving a weak class up
    is a good trade.
    """
    parameters = _unpack(vector, level, design)
    scores, log_profiles = _scores(parameters, design)
    posterior = _posterior(scores)
    fold, _ = _mother_matrix(design)

    folded = np.einsum("itm,md->id", posterior, fold)
    rows = np.arange(design.n)
    truth = np.clip(folded[rows, target], 1e-12, None)
    share = np.full(design.n, 1.0 / design.n) if weights is None else weights
    loss = float(-np.dot(share, np.log(truth)))

    # d(-log P_truth)/d score, for the softmax-then-sum-group structure.
    belongs = fold[:, target].T  # (n, M): is this mode inside the true dastgah
    upstream = posterior * (1.0 - belongs[:, None, :] / truth[:, None, None])
    upstream *= share[:, None, None]

    d_alpha = float(
        np.sum(upstream * np.einsum("itd,md->itm", design.rotated, log_profiles))
    )
    d_transition = float(np.sum(upstream * design.transition))
    d_prior = float(np.sum(upstream * design.log_prior[:, :, None]))
    d_progression = float(np.sum(upstream * design.progression))
    d_bias = upstream.sum(axis=(0, 1))

    d_log_profiles = parameters.alpha * np.einsum("itm,itd->md", upstream, design.rotated)
    # back through the per-mode log_softmax
    profiles = np.exp(log_profiles)
    d_a = d_log_profiles - profiles * d_log_profiles.sum(axis=1, keepdims=True)
    d_sharpen = (d_a * design.log_theory).sum(axis=1)
    d_deviation = d_a.copy()

    if penalty and level == "profiles":
        loss += penalty * float(np.sum(parameters.deviation**2))
        d_deviation += 2.0 * penalty * parameters.deviation

    gradient = [
        np.array([d_alpha, d_transition, d_prior, d_progression]),
        np.array([d_sharpen.sum()]),
    ]
    if level in ("bias", "sharpen", "profiles"):
        gradient.append(d_bias)
    if level in ("sharpen", "profiles"):
        gradient.append(d_sharpen)
    if level == "profiles":
        gradient.append(d_deviation.ravel())
    return loss, np.concatenate(gradient)


def fit(
    design: Design,
    truth: "list[str]",
    *,
    level: str = "bias",
    penalty: float = 1.0,
    config: ScoringConfig = DEFAULT_CONFIG,
    max_iterations: int = 400,
    balanced: bool = False,
) -> Parameters:
    """Maximise the likelihood of the labelled dastgah, tonic latent.

    ``penalty`` is the L2 pull of learned profiles toward the notated radif, and
    only bites at ``level="profiles"``. ``balanced`` weights each dastgah
    equally rather than each recording, which denies the fit the option of
    writing off a class that is hard to separate.
    """
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}, got {level!r}")
    _, dastgahs = _mother_matrix(design)
    unknown = set(truth) - set(dastgahs)
    if unknown:
        raise ValueError(f"labels outside the answer space: {sorted(unknown)}")
    target = np.array([dastgahs.index(k) for k in truth])

    weights = None
    if balanced:
        occurrences = np.bincount(target, minlength=len(dastgahs)).astype(float)
        weights = 1.0 / occurrences[target]
        weights /= weights.sum()

    start = _pack(initial_parameters(design, config), level)
    result = minimize(
        _objective,
        start,
        args=(level, design, target, penalty, weights),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": max_iterations},
    )
    return _unpack(result.x, level, design)


def predict(
    parameters: Parameters, design: Design
) -> tuple[np.ndarray, list[str]]:
    """Most probable dastgah per recording, with the dastgah key order."""
    probabilities, dastgahs = dastgah_probabilities(parameters, design)
    return probabilities.argmax(axis=1), dastgahs
