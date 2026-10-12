"""Score how a performance moves through a dastgah's gushehs, not which notes it uses.

The radif is an ordered traversal. A dastgah opens with its daramad, climbs
away from the tonic through intermediate gushehs, and returns by a forud; that
order is the *seyr*, and it is a property of the dastgah rather than of any
single moment in it.

Everything else in this library scores pitch *content*: a histogram or a bigram
matrix, both of which discard when things happened. For most dastgahs that is
enough, because their gushehs are modally their own. For the ones whose material
is largely borrowed from neighbours it is not, and no amount of audio or
parameters fixes it — 91.7% of Rast-Panjgah's gushehs have a near-twin in
another dastgah, and Panjgah itself is pitch-indistinguishable from Nava's
opening daramad (see docs/data-notes.md). What remains to tell them apart is
the order the shared material is visited in.

This module turns that into a score. A recording is cut into time windows; each
window is matched against a candidate dastgah's own gushehs; and the match is
asked whether it advances through the seyr as the recording advances through
time. The answer is a correlation in [-1, 1]: positive when the performance
traverses that dastgah's order, near zero when the assignment is arbitrary,
negative when it runs backwards.

The score is computed per tonic hypothesis, so it composes with the joint search
over modes and tonics in :mod:`dastgah.core.classify` rather than replacing it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dastgah.radif.gusheh import GushehTemplate
from dastgah.theory import QUARTER_TONES_PER_OCTAVE

N = QUARTER_TONES_PER_OCTAVE

#: Softmax temperature over gusheh similarities within one window.
#:
#: A window rarely contains exactly one gusheh, and the nearest few are often
#: close in profile, so the expected seyr position is taken over a distribution
#: rather than from the single best match. Hard assignment also produces ties,
#: which a rank statistic handles badly.
MATCH_TEMPERATURE = 0.02

#: Windows carrying less pitch content than this are dropped rather than matched.
MIN_WINDOW_WEIGHT = 1e-9

#: Below this many usable windows a correlation is not worth reporting.
MIN_WINDOWS = 4


@dataclass
class Progression:
    """How one (mode, tonic) hypothesis explains the order of a performance."""

    #: Correlation between window time and expected seyr position, in [-1, 1].
    score: float
    #: Expected seyr position per window, each in [0, 1].
    positions: list[float]
    #: Best-matching gusheh name per window, for display.
    names: list[str]
    #: Windows that carried enough pitch content to match.
    n_windows: int


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0)


def gusheh_matrix(templates: "list[GushehTemplate]") -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(G, 24) unit-norm profiles, their normalised seyr positions, and names.

    Positions are scaled to [0, 1] so that dastgahs with different numbers of
    gushehs produce comparable scores.
    """
    profiles = _unit_rows(np.array([t.as_array() for t in templates], dtype=float))
    indices = np.array([t.seyr_index for t in templates], dtype=float)
    if indices.min() < 0:
        raise ValueError("gusheh templates carry no seyr order; rebuild them")
    span = indices.max() - indices.min()
    positions = (indices - indices.min()) / span if span > 0 else np.zeros_like(indices)
    return profiles, positions, [t.name for t in templates]


def progression(
    windows: np.ndarray,
    templates: "list[GushehTemplate]",
    tonic_pc: int,
    *,
    temperature: float = MATCH_TEMPERATURE,
) -> Progression:
    """Score one dastgah-and-tonic hypothesis against a windowed performance.

    ``windows`` is (W, 24) absolute pitch-class weight per time window, in time
    order. Each window is rotated onto ``tonic_pc``, softmax-matched against the
    dastgah's gushehs, and reduced to an expected position in its seyr. The
    returned score is the Pearson correlation of those positions with time.

    Pearson rather than a rank statistic: the positions are already a smooth
    quantity on a meaningful scale, and a performance that lingers in the
    daramad before moving on should not have that flattening ranked away.
    """
    grid = np.atleast_2d(np.asarray(windows, dtype=float))
    if grid.shape[1] != N:
        raise ValueError(f"expected {N} bins per window, got {grid.shape[1]}")
    if not templates:
        return Progression(score=0.0, positions=[], names=[], n_windows=0)

    profiles, positions, names = gusheh_matrix(templates)

    keep = grid.sum(axis=1) > MIN_WINDOW_WEIGHT
    times = np.flatnonzero(keep).astype(float)
    if times.size < MIN_WINDOWS:
        return Progression(score=0.0, positions=[], names=[], n_windows=int(times.size))

    rotated = _unit_rows(np.roll(grid[keep], -tonic_pc, axis=1))
    similarity = rotated @ profiles.T  # (W, G) cosine

    scaled = similarity / max(temperature, 1e-9)
    scaled -= scaled.max(axis=1, keepdims=True)
    weights = np.exp(scaled)
    weights /= weights.sum(axis=1, keepdims=True)

    expected = weights @ positions
    best = [names[i] for i in similarity.argmax(axis=1)]

    return Progression(
        score=_correlation(times, expected),
        positions=[float(x) for x in expected],
        names=best,
        n_windows=int(times.size),
    )


def _correlation(times: np.ndarray, positions: np.ndarray) -> float:
    """Pearson r, defined as 0 when either side does not vary.

    A performance whose every window matches the same gusheh carries no
    evidence about order, and must score neutral rather than undefined.
    """
    t = times - times.mean()
    p = positions - positions.mean()
    denominator = float(np.linalg.norm(t) * np.linalg.norm(p))
    if denominator <= 1e-12:
        return 0.0
    return float(np.dot(t, p) / denominator)


def progression_scores(
    windows: np.ndarray,
    gusheh_templates: dict[str, "list[GushehTemplate]"],
    modes: "list[str]",
    *,
    temperature: float = MATCH_TEMPERATURE,
) -> np.ndarray:
    """(24, M) progression score for every tonic and every mode in ``modes``.

    Shaped to drop straight into the joint search: axis 0 is the tonic
    hypothesis, axis 1 indexes ``modes``. A mode with no gushehs scores zero
    everywhere, which leaves it to be decided on pitch content alone.
    """
    scores = np.zeros((N, len(modes)))
    for column, mode in enumerate(modes):
        templates = gusheh_templates.get(mode) or []
        if len(templates) < 2:
            continue
        for tonic in range(N):
            scores[tonic, column] = progression(
                windows, templates, tonic, temperature=temperature
            ).score
    return scores


def window_histograms(
    events: "list", *, seconds: float = 20.0, duration: float | None = None
) -> np.ndarray:
    """(W, 24) pitch-class weight per fixed-length window, in time order.

    ``events`` are :class:`dastgah.core.audio.NoteEvent`. Fixed windows rather
    than the Viterbi modal segments on purpose: the segmenter already commits to
    a mode, and reusing its output here would feed the classifier its own
    answer. Windows are agnostic.
    """
    if not events:
        return np.zeros((0, N))
    span = duration if duration is not None else max(e.start + e.duration for e in events)
    count = max(1, int(np.ceil(span / seconds)))
    grid = np.zeros((count, N))
    for event in events:
        index = min(count - 1, int(event.start // seconds))
        grid[index, event.pitch_class % N] += event.duration
    return grid
