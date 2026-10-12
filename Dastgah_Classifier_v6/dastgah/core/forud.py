"""Detection of the *forud*, the cadential descent that establishes the tonic.

A forud is the figure a Persian performance uses to come home: the line descends
through the mode and settles on a sustained note, usually followed by a breath.
That terminal note is the *ist*, the tonic — so locating foruds gives direct
evidence of the tonic rather than inferring it from how much each pitch is used.

This matters because the tonic is what a pitch-class distribution cannot supply.
Modes within a family share a pitch collection and differ in which degree is
home, and weighting tonic candidates simply by how long they sound elects the
*shahed* (the reciting tone) instead, which for Shur sits a fourth above the
tonic — exactly the interval that turns Shur into Nava.

Not every phrase ending is a forud. A phrase can close on the shahed or on a
passing degree, so a descent and a note of genuine repose are both required, and
each candidate carries a strength rather than a vote.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from dastgah.core.audio import NoteEvent
from dastgah.theory import QUARTER_TONES_PER_OCTAVE

N = QUARTER_TONES_PER_OCTAVE

#: Silence, in seconds, that ends a phrase.
PHRASE_GAP = 0.25

#: Notes examined at the end of a phrase when looking for a descent.
CADENCE_NOTES = 5

#: A descent of at least this many quarter-tones counts as a full descent.
FULL_DESCENT = 8.0

#: Repose is capped here, so one very long note cannot dominate.
MAX_REPOSE_RATIO = 4.0


@dataclass
class Forud:
    """One cadential descent and the degree it resolves onto."""

    resolution_pc: int
    start: float
    end: float
    descent: float            # quarter-tones fallen into the resolution
    repose: float             # resolution length over the phrase's mean
    silence_after: float
    strength: float           # 0-1

    @property
    def duration(self) -> float:
        return self.end - self.start


def _phrases(events: list[NoteEvent], gap: float) -> list[list[NoteEvent]]:
    """Split note events at silences."""
    if not events:
        return []
    groups: list[list[NoteEvent]] = [[events[0]]]
    for previous, current in zip(events, events[1:]):
        if current.start - (previous.start + previous.duration) >= gap:
            groups.append([current])
        else:
            groups[-1].append(current)
    return groups


def _silence_after(events: list[NoteEvent], phrase: list[NoteEvent], total: float) -> float:
    last = phrase[-1]
    end = last.start + last.duration
    following = [e for e in events if e.start > end]
    return (following[0].start - end) if following else max(0.0, total - end)


def find_foruds(
    events: list[NoteEvent],
    *,
    total_duration: float | None = None,
    gap: float = PHRASE_GAP,
    min_notes: int = 3,
) -> list[Forud]:
    """Find cadential descents and the degrees they resolve onto.

    Strength combines three things a cadence needs: the line must fall into the
    final note, that note must be held longer than the phrase around it, and it
    should be followed by a breath. A phrase that merely stops does not qualify.
    """
    if total_duration is None:
        total_duration = max((e.start + e.duration for e in events), default=0.0)

    found: list[Forud] = []
    for phrase in _phrases(events, gap):
        if len(phrase) < min_notes:
            continue

        resolution = phrase[-1]
        tail = phrase[-CADENCE_NOTES:]
        pitches = np.array([e.quarter_tone for e in tail])

        # How far the line falls into the resolution, and how steadily.
        drop = float(pitches[:-1].max() - pitches[-1])
        steps = np.diff(pitches)
        descending = float(np.mean(steps < 0)) if steps.size else 0.0
        descent_score = min(max(drop, 0.0) / FULL_DESCENT, 1.0) * descending

        mean_duration = float(np.mean([e.duration for e in phrase[:-1]])) or 1e-9
        repose = min(resolution.duration / mean_duration, MAX_REPOSE_RATIO)
        repose_score = min(repose / 2.0, 1.0)

        silence = _silence_after(events, phrase, total_duration)
        silence_score = min(silence / 1.0, 1.0)

        strength = descent_score * (0.5 + 0.5 * repose_score) * (0.5 + 0.5 * silence_score)
        if strength <= 0.0:
            continue

        found.append(
            Forud(
                resolution_pc=resolution.pitch_class,
                start=phrase[0].start,
                end=resolution.start + resolution.duration,
                descent=drop,
                repose=repose,
                silence_after=silence,
                strength=round(float(strength), 4),
            )
        )
    return found


def _recency_weights(
    foruds: list[Forud],
    recency_halflife: float | None,
    total_duration: float | None,
) -> np.ndarray:
    """Per-cadence discount favouring later ones, or ones everywhere if unset.

    A performance may visit several modes and only the closing forud returns to
    the principal tonic.
    """
    if not (recency_halflife and total_duration):
        return np.ones(len(foruds))
    ages = np.array([max(0.0, total_duration - f.end) for f in foruds])
    return np.exp(-np.log(2) * ages / recency_halflife)


def tonic_prior(
    foruds: list[Forud],
    *,
    total_duration: float | None = None,
    recency_halflife: float | None = None,
    smoothing: float = 0.02,
) -> np.ndarray | None:
    """A 24-bin prior over tonics from where cadences resolve.

    ``recency_halflife`` optionally favours later cadences: a performance may
    visit several modes and only the closing forud returns to the principal
    tonic. Returns ``None`` when no cadence was found, so callers can fall back.
    """
    if not foruds:
        return None

    recency = _recency_weights(foruds, recency_halflife, total_duration)
    weights = np.zeros(N, dtype=float)
    for forud, discount in zip(foruds, recency, strict=True):
        weights[forud.resolution_pc] += forud.strength * discount

    if weights.sum() <= 0:
        return None
    prior = weights / weights.sum() + smoothing
    return prior / prior.sum()


def approach_profile(
    events: "list[NoteEvent]", forud: Forud, *, notes: int = 8
) -> np.ndarray | None:
    """How a cadence was reached, as intervals from the note it resolved onto.

    The mirror of :func:`dastgah.radif.templates._cadence_profile`, computed from
    audio instead of notation so the two can be compared. Anchored on the
    resolution rather than the tonic, which is what makes it informative about a
    pair of modes that share a pitch collection: Shur reaches its close from the
    shahed a fourth above, Nava from a fourth below.

    Returns ``None`` when the descent carries too few notes to describe.
    """
    within = [e for e in events if forud.start <= e.start < forud.end]
    if len(within) < 2:
        return None
    tail = within[-notes - 1 : -1] or within[:-1]
    histogram = np.zeros(N)
    for event in tail:
        histogram[(event.pitch_class - forud.resolution_pc) % N] += event.duration
    if histogram.sum() <= 0:
        return None
    return histogram / histogram.sum()


def cadence_agreement(
    foruds: list[Forud],
    events: "list[NoteEvent]",
    cadence_profiles: dict[str, "np.ndarray | None"],
    *,
    recency_halflife: float | None = None,
    total_duration: float | None = None,
) -> "dict[str, np.ndarray] | None":
    """Per mode, how well the cadences resolving on each degree fit that mode.

    Additive evidence rather than a prior, and that distinction is the whole
    point. A prior is a distribution over tonics, so it can only say *where*
    cadences resolved; normalising it cancels any per-mode weighting outright
    when every cadence lands on the same degree. But the discriminating case is
    exactly two modes at the *same* tonic -- a forud onto Nava's tonic reached
    from below against a phrase-rest on Shur's shahed reached from above are the
    same pitch class, and measured on Nava, Shur is the one mode whose cadences
    land more often on its shahed than on its tonic.

    Returns ``{mode: (24,) evidence}``, unnormalised across degrees so that it
    can express absolute support, but **centred across modes** at each degree so
    that only relative fit counts. Without centring the term rewards whichever
    mode has the broadest notated approach, since a flat profile scores a decent
    cosine against anything -- Shur's (33% on the resolution, 21% and 20% on two
    other degrees) against Nava's (40% and 21%). That is the same bias
    ``sharpen`` exists to correct for pitch profiles, and it is a property of the
    template rather than evidence about the recording.

    A degree with no cadence scores zero for every mode, which is neutral
    between them. Modes with no notated approach, and cadences whose approach
    cannot be described, contribute nothing.
    """
    if not foruds:
        return None
    approaches = [approach_profile(events, f) for f in foruds]
    if all(a is None for a in approaches):
        return None
    recency = _recency_weights(foruds, recency_halflife, total_duration)

    out: dict[str, np.ndarray] = {}
    for key, profile in cadence_profiles.items():
        evidence = np.zeros(N)
        if profile is None:
            out[key] = evidence
            continue
        norm_profile = float(np.linalg.norm(profile))
        for forud, approach, discount in zip(foruds, approaches, recency, strict=True):
            if approach is None or norm_profile <= 0:
                continue
            norm = float(np.linalg.norm(approach)) * norm_profile
            if norm <= 0:
                continue
            similarity = float(np.dot(approach, profile) / norm)
            evidence[forud.resolution_pc] += forud.strength * discount * similarity
        out[key] = evidence

    described = [k for k, v in cadence_profiles.items() if v is not None]
    if described:
        mean = np.mean([out[k] for k in described], axis=0)
        for key in described:
            out[key] = out[key] - mean
    return out
