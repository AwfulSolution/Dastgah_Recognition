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

    weights = np.zeros(N, dtype=float)
    for forud in foruds:
        weight = forud.strength
        if recency_halflife and total_duration:
            age = max(0.0, total_duration - forud.end)
            weight *= float(np.exp(-np.log(2) * age / recency_halflife))
        weights[forud.resolution_pc] += weight

    if weights.sum() <= 0:
        return None
    prior = weights / weights.sum() + smoothing
    return prior / prior.sum()
