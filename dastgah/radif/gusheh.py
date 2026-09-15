"""Identification of the *gusheh*: which melody of a dastgah is being played.

A dastgah is not one tune but a repertoire of gushehs sharing its scale. They are
distinguished less by which pitches occur — they largely agree — than by *where
in the range* the melody sits and which degrees it dwells on. A *darāmad* opens
low around the tonic; an *owj* climbs to the peak; a *forud* descends home.

So identification uses two things a mode classifier does not:

``profile``
    the tonic-relative pitch-class distribution, as for modes, but per gusheh;
``tessitura``
    how far above the tonic the melody's centre of gravity sits, in
    quarter-tones within the octave. This is what separates a darāmad from an
    owj, and measuring it relative to the tonic rather than in absolute pitch
    makes it independent of the key a performance is in.

Unlike modes within a family, gushehs within a dastgah are genuinely separable:
their pitch profiles agree at cosine 0.57-0.73, against 0.93 for modes that share
a collection, and their tessituras spread further than the notes within any one
of them.
"""

from __future__ import annotations

import difflib
import re
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from dastgah.radif.parse import Gusheh
from dastgah.theory import QUARTER_TONES_PER_OCTAVE

N = QUARTER_TONES_PER_OCTAVE

#: Romanisation collapses, applied before comparing gusheh names across sources.
_TRANSLITERATIONS = [
    ("aa", "a"), ("ee", "i"), ("oo", "u"), ("ou", "u"), ("ow", "u"),
    ("ey", "ei"), ("kh", "x"), ("gh", "q"), ("sh", "c"), ("ch", "c"),
    ("zh", "j"), ("yy", "y"), ("ss", "s"),
]


def normalise_name(name: str) -> str:
    """Fold a gusheh name to a form comparable across romanisations.

    Sources disagree freely: ``Zirkeshe Salmak`` against ``zirkash_salmak``,
    ``Ouj`` against ``owj``. Parentheticals and izafe particles are dropped and
    tokens sorted, so word order does not matter either.
    """
    text = name.lower()
    text = re.sub(r"\(.*?\)", " ", text)
    text = re.sub(r"[-_.]+", " ", text)
    text = re.sub(r"\b(e|ye|de)\b", " ", text)
    for source, target in _TRANSLITERATIONS:
        text = text.replace(source, target)
    text = re.sub(r"[^a-z ]", "", text)
    return " ".join(sorted(text.split()))


def match_name(name: str, candidates: dict[str, str], cutoff: float = 0.82) -> str | None:
    """Find the candidate gusheh whose name matches ``name``, if any."""
    key = normalise_name(name)
    if not key:
        return None
    if key in candidates:
        return candidates[key]
    close = difflib.get_close_matches(key, list(candidates), n=1, cutoff=cutoff)
    if close:
        return candidates[close[0]]
    for candidate in candidates:
        if candidate and min(len(candidate), len(key)) >= 5:
            if candidate in key or key in candidate:
                return candidates[candidate]
    return None


def tessitura(quarter_tones: np.ndarray, durations: np.ndarray, tonic_pc: int) -> float:
    """Quarter-tones from the tonic to the melody's centre of gravity.

    Signed and confined to (-12, +12]: the melody is placed against whichever
    octave of the tonic it sits nearest. An unsigned 0-24 measure would read a
    melody centred just *below* the tonic as almost an octave *above* it, which
    is the opposite reading.

    Being relative to the tonic and modular in the octave, this is unaffected by
    the key a performance is in or the register it is played in.
    """
    if quarter_tones.size == 0 or durations.sum() <= 0:
        return 0.0
    centre = float(np.average(quarter_tones, weights=durations))
    return float(((centre - tonic_pc + N / 2) % N) - N / 2)


@dataclass
class GushehTemplate:
    """One gusheh's pitch profile and tessitura, relative to its dastgah tonic."""

    mode: str
    name: str
    profile: list[float]      # 24 bins, quarter-tones above the tonic
    tessitura: float          # -12..+12, where the melody sits around the tonic
    tessitura_spread: float
    n_notes: int

    def as_array(self) -> np.ndarray:
        return np.asarray(self.profile, dtype=float)


def _smooth(values: np.ndarray, epsilon: float = 1e-3) -> np.ndarray:
    total = values.sum()
    if total <= 0:
        return np.full(values.shape, 1.0 / values.size)
    smoothed = values / total + epsilon
    return smoothed / smoothed.sum()


def build_gusheh_template(gusheh: Gusheh, tonic_pc: int) -> GushehTemplate:
    quarter = np.array([n.quarter_tones for n in gusheh.notes], dtype=float)
    durations = np.array([n.duration for n in gusheh.notes], dtype=float)

    histogram = np.zeros(N)
    for note in gusheh.notes:
        histogram[(note.pitch_class - tonic_pc) % N] += note.duration

    centre = float(np.average(quarter, weights=durations))
    spread = float(np.sqrt(np.average((quarter - centre) ** 2, weights=durations)))

    return GushehTemplate(
        mode=gusheh.modal_class.key,
        name=gusheh.name,
        profile=[float(x) for x in _smooth(histogram)],
        tessitura=round(tessitura(quarter, durations, tonic_pc), 3),
        tessitura_spread=round(spread, 3),
        n_notes=len(gusheh.notes),
    )


def build_gusheh_templates(
    corpus: list[Gusheh], mode_tonics: dict[str, int]
) -> dict[str, list[GushehTemplate]]:
    """Build a template per gusheh, grouped by dastgah."""
    grouped: dict[str, list[GushehTemplate]] = {}
    for gusheh in corpus:
        tonic = mode_tonics.get(gusheh.modal_class.key)
        if tonic is None or not gusheh.notes:
            continue
        grouped.setdefault(gusheh.modal_class.key, []).append(
            build_gusheh_template(gusheh, tonic)
        )
    return grouped


@dataclass
class GushehMatch:
    name: str
    score: float
    profile_score: float
    tessitura_score: float


def identify_gusheh(
    histogram: np.ndarray,
    observed_tessitura: float,
    candidates: list[GushehTemplate],
    *,
    tessitura_weight: float = 1.0,
    tessitura_sigma: float = 4.0,
) -> list[GushehMatch]:
    """Rank a dastgah's gushehs against an observed excerpt.

    ``histogram`` is 24 bins of tonic-relative weight; ``observed_tessitura`` the
    excerpt's centre of gravity above the tonic. Tessitura distance is measured
    around the octave, since it is an angle rather than a line.
    """
    if not candidates:
        return []
    observed = _smooth(np.asarray(histogram, dtype=float))

    matches: list[GushehMatch] = []
    for template in candidates:
        profile_score = float(np.dot(observed, np.log(template.as_array())))
        gap = abs(observed_tessitura - template.tessitura) % N
        gap = min(gap, N - gap)
        tessitura_score = -0.5 * (gap / tessitura_sigma) ** 2
        matches.append(
            GushehMatch(
                name=template.name,
                score=profile_score + tessitura_weight * tessitura_score,
                profile_score=round(profile_score, 4),
                tessitura_score=round(tessitura_score, 4),
            )
        )
    matches.sort(key=lambda m: m.score, reverse=True)
    return matches


def save_gusheh_templates(
    templates: dict[str, list[GushehTemplate]], path: Path
) -> None:
    import json

    payload = {
        "format": "dastgah-gusheh-templates/1",
        "source": "Radif Corpus (Zenodo 10.5281/zenodo.15742125, CC-BY-4.0)",
        "modes": {
            mode: [asdict(t) for t in sorted(group, key=lambda t: t.name)]
            for mode, group in sorted(templates.items())
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, ensure_ascii=False))


def load_gusheh_templates(path: Path) -> dict[str, list[GushehTemplate]]:
    import json

    payload = json.loads(path.read_text())
    return {
        mode: [GushehTemplate(**t) for t in group]
        for mode, group in payload["modes"].items()
    }
