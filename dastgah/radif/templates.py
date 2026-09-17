"""Derive tonic-relative modal templates from the Radif Corpus.

A template is a duration-weighted distribution over the 24 quarter-tone
intervals above a mode's tonic (*ist*). Expressing it relative to the tonic
makes it transposition-invariant, so a performance in any key can be matched by
scoring every one of the 24 possible tonic hypotheses.

Two degrees are recorded per mode:

``tonic``
    The *ist*, the note of final repose. Estimated as the consensus final note
    across the mode's gushehs, since individual gushehs often close on an
    intermediate degree and only the concluding *forud* returns to the tonic.
``shahed``
    The melodic pivot or reciting tone: the most-sounded degree. This is
    frequently *not* the tonic and is a strong secondary cue.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from dastgah.radif.parse import Gusheh
from dastgah.theory import (
    MODAL_CLASSES_BY_KEY,
    QUARTER_TONES_PER_OCTAVE,
    ModalClass,
    pitch_class_name,
)

#: Floor added to every bin so unseen degrees never produce zero likelihood.
SMOOTHING = 1e-4

#: Aligned cosine above which two modes are treated as one neighbourhood.
FAMILY_SIMILARITY_THRESHOLD = 0.90


@dataclass
class ModalTemplate:
    """A transposition-invariant profile of one modal class."""

    key: str
    tonic_pc: int          # tonic pitch class as notated in the corpus
    shahed_interval: int   # quarter-tones from tonic to the most-sounded degree
    profile: list[float]   # 24 bins, index = quarter-tones above tonic
    n_gushehs: int
    n_notes: int
    tonic_confidence: float  # share of gushehs whose final note is the tonic
    transitions: list[list[float]] | None = None  # 24x24 tonic-relative bigrams
    family: str = ""  # key of the neighbourhood this mode belongs to

    def transition_array(self) -> np.ndarray | None:
        """Tonic-relative note-to-note transition probabilities, if present."""
        if self.transitions is None:
            return None
        return np.asarray(self.transitions, dtype=float)

    @property
    def modal_class(self) -> ModalClass:
        return MODAL_CLASSES_BY_KEY[self.key]

    def as_array(self) -> np.ndarray:
        return np.asarray(self.profile, dtype=float)

    def scale_degrees(self, threshold: float = 0.02) -> list[int]:
        """Intervals carrying at least ``threshold`` of the total duration."""
        arr = self.as_array()
        return [i for i in np.argsort(arr)[::-1] if arr[i] >= threshold]

    def microtonal_degrees(self, threshold: float = 0.02) -> list[int]:
        """Scale degrees that fall on a quarter-tone (koron/sori inflections)."""
        return [d for d in self.scale_degrees(threshold) if d % 2 == 1]


def estimate_tonic(gushehs: list[Gusheh]) -> tuple[int, float]:
    """Consensus tonic pitch class for one mode, with the share of agreement."""
    finals = Counter(g.notes[-1].pitch_class for g in gushehs if g.notes)
    if not finals:
        raise ValueError("no notes from which to estimate a tonic")
    tonic, count = finals.most_common(1)[0]
    return tonic, count / sum(finals.values())


def _weighted_histogram(gushehs: list[Gusheh], tonic_pc: int) -> np.ndarray:
    """Duration-weighted histogram of intervals above ``tonic_pc``."""
    hist = np.zeros(QUARTER_TONES_PER_OCTAVE, dtype=float)
    for gusheh in gushehs:
        for note in gusheh.notes:
            interval = (note.pitch_class - tonic_pc) % QUARTER_TONES_PER_OCTAVE
            hist[interval] += note.duration
    return hist


def _transition_matrix(gushehs: list[Gusheh], tonic_pc: int) -> np.ndarray:
    """Note-to-note transition counts between tonic-relative degrees.

    Melodic motion (*seyr*) is what separates an avaz from its parent dastgah,
    since the two typically share a scale and differ only in emphasis and
    direction.
    """
    n = QUARTER_TONES_PER_OCTAVE
    matrix = np.zeros((n, n), dtype=float)
    for gusheh in gushehs:
        notes = gusheh.notes
        for current, following in zip(notes, notes[1:]):
            i = (current.pitch_class - tonic_pc) % n
            j = (following.pitch_class - tonic_pc) % n
            matrix[i, j] += 1.0
    return matrix


def normalize(hist: np.ndarray, smoothing: float = SMOOTHING) -> np.ndarray:
    """Turn counts into a smoothed probability distribution summing to one.

    Works on any shape, so it applies equally to the 24-bin profile and the
    24x24 transition matrix.
    """
    total = hist.sum()
    if total <= 0:
        return np.full(hist.shape, 1.0 / hist.size)
    smoothed = hist / total + smoothing
    return smoothed / smoothed.sum()


def build_template(key: str, gushehs: list[Gusheh]) -> ModalTemplate:
    """Build one mode's template from its gushehs."""
    tonic_pc, confidence = estimate_tonic(gushehs)
    profile = normalize(_weighted_histogram(gushehs, tonic_pc))

    raw_transitions = _transition_matrix(gushehs, tonic_pc)
    transitions = normalize(raw_transitions.ravel()).reshape(
        QUARTER_TONES_PER_OCTAVE, QUARTER_TONES_PER_OCTAVE
    )

    return ModalTemplate(
        key=key,
        tonic_pc=tonic_pc,
        shahed_interval=int(np.argmax(profile)),
        profile=[float(x) for x in profile],
        n_gushehs=len(gushehs),
        n_notes=sum(len(g) for g in gushehs),
        tonic_confidence=round(confidence, 4),
        transitions=[[float(x) for x in row] for row in transitions],
    )


def build_templates(corpus: list[Gusheh]) -> dict[str, ModalTemplate]:
    """Build templates for every modal class present in ``corpus``."""
    grouped: dict[str, list[Gusheh]] = {}
    for gusheh in corpus:
        grouped.setdefault(gusheh.modal_class.key, []).append(gusheh)
    templates = {key: build_template(key, gs) for key, gs in grouped.items()}

    for key, family in build_families(templates).items():
        templates[key].family = family
    return templates


def aligned_similarity(a: ModalTemplate, b: ModalTemplate) -> float:
    """Best cosine between two profiles over all 24 relative rotations.

    Two modes can share a pitch collection while placing the tonic on different
    degrees of it, which is invisible to a direct comparison but obvious once
    one profile is rotated onto the other.
    """
    first = a.as_array()
    second = b.as_array()
    norm_product = float(np.linalg.norm(first) * np.linalg.norm(second))
    if norm_product <= 0:
        return 0.0
    return max(
        float(np.dot(first, np.roll(second, shift)) / norm_product)
        for shift in range(QUARTER_TONES_PER_OCTAVE)
    )


#: The seven dastgahs are the classes with audio evidence behind them. The six
#: in ``DASTGAHS_WITH_AUDIO`` are those the evaluation archive covers; the avazes
#: and Rast-Panjgah remain in the templates but score 0-22% and 12.8%
#: respectively, so a caller may prefer to keep them out of the running.
DASTGAHS_WITH_AUDIO = (
    "shur", "nava", "homayun", "mahur", "chahargah", "segah",
)


def restrict(
    templates: dict[str, ModalTemplate], keys: "Iterable[str]"
) -> dict[str, ModalTemplate]:
    """Narrow the candidate modes, recomputing families over what remains.

    Scores are computed per template and are independent of which others are
    present, so restricting here gives exactly the answer that picking the best
    surviving candidate afterwards would. What it does change is the family
    partition, which must be recomputed: over the six dastgahs with audio,
    Homayun and Mahur separate and only Shur and Nava remain grouped.
    """
    chosen = [k for k in keys if k in templates]
    if not chosen:
        raise ValueError("no known modes left after restriction")

    narrowed = {
        key: replace(templates[key]) for key in sorted(chosen)
    }
    for key, family in build_families(narrowed).items():
        narrowed[key].family = family
    return narrowed


def build_families(
    templates: dict[str, ModalTemplate],
    threshold: float = FAMILY_SIMILARITY_THRESHOLD,
) -> dict[str, str]:
    """Group modes whose profiles are rotations of one another.

    Modes linked above ``threshold`` are merged transitively, so a family is a
    connected component of the similarity graph. These are the sets a pitch-class
    profile cannot separate: the seven-member group it finds coincides with the
    traditional Shur family, and Chahargah and Segah are the only modes that
    stand alone.

    Each family is keyed by its best-attested member, since that is the mode a
    reader is most likely to recognise the group by.
    """
    keys = sorted(templates)
    parent = {k: k for k in keys}

    def find(key: str) -> str:
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    for index, first in enumerate(keys):
        for second in keys[index + 1 :]:
            if aligned_similarity(templates[first], templates[second]) > threshold:
                root_a, root_b = find(first), find(second)
                if root_a != root_b:
                    parent[root_a] = root_b

    members: dict[str, list[str]] = {}
    for key in keys:
        members.setdefault(find(key), []).append(key)

    assignment: dict[str, str] = {}
    for group in members.values():
        label = max(group, key=lambda k: templates[k].n_gushehs)
        for key in group:
            assignment[key] = label
    return assignment


def family_members(templates: dict[str, ModalTemplate]) -> dict[str, list[str]]:
    """Family key to the modes it contains, in radif order."""
    from dastgah.theory import MODAL_CLASSES_BY_KEY

    order = list(MODAL_CLASSES_BY_KEY)
    grouped: dict[str, list[str]] = {}
    for key, family in sorted(templates.items()):
        if family := templates[key].family or family:
            grouped.setdefault(family, []).append(key)
    return {f: sorted(v, key=order.index) for f, v in grouped.items()}


def save_templates(templates: dict[str, ModalTemplate], path: Path) -> None:
    """Write templates to JSON so inference does not need the corpus."""
    payload = {
        "format": "dastgah-templates/1",
        "source": "Radif Corpus (Zenodo 10.5281/zenodo.15742125, CC-BY-4.0)",
        "templates": {k: asdict(v) for k, v in sorted(templates.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def load_templates(path: Path) -> dict[str, ModalTemplate]:
    """Read templates previously written by :func:`save_templates`."""
    payload = json.loads(path.read_text())
    return {k: ModalTemplate(**v) for k, v in payload["templates"].items()}


def describe(template: ModalTemplate) -> str:
    """One-line human summary, for CLI output and debugging."""
    modal = template.modal_class
    degrees = " ".join(
        f"{d}{'*' if d % 2 else ''}" for d in sorted(template.scale_degrees())
    )
    return (
        f"{modal.display:<26} tonic={pitch_class_name(template.tonic_pc):<3} "
        f"shahed=+{template.shahed_interval:<2} "
        f"n={template.n_gushehs:<3} degrees[{degrees}]"
    )
