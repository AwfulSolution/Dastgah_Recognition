"""Reader for the Radif Corpus symbolic dataset.

The corpus (Kanani et al., Zenodo DOI 10.5281/zenodo.15742125, CC-BY-4.0)
transcribes 228 gushehs of Mirza Abdollah's radif after Talai's notation. Each
gusheh is one CSV of notes with the columns::

    Microtonal pitch, Duration, Pitch (quarter notes), Interval,
    MIDI pitch number, MIDI Bend

Rows whose pitch cell is ``[`` or ``]`` delimit the hierarchical phrase
structure rather than sounding a note.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

from dastgah.theory import ModalClass, modal_class_from_dirname, pitch_class

OPEN_BRACKET = "["
CLOSE_BRACKET = "]"


@dataclass(frozen=True)
class Note:
    """A single sounding note of a gusheh."""

    label: str
    quarter_tones: int
    duration: float
    depth: int  # phrase-bracket nesting level at this note

    @property
    def pitch_class(self) -> int:
        return pitch_class(self.quarter_tones)


@dataclass
class Gusheh:
    """One gusheh: an ordered note sequence tagged with its modal class."""

    name: str
    modal_class: ModalClass
    source: Path
    notes: list[Note] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.notes)

    @property
    def total_duration(self) -> float:
        return sum(n.duration for n in self.notes)

    def pitch_classes(self) -> list[int]:
        return [n.pitch_class for n in self.notes]


def _parse_duration(cell: str) -> float:
    try:
        return float(cell)
    except (TypeError, ValueError):
        return 0.0


def read_gusheh(path: Path) -> Gusheh:
    """Read one gusheh CSV, skipping structure brackets but tracking their depth."""
    modal = modal_class_from_dirname(path.parent.name)
    name = path.stem.split(" - ", 1)[-1].strip()
    notes: list[Note] = []
    depth = 0

    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            label = (row.get("Microtonal pitch") or "").strip()
            if not label:
                continue
            if label == OPEN_BRACKET:
                depth += 1
                continue
            if label == CLOSE_BRACKET:
                depth = max(0, depth - 1)
                continue

            raw_quarter = (row.get("Pitch (quarter notes)") or "").strip()
            if not raw_quarter:
                continue
            try:
                quarter_tones = int(float(raw_quarter))
            except ValueError:
                continue

            notes.append(
                Note(
                    label=label,
                    quarter_tones=quarter_tones,
                    duration=_parse_duration(row.get("Duration", "")),
                    depth=depth,
                )
            )

    return Gusheh(name=name, modal_class=modal, source=path, notes=notes)


def iter_gushehs(corpus_csv_root: Path) -> Iterator[Gusheh]:
    """Yield every gusheh under the corpus ``CSV/`` directory, in radif order."""
    for dastgah_dir in sorted(p for p in corpus_csv_root.iterdir() if p.is_dir()):
        if not dastgah_dir.name[:2].isdigit():
            continue
        for csv_path in sorted(dastgah_dir.glob("*.csv")):
            gusheh = read_gusheh(csv_path)
            if gusheh.notes:
                yield gusheh


def load_corpus(corpus_csv_root: Path) -> list[Gusheh]:
    """Load the whole corpus into memory."""
    return list(iter_gushehs(corpus_csv_root))
