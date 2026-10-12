"""Core music theory for Iranian classical music in 24-tone equal temperament.

Pitch is represented throughout as an integer count of quarter-tones. The
Radif Corpus encodes this directly as ``2 * midi_pitch + bend`` where a bend of
``+2048`` denotes a raised quarter-tone, so middle C (MIDI 60) is 120 and a
pitch class is simply that value modulo 24.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

QUARTER_TONES_PER_OCTAVE = 24
CENTS_PER_QUARTER_TONE = 50

#: Natural note letters to their quarter-tone offset above C.
_LETTER_OFFSET = {"C": 0, "D": 4, "E": 8, "F": 10, "G": 14, "A": 18, "B": 22}

#: Accidental suffixes to their quarter-tone adjustment. ``k`` is *koron*
#: (lowered quarter-tone), ``s`` is *sori* (raised quarter-tone) and ``N`` is an
#: explicit natural cancelling a previous accidental.
_ACCIDENTAL_OFFSET = {"": 0, "#": 2, "b": -2, "k": -1, "s": 1, "N": 0}

#: Canonical spelling of each 24-TET pitch class. Flats are preferred for the
#: semitones and koron over sori for the quarter-tones, matching Talai's
#: notation of the Mirza Abdollah radif.
_PITCH_CLASS_NAMES = [
    "C", "Cs", "Db", "Dk", "D", "Ds", "Eb", "Ek", "E", "Es", "F", "Fs",
    "F#", "Gk", "G", "Gs", "Ab", "Ak", "A", "As", "Bb", "Bk", "B", "Bs",
]

#: The corpus' unmarked octave runs F3-E4 rather than C4-B4, placing it over the
#: radif's typical ambitus. F is therefore 106 while C is 120.
_NATURAL_OCTAVE = {"C": 5, "D": 5, "E": 5, "F": 4, "G": 4, "A": 4, "B": 4}

_LABEL_RE = re.compile(r"^([A-G])(#|b|k|s|N)?([+-]\d+)?$")


class PitchParseError(ValueError):
    """Raised when a Radif Corpus pitch label cannot be interpreted."""


def parse_pitch_label(label: str) -> int:
    """Convert a Radif Corpus pitch label to a quarter-tone number.

    Labels look like ``C``, ``Bb``, ``Ak`` (A koron), ``F#``, ``DN`` (D natural)
    or ``G+1``, where the trailing signed integer is an octave displacement.

    .. warning::
       Octave marks in the corpus are relative to each piece's own register, not
       to a fixed anchor, so the absolute value returned here is only correct up
       to an octave. Pitch *class* is reliable (it agrees with the corpus'
       ``Pitch (quarter notes)`` column for 99.98% of notes), but for absolute
       pitch always prefer that column. This function is intended for display
       and validation, not as the primary pitch source.
    """
    match = _LABEL_RE.match(label.strip())
    if match is None:
        raise PitchParseError(f"unrecognised pitch label: {label!r}")
    letter, accidental, octave = match.groups()
    octave_base = _NATURAL_OCTAVE[letter] + (int(octave) if octave else 0)
    value = _LETTER_OFFSET[letter] + _ACCIDENTAL_OFFSET[accidental or ""]
    return value + QUARTER_TONES_PER_OCTAVE * octave_base


def pitch_class(quarter_tones: int) -> int:
    """Fold a quarter-tone number into its 0-23 pitch class."""
    return quarter_tones % QUARTER_TONES_PER_OCTAVE


def pitch_class_name(pc: int) -> str:
    """Canonical name for a 24-TET pitch class."""
    return _PITCH_CLASS_NAMES[pc % QUARTER_TONES_PER_OCTAVE]


def is_microtonal(pc: int) -> bool:
    """True when a pitch class falls between the 12-TET semitones."""
    return pc % 2 == 1


def hz_to_quarter_tones(hz: float, reference_hz: float = 440.0) -> float:
    """Convert a frequency to a (fractional) quarter-tone number.

    ``reference_hz`` is the tuning of A4, which sits at quarter-tone 138.
    """
    import numpy as np

    return 138.0 + 24.0 * np.log2(hz / reference_hz)


def cents_between(a: float, b: float) -> float:
    """Signed cent distance from quarter-tone number ``a`` to ``b``."""
    return (b - a) * CENTS_PER_QUARTER_TONE


@dataclass(frozen=True)
class ModalClass:
    """One of the 13 modal classes of the Mirza Abdollah radif."""

    key: str
    name: str
    persian: str
    kind: str  # "dastgah" or "avaz"
    parent: str | None = None  # the dastgah an avaz derives from

    @property
    def display(self) -> str:
        prefix = "Dastgāh-e" if self.kind == "dastgah" else "Āvāz-e"
        return f"{prefix} {self.name}"


#: The 13 modal classes, keyed by the Radif Corpus directory ordering.
MODAL_CLASSES: tuple[ModalClass, ...] = (
    ModalClass("shur", "Shūr", "شور", "dastgah"),
    ModalClass("bayat_e_kord", "Bayāt-e Kord", "بیات کرد", "avaz", "shur"),
    ModalClass("dashti", "Dashtī", "دشتی", "avaz", "shur"),
    ModalClass("bayat_e_tork", "Bayāt-e Tork", "بیات ترک", "avaz", "shur"),
    ModalClass("abuata", "Abū'atā", "ابوعطا", "avaz", "shur"),
    ModalClass("afshari", "Afshārī", "افشاری", "avaz", "shur"),
    ModalClass("segah", "Segāh", "سه‌گاه", "dastgah"),
    ModalClass("nava", "Navā", "نوا", "dastgah"),
    ModalClass("homayun", "Homāyūn", "همایون", "dastgah"),
    ModalClass("bayat_e_esfahan", "Bayāt-e Esfahān", "بیات اصفهان", "avaz", "homayun"),
    ModalClass("chahargah", "Chahārgāh", "چهارگاه", "dastgah"),
    ModalClass("mahur", "Māhūr", "ماهور", "dastgah"),
    ModalClass("rast_panjgah", "Rāst-Panjgāh", "راست‌پنجگاه", "dastgah"),
)

MODAL_CLASSES_BY_KEY = {m.key: m for m in MODAL_CLASSES}

#: Maps a Radif Corpus directory name (``"01 - Shur"``) to a modal class key.
_DIR_KEYS = {
    "01": "shur", "02": "bayat_e_kord", "03": "dashti", "04": "bayat_e_tork",
    "05": "abuata", "06": "afshari", "07": "segah", "08": "nava",
    "09": "homayun", "10": "bayat_e_esfahan", "11": "chahargah",
    "12": "mahur", "13": "rast_panjgah",
}


def _normalise(name: str) -> str:
    """Fold a written mode name to a comparable form.

    Romanisations vary widely (``Abu'ata``/``Abuata``, ``Rast-Panjgah``/
    ``Rast Panjgah``, ``Bayat-e Tork``/``bayat_e_tork``), so punctuation,
    spacing and diacritics are stripped before matching.
    """
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", name)
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    return "".join(c for c in stripped.lower() if c.isalnum())


#: Alternate spellings seen in the wild, mapped to modal class keys.
_NAME_ALIASES = {
    "bayatezand": "bayat_e_tork",  # older name for Bayat-e Tork
    "zand": "bayat_e_tork",
    "esfahan": "bayat_e_esfahan",
    "isfahan": "bayat_e_esfahan",
    "abuatta": "abuata",
    "rastpanjgah": "rast_panjgah",
    "rast": "rast_panjgah",
    "homayoun": "homayun",
    "mahoor": "mahur",
    "chaharghah": "chahargah",
    "shoor": "shur",
}

_NORMALISED_LOOKUP = {
    **{_normalise(m.key): m.key for m in MODAL_CLASSES},
    **{_normalise(m.name): m.key for m in MODAL_CLASSES},
    **_NAME_ALIASES,
}


def modal_class_from_name(name: str) -> ModalClass | None:
    """Resolve a written mode name, or ``None`` if it is not recognised.

    Accepts keys (``bayat_e_tork``), display names (``Bayāt-e Tork``) and the
    common alternate romanisations, so dataset folder names can be matched
    without each caller inventing its own table.
    """
    key = _NORMALISED_LOOKUP.get(_normalise(name))
    return MODAL_CLASSES_BY_KEY[key] if key else None


def modal_class_from_dirname(dirname: str) -> ModalClass:
    """Resolve a corpus directory name such as ``"11 - Chahargah"``."""
    number = dirname.strip()[:2]
    try:
        return MODAL_CLASSES_BY_KEY[_DIR_KEYS[number]]
    except KeyError as exc:
        raise ValueError(f"unknown radif directory: {dirname!r}") from exc
