"""Reader for the IRMA dataset's extracted pitch contours.

IRMA (Shahrokhi et al., DLfM 2025, CC-BY-NC) pairs the Karimi and Mirza Abdollah
radif traditions with contours extracted from recordings. Each gusheh has a
``*_pitch_*.csv`` and an ``*_energy_*.csv`` under a ``*_Mp3csv_folder``, with the
dastgah and gusheh identified in the path and filename.

This matters because it is audio-derived rather than notated: evaluating against
it is the first out-of-domain test of templates built from the written radif.

No audio is distributed with the dataset; ``AUDIO_SOURCES.md`` names the
recordings the contours came from.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from dastgah.theory import ModalClass, MODAL_CLASSES_BY_KEY

#: IRMA numbers the modes differently from the Radif Corpus, so its ``D1``-``D13``
#: book identifiers need their own mapping. ``D3`` is Bayat-e Zand, the older
#: name for the avaz this project calls Bayat-e Tork.
IRMA_DASTGAH_KEYS: dict[str, str] = {
    "D1": "shur",
    "D2": "abuata",
    "D3": "bayat_e_tork",
    "D4": "afshari",
    "D5": "dashti",
    "D6": "bayat_e_kord",
    "D7": "mahur",
    "D8": "homayun",
    "D9": "bayat_e_esfahan",
    "D10": "segah",
    "D11": "chahargah",
    "D12": "nava",
    "D13": "rast_panjgah",
}

#: ``D11_G204_M1_A1_T1_pitch_3 Chahargah_zaabol.csv``
_FILENAME_RE = re.compile(
    r"^(?P<dastgah>D\d+)_G(?P<gusheh_id>\d+)_"
    r"(?P<performer>M\d+_A\d+_T\d+)_"
    r"(?P<kind>pitch|energy)_"
    r"(?P<index>\d+)\s+(?P<label>.+)$"
)


@dataclass
class IrmaContour:
    """One gusheh's extracted pitch contour."""

    key: str                # modal class key
    tradition: str          # "Karimi" or "Mirza Abdollah"
    gusheh_id: str
    gusheh_name: str
    performer: str
    source: Path
    times: np.ndarray
    f0_hz: np.ndarray       # 0 or NaN where unvoiced

    @property
    def modal_class(self) -> ModalClass:
        return MODAL_CLASSES_BY_KEY[self.key]

    @property
    def voiced(self) -> np.ndarray:
        return np.isfinite(self.f0_hz) & (self.f0_hz > 0)

    def __len__(self) -> int:
        return int(self.voiced.sum())


def parse_filename(stem: str) -> dict[str, str] | None:
    """Pull the dastgah, gusheh and performer codes out of a contour filename."""
    match = _FILENAME_RE.match(stem)
    if match is None:
        return None
    fields = match.groupdict()
    key = IRMA_DASTGAH_KEYS.get(fields["dastgah"])
    if key is None:
        return None
    fields["key"] = key
    # The label is "<Dastgah>_<gusheh name>"; keep the gusheh part.
    label = fields.pop("label")
    fields["gusheh_name"] = label.split("_", 1)[-1].strip() if "_" in label else label
    return fields


def read_contour(path: Path, tradition: str | None = None) -> IrmaContour | None:
    """Read one ``*_pitch_*.csv``. Returns ``None`` if the name does not parse.

    The files are headerless ``time_seconds,f0_hz`` pairs at roughly a 6 ms hop.
    Unvoiced frames are omitted rather than zero-marked, so gaps in the time
    column are silences.
    """
    fields = parse_filename(path.stem)
    if fields is None:
        return None

    data = np.loadtxt(path, delimiter=",", ndmin=2)
    if data.size == 0 or data.shape[1] < 2:
        return None

    if tradition is None:
        tradition = _tradition_of(path)

    return IrmaContour(
        key=fields["key"],
        tradition=tradition,
        gusheh_id=fields["gusheh_id"],
        gusheh_name=fields["gusheh_name"],
        performer=fields["performer"],
        source=path,
        times=data[:, 0],
        f0_hz=data[:, 1],
    )


def _tradition_of(path: Path) -> str:
    for parent in path.parents:
        if parent.name in {"Karimi", "Mirza Abdollah"}:
            return parent.name
    return "unknown"


def frame_durations(times: np.ndarray, gap_factor: float = 3.0) -> np.ndarray:
    """Seconds attributable to each frame.

    Silences appear as gaps in ``times``. Taking the raw difference would credit
    the frame before a long rest with the whole rest, so deltas are capped at a
    small multiple of the nominal hop.
    """
    if times.size == 0:
        return np.zeros(0)
    if times.size == 1:
        return np.ones(1)

    deltas = np.diff(times)
    hop = float(np.median(deltas))
    capped = np.minimum(deltas, hop * gap_factor)
    return np.append(capped, hop)


def to_pitch_track(contour: IrmaContour):
    """Wrap a contour as a :class:`~dastgah.core.audio.PitchTrack`.

    Reusing the audio pipeline's own container means IRMA is scored through
    exactly the same tuning estimation and binning as a real upload, rather than
    through a parallel code path that might diverge.
    """
    from dastgah.core.audio import PitchTrack
    from dastgah.core.audio import estimate_tuning

    durations = frame_durations(contour.times)
    voiced = contour.voiced
    f0 = np.where(voiced, contour.f0_hz, np.nan)
    reference, concentration = estimate_tuning(f0, voiced, durations)
    hop = float(np.median(np.diff(contour.times))) if contour.times.size > 1 else 0.006

    return PitchTrack(
        times=contour.times,
        f0_hz=f0,
        voiced=voiced,
        confidence=durations,  # weights frames by sounding time
        sr=1,
        hop_length=hop,  # seconds per frame, since sr is 1
        reference_hz=reference,
        tuning_concentration=concentration,
        duration=float(contour.times[-1]) if contour.times.size else 0.0,
    )


def iter_contours(irma_root: Path) -> Iterator[IrmaContour]:
    """Yield every pitch contour under an IRMA checkout."""
    for path in sorted(irma_root.rglob("*_pitch_*.csv")):
        contour = read_contour(path)
        if contour is not None and len(contour) > 0:
            yield contour


def load_contours(irma_root: Path) -> list[IrmaContour]:
    return list(iter_contours(irma_root))
