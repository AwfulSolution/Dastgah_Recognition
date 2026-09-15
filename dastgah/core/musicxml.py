"""MusicXML export of an identified mode.

Writes the scale the analysis found — the tonic and the degrees actually used,
with koron and sori notated properly — so the result can be opened in notation
software rather than only read as numbers.

Microtones are expressed the way MusicXML provides for them: ``<alter>`` takes a
decimal number of semitones, so a koron is ``-0.5`` and a sori ``+0.5``, paired
with the ``quarter-flat`` and ``quarter-sharp`` accidental glyphs. Notation
programs that do not render quarter-tones will still read the correct pitch.

Only the scale is exported, not the performance. Transcribing the melody would
mean quantising rhythm, and the radif's non-metric genres have no beat to
quantise to; an exported rhythm would be an invention rather than a measurement.
"""

from __future__ import annotations

from xml.etree import ElementTree as ET

from dastgah.theory import pitch_class_name

#: Accidental suffix in a pitch-class name -> (semitone alteration, glyph).
_ACCIDENTALS = {
    "": (0.0, "natural"),
    "b": (-1.0, "flat"),
    "#": (1.0, "sharp"),
    "k": (-0.5, "quarter-flat"),   # koron
    "s": (0.5, "quarter-sharp"),   # sori
}

#: Diatonic order, for deciding when an ascending scale crosses into the next
#: octave.
_STEPS = "CDEFGAB"

_DOCTYPE = (
    '<!DOCTYPE score-partwise PUBLIC '
    '"-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
    '"http://www.musicxml.org/dtds/partwise.dtd">'
)


def _split(name: str) -> tuple[str, float, str]:
    """Split a pitch-class name such as ``Ak`` into step, alteration, glyph."""
    step = name[0]
    alter, glyph = _ACCIDENTALS[name[1:]]
    return step, alter, glyph


def _note(parent: ET.Element, step: str, alter: float, glyph: str, octave: int) -> None:
    note = ET.SubElement(parent, "note")
    pitch = ET.SubElement(note, "pitch")
    ET.SubElement(pitch, "step").text = step
    if alter:
        # Integers render as "1" rather than "1.0"; quarter-tones keep a decimal.
        ET.SubElement(pitch, "alter").text = (
            str(int(alter)) if alter == int(alter) else str(alter)
        )
    ET.SubElement(pitch, "octave").text = str(octave)
    ET.SubElement(note, "duration").text = "1"
    ET.SubElement(note, "type").text = "quarter"
    if glyph != "natural":
        ET.SubElement(note, "accidental").text = glyph


def scale_to_musicxml(result) -> str:
    """Render an :class:`~dastgah.core.analyze.AnalysisResult` as MusicXML."""
    score = ET.Element("score-partwise", version="4.0")

    work = ET.SubElement(score, "work")
    ET.SubElement(work, "work-title").text = (
        f"{result.name} — tonic {result.tonic_name}"
    )
    identification = ET.SubElement(score, "identification")
    encoding = ET.SubElement(identification, "encoding")
    ET.SubElement(encoding, "software").text = "dastgah"
    ET.SubElement(encoding, "encoding-description").text = (
        f"Scale identified from {result.source}. "
        f"Mode family {result.family_name} at "
        f"{result.family_confidence * 100:.1f}% confidence; "
        f"{result.name} at {result.confidence * 100:.1f}%. "
        f"Tuning reference {result.reference_hz:.1f} Hz "
        f"({result.reference_cents:+.0f} cents from A440)."
    )

    part_list = ET.SubElement(score, "part-list")
    score_part = ET.SubElement(part_list, "score-part", id="P1")
    ET.SubElement(score_part, "part-name").text = result.name

    part = ET.SubElement(score, "part", id="P1")
    measure = ET.SubElement(part, "measure", number="1")

    attributes = ET.SubElement(measure, "attributes")
    ET.SubElement(attributes, "divisions").text = "1"
    # No key signature: Persian modes are not major or minor, and forcing one
    # would misrepresent the accidentals, which are properties of the mode.
    time = ET.SubElement(attributes, "time")
    ET.SubElement(time, "beats").text = str(max(len(result.degrees) + 1, 2))
    ET.SubElement(time, "beat-type").text = "4"
    clef = ET.SubElement(attributes, "clef")
    ET.SubElement(clef, "sign").text = "G"
    ET.SubElement(clef, "line").text = "2"

    direction = ET.SubElement(measure, "direction", placement="above")
    direction_type = ET.SubElement(direction, "direction-type")
    ET.SubElement(direction_type, "words").text = (
        f"{result.name} · tonic {result.tonic_name} "
        f"({result.tonic_hz:.1f} Hz) · shāhed {result.shahed_name}"
    )

    # Ascend from the tonic through the degrees that were actually used, then
    # close on the octave.
    octave = 4
    previous = -1
    for degree in sorted(result.degrees, key=lambda d: d.interval):
        step, alter, glyph = _split(pitch_class_name(degree.pitch_class))
        index = _STEPS.index(step)
        if previous >= 0 and index <= previous:
            octave += 1
        previous = index
        _note(measure, step, alter, glyph, octave)

    step, alter, glyph = _split(result.tonic_name)
    _note(measure, step, alter, glyph, octave + 1)

    body = ET.tostring(score, encoding="unicode")
    return f'<?xml version="1.0" encoding="UTF-8"?>\n{_DOCTYPE}\n{body}\n'
