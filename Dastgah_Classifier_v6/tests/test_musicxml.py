"""Tests for MusicXML export."""

from dataclasses import dataclass, field
from xml.etree import ElementTree as ET

import pytest

from dastgah.core.analyze import Degree
from dastgah.core.musicxml import scale_to_musicxml


@dataclass
class FakeResult:
    """Minimal stand-in for AnalysisResult, so tests need no audio."""

    source: str = "test.wav"
    name: str = "Dastgāh-e Chahārgāh"
    family_name: str = "Chahārgāh group"
    family_confidence: float = 0.99
    confidence: float = 0.99
    tonic_name: str = "C"
    tonic_hz: float = 261.6
    shahed_name: str = "Ak"
    reference_hz: float = 440.0
    reference_cents: float = 0.0
    degrees: list = field(default_factory=list)


def chahargah() -> FakeResult:
    """C, D-koron, E, F, G, A-koron, B."""
    names = [(0, "C"), (3, "Dk"), (8, "E"), (10, "F"), (14, "G"), (17, "Ak"), (22, "B")]
    return FakeResult(
        degrees=[
            Degree(
                interval=i,
                pitch_class=i,
                name=n,
                weight=0.1,
                microtonal=i % 2 == 1,
                cents_deviation=0.0,
            )
            for i, n in names
        ]
    )


def parse(xml: str) -> ET.Element:
    """Parse past the XML declaration and doctype."""
    return ET.fromstring(xml.split("\n", 2)[2])


def test_export_is_well_formed():
    assert parse(scale_to_musicxml(chahargah())).tag == "score-partwise"


def test_scale_closes_on_the_octave():
    """Seven degrees produce eight notes: the scale plus its upper tonic."""
    root = parse(scale_to_musicxml(chahargah()))
    notes = root.findall(".//note")
    assert len(notes) == 8
    first = notes[0].find("pitch")
    last = notes[-1].find("pitch")
    assert first.find("step").text == last.find("step").text == "C"
    assert int(last.find("octave").text) == int(first.find("octave").text) + 1


def test_koron_is_a_half_flat_not_a_flat():
    """A koron lowers a quarter-tone: alter -0.5, glyph quarter-flat."""
    root = parse(scale_to_musicxml(chahargah()))
    altered = {
        n.find("pitch/step").text: n.find("pitch/alter").text
        for n in root.findall(".//note")
        if n.find("pitch/alter") is not None
    }
    assert altered["D"] == "-0.5"
    assert altered["A"] == "-0.5"
    glyphs = [a.text for a in root.findall(".//accidental")]
    assert glyphs == ["quarter-flat", "quarter-flat"]


def test_naturals_carry_no_alteration():
    root = parse(scale_to_musicxml(chahargah()))
    naturals = [
        n
        for n in root.findall(".//note")
        if n.find("pitch/step").text in {"C", "E", "F", "G", "B"}
    ]
    assert all(n.find("pitch/alter") is None for n in naturals)


def test_octave_advances_when_the_scale_wraps_past_b():
    root = parse(scale_to_musicxml(chahargah()))
    octaves = [int(n.find("pitch/octave").text) for n in root.findall(".//note")]
    assert octaves == sorted(octaves), "an ascending scale must not descend"


def test_no_key_signature_is_written():
    """Persian modes are neither major nor minor; a key would misrepresent them."""
    assert parse(scale_to_musicxml(chahargah())).find(".//key") is None


def test_metadata_records_the_identification():
    xml = scale_to_musicxml(chahargah())
    root = parse(xml)
    assert "Chahārgāh" in root.find(".//work-title").text
    description = root.find(".//encoding-description").text
    assert "test.wav" in description
    assert "440" in description


def test_a_sori_reads_as_a_half_sharp():
    result = chahargah()
    result.degrees = [
        Degree(interval=0, pitch_class=0, name="C", weight=0.5, microtonal=False,
               cents_deviation=0.0),
        Degree(interval=5, pitch_class=5, name="Ds", weight=0.5, microtonal=True,
               cents_deviation=0.0),
    ]
    root = parse(scale_to_musicxml(result))
    assert [a.text for a in root.findall(".//accidental")] == ["quarter-sharp"]
    assert root.findall(".//note")[1].find("pitch/alter").text == "0.5"


def test_an_empty_scale_still_exports():
    result = chahargah()
    result.degrees = []
    root = parse(scale_to_musicxml(result))
    assert len(root.findall(".//note")) == 1  # just the closing tonic
