"""End-to-end analysis of a recording, producing everything the UI displays.

Analysis runs twice over the audio: once globally, for the headline
classification, and once over overlapping windows, to trace where the
performance changes mode. The windowed pass matters because extended radif
performances modulate by design, and a single histogram over a modulating piece
smears the modes together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from dastgah.core.audio import (
    NoteEvent,
    PitchTrack,
    load_audio,
    note_events,
    note_histogram,
    pitch_histogram,
    track_pitch,
    transition_matrix,
)
from dastgah.core.classify import DEFAULT_CONFIG, ScoringConfig, classify
from dastgah.core.forud import Forud, find_foruds, tonic_prior
from dastgah.radif.gusheh import (
    GushehTemplate,
    identify_gusheh,
    load_gusheh_templates,
    tessitura,
)
from dastgah.radif.templates import (
    DASTGAHS_WITH_AUDIO,
    ModalTemplate,
    family_members,
    load_templates,
    restrict,
)
from dastgah.theory import (
    MODAL_CLASSES_BY_KEY,
    QUARTER_TONES_PER_OCTAVE,
    is_microtonal,
    pitch_class_name,
)

N = QUARTER_TONES_PER_OCTAVE

DEFAULT_TEMPLATE_PATH = Path(__file__).resolve().parent.parent / "data" / "templates.json"
DEFAULT_GUSHEH_PATH = Path(__file__).resolve().parent.parent / "data" / "gushehs.json"

#: Gusheh identification is a shortlist, not a verdict: top-1 accuracy is
#: about 22% against 4% for guessing, so several candidates are reported.
GUSHEH_SHORTLIST = 4

#: Softmax temperature for gusheh scores, calibrated on IRMA so the leading
#: candidate's reported probability tracks its measured accuracy: 20.1% mean
#: confidence against 25.3% top-1, expected calibration error 0.090. Chosen on
#: that set, so treat the confidences as indicative rather than held-out.
GUSHEH_TEMPERATURE = 0.5

#: Degrees below this share of total sounding time are treated as passing tones.
DEGREE_THRESHOLD = 0.02

#: A bin this many times weaker than a neighbour is treated as binning leakage.
LEAKAGE_RATIO = 3.0

#: Below this many note events a window is scored from frames instead, since a
#: handful of events is too sparse a histogram to match anything.
MIN_EVENTS_FOR_NOTE_HISTOGRAM = 8


@dataclass
class Degree:
    """One scale degree found in the performance."""

    interval: int              # quarter-tones above the estimated tonic
    pitch_class: int
    name: str
    weight: float              # share of sounding time
    microtonal: bool           # sits on a quarter-tone (koron/sori)
    cents_deviation: float     # measured offset from exact 24-TET


@dataclass
class GushehCandidate:
    """One gusheh the excerpt might be, with a calibrated-ish probability."""

    name: str
    probability: float


@dataclass
class Segment:
    """A stretch of the performance holding one mode."""

    start: float
    end: float
    key: str
    name: str
    tonic_name: str
    confidence: float
    gushehs: list[GushehCandidate] = field(default_factory=list)


@dataclass
class AnalysisResult:
    """Complete analysis of one recording."""

    source: str
    duration: float
    sample_rate: int
    reference_hz: float
    reference_cents: float     # tuning offset from A440
    tuning_concentration: float  # 0-1; how firmly the contour sits on the grid
    voiced_fraction: float
    n_note_events: int
    n_cadences: int

    key: str
    name: str
    persian: str
    kind: str
    confidence: float
    family: str                  # key of the mode family
    family_name: str             # display name of that family
    family_confidence: float     # probability the family is right
    family_members: list[str]    # display names of every mode in it
    tonic_pc: int
    tonic_name: str
    tonic_hz: float
    shahed_interval: int
    shahed_name: str

    ledger: list[dict] = field(default_factory=list)      # all 13, ranked
    gushehs: list[GushehCandidate] = field(default_factory=list)
    degrees: list[Degree] = field(default_factory=list)
    segments: list[Segment] = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        return payload

    def summary(self) -> str:
        covers = ", ".join(self.family_members)
        lines = [
            f"{self.family_name}   confidence {100 * self.family_confidence:.1f}%",
            f"    covers {covers}",
            f"  most likely {self.name} ({self.kind})   "
            f"confidence {100 * self.confidence:.1f}%",
            f"  tonic  {self.tonic_name} @ {self.tonic_hz:.1f} Hz"
            f"   shahed +{self.shahed_interval} ({self.shahed_name})",
            f"  tuning {self.reference_hz:.2f} Hz ({self.reference_cents:+.0f} cents vs A440)"
            f"   grid fit {self.tuning_concentration:.2f}",
            f"  voiced {100 * self.voiced_fraction:.0f}%   {self.n_note_events} events",
        ]
        micro = [d for d in self.degrees if d.microtonal]
        if micro:
            marks = "  ".join(
                f"{d.name}({d.cents_deviation:+.0f}c)" for d in micro
            )
            lines.append(f"  microtonal degrees: {marks}")
        if self.gushehs:
            shortlist = "  ".join(
                f"{g.name} {100 * g.probability:.0f}%" for g in self.gushehs
            )
            lines.append(f"  gusheh shortlist: {shortlist}")
        lines.append("  ledger:")
        for row in self.ledger[:5]:
            lines.append(f"    {row['name']:<24} {100 * row['probability']:5.1f}%")
        return "\n".join(lines)


def _family_confidence(
    ranked: list[tuple[str, float]],
    family_source: dict[str, ModalTemplate],
    family_key: str,
) -> float:
    """Total probability over the answers belonging to one family."""
    return sum(
        p
        for key, p in ranked
        if key in family_source and (family_source[key].family or key) == family_key
    )


def _tonic_prior(
    foruds: list[Forud], duration: float, config: ScoringConfig
) -> np.ndarray | None:
    """Cadence-based tonic evidence, or ``None`` if no cadence was found."""
    if not config.forud_prior_weight:
        return None
    return tonic_prior(
        foruds,
        total_duration=duration,
        recency_halflife=config.forud_recency_halflife,
    )


def _blend_prior(
    prior: np.ndarray | None, histogram: np.ndarray, config: ScoringConfig
) -> np.ndarray | None:
    """Mix cadence evidence with sounding time.

    Cadences are the better signal but a sparse one — a short or heavily
    ornamented excerpt may yield few, and a mode can cadence on a degree other
    than the tonic mid-performance. Blending keeps the sounding-time prior as a
    floor. Measured on two independent sets, the blend raises exact-mode
    accuracy on both while using the cadence signal where it exists.
    """
    if prior is None or histogram.sum() <= 0:
        return None
    weight = config.forud_prior_weight
    return weight * prior + (1.0 - weight) * (histogram / histogram.sum())


def _gushehs(
    events: list[NoteEvent],
    tonic_pc: int,
    mode: str,
    templates: dict[str, list[GushehTemplate]],
    *,
    start: float | None = None,
    end: float | None = None,
    top: int = GUSHEH_SHORTLIST,
) -> list[GushehCandidate]:
    """Shortlist the gushehs of ``mode`` that best fit an excerpt.

    Gushehs of one dastgah share its scale but differ in which degrees they
    dwell on, so they are ranked on the same tonic-relative profile the modes
    use, restricted to the identified mode's own repertoire.
    """
    candidates = templates.get(mode) or []
    if not candidates:
        return []

    within = [
        e
        for e in events
        if (start is None or e.start >= start) and (end is None or e.start < end)
    ]
    if len(within) < MIN_EVENTS_FOR_NOTE_HISTOGRAM:
        return []

    histogram = np.zeros(N)
    for event in within:
        histogram[(event.pitch_class - tonic_pc) % N] += event.duration
    if histogram.sum() <= 0:
        return []

    quarter = np.array([e.quarter_tone for e in within], dtype=float)
    durations = np.array([e.duration for e in within], dtype=float)
    ranked = identify_gusheh(
        histogram, tessitura(quarter, durations, tonic_pc), candidates
    )[:top]
    if not ranked:
        return []

    scores = np.array([m.score for m in ranked]) / GUSHEH_TEMPERATURE
    scores -= scores.max()
    weights = np.exp(scores)
    weights /= weights.sum()
    return [
        GushehCandidate(name=m.name, probability=round(float(w), 4))
        for m, w in zip(ranked, weights, strict=True)
    ]


def _degrees(
    histogram: np.ndarray,
    tonic_pc: int,
    events: list[NoteEvent],
    threshold: float = DEGREE_THRESHOLD,
) -> list[Degree]:
    """Describe the scale degrees the performance actually used."""
    total = histogram.sum()
    if total <= 0:
        return []
    share = histogram / total

    deviation_by_pc: dict[int, list[float]] = {}
    for event in events:
        deviation_by_pc.setdefault(event.pitch_class, []).append(event.cents_deviation)

    found: list[Degree] = []
    for pc in range(N):
        if share[pc] < threshold:
            continue
        # Soft binning spreads a strong note into its neighbours. Left in, that
        # leakage is reported as a koron or sori that was never played, so drop
        # any bin far weaker than an immediate neighbour.
        neighbours = max(share[(pc - 1) % N], share[(pc + 1) % N])
        if share[pc] * LEAKAGE_RATIO < neighbours:
            continue
        interval = (pc - tonic_pc) % N
        deviations = deviation_by_pc.get(pc, [])
        found.append(
            Degree(
                interval=interval,
                pitch_class=pc,
                name=pitch_class_name(pc),
                weight=round(float(share[pc]), 4),
                microtonal=is_microtonal(interval),
                cents_deviation=round(float(np.mean(deviations)), 1) if deviations else 0.0,
            )
        )
    return sorted(found, key=lambda d: d.interval)


def _window_histogram(
    track: PitchTrack,
    events: list[NoteEvent],
    start: float,
    end: float,
) -> np.ndarray:
    """Note-event histogram for one window, falling back to frames if sparse."""
    within = [e for e in events if start <= e.start < end]
    if len(within) >= MIN_EVENTS_FOR_NOTE_HISTOGRAM:
        return note_histogram(within)
    return pitch_histogram(track, start=start, end=end)


def _window_emissions(
    track: PitchTrack,
    events: list[NoteEvent],
    templates: dict[str, ModalTemplate],
    config: ScoringConfig,
    keys: list[str],
    window: float,
    hop: float,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, int]]]:
    """Score every window against every mode.

    Returns window centre times, a ``[n_windows, n_modes]`` matrix of log
    probabilities, and each window's best tonic per mode.
    """
    centres: list[float] = []
    rows: list[np.ndarray] = []
    tonics: list[dict[str, int]] = []

    start = 0.0
    while start + window <= track.duration + 1e-6:
        end = start + window
        histogram = _window_histogram(track, events, start, end)
        if histogram.sum() > 0:
            try:
                result = classify(histogram, templates, config=config)
            except ValueError:
                start += hop
                continue
            marginal = dict(result.ranked_classes())
            best_tonic: dict[str, int] = {}
            for candidate in result.candidates:
                best_tonic.setdefault(candidate.key, candidate.tonic_pc)

            centres.append((start + end) / 2.0)
            rows.append(np.log([max(marginal.get(k, 0.0), 1e-12) for k in keys]))
            tonics.append(best_tonic)
        start += hop

    if not rows:
        return np.zeros(0), np.zeros((0, len(keys))), []
    return np.array(centres), np.vstack(rows), tonics


def _viterbi(emissions: np.ndarray, switch_penalty: float) -> np.ndarray:
    """Most likely mode sequence, charging ``switch_penalty`` per mode change.

    Windows are scored independently and a single window rarely holds enough
    notes to identify a mode on its own, so the raw argmax flickers. Requiring a
    change to be paid for means only sustained evidence moves the decoded path,
    which is also the musical reality: a performance settles into a mode rather
    than alternating every few seconds.

    Because every switch costs the same, the best predecessor of any state is
    either that state itself or the single best state overall, so each step is
    linear in the number of modes rather than quadratic.
    """
    n_windows, n_states = emissions.shape
    if n_windows == 0:
        return np.zeros(0, dtype=int)

    scores = emissions[0].copy()
    backpointers = np.zeros((n_windows, n_states), dtype=int)
    states = np.arange(n_states)

    for t in range(1, n_windows):
        switch_score = scores.max() - switch_penalty
        switch_source = int(scores.argmax())

        hold = scores >= switch_score
        backpointers[t] = np.where(hold, states, switch_source)
        scores = np.where(hold, scores, switch_score) + emissions[t]

    path = np.zeros(n_windows, dtype=int)
    path[-1] = int(scores.argmax())
    for t in range(n_windows - 1, 0, -1):
        path[t - 1] = backpointers[t, path[t]]
    return path


def _segments(
    track: PitchTrack,
    events: list[NoteEvent],
    templates: dict[str, ModalTemplate],
    config: ScoringConfig,
    gusheh_templates: dict[str, list[GushehTemplate]] | None = None,
    tonic_pc: int | None = None,
    *,
    window: float = 20.0,
    hop: float = 5.0,
    switch_penalty: float = 4.0,
    min_segment: float = 15.0,
) -> list[Segment]:
    """Trace where the performance changes mode.

    Overlapping windows are scored, smoothed with Viterbi, and boundaries are
    placed midway between the centres of the windows that disagree — windows
    overlap, so using their edges would produce segments that overlap too.
    """
    if track.duration < window:
        return []

    keys = sorted(templates)
    centres, emissions, tonics = _window_emissions(
        track, events, templates, config, keys, window, hop
    )
    if len(centres) == 0:
        return []

    path = _viterbi(emissions, switch_penalty)

    # Group consecutive windows sharing a mode, then convert to time spans.
    segments: list[Segment] = []
    run_start = 0
    for index in range(1, len(path) + 1):
        if index < len(path) and path[index] == path[run_start]:
            continue

        state = int(path[run_start])
        key = keys[state]
        last = index - 1

        begin = 0.0 if run_start == 0 else (centres[run_start - 1] + centres[run_start]) / 2
        finish = (
            track.duration if index >= len(path) else (centres[last] + centres[index]) / 2
        )

        probabilities = np.exp(emissions[run_start:index, state])
        tonic_counts: dict[int, int] = {}
        for window_index in range(run_start, index):
            tonic = tonics[window_index].get(key)
            if tonic is not None:
                tonic_counts[tonic] = tonic_counts.get(tonic, 0) + 1
        tonic_pc = max(tonic_counts, key=tonic_counts.get) if tonic_counts else 0

        gushehs: list[GushehCandidate] = []
        if gusheh_templates is not None:
            gushehs = _gushehs(
                events,
                tonic_pc,
                key,
                gusheh_templates,
                start=float(begin),
                end=float(finish),
                top=3,
            )

        segments.append(
            Segment(
                start=round(float(begin), 2),
                end=round(float(finish), 2),
                key=key,
                name=MODAL_CLASSES_BY_KEY[key].name,
                tonic_name=pitch_class_name(tonic_pc),
                confidence=round(float(probabilities.mean()), 4),
                gushehs=gushehs,
            )
        )
        run_start = index

    kept = [s for s in segments if s.end - s.start >= min_segment]
    if not kept:
        return []

    # Absorbing a dropped short segment leaves a gap; close it so the segments
    # still tile the recording end to end.
    kept[0].start = 0.0
    kept[-1].end = round(track.duration, 2)
    for previous, following in zip(kept, kept[1:]):
        if previous.end != following.start:
            midpoint = round((previous.end + following.start) / 2, 2)
            previous.end = midpoint
            following.start = midpoint
    return kept


def analyze(
    path: Path | str,
    *,
    templates: dict[str, ModalTemplate] | None = None,
    gusheh_templates: dict[str, list[GushehTemplate]] | None = None,
    config: ScoringConfig = DEFAULT_CONFIG,
    with_segments: bool = True,
    with_gushehs: bool = True,
    answer_space: tuple[str, ...] | None = DASTGAHS_WITH_AUDIO,
) -> AnalysisResult:
    """Analyse a recording and return its modal classification.

    By default the answer is one of the six dastgahs with audio evidence behind
    them. All thirteen templates still score — an avaz profile is evidence for
    its mother dastgah, and folding that evidence in beats leaving the avaz
    templates out by 5 to 15 points — but avaz probability is folded into the
    parent rather than reported separately, since avaz readings are only 0-22%
    accurate. Pass ``answer_space=None`` to get all thirteen classes back.
    """
    path = Path(path)
    if templates is None:
        templates = load_templates(DEFAULT_TEMPLATE_PATH)
    if gusheh_templates is None and with_gushehs:
        gusheh_templates = load_gusheh_templates(DEFAULT_GUSHEH_PATH)

    y, sr = load_audio(path)
    track = track_pitch(y, sr)
    events = note_events(track)

    # Cadences say where the line comes to rest, which is the tonic. Sounding
    # time alone elects the shahed instead, so blend the two.
    foruds = find_foruds(events, total_duration=track.duration)
    prior = _tonic_prior(foruds, track.duration, config)

    # Score from note events so the observation matches how templates are built.
    histogram = note_histogram(events)
    if len(events) < MIN_EVENTS_FOR_NOTE_HISTOGRAM or histogram.sum() <= 0:
        histogram = pitch_histogram(track)
    if histogram.sum() <= 0:
        raise ValueError(
            f"no pitched content detected in {path.name}; "
            "the file may be silent, percussive, or outside the 70-1200 Hz range"
        )
    result = classify(
        histogram,
        templates,
        transitions=transition_matrix(events),
        tonic_prior=_blend_prior(prior, histogram, config),
        config=config,
    )
    ranked = result.ranked_classes()

    if answer_space is None:
        ranked = result.ranked_classes()
        best = result.best
        family_source = templates
    else:
        ranked = result.ranked_dastgahs(answer_space)
        # The tonic must come from whichever profile actually matched, which may
        # be an avaz of the winning dastgah rather than the dastgah itself.
        best = result.best_for_dastgah(ranked[0][0])
        family_source = restrict(templates, answer_space)

    winner = ranked[0][0]
    modal = MODAL_CLASSES_BY_KEY[winner]
    template = templates[best.key]
    family_key = family_source[winner].family or winner
    members = family_members(family_source)
    shahed_pc = (best.tonic_pc + template.shahed_interval) % N
    tonic_hz = track.reference_hz * (2.0 ** ((best.tonic_pc - 138 % N) / 24.0))
    # express the tonic in the octave nearest the performance's own register
    sounding = track.f0_hz[track.voiced & np.isfinite(track.f0_hz)]
    if sounding.size:
        centre = float(np.median(sounding))
        while tonic_hz < centre / 2:
            tonic_hz *= 2
        while tonic_hz > centre * 2:
            tonic_hz /= 2

    return AnalysisResult(
        source=path.name,
        duration=round(track.duration, 2),
        sample_rate=sr,
        reference_hz=round(track.reference_hz, 2),
        reference_cents=round(1200 * float(np.log2(track.reference_hz / 440.0)), 1),
        tuning_concentration=round(track.tuning_concentration, 4),
        voiced_fraction=round(track.voiced_fraction, 4),
        n_note_events=len(events),
        n_cadences=len(foruds),
        key=winner,
        name=modal.display,
        persian=modal.persian,
        kind=modal.kind,
        confidence=round(float(dict(ranked)[winner]), 4),
        family=family_key,
        family_name=f"{MODAL_CLASSES_BY_KEY[family_key].name} group",
        family_confidence=round(float(_family_confidence(ranked, family_source, family_key)), 4),
        family_members=[
            MODAL_CLASSES_BY_KEY[k].name for k in members.get(family_key, [family_key])
        ],
        tonic_pc=best.tonic_pc,
        tonic_name=best.tonic_name,
        tonic_hz=round(tonic_hz, 2),
        shahed_interval=template.shahed_interval,
        shahed_name=pitch_class_name(shahed_pc),
        ledger=[
            {
                "key": k,
                "name": MODAL_CLASSES_BY_KEY[k].display,
                "short_name": MODAL_CLASSES_BY_KEY[k].name,
                "persian": MODAL_CLASSES_BY_KEY[k].persian,
                "kind": MODAL_CLASSES_BY_KEY[k].kind,
                "family": (family_source[k].family or k) if k in family_source else k,
                "probability": round(float(p), 5),
            }
            for k, p in ranked
        ],
        degrees=_degrees(histogram, best.tonic_pc, events),
        gushehs=(
            _gushehs(events, best.tonic_pc, winner, gusheh_templates)
            if gusheh_templates
            else []
        ),
        segments=(
            _segments(
                track, events, templates, config, gusheh_templates, best.tonic_pc
            )
            if with_segments
            else []
        ),
    )
