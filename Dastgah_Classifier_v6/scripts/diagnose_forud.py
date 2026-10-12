"""Which properties of a detected cadence predict that it is a real forud?

    python scripts/diagnose_forud.py --irma data/raw/irma --audio data/raw/kdc

Every attempt to use the cadence prior better has failed, and this says why: the
detector accepts any phrase ending lower than it started, so most of what it
returns is a phrase-rest, and its own ``strength`` does not tell the two apart.

The label is weak but independent. A cadence counts as real when it resolves on
the tonic that best fits the *true* mode's notated profile and bigrams, scored
with the tonic prior switched off so the detector takes no part in judging
itself. That reference is the profile's opinion rather than ground truth, which
is why the interesting output is the *ranking* of features rather than any
absolute number.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.audio import (
    load_audio,
    note_events,
    note_histogram,
    track_pitch,
    transition_matrix,
)
from dastgah.core.classify import DEFAULT_CONFIG
from dastgah.core.forud import PHRASE_GAP, _phrases, find_foruds
from dastgah.core.learn import build_design, initial_parameters
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates
from dastgah.theory import MODAL_CLASSES_BY_KEY

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".m4a", ".ogg"}


def mother(key: str) -> str:
    return MODAL_CLASSES_BY_KEY[key].parent or key


def from_irma(root: Path):
    from dastgah.radif.irma import load_contours, to_pitch_track

    for contour in load_contours(root):
        if mother(contour.key) not in DASTGAHS_WITH_AUDIO:
            continue
        track = to_pitch_track(contour)
        yield mother(contour.key), note_events(track), track.duration


def from_audio(root: Path):
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        key = path.parent.name.lower()
        if key not in MODAL_CLASSES_BY_KEY or mother(key) not in DASTGAHS_WITH_AUDIO:
            continue
        try:
            y, sr = load_audio(path)
            track = track_pitch(y, sr)
        except Exception:  # noqa: BLE001 - a bad file should not stop the run
            continue
        yield mother(key), note_events(track), track.duration


def features(source, templates) -> tuple[list[dict], np.ndarray]:
    kept = []
    for truth, events, duration in source:
        if len(events) < 8:
            continue
        histogram = note_histogram(events)
        if histogram.sum() <= 0:
            continue
        kept.append((truth, events, duration, histogram, transition_matrix(events)))

    design = build_design(
        [{"h": h, "B": b, "log_prior": None} for _, _, _, h, b in kept],
        templates, config=DEFAULT_CONFIG, answer_space=DASTGAHS_WITH_AUDIO,
    )
    theory = initial_parameters(design, DEFAULT_CONFIG)
    scores = (
        theory.alpha * np.einsum("itd,md->itm", design.rotated,
                                 theory.log_profiles(design.log_theory))
        + theory.transition_weight * design.transition
    )
    mothers = np.array(design.mothers)

    rows, labels = [], []
    for index, (truth, events, duration, _, _) in enumerate(kept):
        columns = np.flatnonzero(mothers == truth)
        reference = int(scores[index][:, columns].max(axis=1).argmax())

        pitches = np.array([e.quarter_tone for e in events])
        low, span = pitches.min(), max(pitches.max() - pitches.min(), 1e-9)

        foruds = find_foruds(events, total_duration=duration)
        weight_by_pc: dict[int, float] = {}
        for f in foruds:
            weight_by_pc[f.resolution_pc] = weight_by_pc.get(f.resolution_pc, 0.0) + f.strength
        starts = {p[0].start: p for p in _phrases(events, PHRASE_GAP)}

        for f in foruds:
            phrase = starts.get(f.start)
            if phrase is None:
                continue
            q = np.array([e.quarter_tone for e in phrase])
            rows.append({
                "strength": f.strength,
                "descent": f.descent,
                "repose": f.repose,
                "silence_after": f.silence_after,
                "duration": f.end - f.start,
                "is_phrase_min": float(q[-1] <= q.min() + 1e-9),
                # Where the resolution sits in the performance's own register.
                # The ist is low in a performance's range; the shahed is not.
                "register_position": float((q[-1] - low) / span),
                "consensus": weight_by_pc[f.resolution_pc] / max(sum(weight_by_pc.values()), 1e-9),
                "relative_time": f.end / max(duration, 1e-9),
            })
            labels.append(int(f.resolution_pc == reference))
    return rows, np.array(labels)


def report(name: str, rows: list[dict], labels: np.ndarray) -> None:
    from sklearn.metrics import roc_auc_score

    print(f"\n{name}: {len(labels)} cadences, "
          f"{100 * labels.mean():.1f}% land on the reference tonic")
    print(f"  {'feature':<20} {'AUC':>7}")
    for field in rows[0]:
        values = np.array([r[field] for r in rows])
        print(f"  {field:<20} {roc_auc_score(labels, values):7.3f}")

    def column(field):
        return np.array([r[field] for r in rows])

    repose = np.minimum(column("repose") / 2.0, 1.0)
    silence = np.minimum(column("silence_after") / 1.0, 1.0)
    lowness = 1.0 - column("register_position")
    candidates = {
        "shipped strength": column("strength"),
        "lowness": lowness,
        "lowness x repose": lowness * (0.5 + 0.5 * repose),
        "lowness x repose x silence": lowness * (0.5 + 0.5 * repose) * (0.5 + 0.5 * silence),
    }
    print(f"  {'candidate score':<30} {'AUC':>7} {'top-quartile precision':>24}")
    for label, values in candidates.items():
        cut = np.quantile(values, 0.75)
        selected = values >= cut
        precision = 100 * labels[selected].mean() if selected.any() else float("nan")
        print(f"  {label:<30} {roc_auc_score(labels, values):7.3f} {precision:23.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irma", type=Path, default=None)
    parser.add_argument("--audio", type=Path, nargs="*", default=())
    args = parser.parse_args()

    templates = load_templates(DEFAULT_TEMPLATE_PATH)
    if args.irma and args.irma.exists():
        rows, labels = features(from_irma(args.irma), templates)
        if rows:
            report(f"irma ({args.irma})", rows, labels)
    for root in args.audio:
        if not Path(root).is_dir():
            continue
        rows, labels = features(from_audio(Path(root)), templates)
        if rows:
            report(f"audio ({root})", rows, labels)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
