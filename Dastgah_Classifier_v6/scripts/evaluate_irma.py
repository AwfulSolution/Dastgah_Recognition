"""Evaluate the templates against IRMA's audio-derived pitch contours.

This is the out-of-domain test. The templates are built from the *notated*
Mirza Abdollah radif; IRMA's contours are extracted from recordings of the
*Karimi* radif. Tradition, medium and performer all differ, so nothing here
overlaps the training material and no leave-one-out is needed.

    python scripts/evaluate_irma.py [--irma data/raw/irma]
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dastgah.core.analyze import (
    DEFAULT_TEMPLATE_PATH,
    MIN_EVENTS_FOR_NOTE_HISTOGRAM,
)
from dastgah.core.audio import (
    note_events,
    note_histogram,
    pitch_histogram,
    transition_matrix,
)
from dastgah.core.analyze import _blend_prior, _tonic_prior
from dastgah.core.classify import DEFAULT_CONFIG, ScoringConfig, classify
from dastgah.core.forud import find_foruds
from dastgah.radif.irma import IrmaContour, load_contours, to_pitch_track
from dastgah.radif.templates import load_templates
from dastgah.theory import MODAL_CLASSES_BY_KEY


def parent(key: str) -> str:
    return MODAL_CLASSES_BY_KEY[key].parent or key


def evaluate(
    contours: list[IrmaContour],
    templates: dict,
    config: ScoringConfig,
    *,
    use_transitions: bool = True,
) -> dict:
    correct13 = correct7 = correct_family = 0
    family_of = {k: (t.family or k) for k, t in templates.items()}
    confusions: Counter[tuple[str, str]] = Counter()
    per_class: dict[str, list[int]] = defaultdict(list)
    confidences: list[tuple[float, bool]] = []
    ranks: list[int] = []

    for contour in contours:
        track = to_pitch_track(contour)
        events = note_events(track)

        # Mirror analyze(): score from note events, fall back to frames if sparse.
        histogram = note_histogram(events)
        if len(events) < MIN_EVENTS_FOR_NOTE_HISTOGRAM or histogram.sum() <= 0:
            histogram = pitch_histogram(track)
        if histogram.sum() <= 0:
            continue

        transitions = transition_matrix(events) if use_transitions else None
        foruds = find_foruds(events, total_duration=track.duration)
        prior = _blend_prior(
            _tonic_prior(foruds, track.duration, config), histogram, config
        )

        result = classify(
            histogram, templates, transitions=transitions, tonic_prior=prior, config=config
        )
        ranked = result.ranked_classes()
        predicted, probability = ranked[0]
        truth = contour.key

        hit = predicted == truth
        correct13 += hit
        correct7 += parent(predicted) == parent(truth)
        correct_family += family_of.get(predicted, predicted) == family_of.get(truth, truth)
        confusions[(truth, predicted)] += 1
        per_class[truth].append(int(hit))
        confidences.append((probability, hit))
        ranks.append([k for k, _ in ranked].index(truth) + 1)

    total = len(confidences)
    calibration_error = 0.0
    for low in np.arange(0, 1, 0.1):
        band = [(c, h) for c, h in confidences if low <= c < low + 0.1]
        if band:
            calibration_error += (
                len(band)
                / total
                * abs(np.mean([h for _, h in band]) - np.mean([c for c, _ in band]))
            )

    return {
        "n": total,
        "accuracy_13": correct13 / total,
        "accuracy_7": correct7 / total,
        "accuracy_family": correct_family / total,
        "top3": sum(r <= 3 for r in ranks) / total,
        "mean_rank": float(np.mean(ranks)),
        "mean_confidence": float(np.mean([c for c, _ in confidences])),
        "calibration_error": calibration_error,
        "confusions": confusions,
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--irma", type=Path, default=Path("data/raw/irma"))
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--no-transitions", action="store_true")
    args = parser.parse_args()

    if not args.irma.exists():
        parser.error(f"IRMA not found at {args.irma}; run ./scripts/fetch_data.sh irma")

    contours = load_contours(args.irma)
    if not contours:
        parser.error(f"no pitch contours under {args.irma}")

    templates = load_templates(args.templates)
    report = evaluate(
        contours, templates, DEFAULT_CONFIG, use_transitions=not args.no_transitions
    )

    hours = sum(c.times[-1] for c in contours) / 3600
    present = {c.key for c in contours}
    print(f"IRMA held-out evaluation: {report['n']} contours, {hours:.1f} h")
    print(f"  templates from notated Mirza Abdollah radif; contours from Karimi recordings")
    print(f"  {len(present)} of 13 modes present\n")
    print(f"  13-class accuracy   {100 * report['accuracy_13']:5.1f}%   (chance {100 / len(present):.1f}%)")
    print(f"  7-class accuracy    {100 * report['accuracy_7']:5.1f}%")
    print(f"  top-3 accuracy      {100 * report['top3']:5.1f}%")
    print(f"  FAMILY accuracy     {100 * report['accuracy_family']:5.1f}%")
    print(f"  mean rank of truth  {report['mean_rank']:5.2f} of 13")
    print(f"  mean confidence     {100 * report['mean_confidence']:5.1f}%")
    print(f"  calibration error   {report['calibration_error']:.3f}")

    print("\nPer-class recall:")
    for key in MODAL_CLASSES_BY_KEY:
        hits = report["per_class"].get(key)
        if not hits:
            continue
        print(
            f"  {MODAL_CLASSES_BY_KEY[key].display:<26}"
            f"{100 * sum(hits) / len(hits):5.1f}%  (n={len(hits)})"
        )

    print("\nTop confusions:")
    for (truth, predicted), count in report["confusions"].most_common():
        if truth != predicted and count >= 2:
            print(
                f"  {MODAL_CLASSES_BY_KEY[truth].name:<20} -> "
                f"{MODAL_CLASSES_BY_KEY[predicted].name:<20} {count}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
