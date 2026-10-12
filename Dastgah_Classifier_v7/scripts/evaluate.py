"""Leave-one-out benchmark of the template classifier over the Radif Corpus.

Reproduces the accuracy figures quoted in the README and the web UI. For each
gusheh the templates are rebuilt from the other 228, so no piece is scored
against a template it contributed to.

    python scripts/evaluate.py [--corpus data/raw/radif_corpus/RadifCorpus/CSV]

Caveat: the scoring weights in ScoringConfig were themselves chosen using this
benchmark, so these numbers are optimistic relative to a truly held-out set.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dastgah.core.classify import DEFAULT_CONFIG, ScoringConfig, classify
from dastgah.radif.parse import Gusheh, load_corpus
from dastgah.radif.templates import build_templates
from dastgah.theory import MODAL_CLASSES_BY_KEY

DEFAULT_CORPUS = Path("data/raw/radif_corpus/RadifCorpus/CSV")


def features(gusheh: Gusheh) -> tuple[np.ndarray, np.ndarray]:
    histogram = np.zeros(24)
    transitions = np.zeros((24, 24))
    for note in gusheh.notes:
        histogram[note.pitch_class] += note.duration
    for current, following in zip(gusheh.notes, gusheh.notes[1:]):
        transitions[current.pitch_class, following.pitch_class] += 1.0
    return histogram, transitions


def parent(key: str) -> str:
    return MODAL_CLASSES_BY_KEY[key].parent or key


def evaluate(corpus: list[Gusheh], config: ScoringConfig) -> dict:
    correct13 = correct7 = correct_tonic = 0
    confusions: Counter[tuple[str, str]] = Counter()
    per_class: dict[str, list[int]] = defaultdict(list)
    confidences: list[tuple[float, bool]] = []

    for index, gusheh in enumerate(corpus):
        held_out = corpus[:index] + corpus[index + 1 :]
        templates = build_templates(held_out)
        histogram, transitions = features(gusheh)

        result = classify(histogram, templates, transitions=transitions, config=config)
        ranked = result.ranked_classes()
        predicted, probability = ranked[0]
        truth = gusheh.modal_class.key

        hit = predicted == truth
        correct13 += hit
        correct7 += parent(predicted) == parent(truth)
        if truth in templates:
            correct_tonic += result.best.tonic_pc == templates[truth].tonic_pc
        confusions[(truth, predicted)] += 1
        per_class[truth].append(int(hit))
        confidences.append((probability, hit))

    total = len(corpus)
    calibration_error = 0.0
    for low in np.arange(0, 1, 0.1):
        band = [(c, h) for c, h in confidences if low <= c < low + 0.1]
        if band:
            mean_confidence = float(np.mean([c for c, _ in band]))
            accuracy = float(np.mean([h for _, h in band]))
            calibration_error += len(band) / total * abs(accuracy - mean_confidence)

    return {
        "n": total,
        "accuracy_13": correct13 / total,
        "accuracy_7": correct7 / total,
        "tonic_accuracy": correct_tonic / total,
        "mean_confidence": float(np.mean([c for c, _ in confidences])),
        "calibration_error": calibration_error,
        "confusions": confusions,
        "per_class": per_class,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--sharpen", type=float, default=DEFAULT_CONFIG.sharpen)
    parser.add_argument(
        "--transition-weight", type=float, default=DEFAULT_CONFIG.transition_weight
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        parser.error(
            f"corpus not found at {args.corpus}\n"
            "Download it from https://zenodo.org/records/15742125 and unzip into data/raw/"
        )

    corpus = load_corpus(args.corpus)
    config = ScoringConfig(
        sharpen=args.sharpen, transition_weight=args.transition_weight
    )
    report = evaluate(corpus, config)

    print(f"Leave-one-out over {report['n']} gushehs")
    print(f"  13-class accuracy   {100 * report['accuracy_13']:5.1f}%   (chance 7.7%)")
    print(f"  7-class accuracy    {100 * report['accuracy_7']:5.1f}%   (avaz merged into parent)")
    print(f"  tonic accuracy      {100 * report['tonic_accuracy']:5.1f}%")
    print(f"  mean confidence     {100 * report['mean_confidence']:5.1f}%")
    print(f"  calibration error   {report['calibration_error']:.3f}")

    print("\nPer-class recall:")
    for key in MODAL_CLASSES_BY_KEY:
        hits = report["per_class"].get(key)
        if not hits:
            continue
        modal = MODAL_CLASSES_BY_KEY[key]
        print(
            f"  {modal.display:<26}{100 * sum(hits) / len(hits):5.1f}%  (n={len(hits)})"
        )

    print("\nTop confusions:")
    for (truth, predicted), count in report["confusions"].most_common():
        if truth != predicted and count >= 3:
            print(
                f"  {MODAL_CLASSES_BY_KEY[truth].name:<20} -> "
                f"{MODAL_CLASSES_BY_KEY[predicted].name:<20} {count}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
