"""How much shrinkage toward theory does a fitted model want?

    python scripts/sweep_penalty.py --cache data/cache/nava.pkl

Every free parameter is pulled toward the value the notated radif gives it, so
the penalty interpolates between the hand-built classifier (large penalty) and
an unconstrained fit (zero). The sweep is artist-grouped: a random split would
be meaningless when 11 of Nava's artists play a single dastgah.
"""

from __future__ import annotations

import argparse
import collections
import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import GroupKFold

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.core.learn import (
    LEVELS,
    build_design,
    dastgah_probabilities,
    fit,
    initial_parameters,
)
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

from train_hybrid import load_records  # noqa: E402  - sibling script

PENALTIES = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/nava.pkl"))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-recordings", type=int, default=15)
    parser.add_argument("--balanced", action="store_true")
    args = parser.parse_args()

    templates = load_templates(DEFAULT_TEMPLATE_PATH)
    records = [
        r for r in load_records(args.cache, templates)
        if r["truth"] in DASTGAHS_WITH_AUDIO
    ]
    design = build_design(
        records, templates, config=DEFAULT_CONFIG, answer_space=DASTGAHS_WITH_AUDIO
    )
    truth = [r["truth"] for r in records]
    artists = np.array([r["artist"] for r in records])
    _, dastgahs = dastgah_probabilities(initial_parameters(design), design)
    target = np.array([dastgahs.index(k) for k in truth])
    print(f"{len(records)} recordings, {len(set(artists))} artists, "
          f"{design.n_modes} modal templates folding into {len(dastgahs)} dastgahs")

    reference = np.empty(len(records), dtype=int)
    for i, r in enumerate(records):
        result = classify(
            r["h"], templates, transitions=r["B"],
            tonic_prior=np.exp(r["log_prior"]) if r["log_prior"] is not None else None,
            config=DEFAULT_CONFIG,
        )
        reference[i] = dastgahs.index(result.ranked_dastgahs(DASTGAHS_WITH_AUDIO)[0][0])
    print(f"theory only: {100 * (reference == target).mean():.1f}%\n")

    splits = list(GroupKFold(n_splits=args.folds).split(design.rotated, target, artists))

    def evaluate(level: str, penalty: float) -> tuple[float, float]:
        predicted = np.empty(len(records), dtype=int)
        for train, test in splits:
            parameters = fit(
                design.select(train), [truth[i] for i in train],
                level=level, penalty=penalty, config=DEFAULT_CONFIG,
                balanced=args.balanced,
            )
            probabilities, _ = dastgah_probabilities(parameters, design.select(test))
            predicted[test] = probabilities.argmax(axis=1)
        accuracy = 100 * (predicted == target).mean()
        macro = 100 * np.mean(
            [(predicted[target == d] == d).mean() for d in range(len(dastgahs))
             if (target == d).any()]
        )
        return accuracy, macro

    header = "  ".join(f"{p:>6g}" for p in PENALTIES)
    print(f"artist-grouped {args.folds}-fold accuracy by shrinkage toward theory")
    print(f"{'level':<10} {header}")
    best = (None, None, -1.0)
    for level in LEVELS:
        row = []
        for penalty in PENALTIES:
            accuracy, _ = evaluate(level, penalty)
            row.append(accuracy)
            if accuracy > best[2]:
                best = (level, penalty, accuracy)
        print(f"{level:<10} " + "  ".join(f"{v:6.1f}" for v in row), flush=True)

    level, penalty, accuracy = best
    print(f"\nbest: level={level} penalty={penalty:g} at {accuracy:.1f}%")

    counts = collections.Counter(artists)
    testable = sorted(a for a, n in counts.items() if n >= args.min_recordings)
    held = np.isin(artists, testable)
    predicted = np.empty(len(records), dtype=int)
    for artist in testable:
        test = artists == artist
        parameters = fit(
            design.select(~test), [t for t, keep in zip(truth, ~test) if keep],
            level=level, penalty=penalty, config=DEFAULT_CONFIG, balanced=args.balanced,
        )
        probabilities, _ = dastgah_probabilities(parameters, design.select(test))
        predicted[test] = probabilities.argmax(axis=1)

    def rate(mask, pred):
        return 100 * (pred[mask] == target[mask]).mean()

    def macro(mask, pred):
        return 100 * np.mean(
            [(pred[mask & (target == d)] == d).mean() for d in range(len(dastgahs))
             if (mask & (target == d)).any()]
        )

    print(f"\nleave-one-artist-out over {len(testable)} artists, {held.sum()} recordings")
    print(f"  theory only  {rate(held, reference):5.1f}%   macro {macro(held, reference):5.1f}%")
    print(f"  fitted       {rate(held, predicted):5.1f}%   macro {macro(held, predicted):5.1f}%")

    print(f"\n{'dastgah':<12} {'n':>5} {'theory':>8} {'fitted':>8} {'delta':>7}")
    for d, key in enumerate(dastgahs):
        mask = held & (target == d)
        if not mask.any():
            continue
        t, f = rate(mask, reference), rate(mask, predicted)
        print(f"{key:<12} {mask.sum():5d} {t:7.1f}% {f:7.1f}% {f - t:+6.1f}")

    wins = sum(
        rate(held & (artists == a), predicted) > rate(held & (artists == a), reference)
        for a in testable
    )
    print(f"\nimproved on {wins} of {len(testable)} unseen artists")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
