"""Fit the scoring function's own parameters and score it on unseen artists.

    python scripts/train_hybrid.py --cache data/cache/nava.pkl

Leave-one-artist-out over artists with enough recordings to score, at each
level of :data:`dastgah.core.learn.LEVELS`, against the hand-built classifier
on exactly the same recordings.
"""

from __future__ import annotations

import argparse
import collections
import pickle
import time
from pathlib import Path

import numpy as np

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH, _blend_prior, _tonic_prior
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.core.learn import (
    LEVELS,
    build_design,
    dastgah_probabilities,
    fit,
    initial_parameters,
)
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates


def load_records(cache: Path, templates) -> list[dict]:
    raw = pickle.load(cache.open("rb"))
    if isinstance(raw, dict):
        raw = list(raw.values())
    records = []
    for r in raw:
        histogram = np.asarray(r["h"], dtype=float)
        if histogram.sum() <= 0:
            continue
        prior = _blend_prior(
            _tonic_prior(r.get("foruds") or [], r.get("dur"), DEFAULT_CONFIG),
            histogram,
            DEFAULT_CONFIG,
        )
        records.append(
            {
                "h": histogram,
                "B": r.get("B"),
                "log_prior": np.log(np.asarray(prior, dtype=float) + 1e-9)
                if prior is not None
                else None,
                "truth": r["truth"],
                "artist": r.get("artist", "?"),
                "instrument": r.get("instrument", "?"),
            }
        )
    return records


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/nava.pkl"))
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument("--min-recordings", type=int, default=15)
    parser.add_argument("--penalty", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=400)
    parser.add_argument(
        "--balanced", action="store_true",
        help="weight each dastgah equally, denying the fit the option of "
             "writing off a class it cannot separate",
    )
    args = parser.parse_args()

    templates = load_templates(args.templates)
    records = load_records(args.cache, templates)
    space = tuple(sorted(set(r["truth"] for r in records)))
    print(f"{len(records)} recordings, {len({r['artist'] for r in records})} artists, "
          f"{len(space)} dastgahs")

    print("building design matrices...", end="", flush=True)
    started = time.monotonic()
    design = build_design(records, templates, config=DEFAULT_CONFIG)
    print(f" {time.monotonic() - started:.0f}s")

    truth = [r["truth"] for r in records]
    artists = np.array([r["artist"] for r in records])
    instruments = np.array([r["instrument"] for r in records])
    _, dastgahs = dastgah_probabilities(initial_parameters(design, DEFAULT_CONFIG), design)
    target = np.array([dastgahs.index(k) for k in truth])

    # The hand-built classifier, through its own code path, as the reference.
    template_pred = np.empty(len(records), dtype=int)
    for i, r in enumerate(records):
        result = classify(
            r["h"], templates, transitions=r["B"],
            tonic_prior=np.exp(r["log_prior"]) if r["log_prior"] is not None else None,
            config=DEFAULT_CONFIG,
        )
        template_pred[i] = dastgahs.index(result.ranked_dastgahs(space)[0][0])

    counts = collections.Counter(artists)
    testable = sorted(a for a, n in counts.items() if n >= args.min_recordings)
    held = np.isin(artists, testable)
    print(f"held out: {held.sum()} recordings by {len(testable)} artists\n")

    def rate(mask, pred):
        return 100.0 * (pred[mask] == target[mask]).mean()

    def macro(mask, pred):
        return 100.0 * np.mean(
            [(pred[mask & (target == d)] == d).mean()
             for d in range(len(dastgahs))
             if (mask & (target == d)).any()]
        )

    print(f"{'level':<12} {'params':>7} {'accuracy':>9} {'macro':>7}")
    print(f"{'theory only':<12} {0:>7} {rate(held, template_pred):8.1f}% "
          f"{macro(held, template_pred):6.1f}%")

    predictions = {"theory only": template_pred}
    for level in LEVELS:
        pred = np.full(len(records), -1)
        n_params = 0
        for artist in testable:
            test = artists == artist
            parameters = fit(
                _subset(design, ~test), [t for t, keep in zip(truth, ~test) if keep],
                level=level, penalty=args.penalty, config=DEFAULT_CONFIG,
                max_iterations=args.max_iterations, balanced=args.balanced,
            )
            probabilities, _ = dastgah_probabilities(parameters, _subset(design, test))
            pred[test] = probabilities.argmax(axis=1)
            n_params = _count(level, design.n_modes)
        predictions[level] = pred
        print(f"{level:<12} {n_params:>7} {rate(held, pred):8.1f}% {macro(held, pred):6.1f}%")

    best = max(LEVELS, key=lambda lv: rate(held, predictions[lv]))
    print(f"\nby dastgah ({best} vs theory)")
    print(f"{'dastgah':<14} {'n':>5} {'theory':>8} {'fitted':>8} {'delta':>7}")
    for d, key in enumerate(dastgahs):
        mask = held & (target == d)
        if not mask.any():
            continue
        t, f = rate(mask, template_pred), rate(mask, predictions[best])
        print(f"{key:<14} {mask.sum():5d} {t:7.1f}% {f:7.1f}% {f - t:+6.1f}")

    print(f"\nby instrument ({best} vs theory)")
    for code in sorted(set(instruments)):
        mask = held & (instruments == code)
        if not mask.any():
            continue
        t, f = rate(mask, template_pred), rate(mask, predictions[best])
        print(f"instrument {code:<4} {mask.sum():5d} {t:7.1f}% {f:7.1f}% {f - t:+6.1f}")

    print("\nprediction counts, true vs predicted (over-prediction is the bias term's job)")
    print(f"{'dastgah':<14} {'true':>6} {'theory':>7} {'fitted':>7}")
    for d, key in enumerate(dastgahs):
        print(f"{key:<14} {(held & (target == d)).sum():6d} "
              f"{(held & (template_pred == d)).sum():7d} "
              f"{(held & (predictions[best] == d)).sum():7d}")

    # What a fit on everything learned, for reading against theory.
    print(f"\nfitted on all {len(records)} recordings at level={best}")
    whole = fit(design, truth, level=best, penalty=args.penalty,
                config=DEFAULT_CONFIG, max_iterations=args.max_iterations,
                balanced=args.balanced)
    print(f"  alpha {whole.alpha:.3f}  transition {whole.transition_weight:.3f}  "
          f"prior {whole.prior_weight:.3f}")
    print(f"  {'mode':<18} {'sharpen':>8} {'bias':>8}")
    for index, key in enumerate(design.modes):
        print(f"  {key:<18} {whole.sharpen[index]:8.2f} {whole.bias[index]:8.2f}")
    return 0


def _count(level: str, n_modes: int) -> int:
    n = 4
    if level in ("bias", "sharpen", "profiles"):
        n += n_modes
    if level in ("sharpen", "profiles"):
        n += n_modes
    if level == "profiles":
        n += n_modes * 24
    return n


def _subset(design, mask):
    from dastgah.core.learn import Design

    return Design(
        rotated=design.rotated[mask],
        transition=design.transition[mask],
        log_prior=design.log_prior[mask],
        log_theory=design.log_theory,
        modes=design.modes,
        mothers=design.mothers,
    )


if __name__ == "__main__":
    raise SystemExit(main())
