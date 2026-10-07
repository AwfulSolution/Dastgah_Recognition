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

from dastgah.core.analyze import (
    DEFAULT_GUSHEH_PATH,
    DEFAULT_TEMPLATE_PATH,
    _blend_prior,
    _tonic_prior,
)
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.core.learn import (
    LEVELS,
    build_design,
    dastgah_probabilities,
    fit,
    initial_parameters,
)
from dastgah.core.seyr import progression_scores
from dastgah.radif.gusheh import load_gusheh_templates
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
                "W": r.get("W"),
                # Carried so a caller can rebuild the prior under other weights;
                # without these a sweep of the cadence weights silently does
                # nothing, because tonic_prior([]) returns None and the blend
                # falls back to sounding time for every recording.
                "foruds": r.get("foruds") or [],
                "dur": r.get("dur"),
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
        "--no-seyr", action="store_true",
        help="leave the progression term out, for a like-for-like comparison",
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help="weight each dastgah equally, denying the fit the option of "
             "writing off a class it cannot separate",
    )
    args = parser.parse_args()

    templates = load_templates(args.templates)
    records = load_records(args.cache, templates)

    # Rast-Panjgah is out of the answer space by design: its material is almost
    # entirely borrowed, so it is excluded rather than scored. See the notes.
    records = [r for r in records if r["truth"] in DASTGAHS_WITH_AUDIO]
    space = tuple(sorted(set(r["truth"] for r in records)))
    gushehs = None if args.no_seyr else load_gusheh_templates(DEFAULT_GUSHEH_PATH)
    windowed = sum(1 for r in records if r.get("W") is not None)
    print(f"{windowed} of {len(records)} recordings carry time windows")
    print(f"{len(records)} recordings, {len({r['artist'] for r in records})} artists, "
          f"{len(space)} dastgahs: {', '.join(space)}")

    print("building design matrices...", end="", flush=True)
    started = time.monotonic()
    design = build_design(
        records, templates, config=DEFAULT_CONFIG, gusheh_templates=gushehs,
        answer_space=DASTGAHS_WITH_AUDIO,
    )
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
        template_pred[i] = dastgahs.index(
            result.ranked_dastgahs(DASTGAHS_WITH_AUDIO)[0][0]
        )

    # Progression on its own, with no pitch-content term at all: the most
    # direct test of whether order carries the dastgah.
    if gushehs is not None and np.any(design.progression != 0):
        fold, _ = __import__(
            "dastgah.core.learn", fromlist=["_mother_matrix"]
        )._mother_matrix(design)
        best = design.progression.reshape(len(records), -1).argmax(axis=1)
        mode_of = best % design.n_modes
        alone = np.array([dastgahs.index(design.mothers[m]) for m in mode_of])
        print(f"progression alone (no pitch term): "
              f"{100 * (alone == target).mean():.1f}%  "
              f"chance {100 / len(dastgahs):.1f}%")

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
                design.select(~test), [t for t, keep in zip(truth, ~test) if keep],
                level=level, penalty=args.penalty, config=DEFAULT_CONFIG,
                max_iterations=args.max_iterations, balanced=args.balanced,
            )
            probabilities, _ = dastgah_probabilities(parameters, design.select(test))
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
          f"prior {whole.prior_weight:.3f}  progression {whole.progression_weight:.3f}")
    print(f"  {'mode':<18} {'sharpen':>8} {'bias':>8}")
    for index, key in enumerate(design.modes):
        print(f"  {key:<18} {whole.sharpen[index]:8.2f} {whole.bias[index]:8.2f}")
    return 0


def _count(level: str, n_modes: int) -> int:
    n = 5
    if level in ("bias", "sharpen", "profiles"):
        n += n_modes
    if level in ("sharpen", "profiles"):
        n += n_modes
    if level == "profiles":
        n += n_modes * 24
    return n


if __name__ == "__main__":
    raise SystemExit(main())
