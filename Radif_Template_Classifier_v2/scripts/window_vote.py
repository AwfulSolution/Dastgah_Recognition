"""Score a recording in windows and combine, instead of pooling it into one histogram.

Recording length is the largest effect in this project: the same classifier and
the same parameters score 74.9% on Nava's 75-second median and 89.0% on its
recordings over three minutes. Pooling is the suspect. A performance moves
through gushehs, and a single histogram over the whole of one mixes every
region it visited into one blur, so the longer the recording the more it mixes
-- except the measurement says longer is *better*, which means blurring is not
what limits the short ones.

That leaves two separable questions, and this script exists to tell them apart:

1. Does combining window-level decisions beat pooling, at any length? If a
   performance's modal regions are better read one at a time, this should win on
   long recordings, where there is most to blur.
2. Does it rescue short recordings? A 75-second excerpt holds three or four
   windows; splitting it adds no evidence, so if short recordings are limited by
   how much music they contain rather than by pooling, nothing here will help.

Nothing is trained or fitted. These are decision rules over the existing
classifier, which is why they are worth trying after everything learned has
failed to transfer.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

SPACE = list(DASTGAHS_WITH_AUDIO)

#: Windows holding less pitch content than this share of the fullest window are
#: dropped: a near-silent window votes on almost nothing.
MIN_WINDOW_SHARE = 0.05


def window_probabilities(record, templates, config=DEFAULT_CONFIG):
    """(W, 6) probability per dastgah for each window of one recording.

    Each window is scored with the *recording's* tonic prior rather than its
    own. The tonic is a property of the performance and does not move between
    windows, while the mode may; estimating it afresh from twenty seconds would
    throw away the one thing the whole recording establishes well.

    The transition term is omitted because per-window bigrams are not cached,
    so the pooled comparison in :func:`baselines` drops it too and the two stay
    like for like.
    """
    grid = record.get("W")
    if grid is None:
        return None
    grid = np.atleast_2d(np.asarray(grid, dtype=float))
    weight = grid.sum(axis=1)
    if weight.max() <= 0:
        return None
    keep = weight >= MIN_WINDOW_SHARE * weight.max()
    if not keep.any():
        return None

    prior = np.exp(record["log_prior"]) if record.get("log_prior") is not None else None
    out = []
    for window in grid[keep]:
        result = classify(window, templates, transitions=None, tonic_prior=prior,
                          config=config)
        ranked = dict(result.ranked_dastgahs(SPACE))
        out.append([ranked.get(k, 0.0) for k in SPACE])
    return np.array(out), weight[keep]


def combine(probabilities, weights, how):
    """Reduce per-window probabilities to one answer."""
    if how == "mean":
        return probabilities.mean(axis=0)
    if how == "weighted mean":
        return (probabilities * weights[:, None]).sum(axis=0) / weights.sum()
    if how == "sum of logs":
        # Treats windows as independent observations, which sharpens the result
        # and over-counts evidence when they are not.
        return np.log(probabilities + 1e-12).sum(axis=0)
    if how == "majority":
        votes = collections.Counter(probabilities.argmax(axis=1))
        out = np.zeros(probabilities.shape[1])
        for index, count in votes.items():
            out[index] = count
        return out
    if how == "max":
        return probabilities.max(axis=0)
    raise ValueError(how)


RULES = ("mean", "weighted mean", "sum of logs", "majority", "max")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--caches", type=Path, nargs="+", required=True)
    parser.add_argument("--buckets", type=float, nargs="*",
                        default=[60.0, 120.0, 180.0])
    args = parser.parse_args()

    import sys

    sys.path.insert(0, "scripts")
    from train_hybrid import load_records
    from transfer_test import in_scope

    templates = load_templates(DEFAULT_TEMPLATE_PATH)
    edges = [0.0, *args.buckets, float("inf")]

    for cache in args.caches:
        records = in_scope(load_records(cache, templates))
        records = [r for r in records if r.get("W") is not None]
        if not records:
            print(f"{cache.name}: no cached windows")
            continue

        truth = np.array([r["truth"] for r in records])
        durations = np.array([r.get("dur") or 0.0 for r in records])

        # Pooled baselines: the shipped path, and the same minus transitions so
        # the windowed rules are compared against their own settings.
        shipped, pooled = [], []
        windowed = {rule: [] for rule in RULES}
        usable = []
        for record in records:
            prior = (np.exp(record["log_prior"])
                     if record.get("log_prior") is not None else None)
            full = classify(record["h"], templates, transitions=record["B"],
                            tonic_prior=prior, config=DEFAULT_CONFIG)
            shipped.append(full.ranked_dastgahs(SPACE)[0][0])
            flat = classify(record["h"], templates, transitions=None,
                            tonic_prior=prior, config=DEFAULT_CONFIG)
            pooled.append(flat.ranked_dastgahs(SPACE)[0][0])

            result = window_probabilities(record, templates)
            usable.append(result is not None)
            for rule in RULES:
                if result is None:
                    windowed[rule].append(flat.ranked_dastgahs(SPACE)[0][0])
                else:
                    probabilities, weights = result
                    windowed[rule].append(SPACE[int(combine(probabilities, weights, rule).argmax())])

        shipped = np.array(shipped); pooled = np.array(pooled)
        usable = np.array(usable)
        n_windows = [len(window_probabilities(r, templates)[0])
                     for r in records if window_probabilities(r, templates) is not None]

        print(f"\n=== {cache.name}: {len(records)} recordings, "
              f"{usable.sum()} with usable windows, median {np.median(n_windows):.0f} "
              f"windows each ===")
        print(f"{'rule':<16} {'overall':>8}  " +
              "  ".join(f"{_label(edges, i):>12}" for i in range(len(edges) - 1)))

        def row(name, predicted):
            cells = []
            for i in range(len(edges) - 1):
                mask = (durations >= edges[i]) & (durations < edges[i + 1])
                cells.append(f"{100 * (predicted[mask] == truth[mask]).mean():.1f}% ({mask.sum()})"
                             if mask.any() else "-")
            print(f"{name:<16} {100 * (predicted == truth).mean():7.1f}%  " +
                  "  ".join(f"{c:>12}" for c in cells))

        row("pooled (shipped)", shipped)
        row("pooled, no trans", pooled)
        for rule in RULES:
            row(rule, np.array(windowed[rule]))
    return 0


def _label(edges, i):
    low, high = edges[i], edges[i + 1]
    return f"<{low + (high - low) if False else high:.0f}s" if high != float("inf") else f">{low:.0f}s"


if __name__ == "__main__":
    raise SystemExit(main())
