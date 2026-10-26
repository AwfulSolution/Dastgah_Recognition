"""Fit the pair models the deployed cascade consults, and write their coefficients.

    python scripts/train_pair_models.py --cache data/cache/nava.pkl

Writes `dastgah/data/pair_models.json`: for each confusable pair, a logistic
regression's coefficients and intercept, plus what it was fitted on. Serving
reads those with numpy, so scikit-learn is a training dependency only.

The features come from `dastgah.core.cascade`, not from this file, so what is
fitted here and what runs in production cannot drift.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.cascade import DEFAULT_PAIR_MODEL_PATH, pair_features
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

SPACE = list(DASTGAHS_WITH_AUDIO)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/nava.pkl"))
    parser.add_argument("--out", type=Path, default=DEFAULT_PAIR_MODEL_PATH)
    parser.add_argument("--features", default="all")
    parser.add_argument("--min-errors", type=int, default=25)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, "scripts")
    from train_hybrid import load_records
    from transfer_test import in_scope

    templates = load_templates(DEFAULT_TEMPLATE_PATH)
    records = in_scope(load_records(args.cache, templates))
    print(f"{len(records)} recordings from {args.cache.name}")

    rows = []
    for record in records:
        prior = (np.exp(record["log_prior"])
                 if record.get("log_prior") is not None else None)
        result = classify(record["h"], templates, transitions=record["B"],
                          tonic_prior=prior, config=DEFAULT_CONFIG)
        ranked = result.ranked_dastgahs(SPACE)
        tonics, modes = {}, {}
        for key in SPACE:
            try:
                candidate = result.best_for_dastgah(key)
            except KeyError:
                continue
            tonics[key], modes[key] = candidate.tonic_pc, candidate.key
        rows.append((dict(ranked), tonics, modes, [k for k, _ in ranked]))

    truth = np.array([r["truth"] for r in records])
    artists = np.array([r["artist"] for r in records])
    predicted = np.array([row[3][0] for row in rows])

    confusion = collections.Counter(
        frozenset((t, p)) for t, p in zip(truth, predicted, strict=True) if t != p
    )
    pairs = [tuple(sorted(s)) for s, n in confusion.items()
             if len(s) == 2 and n >= args.min_errors]
    print(f"{len(pairs)} pairs confused at least {args.min_errors} times\n")

    models = []
    for pair in pairs:
        mask = np.isin(truth, pair)
        X = np.array([
            pair_features(records[i]["h"], rows[i][0], rows[i][1], rows[i][2],
                          pair, templates, which=args.features,
                          foruds=records[i].get("foruds"),
                          approaches=records[i].get("approaches"))
            for i in np.flatnonzero(mask)
        ])
        y = (truth[mask] == pair[0]).astype(int)
        groups = artists[mask]

        # An honest accuracy for the artefact, grouped so no artist is on both
        # sides, reported next to the model rather than in a separate note.
        scores = []
        for train, test in GroupKFold(n_splits=args.folds).split(X, y, groups):
            fold = LogisticRegression(max_iter=4000).fit(X[train], y[train])
            scores.append((fold.predict(X[test]) == y[test]).mean())
        held_out = float(np.mean(scores))

        model = LogisticRegression(max_iter=4000).fit(X, y)
        models.append({
            "pair": list(pair),
            "features": args.features,
            "coefficients": [float(c) for c in model.coef_[0]],
            "intercept": float(model.intercept_[0]),
            "trained_on": args.cache.stem,
            "n_recordings": int(mask.sum()),
            "grouped_cv_accuracy": round(held_out, 4),
        })
        print(f"{pair[0]:>10} / {pair[1]:<12} {int(mask.sum()):4d} recordings, "
              f"artist-grouped accuracy {100*held_out:.1f}%")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "format": "dastgah-pair-models/1",
        "note": (
            "Fitted on one corpus. These gain 1.8 points on it and lose on every "
            "other corpus measured: -3.2 KDC, -6.9 IRMA, -5.5/-2.6/-3.8 "
            "leave-one-corpus-out. Carried as a second opinion, never as the answer."
        ),
        "models": models,
    }, indent=1))
    print(f"\n{len(models)} models -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
