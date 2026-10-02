"""Fit on one corpus, test on another. The only evidence that justifies shipping.

    python scripts/transfer_test.py --fit data/cache/nava.pkl --test data/cache/kdc.pkl

Leave-one-artist-out within a corpus says a fitted parameter generalises across
that corpus's performers. It cannot say the parameter generalises across
recording conditions, instrument balance or editorial choices, and this project
has already been fooled once by exactly that gap: a stacked model gained 2-4
points under leave-one-performer-out and transferred nothing.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import numpy as np

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.core.learn import (
    build_design,
    dastgah_probabilities,
    fit,
    initial_parameters,
)
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates
from dastgah.theory import MODAL_CLASSES_BY_KEY

from train_hybrid import load_records  # noqa: E402  - sibling script


def in_scope(records: "list[dict]", *, avazes: bool = True) -> "list[dict]":
    """Fold each avaz onto its mother, then keep the answerable dastgahs.

    ``avazes=False`` additionally drops recordings whose own label is an avaz,
    leaving only ones labelled with a dastgah directly. That makes a test corpus
    comparable to a training corpus that contains no avaz labels at all.
    """
    folded = []
    for r in records:
        modal = MODAL_CLASSES_BY_KEY.get(r["truth"])
        mother = (modal.parent or r["truth"]) if modal else r["truth"]
        if mother not in DASTGAHS_WITH_AUDIO:
            continue
        if not avazes and mother != r["truth"]:
            continue
        folded.append({**r, "truth": mother})
    return folded


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit", type=Path, required=True)
    parser.add_argument("--test", type=Path, nargs="+", required=True)
    parser.add_argument("--level", default="bias")
    parser.add_argument("--penalty", type=float, default=0.0)
    parser.add_argument(
        "--no-avaz", action="store_true",
        help=(
            "drop avaz-labelled recordings from the test corpus, matching a "
            "training corpus that has none"
        ),
    )
    parser.add_argument(
        "--balanced", action="store_true",
        help=(
            "weight each dastgah equally when fitting. The bias term is a class "
            "prior, so an unbalanced fit learns which classes the training "
            "corpus happens to contain -- which is exactly what cannot transfer"
        ),
    )
    args = parser.parse_args()

    templates = load_templates(DEFAULT_TEMPLATE_PATH)

    train = in_scope(load_records(args.fit, templates))
    train_design = build_design(
        train, templates, config=DEFAULT_CONFIG, answer_space=DASTGAHS_WITH_AUDIO
    )
    _, dastgahs = dastgah_probabilities(initial_parameters(train_design), train_design)
    print(f"fitting on {args.fit.name}: {len(train)} recordings, "
          f"{len({r['artist'] for r in train})} artists")
    parameters = fit(
        train_design, [r["truth"] for r in train],
        level=args.level, penalty=args.penalty, config=DEFAULT_CONFIG,
        balanced=args.balanced,
    )
    print(f"  level={args.level} penalty={args.penalty:g} balanced={args.balanced}")
    print(f"  {'mode':<18} {'bias':>7}")
    order = np.argsort(parameters.bias)
    for index in order:
        print(f"  {train_design.modes[index]:<18} {parameters.bias[index]:+7.2f}")

    for cache in args.test:
        test = in_scope(load_records(cache, templates), avazes=not args.no_avaz)
        if not test:
            print(f"\n{cache.name}: nothing in scope")
            continue
        design = build_design(
            test, templates, config=DEFAULT_CONFIG, answer_space=DASTGAHS_WITH_AUDIO
        )
        truth = np.array([dastgahs.index(r["truth"]) for r in test])

        reference = np.empty(len(test), dtype=int)
        for i, r in enumerate(test):
            result = classify(
                r["h"], templates, transitions=r["B"],
                tonic_prior=np.exp(r["log_prior"]) if r["log_prior"] is not None else None,
                config=DEFAULT_CONFIG,
            )
            reference[i] = dastgahs.index(
                result.ranked_dastgahs(DASTGAHS_WITH_AUDIO)[0][0]
            )
        probabilities, _ = dastgah_probabilities(parameters, design)
        predicted = probabilities.argmax(axis=1)

        def rate(pred, mask=None):
            sel = slice(None) if mask is None else mask
            return 100 * (pred[sel] == truth[sel]).mean()

        def macro(pred):
            return 100 * np.mean(
                [(pred[truth == d] == d).mean() for d in range(len(dastgahs))
                 if (truth == d).any()]
            )

        present = sorted({dastgahs[d] for d in truth})
        print(f"\n{cache.name}: {len(test)} recordings, {len(present)} dastgahs present")
        print(f"  theory only  {rate(reference):5.1f}%   macro {macro(reference):5.1f}%")
        print(f"  transferred  {rate(predicted):5.1f}%   macro {macro(predicted):5.1f}%"
              f"   ({rate(predicted) - rate(reference):+.1f})")

        print(f"  {'dastgah':<12} {'n':>5} {'theory':>8} {'fitted':>8} {'delta':>7}")
        for d, key in enumerate(dastgahs):
            mask = truth == d
            if not mask.any():
                continue
            t, f = rate(reference, mask), rate(predicted, mask)
            print(f"  {key:<12} {mask.sum():5d} {t:7.1f}% {f:7.1f}% {f - t:+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
