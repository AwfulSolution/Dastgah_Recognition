"""Sweep the tonic-prior weights, which is where the measured headroom is.

Scoring every class at the tonic the true class prefers reaches 91.0% against
74.9% as shipped, so about 16 points are tonic placement rather than mode
recognition. The forud is what fixes a tonic, and three settings govern how much
the classifier listens to it: how much cadence evidence is mixed with sounding
time (``forud_prior_weight``), how fast older cadences are discounted
(``forud_recency_halflife``) and how much the blended prior counts against pitch
content (``tonic_prior_weight``).

The score decomposes into terms that no weight changes, so they are computed
once per corpus and the sweep itself is arithmetic.

    python scripts/sweep_tonic.py
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.classify import DEFAULT_CONFIG
from dastgah.core.forud import tonic_prior
from dastgah.core.learn import build_design, initial_parameters
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

CORPORA = {
    "nava (1568)": "data/cache/nava.pkl",
    "nava >3min (218)": "data/cache/nava_long.pkl",
    "kdc (189)": "data/cache/kdc.pkl",
    "irma (130)": "data/cache/irma.pkl",
    "shajarian (16)": "data/cache/shajarian.pkl",
}
FORUD_WEIGHTS = (0.0, 0.3, 0.5, 0.7, 0.9, 1.0)
PRIOR_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0)
HALFLIVES = (30.0, 60.0, 120.0, None)


class Corpus:
    """Everything a weight sweep needs, with the fixed terms already summed."""

    def __init__(self, records, templates):
        self.design = build_design(
            records, templates, config=DEFAULT_CONFIG,
            answer_space=DASTGAHS_WITH_AUDIO,
        )
        theory = initial_parameters(self.design, DEFAULT_CONFIG)
        self.alpha = theory.alpha
        self.transition_weight = theory.transition_weight
        self.profile = np.einsum(
            "itd,md->itm", self.design.rotated,
            theory.log_profiles(self.design.log_theory),
        )
        # prior[t] reads the histogram at pitch class t, so sounding time is the
        # histogram itself rather than anything rotated.
        self.sounding = np.array(
            [np.asarray(r["h"], float) / np.asarray(r["h"], float).sum()
             for r in records]
        )
        self.records = records
        self.dastgahs = sorted(set(self.design.mothers))
        self.fold = np.zeros((self.design.n_modes, len(self.dastgahs)))
        for index, mother in enumerate(self.design.mothers):
            self.fold[index, self.dastgahs.index(mother)] = 1.0
        self.target = np.array([self.dastgahs.index(r["truth"]) for r in records])

    def cadence(self, halflife: float | None) -> np.ndarray:
        """(n, 24) cadence prior, falling back to sounding time where absent."""
        out = np.empty_like(self.sounding)
        for i, record in enumerate(self.records):
            prior = tonic_prior(
                record.get("foruds") or [],
                total_duration=record.get("dur"),
                recency_halflife=halflife,
            )
            out[i] = self.sounding[i] if prior is None else prior
        return out

    def accuracy(self, cadence: np.ndarray, forud_weight: float, prior_weight: float) -> float:
        blended = forud_weight * cadence + (1.0 - forud_weight) * self.sounding
        scores = (
            self.alpha * self.profile
            + self.transition_weight * self.design.transition
            + (prior_weight / DEFAULT_CONFIG.temperature)
            * np.log(blended + 1e-9)[:, :, None]
        )
        flat = scores.reshape(scores.shape[0], -1)
        flat = flat - flat.max(axis=1, keepdims=True)
        weights = np.exp(flat).reshape(scores.shape)
        folded = np.einsum("itm,md->id", weights, self.fold)
        return 100.0 * (folded.argmax(axis=1) == self.target).mean()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    args = parser.parse_args()

    import sys

    sys.path.insert(0, "scripts")
    from train_hybrid import load_records
    from transfer_test import in_scope

    templates = load_templates(args.templates)
    corpora = {}
    for label, path in CORPORA.items():
        if not Path(path).exists():
            continue
        records = in_scope(load_records(Path(path), templates))
        if records:
            corpora[label] = Corpus(records, templates)
            print(f"loaded {label}", flush=True)

    shipped = (
        DEFAULT_CONFIG.forud_prior_weight,
        DEFAULT_CONFIG.tonic_prior_weight,
        DEFAULT_CONFIG.forud_recency_halflife,
    )
    print(f"\nshipped: forud_prior_weight={shipped[0]} "
          f"tonic_prior_weight={shipped[1]} halflife={shipped[2]}")
    baseline = {}
    for label, corpus in corpora.items():
        baseline[label] = corpus.accuracy(
            corpus.cadence(shipped[2]), shipped[0], shipped[1]
        )
        print(f"  {label:<20} {baseline[label]:5.1f}%")

    print("\ntonic_prior_weight, holding the shipped blend and halflife")
    print(f"{'corpus':<20} " + "  ".join(f"{w:>5g}" for w in PRIOR_WEIGHTS))
    for label, corpus in corpora.items():
        cadence = corpus.cadence(shipped[2])
        row = [corpus.accuracy(cadence, shipped[0], w) for w in PRIOR_WEIGHTS]
        print(f"{label:<20} " + "  ".join(f"{v:5.1f}" for v in row), flush=True)

    print("\nforud_prior_weight, holding tonic_prior_weight at the shipped 0.25")
    print(f"{'corpus':<20} " + "  ".join(f"{w:>5g}" for w in FORUD_WEIGHTS))
    for label, corpus in corpora.items():
        cadence = corpus.cadence(shipped[2])
        row = [corpus.accuracy(cadence, w, shipped[1]) for w in FORUD_WEIGHTS]
        print(f"{label:<20} " + "  ".join(f"{v:5.1f}" for v in row), flush=True)

    print("\njoint sweep, pooled by corpus size and averaged over corpora")
    sizes = {label: corpus.design.n for label, corpus in corpora.items()}
    total = sum(sizes.values())
    best = []
    for halflife in HALFLIVES:
        cadences = {k: c.cadence(halflife) for k, c in corpora.items()}
        for forud_weight in FORUD_WEIGHTS:
            for prior_weight in PRIOR_WEIGHTS:
                per = {
                    label: corpus.accuracy(cadences[label], forud_weight, prior_weight)
                    for label, corpus in corpora.items()
                }
                pooled = sum(per[k] * sizes[k] for k in per) / total
                mean = float(np.mean(list(per.values())))
                best.append((pooled, mean, forud_weight, prior_weight, halflife, per))

    print(f"{'rank':<5} {'pooled':>7} {'mean':>7} {'forud':>6} {'prior':>6} {'halflife':>9}")
    for rank, item in enumerate(sorted(best, key=lambda x: -x[0])[:8], start=1):
        pooled, mean, fw, pw, hl, _ = item
        print(f"{rank:<5} {pooled:6.1f}% {mean:6.1f}% {fw:6g} {pw:6g} "
              f"{'none' if hl is None else f'{hl:g}':>9}")

    top = max(best, key=lambda x: x[0])
    print(f"\nbest pooled: forud_prior_weight={top[2]:g} tonic_prior_weight={top[3]:g} "
          f"halflife={'none' if top[4] is None else top[4]:g}")
    print(f"{'corpus':<20} {'shipped':>8} {'swept':>8} {'delta':>7}")
    for label in corpora:
        print(f"{label:<20} {baseline[label]:7.1f}% {top[5][label]:7.1f}% "
              f"{top[5][label] - baseline[label]:+6.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
