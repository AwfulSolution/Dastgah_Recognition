"""A second stage for the pairs the first stage confuses.

Stage one ranks all six dastgahs. When its top two are a pair known to be
confusable, a classifier trained on *only that pair* decides between them.

The gate is what makes this safe, and it is worth stating precisely. The second
stage fires only when ``{top1, top2} == {A, B}``, so stage one's answer is
already A or B. A recording whose truth is neither was already wrong and stays
wrong; a recording whose truth is A or B can be fixed or broken. **The cascade's
effect is therefore confined entirely to the pair.** That is the property every
earlier attempt lacked: adding a term to the shared score, or refitting one
template, changes what every class competes against, and probability given to
Nava came out of Shur and Homayun.

    python scripts/cascade.py --cache data/cache/nava.pkl
"""

from __future__ import annotations

import argparse
import collections
import itertools
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates

SPACE = list(DASTGAHS_WITH_AUDIO)


def stage_one(records, templates):
    """Rank the six dastgahs for every recording, keeping each one's tonic."""
    out = []
    for r in records:
        result = classify(
            r["h"], templates, transitions=r["B"],
            tonic_prior=np.exp(r["log_prior"]) if r["log_prior"] is not None else None,
            config=DEFAULT_CONFIG,
        )
        ranked = result.ranked_dastgahs(SPACE)
        probabilities = dict(ranked)
        tonics, modes = {}, {}
        for key in SPACE:
            try:
                candidate = result.best_for_dastgah(key)
            except KeyError:
                continue
            tonics[key] = candidate.tonic_pc
            modes[key] = candidate.key
        out.append({
            "order": [k for k, _ in ranked],
            "probabilities": probabilities,
            "tonics": tonics,
            "modes": modes,
        })
    return out


#: Which features a pair model may use.
#:
#: ``all`` includes the observation re-indexed against each hypothesis's tonic,
#: 48 of the 53 columns. Those are raw pitch-class bins, and every model built
#: on them in this project has failed to transfer between corpora; ``minimal``
#: exists to test whether the cascade's gains survive without them.
#: ``cadence`` is the one set here that reads no tonic-relative pitch profile.
#: Every model in this project built on those 24 bins has failed to transfer
#: between corpora, and the representation itself caps out near 70-75%. A
#: cadence approach is anchored on the note *resolved onto* rather than on the
#: tonic, which makes it asymmetric under rotation -- the property a pitch
#: profile lacks, and the reason Shur and Nava are indistinguishable to one.
FEATURE_SETS = ("all", "minimal", "cadence")


def gate_mask(stages, pair, margin):
    """Which recordings the second stage is allowed to decide.

    Two conditions. The pair must hold the top two places, which is what
    confines the cascade's effect to that pair: stage one has already answered A
    or B, so a recording whose truth is a third dastgah was wrong either way.
    And stage one must be *undecided* between them by at most ``margin``, which
    keeps the second stage away from recordings the first already reads
    confidently.
    """
    wanted = frozenset(pair)
    out = []
    for stage in stages:
        order = stage["order"]
        if frozenset(order[:2]) != wanted:
            out.append(False)
            continue
        probabilities = stage["probabilities"]
        separation = probabilities.get(order[0], 0.0) - probabilities.get(order[1], 0.0)
        out.append(separation <= margin)
    return np.array(out)


def pair_features(record, stage, pair, templates, which="all"):
    """Features for choosing between two (mode, tonic) hypotheses.

    Each hypothesis supplies the observation re-indexed against *its own* tonic,
    so the model sees the same pitch content twice, read two ways. That is the
    whole question for a rotation-confusable pair.
    """
    first, second = pair
    histogram = np.asarray(record["h"], dtype=float)
    histogram = histogram / max(histogram.sum(), 1e-9)

    blocks = []
    if which == "all":
        for key in (first, second):
            tonic = stage["tonics"].get(key, 0)
            blocks.append(np.roll(histogram, -tonic))

    if which == "cadence":
        # How cadences resolving on each hypothesis's tonic were approached,
        # plus how much cadential weight that tonic carries at all. No bin of
        # this is the tonic-relative pitch profile.
        foruds = record.get("foruds") or []
        approaches = record.get("approaches") or []
        for key in (first, second):
            tonic = stage["tonics"].get(key, 0)
            shape, weight = np.zeros(24), 0.0
            for forud, approach in zip(foruds, approaches):
                if approach is None or forud.resolution_pc != tonic:
                    continue
                shape += forud.strength * np.asarray(approach, dtype=float)
                weight += forud.strength
            blocks.append(shape / weight if weight > 0 else shape)
            blocks.append(np.array([weight, float(weight > 0)]))
        return np.concatenate([*blocks, _cadence_fit(record, stage, pair, templates)])

    probabilities = [
        np.log(stage["probabilities"].get(first, 0.0) + 1e-9),
        np.log(stage["probabilities"].get(second, 0.0) + 1e-9),
    ]
    probabilities.append(probabilities[0] - probabilities[1])

    return np.concatenate([*blocks, probabilities,
                           _cadence_fit(record, stage, pair, templates)])


def _cadence_fit(record, stage, pair, templates):
    """Agreement between how each hypothesis's cadences were approached and how
    that mode approaches a close in the notated radif."""
    first, second = pair
    cadence = []
    for key in (first, second):
        template = templates.get(stage["modes"].get(key, key))
        profile = template.cadence_array() if template is not None else None
        approaches = [a for a in (record.get("approaches") or []) if a is not None]
        if profile is None or not approaches:
            cadence.append(0.0)
            continue
        norms = np.linalg.norm(profile) * np.array(
            [np.linalg.norm(a) for a in approaches]
        )
        similarity = np.array([float(np.dot(a, profile)) for a in approaches])
        cadence.append(float(np.mean(similarity / np.maximum(norms, 1e-9))))
    return np.array(cadence)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("data/cache/nava.pkl"))
    parser.add_argument("--transfer", type=Path, nargs="*", default=())
    parser.add_argument("--min-errors", type=int, default=25,
                        help="pairs confused fewer times than this are not worth a stage")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--features", choices=FEATURE_SETS, default="all")
    parser.add_argument(
        "--gate-margin", type=float, default=1.0,
        help=(
            "fire the second stage only when stage one is undecided, that is "
            "when p(top1) - p(top2) is at most this. The default of 1.0 fires "
            "whenever the pair holds the top two, confident or not"
        ),
    )
    parser.add_argument(
        "--train-on", choices=("all", "gated"), default="all",
        help=(
            "'all' trains a pair model on every recording of the two dastgahs, "
            "which is what the gate's independence is for. 'gated' trains only "
            "on the ones stage one could not decide -- a biased subset, being "
            "exactly the hard cases, and smaller"
        ),
    )
    parser.add_argument(
        "--leave-one-corpus-out", type=Path, nargs="*", default=(),
        help=(
            "pool these corpora and test each in turn on pair models trained on "
            "the others. The only way to tell a cascade that cannot generalise "
            "from one trained on a single corpus's habits"
        ),
    )
    args = parser.parse_args()

    import sys

    sys.path.insert(0, "scripts")
    from train_hybrid import load_records
    from transfer_test import in_scope

    templates = load_templates(DEFAULT_TEMPLATE_PATH)
    records = in_scope(load_records(args.cache, templates))
    stages = stage_one(records, templates)
    truth = np.array([r["truth"] for r in records])
    artists = np.array([r["artist"] for r in records])
    predicted = np.array([s["order"][0] for s in stages])
    top_two = [frozenset(s["order"][:2]) for s in stages]

    base = float((predicted == truth).mean())
    print(f"{len(records)} recordings, {len(set(artists))} artists, "
          f"features={args.features}, gate_margin={args.gate_margin:g}")
    print(f"stage one alone: {100 * base:.1f}%\n")

    confusion = collections.Counter()
    for t, p in zip(truth, predicted, strict=True):
        if t != p:
            confusion[frozenset((t, p))] += 1

    print("pairs by how often stage one confuses them, either direction")
    print(f"{'pair':<26} {'errors':>7} {'gated':>7} {'share of all errors':>21}")
    total_errors = int((predicted != truth).sum())
    candidates = []
    for pair_set, count in confusion.most_common():
        if len(pair_set) != 2:
            continue
        gated = sum(1 for t in top_two if t == pair_set)
        a, b = sorted(pair_set)
        print(f"{a + ' / ' + b:<26} {count:7d} {gated:7d} {100*count/total_errors:20.1f}%")
        if count >= args.min_errors:
            candidates.append((a, b))

    print(f"\nbuilding a second stage for {len(candidates)} pair(s), "
          f"artist-grouped {args.folds}-fold\n")

    cascaded = predicted.copy()
    models = {}
    for pair in candidates:
        pair_set = frozenset(pair)
        gate = gate_mask(stages, pair, args.gate_margin)
        trainable = np.isin(truth, pair)
        if args.train_on == "gated":
            trainable = trainable & gate

        X = np.array([
            pair_features(records[i], stages[i], pair, templates, args.features)
            for i in range(len(records))
        ])
        y = (truth == pair[0]).astype(int)

        # Held-out decisions for every recording the gate would fire on.
        decided = np.full(len(records), -1)
        groups = artists[trainable]
        splits = GroupKFold(n_splits=args.folds).split(X[trainable], y[trainable], groups)
        index = np.flatnonzero(trainable)
        for train, test in splits:
            model = LogisticRegression(max_iter=4000).fit(
                X[index[train]], y[index[train]]
            )
            decided[index[test]] = model.predict(X[index[test]])
        models[pair] = LogisticRegression(max_iter=4000).fit(X[trainable], y[trainable])

        acts = gate & trainable & (decided >= 0)
        before = predicted[acts] == truth[acts]
        after = np.where(decided[acts] == 1, pair[0], pair[1]) == truth[acts]
        cascaded[acts] = np.where(decided[acts] == 1, pair[0], pair[1])

        print(f"{pair[0]} / {pair[1]}")
        print(f"  trained on {trainable.sum()} recordings of the two "
              f"({args.train_on})")
        print(f"  gate fires on {gate.sum()}, of which {acts.sum()} are one of the pair")
        print(f"  within the gate: {100*before.mean():.1f}% -> {100*after.mean():.1f}%"
              f"  ({int(after.sum() - before.sum()):+d} recordings)")

    print(f"\noverall: {100*base:.1f}% -> {100*(cascaded == truth).mean():.1f}% "
          f"({100*((cascaded == truth).mean() - base):+.1f})")
    print(f"{'dastgah':<12} {'stage one':>10} {'cascaded':>10} {'delta':>7}")
    for key in SPACE:
        mask = truth == key
        a = 100 * (predicted[mask] == truth[mask]).mean()
        b = 100 * (cascaded[mask] == truth[mask]).mean()
        print(f"{key:<12} {a:9.1f}% {b:9.1f}% {b - a:+6.1f}")

    if args.leave_one_corpus_out:
        pooled = []
        for cache in args.leave_one_corpus_out:
            rs = in_scope(load_records(cache, templates))
            if not rs:
                continue
            st = stage_one(rs, templates)
            pooled.append((cache.name, rs, st))

        print(f"\nleave-one-corpus-out over {len(pooled)} corpora, "
              f"features={args.features}")
        print(f"{'held-out corpus':<18} {'n':>5} {'stage one':>10} {'cascaded':>10} "
              f"{'delta':>7} {'fired':>7}")
        for index, (name, rs, st) in enumerate(pooled):
            others = [p for j, p in enumerate(pooled) if j != index]
            held_truth = np.array([r["truth"] for r in rs])
            held_pred = np.array([x["order"][0] for x in st])
            held_top2 = [frozenset(x["order"][:2]) for x in st]
            out = held_pred.copy()
            fired = 0
            for pair in candidates:
                pair_set = frozenset(pair)
                rows, labels = [], []
                for _, o_rs, o_st in others:
                    for i, r in enumerate(o_rs):
                        if r["truth"] in pair:
                            rows.append(pair_features(r, o_st[i], pair, templates,
                                                      args.features))
                            labels.append(int(r["truth"] == pair[0]))
                if len(set(labels)) < 2:
                    continue
                model = LogisticRegression(max_iter=4000).fit(
                    np.array(rows), np.array(labels)
                )
                gate = gate_mask(st, pair, args.gate_margin)
                if not gate.any():
                    continue
                X = np.array([
                    pair_features(rs[i], st[i], pair, templates, args.features)
                    for i in np.flatnonzero(gate)
                ])
                choice = model.predict(X)
                out[gate] = np.where(choice == 1, pair[0], pair[1])
                fired += int(gate.sum())
            a = 100 * (held_pred == held_truth).mean()
            b = 100 * (out == held_truth).mean()
            print(f"{name:<18} {len(rs):5d} {a:9.1f}% {b:9.1f}% {b - a:+6.1f} {fired:7d}")

    for cache in args.transfer:
        other = in_scope(load_records(cache, templates))
        if not other:
            continue
        other_stages = stage_one(other, templates)
        other_truth = np.array([r["truth"] for r in other])
        other_pred = np.array([s["order"][0] for s in other_stages])
        other_top2 = [frozenset(s["order"][:2]) for s in other_stages]
        out = other_pred.copy()
        fired = 0
        for pair, model in models.items():
            pair_set = frozenset(pair)
            gate = gate_mask(other_stages, pair, args.gate_margin)
            if not gate.any():
                continue
            X = np.array([
                pair_features(other[i], other_stages[i], pair, templates, args.features)
                for i in np.flatnonzero(gate)
            ])
            choice = model.predict(X)
            out[gate] = np.where(choice == 1, pair[0], pair[1])
            fired += int(gate.sum())
        a = 100 * (other_pred == other_truth).mean()
        b = 100 * (out == other_truth).mean()
        print(f"\ntransfer to {cache.name}: {len(other)} recordings, gate fired {fired}")
        print(f"  stage one {a:.1f}%  ->  cascaded {b:.1f}%  ({b - a:+.1f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
