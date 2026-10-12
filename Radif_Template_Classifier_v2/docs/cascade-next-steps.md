# Second-layer classifiers: where this stands, and what to try next

Handoff note for picking the cascade work up in a fresh session. The full
measurement record is in `docs/data-notes.md`; this is only what a next session
needs in order to start.

## What exists

`scripts/cascade.py`. Stage one ranks the six dastgahs; when its top two are a
confusable pair, a classifier trained on only that pair decides between them.

```bash
# from this folder, with ../.venv
../.venv/bin/python scripts/cascade.py --cache data/cache/nava.pkl \
    --transfer data/cache/kdc.pkl data/cache/irma.pkl
../.venv/bin/python scripts/cascade.py --cache data/cache/nava.pkl \
    --leave-one-corpus-out data/cache/nava.pkl data/cache/kdc.pkl data/cache/irma.pkl
```

Flags: `--features {all,minimal}`, `--train-on {all,gated}`, `--min-errors`.

The caches are at the repository root `data/cache/`, reachable from here through
a `data` symlink. They hold pitch histograms, bigram matrices, cadences, their
approach profiles and 20-second windows for five corpora, and they took about
two hours of pYIN to build. **Do not delete them**; they are gitignored and not
recoverable without the source audio.

| cache | records | what it is |
| --- | --- | --- |
| `nava.pkl` | 1,785 | 39 artists, 5 instruments, 7 dastgahs |
| `nava_long.pkl` | 247 | the subset over three minutes |
| `kdc.pkl` | 202 | 6 performers, 12 classes including avazes |
| `irma.pkl` | 144 | Karimi radif contours, 12 classes |
| `shajarian.pkl` | 17 | complete radif, 10-30 minutes each |

## What is established

**The gate is sound.** It fires only when `{top1, top2}` is the pair, so stage
one has already answered A or B and a recording whose truth is a third dastgah
was already wrong. The effect is provably confined to the pair. This is the
property that every earlier attempt lacked, and it should be preserved by
anything built next.

**Within a corpus it gains.** Artist-grouped 5-fold on Nava, 74.9% to 76.7%.

**It fixes same-tonic confusions and not rotations.**

| pair | gated | effect | why |
| --- | --- | --- | --- |
| chahargah / homayun | 292 | **+14** | confused at the *same* tonic |
| chahargah / segah | 64 | +5 | same |
| nava / shur | 399 | **+1 of 374** | a rotation of one shared collection |

This is the most portable finding here. Two modes sharing a pitch collection and
differing only in which degree is home give a second classifier nothing new to
read, because it reads the same pitch content. Pair models are worth building
for same-tonic pairs and are not the tool for rotations.

**It does not transfer.** Trained on Nava: KDC -3.2, IRMA -6.9. Leave-one-corpus-out,
each pair model trained on two corpora and tested on the third: Nava -5.5,
KDC -2.6, IRMA -3.8. Reducing the features from 53 to 5 does not change this, so
it is not the feature count.

## What to try next

Ordered by how much is behind each.

1. **Gate on the margin, not just the identity of the top two.** The gate
   currently fires whenever the pair holds the top two, including when stage one
   is confident. Firing only when `p(top1) - p(top2)` is small would confine the
   second stage to genuinely undecided recordings and shrink the blast radius
   further. Untried, cheap, and the most likely of these to survive transfer.

2. **Pair models on features that are not pitch-class bins.** Everything that has
   failed to transfer in this project reads the 24-bin tonic-relative profile,
   which supports roughly 70-75% and whose learned use never survives a corpus
   change. The cadence approach profiles are already cached per cadence and are
   anchored on the resolution rather than the tonic. A pair model built only on
   those would be the first learned thing here not reading the profile.

3. **A cascade over the long-recording subset.** Stage one is 89.0% on Nava's
   recordings over three minutes against 74.9% overall. The pairs that remain
   confusable at 89% are a different and smaller set, and a second stage has
   less to do but cleaner material to do it on.

4. **More corpora, which is the honest blocker.** Leave-one-corpus-out over three
   corpora is the strictest test available here and everything fails it. With
   five or six recording domains it would be possible to distinguish "cannot
   generalise" from "three domains are not enough to learn what generalises".

## What not to retry

All refuted, with mechanisms in `docs/data-notes.md`: refitting the global
scoring weights, training per-mode template parameters, lowering the sharpening
exponent, stacking a learned model on the classifier output, correcting one
template in isolation, progression through the seyr, the shahed as tonic
evidence in any of three forms, selecting among detected cadences, and the
cadence-approach agreement term.

The largest untouched lever in the whole project is not a classifier at all:
**recording length**. 74.9% at Nava's 75-second median against 89.0% over three
minutes, same classifier and parameters. Scoring several windows of a short
upload and combining them has never been tried.
