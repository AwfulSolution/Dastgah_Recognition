# Dastgah Classifier v4 — theory-guided

v4 starts from the v3 phrase-aware melodic pipeline (pyin note events, tonic-relative
pitch-class/interval/duration histograms, cadence weighting, 30s×6 segments, CatBoost)
and adds what v3 lacked: **explicit Persian music theory**. Measured without album
or performer leakage (see Measurement rules), that v3 pipeline scores 0.523 pooled
accuracy on the 570-track dataset, and its errors are the ones scale-content models
make: Segah↔Shur, weak Homayun, unstable Nava. Those dastgahs differ by note
*function* (shahed, ist, forud, moteghayyer) and by the exact placement of their
neutral intervals, not by note *inventory*.

## Design goals, and what came of them

1. **Ist/forud-guided tonic detection** — *dropped*. The tonic audit
   (`analysis/tonic_audit.py`) compared three independent tonic estimators on
   correct vs misclassified tracks: they agree on ~75% of the misclassified ones.
   The model knows where home is and still picks the wrong dastgah, so tonic
   detection is not the binding constraint.
2. **Shahed features** — *negative in one-hot form* (`--function_features`).
   CatBoost gives the block importance proportional to its dimension count, i.e.
   it is redundant with the existing histograms. Only the phrase-position profile
   inside it earned any importance; a soft-profile rework remains open.
3. **Farhat interval templates** — *refuted* (`--template_features`). See below.
4. **Neutral-interval resolution** — **confirmed, now default.** The koron block
   is the one theory feature that survives replication.
5. **Segment voting with abstention** — *negative* (`cv_segment_vote.py`). See below.

The pattern across five attempts: what worked measures *intonation* at a
resolution the baseline histograms cannot represent. What failed re-encoded
information CatBoost could already extract from the existing features.

## Measurement rules (learned the hard way in v3)

- **Group-aware splits**: same-album/performance tracks must never span train/eval.
  Groups come from ID3 album tags with a per-performer merge for multi-CD radif
  sets (`src/dastgah_v4/grouping.py`).
- **Pooled grouped 5-fold CV for all headline numbers** (`cv_melodic.py`): every
  track is predicted exactly once by a model that never saw its group, with a
  size+class-balanced greedy splitter. Single 86-track test splits swing ±7
  points between seeds, and album leakage inflated them by ~10-20 points.
- Feature caches are shared with v3 (`data/cache` symlinks to v3's). Cache entries
  key on track path + `CORPUS_VERSION` + config signature — deliberately *not* on
  file metadata or content hashes, so that tooling which rewrites audio files in
  place (tag rewrites, timestamp churn) cannot silently invalidate hours of pyin
  extraction. Bump `CORPUS_VERSION` in `src/dastgah_v4/cache.py` when audio at an
  existing path is genuinely replaced; adding new files needs no bump.

### Honest baseline (pooled grouped CV, 570 tracks, catboost 30s×6)

All numbers below are measured on the current feature caches. An earlier table
reported figures ~2 points higher across the board; those came from note caches
computed under a numpy build that has since been replaced (see Environment), and
are retired — do not compare against them.

| config | pooled acc | pooled macro F1 |
|---|---|---|
| v3-parity (`--no_koron_features`) | 0.523 | 0.524 |
| + one-hot function block (`--function_features`) | 0.526\* | 0.527\* |
| + Farhat templates (`--template_features`) | 0.516 | 0.518 |
| **+ koron features (default)** | **0.546** | **0.549** |
| + koron + templates | 0.544 | 0.544 |

\* function block measured on retired caches (0.526 vs 0.542 parity then); its
sign is unlikely to flip but the figure is not directly comparable.

**Segment abstention voting** (`cv_segment_vote.py`: per-segment classification,
probability-averaged track votes, confidence-threshold abstention) measured
**negative** on clean notes: 0.49 pooled acc at every tau vs 0.52 whole-track,
with abstention thresholds flat (±0.4 points). Two lessons: 30-second segments
are too sparse an observation to beat whole-track histograms, and modulating
gushehs fail *confidently* (a Hesar passage votes Shur with conviction), so
confidence-gated abstention cannot rescue them. Kept for reference; the
modulation problem needs a different lever (e.g. explicitly modeling gusheh
structure, not filtering by confidence).

**Koron features** (cents-level intonation histograms over the neutral 2nd/3rd/6th
regions against a drift-robust continuous tonic reference) are the one theory
feature that holds up, and are on by default. +2.3 pooled accuracy over parity,
concentrated where the mechanism predicts: **Segah +8.5 F1** — the class whose
own tonic sits on a neutral degree, and the weakest class under every other
configuration — plus Mahur +4.5 (near-zero koron-ness is itself a clean Mahur
signature) and Chahargah +3.6, against Nava −1.6. This result was measured twice
on independently recomputed features and reproduced in both magnitude and
per-class shape.

**Farhat templates are refuted.** They score below parity alone (0.516), and
adding them on top of koron costs 0.2 — they carry nothing koron does not
already capture, which is unsurprising once stated plainly: both features ask
where the neutral degrees actually sit, and koron asks at 10-cent rather than
50-cent resolution. An earlier measurement showed templates +0.7 with gains
across the Shur family (Nava/Homayun/Shur), read at the time as theory
confirming itself. On recomputed features all three of those classes move
*down* and the only gain lands on Segah. That per-class story was noise fitted
to ±7-point fold variance; the module stays in the tree as a negative result,
not a building block.

Per-class F1 (parity): Chahargah 0.64, Mahur 0.55, Shur 0.52, Homayun 0.51,
Nava 0.49, Segah 0.44. The confusion structure is theory-consistent: Shur
absorbs its family relatives (Mahur/Nava/Segah→Shur), and Chahargah↔Segah
confuse symmetrically. The leaky pre-v4 numbers (0.59-0.66) measured performer
memorization as much as dastgah recognition — melodic features leak performer
identity through tonic conventions and repertoire, not just timbre.

## Environment

Use a virtualenv with numpy 1.26.x on OpenBLAS (Python 3.11 tested). On Apple
Silicon, numpy builds linked against the Accelerate framework (e.g. numpy 2.3)
can segfault inside `librosa.pyin` during feature extraction — check
`numpy.show_config()` if extraction workers die with SIGSEGV.

## Training

Same CLI as v3; koron features are on by default, so this trains the best
measured configuration:

```bash
python Dastgah_Classifier_v4/train_melodic_model.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v4/runs/<run_name> \
  --model_type catboost \
  --trim_silence \
  --num_segments 6 --segment_seconds 30
```

Add `--no_koron_features` to reproduce the v3-parity baseline.

## Evaluation

Headline numbers come from the grouped CV harness, never from a single split:

```bash
python Dastgah_Classifier_v4/cv_melodic.py --out Dastgah_Classifier_v4/runs/cv_default.json
```

The first run pays pyin extraction for the whole corpus (hours); afterwards the
note cache makes a full re-evaluation a matter of rebuilding feature vectors and
refitting (minutes).
