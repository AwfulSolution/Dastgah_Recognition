# Dastgah Classifier v4 — theory-guided

v4 starts from the v3 phrase-aware melodic pipeline (pyin note events, tonic-relative
pitch-class/interval/duration histograms, cadence weighting, 30s×6 segments, CatBoost)
and adds what v3 lacked: **explicit Persian music theory**. v3's honest baseline on the
570-track dataset is test acc ~0.59–0.66 across seeds (macro F1 ~0.59–0.64), and its
errors are exactly the ones scale-content models make: Segah↔Shur, weak Homayun,
unstable Nava. Those dastgahs differ by note *function* (shahed, ist, forud,
moteghayyer), not note *inventory*.

## Design goals

1. **Ist/forud-guided tonic detection** — weight tonic candidates by phrase-final,
   descending-approach evidence instead of raw note frequency. Gated on the tonic
   audit in `analysis/` showing tonic errors matter (they very likely do; every
   downstream feature is tonic-relative).
2. **Shahed features** — emphasis-weighted (duration × energy × phrase position)
   pitch-class profile, plus the tonic→shahed interval as a near-categorical feature.
   The main theory lever for Segah-vs-Shur.
3. **Farhat interval templates** — per-dastgah expected pitch-class profiles built
   from published cents values (Farhat, *The Dastgah Concept in Persian Music*);
   correlations fed as features / priors.
4. **Neutral-interval resolution** — cents-level histograms around the koron region
   (~135–160¢) where 24-TET bins blur the Shur-family distinctions.
5. **Segment voting with abstention** — modulating gushehs (Homayun's Hesar problem)
   abstain rather than poison the track vote.

## Measurement rules (learned the hard way in v3)

- **Group-aware splits**: same-album/performance tracks must never span train/eval;
  filename-normalized grouping lives in the split builder.
- **5-fold grouped CV for all headline numbers** — single 86-track test splits swing
  ±7 points between seeds.
- Feature caches are shared with v3 (`data/cache` symlinks to v3's; per-track note
  caches key on path+mtime+size+config, so identical extraction params cost nothing).

## Environment

Use a virtualenv with numpy 1.26.x on OpenBLAS (Python 3.11 tested). On Apple
Silicon, numpy builds linked against the Accelerate framework (e.g. numpy 2.3)
can segfault inside `librosa.pyin` during feature extraction — check
`numpy.show_config()` if extraction workers die with SIGSEGV.

## Training

Same CLI as v3:

```bash
python Dastgah_Classifier_v4/train_melodic_model.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v4/runs/<run_name> \
  --model_type catboost \
  --trim_silence \
  --num_segments 6 --segment_seconds 30
```
