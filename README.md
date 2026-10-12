# Dastgah_Classification

End-to-end classification project for Persian/Iranian music mode recognition (Dastgah detection).

## Project goal

Given an input audio track, predict which Dastgah it belongs to.

Target classes:

- `Chahargah`
- `Homayun`
- `Mahur`
- `Nava`
- `Segah`
- `Shur`

## Two lines of work

The version folders are **not one continuous progression**. There are two
independent lines, and a higher number does not mean a successor.

**Trained line — v1 through v5.** Supervised models learned from `Training_Data/`,
each version a redesign of the one before it. Shares a dataset, a split protocol
and an evaluation.

**Template line — `Radif_Template_Classifier_v1` and `v2`.** A classifier with no
trained model at all: modal templates derived from Talai's notation of Mirza
Abdollah's radif, scored by searching jointly over modal classes and tonics.
Separate codebase, separate corpora, separate accuracy figures.

It carries its own name and its own numbering from 1 precisely so that nothing
reads as a successor to the trained line. It began as its own repository, it
does not build on `Dastgah_Classifier_v5`, and its numbers are not comparable
with the trained line's because they are not measured on the same data.

| folder | line | what it is |
| --- | --- | --- |
| `Dastgah_Classifier_v1` | trained | Baseline. scikit models (LR/SVM/stacking), CNN variants, Streamlit UI. |
| `Dastgah_Classifier_v2` | trained | Interval-first redesign. pYIN on harmonic audio, interval histograms and transitions, percussion filtering. |
| `Dastgah_Classifier_v3` | trained | Phrase-aware melodic redesign. Note events, track-level tonic, cadence and interval-sequence features. CatBoost on 30s x 6 segments. |
| `Dastgah_Classifier_v4` | trained | Theory-guided. Adds explicit Persian theory to v3 -- shahed, ist, forud, moteghayyer -- since the confusable dastgahs differ by note *function*, not note inventory. |
| `Dastgah_Classifier_v5` | trained | v4 plus optional count-based pitch-class features (`count_features`), as an alternative to duration-weighted histograms. |
| `Radif_Template_Classifier_v1` | template | Template classifier, no training. Six dastgahs with avazes folded into their mothers. |
| `Radif_Template_Classifier_v2` | template | v1 plus a gated pairwise cascade for confusable pairs. Not enabled by default. |

### Why the numbers between the lines cannot be compared

The trained line reports held-out accuracy on `Training_Data/`, roughly 570
tracks. The template line reports accuracy on Nava, KDC, IRMA and a Shajarian
radif set, none of which it was tuned on, and it has no training split at all
because nothing is fitted. A figure from one line placed beside a figure from
the other says nothing.

## Repository structure

- `Training_Data/` -- the dataset the trained line (v1-v5) learns from.

```text
Training_Data/
  Chahargah/
  Homayun/
  Mahur/
  Nava/
  Segah/
  Shur/
```

The template line does not read `Training_Data/`; each of those folders
documents its own corpora.

### v4

- Adds explicit Persian music theory to the v3 pipeline: shahed, ist, forud and
  moteghayyer features, plus grouped cross-validation to keep a performer out of
  both sides of a split.

### v5

- v4 with optional count-based pitch-class features (`count_features`), counting
  notes per pitch class rather than weighting them by duration.

### Radif_Template_Classifier_v1 and v2 (template line)

- No model is trained. Modal templates come from notation, and an unseen
  recording is scored by searching jointly over 13 modal classes and 24 possible
  tonics in a quarter-tone representation.
- `Radif_Template_Classifier_v2` adds a second stage: when the first stage's top
  two are a confusable pair, a classifier trained on only that pair decides
  between them. It is off by default, so v2 answers exactly as v1 unless invoked.
- Each folder's `README.md` and `docs/data-notes.md` carry their own accuracy
  figures and, at more length, the approaches that did not work.

## Typical workflow

1. Train v1 and v2 models on the same split.
2. Compare `acc`, `macro_f1`, and `balanced_acc`.
3. Inspect confusion matrices and per-class report.
4. Keep best model(s) in `runs/*` and use corresponding UI/predict scripts.

## Notes

- Feature caches are disk-based to speed up reruns.
- Manifest/splits rebuild automatically when dataset content changes.
- Some MP3 files may be partially corrupted; warnings can appear during decoding.

## Quick start (from repo root)

### v1

```bash
python Dastgah_Classifier_v1/train_svm.py --data Training_Data
```

```bash
streamlit run Dastgah_Classifier_v1/app.py
```

### v2

```bash
python Dastgah_Classifier_v2/train_interval_model.py --data Training_Data
```

```bash
streamlit run Dastgah_Classifier_v2/app_v2.py
```

### v3

```bash
python Dastgah_Classifier_v3/train_melodic_model.py --data Training_Data
```

```bash
streamlit run Dastgah_Classifier_v3/app_v3.py
```
