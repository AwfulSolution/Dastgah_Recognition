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

## Repository structure

- `Dastgah_Classifier_v1`
  - Baseline production pipeline.
  - Includes scikit models (LR/SVM/stacking), CNN training, and Streamlit UI.
- `Dastgah_Classifier_v2`
  - Interval-first redesign.
  - Focuses on pitch-interval behavior, tonic/cadence patterns, and better filtering of non-pitched percussion-heavy segments.

Shared dataset location (for both):

- `Training_Data/`

Expected dataset layout:

```text
Training_Data/
  Chahargah/
  Homayun/
  Mahur/
  Nava/
  Segah/
  Shur/
```

## Modeling approach

### v1

- Feature-based scikit pipeline + CNN variants.
- Supports mode-aware features and caching.
- Good as stable baseline and comparison target.

### v2

- Extracts interval-centric representations from harmonic content.
- Uses voiced/harmonic gating to suppress low-pitch/no-pitch and percussive segments.
- Designed to better capture Dastgah-defining melodic interval relations.

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
