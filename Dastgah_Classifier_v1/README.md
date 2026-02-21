# Dastgah Classification

This project trains classifiers for 6 Dastgah classes from MP3 full tracks.

## Data layout

```
Training_Data/
  Chahargah/
  Homayun/
  Mahur/
  Nava/
  Segah/
  Shur/
```

## Setup

```
python -m venv .venv
source .venv/bin/activate
pip install -r Dastgah_Classifier_v1/requirements.txt
```

## Train (scikit logistic regression)

```
python Dastgah_Classifier_v1/train_sklearn.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v1/runs/exp_sklearn \
  --segment_seconds 45 \
  --num_segments 10 \
  --trim_silence \
  --cache_dir Dastgah_Classifier_v1/data/cache
```

## Train (SVM)

```
python Dastgah_Classifier_v1/train_svm.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v1/runs/exp_svm \
  --segment_seconds 45 \
  --num_segments 10 \
  --trim_silence \
  --cache_dir Dastgah_Classifier_v1/data/cache
```

## Train (Ensemble: LR + SVM)

```
python Dastgah_Classifier_v1/train_ensemble.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v1/runs/exp_ensemble \
  --segment_seconds 45 \
  --num_segments 10 \
  --trim_silence \
  --cache_dir Dastgah_Classifier_v1/data/cache
```

## Mode-aware features

Add the following flags to any scikit training command:

```
--use_mode_features --mode_pitch_bins 24
```

This adds pitch/tonic/interval/cadence features from `pyin` on top of the baseline features.

## PCA (recommended with mode features)

Add these flags to scikit training commands:

```
--use_pca --pca_variance 0.95
```

This adds PCA after standardization and keeps 95% of feature variance.

## Low-compute mode (for laptop iteration)

Add `--low_compute` to scikit train/predict commands. If you keep default values, it auto-adjusts to:

- `segment_seconds`: `45 -> 25`
- `num_segments`: `10 -> 4`
- `trim_db`: `25 -> 30`
- `mode_pitch_bins`: `24 -> 16`

## Predict

```
python Dastgah_Classifier_v1/predict.py \
  --model Dastgah_Classifier_v1/runs/exp_sklearn/model.joblib \
  --input /full/path/to/file_or_folder \
  --trim_silence \
  --use_mode_features
```

Important: inference flags must match training flags (`use_mode_features`, `mode_pitch_bins`, segment settings), or feature dimensions will not match.

## Web UI

```
streamlit run Dastgah_Classifier_v1/app.py
```

The UI now has controls for:

- mode-aware features
- mode pitch bins
- low-compute mode

## Notes

- Cached features are saved in `Dastgah_Classifier_v1/data/cache` and keyed by feature configuration.
- If you want to regenerate train/val/test split, delete `Dastgah_Classifier_v1/data/splits.json`.
