# Dastgah_Classifier_v2

Interval-first Dastgah classifier focused on melodic interval structure, with explicit filtering for low-pitch/percussive segments (for example Tombak-heavy parts).

## What is different from v1

- Uses pYIN pitch tracking on harmonic audio (HPSS).
- Builds interval-centric features:
  - tonic-normalized interval histogram
  - interval transition matrix
  - cadence histogram (tail-biased)
  - melodic step histogram
  - note-duration histogram
- Filters non-melodic segments using:
  - voiced-frame ratio threshold
  - harmonic-to-percussive energy ratio threshold
  - short voiced-run suppression
- Uses disk cache for per-track features.

## Train

From project root (`/Users/taha/Code/Dastgah_Classification`):

```bash
python Dastgah_Classifier_v2/train_interval_model.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v2/runs/exp_interval_svm_v1 \
  --model_type svm \
  --use_pca \
  --trim_silence \
  --num_segments 8 \
  --segment_seconds 20 \
  --voiced_ratio_threshold 0.30 \
  --min_harmonic_ratio 0.55
```

Automatic behavior:

- If dataset files changed, `manifest.json` is rebuilt automatically.
- If splits are missing/invalid, `splits.json` is rebuilt automatically.
- You do not need to manually declare total file count.

## Predict one file

```bash
python Dastgah_Classifier_v2/predict_interval_model.py \
  --audio /absolute/path/to/audio.mp3 \
  --model_dir Dastgah_Classifier_v2/runs/exp_interval_svm_v1
```

## Web app (Windows and macOS/Linux)

```bash
streamlit run Dastgah_Classifier_v2/app_v2.py
```

Then open the shown localhost URL in browser.

## Notes

- Cache is on disk under `Dastgah_Classifier_v2/data/cache` by default.
- If you change feature settings, cache keys change automatically.
- If some source files are corrupted, librosa/mpg123 warnings can still appear; those tracks are still handled best-effort.
