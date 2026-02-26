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

From project root:

```bash
python Dastgah_Classifier_v2/train_interval_model.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v2/runs/exp_interval_svm_v1 \
  --model_type svm \
  --use_pca \
  --trim_silence \
  --num_workers 4 \
  --num_segments 8 \
  --segment_seconds 20 \
  --voiced_ratio_threshold 0.30 \
  --min_harmonic_ratio 0.55
```

To export the trained run directly to a production-style `models/` slot (same pattern as v1), add:

```bash
  --export_production \
  --models_dir Dastgah_Classifier_v2/models
```

Automatic behavior:

- If dataset files changed, `manifest.json` is rebuilt automatically.
- If splits are missing/invalid, `splits.json` is rebuilt automatically.
- You do not need to manually declare total file count.

## Predict one file

```bash
python Dastgah_Classifier_v2/predict_interval_model.py \
  --audio path/to/audio.mp3 \
  --model_dir Dastgah_Classifier_v2/runs/exp_interval_svm_v1
```

## Promote a run to `models/` (v1-style production slot)

```bash
python Dastgah_Classifier_v2/promote_run_to_models.py \
  --run_dir Dastgah_Classifier_v2/runs/exp_interval_svm_v1 \
  --models_dir Dastgah_Classifier_v2/models
```

This copies:

- `run_dir/model.joblib` -> `models/model.joblib`
- `run_dir/model_config.json` -> `models/model_config.json`
- `run_dir/metrics.json` -> `models/metrics.json` (if available)

## Compare v2 runs

```bash
python Dastgah_Classifier_v2/compare_models_v2.py \
  --runs Dastgah_Classifier_v2/runs \
  --out Dastgah_Classifier_v2/runs/compare_models_v2.md \
  --sort_by test_macro_f1
```

## Web app (Windows and macOS/Linux)

```bash
streamlit run Dastgah_Classifier_v2/app_v2.py
```

Then open the shown localhost URL in browser.

UI modes:

- `Single model`
  - Load from a run directory, or
  - Load production model from `Dastgah_Classifier_v2/models/model.joblib`
- `Compare runs`
  - Run the same uploaded files against multiple run directories

## Optional: one-time dataset conversion to WAV

This can reduce decode issues and speed repeated runs.

```bash
python Dastgah_Classifier_v2/convert_dataset_to_wav.py \
  --input_root Training_Data \
  --output_root Training_Data_wav \
  --sample_rate 22050 \
  --mono \
  --num_workers 4
```

If some files fail conversion, they are logged to:

- `Dastgah_Classifier_v2/logs/wav_conversion_failed_files.txt`
- `Dastgah_Classifier_v2/logs/wav_conversion_failed_details.log`

Then train with:

```bash
python Dastgah_Classifier_v2/train_interval_model.py \
  --data Training_Data_wav \
  --run_dir Dastgah_Classifier_v2/runs/exp_interval_svm_wav_v1 \
  --model_type svm \
  --use_pca \
  --trim_silence \
  --num_workers 4
```

## Notes

- Cache is on disk under `Dastgah_Classifier_v2/data/cache` by default.
- If you change feature settings, cache keys change automatically.
- If some source files are corrupted, librosa/mpg123 warnings can still appear; those tracks are still handled best-effort.
