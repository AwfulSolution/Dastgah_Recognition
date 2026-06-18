# Dastgah_Classifier_v3

Phrase-aware melodic Dastgah classifier.

v3 keeps the v2 idea of pitch/interval classification, but improves the melodic representation before classification:

- estimates tonic once per track from note durations, stable notes, and phrase endings
- converts frame-level pitch into note events before building features
- separates stable notes from short ornaments
- extracts phrase-ending cadence features
- keeps more sequence information through interval transitions and step bigrams
- still filters weak/non-melodic segments with voiced and harmonic-ratio thresholds

## Results (2026-06-11)

Production model: CatBoost, 30s x 6 segments, no PCA, `hop_length=512`, `tonic_strategy=vote`.

| Split | Acc | Macro F1 | Balanced Acc |
|---|---:|---:|---:|
| Val | 0.725 | 0.721 | 0.722 |
| Test | 0.765 | 0.750 | 0.753 |

Notable findings (full tables in `runs/compare_models_v3.md` and `runs/cadence_sweep_results.txt`):

- 30s x 6 segments beats 15s x 3 by ~8 points test accuracy.
- `hop_length=512` matches 256 on val/test at half the pyin cost (5-fold CV confirmed) and is now the default.
- `tonic_strategy=vote` gives the best Homayun F1 (0.653 vs 0.594 pooled in CV) with the lowest fold variance, at equal macro F1; now the default. Pooled tonic estimates flip toward the shahed on shahed-heavy Homayun tracks, rotating interval histograms into Chahargah.
- Cadence-weight sweep (cw 1.6-4.0 x cn 3/5) found the existing 1.6/3 already optimal; heavier cadence weighting amplifies phrase-detection noise and hurts.
- Segment budget is also optimal at 30s x 6: 45s x 6 underperforms on both splits, and 30s x 8 is statistically identical in 5-fold CV at 33% more extraction cost.
- Remaining known weakness: avaz recordings that modulate through foreign gushehs; would need segment-level prediction with voting/abstention.

## Train

From project root (this reproduces the production model):

```bash
python Dastgah_Classifier_v3/train_melodic_model.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v3/runs/<run_name> \
  --model_type catboost \
  --trim_silence \
  --num_workers 4 \
  --num_segments 6 \
  --segment_seconds 30
```

Defaults worth knowing: `--hop_length 512` and `--tonic_strategy vote` (both CV-validated, see Results). Feature extraction caches note events per track in `data/cache`, so reruns that only change vector-stage parameters (cadence weights, tonic strategy) take minutes, not hours.

Supported `--model_type` values are the same as v2:

- `svm`
- `svm_rbf`
- `svm_linear`
- `lr`
- `knn`
- `rf`
- `extratrees`
- `catboost`
- `ensemble`

To export directly to the production `models/` slot:

```bash
python Dastgah_Classifier_v3/train_melodic_model.py \
  --data Training_Data \
  --run_dir Dastgah_Classifier_v3/runs/<run_name> \
  --model_type svm \
  --use_pca \
  --trim_silence \
  --export_production
```

## Predict one file

```bash
python Dastgah_Classifier_v3/predict_melodic_model.py \
  --audio path/to/audio.mp3 \
  --model_dir Dastgah_Classifier_v3/runs/<run_name>
```

## Compare v3 runs

```bash
python Dastgah_Classifier_v3/compare_models_v3.py \
  --runs Dastgah_Classifier_v3/runs \
  --out Dastgah_Classifier_v3/runs/compare_models_v3.md \
  --sort_by test_macro_f1
```

## Web app

```bash
streamlit run Dastgah_Classifier_v3/app_v3.py
```

## Main difference from v2

v2 mostly pools segment-level interval histograms. v3 first turns pitch frames into note events, finds phrase boundaries from voiced gaps, estimates a stronger track-level tonic, and then builds tonic-normalized note, cadence, transition, and short sequence features.
