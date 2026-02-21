# Dastgah_Classification

This repository now contains two versions of the project:

- `Dastgah_Classifier_v1`: existing baseline/scikit/CNN pipeline
- `Dastgah_Classifier_v2`: interval-first pipeline with stronger pitch/percussive filtering

Shared dataset location (for both):

- `Training_Data/`

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
