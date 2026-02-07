# Dastgah Classification

This project trains a simple baseline classifier for 6 Dastgah classes using MP3 full tracks.

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
pip install -r requirements.txt
```

## Train (PyTorch baseline)

```
python train.py --data Training_Data --run_dir runs/exp1
```

The script will:
- build `data/manifest.json`
- create a train/val/test split in `data/splits.json`
- train a small CNN on mel-spectrograms
- save `runs/exp1/model.pt`

## Train (scikit-learn baseline)

```
python train_sklearn.py --data Training_Data --run_dir runs/exp_sklearn --segment_seconds 45 --num_segments 10 --trim_silence --cache_dir data/cache
```

This version avoids PyTorch and trains a multinomial logistic regression on mel-spectrogram summary features.

## Train (SVM)

```
python train_svm.py --data Training_Data --run_dir runs/exp_svm --segment_seconds 45 --num_segments 10 --trim_silence --cache_dir data/cache
```

## Train (Ensemble: LR + SVM)

```
python train_ensemble.py --data Training_Data --run_dir runs/exp_ensemble --segment_seconds 45 --num_segments 10 --trim_silence --cache_dir data/cache
```

## Train (Ensemble: LR + SVM + PyTorch)

```
python train_ensemble_torch.py --data Training_Data --run_dir runs/exp_ensemble_torch --segment_seconds 45 --num_segments 10 --trim_silence --cache_dir data/cache --torch_model runs/exp_torch/model.pt --device cpu
```

## Compare Models

```
python compare_models.py --runs runs --out runs/compare_models.md
```

## Web UI

```
pip install streamlit
streamlit run app.py
```

Default model path in the UI:
`models/model.joblib`

## Notes

- Default segment length is 30s. Adjust with `--segment_seconds`.
- If you want to regenerate splits, delete `data/splits.json`.
