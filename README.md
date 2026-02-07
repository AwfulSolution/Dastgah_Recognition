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
python train.py --data /Users/taha/Code/Dastgah_Classification/Training_Data --run_dir runs/exp1
```

The script will:
- build `data/manifest.json`
- create a train/val/test split in `data/splits.json`
- train a small CNN on mel-spectrograms
- save `runs/exp1/model.pt`

## Train (scikit-learn baseline)

```
python train_sklearn.py --data /Users/taha/Code/Dastgah_Classification/Training_Data --run_dir runs/exp_sklearn --num_segments 6
```

This version avoids PyTorch and trains a multinomial logistic regression on mel-spectrogram summary features.

## Web UI

```
/Users/taha/Code/Dastgah_Classification/.venv/bin/pip install streamlit
/Users/taha/Code/Dastgah_Classification/.venv/bin/streamlit run /Users/taha/Code/Dastgah_Classification/app.py
```

To run again later:
```
/Users/taha/Code/Dastgah_Classification/.venv/bin/streamlit run /Users/taha/Code/Dastgah_Classification/app.py
```

Default model path in the UI:
`models/model.joblib`

## Notes

- Default segment length is 30s. Adjust with `--segment_seconds`.
- If you want to regenerate splits, delete `data/splits.json`.
