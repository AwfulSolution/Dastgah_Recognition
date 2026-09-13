"""Stacked ensemble over several representations of the same corpus.

The base models are deliberately built on different views of the audio, so
that their errors decorrelate: note-event histograms describe which pitches
are used and how phrases resolve, while a spectrogram model sees timbre and
time-frequency texture it has no notion of notes for. A decider is then
fitted on their out-of-fold probabilities.

Protocol, which is what keeps the leakage honest:

  * base models produce out-of-fold probabilities over the corpus using the
    same album/performer-grouped folds as cv_melodic.py, so no base model
    ever predicts a track whose group it trained on;
  * the decider is fitted on those out-of-fold probabilities only;
  * every base model is then refitted on the whole corpus and, together with
    the decider, applied to an external corpus that nothing in the stack has
    ever seen (see eval_external.py / data/kdc_manifest.json).

The external number is the one worth quoting: the decider's fit to the
out-of-fold probabilities is itself an in-sample quantity.
"""

import argparse
import json
import os
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)

from cv_melodic import balanced_group_folds  # noqa: E402
from dastgah_v4 import LABELS  # noqa: E402
from dastgah_v4.data import Track, label_to_index  # noqa: E402
from dastgah_v4.melodic_features import MelodicFeatureConfig, build_track_matrix  # noqa: E402
from dastgah_v4.modeling import build_model  # noqa: E402


class MelodicBase:
    """Note-event features (tonic-relative histograms + koron block) + CatBoost."""

    name = "melodic"

    def __init__(self, args):
        self.args = args
        self.cfg = MelodicFeatureConfig(
            segment_seconds=args.segment_seconds,
            num_segments=args.num_segments,
            trim_silence=True,
            tonic_strategy="vote",
            koron_features=not args.no_koron,
        )

    def prepare(self, corpus_tracks, ext_tracks, l2i):
        self.X, self.y = build_track_matrix(
            corpus_tracks, self.cfg, "train", self.args.seed, self.args.cache_dir,
            l2i, "Melodic (corpus)", self.args.num_workers)
        self.X_ext, self.y_ext = build_track_matrix(
            ext_tracks, self.cfg, "train", self.args.seed, self.args.cache_dir,
            l2i, "Melodic (external)", self.args.num_workers)

    def _model(self):
        return build_model(self.args.model_type, seed=self.args.seed, use_pca=False,
                           pca_variance=0.95, model_jobs=self.args.model_jobs)

    def fit_predict(self, train_idx, pred_idx):
        m = self._model()
        m.fit(self.X[train_idx], self.y[train_idx])
        return m.predict_proba(self.X[pred_idx])

    def fit_full_predict_external(self):
        m = self._model()
        m.fit(self.X, self.y)
        return m.predict_proba(self.X_ext)


class SpectralBase:
    """Log-mel spectrogram segments + a small CNN, averaged to track level.

    Trained per fold on segment rows belonging to the fold's training tracks,
    so the grouped split is respected at the track level.
    """

    name = "spectral"

    def __init__(self, args):
        self.args = args
        self.cfg = MelodicFeatureConfig(
            segment_seconds=args.segment_seconds,
            num_segments=args.num_segments,
            trim_silence=True,
            tonic_strategy="vote",
        )

    def prepare(self, corpus_tracks, ext_tracks, l2i):
        from dastgah_v4.spectral import build_spec_dataset

        self.S, self.sy, self.owner = build_spec_dataset(
            corpus_tracks, self.cfg, "train", self.args.seed, self.args.cache_dir,
            l2i, "Spectrogram (corpus)")
        self.S_ext, self.sy_ext, self.owner_ext = build_spec_dataset(
            ext_tracks, self.cfg, "train", self.args.seed, self.args.cache_dir,
            l2i, "Spectrogram (external)")
        self.n_tracks = len(corpus_tracks)
        self.n_ext = len(ext_tracks)
        # dB spectrograms sit roughly in [-80, 0]; centre them once.
        self.mu = float(self.S.mean())
        self.sd = float(self.S.std()) or 1.0

    def _train_cnn(self, seg_idx):
        import torch
        from torch import nn

        dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        torch.manual_seed(self.args.seed)
        X = torch.from_numpy(((self.S[seg_idx] - self.mu) / self.sd)).unsqueeze(1)
        yv = torch.from_numpy(self.sy[seg_idx])

        def block(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(), nn.MaxPool2d(2))

        net = nn.Sequential(
            block(1, 16), block(16, 32), block(32, 64), block(64, 64),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            nn.Dropout(0.3), nn.Linear(64, len(LABELS)),
        ).to(dev)
        opt = torch.optim.Adam(net.parameters(), lr=1e-3)
        lossf = nn.CrossEntropyLoss()
        bs = self.args.cnn_batch
        net.train()
        for ep in range(self.args.cnn_epochs):
            perm = torch.randperm(len(X))
            tot = 0.0
            for i in range(0, len(X), bs):
                b = perm[i:i + bs]
                xb, yb = X[b].to(dev), yv[b].to(dev)
                opt.zero_grad()
                loss = lossf(net(xb), yb)
                loss.backward()
                opt.step()
                tot += float(loss) * len(b)
            if self.args.cnn_verbose:
                print(f"    epoch {ep + 1}/{self.args.cnn_epochs} loss {tot / len(X):.4f}", flush=True)
        return net, dev

    def _track_probs(self, net, dev, S, owner, n_tracks):
        import torch

        net.eval()
        X = torch.from_numpy(((S - self.mu) / self.sd)).unsqueeze(1)
        outs = []
        with torch.no_grad():
            for i in range(0, len(X), 128):
                outs.append(torch.softmax(net(X[i:i + 128].to(dev)), dim=1).cpu().numpy())
        seg_p = np.vstack(outs)
        probs = np.zeros((n_tracks, len(LABELS)))
        for t in range(n_tracks):
            rows = seg_p[owner == t]
            probs[t] = rows.mean(axis=0) if len(rows) else 1.0 / len(LABELS)
        return probs

    def fit_predict(self, train_idx, pred_idx):
        seg_tr = np.isin(self.owner, train_idx)
        net, dev = self._train_cnn(np.where(seg_tr)[0])
        full = self._track_probs(net, dev, self.S, self.owner, self.n_tracks)
        return full[pred_idx]

    def fit_full_predict_external(self):
        net, dev = self._train_cnn(np.arange(len(self.S)))
        return self._track_probs(net, dev, self.S_ext, self.owner_ext, self.n_ext)


class TimbreBase:
    """Pooled spectral-envelope statistics (MFCC/contrast/centroid) + CatBoost."""

    name = "timbre"

    def __init__(self, args):
        self.args = args
        self.cfg = MelodicFeatureConfig(
            segment_seconds=args.segment_seconds,
            num_segments=args.num_segments,
            trim_silence=True,
            tonic_strategy="vote",
        )

    def prepare(self, corpus_tracks, ext_tracks, l2i):
        from dastgah_v4.spectral import build_timbre_matrix

        self.X, self.y = build_timbre_matrix(
            corpus_tracks, self.cfg, "train", self.args.seed, self.args.cache_dir,
            l2i, "Timbre (corpus)")
        self.X_ext, self.y_ext = build_timbre_matrix(
            ext_tracks, self.cfg, "train", self.args.seed, self.args.cache_dir,
            l2i, "Timbre (external)")

    def _model(self):
        return build_model(self.args.model_type, seed=self.args.seed, use_pca=False,
                           pca_variance=0.95, model_jobs=self.args.model_jobs)

    def fit_predict(self, train_idx, pred_idx):
        m = self._model()
        m.fit(self.X[train_idx], self.y[train_idx])
        return m.predict_proba(self.X[pred_idx])

    def fit_full_predict_external(self):
        m = self._model()
        m.fit(self.X, self.y)
        return m.predict_proba(self.X_ext)


BASES = {"melodic": MelodicBase, "spectral": SpectralBase, "timbre": TimbreBase}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--bases", default="melodic",
                   help="comma-separated base model names: " + ",".join(BASES))
    p.add_argument("--train_manifest", default=os.path.join(REPO, "Dastgah_Classifier_v3", "data", "manifest.json"))
    p.add_argument("--groups", default=os.path.join(ROOT, "data", "groups.json"))
    p.add_argument("--external_manifest", default=os.path.join(ROOT, "data", "kdc_manifest.json"))
    p.add_argument("--cache_dir", default=os.path.join(ROOT, "data", "cache"))
    p.add_argument("--out", default=os.path.join(ROOT, "runs", "stack_ensemble.json"))
    p.add_argument("--model_type", default="catboost")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--model_jobs", type=int, default=4)
    p.add_argument("--num_segments", type=int, default=6)
    p.add_argument("--segment_seconds", type=float, default=30.0)
    p.add_argument("--no_koron", action="store_true")
    p.add_argument("--cnn_epochs", type=int, default=20)
    p.add_argument("--cnn_batch", type=int, default=32)
    p.add_argument("--cnn_verbose", action="store_true")
    return p.parse_args()


def report(tag, y, pred, n_label=None):
    acc = float(accuracy_score(y, pred))
    mf1 = float(f1_score(y, pred, average="macro", zero_division=0))
    per = {LABELS[i]: round(float(v), 3) for i, v in enumerate(
        f1_score(y, pred, average=None, labels=range(len(LABELS)), zero_division=0))}
    print(f"{tag:28} acc {acc:.3f} | macro F1 {mf1:.3f}" + (f"  (n={n_label})" if n_label else ""))
    return {"acc": round(acc, 4), "macro_f1": round(mf1, 4), "per_class_f1": per}


def main():
    args = parse_args()
    names = [n.strip() for n in args.bases.split(",") if n.strip()]
    unknown = [n for n in names if n not in BASES]
    if unknown:
        raise SystemExit(f"unknown base model(s): {unknown}; available: {list(BASES)}")

    with open(args.train_manifest) as f:
        corpus = json.load(f)
    with open(args.groups) as f:
        groups = np.array(json.load(f))
    with open(args.external_manifest) as f:
        ext = [m for m in json.load(f) if os.path.exists(m["path"])]

    l2i = label_to_index()
    corpus_tracks = [Track(path=m["path"], label=m["label"]) for m in corpus]
    ext_tracks = [Track(path=m["path"], label=m["label"]) for m in ext]
    y = np.array([l2i[m["label"]] for m in corpus], dtype=np.int64)
    y_ext = np.array([l2i[m["label"]] for m in ext], dtype=np.int64)
    n_cls = len(LABELS)
    print(f"corpus {len(corpus)} tracks | external {len(ext)} tracks | bases: {names}\n", flush=True)

    bases = {}
    for n in names:
        b = BASES[n](args)
        print(f"[{n}] preparing features...", flush=True)
        b.prepare(corpus_tracks, ext_tracks, l2i)
        bases[n] = b

    folds = list(balanced_group_folds(y, groups, args.folds, args.seed))

    oof = {n: np.zeros((len(y), n_cls)) for n in names}
    for fi, (tr, ev) in enumerate(folds):
        for n in names:
            oof[n][ev] = bases[n].fit_predict(tr, ev)
        print(f"fold {fi}: " + " | ".join(
            f"{n} acc={accuracy_score(y[ev], oof[n][ev].argmax(1)):.3f}" for n in names), flush=True)

    print()
    results = {"bases": {}, "config": {"bases": names, "seed": args.seed, "folds": args.folds}}
    for n in names:
        results["bases"][n] = {"oof": report(f"[{n}] out-of-fold", y, oof[n].argmax(1))}

    ext_probs = {}
    for n in names:
        print(f"[{n}] refitting on full corpus for external prediction...", flush=True)
        ext_probs[n] = bases[n].fit_full_predict_external()
        results["bases"][n]["external"] = report(f"[{n}] external", y_ext, ext_probs[n].argmax(1), len(y_ext))

    if len(names) > 1:
        Z = np.hstack([oof[n] for n in names])
        Z_ext = np.hstack([ext_probs[n] for n in names])
        decider = LogisticRegression(max_iter=2000, multi_class="multinomial")
        decider.fit(Z, y)
        pred_ext = decider.predict(Z_ext)
        print()
        results["decider"] = {
            "external": report("[decider] external", y_ext, pred_ext, len(y_ext)),
            "confusion": confusion_matrix(y_ext, pred_ext, labels=range(n_cls)).tolist(),
        }
        # A simple average is the baseline any decider has to beat.
        avg_pred = np.mean([ext_probs[n] for n in names], axis=0).argmax(1)
        results["mean_vote"] = {"external": report("[mean of bases] external", y_ext, avg_pred, len(y_ext))}
    else:
        print("\n(single base model: no decider to fit)")

    results["labels"] = list(LABELS)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
