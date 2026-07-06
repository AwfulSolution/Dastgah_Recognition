"""Tonic audit: is tonic detection the binding constraint on v3 accuracy?

For each rebaseline run (seed 42 / seed 1337), reload the trained model,
rebuild test features from cache, and split test tracks into correct vs
misclassified. For every track, derive three independent tonic candidates:

  vote    - per-segment vote (v3 production strategy)
  pooled  - argmax of the pooled duration/stable/cadence score
  ist     - phrase-final evidence: last notes of phrases, weighted by
            duration, with a 2x bonus when approached from above
            (a forud is a descending resolution)

plus the dispersion of per-segment tonic estimates. If misclassified tracks
disagree between candidates (or scatter across segments) much more than
correct ones, tonic instability is implicated and v4 should fix the tonic
first; if candidates agree even on misses, the features themselves are the
constraint.
"""

import json
import os
import sys
from collections import Counter

import joblib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))

from dastgah_v4.melodic_features import (  # noqa: E402
    MelodicFeatureConfig,
    _arrays_to_notes,
    _phrase_groups,
    estimate_tonic,
    extract_track_feature,
    notes_signature,
    vote_track_tonic,
)
from dastgah_v4.cache import load_cached_track_notes  # noqa: E402

V3 = os.path.join(REPO, "Dastgah_Classifier_v3")
CACHE = os.path.join(V3, "data", "cache")
RUNS = [
    ("seed42", os.path.join(V3, "runs", "exp_melodic_catboost_newdata_30s6"), os.path.join(V3, "data", "splits.json"), 42),
    ("seed1337", os.path.join(V3, "runs", "exp_melodic_catboost_newdata_30s6_seed1337"), os.path.join(V3, "data", "splits_seed1337.json"), 1337),
]


def ist_tonic(segment_note_lists, cfg) -> tuple[int | None, float]:
    """Phrase-final tonic candidate and its share of total phrase-final weight."""
    scores = np.zeros(cfg.bins_per_octave, dtype=np.float64)
    for seg_notes in segment_note_lists:
        for group in _phrase_groups(seg_notes, cfg):
            last = seg_notes[group[-1]]
            w = float(last.duration_frames)
            if len(group) > 1 and seg_notes[group[-2]].midi_mean > last.midi_mean:
                w *= 2.0  # descending approach: forud-like
            scores[last.pc] += w
    total = scores.sum()
    if total <= 0:
        return None, 0.0
    return int(np.argmax(scores)), float(scores.max() / total)


def segment_tonic_dispersion(segment_note_lists, cfg) -> tuple[int, int]:
    """(#segments casting a tonic vote, #distinct tonics among them)."""
    tonics = []
    for seg_notes in segment_note_lists:
        if not seg_notes or len({n.pc for n in seg_notes}) < 3:
            continue
        t, _, _ = estimate_tonic(seg_notes, cfg)
        tonics.append(t)
    return len(tonics), len(set(tonics))


def audit_run(name, run_dir, splits_path, seed):
    with open(os.path.join(run_dir, "model_config.json")) as f:
        mc = json.load(f)
    labels = mc["labels"]
    cfg = MelodicFeatureConfig(**mc["feature_config"])
    model = joblib.load(os.path.join(run_dir, "model.joblib"))

    with open(os.path.join(V3, "data", "manifest.json")) as f:
        manifest = json.load(f)
    with open(splits_path) as f:
        test_idx = json.load(f)["test"]

    notes_sig = notes_signature(cfg)
    suffix = f"notes-eval-seed{seed}"

    rows = []
    for i in test_idx:
        path, label = manifest[i]["path"], manifest[i]["label"]
        feat = extract_track_feature(path, cfg, "eval", seed, CACHE)
        pred = labels[int(model.predict(feat.reshape(1, -1))[0])]

        arrays = load_cached_track_notes(CACHE, path, notes_sig, suffix)
        if arrays is None:
            print(f"  ! no note cache for {os.path.basename(path)}; skipping candidates")
            continue
        seg_notes, _ = _arrays_to_notes(arrays)
        all_notes = [n for seg in seg_notes for n in seg]

        vote = vote_track_tonic(seg_notes, cfg)
        pooled, pooled_strength, _ = estimate_tonic(all_notes, cfg)
        ist, ist_share = ist_tonic(seg_notes, cfg)
        n_votes, n_distinct = segment_tonic_dispersion(seg_notes, cfg)

        rows.append({
            "track": os.path.basename(path), "label": label, "pred": pred,
            "correct": pred == label, "vote": vote, "pooled": pooled,
            "ist": ist, "ist_share": round(ist_share, 3),
            "pooled_strength": round(pooled_strength, 3),
            "segs_voting": n_votes, "distinct_seg_tonics": n_distinct,
        })

    def stats(subset):
        if not subset:
            return {}
        return {
            "n": len(subset),
            "vote!=ist": round(np.mean([r["vote"] != r["ist"] for r in subset]), 3),
            "vote!=pooled": round(np.mean([r["vote"] != r["pooled"] for r in subset]), 3),
            "all3_agree": round(np.mean([r["vote"] == r["pooled"] == r["ist"] for r in subset]), 3),
            "mean_distinct_seg_tonics": round(np.mean([r["distinct_seg_tonics"] for r in subset]), 2),
            "mean_ist_share": round(np.mean([r["ist_share"] for r in subset]), 3),
        }

    correct = [r for r in rows if r["correct"]]
    wrong = [r for r in rows if not r["correct"]]
    print(f"\n=== {name}: {len(correct)}/{len(rows)} correct ===")
    print("correct:      ", stats(correct))
    print("misclassified:", stats(wrong))
    print("\nmisclassified tracks:")
    for r in sorted(wrong, key=lambda r: (r["label"], r["track"])):
        flags = []
        if r["vote"] != r["ist"]:
            flags.append("VOTE!=IST")
        if r["distinct_seg_tonics"] > 2:
            flags.append("SCATTERED")
        print(f"  [{r['label']:9} -> {r['pred']:9}] {r['track'][:48]:50} "
              f"vote={r['vote']:>2} pooled={r['pooled']:>2} ist={str(r['ist']):>4} "
              f"segs={r['segs_voting']}/{r['distinct_seg_tonics']} {' '.join(flags)}")
    return rows


def main():
    all_rows = {}
    for name, run_dir, splits_path, seed in RUNS:
        all_rows[name] = audit_run(name, run_dir, splits_path, seed)
    out = os.path.join(ROOT, "analysis", "tonic_audit_results.json")
    with open(out, "w") as f:
        json.dump(all_rows, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
