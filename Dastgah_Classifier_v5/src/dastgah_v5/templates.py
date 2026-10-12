"""Theory-derived pitch-class templates per dastgah.

Scale-degree positions (cents above the tonic, octave-collapsed) follow the
interval structures analyzed in Farhat, *The Dastgah Concept in Persian
Music* — quantized here to quarter-tone bins (50 cents), which absorbs the
natural variation of the neutral (koron) intervals across performers and
schools. These are static scaffolds: they ignore moteghayyer inflections and
gusheh modulations, and serve as fixed anchors no performer's habits can
bias — the model measures how a track's observed pitch distribution aligns
with each theoretical scale, rather than with a performer's realization
memorized from training data.

Quarter-tone distinctions the templates preserve (bin = cents / 50):
- Shur vs Nava differ only at the 2nd (150 vs 200 cents) and 6th (800 vs 850).
- Chahargah and Homayun share the signature lower tetrachord (koron 2nd +
  plus-tone to a major 3rd) and differ in the upper (850/1100 vs 800/1000).
"""

from typing import Dict, List

import numpy as np

BINS = 24

# Scale degrees in cents above the tonic.
DASTGAH_DEGREES_CENTS: Dict[str, List[int]] = {
    "Chahargah": [0, 150, 400, 500, 700, 850, 1100],
    "Homayun":   [0, 150, 400, 500, 700, 800, 1000],
    "Mahur":     [0, 200, 400, 500, 700, 900, 1100],
    "Nava":      [0, 200, 300, 500, 700, 850, 1000],
    "Segah":     [0, 150, 350, 500, 650, 850, 1050],
    "Shur":      [0, 150, 300, 500, 700, 800, 1000],
}

# Tonic and fifth anchor the mode; weight them above the remaining degrees.
_DEGREE_WEIGHT_TONIC = 1.5
_DEGREE_WEIGHT_FIFTH = 1.25
_SMEAR = 0.25  # mass spread to each neighbor bin: tolerates intonation drift


def template_matrix(labels: List[str]) -> np.ndarray:
    """(n_labels, BINS) L2-normalized smoothed templates, row order = labels."""
    mat = np.zeros((len(labels), BINS), dtype=np.float64)
    for row, label in enumerate(labels):
        for cents in DASTGAH_DEGREES_CENTS[label]:
            b = int(round(cents / 50.0)) % BINS
            w = _DEGREE_WEIGHT_TONIC if cents == 0 else (_DEGREE_WEIGHT_FIFTH if cents == 700 else 1.0)
            mat[row, b] += w * (1.0 - 2 * _SMEAR)
            mat[row, (b - 1) % BINS] += w * _SMEAR
            mat[row, (b + 1) % BINS] += w * _SMEAR
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, 1e-9)


def template_features(hist: np.ndarray, templates: np.ndarray) -> np.ndarray:
    """Alignment features between a tonic-relative PC histogram and each template.

    Per template: cosine at the estimated tonic, best cosine over all 24
    rotations (tonic-error robust), and whether the best rotation is the
    estimated tonic. Returns (n_templates * 3,) float32.
    """
    h = hist.astype(np.float64)
    norm = np.linalg.norm(h)
    if norm <= 1e-12:
        return np.zeros(templates.shape[0] * 3, dtype=np.float32)
    h = h / norm

    # All cyclic rotations of h: rows are h rotated so bin r becomes the tonic.
    idx = (np.arange(BINS)[None, :] + np.arange(BINS)[:, None]) % BINS
    rotations = h[idx]  # (BINS, BINS)
    sims = rotations @ templates.T  # (BINS rotations, n_templates)

    cos_at_tonic = sims[0]
    best_cos = sims.max(axis=0)
    best_rot_is_tonic = (sims.argmax(axis=0) == 0).astype(np.float64)
    return np.concatenate([cos_at_tonic, best_cos, best_rot_is_tonic]).astype(np.float32)
