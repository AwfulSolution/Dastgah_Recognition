import os
import sys
import tempfile
import unittest


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
V2_SRC = os.path.join(ROOT, "Dastgah_Classifier_v2", "src")
if V2_SRC not in sys.path:
    sys.path.insert(0, V2_SRC)
V3_SRC = os.path.join(ROOT, "Dastgah_Classifier_v3", "src")
if V3_SRC not in sys.path:
    sys.path.insert(0, V3_SRC)

from dastgah_v2.data import Track, build_splits  # noqa: E402
from dastgah_v2.paths import portable_path, resolve_config_path  # noqa: E402
from dastgah_v3.melodic_features import (  # noqa: E402
    MelodicFeatureConfig,
    NoteEvent,
    build_melodic_vector,
    cfg_signature,
    estimate_tonic,
    feature_dim,
    vote_track_tonic,
)


class V2CoreTests(unittest.TestCase):
    def test_build_splits_all_train_when_val_and_test_are_zero(self) -> None:
        tracks = [Track(path=f"/tmp/{i}.wav", label="Shur") for i in range(3)]

        splits = build_splits(tracks, val_split=0.0, test_split=0.0, seed=42)

        self.assertEqual(splits, {"train": [0, 1, 2], "val": [], "test": []})

    def test_build_splits_rejects_negative_split(self) -> None:
        tracks = [Track(path="/tmp/a.wav", label="Shur")]

        with self.assertRaises(ValueError):
            build_splits(tracks, val_split=-0.1, test_split=0.0, seed=42)

    def test_config_paths_round_trip_project_local_paths(self) -> None:
        with tempfile.TemporaryDirectory() as root:
            cache_dir = os.path.join(root, "data", "cache")

            stored = portable_path(cache_dir, root)
            resolved = resolve_config_path(stored, root, default="/unused")

            self.assertEqual(stored, os.path.join("data", "cache"))
            self.assertEqual(resolved, cache_dir)


class V3MelodicFeatureTests(unittest.TestCase):
    def test_estimate_tonic_uses_phrase_endings_and_stable_notes(self) -> None:
        cfg = MelodicFeatureConfig(bins_per_octave=24, cadence_weight=2.0, stable_weight=1.0)
        notes = [
            NoteEvent(0, 4, 7, 63.5, 4),
            NoteEvent(4, 8, 10, 65.0, 4),
            NoteEvent(8, 38, 0, 60.0, 30),
            NoteEvent(80, 84, 7, 63.5, 4),
            NoteEvent(84, 112, 0, 60.0, 28),
        ]

        tonic, strength, profile = estimate_tonic(notes, cfg)

        self.assertEqual(tonic, 0)
        self.assertGreater(strength, 0.0)
        self.assertAlmostEqual(float(profile.sum()), 1.0, places=5)

    def test_melodic_vector_has_configured_dimension(self) -> None:
        cfg = MelodicFeatureConfig()
        notes = [
            NoteEvent(0, 12, 0, 60.0, 12),
            NoteEvent(12, 18, 4, 62.0, 6),
            NoteEvent(30, 44, 0, 60.0, 14),
        ]

        vec = build_melodic_vector(notes, cfg, [{"voiced_ratio": 0.8, "harmonic_ratio": 0.7}])

        self.assertEqual(vec.shape, (feature_dim(cfg),))
        self.assertTrue((vec >= 0).all())

    def test_vote_tonic_majority_beats_confident_drone(self) -> None:
        cfg = MelodicFeatureConfig(tonic_strategy="vote")

        def mknote(start: int, pc: int, dur: int) -> NoteEvent:
            return NoteEvent(start, start + dur, pc, 60.0 + pc / 2, dur)

        seg_a = [mknote(i * 40, [5, 7, 8, 5, 3][i % 5], 30) for i in range(9)] + [mknote(400, 5, 40)]
        seg_b = [mknote(1000 + i * 40, [5, 3, 2, 5][i % 4], 30) for i in range(11)] + [mknote(1500, 5, 40)]
        drone = [mknote(2000 + i * 40, 9, 35) for i in range(12)]

        self.assertEqual(vote_track_tonic([seg_a, seg_b, drone], cfg), 5)

    def test_vote_tonic_returns_none_without_melodic_segments(self) -> None:
        cfg = MelodicFeatureConfig(tonic_strategy="vote")
        drone = [NoteEvent(i * 40, i * 40 + 35, 9, 64.5, 35) for i in range(12)]

        self.assertIsNone(vote_track_tonic([[], []], cfg))
        self.assertIsNone(vote_track_tonic([drone], cfg))

    def test_pooled_cfg_signature_has_no_tonic_strategy_marker(self) -> None:
        self.assertNotIn("-ts", cfg_signature(MelodicFeatureConfig()))
        self.assertTrue(cfg_signature(MelodicFeatureConfig(tonic_strategy="vote")).endswith("-tsvote"))


if __name__ == "__main__":
    unittest.main()
