import hashlib
import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Tuple

os.environ.setdefault("NUMBA_CACHE_DIR", os.path.join("/private", "tmp", "dastgah_numba_cache"))

import librosa
import numpy as np
from tqdm.auto import tqdm

from .cache import (
    load_cached_track_features,
    load_cached_track_notes,
    save_cached_track_features,
    save_cached_track_notes,
)
from .data import Track


@dataclass
class MelodicFeatureConfig:
    sample_rate: int = 22050
    n_fft: int = 2048
    # 512 measured equal to 256 on val/test at half the pyin cost
    # (exp_melodic_catboost_30s6_nopca vs _hop512, 2026-06-11).
    hop_length: int = 512
    bins_per_octave: int = 24
    segment_seconds: float = 30.0
    num_segments: int = 6
    trim_silence: bool = True
    trim_db: int = 25
    voiced_ratio_threshold: float = 0.25
    min_voiced_ms: float = 120.0
    min_harmonic_ratio: float = 0.50
    min_note_ms: float = 90.0
    stable_note_ms: float = 220.0
    phrase_gap_ms: float = 280.0
    cadence_notes: int = 3
    cadence_weight: float = 1.6
    stable_weight: float = 0.8
    step_clip_bins: int = 12
    duration_bins: int = 8
    tonic_strategy: str = "vote"  # "pooled": argmax over all notes; "vote": per-segment tonic vote
    # Note-function features (shahed/ist): which degrees carry emphasis and
    # where phrases resolve, relative to the tonic. These target the modes that
    # share scale material but differ in note function (Segah/Shur families).
    function_features: bool = True


@dataclass
class NoteEvent:
    start: int
    end: int
    pc: int
    midi_mean: float
    duration_frames: int


def function_dim(cfg: MelodicFeatureConfig) -> int:
    # shahed one-hot, vice-shahed one-hot, ist one-hot, shahed-ist relation
    # one-hot, phrase-position profile, plus 6 scalars.
    return cfg.bins_per_octave * 5 + 6 if cfg.function_features else 0


def feature_dim(cfg: MelodicFeatureConfig) -> int:
    bins = cfg.bins_per_octave
    step_bins = cfg.step_clip_bins * 2 + 1
    return (bins * bins) + (bins * 5) + step_bins + (step_bins * step_bins) + cfg.duration_bins + function_dim(cfg) + 16


def cfg_signature(cfg: MelodicFeatureConfig) -> str:
    sig = (
        f"sr{cfg.sample_rate}-fft{cfg.n_fft}-hop{cfg.hop_length}-bins{cfg.bins_per_octave}"
        f"-seg{cfg.segment_seconds}-n{cfg.num_segments}-trim{int(cfg.trim_silence)}-db{cfg.trim_db}"
        f"-vr{cfg.voiced_ratio_threshold}-mv{cfg.min_voiced_ms}-hr{cfg.min_harmonic_ratio}"
        f"-mn{cfg.min_note_ms}-sn{cfg.stable_note_ms}-pg{cfg.phrase_gap_ms}"
        f"-cadn{cfg.cadence_notes}-cadw{cfg.cadence_weight}-stabw{cfg.stable_weight}"
        f"-sc{cfg.step_clip_bins}-dur{cfg.duration_bins}"
    )
    # Appended conditionally so caches built before this option existed stay valid.
    if cfg.tonic_strategy != "pooled":
        sig += f"-ts{cfg.tonic_strategy}"
    if cfg.function_features:
        sig += "-fn1"
    return sig


def notes_signature(cfg: MelodicFeatureConfig) -> str:
    """Cache signature for the note-event stage.

    Only includes parameters that influence segmentation and note extraction
    (hpss/pyin and run-length quantization). Vector-building parameters such as
    cadence weights or tonic_strategy are deliberately excluded so those can be
    iterated without redoing the expensive pitch tracking.
    """
    return (
        f"notes1-sr{cfg.sample_rate}-fft{cfg.n_fft}-hop{cfg.hop_length}-bins{cfg.bins_per_octave}"
        f"-seg{cfg.segment_seconds}-n{cfg.num_segments}-trim{int(cfg.trim_silence)}-db{cfg.trim_db}"
        f"-vr{cfg.voiced_ratio_threshold}-mv{cfg.min_voiced_ms}-hr{cfg.min_harmonic_ratio}"
        f"-mn{cfg.min_note_ms}"
    )


def _notes_to_arrays(segment_note_lists: List[List[NoteEvent]], metas: List[Dict[str, float]]) -> Dict[str, np.ndarray]:
    flat = [note for seg in segment_note_lists for note in seg]
    return {
        "seg_counts": np.array([len(seg) for seg in segment_note_lists], dtype=np.int64),
        "starts": np.array([n.start for n in flat], dtype=np.int64),
        "ends": np.array([n.end for n in flat], dtype=np.int64),
        "pcs": np.array([n.pc for n in flat], dtype=np.int64),
        "midi_means": np.array([n.midi_mean for n in flat], dtype=np.float64),
        "durations": np.array([n.duration_frames for n in flat], dtype=np.int64),
        "voiced_ratios": np.array([m["voiced_ratio"] for m in metas], dtype=np.float64),
        "harmonic_ratios": np.array([m["harmonic_ratio"] for m in metas], dtype=np.float64),
    }


def _arrays_to_notes(arrays: Dict[str, np.ndarray]) -> Tuple[List[List[NoteEvent]], List[Dict[str, float]]]:
    segment_note_lists: List[List[NoteEvent]] = []
    pos = 0
    for count in arrays["seg_counts"].tolist():
        seg = [
            NoteEvent(
                start=int(arrays["starts"][i]),
                end=int(arrays["ends"][i]),
                pc=int(arrays["pcs"][i]),
                midi_mean=float(arrays["midi_means"][i]),
                duration_frames=int(arrays["durations"][i]),
            )
            for i in range(pos, pos + count)
        ]
        segment_note_lists.append(seg)
        pos += count
    metas = [
        {"voiced_ratio": float(v), "harmonic_ratio": float(h)}
        for v, h in zip(arrays["voiced_ratios"].tolist(), arrays["harmonic_ratios"].tolist())
    ]
    return segment_note_lists, metas


def _load_segment(path: str, start_sample: int, cfg: MelodicFeatureConfig) -> np.ndarray:
    # Decode only this window: loading whole tracks made parallel extraction
    # exceed RAM on long recordings (tracks here run up to ~70 minutes).
    segment, _ = librosa.load(
        path,
        sr=cfg.sample_rate,
        mono=True,
        offset=start_sample / float(cfg.sample_rate),
        duration=cfg.segment_seconds,
    )
    if cfg.trim_silence:
        segment, _ = librosa.effects.trim(segment, top_db=cfg.trim_db)
    return segment


def _segment_starts(audio_len: int, sr: int, seconds: float, num_segments: int, mode: str, seed: int) -> List[int]:
    seg_len = int(seconds * sr)
    if audio_len <= seg_len:
        return [0]
    max_start = audio_len - seg_len
    if num_segments <= 1:
        return [max_start // 2]
    if mode == "train":
        rng = np.random.RandomState(seed)
        return rng.randint(0, max_start + 1, size=num_segments).tolist()
    return np.linspace(0, max_start, num_segments).astype(int).tolist()


def _suppress_short_runs(mask: np.ndarray, min_len: int) -> np.ndarray:
    if min_len <= 1 or mask.size == 0:
        return mask
    out = mask.copy()
    start = None
    for i, val in enumerate(mask):
        if val and start is None:
            start = i
        elif (not val) and start is not None:
            if i - start < min_len:
                out[start:i] = False
            start = None
    if start is not None and (mask.size - start) < min_len:
        out[start:] = False
    return out


def _duration_hist(lengths: np.ndarray, n_bins: int) -> np.ndarray:
    if lengths.size == 0:
        return np.zeros(n_bins, dtype=np.float32)
    edges = np.array([1, 2, 3, 4, 6, 8, 12], dtype=np.float32)
    if n_bins != 8:
        q = np.linspace(0, 1, max(n_bins - 1, 2))[1:-1]
        edges = np.quantile(lengths, q) if lengths.size > 1 else np.array([lengths[0]], dtype=np.float32)
    idx = np.digitize(lengths.astype(np.float32), edges, right=True)
    hist = np.bincount(idx, minlength=n_bins).astype(np.float64)
    return (hist / (hist.sum() + 1e-9)).astype(np.float32)


def _extract_segment_notes(segment: np.ndarray, cfg: MelodicFeatureConfig) -> Tuple[List[NoteEvent], Dict[str, float]]:
    if segment.size < cfg.n_fft:
        segment = np.pad(segment, (0, cfg.n_fft - segment.size), mode="constant")

    harmonic, percussive = librosa.effects.hpss(segment)
    h_rms = float(np.sqrt(np.mean(np.square(harmonic))) + 1e-12)
    p_rms = float(np.sqrt(np.mean(np.square(percussive))) + 1e-12)
    harmonic_ratio = h_rms / (h_rms + p_rms + 1e-9)

    f0, voiced_flag, _ = librosa.pyin(
        harmonic,
        sr=cfg.sample_rate,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        frame_length=cfg.n_fft,
        hop_length=cfg.hop_length,
    )
    if f0 is None or len(f0) == 0:
        return [], {"voiced_ratio": 0.0, "harmonic_ratio": harmonic_ratio}

    energy = librosa.feature.rms(y=harmonic, frame_length=cfg.n_fft, hop_length=cfg.hop_length)[0]
    n = min(len(f0), len(energy))
    f0 = np.asarray(f0[:n], dtype=np.float64)
    energy = np.asarray(energy[:n], dtype=np.float64)

    voiced = np.isfinite(f0)
    if voiced_flag is not None:
        voiced = np.logical_and(voiced, np.asarray(voiced_flag[:n], dtype=bool))
    if energy.size > 0:
        voiced = np.logical_and(voiced, energy >= np.percentile(energy, 20))

    min_run = max(1, int(round((cfg.min_voiced_ms / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    voiced = _suppress_short_runs(voiced, min_run)
    voiced_ratio = float(np.mean(voiced)) if voiced.size else 0.0
    if voiced_ratio < cfg.voiced_ratio_threshold or harmonic_ratio < cfg.min_harmonic_ratio:
        return [], {"voiced_ratio": voiced_ratio, "harmonic_ratio": harmonic_ratio}

    midi = librosa.hz_to_midi(f0)
    valid = np.logical_and(voiced, np.isfinite(midi))
    if int(np.sum(valid)) < 6:
        return [], {"voiced_ratio": voiced_ratio, "harmonic_ratio": harmonic_ratio}

    pc = np.full(n, -1, dtype=np.int64)
    pc[valid] = np.mod(np.floor((midi[valid] % 12.0) * (cfg.bins_per_octave / 12.0)).astype(int), cfg.bins_per_octave)

    min_note_frames = max(1, int(round((cfg.min_note_ms / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    notes: List[NoteEvent] = []
    start = None
    current_pc = -1
    for i, p in enumerate(pc):
        if p < 0:
            if start is not None and i - start >= min_note_frames:
                notes.append(NoteEvent(start, i, current_pc, float(np.nanmean(midi[start:i])), i - start))
            start = None
            current_pc = -1
            continue
        if start is None:
            start = i
            current_pc = int(p)
        elif int(p) != current_pc:
            if i - start >= min_note_frames:
                notes.append(NoteEvent(start, i, current_pc, float(np.nanmean(midi[start:i])), i - start))
            start = i
            current_pc = int(p)
    if start is not None and pc.size - start >= min_note_frames:
        notes.append(NoteEvent(start, pc.size, current_pc, float(np.nanmean(midi[start:])), pc.size - start))

    return notes, {"voiced_ratio": voiced_ratio, "harmonic_ratio": harmonic_ratio}


def _phrase_groups(notes: List[NoteEvent], cfg: MelodicFeatureConfig) -> List[List[int]]:
    if not notes:
        return []
    gap_frames = max(1, int(round((cfg.phrase_gap_ms / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    groups: List[List[int]] = [[0]]
    for i in range(1, len(notes)):
        if notes[i].start - notes[i - 1].end >= gap_frames:
            groups.append([i])
        else:
            groups[-1].append(i)
    return groups


def estimate_tonic(notes: List[NoteEvent], cfg: MelodicFeatureConfig) -> Tuple[int, float, np.ndarray]:
    bins = cfg.bins_per_octave
    if not notes:
        return 0, 0.0, np.zeros(bins, dtype=np.float32)

    pc = np.array([n.pc for n in notes], dtype=np.int64)
    dur = np.array([n.duration_frames for n in notes], dtype=np.float64)
    duration_scores = np.bincount(pc, weights=dur, minlength=bins).astype(np.float64)

    stable_cutoff = max(1, int(round((cfg.stable_note_ms / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    stable_mask = dur >= stable_cutoff
    stable_scores = np.bincount(pc[stable_mask], weights=dur[stable_mask], minlength=bins).astype(np.float64)

    cadence_scores = np.zeros(bins, dtype=np.float64)
    for group in _phrase_groups(notes, cfg):
        tail = group[-max(1, cfg.cadence_notes) :]
        for rank, note_idx in enumerate(reversed(tail)):
            weight = cfg.cadence_weight / (rank + 1)
            cadence_scores[notes[note_idx].pc] += weight * notes[note_idx].duration_frames

    scores = duration_scores + cfg.stable_weight * stable_scores + cadence_scores
    tonic = int(np.argmax(scores))
    sorted_scores = np.sort(scores)
    top1 = float(sorted_scores[-1]) if sorted_scores.size else 0.0
    top2 = float(sorted_scores[-2]) if sorted_scores.size > 1 else 0.0
    strength = (top1 - top2) / (top1 + 1e-9)
    return tonic, strength, (scores / (scores.sum() + 1e-9)).astype(np.float32)


def vote_track_tonic(segment_note_lists: List[List[NoteEvent]], cfg: MelodicFeatureConfig) -> int | None:
    """Pick the track tonic by letting each segment vote with its own estimate.

    A single argmax over pooled notes can be flipped by one shahed-heavy
    passage. Each melodically meaningful segment casts one vote (confidence
    only breaks ties) so a degenerate segment cannot dominate the track.
    """
    votes = np.zeros(cfg.bins_per_octave, dtype=np.float64)
    for seg_notes in segment_note_lists:
        if not seg_notes:
            continue
        if len({n.pc for n in seg_notes}) < 3:
            continue  # drones/near-monotone segments carry no tonic information
        tonic, strength, _ = estimate_tonic(seg_notes, cfg)
        votes[tonic] += 1.0 + 0.25 * strength
    if votes.sum() <= 0:
        return None
    return int(np.argmax(votes))


def _safe_hist(vals: np.ndarray, minlength: int, weights: np.ndarray | None = None) -> np.ndarray:
    if vals.size == 0:
        return np.zeros(minlength, dtype=np.float32)
    hist = np.bincount(vals, weights=weights, minlength=minlength).astype(np.float64)
    return (hist / (hist.sum() + 1e-9)).astype(np.float32)


def _function_block(
    notes: List[NoteEvent],
    intervals: np.ndarray,
    durations: np.ndarray,
    groups: List[List[int]],
    cfg: MelodicFeatureConfig,
) -> np.ndarray:
    """Note-function features: shahed (emphasis center), ist (resolution
    center), their relation, and a phrase-position-weighted profile.

    All degree indices are tonic-relative. One-hots are scaled by the share
    of evidence behind them so a weakly-dominant shahed reads differently
    from an overwhelming one.
    """
    bins = cfg.bins_per_octave

    emphasis = np.bincount(intervals, weights=durations, minlength=bins).astype(np.float64)
    emphasis_total = emphasis.sum() + 1e-9
    order = np.argsort(emphasis)[::-1]
    shahed, vice = int(order[0]), int(order[1])
    shahed_share = float(emphasis[shahed] / emphasis_total)
    vice_share = float(emphasis[vice] / emphasis_total)
    shahed_margin = float((emphasis[shahed] - emphasis[vice]) / (emphasis[shahed] + 1e-9))

    shahed_hot = np.zeros(bins, dtype=np.float32)
    shahed_hot[shahed] = shahed_share
    vice_hot = np.zeros(bins, dtype=np.float32)
    vice_hot[vice] = vice_share

    # Ist: phrase-final notes, duration-weighted, doubled when approached
    # from above (a forud resolves downward onto the ist).
    ist_scores = np.zeros(bins, dtype=np.float64)
    for group in groups:
        last = group[-1]
        w = float(durations[last])
        if len(group) > 1 and notes[group[-2]].midi_mean > notes[last].midi_mean:
            w *= 2.0
        ist_scores[intervals[last]] += w
    ist_total = ist_scores.sum()
    ist_hot = np.zeros(bins, dtype=np.float32)
    if ist_total > 0:
        ist = int(np.argmax(ist_scores))
        ist_share = float(ist_scores[ist] / ist_total)
        ist_hot[ist] = ist_share
    else:
        ist, ist_share = 0, 0.0

    rel_hot = np.zeros(bins, dtype=np.float32)
    rel_hot[(shahed - ist) % bins] = 1.0

    # Phrase-position profile: emphasis weighted toward phrase endings,
    # a smooth forud-flavored counterpart to the final-note cadence hist.
    pos_scores = np.zeros(bins, dtype=np.float64)
    for group in groups:
        glen = len(group)
        for j, idx in enumerate(group):
            pos_scores[intervals[idx]] += durations[idx] * ((j + 1) / glen) ** 2
    pos_hist = (pos_scores / (pos_scores.sum() + 1e-9)).astype(np.float32)

    scalars = np.array(
        [
            shahed_share,
            shahed_margin,
            ist_share,
            1.0 if shahed == 0 else 0.0,
            1.0 if ist == 0 else 0.0,
            1.0 if shahed == ist else 0.0,
        ],
        dtype=np.float32,
    )
    return np.concatenate([shahed_hot, vice_hot, ist_hot, rel_hot, pos_hist, scalars])


def build_melodic_vector(
    notes: List[NoteEvent],
    cfg: MelodicFeatureConfig,
    metas: List[Dict[str, float]],
    tonic_override: int | None = None,
) -> np.ndarray:
    bins = cfg.bins_per_octave
    step_bins = cfg.step_clip_bins * 2 + 1
    out_dim = feature_dim(cfg)
    if not notes:
        out = np.zeros(out_dim, dtype=np.float32)
        if metas:
            out[-16] = float(np.mean([m["voiced_ratio"] for m in metas]))
            out[-15] = float(np.mean([m["harmonic_ratio"] for m in metas]))
        return out

    tonic, tonic_strength, tonic_profile = estimate_tonic(notes, cfg)
    if tonic_override is not None:
        tonic = tonic_override
    pc = np.array([n.pc for n in notes], dtype=np.int64)
    durations = np.array([n.duration_frames for n in notes], dtype=np.float64)
    intervals = np.mod(pc - tonic, bins).astype(np.int64)
    stable_cutoff = max(1, int(round((cfg.stable_note_ms / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    stable_mask = durations >= stable_cutoff

    note_hist = _safe_hist(intervals, bins)
    duration_hist_pc = _safe_hist(intervals, bins, weights=durations)
    stable_hist = _safe_hist(intervals[stable_mask], bins, weights=durations[stable_mask])

    groups = _phrase_groups(notes, cfg)
    phrase_end_intervals = np.array([intervals[g[-1]] for g in groups], dtype=np.int64) if groups else np.empty(0, dtype=np.int64)
    cadence_hist = _safe_hist(phrase_end_intervals, bins)
    tonic_profile_rel = np.roll(tonic_profile, -tonic).astype(np.float32)

    trans = np.zeros((bins, bins), dtype=np.float64)
    if intervals.size > 1:
        np.add.at(trans, (intervals[:-1], intervals[1:]), 1.0)
        trans /= trans.sum() + 1e-9

    if intervals.size > 1:
        steps = intervals[1:] - intervals[:-1]
        steps = ((steps + (bins // 2)) % bins) - (bins // 2)
    else:
        steps = np.zeros(0, dtype=np.int64)
    clipped_steps = np.clip(steps, -cfg.step_clip_bins, cfg.step_clip_bins)
    step_hist = _safe_hist(clipped_steps + cfg.step_clip_bins, step_bins)

    step_bigram = np.zeros((step_bins, step_bins), dtype=np.float64)
    if clipped_steps.size > 1:
        s = clipped_steps + cfg.step_clip_bins
        np.add.at(step_bigram, (s[:-1], s[1:]), 1.0)
        step_bigram /= step_bigram.sum() + 1e-9

    note_duration_hist = _duration_hist(durations.astype(np.float32), cfg.duration_bins)

    total_frames = max(notes[-1].end - notes[0].start, 1)
    voiced_mean = float(np.mean([m["voiced_ratio"] for m in metas])) if metas else 0.0
    harmonic_mean = float(np.mean([m["harmonic_ratio"] for m in metas])) if metas else 0.0
    mean_abs_step = float(np.mean(np.abs(clipped_steps))) / max(cfg.step_clip_bins, 1) if clipped_steps.size else 0.0
    std_step = float(np.std(clipped_steps)) / max(cfg.step_clip_bins, 1) if clipped_steps.size else 0.0
    short_cutoff = max(1, int(round((cfg.min_note_ms * 2 / 1000.0) * cfg.sample_rate / cfg.hop_length)))
    ornament_ratio = float(np.mean(durations <= short_cutoff)) if durations.size else 0.0
    stable_ratio = float(np.sum(durations[stable_mask]) / (np.sum(durations) + 1e-9))
    note_density = float(len(notes) / max(total_frames, 1))
    phrase_density = float(len(groups) / max(total_frames, 1))
    phrase_end_to_tonic = float(cadence_hist[0]) if cadence_hist.size else 0.0
    final_to_tonic = 1.0 if intervals[-1] == 0 else 0.0
    median_duration = float(np.median(durations) / max(total_frames, 1))
    mean_duration = float(np.mean(durations) / max(total_frames, 1))
    pitch_span = float((np.max([n.midi_mean for n in notes]) - np.min([n.midi_mean for n in notes])) / 24.0)

    summary = np.array(
        [
            voiced_mean,
            harmonic_mean,
            tonic_strength,
            phrase_end_to_tonic,
            final_to_tonic,
            mean_abs_step,
            std_step,
            ornament_ratio,
            stable_ratio,
            note_density,
            phrase_density,
            min(len(notes) / 200.0, 1.0),
            min(len(groups) / 40.0, 1.0),
            median_duration,
            mean_duration,
            pitch_span,
        ],
        dtype=np.float32,
    )

    parts = [
        note_hist,
        duration_hist_pc,
        stable_hist,
        cadence_hist,
        tonic_profile_rel,
        trans.reshape(-1).astype(np.float32),
        step_hist,
        step_bigram.reshape(-1).astype(np.float32),
        note_duration_hist,
    ]
    if cfg.function_features:
        parts.append(_function_block(notes, intervals, durations, groups, cfg))
    parts.append(summary)  # keep summary last: the empty-notes path indexes from the end
    vec = np.concatenate(parts, axis=0).astype(np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)


def _stable_track_seed(track_path: str, seed: int) -> int:
    digest = hashlib.md5(track_path.encode("utf-8")).hexdigest()[:8]
    return (seed + int(digest, 16)) % (2**31 - 1)


def _compute_track_notes(track_path: str, cfg: MelodicFeatureConfig, mode: str, seed: int) -> Tuple[List[List[NoteEvent]], List[Dict[str, float]]]:
    try:
        duration = float(librosa.get_duration(path=track_path))
    except Exception as exc:
        warnings.warn(f"Could not read audio duration; using zero features for {track_path}: {exc}", RuntimeWarning)
        return [], []
    total_samples = int(duration * cfg.sample_rate)
    if total_samples <= 0:
        return [], []

    seg_len = int(cfg.segment_seconds * cfg.sample_rate)
    starts = _segment_starts(total_samples, cfg.sample_rate, cfg.segment_seconds, cfg.num_segments, mode, seed)
    segment_note_lists: List[List[NoteEvent]] = []
    metas: List[Dict[str, float]] = []
    frame_offset = 0
    for start in starts:
        try:
            segment = _load_segment(track_path, start, cfg)
        except Exception as exc:
            warnings.warn(
                f"Could not decode segment at {start / cfg.sample_rate:.0f}s; skipping it for {track_path}: {exc}",
                RuntimeWarning,
            )
            segment = np.zeros(0, dtype=np.float32)
        if segment.size == 0:
            segment_note_lists.append([])
            metas.append({"voiced_ratio": 0.0, "harmonic_ratio": 0.0})
            frame_offset += seg_len // cfg.hop_length + 1
            continue
        notes, meta = _extract_segment_notes(segment, cfg)
        metas.append(meta)
        segment_note_lists.append(
            [
                NoteEvent(
                    start=note.start + frame_offset,
                    end=note.end + frame_offset,
                    pc=note.pc,
                    midi_mean=note.midi_mean,
                    duration_frames=note.duration_frames,
                )
                for note in notes
            ]
        )
        frame_offset += int(round(len(segment) / cfg.hop_length)) + 1
    return segment_note_lists, metas


def extract_track_feature(track_path: str, cfg: MelodicFeatureConfig, mode: str, seed: int, cache_dir: str) -> np.ndarray:
    sig = cfg_signature(cfg)
    suffix = f"track-{mode}-seed{seed}"
    cached = load_cached_track_features(cache_dir, track_path, sig, suffix)
    if cached is not None:
        return cached

    notes_sig = notes_signature(cfg)
    notes_suffix = f"notes-{mode}-seed{seed}"
    arrays = load_cached_track_notes(cache_dir, track_path, notes_sig, notes_suffix)
    if arrays is not None:
        segment_note_lists, metas = _arrays_to_notes(arrays)
    else:
        segment_note_lists, metas = _compute_track_notes(track_path, cfg, mode, seed)
        save_cached_track_notes(cache_dir, track_path, notes_sig, notes_suffix, _notes_to_arrays(segment_note_lists, metas))

    all_notes: List[NoteEvent] = [note for seg_notes in segment_note_lists for note in seg_notes]
    tonic_override = None
    if cfg.tonic_strategy == "vote":
        tonic_override = vote_track_tonic(segment_note_lists, cfg)

    out = build_melodic_vector(all_notes, cfg, metas, tonic_override=tonic_override)
    save_cached_track_features(cache_dir, track_path, sig, suffix, out)
    return out


def _feature_job(job: Tuple[int, str, MelodicFeatureConfig, str, int, str]) -> Tuple[int, np.ndarray]:
    idx, track_path, cfg, mode, seed, cache_dir = job
    feat = extract_track_feature(track_path, cfg=cfg, mode=mode, seed=seed, cache_dir=cache_dir)
    return idx, feat


def build_track_matrix(
    tracks: List[Track],
    cfg: MelodicFeatureConfig,
    mode: str,
    seed: int,
    cache_dir: str,
    label_to_idx: Dict[str, int],
    progress_label: str | None = None,
    num_workers: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    if not tracks:
        return np.empty((0, feature_dim(cfg)), dtype=np.float32), np.empty((0,), dtype=np.int64)

    jobs: List[Tuple[int, str, MelodicFeatureConfig, str, int, str]] = []
    y = []
    for idx, t in enumerate(tracks):
        track_seed = _stable_track_seed(t.path, seed)
        jobs.append((idx, t.path, cfg, mode, track_seed, cache_dir))
        y.append(label_to_idx[t.label])

    if num_workers <= 1:
        X = []
        iterator = tqdm(jobs, desc=progress_label) if progress_label else jobs
        for job in iterator:
            _, feat = _feature_job(job)
            X.append(feat)
        return np.vstack(X), np.array(y, dtype=np.int64)

    results: List[np.ndarray | None] = [None] * len(jobs)
    # A fresh pool per chunk keeps worker heap growth across many pyin runs
    # bounded. Don't swap this for max_tasks_per_child: with all jobs
    # pre-submitted it deadlocks at the first worker recycle (Python 3.13).
    ctx = multiprocessing.get_context("spawn")
    chunk_size = max(1, num_workers) * 8
    pbar = tqdm(total=len(jobs), desc=progress_label) if progress_label else None
    for chunk_start in range(0, len(jobs), chunk_size):
        chunk = jobs[chunk_start : chunk_start + chunk_size]
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as ex:
            fut_map = {ex.submit(_feature_job, job): job[1] for job in chunk}
            for fut in as_completed(fut_map):
                path = fut_map[fut]
                try:
                    out_idx, feat = fut.result()
                except Exception as exc:
                    raise RuntimeError(f"Feature extraction failed for: {path}") from exc
                results[out_idx] = feat
                if pbar is not None:
                    pbar.update(1)
    if pbar is not None:
        pbar.close()

    X_final = [x for x in results if x is not None]
    return np.vstack(X_final), np.array(y, dtype=np.int64)
