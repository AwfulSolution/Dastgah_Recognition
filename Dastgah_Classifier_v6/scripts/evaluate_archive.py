"""Evaluate against a folder-per-dastgah archive of real recordings.

Expects a directory whose subfolders are named for modal classes::

    Training_Data_wav/
      Chahargah/*.wav
      Homayun/*.wav
      ...

Unlike the radif corpora this is commercial performance audio: ensembles,
modulation, ornament and studio processing. It is the most realistic test
available here, and the only one exercising the pYIN front end end to end.

    python scripts/evaluate_archive.py /path/to/Training_Data_wav

Two accuracies are reported. *Open set* lets the classifier choose among all 13
modal classes. *Closed set* restricts it to the classes the archive actually
contains, which is the fair number when the candidate set is known in advance.
"""

from __future__ import annotations

import argparse
import warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH, MIN_EVENTS_FOR_NOTE_HISTOGRAM
from dastgah.core.audio import (
    note_events,
    note_histogram,
    pitch_histogram,
    track_pitch,
    transition_matrix,
)
from dastgah.core.analyze import _blend_prior, _tonic_prior
from dastgah.core.classify import DEFAULT_CONFIG, classify
from dastgah.core.forud import find_foruds
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates
from dastgah.theory import MODAL_CLASSES_BY_KEY, modal_class_from_name


def folder_to_key(name: str) -> str | None:
    """Resolve a dataset folder name to a modal class key."""
    modal = modal_class_from_name(name)
    return modal.key if modal else None


def analyse_excerpt(
    path: Path, seconds: float | None, position: str = "middle"
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray | None]:
    """Features for one recording, excerpted from the given position.

    Position matters more than it looks: the *forud*, the cadence that fixes the
    tonic, falls at the end of a performance, so a middle excerpt often contains
    none at all.
    """
    import librosa

    offset = 0.0
    if seconds is not None:
        total = librosa.get_duration(path=str(path))
        if position == "end":
            offset = max(0.0, total - seconds)
        elif position == "middle":
            offset = max(0.0, (total - seconds) / 2)

    y, sr = librosa.load(
        str(path), sr=22050, mono=True, offset=offset, duration=seconds
    )
    track = track_pitch(y, sr)
    events = note_events(track)

    histogram = note_histogram(events)
    if len(events) < MIN_EVENTS_FOR_NOTE_HISTOGRAM or histogram.sum() <= 0:
        histogram = pitch_histogram(track)
    foruds = find_foruds(events, total_duration=track.duration)
    prior = _blend_prior(
        _tonic_prior(foruds, track.duration, DEFAULT_CONFIG), histogram, DEFAULT_CONFIG
    )
    return histogram, transition_matrix(events), track.tuning_concentration, prior


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--templates", type=Path, default=DEFAULT_TEMPLATE_PATH)
    parser.add_argument(
        "--seconds",
        type=float,
        default=90.0,
        help="excerpt length from the middle; 0 analyses the whole file",
    )
    parser.add_argument("--limit", type=int, default=0, help="max files per class")
    parser.add_argument(
        "--all-classes",
        action="store_true",
        help="rank all 13 classes instead of folding each avaz into its mother",
    )
    parser.add_argument(
        "--position",
        choices=("start", "middle", "end"),
        default="end",
        help=(
            "where in each recording to take the excerpt. Defaults to the end, "
            "which is where the forud falls and the closest proxy for the whole-"
            "file analysis the library actually performs; a middle excerpt "
            "measures a configuration nothing uses and understates accuracy by "
            "roughly seven points"
        ),
    )
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    if not args.archive.is_dir():
        parser.error(f"not a directory: {args.archive}")
    seconds = None if args.seconds <= 0 else args.seconds

    templates = load_templates(args.templates)
    answer_space = None if args.all_classes else DASTGAHS_WITH_AUDIO
    files: list[tuple[str, Path]] = []
    for folder in sorted(p for p in args.archive.iterdir() if p.is_dir()):
        key = folder_to_key(folder.name)
        if key is None:
            print(f"  skipping unrecognised folder: {folder.name}")
            continue
        found = sorted(
            path
            for pattern in ("*.wav", "*.flac", "*.aiff", "*.aif", "*.mp3", "*.m4a")
            for path in folder.rglob(pattern)
        )
        files.extend((key, p) for p in (found[: args.limit] if args.limit else found))

    if not files:
        parser.error("no recordings found")

    # When the classifier answers with dastgahs, the ground truth has to be
    # folded the same way: a recording of Abu'ata is a correct answer of Shur.
    if answer_space is not None:
        files = [
            (MODAL_CLASSES_BY_KEY[k].parent or k, p)
            for k, p in files
            if (MODAL_CLASSES_BY_KEY[k].parent or k) in answer_space
        ]

    present = sorted({k for k, _ in files})
    family_of = {k: (t.family or k) for k, t in templates.items()}
    open_hits = closed_hits = family_hits = 0
    ranks: list[int] = []
    confusions: Counter[tuple[str, str]] = Counter()
    per_class: dict[str, list[int]] = defaultdict(list)
    concentrations: list[float] = []
    failures = 0

    for index, (truth, path) in enumerate(files, start=1):
        try:
            histogram, transitions, concentration, prior = analyse_excerpt(
                path, seconds, args.position
            )
            if histogram.sum() <= 0:
                failures += 1
                continue
            result = classify(
                histogram,
                templates,
                transitions=transitions,
                tonic_prior=prior,
                config=DEFAULT_CONFIG,
            )
            ranked = (
                result.ranked_classes()
                if answer_space is None
                else result.ranked_dastgahs(answer_space)
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  failed: {path.name}: {exc}")
            failures += 1
            continue

        order = [k for k, _ in ranked]
        predicted = order[0]
        closed = next((k for k in order if k in present), order[0])

        open_hits += predicted == truth
        closed_hits += closed == truth
        family_hits += family_of.get(predicted, predicted) == family_of.get(truth, truth)
        ranks.append(order.index(truth) + 1)
        confusions[(truth, closed)] += 1
        per_class[truth].append(int(closed == truth))
        concentrations.append(concentration)

        if index % 25 == 0:
            print(f"  ...{index}/{len(files)}")

    scored = len(ranks)
    print(f"\nArchive evaluation: {scored} recordings, {len(present)} classes present")
    if failures:
        print(f"  ({failures} files could not be analysed)")
    excerpt = "whole file" if seconds is None else f"{seconds:.0f}s from the {args.position}"
    print(f"  excerpt: {excerpt}\n")
    print(f"  open-set accuracy    {100 * open_hits / scored:5.1f}%   (all 13 classes, chance 7.7%)")
    print(f"  closed-set accuracy  {100 * closed_hits / scored:5.1f}%   "
          f"({len(present)} classes, chance {100 / len(present):.1f}%)")
    print(f"  top-3 (closed)       {100 * sum(1 for r in ranks if r <= 3) / scored:5.1f}%")
    n_families = len({family_of.get(k, k) for k in present})
    print(f"  FAMILY accuracy      {100 * family_hits / scored:5.1f}%   "
          f"({n_families} families present, chance {100 / n_families:.1f}%)")
    print(f"  mean rank of truth   {np.mean(ranks):5.2f} of 13")
    print(f"  mean grid fit        {np.mean(concentrations):5.2f}")

    print("\nPer-class recall (closed set):")
    for key in present:
        hits = per_class[key]
        print(
            f"  {MODAL_CLASSES_BY_KEY[key].display:<26}"
            f"{100 * sum(hits) / len(hits):5.1f}%  (n={len(hits)})"
        )

    print("\nTop confusions (closed set):")
    for (truth, predicted), count in confusions.most_common():
        if truth != predicted and count >= 3:
            print(
                f"  {MODAL_CLASSES_BY_KEY[truth].name:<16} -> "
                f"{MODAL_CLASSES_BY_KEY[predicted].name:<16} {count}x"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
