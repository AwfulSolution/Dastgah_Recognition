"""Cache pitch features for a corpus so evaluations need not redo pYIN.

Writes to ``data/cache/<name>.pkl``, which is gitignored but survives between
sessions — unlike a scratchpad. Resumable: rerunning skips what is already
cached, so an interrupted run costs only the remainder.

    python scripts/extract_features.py nava /path/to/Data --layout nava
    python scripts/extract_features.py kdc data/raw/kdc --layout folder
"""

from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import numpy as np

from dastgah.core.audio import (
    load_audio,
    note_events,
    note_histogram,
    track_pitch,
    transition_matrix,
)
from dastgah.core.forud import find_foruds

AUDIO_SUFFIXES = {".wav", ".flac", ".mp3", ".aiff", ".aif", ".m4a", ".ogg"}

#: Nava encodes instrument_dastgah_artist_track in the filename. The dastgah
#: digits were identified by ear by a Persian-speaking listener and corroborated
#: on four of the seven groups by a second; a third listener read groups 1, 3, 4
#: and 5 differently, so those labels carry real uncertainty.
NAVA_DASTGAH = {
    "0": "shur", "1": "segah", "2": "mahur", "3": "homayun",
    "4": "rast_panjgah", "5": "nava", "6": "chahargah",
}


def _progress(done: int, total: int, cached: int, started: float) -> None:
    """One self-overwriting line, so a long run is watchable."""
    fraction = done / total
    filled = int(42 * fraction)
    elapsed = time.monotonic() - started
    rate = done / elapsed if elapsed > 0 else 0.0
    remaining = (total - done) / rate if rate > 0 else 0.0
    bar = "#" * filled + "." * (42 - filled)
    print(
        f"\r[{bar}] {100 * fraction:5.1f}%  {done}/{total}  "
        f"cached {cached}  {remaining / 60:4.1f} min left   ",
        end="",
        flush=True,
    )
    if done >= total:
        print()


def describe(path: Path, layout: str) -> dict:
    """Pull whatever labels the corpus layout encodes."""
    if layout == "nava":
        instrument, dastgah, artist, _track = path.stem.split("_")[:4]
        return {
            "truth": NAVA_DASTGAH[dastgah],
            "dastgah_code": dastgah,
            "instrument": instrument,
            "artist": artist,
        }
    return {"truth": path.parent.name.lower()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    parser.add_argument("root", type=Path)
    parser.add_argument("--layout", choices=("nava", "folder"), default="folder")
    parser.add_argument("--out", type=Path, default=Path("data/cache"))
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / f"{args.name}.pkl"

    cached: dict[str, dict] = {}
    if cache_path.exists():
        cached = {r["name"]: r for r in pickle.loads(cache_path.read_bytes())}
        print(f"resuming: {len(cached)} already cached")

    files = sorted(p for p in args.root.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES)
    print(f"{len(files)} audio files under {args.root}", flush=True)
    started = time.monotonic()

    for index, path in enumerate(files, start=1):
        if path.stem in cached:
            continue
        try:
            y, sr = load_audio(path)
            track = track_pitch(y, sr)
            events = note_events(track)
        except Exception as exc:  # noqa: BLE001 - a bad file should not stop the run
            print(f"  skip {path.name}: {exc}", flush=True)
            continue
        if len(events) < 8:
            continue
        histogram = note_histogram(events)
        if histogram.sum() <= 0:
            continue

        cached[path.stem] = {
            "name": path.stem,
            "path": str(path),
            "h": histogram,
            "B": transition_matrix(events),
            "foruds": find_foruds(events, total_duration=track.duration),
            "dur": track.duration,
            **describe(path, args.layout),
        }
        if index % 10 == 0 or index == len(files):
            _progress(index, len(files), len(cached), started)
        if index % 25 == 0:
            cache_path.write_bytes(pickle.dumps(list(cached.values())))

    cache_path.write_bytes(pickle.dumps(list(cached.values())))
    print(f"\n{len(cached)} records -> {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
