"""Album/performance grouping for leakage-safe splits.

Tracks from the same album or recording session share performer, instrument,
and recording conditions; letting them span train/eval inflates scores. Group
key preference:

1. Normalized ID3 album tag (via ffprobe), with per-track junk (leading track
   numbers, gusheh names) stripped.
2. Fallback: dastgah label + filename numbering style. This deliberately
   over-groups — merging two albums only costs stratification balance, while
   under-grouping leaks.

A final merge pass folds any group whose key mentions a known radif performer
into one group per performer: their multi-CD sets are single recording
sessions ripped inconsistently.
"""

import json
import os
import re
import subprocess
import unicodedata
from typing import Dict, List, Optional

# Radif performers whose sets appear under inconsistently-tagged albums.
# Extend as the dataset grows.
PERFORMER_MERGE_TOKENS = ["sarvestani", "karimi"]

_NUM_PREFIX = re.compile(r"^[\s\-_.()0-9]+")
_NON_WORD = re.compile(r"[^a-z؀-ۿ]+")


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NUM_PREFIX.sub("", text)
    text = _NON_WORD.sub(" ", text)
    return " ".join(text.split())


def read_album_tag(path: str) -> Optional[str]:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format_tags=album,album_artist,artist",
             "-of", "json", path],
            capture_output=True, text=True, timeout=30,
        )
        tags = json.loads(out.stdout or "{}").get("format", {}).get("tags", {})
    except (subprocess.SubprocessError, json.JSONDecodeError, OSError):
        return None
    tags = {k.lower(): v for k, v in tags.items()}
    album = tags.get("album", "").strip()
    artist = (tags.get("album_artist") or tags.get("artist") or "").strip()
    if not album:
        return None
    key = _normalize(album)
    if not key:
        return None
    artist_key = _normalize(artist)
    return f"{key}|{artist_key}" if artist_key else key


_STYLES = [
    ("num_dot", re.compile(r"^\d+\.\s")),
    ("num_dash", re.compile(r"^\d+\s*-\s*")),
    ("num_space", re.compile(r"^\d+\s")),
    ("word_num", re.compile(r"\(\d+\)\.[a-zA-Z0-9]+$")),
]


def _fallback_group(label: str, basename: str) -> str:
    for name, pat in _STYLES:
        if pat.search(basename):
            return f"fallback:{label}:{name}"
    return f"fallback:{label}:other"


def _merge_performers(key: str) -> str:
    flat = key.replace(" ", "")
    for token in PERFORMER_MERGE_TOKENS:
        if token in flat:
            return f"performer:{token}"
    return key


def build_groups(manifest: List[Dict[str, str]]) -> List[str]:
    """Return one group key per manifest entry (parallel list)."""
    groups: List[str] = []
    for entry in manifest:
        path, label = entry["path"], entry["label"]
        album = read_album_tag(path)
        key = album if album else _fallback_group(label, os.path.basename(path))
        groups.append(_merge_performers(key))
    return groups
