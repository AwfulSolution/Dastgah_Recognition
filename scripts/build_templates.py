"""Rebuild dastgah/data/templates.json from the Radif Corpus.

    python scripts/build_templates.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dastgah.radif.parse import load_corpus
from dastgah.radif.templates import build_templates, describe, save_templates

DEFAULT_CORPUS = Path("data/raw/radif_corpus/RadifCorpus/CSV")
DEFAULT_OUTPUT = Path("dastgah/data/templates.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.corpus.exists():
        parser.error(f"corpus not found at {args.corpus}; run ./scripts/fetch_data.sh radif")

    corpus = load_corpus(args.corpus)
    templates = build_templates(corpus)
    save_templates(templates, args.output)

    print(f"{len(corpus)} gushehs -> {len(templates)} templates -> {args.output}")
    print("(* marks a quarter-tone degree)\n")
    for template in templates.values():
        print(describe(template))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
