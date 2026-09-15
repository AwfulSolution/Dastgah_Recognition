"""Rebuild the template artefacts from the Radif Corpus.

Writes both files the library ships with:

``dastgah/data/templates.json``
    one modal template per dastgah and avaz;
``dastgah/data/gushehs.json``
    one template per gusheh, which depends on the modal tonics above and so is
    built in the same pass.

Both are committed to the repository, so this only needs running when the corpus
or the template code changes. Output is deterministic: rebuilding from an
unchanged corpus reproduces the committed files byte for byte.

    python scripts/build_templates.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

from dastgah.radif.gusheh import build_gusheh_templates, save_gusheh_templates
from dastgah.radif.parse import load_corpus
from dastgah.radif.templates import build_templates, describe, save_templates

DEFAULT_CORPUS = Path("data/raw/radif_corpus/RadifCorpus/CSV")
DEFAULT_TEMPLATES = Path("dastgah/data/templates.json")
DEFAULT_GUSHEHS = Path("dastgah/data/gushehs.json")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--output", type=Path, default=DEFAULT_TEMPLATES)
    parser.add_argument("--gushehs", type=Path, default=DEFAULT_GUSHEHS)
    parser.add_argument(
        "--quiet", action="store_true", help="suppress the per-mode summary"
    )
    args = parser.parse_args()

    if not args.corpus.exists():
        parser.error(f"corpus not found at {args.corpus}; run ./scripts/fetch_data.sh radif")

    corpus = load_corpus(args.corpus)
    templates = build_templates(corpus)
    save_templates(templates, args.output)

    gushehs = build_gusheh_templates(
        corpus, {key: template.tonic_pc for key, template in templates.items()}
    )
    save_gusheh_templates(gushehs, args.gushehs)

    print(f"{len(corpus)} gushehs from {args.corpus}")
    print(f"  {len(templates)} modal templates -> {args.output}")
    print(
        f"  {sum(len(g) for g in gushehs.values())} gusheh templates "
        f"across {len(gushehs)} modes -> {args.gushehs}"
    )
    if not args.quiet:
        print("\n(* marks a quarter-tone degree)\n")
        for template in templates.values():
            print(describe(template))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
