"""Command line entry point: ``dastgah <audio files>``."""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

from dastgah.core.analyze import DEFAULT_TEMPLATE_PATH, analyze
from dastgah.core.classify import ScoringConfig
from dastgah.radif.templates import DASTGAHS_WITH_AUDIO, load_templates, restrict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="dastgah",
        description="Classify the dastgah or avaz of a Persian classical recording.",
    )
    parser.add_argument("audio", nargs="+", type=Path, help="audio files to analyse")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of text")
    parser.add_argument(
        "--templates", type=Path, default=DEFAULT_TEMPLATE_PATH, help="template JSON path"
    )
    parser.add_argument(
        "--no-segments", action="store_true", help="skip the windowed timeline pass"
    )
    parser.add_argument(
        "--dastgahs-only",
        action="store_true",
        help=(
            "consider only the six dastgahs with audio evidence behind them, "
            "leaving out the avazes and Rast-Panjgah. Accuracy on those six is "
            "unchanged, since scores do not depend on which other modes are in "
            "the running; what changes is that no low-confidence avaz reading "
            "is offered, and the families reduce to Shur/Nava plus four "
            "singletons"
        ),
    )
    parser.add_argument("--sharpen", type=float, default=ScoringConfig.sharpen)
    parser.add_argument(
        "--transition-weight", type=float, default=ScoringConfig.transition_weight
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    warnings.filterwarnings("ignore", category=RuntimeWarning)

    templates = load_templates(args.templates)
    if args.dastgahs_only:
        templates = restrict(templates, DASTGAHS_WITH_AUDIO)
    config = ScoringConfig(
        sharpen=args.sharpen, transition_weight=args.transition_weight
    )

    results, failures = [], 0
    for path in args.audio:
        if not path.exists():
            print(f"error: no such file: {path}", file=sys.stderr)
            failures += 1
            continue
        try:
            result = analyze(
                path,
                templates=templates,
                config=config,
                with_segments=not args.no_segments,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user verbatim
            print(f"error: {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue

        results.append(result)
        if not args.json:
            print(f"\n{path.name}")
            print(result.summary())

    if args.json:
        print(json.dumps([r.to_dict() for r in results], indent=2, ensure_ascii=False))

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
