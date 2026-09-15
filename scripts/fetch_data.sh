#!/usr/bin/env bash
# Fetch the reference corpora into data/raw/.
#
#   ./scripts/fetch_data.sh [radif|irma|all]
#
# Neither corpus is vendored into this repository: the Radif Corpus is CC-BY-4.0
# and IRMA is CC-BY-NC, so both are downloaded from their sources and cited.
set -euo pipefail

RAW="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/data/raw"
TARGET="${1:-all}"
mkdir -p "$RAW"

fetch_radif() {
  # Radif Corpus: 229 gushehs of Mirza Abdollah's radif as MIDI, MusicXML and
  # CSV. This is what dastgah/data/templates.json is built from.
  if [ -d "$RAW/radif_corpus/RadifCorpus" ]; then
    echo "radif: already present, skipping"
    return
  fi
  echo "radif: downloading (17 MB, Zenodo 10.5281/zenodo.15742125, CC-BY-4.0)"
  curl -fL --retry 3 -o "$RAW/RadifCorpus.zip" \
    "https://zenodo.org/api/records/15742125/files/RadifCorpus.zip/content"
  unzip -q -o "$RAW/RadifCorpus.zip" -d "$RAW/radif_corpus"
  rm -f "$RAW/RadifCorpus.zip"
  echo "radif: $(find "$RAW/radif_corpus/CSV" -name '*.csv' 2>/dev/null | wc -l) gusheh CSVs"
}

fetch_irma() {
  # IRMA: per-gusheh f0 and energy contours extracted from recordings, plus
  # MIDI and scanned scores, labelled by dastgah and gusheh. The repository is
  # ~2.5 GB, almost all of it scan images, so fetch only the data files.
  if [ -d "$RAW/irma/.git" ]; then
    echo "irma: repository present, completing checkout"
  else
    echo "irma: cloning metadata only (blobless)"
    git clone --filter=blob:none --no-checkout --depth 1 \
      https://github.com/SepiSha/irma-dataset.git "$RAW/irma"
  fi
  git -C "$RAW/irma" sparse-checkout init --no-cone
  git -C "$RAW/irma" sparse-checkout set '*.csv' '*.mid' '*.midi' '*.xlsx' '*.md' 'LICENSE'
  git -C "$RAW/irma" checkout
  echo "irma: $(find "$RAW/irma" -name '*.csv' -not -path '*/.git/*' | wc -l) CSVs, "\
       "$(find "$RAW/irma" -name '*.mid*' -not -path '*/.git/*' | wc -l) MIDI"
}

case "$TARGET" in
  radif) fetch_radif ;;
  irma)  fetch_irma ;;
  all)   fetch_radif; fetch_irma ;;
  *)     echo "usage: $0 [radif|irma|all]" >&2; exit 2 ;;
esac

echo "done. rebuild templates with: python scripts/build_templates.py"
