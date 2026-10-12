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
  local zip="$RAW/RadifCorpus.zip"
  # Zenodo resets long connections often enough that a single attempt is not
  # reliable; --retry-all-errors covers the mid-transfer reset that plain
  # --retry does not. A partial file is removed so a retry cannot extract it.
  if ! curl -fL --no-progress-meter \
       --retry 5 --retry-delay 2 --retry-all-errors --continue-at - \
       -o "$zip" \
       "https://zenodo.org/api/records/15742125/files/RadifCorpus.zip/content"
  then
    rm -f "$zip"
    echo "radif: download failed. Retry, or fetch the archive by hand from" >&2
    echo "       https://zenodo.org/records/15742125 into $RAW/" >&2
    return 1
  fi
  if ! unzip -tq "$zip" >/dev/null 2>&1; then
    rm -f "$zip"
    echo "radif: the downloaded archive is incomplete or corrupt; re-run" >&2
    return 1
  fi
  unzip -q -o "$zip" -d "$RAW/radif_corpus"
  rm -f "$zip"
  # The archive extracts into a RadifCorpus/ subdirectory; counting the level
  # above it silently reported zero even on a good download, which made a
  # successful fetch indistinguishable from a failed one.
  local csvs
  csvs=$(find "$RAW/radif_corpus/RadifCorpus/CSV" -name '*.csv' 2>/dev/null | wc -l | tr -d ' ')
  if [ "$csvs" -lt 200 ]; then
    echo "radif: expected ~229 gusheh CSVs, found $csvs — the download looks incomplete" >&2
    return 1
  fi
  echo "radif: $csvs gusheh CSVs"
}

fetch_irma() {
  # IRMA: per-gusheh f0 and energy contours extracted from recordings, plus
  # MIDI and scanned scores, labelled by dastgah and gusheh. The repository is
  # ~2.5 GB, almost all of it scan images, so fetch only the data files.
  if [ -d "$RAW/irma/.git" ]; then
    echo "irma: repository present, completing checkout"
  else
    echo "irma: cloning metadata only (blobless)"
    if ! git clone --filter=blob:none --no-checkout --depth 1 \
         https://github.com/SepiSha/irma-dataset.git "$RAW/irma"; then
      rm -rf "$RAW/irma"
      echo "irma: clone failed; re-run to resume" >&2
      return 1
    fi
  fi
  git -C "$RAW/irma" sparse-checkout init --no-cone
  git -C "$RAW/irma" sparse-checkout set '*.csv' '*.mid' '*.midi' '*.xlsx' '*.md' 'LICENSE'
  git -C "$RAW/irma" checkout
  local icsv imidi
  icsv=$(find "$RAW/irma" -name '*.csv' -not -path '*/.git/*' | wc -l | tr -d ' ')
  imidi=$(find "$RAW/irma" -name '*.mid*' -not -path '*/.git/*' | wc -l | tr -d ' ')
  if [ "$icsv" -lt 100 ]; then
    echo "irma: expected ~288 CSVs, found $icsv — the checkout looks incomplete" >&2
    return 1
  fi
  echo "irma: $icsv CSVs, $imidi MIDI"
}

case "$TARGET" in
  radif) fetch_radif ;;
  irma)  fetch_irma ;;
  all)   fetch_radif; fetch_irma ;;
  *)     echo "usage: $0 [radif|irma|all]" >&2; exit 2 ;;
esac

echo "done. rebuild templates with: python scripts/build_templates.py"
