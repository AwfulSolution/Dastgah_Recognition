"""Fetch the KUG Dastgāhi Corpus (KDC) as an external evaluation set.

KDC is a collection of solo recordings made for computational research at the
Institute of Ethnomusicology, KUG, and published through their PHAIDRA
repository under CC BY-NC-ND 4.0. It matters here because its performers
appear nowhere in this project's training corpus, which makes it a test of
whether the features describe the mode or the performer.

Two details decide what is usable:

  * the collection mixes full performances with isolated šāhed/ist reference
    tones (a couple of seconds each, titled "... (Šāhed)" / "... (Ist)").
    Only performances are kept.
  * it covers all 7 dastgāhs and 5 āvāzes; only the six dastgāhs this project
    has labels for are downloaded by default. Pass --include_avaz to also
    fetch the āvāz recordings, which are useful for asking whether the model
    places family relatives with their parent dastgāh.

Audio is written to data/kdc_audio/<label>/ (gitignored) and a manifest to
data/kdc_manifest.json. Nothing is redistributed by this repository.
"""

import argparse
import json
import os
import re
import ssl
import time
import urllib.request

import certifi

ROOT = os.path.dirname(os.path.abspath(__file__))
COLLECTION = "o:127195"
INFO = "https://phaidra.kug.ac.at/api/object/{}/info"
META = "https://phaidra.kug.ac.at/api/object/{}/uwmetadata"
OCTETS = "https://phaidra.kug.ac.at/api/object/{}/octets"

# The framework Python build on some systems ships no usable CA bundle.
SSL_CTX = ssl.create_default_context(cafile=certifi.where())

DASTGAH = {
    "Čāhārgāh": "Chahargah", "Homāyun": "Homayun", "Māhur": "Mahur",
    "Navā": "Nava", "Segāh": "Segah", "Šur": "Shur",
}
AVAZ = {
    "Abuatā": "Abuata", "Afšāri": "Afshari", "Bayāt-e Tork": "Bayat-e Tork",
    "Dašti": "Dashti", "Esfahān": "Esfahan",
}
EXT = {"audio/flac": ".flac", "audio/x-flac": ".flac", "audio/wav": ".wav"}


def get_json(url):
    req = urllib.request.Request(url, headers={"User-Agent": "dastgah-research/1.0"})
    with urllib.request.urlopen(req, timeout=90, context=SSL_CTX) as r:
        return json.load(r)


def flatten(node, out):
    if isinstance(node, list):
        for n in node:
            flatten(n, out)
        return
    if not isinstance(node, dict):
        return
    name, val = node.get("xmlname", ""), node.get("ui_value")
    if name and val:
        out.setdefault(name, []).append(val)
    for c in node.get("children", []) or []:
        flatten(c, out)


def mode_of(alt_title):
    return re.split(r"\s+Dar", alt_title.strip())[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio_dir", default=os.path.join(ROOT, "data", "kdc_audio"))
    ap.add_argument("--manifest", default=os.path.join(ROOT, "data", "kdc_manifest.json"))
    ap.add_argument("--include_avaz", action="store_true",
                    help="also fetch āvāz recordings (labels outside this project's six classes)")
    ap.add_argument("--delay", type=float, default=0.4, help="seconds between requests")
    args = ap.parse_args()

    labels = dict(DASTGAH)
    if args.include_avaz:
        labels.update(AVAZ)

    print(f"listing collection {COLLECTION} ...", flush=True)
    ids = get_json(INFO.format(COLLECTION))["info"]["haspart"]
    print(f"{len(ids)} objects", flush=True)

    manifest = []
    for i, oid in enumerate(ids):
        try:
            f = {}
            flatten(get_json(META.format(oid))["metadata"]["uwmetadata"], f)
        except Exception as exc:
            print(f"  !! metadata {oid}: {exc}", flush=True)
            continue
        alt = (f.get("alt_title") or [""])[0]
        if not alt or "(Šāhed)" in alt or "(Ist)" in alt:
            continue  # isolated reference tones, not performances
        label = labels.get(mode_of(alt))
        if not label:
            continue

        fmt = (f.get("format") or [""])[0]
        outdir = os.path.join(args.audio_dir, label)
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"{oid.replace(':', '_')}{EXT.get(fmt, '.flac')}")
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            try:
                req = urllib.request.Request(OCTETS.format(oid),
                                             headers={"User-Agent": "dastgah-research/1.0"})
                with urllib.request.urlopen(req, timeout=300, context=SSL_CTX) as r:
                    data = r.read()
                with open(path, "wb") as fh:
                    fh.write(data)
            except Exception as exc:
                print(f"  !! audio {oid}: {exc}", flush=True)
                continue
            time.sleep(args.delay)
        performer = " ".join(f"{(f.get('firstname') or [''])[0]} {(f.get('lastname') or [''])[0]}".split())
        manifest.append({"path": path, "label": label, "kdc_id": oid,
                         "title": alt, "performer": performer})
        if i % 25 == 0:
            print(f"  {i}/{len(ids)} scanned, {len(manifest)} kept", flush=True)

    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)
    with open(args.manifest, "w") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
    counts = {}
    for m in manifest:
        counts[m["label"]] = counts.get(m["label"], 0) + 1
    print(f"\nwrote {args.manifest}: {len(manifest)} recordings")
    print("by label:", dict(sorted(counts.items())))


if __name__ == "__main__":
    main()
