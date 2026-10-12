# Dastgāh Classifier

Identifies the **dastgāh** or **āvāz** of a Persian classical recording, along with
its tonic, its quarter-tone degrees, and where it changes mode.

Classification runs entirely on pitch: an f0 contour is folded into a 24-tone
equal-tempered pitch-class histogram and matched against scale templates derived
from Mirza Abdollah's radif. Nothing is trained on audio, so the system works on
tār, setār, ney, kamānche, santur or voice without having seen any of them.

## Current accuracy

The library answers over **six dastgāhs** — Shūr, Navā, Homāyūn, Māhūr,
Chahārgāh and Segāh — with each āvāz folded into its mother, so a recording of
Dashti is a correct answer of Shūr. Rāst-Panjgāh is deliberately outside the
answer space; [docs/data-notes.md](docs/data-notes.md) records why, and removing
it is worth 11.5 points.

Every figure below is the shipped configuration, untrained.

**On Nava — the broadest evidence available.** 1,568 in-scope recordings by
**37 artists** across five instruments (BabaAli & Mohammadi, University of
Tehran; obtained by request). Nothing in the system was tuned against it: the
templates come from notation and every weight was fixed before Nava arrived.

| Metric | Result |
| --- | --- |
| Accuracy over six dastgāhs | **74.9%** (chance 16.7%) |
| Macro average over classes | 74.6% |
| Truth in the top two | 89.4% |

Per-class: Māhūr 86.8%, Segāh 80.6%, Homāyūn 76.7%, Shūr 74.1%,
Chahārgāh 73.4%, Navā 56.2%.

**Recording length dominates everything else.** The same classifier, same
parameters, on the same corpus split by duration:

| | recordings | accuracy |
| --- | --- | --- |
| all of Nava (median 75 s) | 1,568 | 74.9% |
| Nava over three minutes | 218 | **89.0%** |

A 75-second excerpt does not contain enough of a performance to identify its
mode. Quote 74.9% for short excerpts and 89.0% for whole performances; one
number for both would misrepresent either.

**On other corpora**, where the recordings are shorter, the performers fewer,
or both: IRMA's Karimi-radif contours score 69.2% over 130 in-scope items; KDC's
189 in-scope recordings score 57.7%; sixteen complete Shajarian radif
performances score 50.0% — those last are 10 to 30 minutes each and traverse
many gushehs, which is the hardest case for a single pooled histogram rather
than the easiest.

**On the notated radif itself.** Leave-one-out over the 229 gushehs: 60.3%
across 13 classes, 69.4% at 7. An upper bound rather than a forecast, since the
weights were tuned on that split.

**Training does not improve any of this.** Fourteen configurations were fitted
and measured; all improve the corpus they were fitted on and degrade every other
one. The notes record each.

## What it reports

Because a pitch-class profile separates mode *families* far more reliably than
the modes inside one, the classifier leads with the family and offers the mode as
a best guess within it:

```
Shūr group   confidence 89.2%
    covers Shūr, Bayāt-e Kord, Dashtī, Bayāt-e Tork, Abū'atā, Afshārī, Navā
  most likely Dastgāh-e Shūr (dastgah)   confidence 52.5%
  tonic  G @ 391.4 Hz   shahed +10 (C)
```

Chahārgāh and Segāh are families of one, so for those the two readings coincide
and only the mode is shown.

Families are derived at template-build time from the template geometry itself:
modes whose profiles exceed cosine 0.90 at their best rotational alignment are
merged transitively, and each family is keyed by its best-attested member. They
are confusion neighbourhoods, not an editorial taxonomy — the seven-member group
happens to coincide with the traditional Shūr family, but the four-member one
puts Homāyūn alongside Māhūr, which no theorist would.

| Evaluation | Exact mode | Family |
| --- | --- | --- |
| Unseen performers (KDC) | 53.3% | **64.3%** |
| Development archive | 74.1% | **83.8%** |
| Karimi radif (IRMA) | 40.3% | **78.5%** |

## Install

```bash
uv venv && uv pip install -e ".[api,dev]"
```

## Use

The web UI shows the classification, the modal probability ledger grouped by
family, the 24 quarter-tone scale degrees, a modal timeline, gusheh shortlists,
and a waveform you can play back — the detected mode and gusheh update as the
playhead crosses each segment, so a claimed modulation can be listened to rather
than taken on trust. Results export as JSON or as MusicXML with koron and sori
notated.

```bash
# command line
dastgah recording.wav
dastgah *.flac --json

# HTTP API
uvicorn dastgah.api.server:app --port 8000

# web UI (expects the API on :8000)
cd web && npm install && npm run dev
```

## Performance

Analysis runs at roughly **55x realtime** on an M4, measured end to end on real
recordings from 2.6 to 20.8 minutes (46x to 62x, median 56.6x). A ten-minute
upload takes about eleven seconds and a half-hour recording about thirty; pYIN
pitch tracking dominates, and the cost is close to linear in duration.

That is fast enough that the whole archive of 340 recordings, some 24 hours of
audio, evaluates in under half an hour.

## How it works

1. **f0 tracking** — pYIN over 70–1200 Hz.
2. **Tuning estimation** — the reference pitch is found per performance by
   maximising how tightly the contour concentrates on 24-TET bin centres.
   Persian ensembles do not tune to A440, and a misplaced grid destroys the
   koron/sori distinctions the whole method rests on. The reference is only
   identifiable modulo a quarter-tone, which is sufficient for gridding.
3. **Soft 24-TET binning** — each frame's weight is split between its two
   nearest quarter-tone bins, so vibrato and glissando contribute to both.
4. **Joint mode and tonic search** — every one of the 13 modes is scored against
   all 24 tonic hypotheses. The score combines pitch-class likelihood, a
   note-transition term, and a prior favouring sustained tonics.
5. **Windowed pass** — the recording is re-analysed in 12-second windows and
   neighbouring windows sharing a mode are merged, because extended radif
   performances modulate and one global histogram smears them together.

Three details carry most of the accuracy:

- **Score note events, not frames.** This is the single largest effect measured.
  A frame-level histogram of real audio has about 1.5 bits more entropy than the
  notation the templates come from — broader than *every* template — so the
  classifier degenerates into picking whichever template is most permissive. On
  the held-out set that collapsed 103 of 144 predictions onto Rāst-Panjgāh, for
  18.8% accuracy. Building the histogram from note events instead took it to
  **40.3%**.

- **Template sharpening.** Profiles are raised to the third power before
  scoring. Without it, modes with broad templates attract everything, since a
  permissive distribution assigns decent likelihood to any input. This took
  13-class accuracy from 46% to 55%.
- **Tonic ≠ most-played note.** The tonic (*ist*) is estimated as the consensus
  *final* note across a mode's gushehs. The most-sounded degree is usually the
  *shāhed*, a different scale degree, and both are reported.

## Data

Neither corpus is vendored; `./scripts/fetch_data.sh` downloads them.

- **[Radif Corpus](https://zenodo.org/records/15742125)** (CC-BY-4.0) — 229
  gushehs of Mirza Abdollah's radif as MIDI/MusicXML/CSV with quarter-tone
  annotation, transcribed after Dariush Talai. The templates are built from
  this. Pitch is read from the absolute `Pitch (quarter notes)` column, because
  the corpus' octave marks are relative to each piece's own register.
- **[IRMA](https://github.com/SepiSha/irma-dataset)** (CC-BY-NC) — 144 per-gusheh
  f0 and energy contours (4.6 h) extracted from recordings of the Karimi radif,
  plus MIDI, scanned scores and theoretical scale tables, labelled by dastgāh
  *and gusheh*. Used as the held-out evaluation set. The repository is ~2.5 GB,
  almost all scan images; the fetch script takes only the data files.

## Known limitations

- **Recording length is the single biggest factor.** 74.9% on Nava's
  75-second median, 89.0% on its recordings over three minutes. A short excerpt
  does not contain enough of a performance to identify its mode.
- **Tonic placement, not mode recognition, is the largest error source.** Given
  the true class's own tonic the classifier reaches 91.0%, so about 16 of the 25
  missing points are the tonic being put in the wrong place. Shūr and Navā share
  a pitch collection — aligned cosine 0.931 — and 80 of their 111 mutual errors
  sit exactly a fourth or a fifth away from the right tonic.
- **A pitch-class profile cannot separate every pair.** Even with the tonic
  given, 9.0 points remain. Chahārgāh confused with Homāyūn is 12% of all errors
  and two thirds of those are at the *same* tonic, so that pair is a genuine
  modal confusion rather than a rotation.
- **The 24-bin profile is the ceiling, not the parameters.** No learned
  classifier on those features beats the templates: 15-nearest-neighbours 70.1%,
  logistic regression 68.8%, 1-nearest-neighbour 67.7%, templates 74.9%, all
  artist-grouped. Fourteen fitted configurations were measured and none
  transferred between corpora.
- **Rāst-Panjgāh is out of scope.** 91.7% of its gushehs have a near-twin in
  another dastgāh and Panjgāh itself is pitch-indistinguishable from Navā's
  opening darāmad. Removing it is worth 11.5 points.
- **Monophonic assumption.** pYIN tracks one voice; ensemble recordings with
  independent simultaneous melodies degrade. One of Nava's five instruments
  scores 50.8% against 65-69% for the others, and three different models failed
  to close that gap, which points at pitch tracking rather than classification.
- **Āvāzes are answered as their mother dastgāh**, not in their own right.
  Folding beats dropping them, but an āvāz-specific answer is not available.
- **No corpus here is fully clean.** KDC and IRMA have both informed design
  decisions; the Shajarian radif and Nava's long recordings have not.

## Next steps

Ranked by measured headroom rather than by appeal.

1. **Tonic placement — up to 16 points.** The forud is what fixes the tonic, and
   the cadence prior currently carries a weight of 0.25 against pitch content.
   Both a weight sweep over the six-class answer space and better cadence
   detection target a deficit that has now been measured rather than guessed.
   Note the 91.0% is an oracle that consults the truth, so it bounds the prize
   rather than promising it.
2. **More audio per decision — up to 14 points.** 89.0% on recordings over three
   minutes is the same classifier on longer input. For short uploads, scoring
   several windows and combining them is untried.
3. **A feature that separates modes sharing a pitch collection — 9 points.**
   This is the hard residual and the one place a genuinely new representation is
   required. Scoring progression through the seyr was tried and failed: it reads
   order correctly but a performance advances through every dastgāh's seyr almost
   equally, because 70-92% of every dastgāh's gushehs have a near-twin elsewhere.
4. **Gusheh identification.** IRMA labels every contour with its gusheh, so the
   references exist, and it does not depend on solving 3.

What is already ruled out, with the evidence in
[docs/data-notes.md](docs/data-notes.md): refitting the global scoring weights
(worth nothing on four corpora), training per-mode parameters (improves the
fitted corpus, degrades every other), lowering the sharpening exponent, stacking
a learned model on the classifier output, and correcting one template in
isolation — the answer space is a softmax, so probability given to Navā is taken
from Shūr and Homāyūn.
