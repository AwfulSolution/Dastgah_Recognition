# Dastgāh Classifier

Identifies the **dastgāh** or **āvāz** of a Persian classical recording, along with
its tonic, its quarter-tone degrees, and where it changes mode.

Classification runs entirely on pitch: an f0 contour is folded into a 24-tone
equal-tempered pitch-class histogram and matched against scale templates derived
from Mirza Abdollah's radif. Nothing is trained on audio, so the system works on
tār, setār, ney, kamānche, santur or voice without having seen any of them.

## Current accuracy

Three evaluations. The spread between them is the important part: the
more the test material differs from the notated radif the templates come from,
the more the accuracy falls.

**On real recordings (the number that matters).** 340 commercial performances,
~24 hours, six dastgahs, supplied as a folder-per-class archive. Measured on
whole recordings, exactly as the library analyses them
(`python scripts/evaluate_archive.py <dir> --seconds 0`). The templates have
never seen this audio, so it is held out by construction:

| Metric | Result |
| --- | --- |
| Accuracy over the six dastgahs | **74.1%** (chance 16.7%) |
| Top-3 | **97.6%** |
| Mode family | **83.8%** (chance 25%) |
| Mean rank of the true dastgah | 1.36 of 13 |

Per-class: Māhūr 93.9%, Chahārgāh 86.0%, Homāyūn 79.6%, Segāh 77.1%, Shūr 60.3%,
Navā 53.6%.

Folding each avaz into its mother dastgah rather than reporting it separately is
worth **+6.7 points** over ranking all thirteen classes and taking the best of
the six (74.1% against 67.4%), and lifts Shūr from 12.8% to 60.3%. Shūr has five
avazes, and their evidence was previously scattered across classes nobody wanted
as an answer while Navā absorbed Shūr's territory.

**On the Karimi radif.** 144 IRMA pitch contours, 4.6 hours, scored against
templates built from the notated Mirza Abdollah radif — a different tradition and
medium (`python scripts/evaluate_irma.py`): 40.3% over 13 classes, 47.2% at 7,
top-3 65.3%, calibration error 0.096.

**On the radif itself.** Leave-one-out over the 229 notated gushehs
(`python scripts/evaluate.py`): 60.3% / 69.4%, tonic 65.5%. Treat this as an
upper bound, not a forecast — the scoring weights were tuned on that same split,
and real recordings are messier than notation.

Reported confidence is calibrated in both settings (43.6% mean confidence against
40.3% accuracy out of domain), so a 50% reading genuinely means a coin-flip
between the leading candidates.

### Accuracy is very uneven, and the pattern is structural

On real recordings, five of six dastgahs land between 57% and 86% — but Shūr
collapses to 10.3%, and 49 of its 78 recordings are called Navā:

| Class | Recall | | Class | Recall |
| --- | --- | --- | --- |
| Māhūr | 86.4% | | Segāh | 72.9% |
| Homāyūn | 75.5% | | Navā | 57.1% |
| Chahārgāh | 72.1% | | **Shūr** | **10.3%** |

That single confusion is 14% of the whole dataset, and it is not a defect of the
templates. Shūr and Navā **share a pitch collection, rotated by a fourth**:

```
Shūr on G : G  Ak  Bb  C  Dk  D  Eb  F
Navā on C : C  D   Eb  F  G   Ak  Bb
```

Their profiles reach cosine 0.931 once aligned. Any method scoring pitch content
against a tonic hypothesis can therefore place the tonic a fourth away and
recover an almost perfect match. Separating them needs note *function* — which
degree is the ist, which the shahed — not better pitch measurement.

This is not specific to Shūr. Comparing every pair of templates at its best
rotational alignment, 21 of 78 pairs exceed cosine 0.85, and treating pairs above
0.90 as indistinguishable collapses the 13 modes into **four components that
reproduce the traditional families** — with Chahārgāh and Segāh the only modes
standing alone, and the only two that classify well.

Asking for the family instead of the mode, on 168 balanced real recordings:

| Question | Accuracy |
| --- | --- |
| Exact mode (13 classes) | 48.2% |
| **Mode family (4 components)** | **72.6%** (chance 25%) |

**A tonic-relative pitch-class profile identifies the mode family, not the
mode.** Four fixes were tried against the Shūr case and all measured neutral or
worse; see [docs/data-notes.md](docs/data-notes.md) for the numbers.

On the Karimi radif, where the 13-class set includes the avazes, the split runs
along dastgāh versus āvāz instead:

| Dastgāh | | Āvāz | |
| --- | --- | --- | --- |
| Chahārgāh | 70% | Abū'atā | 22% |
| Homāyūn | 67% | Bayāt-e Tork | 17% |
| Navā | 64% | Dashtī | 11% |
| Rāst-Panjgāh | 64% | Afshārī | 0% |
| Shūr | 53% | Bayāt-e Esfahān | 0% |
| Māhūr | 45% | | |
| Segāh | 30% | | |

An āvāz shares its scale with its parent dastgāh and differs in melodic emphasis,
which a pitch-distribution method cannot see. **The dastgāhs are usable; the
āvāzes are not.** Fixing that needs melodic-contour modelling, not better
templates.

For context, published work reports around 86% F1 on a **7-class** task with a
**trained** model ([AzarNet](https://arxiv.org/pdf/1812.07017)). This is a
13-class untrained baseline and is meant as a floor to beat.

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
| Real recordings (archive) | 74.1% | **83.8%** |
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

- **Monophonic assumption.** pYIN tracks one voice. Ensemble recordings with
  independent simultaneous melodies will degrade.
- **Āvāz vs parent dastgāh.** An āvāz shares its scale with its parent and
  differs mainly in melodic emphasis, so Shūr/Dashtī-type confusions are
  inherent to a pitch-distribution method. The transition term helps a little;
  real separation needs melodic-contour modelling.
- **Āvāzes are not usable**, as above. Only the seven dastgāhs are.
- **Hyperparameters were tuned on the corpus leave-one-out split.** They were
  not re-tuned on IRMA — the same `sharpen=3.0` is best in both — but the
  corpus figure remains optimistic for that reason.
- **The held-out set is one tradition, one set of performers.** IRMA's contours
  are all Karimi radif; broader performer and instrument variety is untested.
- **Gusheh identification is not implemented.** Only modal segmentation is.
- **Monophonic pitch input.** IRMA's contours and pYIN both assume one voice.

## Next steps

Separating modes *within* a family was attempted with a learned model and did
not work: seven functional tonic features plus the template margin scored 56.0%
under leave-one-performer-out against a 58.2% majority-class baseline, and every
attempt to correct the underlying score bias merely moved the error from one
class to the other. The full account, including the performer confound that
shapes any experiment on this data, is in [docs/data-notes.md](docs/data-notes.md).

On current evidence the family layer is the ceiling for pitch-based
classification. Worthwhile directions from here:

1. **More performers.** Nine groups, with the label nearly determined by
   performer outside one of them, is too few to establish whether within-family
   signal generalises. This is the cheapest way to change the answer.
2. **Phrase-level rather than recording-level features.** Everything tried so far
   summarises a whole recording. The *forud* is a local event, and a 90-second
   excerpt from the middle of a performance rarely contains one.
3. **Gusheh identification** — IRMA labels every contour with its gusheh, so the
   references exist, and it does not depend on solving the within-family problem.
4. **The drone.** Persian ensemble practice often sounds the tonic continuously
   beneath the melody; monophonic f0 tracking discards it by design.
