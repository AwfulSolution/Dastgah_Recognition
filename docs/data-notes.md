# Corpus notes

Findings from working with the source data that are not obvious from the papers.

## Radif Corpus

- **229 gusheh CSVs, 43,441 notes**, matching the paper's count.
- Schema: `Microtonal pitch, Duration, Pitch (quarter notes), Interval, MIDI pitch number, MIDI Bend`.
- `Pitch (quarter notes)` is an absolute 24-TET integer equal to
  `2 * MIDI pitch + bend`, where a bend of `+2048` is a raised quarter-tone.
  Middle C is 120.
- Rows whose pitch cell is `[` or `]` delimit phrase structure and sound nothing.
- **Octave marks are register-relative.** `C+1` is 120 in some pieces and 144 in
  others, because the notated octave is anchored to each piece's own tessitura.
  Pitch *class* parsed from the label agrees with the absolute column for
  99.98% of notes, but absolute pitch does not. Always read the column.
- Accidental suffixes: `k` koron, `s` sori, `N` explicit natural, plus `#`/`b`.
- **Eight transcription errors**, all in two Shūr files (`Ghajar`,
  `Shahnaze kot ya asheqkosh`), where the label and the quarter-tone column
  disagree on pitch class. Left as-is; the column is treated as authoritative.
- Class sizes are very uneven: Māhūr 34, Chahārgāh 32, Shūr 29 … Afshārī 4,
  Bayāt-e Esfahān 5. This is the main driver of per-class accuracy.
- Tonics recovered from consensus final notes match theory: Shūr→G, Māhūr→C,
  Chahārgāh→C, Rāst-Panjgāh→F, Homāyūn→G, and **Segāh→A-koron**, a tonic that
  sits on a quarter-tone and cannot be represented in 12-TET at all.

## IRMA

- ~21,000 files, ~2.5 GB, but **18,700 of those are scanned score images**.
  Clone blobless and sparse-checkout the data files only; that yields 225 MB.
- The useful part is `*_pitch_*.csv` and `*_energy_*.csv` under each
  `*_Mp3csv_folder`: f0 and energy contours extracted from recordings, one pair
  per gusheh, with the dastgāh (`D1`–`D13`) and gusheh name in the filename.
- **144 pitch contours, 4.6 hours**, covering 12 of the 13 modes. All of them
  are from the **Karimi** tradition; the Mirza Abdollah tree carries scores and
  MIDI but no contours. Bayāt-e Kord (`D6`) has no contours at all.
- Contour CSVs are **headerless `time_seconds,f0_hz`** at roughly a 6 ms hop.
  **Unvoiced frames are omitted rather than zero-marked**, so gaps in the time
  column are silences — take frame durations from the time deltas, capped, or a
  single frame before a long rest is credited with the whole rest.
- **IRMA's `D1`–`D13` numbering is its own and does not match the Radif Corpus'
  `01`–`13` directories.** For example IRMA `D2` is Abū'atā while the corpus'
  `02` is Bayāt-e Kord. `D3` is listed as *Bayāt-e Zand*, the older name for
  Bayāt-e Tork. The mapping lives in `dastgah/radif/irma.py`.
- Also carries ~550 MIDI, ~294 Finale `.musx`, and 29 theoretical scale tables
  (Ja'farzadeh and Talai) as `.xlsx`. The tables are unused so far and are an
  independent check on the templates derived from the corpus.
- **No audio is vendored.** `AUDIO_SOURCES.md` points at recordings that must be
  obtained separately.

### Why this set is worth having

It is genuinely out of domain: the templates come from *notated* Mirza Abdollah,
these contours from *recorded* Karimi. That gap exposed the single biggest bug in
the pipeline — scoring frame-level histograms rather than note events. Frame
histograms of real audio average 3.83 bits of entropy against 2.37 for the
notation, which is broader than every template, so the classifier degenerated
into choosing the most permissive one (103 of 144 predictions went to
Rāst-Panjgāh). Nothing in the corpus leave-one-out could have surfaced this,
because notated observations are already sharp.


## The Shūr / Navā confusion

On 340 real recordings Shūr scores 10.3% recall and 49 of its 78 tracks are
called Navā — 14% of the entire dataset in one confusion, while every other
dastgāh sits between 57% and 86%.

**Cause.** The two modes share a pitch collection related by a perfect fourth.
Rotating the Navā profile by +10 quarter-tones lifts its cosine against Shūr from
0.621 to **0.931**, and on 22 of 25 sampled Shūr recordings the best Navā tonic
sits exactly +10 quarter-tones above the best Shūr tonic. The classifier has the
right notes and picks the wrong home.

This is a tonic-identification problem, not a pitch-resolution one. Distinguishing
the two requires knowing which degree functions as the *ist* — information that a
pitch-class distribution does not carry at any resolution.

**Two fixes tried, both refuted.**

1. *A quarter-tone tonic shift.* Shūr sits one quarter-tone below Navā on three
   degrees (+3/+4, +13/+14, +16/+17), so a flat tonic estimate would map one onto
   the other. Measured: the offset is +10 quarter-tones in 22 of 25 files and
   never ±1. Refuted.
2. *A phrase-final (forud) tonic prior.* Weighting tonic candidates by notes that
   come to rest before a silence, on the theory that the cadential descent marks
   the ist. Measured on a 108-recording balanced sample: 61.1% with the existing
   duration prior against 57.4% at the best phrase-final weight, degrading
   monotonically to 40.7%. Refuted — plausibly because a 90-second excerpt taken
   from the middle of a performance rarely contains a true forud.

**Note on intonation.** Real recordings do place pitch peaks 8-20 cents off the
24-TET grid, and frame-level grid fit drops to 0.2-0.35 against 0.97 on
synthetic audio, driven by ornament and glissando rather than mistuning. That is
a genuine limitation of a 50-cent grid, but it is *not* the cause of the dominant
error above, and estimating the tuning reference from sustained note centres
instead of all frames raised grid fit (0.18 to 0.42) while changing no
predictions at all.


## The representational ceiling

The Shūr/Navā confusion turned out to be one instance of a general property, and
the general property is the most important result of the work so far.

Comparing every pair of modal templates at its best rotational alignment, **21 of
the 78 pairs exceed cosine 0.85** and 12 exceed 0.90. Treating pairs above 0.90
as indistinguishable and taking connected components collapses the 13 modes into
**four groups** — which turn out to be the traditional families:

| Component | Members |
| --- | --- |
| 7 | Shūr, Navā, Dashtī, Abū'atā, Afshārī, Bayāt-e Tork, Bayāt-e Kord |
| 4 | Homāyūn, Bayāt-e Esfahān, Māhūr, Rāst-Panjgāh |
| 1 | Chahārgāh |
| 1 | Segāh |

The clustering was derived purely from template geometry and recovers the Shūr
family exactly as the tradition groups it. Chahārgāh (unique augmented seconds)
and Segāh (tonic on a koron degree) are the only modes that stand alone.

This predicts the measured per-class results and is confirmed by them: the two
singleton components are the two classes that work (Chahārgāh 82%, Segāh 61-72%)
while everything inside a large component sits between 0% and 43%.

Measured on 168 balanced real recordings:

| Question asked | Accuracy |
| --- | --- |
| Exact mode (13 classes) | 48.2% |
| **Mode family (4 components)** | **72.6%** (chance 25%) |

If a component were wholly indistinguishable and the answer guessed within it,
the 13-class ceiling would be 30.8%. The measured 48.2% is above that, so the
representation does carry *some* within-family information — but not much.

**Conclusion: a tonic-relative pitch-class profile identifies the mode family,
not the mode.** That is a property of the repertoire, not of this implementation:
the modes within a family genuinely share pitch collections, differing by which
degree functions as the ist and by melodic trajectory. No refinement of pitch
statistics can separate them.

### Four attempted fixes, all refuted

Every one targeted the Shūr/Navā case and was measured, not assumed:

| Attempt | Result |
| --- | --- |
| Quarter-tone tonic shift | Refuted — the offset is +10 quarter-tones in 22 of 25 files, never ±1 |
| Phrase-final (forud) tonic prior | Refuted — 61.1% to 57.4%, degrading monotonically with weight |
| Global register prior | Mixed — Shūr 0% to 40%, but overall 56.7% to 47.8% |
| Register tie-break on rotation-related pairs only | Refuted — 47.6% against a 48.2% baseline |

The register result is the informative one: the signal is real (the corpus keeps
32.3% of Shūr duration below its tonic against 46.3% for Navā) and it does fix
the target class, but commercial recordings vary in register far more than
notated radif does, so applying it globally injects more noise into the classes
that already work than it recovers from the broken one.

### What this implies for the next step

Separating within a family needs note *function* over time — which degree the
melody treats as home, which it recites on, how phrases descend — not a better
summary of which pitches occurred. That is the seyr, and it is a sequence
model's problem, not a histogram's.

A useful interim product change: report the family confidently and the mode
tentatively, since the family answer is both more accurate and better calibrated.

## Can a learned model separate modes within a family?

Short answer: not from monophonic f0, on the evidence available here.

The family layer is reliable (72-78%) and the remaining loss is concentrated in
one place. Measured on 168 balanced real recordings:

| | |
| --- | --- |
| Family accuracy | 72.0% |
| Exact mode | 50.0% |
| **Oracle, perfect within-family** | **72.0%, so +22.0 points are available** |

Within a family the problem is small: on this archive the Shūr family reduces to
Shūr vs Navā and the Māhūr family to Homāyūn vs Māhūr, both binary. Shūr vs Navā
scored **33.3% — worse than a coin flip**, which suggested a systematic bias
rather than absent information.

### The confound that shaped the experiment

Class is almost perfectly confounded with performer. Across 134 archive
recordings there are only **nine performer groups**, and outside one of them the
performer effectively determines the label:

| Group | Shūr | Navā |
| --- | --- | --- |
| Hossein Alizadeh | 33 | 30 |
| M.R. Shajarian (solo) | 30 | 0 |
| Grohe Sheyda / Aaref | 10 | 0 |
| Shajarian collaborations | 0 | 22 |
| others | 5 | 4 |

A random split would score well by memorising performers. Only Alizadeh holds
performer constant, so that is the controlled test; cross-performer transfer and
IRMA are separate, harder questions. Album and performer tags survive in the MP3
originals under `Training_Data/` even though the WAV conversion stripped them.

### What was tried

Seven functional features, each a *difference* between the Shūr tonic hypothesis
and the Navā one, asking of each candidate degree whether it behaves like a
tonic: do phrases rest on it, is it a melodic sink, is it held long, where does
it sit in the register, is it reached by descent, does the piece end on it, what
share of time does it take. Plus the existing template score margin.

| Feature set | Within-Alizadeh (5 seeds) | Trained on archive, tested on IRMA |
| --- | --- | --- |
| Functional only | 56.8% ±4.5 | 82.8% |
| Template margin only | 75.9% ±0.6 | 65.5% |
| Both | 65.1% | 82.8% |

The results **invert between datasets**, which is the signature of fitting
dataset-specific quirks rather than modal structure. The decisive test settles
it:

**Leave-one-performer-out over all nine groups: 56.0% learned, against 58.2% for
always guessing the majority class.** The features do not generalise across
performers.

### The bias is real, but correcting it only moves the error

The margin distribution explains the sub-chance result: **both classes have a
negative mean margin** (Shūr -0.41, Navā -0.76), so splitting at zero puts nearly
everything on the Navā side.

Three corrections were measured:

1. *Fitted threshold.* Archive 49.3% to 69.4%, but the optimum fitted on IRMA
   (-0.396) differs from the one fitted on the archive (-0.733), and each
   degrades the other set.
2. *Corpus-derived per-template offsets*, each template's mean best score over
   the corpus, computed without labels. Helped the binary case (archive 49.3% to
   68.7%) and **destroyed the 13-class problem: 40.3% to 16.7%**, family 78.5% to
   40.3%. The offset absorbs how often a template is legitimately correct, not
   just how permissive it is.
3. *Family-centred offsets*, the same correction centred within each family so
   cross-family comparison is untouched. Closed-set accuracy rose slightly
   (58.9% to 61.3%) but the per-class breakdown shows why it is not a fix:

   | Class | Uncalibrated | Calibrated | Delta |
   | --- | --- | --- | --- |
   | Shūr | 3.6% | 28.6% | +25.0 |
   | Navā | 46.4% | 3.6% | **-42.9** |

   The correction moves the starvation from Shūr to Navā. Open-set accuracy fell
   50.0% to 44.6% and family accuracy 72.0% to 68.5%, so it was reverted.

That a threshold shift merely trades one class's recall for the other's is the
clearest evidence that the pair carries almost no discriminative signal in this
representation: there is no threshold that separates them because the two score
distributions overlap almost entirely.

### Conclusion

**The family layer is the honest ceiling for pitch-based classification.**
Separating modes inside a family needs information that a monophonic f0 contour
summarised over a whole recording does not appear to carry — plausibly phrase
level structure, or the interaction of melody with the accompanying drone, or
simply more performers than nine.

Anyone continuing should note the measurement requirements: group by performer,
report leave-one-performer-out, and treat any result that inverts between two
datasets as noise.

## Cadence detection (forud)

The one change that moved the within-family number, and the reason the earlier
attempts failed.

Every tonic estimate up to this point weighted candidates by **sounding time**.
That is the wrong quantity: the most-sounded degree of a Persian mode is usually
the *shahed*, the reciting tone, not the *ist*. For Shur the shahed sits a fourth
above the tonic — exactly the interval that turns Shur into Nava. The estimator
was electing the shahed and the classifier was faithfully reporting the mode that
has its tonic there.

`dastgah/core/forud.py` looks for the figure that actually establishes the tonic:
a descent settling onto a held note, followed by a breath. A phrase ending is not
enough — phrases close on the shahed and on passing degrees all the time — so a
candidate needs all three of **descent** (how far and how steadily the line falls
into the final note), **repose** (that note held longer than the phrase around
it) and **silence after**, and carries a strength rather than a vote. An optional
recency half-life favours later cadences, since a performance may visit several
modes and only the closing forud returns to the principal tonic.

### Why it is blended rather than substituted

Cadence evidence is better but sparse, and a mode can cadence away from its tonic
mid-performance. Pure cadence evidence measured *worse* than sounding time (IRMA
38.9% against 40.3%); the blend measured better than either. Shipped at
`forud_prior_weight = 0.30`, chosen by sweeping both datasets:

| Blend weight | IRMA exact | IRMA family | IRMA Shūr | Archive exact | Archive family | Archive Shūr |
| --- | --- | --- | --- | --- | --- | --- |
| 0.00 (before) | 40.3% | 78.5% | 53.3% | 59.3% | 85.2% | 11.1% |
| 0.20 | 41.7% | 77.1% | 66.7% | 59.3% | 85.2% | 11.1% |
| **0.30** | **42.4%** | 76.4% | **66.7%** | **61.1%** | **87.0%** | **22.2%** |
| 0.50 | 41.7% | 74.3% | 66.7% | 61.1% | 87.0% | 22.2% |

Exact-mode accuracy and Shūr recall both improve on **two independent datasets**,
which is what justified shipping it — the earlier tonic-prior retune was rejected
precisely because the two sets disagreed there. Family accuracy is a wash: -2.1
on IRMA against +1.8 on the archive, three contours and one recording
respectively, both inside noise.

### The excerpt-position mistake this uncovered

Looking for cadences forced whole-recording analysis and exposed a measurement
error running through everything before it: the evaluators sampled **90 seconds
from the middle** of each recording. The forud is at the end. Sampling the end
instead of the middle was worth, on its own:

| Excerpt | open-13 | closed-6 | family |
| --- | --- | --- | --- |
| start | 52.8% | 59.7% | 77.8% |
| middle | 54.2% | 62.5% | 75.0% |
| **end** | **55.6%** | **70.8%** | **81.9%** |

The product always analysed whole files, so this understated it rather than
misreporting it — but it also explains why the earlier phrase-final tonic prior
was refuted. It was looking for a cadence in a stretch of music that does not
contain one.

### What this does not fix

Shūr remains the weakest class (22.2% on the archive, 66.7% on IRMA's cleaner
solo radif). The family ceiling stands: this is a better tonic estimator, not a
solution to within-family separation.

## Excerpt position, confirmed at full scale

The end-of-recording finding was measured on 72 recordings and then confirmed on
all 340. Identical pipeline, identical settings, only the 90-second window moved:

| Metric | 90s from the middle | 90s from the end |
| --- | --- | --- |
| Open-set (13 classes) | 49.7% | **56.5%** |
| Closed-set (6 classes) | 59.4% | **67.1%** |
| Mode family | 74.7% | **80.6%** |
| Top-3 | 79.7% | **86.5%** |
| Mean rank | 2.28 | **1.99** |

Homāyūn gains most (75.5% to 91.8%), then Navā (55.4% to 66.1%) and Chahārgāh
(74.4% to 81.4%). Shūr moves 9.0% to 14.1% and stays the outlier, as the
within-family ceiling predicts.

Seven points across every metric, from *where* the audio is sampled. The forud is
a local event at the close of a performance, and a mid-performance excerpt
frequently contains none — which is also why an early attempt at a phrase-final
tonic prior measured as refuted. `evaluate_archive.py` now defaults to the end;
a middle excerpt measures a configuration the library never uses, since
`analyze()` has always read whole files.

## Whole file against end-window: what the library should analyse

`analyze()` reads whole recordings; the evaluator excerpts for speed. Those had
never been compared on the same recordings, leaving open whether the published
figures described the shipped behaviour. On 54 whole recordings:

| Configuration | open-13 | closed-6 | family |
| --- | --- | --- | --- |
| **whole file (what `analyze()` does)** | 61.1% | 74.1% | **87.0%** |
| last 180s | 61.1% | 75.9% | 83.3% |
| last 90s | 63.0% | 74.1% | 79.6% |
| last 60s | 53.7% | 66.7% | 72.2% |
| whole file, recency half-life 240s | 61.1% | 75.9% | 87.0% |
| whole file, recency half-life 60s | 59.3% | 72.2% | 83.3% |
| whole file, recency half-life 30s | 57.4% | 66.7% | 77.8% |

**No change needed.** Whole-file analysis has the best family accuracy, which is
the metric the product leads with, and end-windowing costs it up to 7 points.
Recency weighting does not help: the only half-life matching whole-file (240s) is
long enough to be barely any weighting, and shorter ones degrade steadily.
Per-class differences between whole-file and last-90s are all exactly two
recordings at n=9 per class, and cancel out.

This does not contradict the excerpt-position result, which answered a different
question. *If you must excerpt, take the end*, because that is where the forud
falls. *If you can read the whole recording, do that*, because you get the forud
and everything else. Only below about 60 seconds does losing material outweigh
gaining the cadence.

One consequence: the headline archive figures were measured on 90-second
excerpts and therefore **understate** the library slightly, most visibly on
family accuracy (79.6% excerpted against 87.0% whole-file on this subset). A
whole-file run over all 340 recordings would settle the margin but costs roughly
three hours of pYIN; the subset is enough to establish that no code change is
warranted.

## Gusheh templates from audio: tested, and worse than notation

**Superseded note.** This section first concluded the experiment was impossible.
That was judged on IRMA alone and was wrong: the archive's filenames name the
gusheh (*Mokhalef*, *Bidād*, *Razavi*, *Zābol*, *Hesār*), and 146 of its 340
recordings match a corpus gusheh. Pooled with IRMA that gives 237 labelled
recordings over 132 gushehs, of which **55 have two or more examples covering 160
recordings** — against 8 gushehs and 19 recordings from IRMA alone. The original
reasoning is kept below.

Because 47 of those gushehs have examples in *both* sources, the experiment needs
no leave-one-out: templates were built from the archive and tested on IRMA
contours — different tradition, different recordings, different performers.
Candidates remained the mode's full gusheh list from notation, with audio
substituted only where available, so the audio condition was never scored against
a smaller candidate set.

| Gusheh templates | top-1 | top-3 | mean rank |
| --- | --- | --- | --- |
| **notation only (current)** | **25.3%** | **42.9%** | **7.9** |
| blend 30% audio | 23.1% | 40.7% | 8.4 |
| blend 50% audio | 20.9% | 34.1% | 9.3 |
| blend 70% audio | 19.8% | 31.9% | 10.0 |
| audio only where available | 22.0% | 30.8% | 10.0 |

Restricted to the 58 test items that have an archive template, the same ordering
holds (20.7% down to 17.2%). Degradation is monotonic in the mixing weight, so
this is a real effect rather than noise.

**The likely cause is that the labels are track-level while gushehs are sections
within a track.** A three-minute commercial recording titled *Razavi* is not
three minutes of Razavi: it opens with a darāmad, passes through the named
gusheh, and usually closes with a forud. Every audio template is therefore
blended with its neighbours, while a notated gusheh has exact boundaries. This is
also why IRMA serves well as a *test* set — its contours are per-gusheh
extractions rather than whole tracks.

What would work is per-gusheh segmented audio, which is a annotation problem
rather than a volume problem. More whole tracks will not help; the same 146
recordings cut at gusheh boundaries very likely would.

Gusheh identification therefore stays notation-based, at 25% top-1 and 43% top-3
against 4% and 13% for guessing.

### Original note, written before the archive labels were considered



Gusheh templates are built from notation, and the obvious improvement is to build
them from audio instead — the same move that took mode accuracy from 18.8% to
40.3%, since performance practice differs from the written radif.

It cannot be done here. IRMA's 91 gusheh-labelled contours cover **80 distinct
gushehs**: 72 appear exactly once, five twice, three three times. Under
leave-one-out, 72 of the 80 would have no examples left and could not be
identified at all, and only 19 contours have a sibling to learn from.

An audio-derived template would be a single recording evaluated against itself.
No number produced that way would mean anything, so none was.

What would change this is more recordings per gusheh, not a better method — the
same conclusion the within-family work reached about performers. Until then
gusheh identification stays notation-based, at 25% top-1 and 48% top-3 against
4% and 13% for guessing.

### Settled at full scale: the excerpt gap was noise

The section above reported, from 54 recordings, that whole-file analysis beat
90-second end excerpts by about seven points of family accuracy, and inferred
that the headline figures understated the library. **Measured on all 340, that
inference was wrong.**

| Metric | 90s from the end | Whole file | Difference |
| --- | --- | --- | --- |
| Open-set (13 classes) | 56.5% | 57.1% | +0.6 |
| Closed-set (6 classes) | 67.1% | 67.4% | +0.3 |
| Mode family | 80.6% | 81.2% | +0.6 |
| Top-3 | 86.5% | 88.2% | +1.7 |
| Mean rank | 1.99 | 1.88 | — |

Whole-file is better, but by well under a point on the headline metrics rather
than by seven. Seven points across 54 recordings is roughly four files, which is
what a difference of that size meant there.

The useful conclusion is the reverse of the earlier one: **a 90-second excerpt
from the end is an excellent proxy for the whole recording**, costing almost
nothing while running about ten times faster. The README now quotes whole-file
figures because that is what the library does, and the evaluator keeps end
excerpts as its default because they are cheap and faithful.

Per-class, whole-file against end-excerpt, the differences do not point one way:
Navā gains (66.1% to 71.4%) while Homāyūn loses (91.8% to 79.6%). Shūr sits at
12.8%, between the middle-excerpt 9.0% and end-excerpt 14.1%, and the
within-family ceiling is unmoved.

## Reproducibility check

Verified by cloning the pushed branch from GitHub into an empty directory and
building from nothing:

| Step | Result |
| --- | --- |
| `uv pip install -e ".[api,dev]"` | clean |
| `pytest` before any corpus is fetched | 122 passed, 2 skipped |
| `./scripts/fetch_data.sh radif` | 229 gusheh CSVs from Zenodo |
| `python scripts/build_templates.py` | both artefacts rebuilt |
| `templates.json` against the committed file | **byte-identical** |
| `gushehs.json` against the committed file | **byte-identical** |
| `pytest` after the rebuild | 122 passed, 2 skipped |
| `scripts/evaluate.py` | 60.3% / 69.4%, matching the README |
| CLI on a recording | correct |

So the claim that the templates derive from the Radif Corpus is not merely
documented but checkable: anyone can delete both JSON files and regenerate them
exactly.

Three defects surfaced, all now fixed.

**Nothing built `gushehs.json`.** It had been produced ad hoc, so a fresh clone
could not regenerate it and the provenance of all 229 gusheh templates rested on
an artefact nobody could reproduce. `build_templates.py` now writes both.

**The fetch could not survive a transient network error.** Zenodo reset the
connection at 42% on the first attempt and the script gave up, leaving a partial
file that a rerun would have tried to unzip. It now retries with
`--retry-all-errors`, deletes partial downloads, and verifies the archive with
`unzip -t` before extracting.

**The fetch reported success after doing nothing.** Its CSV count looked in
`radif_corpus/CSV` while the archive extracts to `radif_corpus/RadifCorpus/CSV`,
so it printed "0 gusheh CSVs" and then "done" — a good download and a failed one
were indistinguishable. Both counts are now correct and assert a plausible file
count rather than reporting whatever they find.

The IRMA half of the fetch was exercised earlier in development but not re-run
here, since it is a 225 MB sparse checkout; its error handling was hardened in
the same pass but is not covered by this check.

## Restricting to the six dastgahs with audio

`--dastgahs-only` drops the six avazes and Rast-Panjgah from the running,
leaving Shur, Nava, Homayun, Mahur, Chahargah and Segah — exactly the classes
the evaluation archive covers.

**It does not change accuracy at all**, and cannot. Each (mode, tonic) score is
computed from that template alone, so removing other templates cannot reorder the
survivors: picking the best of the six afterwards and scoring only the six give
identical answers. Measured at exactly +0.0 points on both IRMA (n=81) and the
archive (n=168), which is the arithmetic rather than a close call.

What it changes is what gets reported, and the family partition:

| Over all 13 | Over the six |
| --- | --- |
| Shur group (7 members) | **Shur group: Shur, Nava** |
| Mahur group (4 members) | Homayun, Mahur — now separate singletons |
| Chahargah, Segah singletons | unchanged |

The family layer stops being an abstraction over a seven-mode blur and becomes a
precise statement: the only modal distinction the method cannot make among these
six is Shur against Nava. Homayun and Mahur separating is consistent with that
pair measuring 94.8% under leave-one-performer-out.

The case for using it is not performance but honesty: avaz readings run 0-22% and
Rast-Panjgah 12.8%, so offering them costs the user more than withholding them.
The templates stay in the file either way, so nothing is lost.

Calibration survives the change. The softmax temperature was fitted over 13
classes, and restricting redistributes probability over fewer, but T=0.5 remains
the best setting for the reduced set as well (ECE 0.092 against 0.043). The full
set runs slightly underconfident, the restricted set slightly overconfident, both
around three points.

## Folding avazes into their mother dastgah

Avaz readings run 0-22% accurate, so they are not worth reporting as answers.
There are two ways to stop reporting them and they are not equivalent.

Dropping the avaz templates from the running measured **worse**. An avaz is a
branch of its parent, so a match against the Dashti profile is evidence for Shur;
removing the template discards that evidence. Keeping all thirteen templates
scoring and folding avaz probability into the mother beats dropping them by 14.7
points on IRMA and 5.4 on the archive — and the archive contains no avaz
recordings at all, so the gain is not about classifying avazes. An avaz profile
simply covers parts of its parent's territory that the parent's own profile
covers less well.

On all 340 recordings, whole files:

| Metric | Rank 13, take best of six | Fold into mother |
| --- | --- | --- |
| Accuracy over six dastgahs | 67.4% | **74.1%** |
| Top-3 | 88.2% | **97.6%** |
| Family | 81.2% | **83.8%** |
| Mean rank | 1.88 | **1.36** |

The per-class effect is concentrated where the structure predicts. **Shur rises
from 12.8% to 60.3%**, the largest single improvement measured in this work. Shur
has five avazes; their probability had been spread across classes that were never
going to be the answer, while Navā absorbed the territory. Shur→Navā confusions
fall from 50 to 16, and Navā drops from 71.4% to 53.6%, having been inflated by
the same imbalance.

### Temperature stopped being a display setting

Folding sums probabilities *after* the softmax, so the temperature now decides
which dastgah wins rather than only how confident the answer looks. It was set
to 0.4 on calibration grounds while folding was still optional, and that choice
was not revisited when folding became the default and the criterion changed.

Measured on accuracy under folding:

| T | whole-file | IRMA | whole ECE | IRMA ECE |
| --- | --- | --- | --- | --- |
| 0.3 | 75.9% | 63.1% | **0.081** | **0.132** |
| 0.4 | 74.1% | 66.2% | 0.203 | 0.141 |
| **0.5** | **79.6%** | 69.2% | 0.229 | 0.176 |
| 0.7 | 64.8% | **71.5%** | 0.223 | 0.275 |

The two sets disagree on the optimum — whole recordings prefer 0.5, IRMA's
contours 0.7 — and calibration would prefer 0.3 on both. 0.5 is best on average
for accuracy (74.4% against 70.2% at 0.4) and is the shipped value; confidence
consequently runs a few points optimistic. Sum-pooling buys accuracy at the cost
of a decision that depends on a calibration parameter, which is a real wart.
Deciding by strongest member instead would remove the dependence but measured
6-9 points worse.

## Stacking a learned model on the classifier's own output

The base classifier collapses 312 scored hypotheses into one answer, discarding
the shape of the score profile. A meta-model reading that whole profile — all 13
marginals, the 6 folded totals, the top confidence and the top-two margin — is a
natural way to recover it, and unlike the hand-crafted features tried earlier it
needs no new musical insight.

The routing precondition holds, which it did not in earlier work: confidence
separates correct from incorrect answers with **AUC 0.809** (65.7% mean
confidence when right against 45.8% when wrong). Deferring the least-confident
20% captures 25 of 60 errors, so a perfect second stage would reach 79.2%.

Under leave-one-performer-out the stacker measured **+2 to +4 points** (64.3%
base against 66-68.5%) across a broad regularisation plateau, C from 0.02 to 0.2.
The breadth looked like evidence the gain was real.

**It is not.** The archive has 13 performer groups but two of them — Hossein
Alizadeh and Mohammad-Reza Shajarian, after merging collaboration albums — hold
310 of the 340 recordings. Leave-one-performer-out is therefore very nearly
"train on one of those two, test on the other", and a model can score well on it
by learning which of the two it is listening to.

Training on those two and testing on the **five remaining artists**, 30
recordings the model never sees:

| | base | stacker |
| --- | --- | --- |
| held-out artists, C=0.02 | 50.0% | 46.7% |
| held-out artists, C=0.1 | 50.0% | 53.3% |
| held-out artists, C=0.3 | 50.0% | 50.0% |

Identical for four of the five artists; the only movement is one recording of
Nazeri's eight. The leave-one-performer-out gain does not transfer, so it was
the two dominant performers trading places, not modal learning.

Not shipped. It would also have ended the property that makes this system's
numbers unusually trustworthy — that nothing was trained on the evaluation audio.

**A limitation this exposed.** The base system scores 50.0% on those 30
recordings against 64.3% on the two dominant artists. Part of that is sample
skew — Grohe Sheyda is entirely Shūr, Gorouh Moulana entirely Māhūr, and n=30 is
thin — but the headline figure does lean on two performers, and a third would
test it far better than more recordings from the same two.

## The number on performers the system has never seen

The KDC corpus (Nikzat & Caro Repetto, ISMIR 2022) is 92 solo recordings by four
professional musicians — Mas'ud Sho'ari, Reza Zalpour, Mohammad Khodadadi and
Farahnaz Sahebgalam — each covering all six dastgahs. None appears in the
development archive, and no weight here was chosen against it.

| Metric | Development archive | **KDC, unseen performers** |
| --- | --- | --- |
| Accuracy over six dastgahs | 74.1% | **55.4%** |
| Top-3 | 97.6% | 84.8% |
| Family | 83.8% | 67.4% |
| Mean rank | 1.36 | 1.98 |

**A 19-point drop.** It is corroborated independently: on the archive's own five
minority artists (30 recordings) the system scores 50% against 64% on the two
dominant ones. Two separate held-out sets agree, so this is the generalisation
gap rather than a quirk of either.

Per-class the failure is not uniform. Māhūr scores 100% on KDC, 15 of 15. Segāh
collapses from 77.1% to 23.1% and Navā from 53.6% to 25.0%.

Why it drops is not established. Candidates: KDC is solo instrumental where much
of the archive is ensemble and voice; its recordings average 74 seconds against
the archive's 162, so each carries less evidence; and every scoring weight was
tuned against the archive, so some of the 74.1% is fitted to it. The templates
themselves come from notation and cannot have been fitted to either.

**The headline was moved.** 55.4% is now what the README leads with, as the
honest expectation for a new performer; 74.1% is stated as an upper bound on
material resembling the development set. Anyone quoting a single figure for this
system should quote the lower one.

## KDC in full: 273 recordings, six performers, every mode but one

The KUG Dastgāhi Corpus is 273 objects at
[phaidra.kug.ac.at/o:127195](https://phaidra.kug.ac.at/detail/o:127195), of which
92 were already present in this project's wider tree. Fetching the rest gives
3.8 hours across **12 of the 13 modes** — only Bayāt-e Kord is absent — performed
by six musicians: Mas'ud Sho'ari, Reza Zalpour, Mohammad Khodadadi, Farahnaz
Sahebgalam, Negar Bouban and Pouya Khoshravesh. None appears in the development
archive.

**Licence: CC BY-NC-ND 4.0**, not the CC-BY-4.0 the Zenodo record implies. Local
evaluation is fine; redistributing the audio or data derived from it is not.

The recordings are darāmads and average 50 seconds, against 162 for the archive,
so absolute accuracy is expected to sit lower: excerpts under a minute measured
several points worse in the windowing tests.

### The generalisation gap, confirmed at scale

| | development archive | KDC (92 subset) | **KDC (255)** |
| --- | --- | --- | --- |
| Accuracy over six dastgahs | 74.1% | 55.4% | **53.3%** |
| Family | 83.8% | 67.4% | 64.3% |

Nearly three times the material gives the same answer. Together with the archive's
own minority artists (50%), three held-out measurements now agree that the
development figure overstates performance on a new performer by roughly twenty
points.

### Avazes cannot be identified directly, measured on avaz audio

This corpus contains the first real āvāz recordings available to the project —
the archive has none and IRMA supplies contours rather than audio. Ranking all
thirteen classes rather than folding:

| | folded to dastgah | all 13 classes |
| --- | --- | --- |
| accuracy | **53.3%** | 19.8% |
| top-3 | 80.0% | 41.4% |
| mean rank | 2.15 | 5.08 |

Per class in the thirteen-way setting: Navā 40.0%, Rāst-Panjgāh 38.9%,
Chahārgāh 38.5%, Dashtī 28.6%, Māhūr 27.3%, Segāh 23.8%, Homāyūn 18.2%,
Bayāt-e Tork 15.0%, Abū'atā 11.1%, and **Afshārī, Bayāt-e Esfahān and Shūr at
0.0%**.

Shūr scoring zero is the pathology folding exists to fix: its own āvāzes absorb
it, so recordings of Shūr are returned as Dashtī or Abū'atā. Folding is worth
**+33.5 points** here, decided on material that is actually āvāz rather than
inferred from dastgāh recordings.

Rāst-Panjgāh appears here for the first time with audio behind it, at 38.9% in
the thirteen-way setting. It remains outside the default answer space, since the
development archive contains none and the six-way figures cannot speak to it.

## Nava settles the sharpen question, and reverses an earlier call

Nava (BabaAli & Mohammadi, University of Tehran) is 1,785 solo recordings, 54.9
hours, by **39 artists** across five instruments and all seven dastgahs — five
times the recordings and six times the performers of anything else here, and
balanced on both axes.

Its filenames encode `instrument_dastgah_artist_track` with no accompanying
documentation. The dastgah digits were identified by ear by the project owner,
corroborated on four of the seven groups by a second listener, with a third
listener reading groups 1, 3, 4 and 5 differently. Those four carry real label
uncertainty. The ordering implied by the paper's abstract is wrong on at least
one count: all three listeners and the classifier independently read group 6 as
Chahārgāh where that ordering gives Rāst-Panjgāh.

### The sharpen exponent

| sharpen | Nava (1785) | archive (340) | KDC (255) | IRMA (130) |
| --- | --- | --- | --- | --- |
| 2.0 | 60.6% | 69.4% | **56.9%** | **72.3%** |
| **3.0** | **63.8%** | **74.1%** | 53.3% | 69.2% |

On Nava accuracy rises monotonically with the exponent: 55.6% at 1.5, 60.6% at
2.0, 62.5% at 2.5, 63.8% at 3.0.

It had been lowered to 2.0 on the strength of KDC and IRMA. That was wrong, and
instructively so: those are the two smallest corpora, and the decision rested on
them because nothing larger with performer diversity existed yet. Nava and the
archive — the two largest — both prefer 3.0, so it is restored.

### Performer and instrument, finally separable in part

Accuracy across the 20 artists with at least 15 recordings: **mean 64.2%, sd
11.7, range 41-86%**. Performer sensitivity is therefore real but smaller than
the 21-point archive-to-KDC drop suggested; some of that gap was corpus and
recording-length effects attributed to performers.

By instrument, four of the five score 65-69% and the fifth scores **50.8%** — a
15 to 19 point deficit, the first direct evidence that instrument matters
independently.

The two cannot be fully disentangled: **30 of the 39 artists play exactly one
instrument**. The weakest instrument has 10 artists behind it, so it is not one
player's idiosyncrasy, but the design does not permit a clean separation.

## Nava is the first corpus here where training is defensible

39 artists, 1,785 recordings, near-balanced over seven dastgahs and five
instruments. Everything below is **artist-grouped**: a random split is
meaningless here, because 11 of the 39 artists play exactly one dastgah, so
artist identity leaks the label outright.

### The feature representation decides everything

Artist-grouped 5-fold, against the untrained templates' 63.8%:

| features | model | grouped | in-fold |
| --- | --- | --- | --- |
| absolute 24-bin profile | logreg | 12.2% | 34.5% |
| absolute 24-bin profile | gbt | 39.0% | 100.0% |
| tonic-rotated profile | logreg | 61.3% | 62.6% |
| tonic-rotated profile | gbt | 65.9% | 100.0% |
| template log-probs | logreg | **66.8%** | 68.6% |
| rotated + log-probs | gbt | 67.1% | 100.0% |

The first two rows are the warning. A boosted model on the **untransposed**
histogram memorises its training fold perfectly and then scores 39% on unseen
artists — 24 points *below* the untrained templates. It is learning instrument
tuning and player register, not mode. Any protocol that did not group by artist
would have reported it as excellent.

Rotating the profile to the template-estimated tonic is what makes learning
possible at all: the same model goes 39% to 65.9% on identical pitch data, only
re-indexed. **A trained model here is not an alternative to the template system;
it depends on it for the tonic.**

### Leave one artist out: the gain transfers, unlike stacking

Over the 20 artists with at least 15 recordings (1,691 held-out predictions):
**templates 63.4%, model 67.6%, better on 15 of 20 artists.** Macro-averaged,
which weights each dastgah equally: **62.6% to 66.4%**.

This is a real difference from the stacking attempt recorded above, where a
LOPO gain of 2-4 points vanished entirely on unseen artists because two
dominant performers were trading places. Here every predicted artist is unseen.

The gain lands where the templates are weakest. Five of the six artists the
templates scored below 57% gain 9 to 18 points (artist 02: 54.9% to 72.5%;
artist 25: 56.6% to 72.3%). The templates encode one notated radif, so a player
far from Talai's reading is exactly where 38 other players have something to
add. Artist 39 is the counterexample: worst at 41.1%, and the model drops it to
26.8%.

### What moves, by class

| dastgah | templates | model | delta |
| --- | --- | --- | --- |
| chahargah | 70.8% | 86.2% | +15.4 |
| nava | 53.6% | 65.8% | +12.2 |
| homayun | 67.4% | 73.3% | +5.9 |
| segah | 77.8% | 82.7% | +4.9 |
| shur | 73.5% | 75.8% | +2.3 |
| mahur | 57.5% | 59.1% | +1.5 |
| rast_panjgah | 37.7% | **22.2%** | **-15.5** |

Navā moving 12 points is notable on its own: six hand-built features and one
learned model all failed to separate Shūr from Navā earlier in this project.

The Rāst-Panjgāh collapse is **not** the model trading a weak class away to buy
the others, which is what the table looks like at first glance. The prediction
counts give the real mechanism. Templates *over*-predict Rāst-Panjgāh badly —
268 predictions against 207 true recordings — and those 61 false positives are
stolen from other classes. The model corrects the over-prediction and overshoots
into under-prediction, 132 predictions for 207 recordings. Other classes gain
from the correction; Rāst-Panjgāh recall pays for the overshoot.

Chahārgāh's +15.4 is independent of it: of the 44 recordings the model rescues,
**34 were called Homāyūn** by the templates and only 4 Rāst-Panjgāh. That is a
learned discrimination, not reallocated probability.

Where true Rāst-Panjgāh actually goes is **Māhūr**: 40.6% under the templates,
48.3% under the model. That confusion is theory, not noise — Rāst-Panjgāh and
Māhūr share essentially the same pitch collection and differ in seyr and
emphasis, which a pitch-class profile cannot see. Rāst-Panjgāh is also dastgah
code 4, one of the disputed label groups, so noise may contribute; but the
over-prediction mechanism explains the movement without it.

### Instrument 4's deficit is not a data-volume problem

It goes 50.8% to 56.4% — the same +5.7 every other instrument gains (except
instrument 3, flat at -0.6). Training does not close the 15-point gap, so
whatever that instrument does is invisible to a pitch-class profile.

### If this ships

The boosted model hits 100% in-fold on every feature set, so 67.6% is
regularisation-limited, not capacity-limited. A plain multinomial logistic
regression on the template log-probs alone reaches 66.8% grouped from 68.6%
in-fold — nearly the same generalisation from a far simpler hypothesis, and one
that degrades gracefully. That is the version to prefer.

Two things to settle first: the group 1/3/4/5 labels with the Nava authors, and
whether a ~4 point mean gain justifies giving up the system's current property
that every answer is traceable to notated radif theory rather than to 39
performers' habits.

## A progression scorer: built, measured, not shipped

`dastgah/core/seyr.py` scores the *order* a performance visits a dastgah's
gushehs in, which is the one thing a pitch histogram and a bigram matrix both
discard. Windows of 20s are matched against a candidate dastgah's own gushehs,
reduced to an expected position in its seyr (0 at the daramad, 1 at the last
gusheh), and correlated against time.

It demonstrably reads order and nothing else. Synthetic traversals in radif
order score above 0.8 for all six in-scope dastgahs; the same material reversed
scores below -0.8; shuffled, it collapses below 40% of the ordered score; and
pooling a whole traversal into one repeated window scores exactly 0.0, so no
pitch content leaks into it. The joint search recovers the right mode and the
right tonic for every one.

### It does not separate dastgahs

Two corpora with recordings long enough to contain a traversal. Shajarian's
complete radif — 16 in-scope performances, 2 to 30 minutes — and Nava's long
tail, 218 in-scope recordings over three minutes by 30 artists.

| | Shajarian (chance 16.7%) | Nava >3min (chance 14.3%) |
| --- | --- | --- |
| pitch content only | 50.0% | 88.1% |
| progression only | 25.0% | 21.1% |
| progression only, permutation-calibrated | 18.8% | — |

Above chance, and nowhere near usable. The raw version first collapsed onto
Shur in 9 of 16 Shajarian performances, which is a flexibility artifact rather
than a signal: **Shur folds six modal templates and 68 gushehs, Mahur folds one
and 34**, so a max over modes gives Shur six shots at a spurious correlation.
A permutation null over window order removes the collapse — predictions spread
across all six — without improving accuracy.

Underneath the argmax the signal is real but non-specific:

```
mean rank of the true dastgah : 2.94   (chance 3.50)
true in top 2                 : 50.0%  (chance 33.3%)
mean z of the true dastgah    : +3.32
mean z of the five others     : +3.00
paired difference             : +0.32   Wilcoxon p=0.464, n=16
```

A performance does advance through its own dastgah's seyr — positive z in 88%
of cases — and advances through every *other* dastgah's seyr almost as much.

### A wrong explanation, recorded so it is not repeated

The obvious reading is that this is a register detector: if seyr order tracked
tessitura, then "position rises with time" would just mean the performer went
up, which all radif does. **That is false.** The mean within-dastgah Spearman
correlation between seyr position and tessitura is **+0.027**, and the
trajectories disagree in sign — Homayun climbs +6.45 quarter-tones from its
opening third to its closing third while Chahargah descends -4.96.

What survives is the borrowing result one level up. It was never specific to
Rast-Panjgah: 70-92% of *every* dastgah's gushehs have a >0.95 aligned-cosine
twin elsewhere. A window's soft match therefore spreads across many dastgahs'
gushehs, and whichever dastgah happens to carry a compatible ordering scores
well. The order is informative about the performance and not yet attributable
to a dastgah, because the things being ordered are not themselves
distinguishable.

### As a fitted term it earns nothing

Wired into `learn.py` as a fourth additive term with its weight initialised at
zero, so the fitted value measures what order adds after pitch content:

    alpha 4.483   transition 7.032   prior 1.214   progression -0.239

Slightly negative. Leave-one-artist-out accuracy is identical with the term and
without it (84.1% / macro 83.0% either way). Not shipped: `seyr.py` is built,
tested and unused by the default pipeline.

## Length, not the model, was most of the problem

The same untrained classifier, same parameters:

| | recordings | accuracy | macro |
| --- | --- | --- | --- |
| all of Nava (median 75s) | 1,785 | 63.4% | 62.6% |
| Nava over three minutes | 218 | **88.1%** | **87.4%** |

Nothing was fitted to produce that. A 75-second excerpt does not contain enough
of a performance to identify its mode, and most of the 63.4% that the fitted
model improved on was an excerpt-length ceiling rather than a modelling failure.

That also reverses where fitting helps. On the short-excerpt corpus, fitting 17
parameters gained 4 points over theory. On the long recordings it **loses**
four to five:

| level | params | Nava >3min |
| --- | --- | --- |
| theory only | 0 | **88.1%** |
| weights | 5 | 82.8% |
| bias | 18 | 82.8% |
| sharpen | 31 | 84.1% |
| profiles | 343 | 84.1% |

Leave-one-artist-out here holds out 151 of 218 recordings, leaving roughly 67
to fit on, against a baseline already at 88%. Fitting helps where the baseline
is weak and the excerpt is short; it overfits where the baseline is strong and
the corpus is small. Neither regime recommends shipping a fitted model on this
evidence.

## Dropping Rast-Panjgah is worth 11.5 points, before any fitting

Restricting the answer space to the six dastgahs that remain in scope, with
avazes still folded into their mothers, takes the untrained classifier from
**63.4% to 74.9%** over 1,568 Nava recordings by 37 artists. Nothing was fitted
to produce that. It is the largest single improvement in the project, and it
came from removing a class the representation cannot hold rather than from any
modelling.

The restriction is now applied when the design is built rather than when results
are reported. That matters for training and not for inference: renormalising
after the softmax is identical to restricting the softmax, so predictions are
unchanged, but a mode left in the objective that can never be the answer trains
the fit to push probability away from it -- work inference discards for free.

## Shrinkage toward theory, and what the extra capacity is worth

Every free parameter is now pulled toward the value the notated radif gives it,
so the penalty interpolates between the hand-built classifier at one end and an
unconstrained fit at the other. A test asserts the limit: at penalty 1e6 the fit
returns theory's parameters to 1e-3.

Artist-grouped 5-fold over the 1,568, against 74.9% for theory alone:

| level | params | 0 | 0.003 | 0.01 | 0.03 | 0.1 | 0.3 | 1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| weights | 5 | 74.7 | 74.7 | 74.6 | 74.6 | 74.4 | 74.6 | 74.8 |
| bias | 18 | **78.4** | 77.4 | 76.7 | 75.6 | 74.7 | 74.7 | 74.9 |
| sharpen | 31 | 77.7 | 76.8 | 76.9 | 76.8 | 75.8 | 75.4 | 74.9 |
| profiles | 343 | 76.3 | 77.4 | 76.9 | 77.3 | 75.6 | 75.4 | 75.0 |

Two expectations were wrong. Shrinkage does not rescue the larger models --
`sharpen` and `profiles` peak at or near zero penalty and never beat the
18-parameter `bias` model, so their extra capacity is not being wasted by
overfitting, it simply is not useful. And `bias` wants *no* shrinkage: 1,568
recordings determine 13 per-mode offsets well enough that pulling them toward
theory only destroys information.

Refitting the four global scalars is worth nothing, for the third corpus running.

Leave-one-artist-out at `bias`, penalty 0, over the 20 artists with at least 15
recordings (1,484 recordings): **74.3% to 77.8%**, macro 73.9% to 77.6%, better
on 15 of 20 unseen artists. Nava gains 11.3 and Chahargah 9.2; only Mahur loses,
from 86.5%.

## That gain does not transfer, and not for the reason it first appears

Fit on Nava, test on KDC -- 189 in-scope recordings, different performers,
different provenance:

| | accuracy | macro |
| --- | --- | --- |
| theory only | **57.7%** | 51.7% |
| transferred | 49.2% | 55.0% |

Eight and a half points worse. The fitted biases say why:

    bayat_e_tork  -12.61    afshari  -12.38    abuata  -3.79

The fit crushes Shur's avaz templates, because **Nava contains no
avaz-labelled recordings at all** -- its seven classes are all dastgahs. Within
Nava, any probability an avaz template absorbs is noise, and suppressing it is
correct. In KDC the avaz templates are what detects avaz recordings, and those
are 83 of 202. Shur falls 26.4 points, from 65.5% to 39.1%.

So the avaz biases are not merely unreliable when fitted on Nava, they are
**unidentifiable** from it: the corpus carries no evidence about them, and
whatever the fit puts there is an artifact of the labels' absence.

That is the obvious reading, and a control refutes it as the whole story. Every
configuration loses on KDC, including one with no per-mode bias at all:

| configuration | params | KDC | vs theory |
| --- | --- | --- | --- |
| theory only | 0 | **57.7%** | - |
| bias, no shrinkage | 18 | 49.2% | -8.5 |
| bias, class-balanced | 18 | 49.7% | -7.9 |
| sharpen | 31 | 51.9% | -5.8 |
| **weights only** | **5** | **50.8%** | **-6.9** |
| bias, penalty 0.1 | 18 | 55.6% | -2.1 |

`weights` is four global scalars and a progression weight -- no class prior
anywhere -- and it still loses 6.9 points. The damage also falls monotonically as
the penalty pulls the fit back toward theory, reaching zero at full shrinkage.
So the honest statement is broader than the avaz story: **Nava-optimal
parameters are KDC-suboptimal across the board**, and the avaz biases are the
largest single contributor rather than the only one.

One thing does transfer, in every configuration tested:

| | theory | bias | sharpen | weights | balanced |
| --- | --- | --- | --- | --- | --- |
| nava | 21.1% | 42.1% | 42.1% | 47.4% | 52.6% |
| shur | 65.5% | 39.1% | 41.4% | 54.0% | 36.8% |

Nava gains 21 to 32 points on a different corpus under every fit, which is
signal rather than artifact. The accuracy loss is carried almost entirely by
Shur, which is 87 of KDC's 189 in-scope recordings and nearly all folded avaz.

Two lessons worth keeping separate from the numbers:

**Leave-one-artist-out does not test what broke here.** It controls for the
performer. What differs between Nava and KDC is the label distribution and the
recording conditions, which no amount of performer-grouping within one corpus
can expose. The earlier stacking failure was the same shape with a different
mechanism, and the argument that 20 artists made this case stronger was true and
beside the point.

**Macro average hid it.** Accuracy fell 8.5 points while the macro average
*rose* 3.3, because the fit helps the rare classes and destroys the dominant
one. Reporting macro alone would have called this a success.

## Freezing the modes a corpus never names: principled, and worse

The transfer failure decomposes cleanly. On KDC's 106 dastgah-labelled
recordings a Nava fit goes **50.0% to 63.2%** at `sharpen`; on its 83
avaz-labelled ones it goes **67.6% to 33.7%**. So the fit learns something real
about the modes and destroys the avaz calibration Nava cannot speak to.

Holding every mode the training labels never name at theory's values should fix
that. It made things worse: **-10.6 points** transferring to full KDC, against
-8.5 for not freezing.

The exponent is not separable per mode. A fit moves all six observed dastgahs
from 3.0 to roughly 1.85 and leaves the six frozen avazes at 3.0, and since a
sharper profile is more selective than a flatter one, the two groups' scores stop
being on a comparable scale. Tying an avaz to its *mother's fitted* values rather
than to theory's is the shape a fix would have to take. `freeze_unobserved`
defaults off, with the measurement recorded on the function.

## Those fitted exponents do not imply a better global one

All six landing near 1.85 looks like a single miscalibrated constant rather than
six modal properties, and `sharpen=3.0` was chosen while Rast-Panjgah was still
answerable -- a broad template that attracts everything being exactly what needs
heavy sharpening to suppress. KDC and IRMA had both preferred 2.0 earlier.

Swept over the six-class answer space:

| corpus | 1.25 | 1.5 | 1.75 | 2.0 | 2.25 | 2.5 | 3.0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| nava (1568) | 59.5 | 64.5 | 69.0 | 71.2 | 72.6 | 73.3 | **74.9** |
| kdc (189) | 57.1 | 61.9 | **64.6** | 61.4 | 58.2 | 56.6 | 57.7 |
| shajarian radif (16) | 62.5 | **68.8** | 56.2 | 50.0 | 50.0 | 50.0 | 50.0 |
| nava >3min (218) | 69.3 | 75.7 | 80.7 | 81.2 | 84.4 | 87.6 | **89.0** |
| pooled by size | 60.4 | 65.5 | 69.8 | 71.2 | 72.3 | 73.1 | **74.6** |

Refuted. Nava rises monotonically to 3.0 and the long recordings reach 89.0%
there. The inference was wrong because the exponent is not interpretable in
isolation: `alpha` scales the whole profile term and the same fit raised it from
2.0 to 4.483, so a lower exponent and a higher scale are partly interchangeable.
Reading one fitted parameter without its coupled partner is the error.

**sharpen stays 3.0.** The corpus disagreement is unchanged and already recorded
above: the two smallest corpora prefer a lower exponent, the two largest a higher
one.

## What this line of work shipped

Nothing beyond what was already default. `analyze()` already answers over the six
dastgahs, so the 74.9% is what the library does today.

| finding | evidence | status |
| --- | --- | --- |
| drop Rast-Panjgah from the answer space | 63.4% to 74.9%, nothing fitted | shipped (already the default) |
| restrict the answer space at design time | train/test agreement; inference unchanged | shipped, training only |
| fitted per-mode parameters, dastgah audio | KDC 50.0% to 63.2% | real, not shipped |
| fitted per-mode parameters, avaz audio | KDC 67.6% to 33.7% | blocks shipping |
| freeze modes the labels never name | -10.6 vs -8.5 | refuted |
| refit the four global scalars | nothing, on four corpora | dead |
| lower the global exponent | refuted by the sweep above | dead |
| progression through the seyr | fitted weight -0.239 | dead |

`learn.py` is what diagnosed all of it and `transfer_test.py` is the gate any
future change should pass: within-corpus cross-validation chose `bias` over
`sharpen` on Nava (78.4% against 77.7%) and cross-corpus transfer chose the
opposite (63.2% against 61.3%), so corpus-internal model selection is not
sufficient here.

## Every way of transferring a fitted model, and the count

Fourteen configurations, all measured against the untrained classifier on the
same recordings. Nothing beats theory by more than noise.

Fit on Nava (1,568 recordings, 37 artists, no avaz labels):

| variant | KDC (189) | IRMA (130) |
| --- | --- | --- |
| theory only | **57.7%** | **69.2%** |
| bias | -8.5 | |
| bias, class-balanced | -7.9 | |
| bias, penalty 0.1 | -2.1 | |
| sharpen | -5.8 | |
| weights only | -6.9 | |
| avazes frozen at theory | -10.6 | |
| **avazes tied to their mother** | **-4.2** | **-3.1** |
| avazes tied, bias | -7.9 | -5.4 |
| nava template only | -6.9 | -2.3 |

Fit on KDC (189 recordings, avaz labels present):

| variant | IRMA | Nava |
| --- | --- | --- |
| sharpen | -5.4 | -10.3 |
| bias | -9.2 | -9.7 |

Fit on Nava and KDC jointly (1,757 recordings):

| variant | IRMA |
| --- | --- |
| sharpen | +0.0 |
| bias | +1.5 |
| nava template only | +2.3 |

### Tying is the right shape and not enough

Pointing each unobserved mode at its mother's *fitted* parameters halves the
damage freezing caused (-4.2 against -10.6) and beats leaving them free (-5.8).
The tied biases confirm the mechanism was understood: Shur's five avazes carry
Shur's own -0.07 instead of the -12.61 an unconstrained fit gave Bayat-e Tork.
It still does not reach theory.

### A corpus with avaz labels is necessary and not sufficient

KDC has the labels Nava lacks and is the worst corpus to fit on: 87 of its 189
in-scope recordings are Shur, because five avazes fold into it, so the fit
learns to answer Shur. Shur gains 17 to 22 points and everything else collapses
-- Nava falls to 6.6% on Nava's own recordings. 189 recordings dominated by one
folded class cannot calibrate twelve templates.

Adding those 189 to Nava's 1,568 moves transfer from about -4 to about zero,
which is the right direction and inside the noise of a 130-recording test.

### Correcting one template is zero-sum

Nava is the one template that improves on all three corpora -- +11.3 under
leave-one-artist-out within Nava, +26.3 on KDC, +14.3 on IRMA -- and it is the
mode theory reads worst, which fits its 0.931 aligned cosine against Shur.
Fitting it alone and pinning the other eleven at theory still loses 6.9 points
on KDC, because the score is a softmax over all twelve templates:

    recordings Nava's correction gains :  +5 of 19
    recordings the others lose         : -18  (Shur -10 of 87, Homayun -6 of 31)
    net                                : -13 of 189

Probability handed to Nava is taken from Shur and Homayun. There is no such
thing as repairing one template in isolation in a joint answer space, and the
templates it steals from are larger than the one it fixes.

### The conclusion this line reaches

The notated radif's parameters are already close to cross-corpus optimal.
Fitting reliably improves accuracy within whatever corpus it is fitted on --
Nava 74.9% to 77.8% under leave-one-artist-out over 20 unseen artists -- and
reliably degrades it on any other corpus. The gains are real and corpus-local;
the parameters theory supplies are what generalise.
