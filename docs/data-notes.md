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
