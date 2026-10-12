# Draft: Nava dataset access request

**To:** Prof. Bagher BabaAli — School of Mathematics, Statistics and Computer
Science, College of Science, University of Tehran.
*(Email address not published in the papers I could reach. It is usually given
as the corresponding author's address in BabaAli & Mohseni, "On the
effectiveness of self-supervised pre-trained models for Persian traditional
music information retrieval", J. Audio Speech Music Proc. 2025 — or via his
University of Tehran staff page.)*

**Cc:** the paper's second author, Mohammadi, if an address is available.

**Subject:** Request for access to the Nava database for dastgāh recognition research

---

Dear Professor BabaAli,

I am writing to ask whether the Nava database might be made available for
academic use. I am working on automatic dastgāh recognition and have read your
paper with Mohammadi, "Nava: A Persian Traditional Music Database for the Dastgah
and Instrument Recognition Tasks" (Advanced Signal Processing, 2019).

I have built a dastgāh classifier that uses no trained audio model. It derives
24-tone scale templates from the Radif Corpus — the symbolic transcription of
Mirza Abdollah's radif — and matches recordings against them by pitch content,
tonic estimation and cadence detection. On six dastgāhs it reaches 74% on a
340-recording archive.

That figure is the reason I am writing. Two performers account for 310 of those
340 recordings. Evaluated on the KDC corpus, whose four musicians appear nowhere
in my development data, the same system scores 55%. The nineteen-point gap is
confirmed by a second held-out split, so it is a property of the method rather
than of one test set: much of what looks like modal recognition is sensitive to
performer.

I cannot currently tell how much of that gap is performer identity, instrument,
recording condition or repertoire, because no corpus available to me has enough
artists to separate them. Nava, with 1,786 solos by 40 performers across five
instruments and seven dastgāhs, is the only collection I know of that would
allow a proper performer-grouped evaluation.

If access were possible, I would use it solely to measure generalisation across
performers and instruments, with grouped splits so that no artist appears in both
training and evaluation. I would be glad to:

- cite the database as you specify in any resulting work;
- share my results on Nava with you before publishing anything, including
  negative results;
- make the analysis code available to you — it is already open, and the method
  requires no training, so reproducing any figure needs only the audio and a
  single command.

I would of course accept whatever licensing or usage conditions you set, and I am
happy to sign an agreement or work through my institution if that is the normal
route.

Thank you for assembling and documenting the database, and for considering this.

With best regards,

[name]
[affiliation, if any]
[email]
[link to the code repository, optional]
