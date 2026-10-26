import type { AnalysisResult } from "../lib/types";

/** Display names for the two dastgahs a pair model decides between. */
const NAMES: Record<string, string> = {
  shur: "Shūr",
  nava: "Navā",
  homayun: "Homāyūn",
  mahur: "Māhūr",
  chahargah: "Chahārgāh",
  segah: "Segāh",
};

const name = (key: string) => NAMES[key] ?? key;

/**
 * Shown only when a second model was consulted, which happens when the top two
 * answers are a pair the classifier is known to confuse.
 *
 * The disagreement is the useful part. A pair model gains on the corpus it was
 * fitted to and loses on every other one, so it is not evidence that the answer
 * is wrong — it marks a recording where the two readings part and an ear
 * decides. Saying that plainly matters more than the number.
 */
export default function SecondOpinion({ result }: { result: AnalysisResult }) {
  const opinion = result.second_opinion;
  if (!opinion) return null;

  const [a, b] = opinion.pair;
  const disagrees = !opinion.agrees;

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Second opinion</span>
        <span className={`label-mono ${disagrees ? "text-secondary" : "text-tertiary"}`}>
          {disagrees ? "disagrees" : "agrees"}
        </span>
      </header>

      <div className="p-5 space-y-3">
        <p className="text-sm text-secondary">
          {name(a)} and {name(b)} are confused often enough to be worth a
          dedicated check, so a model trained on only those two was asked.
        </p>

        <div className="flex items-baseline gap-3">
          <span className="label-mono text-tertiary">it reads</span>
          <span
            className={`text-lg font-semibold ${
              disagrees ? "text-secondary" : "text-primary"
            }`}
          >
            {name(opinion.choice)}
          </span>
          <span className="label-mono text-tertiary">
            {(opinion.confidence * 100).toFixed(0)}%
          </span>
        </div>

        {disagrees ? (
          <p className="text-sm text-secondary">
            This recording is one the two readings part on. The answer above
            stands — the second model is less reliable on unfamiliar recordings —
            but if you know the repertoire, this is where your ear is worth more
            than either of them.
          </p>
        ) : (
          <p className="text-sm text-tertiary">
            Both readings agree, which is the ordinary case for this pair.
          </p>
        )}

        <p className="label-mono text-tertiary pt-1">
          trained on {opinion.n_recordings} recordings of the pair ·{" "}
          {opinion.trained_on} · loses accuracy on corpora it was not fitted to
        </p>
      </div>
    </section>
  );
}
