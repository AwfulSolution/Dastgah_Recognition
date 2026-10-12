import type { AnalysisResult } from "../lib/types";
import { accidental } from "../lib/theory";

/** Confidence is calibrated, so colour it honestly rather than always gold. */
function confidenceTone(confidence: number) {
  if (confidence >= 0.7) return { text: "text-primary", label: "strong match" };
  if (confidence >= 0.45) return { text: "text-secondary", label: "probable match" };
  return { text: "text-tertiary", label: "weak — treat as a shortlist" };
}

export default function ClassificationPanel({ result }: { result: AnalysisResult }) {
  const microtonal = result.degrees.filter((d) => d.microtonal);
  // A family of one adds nothing over the mode itself, so don't show both.
  const familyIsInformative = result.family_members.length > 1;
  const modeTone = confidenceTone(result.confidence);
  const familyTone = confidenceTone(result.family_confidence);

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Identified Radif Class</span>
        <span className="label-mono">
          {familyIsInformative ? "Family" : "Confidence"}{" "}
          <span className={`${familyTone.text} font-semibold`}>
            {(result.family_confidence * 100).toFixed(1)}%
          </span>
        </span>
      </header>

      <div className="p-5">
        {familyIsInformative ? (
          <>
            <p className="label-mono mb-1">Mode family</p>
            <h2 className="font-serif text-[34px] font-semibold leading-[1.1] text-primary">
              {result.family_name}
            </h2>
            <p className="mt-1.5 font-mono text-[11px] leading-relaxed text-outline">
              {result.family_members.join(" · ")}
            </p>

            <div className="mt-4 rounded border border-hairline bg-surface-container px-3 py-2.5">
              <div className="flex items-baseline justify-between gap-3">
                <div>
                  <p className="label-mono">Most likely within the family</p>
                  <p className="mt-0.5 font-serif text-[22px] text-on-surface">
                    {result.name}
                  </p>
                </div>
                <div className="shrink-0 text-right">
                  <p className={`font-mono text-[15px] ${modeTone.text}`}>
                    {(result.confidence * 100).toFixed(1)}%
                  </p>
                  <p className="font-serif text-[18px] text-primary/80" dir="rtl">
                    {result.persian}
                  </p>
                </div>
              </div>
              <p className="mt-2 border-t border-hairline pt-2 text-[12px] leading-relaxed text-on-surface/60">
                Modes in one family share a pitch collection and differ by which
                degree acts as the tonic. The family reading is the reliable one;
                the mode within it is a best guess.
              </p>
            </div>
          </>
        ) : (
          <div className="flex items-start justify-between gap-4">
            <div>
              <p className="label-mono mb-1">
                {result.kind === "dastgah" ? "Primary Dastgāh" : "Āvāz"}
              </p>
              <h2 className="font-serif text-[40px] font-semibold leading-[1.1] text-primary">
                {result.name}
              </h2>
              <p className={`mt-2 font-mono text-[11px] ${modeTone.text}`}>
                {modeTone.label} — this mode has no close relatives in the radif
              </p>
            </div>
            <p className="font-serif text-[34px] leading-tight text-primary/90" dir="rtl">
              {result.persian}
            </p>
          </div>
        )}

        <dl className="mt-5 grid grid-cols-1 gap-px overflow-hidden rounded border border-hairline bg-hairline sm:grid-cols-3">
          <Cell
            label="Tonic (ist)"
            value={result.tonic_name}
            sub={
              result.n_cadences > 0
                ? `${result.tonic_hz.toFixed(1)} Hz · ${result.n_cadences} cadences`
                : `${result.tonic_hz.toFixed(1)} Hz`
            }
          />
          <Cell
            label="Shāhed (pivot)"
            value={result.shahed_name}
            sub={`+${result.shahed_interval} quarter-tones`}
          />
          <Cell
            label="Tuning reference"
            value={`${result.reference_hz.toFixed(1)} Hz`}
            sub={`${result.reference_cents >= 0 ? "+" : ""}${result.reference_cents.toFixed(0)}¢ vs A440`}
          />
        </dl>

        <div className="mt-4">
          <p className="label-mono mb-2">Microtone signature</p>
          {microtonal.length === 0 ? (
            <p className="font-mono text-[12px] text-outline">
              No quarter-tone degrees above threshold — the excerpt reads as 12-TET.
            </p>
          ) : (
            <div className="flex flex-wrap gap-2">
              {microtonal.map((d) => {
                const koron = accidental(d.name) === "koron";
                return (
                  <span
                    key={d.pitch_class}
                    title={`${(d.weight * 100).toFixed(1)}% of sounding time`}
                    className={[
                      "rounded-full border px-2 py-0.5 font-mono text-[11px]",
                      koron
                        ? "border-tertiary-container/60 bg-tertiary-container/10 text-tertiary"
                        : "border-secondary/50 bg-secondary/10 text-secondary",
                    ].join(" ")}
                  >
                    {d.name} ({koron ? "koron" : "sori"}){" "}
                    {d.cents_deviation >= 0 ? "+" : ""}
                    {d.cents_deviation.toFixed(0)}¢
                  </span>
                );
              })}
            </div>
          )}
        </div>

        <p className="mt-5 border-t border-hairline pt-3 font-mono text-[11px] leading-relaxed text-outline">
          {result.n_note_events} note events · {result.n_cadences} cadences ·{" "}
          {(result.voiced_fraction * 100).toFixed(0)}% voiced · grid fit{" "}
          {result.tuning_concentration.toFixed(2)} · {result.duration.toFixed(1)}s @{" "}
          {(result.sample_rate / 1000).toFixed(1)} kHz
        </p>
      </div>
    </section>
  );
}

function Cell({ label, value, sub }: { label: string; value: string; sub: string }) {
  return (
    <div className="bg-surface-container px-3 py-2.5">
      <p className="label-mono">{label}</p>
      <p className="mt-0.5 font-serif text-[20px] text-on-surface">{value}</p>
      <p className="font-mono text-[11px] text-outline">{sub}</p>
    </div>
  );
}
