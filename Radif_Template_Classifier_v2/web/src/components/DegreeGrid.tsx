import type { AnalysisResult } from "../lib/types";
import { pitchClassName, isMicrotonal } from "../lib/theory";

/**
 * The 24 quarter-tone degrees above the tonic. Degrees the performance actually
 * used are filled in proportion to their share of sounding time; the rest stay
 * empty, so the mode's scale reads directly off the grid.
 */
export default function DegreeGrid({ result }: { result: AnalysisResult }) {
  const byInterval = new Map(result.degrees.map((d) => [d.interval, d]));
  const maxWeight = Math.max(...result.degrees.map((d) => d.weight), 1e-6);

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">24 Microtonal Scale Degrees</span>
        <span className="label-mono text-outline">
          tonic {result.tonic_name} = {result.tonic_hz.toFixed(1)} Hz
        </span>
      </header>

      <div className="p-4">
        <div className="grid grid-cols-6 gap-1.5 sm:grid-cols-12">
          {Array.from({ length: 24 }, (_, interval) => {
            const degree = byInterval.get(interval);
            const micro = isMicrotonal(interval);
            const name = pitchClassName(result.tonic_pc + interval);
            const isTonic = interval === 0;
            const isShahed = interval === result.shahed_interval;
            const fill = degree ? (degree.weight / maxWeight) * 100 : 0;

            return (
              <div
                key={interval}
                title={
                  degree
                    ? `+${interval} quarter-tones · ${(degree.weight * 100).toFixed(1)}% · ${degree.cents_deviation >= 0 ? "+" : ""}${degree.cents_deviation}¢`
                    : `+${interval} quarter-tones · unused`
                }
                className={[
                  "relative flex h-14 flex-col items-center justify-center overflow-hidden rounded border",
                  degree
                    ? isTonic
                      ? "border-primary bg-primary/10 shadow-rim"
                      : micro
                        ? "border-secondary/60 bg-secondary/[0.07]"
                        : "border-hairline bg-surface-container"
                    : "border-hairline/50 bg-surface-container-lowest",
                ].join(" ")}
              >
                <div
                  className={`absolute inset-x-0 bottom-0 ${isTonic ? "bg-primary/30" : micro ? "bg-secondary/25" : "bg-surface-container-highest/70"}`}
                  style={{ height: `${fill}%` }}
                />
                <span
                  className={[
                    "relative font-mono text-[11px]",
                    degree
                      ? isTonic
                        ? "text-primary"
                        : micro
                          ? "text-secondary"
                          : "text-on-surface"
                      : "text-outline/50",
                  ].join(" ")}
                >
                  {name}
                </span>
                <span className="relative font-mono text-[9px] text-outline">
                  +{interval}
                </span>
                {isShahed && !isTonic && (
                  <span className="absolute right-1 top-1 h-1 w-1 rounded-full bg-primary" />
                )}
              </div>
            );
          })}
        </div>

        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[10px] text-outline">
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-sm border border-primary bg-primary/20 align-middle" />
            tonic (ist)
          </span>
          <span>
            <span className="mr-1 inline-block h-2 w-2 rounded-sm border border-secondary/60 bg-secondary/20 align-middle" />
            quarter-tone degree (koron / sori)
          </span>
          <span>
            <span className="mr-1 inline-block h-1 w-1 rounded-full bg-primary align-middle" />
            shāhed
          </span>
        </div>
      </div>
    </section>
  );
}
