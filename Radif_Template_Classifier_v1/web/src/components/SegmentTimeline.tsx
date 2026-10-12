import type { AnalysisResult } from "../lib/types";
import { formatTime } from "../lib/api";

const PALETTE = [
  "bg-primary/70",
  "bg-secondary/60",
  "bg-tertiary-container/55",
  "bg-primary-container/60",
  "bg-secondary-container/60",
];

/**
 * Where the performance changes mode. Extended radif performances modulate by
 * design, so a single label over the whole recording hides real structure.
 *
 * Each window carries less evidence than the whole recording, so these labels
 * are measurably less reliable than the headline classification. The panel says
 * so rather than presenting them as equally trustworthy.
 */
export default function SegmentTimeline({ result }: { result: AnalysisResult }) {
  if (result.segments.length === 0) {
    return (
      <section className="panel">
        <header className="panel-header">
          <span className="label-mono text-secondary">Modal Timeline</span>
        </header>
        <p className="px-4 py-5 font-mono text-[12px] text-outline">
          Recording is shorter than one 20-second analysis window — no timeline to show.
        </p>
      </section>
    );
  }

  const colourFor = new Map<string, string>();
  result.segments.forEach((segment) => {
    if (!colourFor.has(segment.key))
      colourFor.set(segment.key, PALETTE[colourFor.size % PALETTE.length]);
  });

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Modal Timeline</span>
        <span className="label-mono text-outline">
          {result.segments.length} segment{result.segments.length === 1 ? "" : "s"} ·
          20s windows
        </span>
      </header>

      <div className="p-4">
        <p className="mb-3 text-[13px] leading-relaxed text-on-surface/70">
          Overlapping 20-second windows, smoothed so a mode change needs sustained
          evidence.{" "}
          <span className="text-tertiary">
            Each window sees less than the whole recording, so these labels are
            less reliable than the classification above — read them as structure,
            not verdicts.
          </span>
        </p>
        <div className="flex h-9 w-full overflow-hidden rounded border border-hairline">
          {result.segments.map((segment, index) => {
            const width = ((segment.end - segment.start) / result.duration) * 100;
            return (
              <div
                key={index}
                title={`${segment.name} · ${formatTime(segment.start)}–${formatTime(segment.end)} · ${(segment.confidence * 100).toFixed(0)}%`}
                style={{ width: `${width}%` }}
                className={`${colourFor.get(segment.key)} flex items-center justify-center border-r border-background/40 last:border-r-0`}
              >
                <span className="truncate px-1 font-mono text-[10px] text-background">
                  {segment.name}
                </span>
              </div>
            );
          })}
        </div>

        <div className="mt-1 flex justify-between font-mono text-[10px] text-outline">
          <span>0:00</span>
          <span>{formatTime(result.duration)}</span>
        </div>

        <ul className="mt-3 space-y-1">
          {result.segments.map((segment, index) => (
            <li
              key={index}
              className="flex items-baseline justify-between border-b border-hairline/40 pb-1 last:border-0"
            >
              <span className="font-serif text-[13px]">
                <span className={`mr-2 inline-block h-2 w-2 rounded-sm align-middle ${colourFor.get(segment.key)}`} />
                {segment.name}
                <span className="ml-2 font-mono text-[10px] text-outline">
                  tonic {segment.tonic_name}
                </span>
              </span>
              <span className="font-mono text-[11px] text-on-surface-variant">
                {formatTime(segment.start)}–{formatTime(segment.end)} ·{" "}
                {(segment.confidence * 100).toFixed(0)}%
              </span>
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
