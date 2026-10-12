import type { AnalysisResult, GushehCandidate } from "../lib/types";
import { formatTime } from "../lib/api";

/**
 * Which melody of the dastgah is being played, over time.
 *
 * A dastgah is a repertoire, not a single tune, and its gushehs share its
 * scale — so this is a shortlist rather than a verdict. Measured against the
 * Karimi radif it puts the right gusheh first 25% of the time and inside the
 * top three 48%, against 4% and 13% for guessing among ~23 candidates.
 */
function Shortlist({ gushehs }: { gushehs: GushehCandidate[] }) {
  if (gushehs.length === 0) {
    return (
      <span className="font-mono text-[11px] text-outline">
        too few notes to rank
      </span>
    );
  }
  const [leading, ...rest] = gushehs;
  return (
    <span className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
      <span className="font-serif text-[14px] text-primary">{leading.name}</span>
      <span className="font-mono text-[11px] text-primary/70">
        {(leading.probability * 100).toFixed(0)}%
      </span>
      {rest.map((g) => (
        <span key={g.name} className="font-mono text-[11px] text-outline">
          · {g.name} {(g.probability * 100).toFixed(0)}%
        </span>
      ))}
    </span>
  );
}

export default function GushehTimeline({ result }: { result: AnalysisResult }) {
  const hasAny =
    result.gushehs.length > 0 || result.segments.some((s) => s.gushehs.length > 0);
  if (!hasAny) return null;

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Gusheh Identification</span>
        <span className="label-mono text-outline">
          {result.name.replace(/^(Dastgāh-e|Āvāz-e)\s*/, "")} repertoire
        </span>
      </header>

      <div className="p-4">
        <p className="mb-3 text-[13px] leading-relaxed text-on-surface/70">
          A dastgāh is a repertoire of gushehs sharing one scale, so these are
          ranked candidates rather than an identification.{" "}
          <span className="text-tertiary">
            The right gusheh is first about a quarter of the time and in the top
            three about half — against 4% and 13% for guessing.
          </span>
        </p>

        {result.gushehs.length > 0 && (
          <div className="mb-4 rounded border border-hairline bg-surface-container px-3 py-2.5">
            <p className="label-mono mb-1.5">Across the whole recording</p>
            <Shortlist gushehs={result.gushehs} />
          </div>
        )}

        {result.segments.some((s) => s.gushehs.length > 0) && (
          <>
            <p className="label-mono mb-2">Segment by segment</p>
            <ol className="space-y-2">
              {result.segments.map((segment, index) => (
                <li
                  key={index}
                  className="border-b border-hairline/40 pb-2 last:border-0 last:pb-0"
                >
                  <div className="mb-1 flex items-baseline gap-2">
                    <span className="font-mono text-[11px] text-on-surface-variant">
                      {formatTime(segment.start)}–{formatTime(segment.end)}
                    </span>
                    <span className="font-serif text-[12px] text-on-surface/70">
                      {segment.name}
                    </span>
                  </div>
                  <Shortlist gushehs={segment.gushehs} />
                </li>
              ))}
            </ol>
          </>
        )}
      </div>
    </section>
  );
}
