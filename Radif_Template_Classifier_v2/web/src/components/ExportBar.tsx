import type { AnalysisResult } from "../lib/types";

function download(contents: string, filename: string, mime: string) {
  const url = URL.createObjectURL(new Blob([contents], { type: mime }));
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

/** Strip the extension so exports sit next to the recording they describe. */
const stem = (source: string) => source.replace(/\.[^.]+$/, "") || "analysis";

export default function ExportBar({ result }: { result: AnalysisResult }) {
  const base = stem(result.source);

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Export</span>
        <span className="label-mono text-outline">
          {result.n_note_events} events · {result.degrees.length} degrees
        </span>
      </header>

      <div className="flex flex-wrap items-center gap-2 p-4">
        <button
          type="button"
          onClick={() =>
            download(
              JSON.stringify(result, null, 2),
              `${base}.dastgah.json`,
              "application/json",
            )
          }
          className="rounded border border-secondary/50 bg-secondary/10 px-3 py-1.5 font-mono text-[12px] text-secondary transition hover:bg-secondary/20"
        >
          JSON report
        </button>

        <button
          type="button"
          disabled={!result.musicxml}
          title={
            result.musicxml
              ? "The identified scale, with koron and sori notated"
              : "Not available for this analysis"
          }
          onClick={() =>
            result.musicxml &&
            download(
              result.musicxml,
              `${base}.musicxml`,
              "application/vnd.recordare.musicxml+xml",
            )
          }
          className="rounded border border-primary/50 bg-primary/10 px-3 py-1.5 font-mono text-[12px] text-primary transition hover:bg-primary/20 disabled:cursor-not-allowed disabled:opacity-40"
        >
          MusicXML (24-TET)
        </button>

        <p className="ml-auto max-w-md text-right text-[12px] leading-relaxed text-outline">
          JSON carries the full analysis. MusicXML carries the identified{" "}
          <span className="text-on-surface/70">scale</span>, not the melody —
          transcribing rhythm would mean inventing a beat the non-metric radif
          does not have.
        </p>
      </div>
    </section>
  );
}
