import { useCallback, useState } from "react";
import Dropzone from "./components/Dropzone";
import ClassificationPanel from "./components/ClassificationPanel";
import ProbabilityLedger from "./components/ProbabilityLedger";
import DegreeGrid from "./components/DegreeGrid";
import SegmentTimeline from "./components/SegmentTimeline";
import GushehTimeline from "./components/GushehTimeline";
import ExportBar from "./components/ExportBar";
import WaveformScrubber from "./components/WaveformScrubber";
import { analyzeFile, ApiError } from "./lib/api";
import type { AnalysisResult } from "./lib/types";

export default function App() {
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [audio, setAudio] = useState<File>();

  const handleFile = useCallback(async (file: File) => {
    setBusy(true);
    setError(null);
    setAudio(file);
    try {
      setResult(await analyzeFile(file));
    } catch (caught) {
      setError(
        caught instanceof ApiError ? caught.message : "Unexpected error during analysis.",
      );
      setResult(null);
    } finally {
      setBusy(false);
    }
  }, []);

  return (
    <div className="min-h-screen bg-[#070a12]">
      <header className="sticky top-0 z-10 border-b border-hairline bg-[#070a12]/90 backdrop-blur">
        <div className="mx-auto flex max-w-[1400px] items-center justify-between px-6 py-3">
          <div className="flex items-baseline gap-3">
            <h1 className="font-serif text-[19px] font-semibold text-primary">
              Dastgāh AI
            </h1>
            <span className="label-mono">Modal Audio Intelligence</span>
          </div>
          <span className="label-mono text-outline">
            Radif Corpus · 229 gushehs · 6 dastgāhs
          </span>
        </div>
      </header>

      <main className="mx-auto max-w-[1400px] px-6 py-6">
        <section className="panel mb-5 p-5">
          <h2 className="font-serif text-[26px] font-semibold leading-tight text-primary">
            Dastgāh &amp; Āvāz Audio Classifier
          </h2>
          <p className="mt-2 max-w-3xl text-[14px] leading-relaxed text-on-surface/75">
            Upload a solo performance — tār, setār, ney, kamānche, santur or vocal
            āvāz — to identify its modal class, tonic and quarter-tone degrees. Pitch
            is analysed in 24-tone equal temperament against scale templates derived
            from Mirza Abdollah's radif.
          </p>
          <div className="mt-3 flex flex-wrap gap-2 font-mono text-[10px] uppercase tracking-wider">
            {["7 dastgāhs", "6 āvāzes", "24-TET koron / sori", "tonic estimation"].map(
              (chip) => (
                <span
                  key={chip}
                  className="rounded border border-hairline bg-surface-container px-2 py-1 text-on-surface-variant"
                >
                  {chip}
                </span>
              ),
            )}
          </div>
        </section>

        <div className="grid gap-5 lg:grid-cols-2">
          <Dropzone onFile={handleFile} busy={busy} filename={audio?.name} />

          <section className="panel p-5">
            <p className="label-mono mb-2 text-tertiary">Interpreting the output</p>
            <p className="text-[13px] leading-relaxed text-on-surface/75">
              Confidence is calibrated against measured accuracy, so a reading near
              50% genuinely means a coin-flip between the top candidates — read the
              ledger, not just the headline. Each āvāz is folded into its mother
              dastgāh rather than reported separately: āvāz readings run 0–22%
              accurate, while their templates remain useful as evidence{" "}
              <em>for</em> the parent. Folding beats leaving them out by{" "}
              <span className="font-mono text-primary">5 to 15 points</span>.
            </p>
            <p className="mt-2 text-[13px] leading-relaxed text-on-surface/75">
              The limit is structural. Modes that share a pitch collection cannot
              be separated by measuring pitch content, however finely. Among
              these six that leaves exactly one unresolved pair — Shūr and Navā,
              the same collection rotated by a fourth — which the family reading
              reports honestly instead of guessing between.{" "}
              <span className="text-tertiary">
                Inside that pair, read the ranking as a shortlist, not a verdict.
              </span>
            </p>
          </section>
        </div>

        {error && (
          <div
            role="alert"
            className="mt-5 rounded-lg border border-error/40 bg-error-container/20 px-4 py-3"
          >
            <p className="font-mono text-[12px] text-error">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-5 grid gap-5 lg:grid-cols-2">
            <ClassificationPanel result={result} />
            <ProbabilityLedger result={result} />
            {audio && (
              <div className="lg:col-span-2">
                <WaveformScrubber file={audio} result={result} />
              </div>
            )}
            <div className="lg:col-span-2">
              <DegreeGrid result={result} />
            </div>
            <div className="lg:col-span-2">
              <SegmentTimeline result={result} />
            </div>
            <div className="lg:col-span-2">
              <GushehTimeline result={result} />
            </div>
            <div className="lg:col-span-2">
              <ExportBar result={result} />
            </div>
          </div>
        )}
      </main>

      <footer className="mx-auto max-w-[1400px] px-6 pb-8 pt-2">
        <p className="font-mono text-[10px] leading-relaxed text-outline">
          Templates derived from the Radif Corpus (Kanani et al., Zenodo
          10.5281/zenodo.15742125, CC-BY-4.0), transcribed after Dariush Talai's
          notation of the Mirza Abdollah radif.
        </p>
      </footer>
    </div>
  );
}
