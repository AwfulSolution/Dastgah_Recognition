import { useCallback, useRef, useState } from "react";

interface Props {
  onFile: (file: File) => void;
  busy: boolean;
  filename?: string;
}

const ACCEPT = ".wav,.flac,.aiff,.aif,.mp3,.m4a,.ogg,.opus";

export default function Dropzone({ onFile, busy, filename }: Props) {
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer.files?.[0];
      if (file) onFile(file);
    },
    [onFile],
  );

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono">Acoustic Ingestion Stage</span>
        <span className="label-mono text-outline">WAV · FLAC · AIFF · MP3 · ≤100 MB</span>
      </header>

      <div className="p-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => !busy && input.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => e.key === "Enter" && !busy && input.current?.click()}
          aria-label="Upload an audio file for analysis"
          className={[
            "flex cursor-pointer flex-col items-center justify-center gap-3 rounded-lg border-[1.5px] border-dashed px-6 py-10 text-center transition",
            dragging
              ? "border-primary bg-primary/[0.06] shadow-glow-gold"
              : "border-[#283556] hover:border-outline",
            busy ? "pointer-events-none opacity-60" : "",
          ].join(" ")}
        >
          <input
            ref={input}
            type="file"
            accept={ACCEPT}
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) onFile(file);
              e.target.value = "";
            }}
          />

          {busy ? (
            <>
              <div className="h-8 w-8 animate-spin rounded-full border-2 border-primary border-t-transparent" />
              <p className="font-mono text-[13px] text-primary">Tracking pitch contour…</p>
              <p className="label-mono text-outline">
                pYIN f0 · tuning estimation · 24-TET binning
              </p>
            </>
          ) : (
            <>
              <p className="text-[15px]">
                Drag and drop audio here, or{" "}
                <span className="text-primary underline underline-offset-2">
                  browse filesystem
                </span>
              </p>
              <p className="label-mono text-outline">
                Monophonic and heterophonic solo performance analyses best
              </p>
            </>
          )}
        </div>

        {filename && !busy && (
          <div className="mt-3 flex items-center gap-2 rounded border border-hairline bg-surface-container px-3 py-2">
            <span className="font-mono text-[12px] text-secondary">{filename}</span>
          </div>
        )}
      </div>
    </section>
  );
}
