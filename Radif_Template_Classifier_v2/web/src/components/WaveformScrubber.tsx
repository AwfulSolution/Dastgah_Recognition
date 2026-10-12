import { useCallback, useEffect, useRef, useState } from "react";
import type { AnalysisResult } from "../lib/types";
import { formatTime } from "../lib/api";

/** Canvas needs literal colours; these track the DESIGN.md tokens. */
const COLOURS = {
  wave: "#2f3445",
  wavePlayed: "#59dad1",
  playhead: "#f2ca50",
  grid: "rgba(40, 53, 86, 0.6)",
};

/** Segment bands, cycling so neighbouring modes stay distinguishable. */
const BANDS = [
  "rgba(242, 202, 80, 0.16)",
  "rgba(89, 218, 209, 0.14)",
  "rgba(255, 151, 127, 0.14)",
  "rgba(212, 175, 55, 0.14)",
];

/** Reduce raw samples to per-pixel peaks, which is all the canvas can show. */
function toPeaks(buffer: AudioBuffer, buckets: number): Float32Array {
  const channel = buffer.getChannelData(0);
  const size = Math.floor(channel.length / buckets) || 1;
  const peaks = new Float32Array(buckets);
  for (let i = 0; i < buckets; i++) {
    let max = 0;
    const start = i * size;
    const end = Math.min(start + size, channel.length);
    for (let j = start; j < end; j++) {
      const value = Math.abs(channel[j]);
      if (value > max) max = value;
    }
    peaks[i] = max;
  }
  return peaks;
}

interface Props {
  file: File;
  result: AnalysisResult;
}

export default function WaveformScrubber({ file, result }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const audioRef = useRef<HTMLAudioElement>(null);
  const [peaks, setPeaks] = useState<Float32Array | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [playing, setPlaying] = useState(false);
  const [time, setTime] = useState(0);

  const duration = result.duration || 1;

  // Decode once per file. AudioContext is closed on cleanup so repeated
  // uploads do not leak decoders.
  useEffect(() => {
    let cancelled = false;
    const context = new AudioContext();
    setPeaks(null);
    setError(null);

    file
      .arrayBuffer()
      .then((bytes) => context.decodeAudioData(bytes))
      .then((buffer) => {
        if (!cancelled) setPeaks(toPeaks(buffer, 1400));
      })
      .catch(() => {
        if (!cancelled) setError("This browser could not decode the audio for display.");
      })
      .finally(() => void context.close());

    return () => {
      cancelled = true;
    };
  }, [file]);

  // Object URL for playback, revoked when the file changes.
  const [src, setSrc] = useState<string>();
  useEffect(() => {
    const url = URL.createObjectURL(file);
    setSrc(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !peaks) return;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    canvas.width = width * ratio;
    canvas.height = height * ratio;

    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.scale(ratio, ratio);
    ctx.clearRect(0, 0, width, height);

    // Modal segments behind the waveform, so a mode change is visible.
    const colourFor = new Map<string, string>();
    result.segments.forEach((segment) => {
      if (!colourFor.has(segment.key))
        colourFor.set(segment.key, BANDS[colourFor.size % BANDS.length]);
      ctx.fillStyle = colourFor.get(segment.key)!;
      const x = (segment.start / duration) * width;
      ctx.fillRect(x, 0, ((segment.end - segment.start) / duration) * width, height);
    });

    const middle = height / 2;
    ctx.strokeStyle = COLOURS.grid;
    ctx.beginPath();
    ctx.moveTo(0, middle);
    ctx.lineTo(width, middle);
    ctx.stroke();

    const playedTo = (time / duration) * width;
    for (let x = 0; x < width; x++) {
      const peak = peaks[Math.floor((x / width) * peaks.length)] ?? 0;
      const amplitude = Math.max(peak * middle * 0.92, 0.5);
      ctx.strokeStyle = x <= playedTo ? COLOURS.wavePlayed : COLOURS.wave;
      ctx.beginPath();
      ctx.moveTo(x + 0.5, middle - amplitude);
      ctx.lineTo(x + 0.5, middle + amplitude);
      ctx.stroke();
    }

    ctx.strokeStyle = COLOURS.playhead;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(playedTo, 0);
    ctx.lineTo(playedTo, height);
    ctx.stroke();
    ctx.lineWidth = 1;
  }, [peaks, time, duration, result.segments]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, [draw]);

  const seek = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const audio = audioRef.current;
    const canvas = canvasRef.current;
    if (!audio || !canvas) return;
    const bounds = canvas.getBoundingClientRect();
    const fraction = (event.clientX - bounds.left) / bounds.width;
    audio.currentTime = Math.max(0, Math.min(1, fraction)) * duration;
    setTime(audio.currentTime);
  };

  const toggle = () => {
    const audio = audioRef.current;
    if (!audio) return;
    if (audio.paused) void audio.play();
    else audio.pause();
  };

  const segmentAt = result.segments.find((s) => time >= s.start && time < s.end);

  return (
    <section className="panel">
      <header className="panel-header justify-between">
        <span className="label-mono text-secondary">Waveform &amp; Transport</span>
        <span className="label-mono text-outline">{result.source}</span>
      </header>

      <div className="p-4">
        <div className="mb-3 flex items-center gap-3">
          <button
            type="button"
            onClick={toggle}
            aria-label={playing ? "Pause" : "Play"}
            className="flex h-9 w-9 shrink-0 items-center justify-center rounded border border-primary/50 bg-primary/10 text-primary transition hover:bg-primary/20"
          >
            <span className="font-mono text-[13px]">{playing ? "❚❚" : "▶"}</span>
          </button>

          <span className="font-mono text-[13px] text-on-surface">
            {formatTime(time)}{" "}
            <span className="text-outline">/ {formatTime(duration)}</span>
          </span>

          {segmentAt && (
            <span className="ml-auto flex items-baseline gap-2">
              <span className="label-mono">playing</span>
              <span className="font-serif text-[14px] text-primary">
                {segmentAt.name}
              </span>
              {segmentAt.gushehs[0] && (
                <span className="font-mono text-[11px] text-outline">
                  · {segmentAt.gushehs[0].name}?
                </span>
              )}
            </span>
          )}
        </div>

        {error ? (
          <p className="rounded border border-hairline bg-surface-container px-3 py-6 text-center font-mono text-[12px] text-outline">
            {error}
          </p>
        ) : peaks ? (
          <canvas
            ref={canvasRef}
            onClick={seek}
            role="slider"
            tabIndex={0}
            aria-label="Seek within the recording"
            aria-valuemin={0}
            aria-valuemax={Math.round(duration)}
            aria-valuenow={Math.round(time)}
            onKeyDown={(e) => {
              const audio = audioRef.current;
              if (!audio) return;
              if (e.key === " ") {
                e.preventDefault();
                toggle();
              }
              if (e.key === "ArrowRight") audio.currentTime += 5;
              if (e.key === "ArrowLeft") audio.currentTime -= 5;
            }}
            className="h-24 w-full cursor-pointer rounded border border-hairline bg-surface-container-lowest"
          />
        ) : (
          <div className="flex h-24 items-center justify-center rounded border border-hairline bg-surface-container-lowest">
            <span className="font-mono text-[12px] text-outline">
              decoding waveform…
            </span>
          </div>
        )}

        <div className="mt-1 flex justify-between font-mono text-[10px] text-outline">
          <span>0:00</span>
          <span>click to seek · space to play · ← → to skip 5s</span>
          <span>{formatTime(duration)}</span>
        </div>

        <audio
          ref={audioRef}
          src={src}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onTimeUpdate={(e) => setTime(e.currentTarget.currentTime)}
          onEnded={() => setPlaying(false)}
          className="hidden"
        />
      </div>
    </section>
  );
}
