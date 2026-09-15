/** Mirrors the naming in `dastgah.theory`. */
export const PITCH_CLASS_NAMES = [
  "C", "Cs", "Db", "Dk", "D", "Ds", "Eb", "Ek", "E", "Es", "F", "Fs",
  "F#", "Gk", "G", "Gs", "Ab", "Ak", "A", "As", "Bb", "Bk", "B", "Bs",
];

export const pitchClassName = (pc: number) => PITCH_CLASS_NAMES[((pc % 24) + 24) % 24];

export const isMicrotonal = (interval: number) => Math.abs(interval % 2) === 1;

/** Koron lowers a quarter-tone, sori raises one. */
export function accidental(name: string): "koron" | "sori" | null {
  if (name.endsWith("k")) return "koron";
  if (name.endsWith("s")) return "sori";
  return null;
}
