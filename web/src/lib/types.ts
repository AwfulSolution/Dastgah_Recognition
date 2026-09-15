/** Mirrors `dastgah.core.analyze.AnalysisResult`. */

export interface Degree {
  interval: number; // quarter-tones above the tonic
  pitch_class: number;
  name: string;
  weight: number; // share of sounding time
  microtonal: boolean; // koron or sori
  cents_deviation: number;
}

export interface GushehCandidate {
  name: string;
  probability: number;
}

export interface Segment {
  start: number;
  end: number;
  key: string;
  name: string;
  tonic_name: string;
  confidence: number;
  gushehs: GushehCandidate[];
  musicxml?: string;
}

export interface LedgerRow {
  key: string;
  name: string;
  short_name: string;
  persian: string;
  kind: "dastgah" | "avaz" | string;
  family: string;
  probability: number;
}

export interface AnalysisResult {
  source: string;
  duration: number;
  sample_rate: number;
  reference_hz: number;
  reference_cents: number;
  tuning_concentration: number;
  voiced_fraction: number;
  n_note_events: number;
  n_cadences: number;

  key: string;
  name: string;
  short_name: string;
  persian: string;
  kind: string;
  confidence: number;
  family: string;
  family_name: string;
  family_confidence: number;
  family_members: string[];
  tonic_pc: number;
  tonic_name: string;
  tonic_hz: number;
  shahed_interval: number;
  shahed_name: string;

  ledger: LedgerRow[];
  gushehs: GushehCandidate[];
  degrees: Degree[];
  segments: Segment[];

  /** Identified scale as MusicXML; supplied by the API, absent otherwise. */
  musicxml?: string;
}
