import type { AnalysisResult } from "./types";

export class ApiError extends Error {}

/** Upload a recording for analysis. */
export async function analyzeFile(file: File): Promise<AnalysisResult> {
  const body = new FormData();
  body.append("file", file);

  let response: Response;
  try {
    response = await fetch("/api/analyze", { method: "POST", body });
  } catch {
    throw new ApiError(
      "Could not reach the analysis server. Start it with: uvicorn dastgah.api.server:app",
    );
  }

  if (!response.ok) {
    let detail = `Analysis failed (HTTP ${response.status})`;
    try {
      const payload = await response.json();
      if (payload?.detail) detail = String(payload.detail);
    } catch {
      /* keep the status-code message */
    }
    throw new ApiError(detail);
  }

  return (await response.json()) as AnalysisResult;
}

export const formatTime = (seconds: number) => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${String(s).padStart(2, "0")}`;
};
