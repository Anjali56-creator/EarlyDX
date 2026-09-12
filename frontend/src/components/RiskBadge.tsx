import type { PredictionResponse } from "../types";

const COPY: Record<string, { icon: string; label: string; text: string }> = {
  HIGH: { icon: "▲", label: "High risk estimate", text: "The model's score is above its validation-derived HIGH cut-off for this condition." },
  MODERATE: { icon: "●", label: "Moderate risk estimate", text: "The model's score falls between its validation-derived LOW and HIGH cut-offs." },
  LOW: { icon: "▼", label: "Low risk estimate", text: "The model's score is below its validation-derived LOW cut-off for this condition." },
};

/** Risk is communicated with explicit wording and an icon; the tint is only a
 * secondary cue so the level is never colour-alone. */
export function RiskBadge({ result }: { result: PredictionResponse }) {
  const pct = Math.round(result.risk_score * 100);
  const c = COPY[result.risk_level] ?? COPY.MODERATE;
  return (
    <div className={`risk-banner risk-${result.risk_level}`} role="status">
      <div>
        <div className="lvl"><span className="risk-icon" aria-hidden="true">{c.icon}</span>{c.label}</div>
        <div className="risk-disease">{result.disease}</div>
        <div>{c.text}</div>
        <div className="score">Model {result.model_version} · {result.model_algorithm ?? "—"} · {result.calibrated ? "calibrated probability" : "uncalibrated score"}</div>
      </div>
      <div className="risk-score-block">
        <div className="risk-score-label">{result.calibrated ? "Estimated risk (calibrated)" : "Model score"}</div>
        <div className="risk-score-pct">{pct}%</div>
        <div className="risk-score-label">{result.risk_level} RISK ESTIMATE</div>
      </div>
    </div>
  );
}
