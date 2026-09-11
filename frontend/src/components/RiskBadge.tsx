import type { PredictionResponse } from "../types";

const COPY: Record<string, string> = {
  HIGH: "The model estimates a high risk based on the information provided.",
  MODERATE: "The model estimates a moderate risk based on the information provided.",
  LOW: "The model estimates a low risk based on the information provided.",
};

export function RiskBadge({ result }: { result: PredictionResponse }) {
  const pct = Math.round(result.risk_score * 100);
  return (
    <div className={`risk-banner risk-${result.risk_level}`}>
      <div className="lvl">{result.risk_level} RISK ESTIMATE</div>
      <div className="risk-disease">{result.disease}</div>
      <div className="risk-score-block">
        <div className="risk-score-label">
          {result.calibrated ? "Estimated risk score (calibrated)" : "Estimated risk score"}
        </div>
        <div className="risk-score-pct">{pct}%</div>
      </div>
      <div>{COPY[result.risk_level]}</div>
      <div className="score">Model: {result.model_version}</div>
    </div>
  );
}
