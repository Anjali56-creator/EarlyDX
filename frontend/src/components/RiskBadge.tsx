import type { PredictionResponse } from "../types";

const COPY: Record<string, string> = {
  HIGH: "The model estimates a HIGH risk based on the information provided.",
  MODERATE: "The model estimates a MODERATE risk based on the information provided.",
  LOW: "The model estimates a LOW risk based on the information provided.",
};

export function RiskBadge({ result }: { result: PredictionResponse }) {
  return (
    <div className={`risk-banner risk-${result.risk_level}`}>
      <div className="lvl">{result.risk_level} RISK — {result.disease}</div>
      <div>{COPY[result.risk_level]}</div>
      <div className="score">
        {result.calibrated ? "Calibrated risk score" : "Risk score"}: {result.risk_score.toFixed(3)}
        {" · "}model {result.model_version}
      </div>
    </div>
  );
}
