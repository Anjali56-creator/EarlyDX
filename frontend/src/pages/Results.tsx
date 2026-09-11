import { Link } from "react-router-dom";
import { RiskBadge } from "../components/RiskBadge";
import type { PredictionResponse } from "../types";

function load(): PredictionResponse | null {
  try {
    const raw = sessionStorage.getItem("earlydx:lastResult");
    return raw ? (JSON.parse(raw) as PredictionResponse) : null;
  } catch {
    return null;
  }
}

export function Results() {
  const result = load();

  if (!result) {
    return (
      <>
        <h1>Risk Assessment</h1>
        <p className="sub">No assessment yet. <Link to="/assessment">Start a new assessment</Link>.</p>
      </>
    );
  }

  const perf = result.model_performance;
  const thresholds = result.risk_thresholds;
  const hasZone = thresholds?.low_cut != null && thresholds?.high_cut != null;

  return (
    <>
      <h1>Risk Assessment</h1>
      <p className="sub">This is a model-based estimate, not a diagnosis.</p>

      <RiskBadge result={result} />

      {hasZone && (
        <>
          <h2>Risk zone</h2>
          <p className="muted">
            These bands come from this model's own validation-derived thresholds (see below) —
            they are not clinical risk categories.
          </p>
          <div className="zone-bar">
            <div className="zone-seg zone-low" style={{ flexGrow: thresholds!.low_cut! }}>LOW</div>
            <div className="zone-seg zone-moderate" style={{ flexGrow: thresholds!.high_cut! - thresholds!.low_cut! }}>MODERATE</div>
            <div className="zone-seg zone-high" style={{ flexGrow: 1 - thresholds!.high_cut! }}>HIGH</div>
          </div>
          <div className="zone-labels">
            <span>0.0</span>
            <span>{thresholds!.low_cut!.toFixed(3)}</span>
            <span>{thresholds!.high_cut!.toFixed(3)}</span>
            <span>1.0</span>
          </div>
        </>
      )}

      <h2>Classification</h2>
      <table>
        <tbody>
          <tr><th>Model score</th><td>{result.risk_score.toFixed(3)}</td></tr>
          {result.predicted_class != null && (
            <tr><th>Classification</th><td>{result.predicted_class === 1 ? "Positive prediction" : "Negative prediction"}</td></tr>
          )}
          {result.decision_threshold != null && (
            <tr><th>Decision threshold</th><td>{result.decision_threshold.toFixed(2)} (the cut used for this model's reported accuracy/precision/recall)</td></tr>
          )}
        </tbody>
      </table>

      <h2>Why this result was produced</h2>
      <p className="muted">
        Ranked by the trained model's permutation importance. "Direction" compares your value with the
        median of the model's training population. These factors describe model behaviour, not causation.
      </p>
      <ul className="factors">
        {result.important_features.map((f) => (
          <li key={f.feature}>
            <strong>{f.feature}</strong>: {f.value}
            {f.population_median != null && <> (typical ≈ {f.population_median})</>} — {f.direction},{" "}
            {f.impact} impact
          </li>
        ))}
      </ul>

      <h2>Model information</h2>
      <table>
        <tbody>
          <tr><th>Model version</th><td className="mono">{result.model_version}</td></tr>
          <tr><th>Algorithm</th><td>{result.model_algorithm ?? "—"}</td></tr>
          <tr><th>Calibration</th><td>{result.calibrated ? "calibrated" : "not calibrated"}</td></tr>
          <tr><th>Test ROC-AUC</th><td>{perf.test_roc_auc ?? "—"}</td></tr>
          <tr><th>Test PR-AUC</th><td>{perf.test_pr_auc ?? "—"}</td></tr>
          <tr><th>Sensitivity (recall)</th><td>{perf.test_recall_sensitivity ?? "—"}</td></tr>
          <tr><th>Specificity</th><td>{perf.test_specificity ?? "—"}</td></tr>
          <tr><th>CV ROC-AUC (mean)</th><td>{perf.cv_roc_auc_mean ?? "—"}</td></tr>
        </tbody>
      </table>
      {result.threshold_policy && <p className="muted">Risk-zone policy: {result.threshold_policy}</p>}

      <div className="disclaimer" style={{ marginTop: 20 }}>{result.disclaimer}</div>
    </>
  );
}
