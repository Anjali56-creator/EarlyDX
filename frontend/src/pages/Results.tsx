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
        <h1>Results</h1>
        <p className="sub">No assessment yet. <Link to="/assessment">Run one</Link>.</p>
      </>
    );
  }

  const perf = result.model_performance;

  return (
    <>
      <h1>Results</h1>
      <RiskBadge result={result} />

      <h2>Contributing factors</h2>
      <p className="muted">
        Ranked by the trained model's permutation importance. "Direction" compares your value with the
        median of the model's training population. This describes model behaviour, not causation.
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

      <h2>Model</h2>
      <table>
        <tbody>
          <tr><th>Version</th><td>{result.model_version} ({result.model_algorithm})</td></tr>
          <tr><th>Calibrated</th><td>{result.calibrated ? "yes" : "no"}</td></tr>
          <tr><th>Test ROC-AUC</th><td>{perf.test_roc_auc ?? "—"}</td></tr>
          <tr><th>Test PR-AUC</th><td>{perf.test_pr_auc ?? "—"}</td></tr>
          <tr><th>Test recall / sensitivity</th><td>{perf.test_recall_sensitivity ?? "—"}</td></tr>
          <tr><th>Test specificity</th><td>{perf.test_specificity ?? "—"}</td></tr>
          <tr><th>CV ROC-AUC (mean)</th><td>{perf.cv_roc_auc_mean ?? "—"}</td></tr>
        </tbody>
      </table>
      {result.threshold_policy && <p className="muted">Risk levels: {result.threshold_policy}</p>}

      <div className="disclaimer" style={{ marginTop: 20 }}>{result.disclaimer}</div>
    </>
  );
}
