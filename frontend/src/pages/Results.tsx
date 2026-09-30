import { Link } from "react-router-dom";
import { RiskBadge } from "../components/RiskBadge";
import type { PredictionResponse } from "../types";
import { humanizeLabel, humanizeOption } from "../utils/format";

export interface StoredInputs {
  disease_key: string;
  values: Record<string, number | string>;
  units: Record<string, string>;
}

function load<T>(key: string): T | null {
  try {
    const raw = sessionStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : null;
  } catch {
    return null;
  }
}


export function Results() {
  const result = load<PredictionResponse>("earlydx:lastResult");
  const inputs = load<StoredInputs>("earlydx:lastInputs");

  if (!result) {
    return (
      <>
        <h1>Risk Assessment</h1>
        <p className="sub">No assessment has been run in this browser session yet.</p>
        <div className="cta-card">
          <div>
            <div className="cta-title">Nothing to show yet</div>
            <p className="cta-sub">Results appear here after you submit an assessment. They are kept only in this tab.</p>
          </div>
          <Link to="/assessment" className="btn-link">Start an assessment</Link>
        </div>
      </>
    );
  }

  const thresholds = result.risk_thresholds;
  const hasZone = thresholds?.low_cut != null && thresholds?.high_cut != null;
  const inputsMatch = inputs && inputs.disease_key === result.disease_key;
  const pctScore = (result.risk_score * 100).toFixed(1);

  return (
    <>
      <span className="eyebrow">Step 04 · Result</span>
      <h1>Risk Assessment</h1>
      <p className="sub">
        Risk estimate for {result.disease}. An estimate only, not a diagnosis.
      </p>

      <RiskBadge result={result} />

      <div className="result-grid">
        <div className="card"><div className="k">Condition</div><div className="v">{result.disease}</div></div>
        <div className="card">
          <div className="k">Prediction</div>
          <div className="v">{result.predicted_class == null ? result.risk_level : result.predicted_class === 1 ? "Positive" : "Negative"}</div>
          <div className="hint">{result.predicted_class == null ? "validation-derived level" : `at the ${result.decision_threshold?.toFixed(2) ?? "0.50"} decision threshold`}</div>
        </div>
        <div className="card">
          <div className="k">{result.calibrated ? "Probability" : "Model score"}</div>
          <div className="v">{pctScore}%</div>
          <div className="hint">{result.calibrated ? "calibrated risk score" : "uncalibrated ranking score"}</div>
        </div>
      </div>

      <div className="disclaimer">{result.disclaimer} This is a statistical research prototype; it has not been clinically
        validated and must not replace professional medical advice.</div>

      <div className="two-col">
        <div>
          <h2>Prediction summary</h2>
          <table>
            <tbody>
              <tr>
                <th>Model score</th>
                <td>
                  {result.risk_score.toFixed(3)} ({pctScore}%)
                  <div className="muted">
                    {result.calibrated
                      ? "Calibrated probability: among training-like cases scoring this, roughly this share were positive."
                      : "Uncalibrated model output — a ranking score, not a probability."}
                  </div>
                </td>
              </tr>
              <tr><th>Risk level</th><td>{result.risk_level} (validation-derived bands, see below)</td></tr>
              {result.predicted_class != null && (
                <tr>
                  <th>Binary classification</th>
                  <td>
                    {result.predicted_class === 1 ? "Positive" : "Negative"}
                    {result.decision_threshold != null && (
                      <span className="muted"> at the {result.decision_threshold.toFixed(2)} threshold used for the reported accuracy / precision / recall / F1</span>
                    )}
                  </td>
                </tr>
              )}
              <tr><th>Calibration</th><td>{result.calibrated ? "calibrated" : "not calibrated"}</td></tr>
            </tbody>
          </table>
        </div>

        <div>
          <h2>Your inputs</h2>
          {inputsMatch ? (
            <table>
              <tbody>
                {Object.entries(inputs!.values).map(([k, v]) => (
                  <tr key={k}>
                    <th>{humanizeLabel(k)}</th>
                    <td>
                      {typeof v === "string" ? humanizeOption(v) : v}
                      {inputs!.units[k] && <span className="muted"> {inputs!.units[k]}</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="muted">Input values for this result are not available in this session.</p>
          )}
        </div>
      </div>

      {hasZone && (
        <>
          <h2>Where the score falls</h2>
          <p className="muted">
            Bands come from this model's own validation run (targets: sensitivity and specificity ≥ 0.85) —
            they are not clinical risk categories.
          </p>
          <div className="zone-bar" aria-label={`Score ${result.risk_score.toFixed(3)} on a 0 to 1 scale`}>
            <div className="zone-seg zone-low" style={{ flexGrow: thresholds!.low_cut! }}>LOW</div>
            <div className="zone-seg zone-moderate" style={{ flexGrow: thresholds!.high_cut! - thresholds!.low_cut! }}>MODERATE</div>
            <div className="zone-seg zone-high" style={{ flexGrow: 1 - thresholds!.high_cut! }}>HIGH</div>
          </div>
          <div className="zone-marker-track" aria-hidden="true">
            <span className="zone-marker" style={{ left: `${result.risk_score * 100}%` }} title={`your score ${result.risk_score.toFixed(3)}`} />
          </div>
          <div className="zone-labels">
            <span>0.0</span>
            <span>{thresholds!.low_cut!.toFixed(3)}</span>
            <span>{thresholds!.high_cut!.toFixed(3)}</span>
            <span>1.0</span>
          </div>
          {result.threshold_policy && <p className="muted">Policy from the training run: {result.threshold_policy}</p>}
        </>
      )}

      <h2>Inputs the model relies on most</h2>
      <p className="muted">
        <strong>Method:</strong> global permutation importance of the trained model (computed once during
        training on held-out data), combined with where your value sits against the training population's
        median and interquartile range. This describes what the model weighs in general — it is{" "}
        <strong>not</strong> a per-case causal explanation and does not use SHAP or per-prediction attribution.
      </p>
      {result.important_features.length ? (
        <div className="table-scroll">
          <table>
            <thead>
              <tr><th>Feature</th><th>Your value</th><th>Training median</th><th>Relative position</th><th>Model reliance</th></tr>
            </thead>
            <tbody>
              {result.important_features.map((f) => (
                <tr key={f.feature}>
                  <td>{humanizeLabel(f.feature)}</td>
                  <td>{f.value}</td>
                  <td>{f.population_median ?? "—"}</td>
                  <td>{f.direction}</td>
                  <td>
                    <span className={`pill ${f.impact === "high" ? "no" : ""}`}>{f.impact}</span>{" "}
                    <span className="muted mono">({f.importance})</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="muted">No importance data was stored for this model.</p>
      )}

      <div className="form-actions">
        <Link to="/assessment" className="btn-link">Run another assessment</Link>
      </div>
    </>
  );
}
