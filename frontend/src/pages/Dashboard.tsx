import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { DatasetEntry, DiseaseInfo, EvaluationSummary, Health, ModelEntry, RecentAssessments } from "../types";

const fmt = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));

const PIPELINE: { step: string; what: string; to: string }[] = [
  { step: "Problem", what: "Early, transparent risk screening across 12 conditions — without pretending to diagnose.", to: "/about" },
  { step: "Data", what: "Public, de-identified datasets with documented source, licence and limitations.", to: "/datasets" },
  { step: "ML models", what: "One leakage-safe scikit-learn pipeline per condition; 3 algorithms compared per condition.", to: "/models" },
  { step: "Evaluation", what: "Validation-split comparison, one held-out test, cross-validation, calibration.", to: "/validation" },
  { step: "Risk prediction", what: "Score + validation-derived risk level from the selected, calibrated model.", to: "/assessment" },
  { step: "Results", what: "Score, inputs, model reliance, and the model's own test metrics side by side.", to: "/results" },
];

export function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [evals, setEvals] = useState<EvaluationSummary[]>([]);
  const [recent, setRecent] = useState<RecentAssessments | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.health(),
      api.diseases(),
      api.models(),
      api.datasets(),
      api.evaluationSummary().catch(() => ({ count: 0, models: [] as EvaluationSummary[] })),
      api.recentAssessments(8).catch(() => null),
    ])
      .then(([h, d, m, ds, ev, r]) => {
        setHealth(h);
        setDiseases(d.diseases);
        setModels(m.models);
        setDatasets(ds.datasets);
        setEvals(ev.models);
        setRecent(r);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [attempt]);

  const withModel = diseases.filter((d) => d.model_available).length;
  const withoutModel = diseases.filter((d) => !d.model_available);
  const verifiedDatasets = datasets.filter((d) => d.status === "verified").length;
  const totalRecords = datasets.reduce((n, d) => n + (Number(d.records) || 0), 0);
  const algorithms = Array.from(new Set(models.map((m) => m.algorithm))).sort();
  const baselineKept = evals.filter((e) => e.selected_algorithm === e.baseline_algorithm).length;
  const displayName = (key: string) => diseases.find((d) => d.key === key)?.display_name ?? key;

  return (
    <>
      <h1>EarlyDX — multi-disease early risk assessment</h1>
      <p className="sub">
        A research prototype that compares machine-learning classifiers per condition, deploys the best-justified
        one, and shows every number it relies on.
      </p>
      <div className="disclaimer">
        EarlyDX produces statistical risk estimates from public datasets. It is not a medical device, is not
        clinically validated, and does not diagnose.
      </div>

      {error && <ApiErrorBox message={error} onRetry={() => setAttempt((a) => a + 1)} />}
      {loading && !error && <Loading what="dashboard" />}

      <div className="cta-card">
        <div>
          <div className="cta-title">Start a new assessment</div>
          <p className="cta-sub">
            Choose one of {withModel || "the"} supported conditions, enter only the inputs its model needs, and get a
            calibrated risk score with the model's own validation metrics.
          </p>
        </div>
        <Link to="/assessment" className="btn-link">New Assessment</Link>
      </div>

      <h2>How EarlyDX works</h2>
      <ol className="pipeline">
        {PIPELINE.map((p, i) => (
          <li key={p.step}>
            <Link to={p.to} className="pipeline-step">
              <span className="pipeline-num">{i + 1}</span>
              <span className="pipeline-title">{p.step}</span>
              <span className="pipeline-what">{p.what}</span>
            </Link>
          </li>
        ))}
      </ol>

      <h2>At a glance</h2>
      <div className="grid">
        <div className="card">
          <div className="k">Conditions in scope</div>
          <div className="v">{diseases.length || "—"}</div>
          <div className="hint">{withModel} with a trained model · {withoutModel.length} documented as unavailable</div>
        </div>
        <div className="card">
          <div className="k">Models served</div>
          <div className="v">{health ? health.models_loaded : "—"}</div>
          <div className="hint">{algorithms.length ? `Algorithms in use: ${algorithms.join(", ")}` : "—"}</div>
        </div>
        <div className="card">
          <div className="k">Datasets</div>
          <div className="v">{datasets.length || "—"}</div>
          <div className="hint">
            {verifiedDatasets} verified against source{totalRecords ? ` · ${totalRecords.toLocaleString()} records total` : ""}
          </div>
        </div>
        <div className="card">
          <div className="k">Candidates compared</div>
          <div className="v">{evals.length ? evals.reduce((n, e) => n + e.candidates_compared.length, 0) : "—"}</div>
          <div className="hint">
            {evals.length ? `${evals.length} conditions × 3 algorithms · baseline kept for ${baselineKept}` : "—"}
          </div>
        </div>
      </div>

      <h2>Comparative model analysis</h2>
      <p className="muted">
        For each condition a LogisticRegression baseline was compared with tree-ensemble challengers on the
        validation split; the winner was evaluated once on a held-out test split. Full tables, confusion matrices
        and calibration on the <Link to="/validation">Validation page</Link>.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Condition</th>
              <th>Deployed model</th>
              <th>Compared against</th>
              <th title="held-out test split">Test ROC-AUC</th>
              <th title="held-out test split">Recall</th>
              <th title="held-out test split">F1</th>
              <th>Calibration</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {evals.map((e) => (
              <tr key={e.disease_key}>
                <td>{e.disease}</td>
                <td>
                  <span className="mono">{e.selected_algorithm}</span>
                  {e.selected_algorithm === e.baseline_algorithm && <span className="pill" style={{ marginLeft: 6 }}>baseline kept</span>}
                </td>
                <td className="muted">{e.candidates_compared.filter((c) => c !== e.selected_algorithm).join(", ")}</td>
                <td>{fmt(e.test_metrics.roc_auc)}</td>
                <td>{fmt(e.test_metrics.recall_sensitivity)}</td>
                <td>{fmt(e.test_metrics.f1)}</td>
                <td>{e.calibration_method ?? "—"}</td>
                <td><Link to="/validation" className="details-link">Details</Link></td>
              </tr>
            ))}
            {!loading && evals.length === 0 && (
              <tr><td colSpan={8} className="muted">No evaluation data available from the API.</td></tr>
            )}
          </tbody>
        </table>
      </div>

      {withoutModel.length > 0 && (
        <div className="status-note">
          <strong>Not modelled in this version:</strong>{" "}
          {withoutModel.map((d) => `${d.display_name} — ${d.unavailable_reason ?? "no model"}`).join(" · ")}
        </div>
      )}

      <div className="two-col" style={{ marginTop: 20 }}>
        <div>
          <h2>System status</h2>
          <div className="status-panel">
            <div className="status-row">
              <span className="status-label">API</span>
              <span className={`pill ${health ? "ok" : "no"}`}>{health ? "Operational" : error ? "Unreachable" : "…"}</span>
            </div>
            <div className="status-row">
              <span className="status-label">Models loaded</span>
              <span className={`pill ${health && health.models_loaded === models.length ? "ok" : "no"}`}>
                {health?.models_loaded ?? "—"}/{models.length || "—"}
              </span>
            </div>
            <div className="status-row">
              <span className="status-label">Assessment history (database)</span>
              <span className={`pill ${health?.database.available ? "ok" : "no"}`}>
                {health ? (health.database.available ? "Operational" : "Unavailable — predictions still work") : "…"}
              </span>
            </div>
          </div>
        </div>
        <div>
          <h2>Recent assessments</h2>
          {recent?.available && recent.results.length > 0 ? (
            <>
              <p className="muted">{recent.total_sessions} assessment sessions stored. Only condition, model, score and level are kept — never inputs.</p>
              <div className="table-scroll">
                <table>
                  <thead><tr><th>When</th><th>Condition</th><th>Score</th><th>Level</th></tr></thead>
                  <tbody>
                    {recent.results.map((r) => (
                      <tr key={`${r.session_id}-${r.disease_key}`}>
                        <td className="muted">{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
                        <td>{displayName(r.disease_key)}</td>
                        <td>{r.risk_score.toFixed(3)}</td>
                        <td><span className={`pill ${r.risk_level === "LOW" ? "ok" : "no"}`}>{r.risk_level}</span></td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="muted">
              {recent && !recent.available
                ? "No database is configured on this deployment, so history is not persisted."
                : loading ? "…" : "No assessments have been stored yet."}
            </p>
          )}
        </div>
      </div>
    </>
  );
}
