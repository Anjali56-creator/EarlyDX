import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { DatasetEntry, DiseaseInfo, EvaluationSummary, Health, ModelEntry, RecentAssessments } from "../types";


/* The research pipeline, expressed as data so each step can be selected and
 * explained. Copy describes what the code in ml/common/framework.py does. */
const PIPELINE: { step: string; short: string; detail: string; to: string; cta: string }[] = [
  {
    step: "Dataset",
    short: "Public, de-identified data with documented provenance.",
    detail:
      "Each condition uses a real public dataset that is downloaded, sha256-verified and inspected before training. Source, licence, size, class balance, target definition and known limitations are recorded in the dataset registry — 'unknown' is never replaced with a guess.",
    to: "/datasets",
    cta: "Browse datasets",
  },
  {
    step: "Preprocessing",
    short: "Imputation, scaling, encoding — fitted on the training split only.",
    detail:
      "A stratified 60 / 20 / 20 train / validation / test split is made first. Every preprocessing step lives inside a scikit-learn Pipeline and is fitted on the training split alone, so no information from validation or test rows leaks into scaling, imputation or encoding.",
    to: "/about",
    cta: "Read the method",
  },
  {
    step: "Risk assessment",
    short: "Validated model → calibrated score → labelled risk level.",
    detail:
      "The deployed pipeline scores your inputs; validation-derived thresholds turn the score into LOW / MODERATE / HIGH. The result is shown next to the model's own test metrics and the inputs it relies on most, with the disclaimer that this is a statistical estimate, not a diagnosis.",
    to: "/assessment",
    cta: "Start an assessment",
  },
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
  const [step, setStep] = useState(2);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.health(),
      api.diseases(),
      api.models(),
      api.datasets(),
      api.evaluationSummary().catch(() => ({ count: 0, models: [] as EvaluationSummary[] })),
      api.recentAssessments(6).catch(() => null),
    ])
      .then(([h, d, m, ds, ev, r]) => {
        setHealth(h); setDiseases(d.diseases); setModels(m.models); setDatasets(ds.datasets); setEvals(ev.models); setRecent(r);
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [attempt]);

  const withModel = diseases.filter((d) => d.model_available).length;
  const activeModels = models.filter((m) => m.status === "active");
  const experimental = models.length - activeModels.length;
  const withoutModel = diseases.filter((d) => !d.model_available);
  const verifiedDatasets = datasets.filter((d) => d.status === "verified").length;
  const totalRecords = datasets.reduce((n, d) => n + (Number(d.records) || 0), 0);
  const candidatesCompared = evals.reduce((n, e) => n + e.candidates_compared.length, 0);
  const displayName = (key: string) => diseases.find((d) => d.key === key)?.display_name ?? key;
  const P = PIPELINE[step];
  const apiDown = !!error && !health;

  return (
    <>
      <section className="hero">
        <div className="hero-copy">
          <span className="eyebrow">Research prototype · ML comparative analysis</span>
          <h1>EarlyDX</h1>
          <p className="hero-tagline">Multi-Disease Early Risk Assessment</p>
          <p>
            EarlyDX trains one independent, leakage-safe machine-learning pipeline per condition on public
            de-identified datasets, compares candidate algorithms on held-out data, and serves the best-justified
            model with every metric it relies on visible — a transparent alternative to single-number "disease
            prediction" demos.
          </p>
          <div className="hero-actions">
            <Link to="/assessment" className="btn-link">Start New Assessment</Link>
          </div>
          <div className="hero-live" aria-live="polite">
            <span className="pill">
              {health ? `✓ API operational · ${health.models_loaded} models loaded` : apiDown ? "✕ API unreachable" : "Connecting to API…"}
            </span>
            <span className="pill">Not a medical device · not a diagnosis</span>
          </div>
        </div>

        {/* Right side: the ML pipeline, annotated with live figures from the API. */}
        <aside className="hero-viz" aria-label="EarlyDX ML pipeline">
          <div className="hero-viz-title">EarlyDX ML pipeline</div>
          <ol className="viz-steps">
            <li>
              <span className="viz-k">Dataset</span>
              <span className="viz-v">{verifiedDatasets ? `${verifiedDatasets} public datasets · ${totalRecords.toLocaleString()} records` : "public, de-identified datasets"}</span>
            </li>
            <li>
              <span className="viz-k">Preprocessing</span>
              <span className="viz-v">stratified 60 / 20 / 20 split · pipeline fitted on train only</span>
            </li>
            <li>
              <span className="viz-k">Risk assessment</span>
              <span className="viz-v">{health ? `${health.models_loaded} conditions available` : "calibrated score → labelled level"}</span>
            </li>
          </ol>
        </aside>
      </section>

      {error && <ApiErrorBox message={error} onRetry={() => setAttempt((a) => a + 1)} />}
      {loading && !error && <Loading what="live project data" />}

      <div className="grid">
        <Link to="/diseases" className="card card-link">
          <div className="k">Supported conditions</div>
          <div className="v">{diseases.length ? `${withModel} / ${diseases.length}` : "—"}</div>
          <div className="hint">
            {diseases.length ? <><strong>{withModel}</strong> conditions deployable with a trained model<br /><strong>{withoutModel.length}</strong> in scope but documented as unavailable</> : "loading…"}
          </div>
        </Link>
        <Link to="/models" className="card card-link">
          <div className="k">Models</div>
          <div className="v">{activeModels.length || "—"}</div>
          <div className="hint">
            {evals.length ? <><strong>{activeModels.length}</strong> active models served by the API — one per condition<br />selected from <strong>{candidatesCompared}</strong> trained candidates{experimental ? ` · ${experimental} experimental, not promoted` : ""}</> : "loading…"}
          </div>
        </Link>
        <Link to="/datasets" className="card card-link">
          <div className="k">Datasets</div>
          <div className="v">{verifiedDatasets || "—"}</div>
          <div className="hint">
            {totalRecords ? <><strong>{verifiedDatasets}</strong> public datasets verified against source (sha256)<br /><strong>{totalRecords.toLocaleString()}</strong> records in total</> : "loading…"}
          </div>
        </Link>
      </div>

      <h2>How EarlyDX works</h2>
      <p className="muted">Select a stage to see what actually happens there.</p>
      <ol className="pipeline" role="tablist" aria-label="Research pipeline">
        {PIPELINE.map((p, i) => (
          <li key={p.step} role="presentation">
            <button
              type="button"
              role="tab"
              aria-selected={i === step}
              className={`pipeline-step ${i === step ? "active" : ""}`}
              onClick={() => setStep(i)}
            >
              <span className="pipeline-num">{i + 1}</span>
              <span className="pipeline-title">{p.step}</span>
              <span className="pipeline-what">{p.short}</span>
            </button>
          </li>
        ))}
      </ol>
      <div className="pipeline-detail fade-in" key={step} role="tabpanel">
        <div>
          <strong>{step + 1}. {P.step}</strong>
          <p>{P.detail}</p>
        </div>
        <Link to={P.to} className="btn-secondary btn-sm">{P.cta} →</Link>
      </div>

      <h2>Recent assessments</h2>
      {recent?.available && recent.results.length > 0 ? (
        <>
          <p className="muted">{recent.total_sessions} assessment sessions stored on this deployment. Only condition, model, score and level are kept — inputs are never stored.</p>
          <div className="table-scroll">
            <table>
              <thead><tr><th>When</th><th>Condition</th><th>Model</th><th>Score</th><th>Level</th></tr></thead>
              <tbody>
                {recent.results.map((r) => (
                  <tr key={`${r.session_id}-${r.disease_key}`}>
                    <td className="muted">{r.created_at ? new Date(r.created_at).toLocaleString() : "—"}</td>
                    <td>{displayName(r.disease_key)}</td>
                    <td className="mono">{r.model_id}</td>
                    <td>{r.risk_score.toFixed(3)}</td>
                    <td><span className={`pill ${r.risk_level === "LOW" ? "ok" : r.risk_level === "HIGH" ? "no" : ""}`}>{r.risk_level === "HIGH" ? "▲" : r.risk_level === "LOW" ? "▼" : "●"} {r.risk_level}</span></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      ) : (
        <div className="empty-state">
          <div className="empty-icon">◎</div>
          {recent && !recent.available
            ? "History is not persisted on this deployment (no database configured). Predictions still work."
            : loading ? "Checking stored assessments…" : "No assessments have been recorded yet."}
          <div style={{ marginTop: 12 }}><Link to="/assessment" className="btn-secondary btn-sm">Run the first assessment</Link></div>
        </div>
      )}

      <div className="two-col" style={{ marginTop: 24 }}>
        <div className="status-panel">
          <div className="status-row"><span className="status-label">API <span className="muted">/health</span></span><span className={`pill ${health ? "ok" : "no"}`}>{health ? "✓ Operational" : apiDown ? "✕ Unreachable" : "… checking"}</span></div>
          <div className="status-row"><span className="status-label">Models loaded</span><span className={`pill ${health && health.models_loaded === activeModels.length ? "ok" : "no"}`}>{health ? `${health.models_loaded} / ${activeModels.length} active${Object.keys(health.model_load_errors).length ? ` · ${Object.keys(health.model_load_errors).length} failed` : ""}` : apiDown ? "unknown — API unreachable" : "… checking"}</span></div>
          <div className="status-row"><span className="status-label">Assessment history (database)</span><span className={`pill ${health?.database.available ? "ok" : "no"}`}>{health ? (health.database.available ? "✓ Operational" : "Degraded — predictions still work") : apiDown ? "unknown — API unreachable" : "… checking"}</span></div>
        </div>
        <div className="disclaimer" style={{ alignSelf: "start" }}>
          EarlyDX produces statistical risk estimates from public datasets. It is not a medical device, is not
          clinically validated, and does not diagnose. Consult a qualified clinician for medical concerns.
        </div>
      </div>
    </>
  );
}
