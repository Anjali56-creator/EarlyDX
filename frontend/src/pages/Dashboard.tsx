import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { DatasetEntry, DiseaseInfo, EvaluationSummary, Health, ModelEntry, RecentAssessments } from "../types";

const fmt = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));

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
    step: "Model training",
    short: "Three candidate algorithms, tuned with cross-validated grid search.",
    detail:
      "A LogisticRegression baseline is trained alongside RandomForest and GradientBoosting challengers. Each is tuned with GridSearchCV under stratified 5-fold cross-validation on the training split, then calibrated (none / sigmoid / isotonic) by validation Brier score.",
    to: "/models",
    cta: "See the models",
  },
  {
    step: "Model comparison",
    short: "Candidates compared on validation; the winner tested once.",
    detail:
      "Candidates are compared on the validation split. The interpretable baseline is kept unless a challenger beats it by at least 0.01 ROC-AUC. The selected model is evaluated exactly once on the held-out test split — accuracy, precision, recall, F1, ROC-AUC, PR-AUC and the confusion matrix are all stored.",
    to: "/validation",
    cta: "View model comparison",
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
  const [step, setStep] = useState(3);

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
  const baselineKept = evals.filter((e) => e.selected_algorithm === e.baseline_algorithm).length;
  const candidatesCompared = evals.reduce((n, e) => n + e.candidates_compared.length, 0);
  const displayName = (key: string) => diseases.find((d) => d.key === key)?.display_name ?? key;
  const bestByAuc = evals.length ? [...evals].sort((a, b) => (b.test_metrics.roc_auc ?? 0) - (a.test_metrics.roc_auc ?? 0)) : [];
  const weakest = bestByAuc[bestByAuc.length - 1];
  const P = PIPELINE[step];
  const algorithms = Array.from(new Set(evals.map((e) => e.selected_algorithm)));
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
            <Link to="/validation" className="btn-secondary">Explore Validation</Link>
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
              <span className="viz-k">Candidate models</span>
              <span className="viz-v">{evals.length ? `${candidatesCompared} candidates · LogisticRegression baseline vs. tree ensembles` : "3 algorithms per condition"}</span>
            </li>
            <li>
              <span className="viz-k">Model comparison</span>
              <span className="viz-v">{evals.length ? `baseline kept ${baselineKept}× · ensemble promoted ${evals.length - baselineKept}× · one held-out test` : "validation split · ≥ 0.01 ROC-AUC rule"}</span>
            </li>
            <li>
              <span className="viz-k">Risk assessment</span>
              <span className="viz-v">{health ? `${health.models_loaded} models served · ${algorithms.length} algorithms in use` : "calibrated score → labelled level"}</span>
            </li>
          </ol>
          {bestByAuc.length > 0 && (
            <div className="viz-strip" aria-label="Held-out test ROC-AUC per deployed model">
              <div className="viz-strip-title">Held-out test ROC-AUC · {bestByAuc.length} deployed models</div>
              <div className="viz-bars">
                {bestByAuc.map((e) => (
                  <Link to="/validation" key={e.disease_key} className="viz-bar" title={`${e.disease}: ${fmt(e.test_metrics.roc_auc)} (${e.selected_algorithm})`}>
                    <span style={{ height: `${Math.max(4, (e.test_metrics.roc_auc ?? 0) * 100)}%` }} />
                  </Link>
                ))}
              </div>
              <div className="viz-strip-foot"><span>{fmt(bestByAuc[0]?.test_metrics.roc_auc, 2)} {bestByAuc[0]?.disease}</span><span>{fmt(weakest?.test_metrics.roc_auc, 2)} {weakest?.disease}</span></div>
            </div>
          )}
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
        <Link to="/validation" className="card card-link">
          <div className="k">Validation</div>
          <div className="v">{evals.length ? `${evals.filter((e) => (e.test_metrics.roc_auc ?? 0) >= 0.9).length} / ${evals.length}` : "—"}</div>
          <div className="hint">
            {evals.length ? <>models with held-out test ROC-AUC ≥ 0.90<br />(a display threshold — all {evals.length} were evaluated once on a held-out test split)</> : "loading…"}
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

      <h2>Research &amp; validation</h2>
      <p className="muted">
        Held-out test results of the deployed model for each condition. Every number below was written by the
        training run and is read verbatim from the API.
      </p>
      <div className="two-col">
        <div>
          <div className="table-scroll">
            <table>
              <thead>
                <tr><th>Condition</th><th>Deployed model</th><th title="held-out test split">ROC-AUC</th><th title="held-out test split">Recall</th><th title="held-out test split">F1</th></tr>
              </thead>
              <tbody>
                {bestByAuc.map((e) => (
                  <tr key={e.disease_key}>
                    <td>{e.disease}</td>
                    <td className="mono">{e.selected_algorithm}{e.selected_algorithm === e.baseline_algorithm ? " ·baseline" : ""}</td>
                    <td>{fmt(e.test_metrics.roc_auc)}</td>
                    <td>{fmt(e.test_metrics.recall_sensitivity)}</td>
                    <td>{fmt(e.test_metrics.f1)}</td>
                  </tr>
                ))}
                {!loading && evals.length === 0 && <tr><td colSpan={5} className="muted">No evaluation data available from the API.</td></tr>}
              </tbody>
            </table>
          </div>
        </div>
        <div>
          <div className="callout" style={{ marginBottom: 12 }}>
            <strong>What the comparison shows.</strong>{" "}
            {evals.length ? (
              <>
                The LogisticRegression baseline was kept for {baselineKept} of {evals.length} conditions; a tree ensemble
                won the rest. {bestByAuc[0]?.disease} has the highest test ROC-AUC ({fmt(bestByAuc[0]?.test_metrics.roc_auc)});{" "}
                {weakest?.disease} the lowest ({fmt(weakest?.test_metrics.roc_auc)}) — kept for transparency, not merit.
                Accuracy alone is not used for selection: for screening, recall and PR-AUC matter more.
              </>
            ) : "Loading…"}
          </div>
          <Link to="/validation" className="btn-link">View Model Comparison</Link>
          {withoutModel.length > 0 && (
            <p className="muted" style={{ marginTop: 14 }}>
              <strong>Not modelled:</strong> {withoutModel.map((d) => d.display_name).join(", ")} — no dataset with
              acceptable provenance or a rigorously defined target. Shown, not hidden.
            </p>
          )}
        </div>
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
