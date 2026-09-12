import { useEffect, useState } from "react";
import { Link, useLocation } from "react-router-dom";
import { api } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { EvaluationSummary, ModelEntry } from "../types";

const fmt = (v: number | null | undefined, d = 3) => (v == null ? "—" : Number(v).toFixed(d));

const METRIC_LABELS: Record<string, string> = {
  test_roc_auc: "Test ROC-AUC",
  test_pr_auc: "Test PR-AUC",
  test_accuracy: "Test accuracy",
  test_recall_sensitivity: "Test recall (sensitivity)",
  test_specificity: "Test specificity",
  cv_roc_auc_mean: "Cross-validation ROC-AUC (mean)",
};

export function Models() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [evals, setEvals] = useState<Record<string, EvaluationSummary>>({});
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);
  const { hash } = useLocation();

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([
      api.models(),
      api.evaluationSummary().catch(() => ({ count: 0, models: [] as EvaluationSummary[] })),
    ])
      .then(([m, ev]) => {
        setModels(m.models);
        setEvals(Object.fromEntries(ev.models.map((e) => [e.model_version, e])));
      })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [attempt]);

  // open + scroll to the model linked from another page (/models#diabetes-v1)
  useEffect(() => {
    if (!hash || loading) return;
    const el = document.getElementById(hash.slice(1)) as HTMLDetailsElement | null;
    if (el) { el.open = true; el.scrollIntoView({ block: "start" }); }
  }, [hash, loading]);

  return (
    <>
      <span className="eyebrow">Model registry</span>
      <h1>Models</h1>
      <p className="sub">
        Every model version the API can serve. Each was selected from three candidates on a validation split and
        then evaluated once on a held-out test split; every prediction records which version produced it.
      </p>

      {error && <ApiErrorBox message={error} onRetry={() => setAttempt((a) => a + 1)} />}
      {loading && !error && <Loading what="model registry" />}

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Condition</th>
              <th>Algorithm</th>
              <th title="held-out test split, 0.5 threshold">Accuracy</th>
              <th title="held-out test split, 0.5 threshold">Precision</th>
              <th title="held-out test split, 0.5 threshold">Recall</th>
              <th title="held-out test split, 0.5 threshold">F1</th>
              <th title="held-out test split">ROC-AUC</th>
              <th>Calibration</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => {
              const e = evals[m.model_id];
              return (
                <tr key={m.model_id}>
                  <td className="mono"><a href={`#${m.model_id}`}>{m.model_id}</a></td>
                  <td>{e?.disease ?? m.disease}</td>
                  <td>{m.algorithm}</td>
                  <td>{fmt(e?.test_metrics.accuracy ?? m.metrics.test_accuracy)}</td>
                  <td>{fmt(e?.test_metrics.precision)}</td>
                  <td>{fmt(e?.test_metrics.recall_sensitivity ?? m.metrics.test_recall_sensitivity)}</td>
                  <td>{fmt(e?.test_metrics.f1)}</td>
                  <td>{fmt(e?.test_metrics.roc_auc ?? m.metrics.test_roc_auc)}</td>
                  <td>{m.calibration}</td>
                  <td><span className={`pill ${m.loaded ? "ok" : "no"}`}>{m.loaded ? "loaded" : m.status}</span></td>
                </tr>
              );
            })}
            {!loading && models.length === 0 && <tr><td colSpan={10} className="muted">No models registered.</td></tr>}
          </tbody>
        </table>
      </div>
      <p className="muted">
        Accuracy alone is misleading on imbalanced medical data — a model can score well by predicting the majority
        class. Recall (missed cases) and PR-AUC are the metrics to weigh for screening. See{" "}
        <Link to="/validation">Validation</Link> for the per-condition candidate comparison and confusion matrices.
      </p>

      <h2>Model details</h2>
      {models.map((m) => {
        const e = evals[m.model_id];
        return (
          <details className="model-details" id={m.model_id} key={m.model_id}>
            <summary>{m.model_id} — {m.algorithm} for {e?.disease ?? m.disease}</summary>
            <table>
              <tbody>
                <tr><th>Model version</th><td className="mono">{m.model_id}</td></tr>
                <tr><th>Condition</th><td>{e?.disease ?? m.disease}</td></tr>
                <tr><th>Algorithm</th><td>{m.algorithm}</td></tr>
                {e && (
                  <tr>
                    <th>Selected from</th>
                    <td>
                      {e.candidates_compared.join(", ")}
                      {e.selected_algorithm === e.baseline_algorithm
                        ? " — the interpretable baseline was kept (no challenger beat it by ≥ 0.01 validation ROC-AUC)."
                        : ` — beat the ${e.baseline_algorithm} baseline by ≥ 0.01 validation ROC-AUC.`}
                    </td>
                  </tr>
                )}
                <tr><th>Input features</th><td>{m.features?.length ? m.features.join(", ") : "—"}</td></tr>
                <tr><th>Dataset (registry key)</th><td>{m.dataset}</td></tr>
                <tr><th>Trained</th><td>{m.training_date}</td></tr>
                <tr><th>Calibration</th><td>{m.calibration} (chosen by validation Brier score)</td></tr>
                {e?.cv_roc_auc && (
                  <tr><th>CV ROC-AUC</th><td>{fmt(e.cv_roc_auc.mean)} ± {fmt(e.cv_roc_auc.std)}</td></tr>
                )}
                <tr><th>Status</th><td>{m.status}{m.loaded ? " (loaded in this API instance)" : ""}</td></tr>
                <tr>
                  <th>Registry metrics</th>
                  <td>
                    {Object.entries(m.metrics).map(([k, v]) => (
                      <div key={k}>{METRIC_LABELS[k] ?? k}: {v}</div>
                    ))}
                  </td>
                </tr>
                <tr>
                  <th>More</th>
                  <td><Link to="/validation">Candidate comparison, confusion matrix, calibration →</Link></td>
                </tr>
              </tbody>
            </table>
          </details>
        );
      })}
    </>
  );
}
