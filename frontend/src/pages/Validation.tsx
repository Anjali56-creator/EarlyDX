import { useCallback, useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { ClassificationMetrics, DiseaseInfo, EvaluationDetail, ValidationSamples } from "../types";
import { humanizeLabel } from "../utils/format";

const fmt = (v: number | null | undefined, d = 3) => (v == null ? "—" : v.toFixed(d));
const pct = (v: number | null | undefined) => (v == null ? "—" : `${(v * 100).toFixed(1)}%`);

const METRIC_COLS: { key: keyof ClassificationMetrics; label: string; title: string }[] = [
  { key: "accuracy", label: "Accuracy", title: "(TP+TN)/N at the 0.5 threshold" },
  { key: "precision", label: "Precision", title: "TP/(TP+FP) at the 0.5 threshold" },
  { key: "recall_sensitivity", label: "Recall (sensitivity)", title: "TP/(TP+FN) at the 0.5 threshold" },
  { key: "specificity", label: "Specificity", title: "TN/(TN+FP) at the 0.5 threshold" },
  { key: "f1", label: "F1", title: "Harmonic mean of precision and recall" },
  { key: "roc_auc", label: "ROC-AUC", title: "Threshold-independent ranking quality" },
  { key: "pr_auc", label: "PR-AUC", title: "Area under precision-recall curve; more informative than ROC-AUC under class imbalance" },
];

function ConfusionMatrix({ cm, n }: { cm: ClassificationMetrics["confusion_matrix"]; n: number | null }) {
  if (!cm) return <p className="muted">Confusion matrix not stored for this run.</p>;
  return (
    <table className="cm">
      <thead>
        <tr><th></th><th>Predicted negative</th><th>Predicted positive</th></tr>
      </thead>
      <tbody>
        <tr><th>Actual negative</th><td className="cm-ok">TN {cm.tn}</td><td className="cm-bad">FP {cm.fp}</td></tr>
        <tr><th>Actual positive</th><td className="cm-bad">FN {cm.fn}</td><td className="cm-ok">TP {cm.tp}</td></tr>
      </tbody>
      {n != null && <tfoot><tr><td colSpan={3} className="muted">n = {n} held-out test records</td></tr></tfoot>}
    </table>
  );
}

export function Validation() {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [ev, setEv] = useState<EvaluationDetail | null>(null);
  const [samples, setSamples] = useState<ValidationSamples | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [attempt, setAttempt] = useState(0);

  const loadDiseases = useCallback(() => {
    setError(null);
    api.diseases().then((d) => {
      setDiseases(d.diseases);
      const first = d.diseases.find((x) => x.model_available);
      if (first) setSelected((cur) => cur || first.key);
    }).catch((e) => { setError(String(e.message ?? e)); setLoading(false); });
  }, []);

  useEffect(loadDiseases, [loadDiseases]);

  useEffect(() => {
    if (!selected) return;
    setEv(null);
    setSamples(null);
    setError(null);
    setLoading(true);
    Promise.all([
      api.evaluation(selected),
      api.validationSamples(selected).catch((e) => {
        if (e instanceof ApiError && e.status === 404) return null;
        throw e;
      }),
    ])
      .then(([e, s]) => { setEv(e); setSamples(s); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [selected, attempt]);

  const available = useMemo(() => diseases.filter((d) => d.model_available), [diseases]);

  const candidateRows = useMemo(() => {
    if (!ev) return [];
    return Object.entries(ev.candidates).sort(([a], [b]) => {
      if (a === ev.selected_algorithm) return -1;
      if (b === ev.selected_algorithm) return 1;
      return a.localeCompare(b);
    });
  }, [ev]);

  const best = ev?.best_validation_candidate_by_roc_auc ?? null;
  const baselineKept = ev ? ev.selected_algorithm === ev.baseline_algorithm : false;
  const bestAuc = best && ev ? ev.candidates[best]?.roc_auc : null;
  const selectedAuc = ev ? ev.candidates[ev.selected_algorithm]?.roc_auc : null;

  return (
    <>
      <h1>Validation &amp; model comparison</h1>
      <p className="sub">
        How each model was chosen and how it performs on data it never saw. Every number on this
        page was written by the training run — nothing is recomputed or estimated here.
      </p>
      <div className="disclaimer">
        This is model validation, not diagnosis. Candidate models are compared on the <strong>validation</strong>{" "}
        split; the chosen model is then evaluated <strong>once</strong> on a <strong>test</strong> split that was
        set aside before training and never used for tuning, calibration or threshold selection.
      </div>

      <div className="selector-card">
        <div className="field" style={{ marginBottom: 0 }}>
          <label htmlFor="val-disease">Condition</label>
          <select id="val-disease" value={selected} onChange={(e) => setSelected(e.target.value)} disabled={!available.length}>
            {!available.length && <option value="">Loading conditions…</option>}
            {available.map((d) => (
              <option key={d.key} value={d.key}>{d.display_name}</option>
            ))}
          </select>
        </div>
      </div>

      {error && <ApiErrorBox message={error} onRetry={() => (selected ? setAttempt((a) => a + 1) : loadDiseases())} />}
      {loading && !error && <Loading what="evaluation results" />}

      {ev && (
        <>
          <h2>1. Which model performs best — and which one was chosen?</h2>
          <p className="muted">
            Candidates trained on the training split ({ev.split.sizes?.train ?? "—"} records) and compared on the
            validation split ({ev.split.sizes?.val ?? "—"} records) at the 0.5 threshold. Selection rule from the
            training run: <em>{ev.selection_rule ?? "—"}</em>
          </p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Candidate</th>
                  {METRIC_COLS.map((c) => <th key={c.key} title={c.title}>{c.label}</th>)}
                  <th>Role</th>
                </tr>
              </thead>
              <tbody>
                {candidateRows.map(([name, m]) => {
                  const isSel = name === ev.selected_algorithm;
                  const isBest = name === best;
                  return (
                    <tr key={name} className={isSel ? "row-selected" : ""}>
                      <td className="mono">{name}</td>
                      {METRIC_COLS.map((c) => (
                        <td key={c.key} className={isBest && c.key === "roc_auc" ? "cell-best" : ""}>
                          {fmt(m[c.key] as number | null)}
                        </td>
                      ))}
                      <td>
                        {isSel && <span className="pill ok">selected · deployed</span>}{" "}
                        {name === ev.baseline_algorithm && <span className="pill">baseline</span>}{" "}
                        {isBest && !isSel && <span className="pill">highest val. ROC-AUC</span>}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <div className="callout">
            <strong>Verdict for {ev.disease}:</strong>{" "}
            {best === ev.selected_algorithm ? (
              <>
                <span className="mono">{ev.selected_algorithm}</span> had the highest validation ROC-AUC ({fmt(bestAuc)})
                {baselineKept
                  ? " — the interpretable baseline was already the best candidate, so it was kept."
                  : ` and beat the ${ev.baseline_algorithm} baseline (${fmt(ev.baseline_algorithm ? ev.candidates[ev.baseline_algorithm]?.roc_auc : null)}) by at least 0.01, so it was promoted.`}
              </>
            ) : (
              <>
                <span className="mono">{best}</span> scored the highest validation ROC-AUC ({fmt(bestAuc)}), but{" "}
                <span className="mono">{ev.selected_algorithm}</span> ({fmt(selectedAuc)}) was deployed because the
                selection rule keeps the interpretable baseline unless a challenger wins by a clear margin (≥ 0.01).
              </>
            )}{" "}
            Ranking is by ROC-AUC because it is threshold-independent; for medical screening, <strong>recall
            (sensitivity)</strong> and <strong>PR-AUC</strong> matter more than raw accuracy, which can look good
            simply by predicting the majority class. Positive rate in this dataset: {pct(ev.target_positive_rate)}.
          </div>

          <h2>2. Held-out test performance of the deployed model</h2>
          <p className="muted">
            <span className="mono">{ev.model_version}</span> · {ev.selected_algorithm} · evaluated once on{" "}
            {ev.test_metrics.n ?? "—"} test records
            {ev.test_metrics.class_distribution && (
              <> ({ev.test_metrics.class_distribution["1"]} positive / {ev.test_metrics.class_distribution["0"]} negative)</>
            )}
            {" "}at threshold {fmt(ev.test_metrics.threshold, 2)}.
          </p>
          <div className="grid metrics-grid">
            {METRIC_COLS.map((c) => (
              <div className="card" key={c.key} title={c.title}>
                <div className="k">{c.label}</div>
                <div className="v">{fmt(ev.test_metrics[c.key] as number | null)}</div>
                {ev.cross_validation.summary?.[c.key] && (
                  <div className="hint">
                    CV mean {fmt(ev.cross_validation.summary[c.key].mean)} ± {fmt(ev.cross_validation.summary[c.key].std)}
                  </div>
                )}
              </div>
            ))}
          </div>

          <div className="two-col">
            <div>
              <h3>Confusion matrix (test, threshold {fmt(ev.test_metrics.threshold, 2)})</h3>
              <ConfusionMatrix cm={ev.test_metrics.confusion_matrix} n={ev.test_metrics.n} />
              <p className="muted">
                For early detection a <strong>false negative</strong> (missed case) is usually the costlier error.
                {ev.test_metrics_at_screening_threshold?.confusion_matrix && (
                  <>
                    {" "}At the model's validation-derived screening cut ({fmt(ev.test_metrics_at_screening_threshold.threshold)}),
                    recall rises to {fmt(ev.test_metrics_at_screening_threshold.recall_sensitivity)} with{" "}
                    {ev.test_metrics_at_screening_threshold.confusion_matrix.fn} false negatives, at the cost of specificity{" "}
                    {fmt(ev.test_metrics_at_screening_threshold.specificity)}.
                  </>
                )}
              </p>
            </div>
            <div>
              <h3>Cross-validation &amp; calibration</h3>
              <table>
                <tbody>
                  <tr><th>CV folds</th><td>{ev.cross_validation.folds ?? "—"} (stratified, train split only)</td></tr>
                  <tr>
                    <th>CV ROC-AUC</th>
                    <td>
                      {ev.cross_validation.summary?.roc_auc
                        ? `${fmt(ev.cross_validation.summary.roc_auc.mean)} ± ${fmt(ev.cross_validation.summary.roc_auc.std)}`
                        : "—"}
                    </td>
                  </tr>
                  <tr><th>Calibration method</th><td>{ev.calibration.method ?? "—"} (chosen by validation Brier score)</td></tr>
                  {ev.calibration.validation_brier && (
                    <tr>
                      <th>Validation Brier</th>
                      <td>
                        {Object.entries(ev.calibration.validation_brier).map(([k, v]) => (
                          <div key={k} className={k === ev.calibration.method ? "cell-best" : ""}>{k}: {fmt(v, 4)}</div>
                        ))}
                      </td>
                    </tr>
                  )}
                  <tr><th>Test Brier</th><td>{fmt(ev.test_metrics.brier, 4)}</td></tr>
                  {ev.risk_thresholds && (
                    <tr>
                      <th>Risk-level cuts</th>
                      <td>LOW &lt; {fmt(ev.risk_thresholds.low_cut)} · HIGH ≥ {fmt(ev.risk_thresholds.high_cut)}</td>
                    </tr>
                  )}
                  <tr><th>ROC curve</th><td className="muted">{ev.unavailable.roc_curve_points}</td></tr>
                </tbody>
              </table>
            </div>
          </div>

          {ev.permutation_importance && ev.permutation_importance.length > 0 && (
            <>
              <h3>Permutation importance (held-out test split, 10 repeats)</h3>
              <p className="muted">
                Drop in ROC-AUC when each feature is shuffled. Measures what the trained model relies on — not
                causation, and not a clinical statement about the feature.
              </p>
              <div className="table-scroll">
                <table>
                  <thead><tr><th>Feature</th><th>Importance</th><th></th></tr></thead>
                  <tbody>
                    {[...ev.permutation_importance]
                      .sort((a, b) => b.importance - a.importance)
                      .map((f) => {
                        const max = Math.max(...ev.permutation_importance!.map((x) => x.importance), 1e-9);
                        return (
                          <tr key={f.feature}>
                            <td>{humanizeLabel(f.feature)}</td>
                            <td className="mono">{fmt(f.importance, 4)} ± {fmt(f.std, 4)}</td>
                            <td style={{ width: "40%" }}>
                              <div className="bar"><span style={{ width: `${Math.max(0, (f.importance / max) * 100)}%` }} /></div>
                            </td>
                          </tr>
                        );
                      })}
                  </tbody>
                </table>
              </div>
            </>
          )}

          {(ev.limitations || ev.ethical_note) && (
            <div className="status-note">
              <strong>Limitations:</strong> {ev.limitations}
              {ev.ethical_note && <> <br /><strong>Ethical note:</strong> {ev.ethical_note}</>}
            </div>
          )}

          <h2 style={{ marginTop: 24 }}>3. Individual held-out records</h2>
          {samples ? (
            <>
              <p className="muted">Model: <span className="mono">{samples.model_version}</span> — {samples.note}</p>
              <div className="table-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>#</th>
                      <th>Actual outcome</th>
                      <th>Model prediction</th>
                      <th>Risk score</th>
                      <th>Risk level</th>
                      <th>Result</th>
                    </tr>
                  </thead>
                  <tbody>
                    {samples.samples.map((s, i) => {
                      const correct = s.predicted_outcome === s.actual_outcome;
                      return (
                        <tr key={i}>
                          <td>{i + 1}</td>
                          <td>{s.actual_outcome === 1 ? "Positive" : "Negative"}</td>
                          <td>{s.predicted_outcome === 1 ? "Positive" : "Negative"}</td>
                          <td>{s.predicted_probability.toFixed(3)}</td>
                          <td><span className={`pill ${s.predicted_risk_level === "LOW" ? "ok" : "no"}`}>{s.predicted_risk_level}</span></td>
                          <td>
                            <span className={`pill ${correct ? "ok" : "no"}`}>
                              {correct ? "Correct classification" : "Incorrect classification"}
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </>
          ) : (
            <p className="muted">
              Individual test rows were not captured for this model's training run. The aggregate test metrics
              above are still from the same held-out split.
            </p>
          )}
        </>
      )}
    </>
  );
}
