import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { DatasetEntry, DiseaseInfo, ModelEntry } from "../types";
import { humanizeLabel } from "../utils/format";

export function Diseases() {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([api.diseases(), api.models(), api.datasets()])
      .then(([d, m, ds]) => { setDiseases(d.diseases); setModels(m.models); setDatasets(ds.datasets); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [attempt]);

  const modelFor = (key: string) => models.find((m) => m.disease === key && m.loaded) ?? models.find((m) => m.disease === key);
  const datasetFor = (key: string) => datasets.find((d) => d.disease === key);
  const groups = Array.from(new Set(diseases.map((d) => d.group))).sort();
  const unavailable = diseases.filter((d) => !d.model_available);

  return (
    <>
      <span className="eyebrow">Scope</span>
      <h1>Conditions</h1>
      <p className="sub">
        The {diseases.length || 12} conditions in scope, grouped by clinical area, with the exact inputs each
        model requires and the dataset it was trained on.
      </p>

      {error && <ApiErrorBox message={error} onRetry={() => setAttempt((a) => a + 1)} />}
      {loading && !error && <Loading what="conditions" />}

      {unavailable.length > 0 && (
        <div className="status-note">
          <strong>No model in this version:</strong>{" "}
          {unavailable.map((d) => d.display_name).join(", ")} — reasons are listed with each condition below.
        </div>
      )}

      {groups.map((g) => (
        <section key={g}>
          <h2 style={{ textTransform: "capitalize" }}>{g}</h2>
          <div className="card-list">
            {diseases.filter((d) => d.group === g).map((d) => {
              const m = modelFor(d.key);
              const ds = datasetFor(d.key);
              return (
                <div className="card condition-card" key={d.key} id={d.key}>
                  <div className="condition-head">
                    <div>
                      <div className="cta-title">{d.display_name}</div>
                      <div className="muted">
                        {d.model_available && m
                          ? <>Model <span className="mono">{m.model_id}</span> · {m.algorithm} · test ROC-AUC {m.metrics.test_roc_auc ?? "—"}</>
                          : <>{d.unavailable_reason ?? "Not implemented in this version."}</>}
                      </div>
                    </div>
                    <span className={`pill ${d.model_available ? "ok" : "no"}`}>
                      {d.model_available ? "Model available" : "Unavailable"}
                    </span>
                  </div>

                  {d.model_available && (
                    <div className="condition-body">
                      <div>
                        <div className="k">Required inputs ({d.required_features.length})</div>
                        <ul className="inputs-list inputs-cols">
                          {d.required_features.map((f) => <li key={f}>{humanizeLabel(f)}</li>)}
                        </ul>
                      </div>
                      <div>
                        <div className="k">Training data</div>
                        {ds ? (
                          <p className="muted" style={{ margin: "4px 0" }}>
                            {ds.dataset_name}
                            {ds.records ? <> · {ds.records.toLocaleString()} records</> : null}
                            {ds.target ? <><br />Target: {ds.target}</> : null}
                          </p>
                        ) : <p className="muted">—</p>}
                        <div className="condition-links">
                          <Link to="/assessment">Assess</Link>
                          <Link to="/validation">Validation</Link>
                          <Link to={`/models#${m?.model_id ?? ""}`}>Model</Link>
                          <Link to="/datasets">Dataset</Link>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </>
  );
}
