import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DatasetEntry, DiseaseInfo, Health, ModelEntry } from "../types";

export function Dashboard() {
  const [health, setHealth] = useState<Health | null>(null);
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([api.health(), api.diseases(), api.models(), api.datasets()])
      .then(([h, d, m, ds]) => {
        setHealth(h);
        setDiseases(d.diseases);
        setModels(m.models);
        setDatasets(ds.datasets);
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="err">Could not reach the API: {error}</p>;

  const withModel = diseases.filter((d) => d.model_available).length;
  const verifiedDatasets = datasets.filter((d) => d.status === "verified").length;

  return (
    <>
      <h1>Dashboard</h1>
      <p className="sub">Overview of supported conditions, trained models and datasets.</p>
      <div className="disclaimer">
        EarlyDX is a research prototype. It produces risk assessments, not diagnoses, and is not
        clinically validated.
      </div>

      <div className="grid">
        <div className="card"><div className="k">Supported diseases</div><div className="v">{diseases.length}</div></div>
        <div className="card"><div className="k">Models available</div><div className="v">{withModel}</div></div>
        <div className="card"><div className="k">Models loaded</div><div className="v">{health?.models_loaded ?? "—"}</div></div>
        <div className="card"><div className="k">Verified datasets</div><div className="v">{verifiedDatasets}</div></div>
        <div className="card"><div className="k">Database</div><div className="v">{health?.database.available ? "up" : "off"}</div></div>
      </div>

      <h2>Model status</h2>
      <table>
        <thead><tr><th>Model</th><th>Disease</th><th>Algorithm</th><th>Test ROC-AUC</th><th>Calibration</th><th>Status</th></tr></thead>
        <tbody>
          {models.map((m) => (
            <tr key={m.model_id}>
              <td>{m.model_id}</td>
              <td>{m.disease}</td>
              <td>{m.algorithm}</td>
              <td>{m.metrics.test_roc_auc ?? "—"}</td>
              <td>{m.calibration}</td>
              <td><span className={`pill ${m.loaded ? "ok" : "no"}`}>{m.loaded ? "loaded" : m.status}</span></td>
            </tr>
          ))}
          {models.length === 0 && <tr><td colSpan={6} className="muted">No models trained yet.</td></tr>}
        </tbody>
      </table>

      <h2>Dataset coverage</h2>
      <table>
        <thead><tr><th>Disease</th><th>Dataset</th><th>Records</th><th>Status</th></tr></thead>
        <tbody>
          {datasets.map((d) => (
            <tr key={d.disease}>
              <td>{d.disease}</td>
              <td>{d.dataset_name}</td>
              <td>{d.records || "—"}</td>
              <td><span className={`pill ${d.status === "verified" ? "ok" : "no"}`}>{d.status}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
