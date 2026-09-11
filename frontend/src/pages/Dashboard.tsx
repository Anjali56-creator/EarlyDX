import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
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

      <div className="cta-card">
        <div>
          <div className="cta-title">Start New Assessment</div>
          <p className="cta-sub">Assess risk using the available EarlyDX prediction models.</p>
        </div>
        <Link to="/assessment" className="btn-link">Start New Assessment</Link>
      </div>

      <div className="grid">
        <div className="card">
          <div className="k">Supported Conditions</div>
          <div className="v">{diseases.length}</div>
          <div className="hint">Conditions EarlyDX is designed to assess</div>
        </div>
        <div className="card">
          <div className="k">Active Models</div>
          <div className="v">{withModel}</div>
          <div className="hint">Conditions with a trained model available for prediction</div>
        </div>
        <div className="card">
          <div className="k">Verified Datasets</div>
          <div className="v">{verifiedDatasets}</div>
          <div className="hint">Datasets verified against their original source</div>
        </div>
      </div>

      <h2>System status</h2>
      <div className="status-panel">
        <div className="status-row">
          <span className="status-label">Database</span>
          <span className={`pill ${health?.database.available ? "ok" : "no"}`}>
            {health?.database.available ? "Operational" : "Unavailable"}
          </span>
        </div>
        <div className="status-row">
          <span className="status-label">Models</span>
          <span className="pill ok">{health?.models_loaded ?? "—"}/{models.length || "—"} loaded</span>
        </div>
        <div className="status-row">
          <span className="status-label">API</span>
          <span className="pill ok">Operational</span>
        </div>
      </div>

      <h2>Model status</h2>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Model</th>
              <th>Condition</th>
              <th>Algorithm</th>
              <th>Test ROC-AUC</th>
              <th>Calibration</th>
              <th>Status</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {models.map((m) => (
              <tr key={m.model_id}>
                <td className="mono">{m.model_id}</td>
                <td>{m.disease}</td>
                <td>{m.algorithm}</td>
                <td>{m.metrics.test_roc_auc ?? "—"}</td>
                <td>{m.calibration}</td>
                <td><span className={`pill ${m.loaded ? "ok" : "no"}`}>{m.loaded ? "loaded" : m.status}</span></td>
                <td><Link to={`/models#${m.model_id}`} className="details-link">View details</Link></td>
              </tr>
            ))}
            {models.length === 0 && <tr><td colSpan={7} className="muted">No models trained yet.</td></tr>}
          </tbody>
        </table>
      </div>

      <h2>Dataset coverage</h2>
      <div className="table-scroll">
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
      </div>
    </>
  );
}
