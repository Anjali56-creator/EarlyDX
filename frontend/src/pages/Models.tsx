import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { ModelEntry } from "../types";

export function Models() {
  const [models, setModels] = useState<ModelEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.models().then((m) => setModels(m.models)).catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="err">{error}</p>;

  return (
    <>
      <h1>Models</h1>
      <p className="sub">Registered model versions. Every prediction records which version produced it.</p>

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
              </tr>
            ))}
            {models.length === 0 && <tr><td colSpan={6} className="muted">No models trained yet.</td></tr>}
          </tbody>
        </table>
      </div>

      {models.map((m) => (
        <details className="model-details" id={m.model_id} key={m.model_id}>
          <summary>{m.model_id} — view details</summary>
          <table>
            <tbody>
              <tr><th>Model version</th><td className="mono">{m.model_id}</td></tr>
              <tr><th>Condition</th><td>{m.disease}</td></tr>
              <tr><th>Algorithm</th><td>{m.algorithm}</td></tr>
              <tr><th>Dataset</th><td>{m.dataset}</td></tr>
              <tr><th>Trained</th><td>{m.training_date}</td></tr>
              <tr><th>Calibration</th><td>{m.calibration}</td></tr>
              <tr><th>Status</th><td>{m.status}{m.loaded ? " (loaded)" : ""}</td></tr>
              <tr>
                <th>Metrics</th>
                <td>
                  {Object.entries(m.metrics).map(([k, v]) => (
                    <div key={k}>{k}: {v}</div>
                  ))}
                </td>
              </tr>
            </tbody>
          </table>
        </details>
      ))}
    </>
  );
}
