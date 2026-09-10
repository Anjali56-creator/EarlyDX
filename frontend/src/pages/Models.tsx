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
      {models.length === 0 && <p className="muted">No models trained yet.</p>}
      {models.map((m) => (
        <div className="card" key={m.model_id} style={{ marginBottom: 14 }}>
          <h2 style={{ marginTop: 0 }}>{m.model_id}</h2>
          <table>
            <tbody>
              <tr><th>Disease</th><td>{m.disease}</td></tr>
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
        </div>
      ))}
    </>
  );
}
