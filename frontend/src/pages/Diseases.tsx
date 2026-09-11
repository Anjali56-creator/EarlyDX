import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DiseaseInfo } from "../types";
import { humanizeLabel } from "../utils/format";

export function Diseases() {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.diseases().then((d) => setDiseases(d.diseases)).catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="err">{error}</p>;

  const unavailable = diseases.filter((d) => !d.model_available);

  return (
    <>
      <h1>Diseases</h1>
      <p className="sub">The 12 conditions in scope and whether a trained model is available.</p>

      {unavailable.length > 0 && (
        <div className="status-note">
          No model available: {unavailable.map((d) => d.display_name).join(", ")}.
        </div>
      )}

      <div className="table-scroll">
        <table>
          <thead>
            <tr><th>Condition</th><th>Group</th><th>Model status</th><th>Required inputs</th></tr>
          </thead>
          <tbody>
            {diseases.map((d) => (
              <tr key={d.key}>
                <td>{d.display_name}</td>
                <td>{d.group}</td>
                <td>
                  <span className={`pill ${d.model_available ? "ok" : "no"}`}>
                    {d.model_available ? "Available" : "Unavailable"}
                  </span>
                </td>
                <td>
                  {d.model_available ? (
                    <details>
                      <summary>{d.required_features.length} inputs</summary>
                      <ul className="inputs-list">
                        {d.required_features.map((f) => (
                          <li key={f}>{humanizeLabel(f)}</li>
                        ))}
                      </ul>
                    </details>
                  ) : (
                    <span className="muted">{d.unavailable_reason ?? "not implemented in this version"}</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}
