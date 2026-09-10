import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DiseaseInfo } from "../types";

export function Diseases() {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.diseases().then((d) => setDiseases(d.diseases)).catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="err">{error}</p>;

  return (
    <>
      <h1>Diseases</h1>
      <p className="sub">The 12 conditions in scope and whether a trained model is available.</p>
      <table>
        <thead>
          <tr><th>Condition</th><th>Group</th><th>Model</th><th>Required inputs / reason</th></tr>
        </thead>
        <tbody>
          {diseases.map((d) => (
            <tr key={d.key}>
              <td>{d.display_name}</td>
              <td>{d.group}</td>
              <td>
                <span className={`pill ${d.model_available ? "ok" : "no"}`}>
                  {d.model_available ? "available" : "unavailable"}
                </span>
              </td>
              <td className="muted">
                {d.model_available
                  ? d.required_features.join(", ")
                  : d.unavailable_reason}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
