import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DatasetEntry } from "../types";

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.datasets().then((d) => setDatasets(d.datasets)).catch((e) => setError(String(e.message ?? e)));
  }, []);

  if (error) return <p className="err">{error}</p>;

  return (
    <>
      <h1>Datasets</h1>
      <p className="sub">
        Every model is trained on a real public dataset. Values here are measured, not assumed;
        "unknown" means not yet verified.
      </p>
      <table>
        <thead>
          <tr><th>Disease</th><th>Dataset</th><th>Source</th><th>Licence</th><th>Records</th><th>Status</th></tr>
        </thead>
        <tbody>
          {datasets.map((d) => (
            <tr key={d.disease}>
              <td>{d.disease}</td>
              <td>
                {d.dataset_name}
                {d.limitations && d.limitations !== "unknown" && (
                  <div className="muted">Limitations: {d.limitations}</div>
                )}
              </td>
              <td className="muted">
                {d.source_url && d.source_url !== "unknown" ? (
                  <a href={d.source_url} target="_blank" rel="noreferrer">{d.source}</a>
                ) : (
                  d.source
                )}
              </td>
              <td className="muted">{d.license}</td>
              <td>{d.records || "—"}</td>
              <td><span className={`pill ${d.status === "verified" ? "ok" : "no"}`}>{d.status}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
    </>
  );
}
