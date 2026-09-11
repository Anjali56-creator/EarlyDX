import { useEffect, useState } from "react";
import { api } from "../api/client";
import type { DatasetEntry } from "../types";

function isKnown(v: string | undefined | null): boolean {
  return !!v && v.toLowerCase() !== "unknown";
}

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
        "unknown" means not yet verified — it is never replaced with a guess.
      </p>
      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Disease</th>
              <th>Dataset</th>
              <th>Records</th>
              <th>Source</th>
              <th>License</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => (
              <tr key={d.disease}>
                <td>{d.disease}</td>
                <td>{d.dataset_name}</td>
                <td>{d.records || "—"}</td>
                <td className={isKnown(d.source) ? "" : "muted"}>
                  {isKnown(d.source_url) ? (
                    <a href={d.source_url} target="_blank" rel="noreferrer">{d.source}</a>
                  ) : isKnown(d.source) ? (
                    d.source
                  ) : (
                    <em>Unknown</em>
                  )}
                </td>
                <td className={isKnown(d.license) ? "" : "muted"}>
                  {isKnown(d.license) ? d.license : <em>Unknown</em>}
                </td>
                <td><span className={`pill ${d.status === "verified" ? "ok" : "no"}`}>{d.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card-list">
        {datasets.map((d) => (
          isKnown(d.limitations) && (
            <div className="card limitations-card" key={d.disease}>
              <div className="k">{d.disease} — limitations</div>
              <p className="muted" style={{ margin: "6px 0 0" }}>{d.limitations}</p>
            </div>
          )
        ))}
      </div>
    </>
  );
}
