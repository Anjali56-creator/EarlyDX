import { useEffect, useState } from "react";
import { api } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { DatasetEntry, DiseaseInfo } from "../types";
import { humanizeLabel } from "../utils/format";

function isKnown(v: string | undefined | null): boolean {
  return !!v && v.toLowerCase() !== "unknown";
}

export function Datasets() {
  const [datasets, setDatasets] = useState<DatasetEntry[]>([]);
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setLoading(true);
    setError(null);
    Promise.all([api.datasets(), api.diseases()])
      .then(([d, dz]) => { setDatasets(d.datasets); setDiseases(dz.diseases); })
      .catch((e) => setError(String(e.message ?? e)))
      .finally(() => setLoading(false));
  }, [attempt]);

  const displayName = (key: string) => diseases.find((d) => d.key === key)?.display_name ?? key;
  const total = datasets.reduce((n, d) => n + (Number(d.records) || 0), 0);

  return (
    <>
      <span className="eyebrow">Dataset registry</span>
      <h1>Datasets</h1>
      <p className="sub">
        Every model is trained on a real, public, de-identified dataset. Values here were measured by{" "}
        <span className="mono">scripts/inspect_dataset.py</span> after download; "unknown" means not yet verified
        and is never replaced with a guess.
      </p>

      {error && <ApiErrorBox message={error} onRetry={() => setAttempt((a) => a + 1)} />}
      {loading && !error && <Loading what="dataset registry" />}

      {datasets.length > 0 && (
        <div className="grid">
          <div className="card">
            <div className="k">Datasets</div>
            <div className="v">{datasets.length}</div>
            <div className="hint">{datasets.filter((d) => d.status === "verified").length} verified against source (sha256)</div>
          </div>
          <div className="card">
            <div className="k">Records total</div>
            <div className="v">{total.toLocaleString()}</div>
            <div className="hint">across all training datasets</div>
          </div>
          <div className="card">
            <div className="k">Smallest / largest</div>
            <div className="v">
              {Math.min(...datasets.map((d) => Number(d.records) || Infinity)).toLocaleString()} /{" "}
              {Math.max(...datasets.map((d) => Number(d.records) || 0)).toLocaleString()}
            </div>
            <div className="hint">small datasets → wide confidence intervals on every metric</div>
          </div>
        </div>
      )}

      <div className="table-scroll">
        <table>
          <thead>
            <tr>
              <th>Condition</th>
              <th>Dataset</th>
              <th>Records</th>
              <th>Positive class</th>
              <th>Features</th>
              <th>Source</th>
              <th>License</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {datasets.map((d) => {
              const pos = d.class_distribution?.["1"];
              const neg = d.class_distribution?.["0"];
              const rate = pos != null && neg != null && pos + neg > 0 ? (pos / (pos + neg)) * 100 : null;
              return (
                <tr key={d.disease}>
                  <td>{displayName(d.disease)}</td>
                  <td>{d.dataset_name}</td>
                  <td>{d.records ? d.records.toLocaleString() : "—"}</td>
                  <td>{rate != null ? `${rate.toFixed(1)}% (${pos}/${pos! + neg!})` : "—"}</td>
                  <td>{d.features?.length ?? "—"}</td>
                  <td className={isKnown(d.source) ? "" : "muted"}>
                    {isKnown(d.source_url) ? (
                      <a href={d.source_url} target="_blank" rel="noreferrer">{d.source}</a>
                    ) : isKnown(d.source) ? d.source : <em>Unknown</em>}
                  </td>
                  <td className={isKnown(d.license) ? "" : "muted"}>{isKnown(d.license) ? d.license : <em>Unknown</em>}</td>
                  <td><span className={`pill ${d.status === "verified" ? "ok" : "no"}`}>{d.status}</span></td>
                </tr>
              );
            })}
            {!loading && datasets.length === 0 && <tr><td colSpan={8} className="muted">No datasets registered.</td></tr>}
          </tbody>
        </table>
      </div>

      <h2>Per-dataset detail</h2>
      <p className="muted">
        Target definition, input features and the preprocessing applied inside each model's pipeline (always fitted on
        the training split only, so no information from validation or test rows leaks into scaling or imputation).
      </p>
      {datasets.map((d) => (
        <details className="model-details" key={d.disease} id={`dataset-${d.disease}`}>
          <summary>{displayName(d.disease)} — {d.dataset_name}</summary>
          <table>
            <tbody>
              <tr><th>Target</th><td>{isKnown(d.target) ? d.target : <em className="muted">unknown</em>}</td></tr>
              {d.class_distribution && (
                <tr>
                  <th>Class distribution</th>
                  <td>
                    negative (0): {d.class_distribution["0"]?.toLocaleString() ?? "—"} · positive (1):{" "}
                    {d.class_distribution["1"]?.toLocaleString() ?? "—"}
                  </td>
                </tr>
              )}
              <tr>
                <th>Features ({d.features?.length ?? 0})</th>
                <td>{d.features?.length ? d.features.map(humanizeLabel).join(", ") : "—"}</td>
              </tr>
              <tr><th>Preprocessing</th><td>{isKnown(d.preprocessing) ? d.preprocessing : <em className="muted">not documented</em>}</td></tr>
              <tr><th>Limitations</th><td>{isKnown(d.limitations) ? d.limitations : <em className="muted">not documented</em>}</td></tr>
              <tr><th>Source</th><td>{isKnown(d.source_url) ? <a href={d.source_url} target="_blank" rel="noreferrer">{d.source_url}</a> : d.source}</td></tr>
            </tbody>
          </table>
        </details>
      ))}
    </>
  );
}
