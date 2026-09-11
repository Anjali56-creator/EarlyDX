import { useEffect, useMemo, useState } from "react";
import { api, ApiError } from "../api/client";
import type { DiseaseInfo, ValidationSamples } from "../types";

export function Validation() {
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [data, setData] = useState<ValidationSamples | null>(null);
  const [notAvailable, setNotAvailable] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    api.diseases().then((d) => {
      setDiseases(d.diseases);
      const first = d.diseases.find((x) => x.model_available);
      if (first) setSelected(first.key);
    }).catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setData(null);
    setNotAvailable(null);
    setError(null);
    api.validationSamples(selected)
      .then(setData)
      .catch((e) => {
        if (e instanceof ApiError && e.status === 404) {
          setNotAvailable("No stored validation samples for this model yet.");
        } else {
          setError(String(e.message ?? e));
        }
      });
  }, [selected]);

  const available = useMemo(() => diseases.filter((d) => d.model_available), [diseases]);

  return (
    <>
      <h1>Validation</h1>
      <p className="sub">Compare the model against known, held-out test-set records.</p>
      <div className="disclaimer">
        This is model validation, not diagnosis. Every record below is from the test split that
        was set aside before training and never used for training, hyperparameter tuning,
        calibration, or threshold selection — so these are genuine, unseen evaluations of the model.
      </div>

      <div className="selector-card">
        <div className="field" style={{ marginBottom: 0 }}>
          <label htmlFor="val-disease">Condition</label>
          <select id="val-disease" value={selected} onChange={(e) => setSelected(e.target.value)}>
            {available.map((d) => (
              <option key={d.key} value={d.key}>{d.display_name}</option>
            ))}
          </select>
        </div>
      </div>

      {error && <p className="err">{error}</p>}
      {notAvailable && <p className="muted">{notAvailable}</p>}

      {data && (
        <>
          <p className="muted">Model: <span className="mono">{data.model_version}</span> — {data.note}</p>
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>#</th>
                  <th>Actual outcome</th>
                  <th>Model prediction</th>
                  <th>Risk score</th>
                  <th>Risk level</th>
                  <th>Result</th>
                </tr>
              </thead>
              <tbody>
                {data.samples.map((s, i) => {
                  const correct = s.predicted_outcome === s.actual_outcome;
                  return (
                    <tr key={i}>
                      <td>{i + 1}</td>
                      <td>{s.actual_outcome === 1 ? "Positive" : "Negative"}</td>
                      <td>{s.predicted_outcome === 1 ? "Positive" : "Negative"}</td>
                      <td>{s.predicted_probability.toFixed(3)}</td>
                      <td><span className={`pill ${s.predicted_risk_level === "LOW" ? "ok" : "no"}`}>{s.predicted_risk_level}</span></td>
                      <td>
                        <span className={`pill ${correct ? "ok" : "no"}`}>
                          {correct ? "Correct classification" : "Incorrect classification"}
                        </span>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </>
  );
}
