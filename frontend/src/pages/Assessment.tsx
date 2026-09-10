import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import type { DiseaseInfo, FeatureSchema } from "../types";

export function Assessment() {
  const nav = useNavigate();
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [schema, setSchema] = useState<FeatureSchema | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    api.diseases().then((d) => {
      setDiseases(d.diseases);
      const first = d.diseases.find((x) => x.model_available);
      if (first) setSelected(first.key);
    }).catch((e) => setError(String(e.message ?? e)));
  }, []);

  useEffect(() => {
    if (!selected) return;
    setSchema(null);
    setError(null);
    api.schema(selected)
      .then((s) => {
        setSchema(s);
        setValues(Object.fromEntries(s.required_features.map((f) => [f, ""])));
      })
      .catch((e) => setError(String(e.message ?? e)));
  }, [selected]);

  const available = useMemo(() => diseases.filter((d) => d.model_available), [diseases]);
  const unavailable = useMemo(() => diseases.filter((d) => !d.model_available), [diseases]);

  const complete = schema?.required_features.every((f) => values[f] !== "" && !Number.isNaN(Number(values[f])));

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!schema) return;
    setBusy(true);
    setError(null);
    try {
      const features = Object.fromEntries(
        schema.required_features.map((f) => [f, Number(values[f])]),
      );
      const result = await api.predict(schema.disease, features);
      sessionStorage.setItem("earlydx:lastResult", JSON.stringify(result));
      nav("/results");
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : String(err);
      setError(msg);
    } finally {
      setBusy(false);
    }
  }

  return (
    <>
      <h1>Assessment</h1>
      <p className="sub">Pick a condition, then enter only the values that condition's model needs.</p>
      <div className="disclaimer">
        This tool estimates risk from a statistical model. It does not diagnose disease and is not a
        substitute for a clinician.
      </div>

      <div className="field">
        <label htmlFor="disease">Condition</label>
        <select id="disease" value={selected} onChange={(e) => setSelected(e.target.value)}>
          {available.map((d) => (
            <option key={d.key} value={d.key}>{d.display_name}</option>
          ))}
        </select>
        {unavailable.length > 0 && (
          <p className="muted">
            No model yet: {unavailable.map((d) => d.display_name).join(", ")}.
          </p>
        )}
      </div>

      {schema && (
        <form onSubmit={submit}>
          {schema.required_features.map((f) => {
            const d = schema.feature_details[f];
            return (
              <div className="field" key={f}>
                <label htmlFor={f}>
                  {f} <span className="unit">({d.unit})</span>
                </label>
                <input
                  id={f}
                  name={f}
                  type="number"
                  step="any"
                  min={d.min}
                  max={d.max}
                  value={values[f] ?? ""}
                  onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                />
              </div>
            );
          })}
          <button type="submit" disabled={!complete || busy}>
            {busy ? "Scoring…" : "Submit assessment"}
          </button>
          {error && <p className="err">{error}</p>}
        </form>
      )}
    </>
  );
}
