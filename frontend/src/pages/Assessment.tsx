import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, ApiError } from "../api/client";
import { ApiErrorBox, Loading } from "../components/PageState";
import type { DiseaseInfo, FeatureSchema } from "../types";
import type { StoredInputs } from "./Results";
import { humanizeLabel, humanizeOption } from "../utils/format";

export function Assessment() {
  const nav = useNavigate();
  const [diseases, setDiseases] = useState<DiseaseInfo[]>([]);
  const [selected, setSelected] = useState<string>("");
  const [schema, setSchema] = useState<FeatureSchema | null>(null);
  const [values, setValues] = useState<Record<string, string>>({});
  const [touched, setTouched] = useState<Record<string, boolean>>({});
  const [error, setError] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [loadingDiseases, setLoadingDiseases] = useState(true);
  const [busy, setBusy] = useState(false);
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    setLoadingDiseases(true);
    setLoadError(null);
    api.diseases().then((d) => {
      setDiseases(d.diseases);
      const first = d.diseases.find((x) => x.model_available);
      if (first) setSelected((cur) => cur || first.key);
    }).catch((e) => setLoadError(String(e.message ?? e)))
      .finally(() => setLoadingDiseases(false));
  }, [attempt]);

  useEffect(() => {
    if (!selected) return;
    setSchema(null);
    setTouched({});
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

  function fieldInvalid(f: string): boolean {
    if (!schema) return false;
    const d = schema.feature_details[f];
    const v = values[f];
    if (v === "" || v == null) return true;
    if (d.type === "categorical") return false;
    if (Number.isNaN(Number(v))) return true;
    const n = Number(v);
    // 0 is an explicit "not measured" sentinel for this feature: the model's
    // own preprocessing treats it as missing and imputes it, so it is valid
    // input even though it falls outside [min, max]. Matches the backend's
    // validation in backend/app/schemas/__init__.py.
    if (d.zero_is_missing && n === 0) return false;
    if (d.min != null && n < d.min) return true;
    if (d.max != null && n > d.max) return true;
    return false;
  }

  const complete = schema?.required_features.every((f) => !fieldInvalid(f)) ?? false;
  const filled = schema ? schema.required_features.filter((f) => values[f] !== "" && values[f] != null).length : 0;
  // guided-workflow stage: 0 condition · 1 information · 2 assessment (ready/running) · 3 result (on /results)
  const stage = !selected || !schema ? 0 : busy ? 2 : complete ? 2 : 1;

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    if (!schema) return;
    if (!complete) {
      setTouched(Object.fromEntries(schema.required_features.map((f) => [f, true])));
      return;
    }
    setBusy(true);
    setError(null);
    try {
      const features = Object.fromEntries(
        schema.required_features.map((f) => [
          f,
          schema.feature_details[f].type === "categorical" ? values[f] : Number(values[f]),
        ]),
      );
      const result = await api.predict(schema.disease, features);
      const stored: StoredInputs = {
        disease_key: schema.disease,
        values: features,
        units: Object.fromEntries(schema.required_features.map((f) => [f, schema.feature_details[f].unit ?? ""])),
      };
      sessionStorage.setItem("earlydx:lastResult", JSON.stringify(result));
      sessionStorage.setItem("earlydx:lastInputs", JSON.stringify(stored));
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
      <span className="eyebrow">Guided workflow</span>
      <h1>New Assessment</h1>
      <p className="sub">Pick a condition, then enter only the values that condition's model needs. Fields change with the condition.</p>

      <ol className="stepper" aria-label="Assessment progress">
        {["Condition", "Information", "Assessment", "Result"].map((label, i) => (
          <li key={label}>
            <span className={`step ${i < stage ? "done" : ""} ${i === stage ? "current" : ""}`} aria-current={i === stage ? "step" : undefined}>
              <span className="step-num">{i < stage ? "✓" : `0${i + 1}`}</span>
              <span className="step-label">{label}</span>
            </span>
            {i < 3 && <span className="step-arrow" aria-hidden="true">→</span>}
          </li>
        ))}
      </ol>

      <div className="disclaimer">
        This tool estimates risk from a statistical model. It does not diagnose disease and is not a
        substitute for a clinician.
      </div>

      <h2>01 · Select a condition</h2>
      <div className="selector-card">
        <div className="field" style={{ marginBottom: 0 }}>
          <label htmlFor="disease">Condition</label>
          <select
            id="disease"
            value={selected}
            onChange={(e) => setSelected(e.target.value)}
            disabled={loadingDiseases || !available.length}
          >
            {loadingDiseases && <option value="">Loading conditions…</option>}
            {!loadingDiseases && !available.length && <option value="">No conditions available</option>}
            {available.map((d) => (
              <option key={d.key} value={d.key}>{d.display_name}</option>
            ))}
          </select>
        </div>
        {loadingDiseases && <Loading what="conditions" />}
        {loadError && <ApiErrorBox message={loadError} onRetry={() => setAttempt((a) => a + 1)} />}
        {!loadingDiseases && !loadError && available.length === 0 && (
          <p className="muted" style={{ marginTop: 8 }}>The API returned no conditions with a trained model.</p>
        )}
        {selected && schema && (
          <span className="pill ok" style={{ marginTop: 8, display: "inline-block" }}>
            Model available · {schema.required_features.length} inputs · <span className="mono">{schema.model_id}</span>
          </span>
        )}
        {unavailable.length > 0 && (
          <div className="status-note">
            No trained model available: {unavailable.map((d) => d.display_name).join(", ")}.
          </div>
        )}
      </div>

      {selected && !schema && !error && <Loading what="the input form" />}
      {error && !schema && <ApiErrorBox message={error} onRetry={() => setSelected((s) => s + "")} />}

      {schema && (
        <form onSubmit={submit} noValidate className="fade-in" key={schema.model_id}>
          <h2>02 · Enter the information this model needs</h2>
          <p className="muted" style={{ marginBottom: 12 }}>
            {filled} of {schema.required_features.length} fields filled · required inputs come from{" "}
            <span className="mono">{schema.model_id}</span>'s feature schema; nothing else is collected.
          </p>
          <div className="form-card">
          <div className="form-grid">
            {schema.required_features.map((f) => {
              const d = schema.feature_details[f];
              const invalid = touched[f] && fieldInvalid(f);
              return (
                <div className="field" key={f}>
                  <label htmlFor={f}>
                    {humanizeLabel(f)} <span className="req-mark" aria-hidden="true">*</span>
                  </label>
                  {d.unit && <div className="field-desc">{d.unit}</div>}
                  {d.zero_is_missing && (
                    <div className="field-desc">
                      Enter 0 if this was not measured — the model treats 0 as missing and imputes it.
                    </div>
                  )}
                  {d.type === "categorical" ? (
                    <select
                      id={f}
                      name={f}
                      required
                      aria-invalid={invalid}
                      value={values[f] ?? ""}
                      onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                      onBlur={() => setTouched((t) => ({ ...t, [f]: true }))}
                    >
                      <option value="" disabled>
                        select…
                      </option>
                      {(d.options ?? []).map((o) => (
                        <option key={o} value={o}>
                          {humanizeOption(o)}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      id={f}
                      name={f}
                      type="number"
                      step="any"
                      min={d.zero_is_missing ? 0 : d.min}
                      max={d.max}
                      required
                      aria-invalid={invalid}
                      value={values[f] ?? ""}
                      onChange={(e) => setValues((v) => ({ ...v, [f]: e.target.value }))}
                      onBlur={() => setTouched((t) => ({ ...t, [f]: true }))}
                    />
                  )}
                  {invalid && (
                    <div className="field-error">
                      {values[f] === ""
                        ? "This value is required."
                        : d.zero_is_missing
                          ? `Enter 0 (not measured), or a value between ${d.min} and ${d.max}.`
                          : `Enter a value between ${d.min} and ${d.max}.`}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
          </div>
          <h2>03 · Run the assessment</h2>
          <div className="form-actions">
            <button type="submit" disabled={busy}>
              {busy ? <><span className="spinner" aria-hidden="true" /> Running assessment…</> : "Run Assessment →"}
            </button>
            <span className="muted">
              {complete ? "All inputs valid — the model will score them and open the result." : "Complete every field with a valid value to enable the assessment."}
            </span>
          </div>
          {!complete && Object.keys(touched).length > 0 && (
            <p className="field-error" style={{ marginTop: 8 }}>Some fields are missing or out of range — they are marked above.</p>
          )}
          {error && <p className="err" role="alert">{error}</p>}
        </form>
      )}
    </>
  );
}
