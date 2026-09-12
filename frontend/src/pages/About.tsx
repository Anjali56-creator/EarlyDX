import { Link } from "react-router-dom";

const API_BASE = (import.meta.env.VITE_API_BASE ?? "").replace(/\/+$/, "");

export function About() {
  return (
    <>
      <span className="eyebrow">Project</span>
      <h1>About EarlyDX</h1>
      <p className="sub">Objective, method, architecture, and what this system is not.</p>

      <h2>Objective</h2>
      <p>
        EarlyDX is a final-year research prototype exploring what a <em>transparent</em>, multi-disease
        risk-screening tool looks like when every number shown to the user is traceable to an executed training
        run: real public datasets with documented provenance, one independently trained classifier per condition, a
        comparative analysis of candidate algorithms, and validation-derived (not hand-picked) decision thresholds.
      </p>

      <h2>Problem statement</h2>
      <p>
        Many "disease prediction" demos report a single accuracy figure on a small dataset and present a binary
        verdict. That hides three things a reviewer needs: how the model was <strong>chosen</strong> against
        alternatives, how it performs on data it <strong>never saw</strong> (including the errors it makes), and
        how far the training population is from the person using it. EarlyDX is built to expose all three rather
        than to maximise a headline metric.
      </p>

      <h2>ML approach — comparative model analysis</h2>
      <ol>
        <li><strong>Dataset</strong> — downloaded, sha256-verified and inspected (<span className="mono">scripts/inspect_dataset.py</span>); source, licence, class balance and limitations recorded in the dataset registry.</li>
        <li><strong>Preprocessing</strong> — per-condition scikit-learn pipeline (imputation, scaling, encoding) fitted on the <em>training split only</em>, so no leakage into validation or test.</li>
        <li><strong>Split</strong> — stratified 60 / 20 / 20 train / validation / test, fixed <span className="mono">random_state = 42</span>.</li>
        <li><strong>Candidates</strong> — a LogisticRegression baseline versus RandomForest and GradientBoosting (HistGradientBoosting is also available in the framework), each tuned with <span className="mono">GridSearchCV</span> + stratified 5-fold CV on the training split.</li>
        <li><strong>Model selection</strong> — candidates compared on the validation split; the interpretable baseline is kept unless a challenger beats it by ≥ 0.01 ROC-AUC. Selection deliberately favours sensitivity and calibration over raw accuracy.</li>
        <li><strong>Calibration</strong> — none / sigmoid / isotonic chosen by validation Brier score, so a "risk score" is a probability only when labelled calibrated.</li>
        <li><strong>Thresholds</strong> — LOW / MODERATE / HIGH cut points derived from validation sensitivity and specificity targets (0.85), never invented.</li>
        <li><strong>Evaluation</strong> — <em>one</em> held-out test evaluation: accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC, confusion matrix, Brier score and calibration bins; plus permutation importance on the test split.</li>
        <li><strong>Deployment</strong> — the serialised pipeline, feature schema, metadata and registry entry are committed and served unchanged by the API.</li>
      </ol>
      <p className="muted">
        All of this is visible in the app: <Link to="/validation">Validation &amp; model comparison</Link> shows the
        candidate table, verdict and confusion matrix for every condition; <Link to="/models">Models</Link> and{" "}
        <Link to="/datasets">Datasets</Link> show the registries the API reads from.
      </p>

      <h2>Architecture</h2>
      <pre className="arch">{`Browser (React 18 + TypeScript + Vite)          hosted on Vercel
        │  HTTPS, JSON — single API client (src/api/client.ts)
        ▼
FastAPI backend (Python 3.11, Pydantic v2)     hosted on Render
        ├── /diseases /schema/{d} /models /datasets      registries + feature schemas
        ├── /predict/{d} /predict/all                    validated input → trained pipeline → score, level, factors
        ├── /evaluation /evaluation/{d} /validation/{d}  stored training-run evaluation (never recomputed)
        └── /assessments/recent /health
        │
        ├── models/*.joblib + model_registry.json        one scikit-learn pipeline per condition
        ├── ml/<condition>/model_metadata.json           candidate comparison, test metrics, thresholds
        └── SQLAlchemy → SQLite (default) / PostgreSQL   best-effort assessment history (no inputs stored)`}</pre>
      <p>
        <strong>API boundary.</strong> The frontend never touches model files or datasets; it only calls the
        endpoints above. In production the base URL is injected at build time (<span className="mono">VITE_API_BASE</span>);
        in local development requests go to <span className="mono">/api</span> and the Vite dev server proxies them to the
        backend on port 8000. The backend allows only the deployed frontend origin via CORS.
        {API_BASE && <> This build talks to <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">{API_BASE}</a>.</>}
      </p>

      <h2>Technology stack</h2>
      <table>
        <tbody>
          <tr><th>Frontend</th><td>React 18, TypeScript 5, Vite 5, React Router 6 · Vitest + Testing Library</td></tr>
          <tr><th>Backend</th><td>Python 3.11, FastAPI, Pydantic v2, Uvicorn · pytest + httpx</td></tr>
          <tr><th>Machine learning</th><td>scikit-learn 1.4 (pipelines, GridSearchCV, CalibratedClassifierCV, permutation importance), pandas, NumPy, joblib</td></tr>
          <tr><th>Storage</th><td>SQLAlchemy 2 — SQLite by default, PostgreSQL (psycopg 3) optional</td></tr>
          <tr><th>Deployment</th><td>Vercel (static frontend) · Render (API web service, Blueprint in <span className="mono">render.yaml</span>)</td></tr>
        </tbody>
      </table>

      <h2>Limitations</h2>
      <ul>
        <li>Datasets are small (195 – 2 111 records) and several come from a single hospital, region or demographic group; every metric therefore has a wide confidence interval and none of the models has been externally validated.</li>
        <li>Some datasets encode missing values as zeros or are imbalanced; see each dataset's documented limitations.</li>
        <li>The Parkinson's model is weak (test ROC-AUC well below the others) and is kept only for transparency.</li>
        <li>"Inputs the model relies on" uses global permutation importance, not per-case attribution — it describes model behaviour, not causes.</li>
        <li>Two in-scope conditions (Stroke, Hypertension) have no model because no dataset with acceptable provenance or a rigorously defined target was found.</li>
      </ul>

      <h2>What EarlyDX is not</h2>
      <div className="disclaimer">
        Not a medical device. Not clinically validated. Not a diagnosis. Not HIPAA / GDPR compliant. A risk score is
        a statistical model output, not a statement about anyone's health — consult a qualified clinician for medical
        concerns.
      </div>

      <h2>Documentation</h2>
      <p>
        <span className="mono">README.md</span>, <span className="mono">ARCHITECTURE.md</span>,{" "}
        <span className="mono">DATASETS.md</span>, <span className="mono">MODEL_CARD.md</span> (generated from the
        evaluation runs) and <span className="mono">LIMITATIONS.md</span> in the repository.
        {API_BASE && <> Interactive API docs: <a href={`${API_BASE}/docs`} target="_blank" rel="noreferrer">{API_BASE}/docs</a>.</>}
      </p>
    </>
  );
}
