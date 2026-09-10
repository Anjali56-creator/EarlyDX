## EarlyDX — Architecture

**EarlyDX** is a multi-disease *early risk assessment* platform. It trains one
independent ML pipeline per disease on real, publicly available, de-identified
datasets, and serves per-disease risk scores through a FastAPI backend and a
React + TypeScript frontend.

> EarlyDX produces **risk assessments, not diagnoses**. It is a portfolio /
> research prototype. It is **not** a medical device and is **not** clinically
> validated.

---

## 1. Design principles

1. **One disease = one pipeline = one model artifact.** Datasets are never merged
   into a single dataframe. Each disease owns its feature schema, preprocessing,
   model, metrics, and version.
2. **Real data only for real metrics.** Synthetic data is confined to frontend
   demos and tests, and is always labelled `DEMO / SYNTHETIC DATA — NOT REAL
   PATIENT DATA`. It never trains a production model or produces a reported
   metric.
3. **Leakage-safe by construction.** Every parameter-learning transform lives
   inside a scikit-learn `Pipeline` that is fitted on the training split only.
   Splitting happens before any fitting.
4. **Additive extension.** Adding disease N+1 means adding a directory and a
   registry entry, not editing shared code paths.
5. **Honest outputs.** If a model's probabilities are not calibrated, the API
   returns `risk_score`, not `probability_of_disease`.

---

## 2. Repository layout

```text
EarlyDX/
├── ARCHITECTURE.md
├── DATASETS.md
├── MODEL_CARD.md          # created as models are trained
├── LIMITATIONS.md
├── README.md
│
├── data/
│   ├── raw/<disease>/      # git-ignored; populated by scripts/download or user
│   ├── processed/<disease>/# git-ignored
│   ├── metadata/           # per-dataset inspection reports (committed)
│   └── dataset_registry.json
│
├── ml/
│   ├── common/
│   │   ├── preprocessing/  # reusable transformers, column-spec helpers
│   │   ├── evaluation/     # metric computation, plots, CV runners
│   │   ├── model_registry/ # read/write model_registry.json, artifact IO
│   │   └── utilities/      # seeding, logging, IO, schema validation
│   ├── diabetes/
│   ├── heart_disease/
│   ├── kidney_disease/
│   ├── liver_disease/
│   ├── breast_cancer/
│   ├── parkinsons/
│   ├── heart_failure/
│   ├── stroke/
│   ├── hypertension/
│   ├── thyroid/
│   ├── obesity/
│   └── maternal_health/
│       ├── load.py         # dataset loading + validation
│       ├── eda.ipynb       # exploratory analysis (committed, outputs cleared)
│       ├── pipeline.py     # sklearn Pipeline: clean -> impute -> encode -> scale
│       ├── train.py        # split -> baseline -> candidate -> CV -> eval -> serialize
│       ├── evaluate.py     # metrics on held-out test set
│       ├── feature_schema.json
│       ├── model_metadata.json
│       └── artifacts/      # git-ignored .joblib
│
├── models/                 # promoted artifacts consumed by the backend
│   └── model_registry.json
│
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── config/         # pydantic-settings, env loading
│   │   ├── api/            # routers: health, diseases, models, datasets, predict
│   │   ├── schemas/        # pydantic request/response models per disease
│   │   ├── services/       # prediction orchestration, registry access
│   │   ├── models/         # artifact loader + in-memory model cache
│   │   ├── preprocessing/  # shared with ml/common via a small package
│   │   ├── explainability/ # SHAP / coefficients / permutation importance
│   │   └── database/       # SQLAlchemy models, session, migrations (alembic)
│   └── tests/
│
├── frontend/
│   └── src/
│       ├── pages/          # dashboard, assessment, results, diseases, models,
│       │                   #   datasets, about
│       ├── components/
│       ├── api/            # typed client, generated from OpenAPI
│       └── types/
│
├── scripts/
│   ├── download_<disease>.py   # documented, licence-aware fetchers
│   └── inspect_dataset.py      # duplicates / leakage / missing / imbalance report
│
└── docker-compose.yml     # postgres + backend + frontend for local dev
```

---

## 3. ML pipeline contract

Every `ml/<disease>/` module exposes the same functions so `ml/common` can drive
them uniformly:

| Step | Function | Notes |
|------|----------|-------|
| Load | `load.load_raw() -> DataFrame` | Reads from `data/raw/<disease>/`; raises if absent. |
| Validate | `load.validate(df)` | Row count, column names/dtypes, target present, value ranges. |
| Split | `train.split(df)` | Stratified train/val/test (60/20/20 default), fixed `RANDOM_STATE = 42`. |
| Pipeline | `pipeline.build() -> sklearn.Pipeline` | Impute -> encode -> scale; all fitted on train only. |
| Baseline | Logistic Regression (class-weighted) | Always trained for comparison. |
| Candidate | Decision Tree / Random Forest / Gradient Boosting / XGBoost | Chosen per dataset, justified in `MODEL_CARD.md`. |
| CV | Stratified k-fold (k=5) on train+val | Reports mean +/- std for each metric. |
| Evaluate | `evaluate.report(model, test)` | Metrics below, from a single held-out test set. |
| Serialize | `joblib.dump` to `artifacts/` | Pipeline + model in one object. |
| Schema | `feature_schema.json` | Required features, types, units, allowed ranges. |
| Metadata | `model_metadata.json` | Algorithm, dataset ref, date, seed, metrics, calibration method. |

**Reported metrics (all from executed evaluation, never hand-written):**
accuracy, precision, recall/sensitivity, specificity, F1, ROC-AUC, PR-AUC,
confusion matrix, per-fold CV scores, class distribution. Model selection
prioritises **sensitivity / low false-negative rate** for early detection, not
accuracy.

**Leakage checks** (run by `scripts/inspect_dataset.py`, recorded in
`data/metadata/<disease>.md`): exact/near-duplicate rows, duplicate patients,
features that are deterministic functions of the target, target present among
predictors, pre-split normalisation, temporal ordering where relevant.

**Calibration:** after selecting a candidate, reliability curve + Brier score on
the validation split. If poorly calibrated, apply `CalibratedClassifierCV`
(Platt or isotonic, chosen by validation Brier) and document it; otherwise the
output field is named `risk_score`.

---

## 4. Model registry

`models/model_registry.json` — one entry per promoted model:

```json
{
  "model_id": "diabetes-v1",
  "disease": "diabetes",
  "version": "v1",
  "algorithm": "RandomForestClassifier",
  "dataset": "pima-indians-diabetes | cdc-diabetes-health-indicators",
  "training_date": "YYYY-MM-DD",
  "features": ["glucose", "bmi", "age", "..."],
  "metrics": { "roc_auc": 0.0, "recall": 0.0, "pr_auc": 0.0 },
  "calibration": "none | platt | isotonic",
  "status": "active | deprecated | experimental",
  "artifact_path": "models/diabetes-v1.joblib"
}
```

The backend loads artifacts listed here, and every prediction response echoes the
`model_id` / `version` that produced it.

---

## 5. Backend

**Stack:** Python 3.11, FastAPI, pydantic v2, SQLAlchemy 2 + Alembic, joblib,
scikit-learn, SHAP. Served by uvicorn.

**Endpoints**

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/health` | Liveness + loaded-model count. |
| `GET` | `/diseases` | Supported diseases + whether a model is available. |
| `GET` | `/models` | Registry contents (version, algorithm, metrics, status). |
| `GET` | `/datasets` | Dataset registry contents. |
| `GET` | `/schema/{disease}` | Required feature schema for a disease form. |
| `POST` | `/predict/{disease}` | Validate against schema -> run pipeline -> risk score + explanation. |
| `POST` | `/predict/all` | Runs only the diseases whose required features are fully satisfied by the payload. Missing-feature diseases are returned in a `skipped` list with the reason. Never fills medical values with defaults. |

**Prediction flow**

```text
request -> pydantic schema (strict, per disease)
        -> feature-schema completeness check
        -> load cached pipeline (joblib)
        -> pipeline.predict_proba -> risk_score
        -> risk_level via documented thresholds (per model, in metadata)
        -> explainability: SHAP (tree/boosting) or standardized coefficients (LR)
        -> persist session + result (no PII) -> response
```

**Response shape**

```json
{
  "disease": "Diabetes",
  "risk_score": 0.82,
  "risk_level": "HIGH",
  "model_version": "diabetes-v1",
  "important_features": [{ "feature": "glucose", "impact": "high" }],
  "calibrated": false,
  "disclaimer": "This is a risk assessment and not a medical diagnosis."
}
```

Thresholds for `LOW / MODERATE / HIGH` are stored per model in
`model_metadata.json` (derived on the validation split, e.g. by target
sensitivity), never hard-coded in the API.

---

## 6. Database (PostgreSQL)

| Table | Key columns | Notes |
|-------|-------------|-------|
| `assessment_session` | `id`, `created_at`, `client_context` | One per submitted assessment. No name/DOB/contact fields. |
| `prediction_result` | `id`, `session_id`, `disease`, `model_id`, `risk_score`, `risk_level`, `created_at` | One row per disease evaluated. |
| `model_version` | `model_id`, `disease`, `version`, `algorithm`, `metrics_json`, `status`, `artifact_path` | Mirror of the registry for querying/audit. |
| `dataset_metadata` | `disease`, `dataset_name`, `source`, `license`, `records`, `status` | Mirror of `dataset_registry.json`. |
| `audit_log` | `id`, `ts`, `action`, `entity`, `detail_json` | Model loads, prediction calls, registry changes. |

Demo/synthetic sessions are flagged with `client_context = "demo"` and are
excluded from any aggregate shown as "real".

---

## 7. Frontend

**Stack:** React + TypeScript + Vite, TanStack Query, a typed API client
generated from the backend OpenAPI schema.

| Route | Content |
|-------|---------|
| `/dashboard` | Supported-disease count, model count + status, dataset coverage, recent assessments, risk distribution. Clearly separates demo vs real aggregates. |
| `/assessment` | Disease picker -> dynamically rendered form from `/schema/{disease}` (demographics, vitals, labs, lifestyle, history — only fields the model needs). |
| `/results` | Risk level banner (HIGH / MODERATE / LOW), score, contributing factors, model version + its evaluation metrics, disclaimer. Uses "the model estimates ...", never "you have ...". |
| `/diseases` | The 12 conditions, model availability, feature requirements. |
| `/models` | Registry view: algorithm, metrics, calibration, status. |
| `/datasets` | Registry view: source, licence, records, limitations, status. |
| `/about` | Purpose, non-clinical disclaimer, methodology, links to docs. |

---

## 8. Configuration & security

- All secrets via environment variables; `.env` git-ignored, `.env.example`
  committed.
- Strict pydantic schemas; server-side range validation from the feature schema.
- Structured JSON logging; DB credentials only from env; least-privilege DB user.
- **No compliance claims.** The UI and docs state explicitly that EarlyDX is not
  HIPAA/GDPR compliant, not medically certified, and not clinically validated.

---

## 9. Testing

- **ML:** preprocessing transforms, feature-schema validation, artifact load,
  prediction on a fixed synthetic row, missing-value handling, out-of-range
  rejection.
- **Backend:** every GET endpoint, `/predict/{disease}` happy path + invalid +
  missing-required-feature, `/predict/all` partial-coverage behaviour.
- **Frontend:** the select-disease -> enter-data -> submit -> render-result ->
  render-explanation flow.
- Tests are fixed by changing the implementation, never by weakening the test.

---

## 10. Local run

```text
docker compose up        # postgres
cd backend && uvicorn app.main:app --reload
cd frontend && npm run dev
```

Full instructions live in `README.md` (added in Phase 3).
