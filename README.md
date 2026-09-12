# EarlyDX

**Multi-disease early risk assessment with comparative ML model analysis — a research prototype, not a diagnostic system.**

EarlyDX trains one independent, leakage-safe machine-learning pipeline per disease on
real, publicly available, de-identified datasets, and serves per-disease **risk
estimates** through a FastAPI backend and a React + TypeScript frontend. Every number
the app shows — a metric, a threshold, an explanation — comes from an executed
training/evaluation run. Nothing is hand-written or invented.

> **Not a medical device. Not clinically validated. Not a diagnosis.**
> A risk score is a model output, not a statement about your health.
> See [`LIMITATIONS.md`](LIMITATIONS.md).

---

## Live Demo

**▶ https://early-dx.vercel.app**

| | URL | Hosted on |
|---|---|---|
| Live frontend | https://early-dx.vercel.app | Vercel |
| Backend API | https://earlydx-api.onrender.com | Render |
| API documentation (Swagger UI) | https://earlydx-api.onrender.com/docs | Render |
| API health check | https://earlydx-api.onrender.com/health | Render |

The React frontend is a static Vite build served by **Vercel**; the FastAPI backend and
the trained models run as a Python web service on **Render**. The frontend calls the
Render API directly over HTTPS (see [Deployment](#deployment)).

> The backend runs on Render's free tier, which spins down after ~15 minutes of
> inactivity. The first request after idle can take **up to a minute** while the
> instance wakes up and loads the 10 model artifacts — later requests are fast.

---

## Screenshots

| Dashboard | Assessment |
|---|---|
| ![Dashboard](docs/screenshots/dashboard.png) | ![Assessment](docs/screenshots/assessment.png) |

| Results | Validation |
|---|---|
| ![Results](docs/screenshots/results.png) | ![Validation](docs/screenshots/validation.png) |

<details>
<summary>More screenshots — Diseases, Models, Datasets</summary>

| Diseases | Models |
|---|---|
| ![Diseases](docs/screenshots/diseases.png) | ![Models](docs/screenshots/models.png) |

![Datasets](docs/screenshots/datasets.png)

</details>

All screenshots above are from the app actually running end to end — the Results page
shows a real prediction for the diabetes model, and the Validation page shows the
model's real held-out test-set predictions against their known labels.

---

## What EarlyDX is

A portfolio / research project exploring what a transparent, multi-disease risk-screening
tool looks like when every claim is traceable to code: real datasets with documented
provenance and limitations, one scikit-learn pipeline per disease trained with
cross-validated hyperparameter search, validation-derived (not invented) decision
thresholds, and a factor layer built on the model's own permutation importance.

### Problem statement

Typical "disease prediction" demos report one accuracy figure on a small dataset and return a
binary verdict. That hides what a reviewer actually needs to judge the work: how the model
was **chosen against alternatives**, how it behaves on data it **never saw** (including the
errors it makes), and how far the training population is from the person using it. EarlyDX
is built to expose all three — per condition, in the UI — rather than to maximise a headline
metric.

**How it works:**

1. Pick a supported condition on the Assessment page.
2. Enter only the values that condition's model needs (with plain-language labels and
   units — the raw feature keys the model actually trained on are unchanged underneath).
3. EarlyDX sends those values to that disease's trained model.
4. The model returns a risk score and, where its own validation run defined them, a
   risk level.
5. The Results page shows the score together with the model's real performance metrics,
   its own explanation of which inputs it weighed most, and the research disclaimer.

---

## Features

- **Per-condition risk assessment** — a form generated from each model's own feature
  schema (plain-language labels and units); missing medical values are never
  silently defaulted.
- **Assess-all mode** (`POST /predict/all`) — runs every model whose full feature set is
  satisfied and reports the rest as *skipped*, with the reason.
- **Results with provenance** — calibrated score with its position on the model's own
  risk bands, the inputs you entered, the model's test-set accuracy / precision / recall /
  F1 / ROC-AUC / PR-AUC, and the inputs the model relies on most (global permutation
  importance — labelled as such, not as a per-case explanation).
- **Validation & model comparison page** — for every condition: the three candidate
  algorithms side by side (accuracy, precision, recall, specificity, F1, ROC-AUC, PR-AUC on
  the validation split), the selection verdict and why, the deployed model's held-out test
  metrics, confusion matrix, cross-validation mean ± std, calibration-method comparison by
  Brier score, permutation importance, and the real held-out rows where they were stored.
- **Transparent model registry** — algorithm, version, candidates compared, calibration
  method and test metrics for every served model, straight from the registry and metadata.
- **Dataset registry** — source, licence, size, class balance, target definition, features,
  preprocessing and documented limitations of every dataset.
- **Research dashboard** — the Problem → Data → Models → Evaluation → Prediction → Results
  pipeline, a comparative-analysis table across all conditions, system status, and recent
  assessments (when a database is configured; inputs are never stored).
- **Honest unavailability** — the 2 conditions without a model are listed with the real
  reason instead of being hidden.
- **Best-effort assessment history** — persisted when a database is configured;
  predictions work with or without it.

## Tech stack

| Layer | Technology |
|---|---|
| Frontend | React 18, TypeScript 5, Vite 5, React Router 6; Vitest + Testing Library |
| Backend | Python 3.11, FastAPI, Pydantic v2 / pydantic-settings, Uvicorn |
| ML | scikit-learn 1.4 (pipelines, `GridSearchCV`, calibration, permutation importance), pandas, NumPy, joblib |
| Data / storage | SQLAlchemy 2 — SQLite by default, PostgreSQL (psycopg 3) optional |
| Testing | pytest + httpx (backend and ML), Vitest + jsdom (frontend) |
| Deployment | Vercel (frontend, `frontend/vercel.json`), Render (backend, `render.yaml`), Docker Compose (local Postgres) |

---

## Status

| Condition | Group | Model |
|---|---|---|
| Diabetes | metabolic | ✅ trained (RandomForest, ROC-AUC 0.83) |
| Breast Cancer | oncology | ✅ trained (LogisticRegression, ROC-AUC 0.99) |
| Heart Disease | cardiovascular | ✅ trained (LogisticRegression, ROC-AUC 0.96) |
| Chronic Kidney Disease | renal | ✅ trained (LogisticRegression, ROC-AUC 1.00) |
| Liver Disease | hepatic | ✅ trained (RandomForest, ROC-AUC 0.79) |
| Heart Failure | cardiovascular | ✅ trained (RandomForest, ROC-AUC 0.79) |
| Thyroid Disease | endocrine | ✅ trained (RandomForest, ROC-AUC 1.00) |
| Obesity | metabolic | ✅ trained (RandomForest, ROC-AUC 0.96) |
| Maternal Health Risk | obstetric | ✅ trained (RandomForest, ROC-AUC 0.95) |
| Parkinson's Disease | neurological | ✅ trained (GradientBoosting, ROC-AUC 0.57 — weak; kept for transparency, see `MODEL_CARD.md`) |
| Stroke | cardiovascular | ⛔ no model — no public dataset with acceptable provenance/licensing found without personal-account credentials |
| Hypertension | cardiovascular | ⛔ no model — no rigorously defined public target for a real V1 label |

**10 of 12** in-scope conditions have a trained, actively-served model. The other 2 are
shown in the app as unavailable, with the real reason, rather than silently omitted.

### Comparative model analysis

For every condition the same three candidates were tuned (`GridSearchCV`, stratified 5-fold
CV on the training split) and compared on the validation split. The selection rule is
deliberately conservative: **keep the interpretable `LogisticRegression` baseline unless a
challenger beats it by ≥ 0.01 validation ROC-AUC.** The winner is then evaluated **once** on
the held-out test split. Numbers below are from that test evaluation at the 0.5 threshold
(`ml/<disease>/model_metadata.json`, surfaced by `GET /evaluation`).

| Condition | Deployed | Compared against | Test n | Accuracy | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|---|
| Breast Cancer | LogisticRegression (baseline kept) | RandomForest, GradientBoosting | 114 | 0.974 | 0.952 | 0.964 | 0.994 |
| Chronic Kidney Disease | LogisticRegression (baseline kept) | RandomForest, GradientBoosting | 80 | 0.988 | 0.980 | 0.990 | 1.000 |
| Heart Disease | LogisticRegression (baseline kept) | RandomForest, GradientBoosting | 61 | 0.869 | 0.929 | 0.867 | 0.958 |
| Diabetes | RandomForest | LogisticRegression, GradientBoosting | 154 | 0.747 | 0.593 | 0.621 | 0.825 |
| Heart Failure | RandomForest | LogisticRegression, GradientBoosting | 60 | 0.733 | 0.474 | 0.529 | 0.789 |
| Liver Disease | RandomForest | LogisticRegression, GradientBoosting | 117 | 0.735 | 0.843 | 0.819 | 0.794 |
| Maternal Health Risk | RandomForest | LogisticRegression, GradientBoosting | 86 | 0.884 | 0.870 | 0.800 | 0.949 |
| Obesity | RandomForest | LogisticRegression, GradientBoosting | 423 | 0.910 | 0.882 | 0.901 | 0.964 |
| Thyroid Disease | RandomForest | LogisticRegression, GradientBoosting | 43 | 1.000 | 1.000 | 1.000 | 1.000 |
| Parkinson's Disease | GradientBoosting | LogisticRegression, RandomForest | 43 | 0.721 | 1.000 | 0.838 | 0.573 |

Recall, specificity, precision, PR-AUC, confusion matrices and calibration for every row are
on the app's **Validation** page and in [`MODEL_CARD.md`](MODEL_CARD.md). Two honest
caveats the comparison makes visible:

- **Accuracy alone is not enough for medical screening.** The Parkinson's model scores
  0.72 accuracy, 1.00 recall and 0.84 F1, yet its test confusion matrix is
  `tn=0, fp=12, fn=0, tp=31` — at the 0.5 threshold it predicts every case positive, so
  those figures merely reflect the 72 % positive class. Its ROC-AUC of 0.57 is the honest number. It is kept only for
  transparency.
- **Perfect scores on tiny datasets are a warning, not a triumph.** Kidney (400 rows) and
  Thyroid (215 rows) reach ROC-AUC 1.00 on test splits of 80 and 43 records; that says more
  about the datasets than about generalisation. None of the models is externally validated.
Full per-model detail — algorithm selection, cross-validation, calibration comparison,
hyperparameter search, confusion matrix, permutation importance — is in
[`MODEL_CARD.md`](MODEL_CARD.md) and each model's `model_metadata.json`.

Diabetes test-set performance (stratified 60/20/20 split, `random_state=42`):

| metric | value |
|---|---|
| ROC-AUC | 0.8255 |
| PR-AUC | 0.7229 |
| Recall / sensitivity @ 0.5 | 0.5926 |
| Specificity @ 0.5 | 0.83 |
| Calibration | isotonic |

A tuned candidate (`diabetes-v2`, cross-validated hyperparameter search + an added
HistGradientBoosting candidate) was evaluated against this model and scored slightly
*worse* on the held-out test set — it was **not promoted**. That comparison is kept on
disk (`ml/diabetes/model_metadata.v2-experimental.json`) as a documented, honest result
rather than discarded.

---

## Architecture

### Project structure

```
data/          real datasets (git-ignored) + registry + inspection reports
ml/common/     reusable pipeline framework — preprocessing, CV, calibration,
               hyperparameter search, threshold selection, evaluation, model registry
ml/<disease>/  one pipeline per disease: load, spec, train
models/        trained joblib artifacts (committed — see "Model artifacts" below)
               + model_registry.json
backend/       FastAPI app — health, diseases, models, datasets, schema,
               predict, predict/all, evaluation, validation, assessments/recent
frontend/      React + TypeScript — dashboard, assessment, results, conditions,
               models, datasets, validation & model comparison, about
```

```
 Browser  ──►  React/Vite frontend  ──►  FastAPI backend  ──►  trained sklearn
(Vercel)                                    (Render)            pipeline (.joblib)
                                                │
                                                ▼
                                     SQLite (dev) / Postgres (optional, prod)
                                     — best-effort assessment history only;
                                       predictions work with or without it
```

### Frontend ↔ backend flow

1. The browser loads the static React build from Vercel.
2. Every API call goes through one client, `frontend/src/api/client.ts`. Its base URL
   is `VITE_API_BASE` (baked in at build time); when unset — i.e. in local development —
   it falls back to `/api`, which the Vite dev server proxies to `localhost:8000`.
3. FastAPI validates the request against the disease's feature schema, runs the
   trained scikit-learn pipeline loaded from `models/*.joblib`, and returns the risk
   score, level, metrics and contributing factors. Evaluation endpoints read the stored
   training-run metadata verbatim — nothing is recomputed or estimated at request time.
4. The backend's CORS allow-list (`EARLYDX_CORS_ORIGINS`) permits only the deployed
   frontend origin(s) in production.

Full detail: [`ARCHITECTURE.md`](ARCHITECTURE.md).

### Research pipeline

```
Dataset → Preprocessing → Feature schema → Candidate training (GridSearchCV)
        → Validation-split comparison → Model selection → Calibration → Thresholds
        → ONE held-out test evaluation → Serialise + register → Deploy → Risk prediction
```

Every disease follows the identical, leakage-audited sequence in
`ml/common/framework.py`:

```
load → validate → stratified 60/20/20 split
     → hyperparameter search (GridSearchCV + StratifiedKFold, train split only)
     → candidate comparison on validation (LogisticRegression baseline vs.
       RandomForest / GradientBoosting / HistGradientBoosting)
     → cross-validation of the selected model
     → calibration choice (none / sigmoid / isotonic) by validation Brier score
     → decision thresholds derived from validation sensitivity/specificity targets
     → ONE held-out test evaluation
     → permutation importance + a bounded held-out validation sample
     → serialise artifact + feature schema + metadata + registry entry
```

The test split is touched exactly once, for final evaluation — never for tuning,
calibration, or threshold selection.

### Model artifacts

The `.joblib` files backing every **active** registry entry are committed directly to
this repo (`models/*.joblib`, ~53 MB total, no Git LFS needed — every file is well
under normal size limits). They are not regenerated at deploy time: production loads
them straight from the checked-out repo. Experimental/candidate artifacts that were not
promoted (e.g. `diabetes-v2`) are intentionally left out of git via `.gitignore`.

---

## Deployment

The production app uses the standard static-frontend / API-backend split:

```
Frontend  →  Vercel   https://early-dx.vercel.app
             (frontend/vercel.json — Vite build, SPA rewrite for client-side routing)
Backend   →  Render   https://earlydx-api.onrender.com
             (render.yaml — Blueprint: build/start commands, /health check, env vars)
```

**Vercel** — import the repo, set *Root Directory* to `frontend` (framework: Vite), and
add the `VITE_API_BASE` environment variable. `frontend/vercel.json` supplies the build
command, output directory and the SPA rewrite.

**Render** — *New → Blueprint* on the repo; `render.yaml` defines the `earlydx-api`
web service (`pip install -r backend/requirements.txt`, `uvicorn backend.app.main:app`).
The model `.joblib` files are loaded from the checked-out repo at start-up, so no
training happens at deploy time.

### Environment variables

Everything is environment-variable driven — no hardcoded `localhost` in production code.
No secrets are required to run the app.

| Where | Variable | Required | Purpose |
|---|---|---|---|
| Vercel (build-time) | `VITE_API_BASE` | Yes, in production | Full backend origin, e.g. `https://earlydx-api.onrender.com` — no trailing slash, no `/api`. Vite bakes it into the bundle, so changing it needs a redeploy. Unset locally → `/api` dev proxy. |
| Render | `EARLYDX_CORS_ORIGINS` | Yes | Comma-separated list of allowed frontend origins (exact, no trailing slash). Set in `render.yaml`; never `*` in production. |
| Render | `EARLYDX_ENV`, `EARLYDX_LOG_LEVEL` | Set by `render.yaml` | `production` / `INFO`. |
| Render | `EARLYDX_DATABASE_URL` | No | `postgresql+psycopg://…` URL. If unset, the API falls back to a local SQLite file and still serves predictions — it just won't persist assessment history across restarts. |

See `render.yaml` and `.env.example` for the full local variable list.

---

## Local setup

Prerequisites: Python 3.11, Node 18+, (optional) Docker for local PostgreSQL.

```bash
git clone https://github.com/Anjali56-creator/EarlyDX.git
cd EarlyDX
```

```bash
# 1. Python environment
python -m venv .venv
.venv/Scripts/activate            # Windows;  source .venv/bin/activate on Unix
pip install -r requirements.txt -r backend/requirements.txt

# 2. Frontend
cd frontend && npm install && cd ..

# 3. Environment file
cp .env.example .env
```

---

## Run locally

```bash
# optional: PostgreSQL (otherwise the API falls back to a local SQLite file)
docker compose up -d

# backend  (http://localhost:8000, docs at /docs)
uvicorn backend.app.main:app --reload

# frontend (http://localhost:5173, proxies /api -> :8000)
cd frontend && npm run dev
```

Locally, leave `VITE_API_BASE` unset: the frontend calls `/api/...` and the Vite dev
server (`frontend/vite.config.ts`) forwards it to the backend on port 8000. The 10
trained model artifacts are committed, so predictions work immediately without
retraining.

---

## Reproduce a model

```bash
python scripts/download_diabetes.py         # fetch the real dataset into data/raw/
python scripts/inspect_dataset.py diabetes   # measured inspection report
python scripts/train_disease.py diabetes     # train, evaluate, calibrate, serialise
```

Any of the 10 trained diseases can be substituted for `diabetes`. This writes the
`models/<disease>-v1.joblib` artifact, `ml/<disease>/feature_schema.json`,
`ml/<disease>/model_metadata.json`, updates `models/model_registry.json`, and
regenerates that disease's section of `MODEL_CARD.md`.

---

## API

Interactive Swagger docs for the live API: https://earlydx-api.onrender.com/docs
(locally: http://localhost:8000/docs). All paths are served at the API root — there is
no `/api` prefix on the backend; that prefix exists only in the local Vite proxy.

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | liveness, loaded-model count, DB status |
| GET | `/diseases` | the 12 conditions + model availability |
| GET | `/models` | model registry |
| GET | `/datasets` | dataset registry, including documented limitations |
| GET | `/schema/{disease}` | required feature schema for a form |
| POST | `/predict/{disease}` | risk score + risk level + contributing factors |
| POST | `/predict/all` | runs only the models whose full feature schema is satisfied; others returned in `skipped` |
| GET | `/evaluation` | comparative summary for every loaded model: deployed vs. candidate algorithms, test metrics, calibration, CV ROC-AUC |
| GET | `/evaluation/{disease}` | full stored evaluation: per-candidate validation metrics, held-out test metrics with confusion matrix and calibration bins, CV summary, thresholds, permutation importance |
| GET | `/validation/{disease}` | a bounded sample of held-out test-set rows with real labels vs. the model's own predictions (stored for Diabetes) |
| GET | `/assessments/recent` | most recent persisted assessments — condition, model, score, level, time; inputs are never stored |

`/predict/*` never fills a missing medical value with a default.

---

## Testing

```bash
pytest                       # ML + backend — 40 tests (backend/tests, ml/tests)
cd frontend && npm test      # frontend — assessment→results flow + validation page (vitest)
```

---

## Documentation

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — system design and the per-disease pipeline contract
- [`DATASETS.md`](DATASETS.md) — every candidate dataset, source, licence, limitations
- [`MODEL_CARD.md`](MODEL_CARD.md) — every trained model, generated from evaluation runs
- [`LIMITATIONS.md`](LIMITATIONS.md) — dataset bias, class imbalance, false negatives, non-clinical status

---

## Disclaimer

EarlyDX is a student research / portfolio prototype built to explore transparent,
traceable machine-learning risk screening. It is **not a medical device**, has **not**
been clinically validated, and must not be used to diagnose, treat, or make decisions
about any person's health. Its models are trained on small, public, de-identified
datasets with documented biases and limitations (see [`LIMITATIONS.md`](LIMITATIONS.md)
and [`DATASETS.md`](DATASETS.md)); some perform weakly, and one (Parkinson's) is kept
only for transparency. Always consult a qualified healthcare professional for medical
concerns.
