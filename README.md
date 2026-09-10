## EarlyDX — Multi-Disease Early Risk Detection Platform

EarlyDX trains one independent machine-learning pipeline per disease on real,
publicly available, de-identified datasets, and serves per-disease **risk
assessments** through a FastAPI backend and a React + TypeScript frontend.

> **Not a medical device. Not clinically validated. Not a diagnosis.**
> EarlyDX is a portfolio / research prototype. See `LIMITATIONS.md`.

---

### Status

| Area | State |
|------|-------|
| Architecture & dataset research | complete — see `ARCHITECTURE.md`, `DATASETS.md` |
| Diabetes pipeline (Phase 3) | **complete, end-to-end, verified** |
| Reusable ML framework (Phase 4) | in progress (`ml/common`) |
| Remaining 9 disease models | not yet implemented |
| Stroke, Hypertension | shipped as `model: unavailable` (see `DATASETS.md`) |

The diabetes model is trained on the real **Pima Indians Diabetes Database**
(768 records). All reported metrics come from an executed evaluation run; none
are hand-written.

Diabetes test-set performance (stratified 60/20/20, `random_state=42`):

| metric | value |
|--------|-------|
| ROC-AUC | 0.83 |
| PR-AUC | 0.72 |
| recall / sensitivity @ 0.5 | 0.59 |
| recall @ screening cut (0.29) | 0.83 |
| specificity @ 0.5 | 0.83 |

Exact numbers, confusion matrix, cross-validation and calibration details are in
`MODEL_CARD.md` and `ml/diabetes/model_metadata.json`.

---

### Architecture

```
data/        real datasets (git-ignored) + registry + inspection reports
ml/common/   reusable pipeline infrastructure (preprocessing, evaluation, registry)
ml/<disease>/ one pipeline per disease: load, pipeline, train, evaluate
models/      promoted joblib artifacts (git-ignored) + model_registry.json
backend/     FastAPI app (health, diseases, models, datasets, schema, predict)
frontend/    React + TypeScript (dashboard, assessment, results, ...)
```

Full detail: `ARCHITECTURE.md`.

---

### Installation

Prerequisites: Python 3.11, Node 18+, (optional) Docker for PostgreSQL.

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

### Reproduce the diabetes model

```bash
python scripts/download_diabetes.py     # fetch the real dataset into data/raw/
python scripts/inspect_dataset.py diabetes   # measured inspection report
python -m ml.diabetes.train             # train, evaluate, calibrate, serialise
```

This writes `models/diabetes-v1.joblib`, `ml/diabetes/feature_schema.json`,
`ml/diabetes/model_metadata.json`, updates `models/model_registry.json` and
regenerates the `diabetes-v1` section of `MODEL_CARD.md`.

---

### Run locally

```bash
# optional: PostgreSQL (otherwise the API falls back to a local SQLite file)
docker compose up -d

# backend  (http://localhost:8000, docs at /docs)
uvicorn backend.app.main:app --reload

# frontend (http://localhost:5173, proxies /api -> :8000)
cd frontend && npm run dev
```

---

### API

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | liveness, loaded-model count, DB status |
| GET | `/diseases` | the 12 conditions + model availability |
| GET | `/models` | model registry |
| GET | `/datasets` | dataset registry |
| GET | `/schema/{disease}` | required feature schema for a form |
| POST | `/predict/{disease}` | risk score + risk level + contributing factors |
| POST | `/predict/all` | runs only the models whose full feature schema is satisfied; others returned in `skipped` |

`/predict/all` never fills missing medical values with defaults.

---

### Testing

```bash
pytest                       # ML + backend  (skips model tests if not trained)
cd frontend && npm test      # critical UI flow (vitest)
```

---

### Supported diseases (target scope)

Diabetes, Heart Disease, Chronic Kidney Disease, Liver Disease, Breast Cancer,
Parkinson's Disease, Heart Failure, Stroke, Hypertension, Thyroid Disease,
Obesity, Maternal Health Risk.

Only Diabetes has a trained model in this version.

---

### Documentation

- `ARCHITECTURE.md` — system design and the per-disease pipeline contract
- `DATASETS.md` — every candidate dataset, source, licence, limitations
- `MODEL_CARD.md` — every trained model, generated from evaluation runs
- `LIMITATIONS.md` — dataset bias, class imbalance, false negatives, non-clinical status
