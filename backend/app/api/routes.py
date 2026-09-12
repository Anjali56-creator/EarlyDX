"""All EarlyDX HTTP endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path as PathParam
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app import database as db_mod
from backend.app.database import get_session, record_assessment, recent_assessments
from backend.app.schemas import (
    PredictAllResponse,
    PredictionResponse,
    REQUEST_MODELS,
)
from backend.app.services import prediction as prediction_service
from backend.app.services.catalog import DISEASES, UNAVAILABLE_REASONS, display_name
from backend.app.services.model_loader import STORE
from ml.common.utilities import read_json

router = APIRouter()
_settings = get_settings()


def _dataset_registry() -> dict[str, Any]:
    path = _settings.data_path / "dataset_registry.json"
    return read_json(path) if path.exists() else {}


def _model_registry() -> dict[str, Any]:
    path = _settings.models_path / "model_registry.json"
    return read_json(path) if path.exists() else {"models": {}}


def _schema_for(disease: str) -> dict[str, Any] | None:
    path = Path(_settings.models_path).parent / "ml" / disease / "feature_schema.json"
    return read_json(path) if path.exists() else None


# ── metadata endpoints ─────────────────────────────────────────────────────
@router.get("/health", tags=["meta"])
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "models_loaded": STORE.count(),
        "model_load_errors": STORE.errors(),
        "database": {"available": db_mod.DB_AVAILABLE, "error": db_mod.DB_ERROR},
        "disclaimer": "EarlyDX is a research prototype and not a medical device.",
    }


@router.get("/diseases", tags=["meta"])
def list_diseases() -> dict[str, Any]:
    available = set(STORE.available())
    out = []
    for key, info in DISEASES.items():
        schema = _schema_for(key)
        out.append(
            {
                "key": key,
                "display_name": info["display_name"],
                "group": info["group"],
                "model_available": key in available,
                "unavailable_reason": UNAVAILABLE_REASONS.get(key)
                if key not in available
                else None,
                "required_features": schema["required_features"] if schema else [],
            }
        )
    return {"count": len(out), "diseases": out}


@router.get("/models", tags=["meta"])
def list_models() -> dict[str, Any]:
    registry = _model_registry()
    loaded = set(STORE.available())
    models = []
    for entry in registry.get("models", {}).values():
        models.append({**entry, "loaded": entry["disease"] in loaded})
    return {"count": len(models), "models": models}


@router.get("/datasets", tags=["meta"])
def list_datasets() -> dict[str, Any]:
    reg = _dataset_registry()
    datasets = [v for k, v in reg.items() if not k.startswith("_")]
    return {"count": len(datasets), "datasets": datasets, "meta": reg.get("_meta", {})}


@router.get("/schema/{disease}", tags=["meta"])
def get_schema(disease: str = PathParam(...)) -> dict[str, Any]:
    schema = _schema_for(disease)
    if schema is None:
        raise HTTPException(
            status_code=404,
            detail=f"no feature schema for '{disease}' "
            f"({UNAVAILABLE_REASONS.get(disease, 'unknown disease')})",
        )
    return schema


@router.get("/validation/{disease}", tags=["research"])
def validation_samples(disease: str = PathParam(...)) -> dict[str, Any]:
    """A bounded sample of held-out TEST rows (never seen in training/tuning/
    calibration/threshold selection) with their real label and the model's own
    prediction, for research validation display. Not diagnosis; not a live
    prediction — these are stored at training time from the test split."""
    bundle = STORE.get(disease)
    if bundle is None:
        raise HTTPException(
            status_code=404,
            detail=f"no model available for '{disease}' "
            f"({UNAVAILABLE_REASONS.get(disease, 'not implemented in this version')})",
        )
    vs = bundle.metadata.get("validation_samples")
    if not vs or not vs.get("samples"):
        raise HTTPException(
            status_code=404,
            detail=f"no stored validation samples for '{disease}' "
            "(this model was trained before validation samples were captured)",
        )
    return {
        "disease": display_name(disease),
        "disease_key": disease,
        "model_version": bundle.model_id,
        "note": vs["note"],
        "samples": vs["samples"],
    }


# ── evaluation endpoints ──────────────────────────────────────────────────
# Every number below is read verbatim from ml/<disease>/model_metadata.json,
# which scripts/train_disease.py writes at the end of a real training run.
# Nothing is computed or estimated at request time.

_CANDIDATE_METRIC_KEYS = (
    "accuracy",
    "precision",
    "recall_sensitivity",
    "specificity",
    "f1",
    "roc_auc",
    "pr_auc",
)


def _evaluation_for(disease: str) -> dict[str, Any] | None:
    bundle = STORE.get(disease)
    if bundle is None:
        return None
    meta = bundle.metadata
    test = meta.get("test_metrics", {}) or {}
    candidates = meta.get("validation_metrics_by_candidate", {}) or {}
    cv = meta.get("cross_validation", {}) or {}
    calib = meta.get("calibration", {}) or {}
    split = meta.get("split", {}) or {}

    # Best validation candidate by ROC-AUC (threshold-independent). The
    # *selected* model may differ: the training rule prefers the interpretable
    # LogisticRegression baseline unless a candidate beats it by >= 0.01.
    best_by_auc = None
    if candidates:
        best_by_auc = max(
            candidates.items(), key=lambda kv: kv[1].get("roc_auc") or -1.0
        )[0]

    return {
        "disease": display_name(disease),
        "disease_key": disease,
        "model_version": bundle.model_id,
        "selected_algorithm": meta.get("algorithm"),
        "baseline_algorithm": meta.get("baseline_algorithm"),
        "selection_rule": meta.get("selection_rule"),
        "best_validation_candidate_by_roc_auc": best_by_auc,
        "training_date": meta.get("training_date"),
        "dataset": meta.get("dataset", {}),
        "split": split,
        "target_positive_rate": meta.get("target_positive_rate"),
        # one row per algorithm compared on the VALIDATION split at 0.5
        "candidates": {
            name: {
                "n": m.get("n"),
                "threshold": m.get("threshold"),
                **{k: m.get(k) for k in _CANDIDATE_METRIC_KEYS},
                "confusion_matrix": m.get("confusion_matrix"),
            }
            for name, m in candidates.items()
        },
        # the selected model's ONE held-out TEST evaluation
        "test_metrics": {
            "n": test.get("n"),
            "threshold": test.get("threshold"),
            "class_distribution": test.get("class_distribution"),
            **{k: test.get(k) for k in _CANDIDATE_METRIC_KEYS},
            "confusion_matrix": test.get("confusion_matrix"),
            "brier": (test.get("calibration") or {}).get("brier"),
            "calibration_bins": (test.get("calibration") or {}).get("bins"),
        },
        "test_metrics_at_screening_threshold": meta.get(
            "test_metrics_at_low_threshold_screening"
        ),
        "cross_validation": {
            "folds": cv.get("folds"),
            "summary": cv.get("summary"),
        },
        "calibration": {
            "method": calib.get("method"),
            "validation_brier": calib.get("validation_brier"),
        },
        "risk_thresholds": meta.get("risk_thresholds"),
        "permutation_importance": meta.get("permutation_importance"),
        # honest flags for what the training run did NOT persist
        "unavailable": {
            "roc_curve_points": "not stored by the training run; only the scalar ROC-AUC is recorded"
        },
        "limitations": meta.get("limitations"),
        "ethical_note": meta.get("ethical_note"),
        "has_validation_samples": bool(
            (meta.get("validation_samples") or {}).get("samples")
        ),
    }


@router.get("/evaluation", tags=["research"])
def evaluation_summary() -> dict[str, Any]:
    """Compact comparative summary for every loaded model: selected algorithm
    vs. the candidates it was compared against, with held-out test metrics."""
    out = []
    for key in STORE.available():
        ev = _evaluation_for(key)
        if ev is None:
            continue
        out.append(
            {
                "disease": ev["disease"],
                "disease_key": key,
                "model_version": ev["model_version"],
                "selected_algorithm": ev["selected_algorithm"],
                "baseline_algorithm": ev["baseline_algorithm"],
                "best_validation_candidate_by_roc_auc": ev[
                    "best_validation_candidate_by_roc_auc"
                ],
                "candidates_compared": sorted(ev["candidates"].keys()),
                "test_metrics": {
                    k: ev["test_metrics"].get(k)
                    for k in ("n", *_CANDIDATE_METRIC_KEYS)
                },
                "calibration_method": ev["calibration"]["method"],
                "cv_roc_auc": (ev["cross_validation"]["summary"] or {}).get("roc_auc"),
            }
        )
    return {"count": len(out), "models": out}


@router.get("/evaluation/{disease}", tags=["research"])
def evaluation_detail(disease: str = PathParam(...)) -> dict[str, Any]:
    """Full stored evaluation for one model: candidate comparison on the
    validation split, the single held-out test evaluation (incl. confusion
    matrix and calibration bins), cross-validation summary, thresholds and
    permutation importance."""
    ev = _evaluation_for(disease)
    if ev is None:
        raise HTTPException(
            status_code=404,
            detail=f"no model available for '{disease}' "
            f"({UNAVAILABLE_REASONS.get(disease, 'not implemented in this version')})",
        )
    return ev


@router.get("/assessments/recent", tags=["meta"])
def assessments_recent(
    limit: int = 10, db: Session = Depends(get_session)
) -> dict[str, Any]:
    """Most recent persisted assessments (disease, model, score, level, time).
    No input features are ever stored, so nothing personal is returned. Empty
    when no database is configured or it is unavailable."""
    limit = max(1, min(limit, 50))
    return recent_assessments(db, limit=limit)


# ── prediction endpoints ──────────────────────────────────────────────────
# NOTE: /predict/all is declared before /predict/{disease} so the literal path
# is matched first and never captured as disease="all".
@router.post("/predict/all", response_model=PredictAllResponse, tags=["predict"])
def predict_all(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_session),
) -> PredictAllResponse:
    raw = payload.get("features", payload)
    features: dict[str, Any] = {}
    for k, v in raw.items():
        if k in {"features", "client_context"} or v is None:
            continue
        # numeric strings become floats; genuine categoricals pass through as-is
        if isinstance(v, str):
            try:
                features[k] = float(v)
            except ValueError:
                features[k] = v
        else:
            features[k] = v

    if STORE.count() == 0:
        raise HTTPException(status_code=409, detail="no models are currently loaded")

    outcome = prediction_service.predict_all(features)
    client_context = str(payload.get("client_context", "app"))
    session_id = record_assessment(
        db, mode="all", client_context=client_context, results=outcome["results"]
    )
    outcome["session_id"] = session_id
    return PredictAllResponse(**outcome)


@router.post("/predict/{disease}", response_model=PredictionResponse, tags=["predict"])
def predict_disease(
    disease: str,
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_session),
) -> PredictionResponse:
    if disease not in DISEASES:
        raise HTTPException(status_code=404, detail=f"unknown disease '{disease}'")
    if disease not in REQUEST_MODELS or STORE.get(disease) is None:
        raise HTTPException(
            status_code=409,
            detail=f"no model available for '{display_name(disease)}' "
            f"({UNAVAILABLE_REASONS.get(disease, 'not implemented in this version')})",
        )

    model_cls = REQUEST_MODELS[disease]
    try:
        features = model_cls(**payload).model_dump()
    except Exception as exc:  # pydantic ValidationError
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        result = prediction_service.predict_one(disease, features)
    except prediction_service.MissingFeatures as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except prediction_service.PredictionError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    client_context = str(payload.get("client_context", "app"))
    session_id = record_assessment(
        db, mode="single", client_context=client_context, results=[result]
    )
    result["session_id"] = session_id
    return PredictionResponse(**result)
