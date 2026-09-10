"""All EarlyDX HTTP endpoints."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Path as PathParam
from sqlalchemy.orm import Session

from backend.app.config import get_settings
from backend.app import database as db_mod
from backend.app.database import get_session, record_assessment
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


# ── prediction endpoints ──────────────────────────────────────────────────
# NOTE: /predict/all is declared before /predict/{disease} so the literal path
# is matched first and never captured as disease="all".
@router.post("/predict/all", response_model=PredictAllResponse, tags=["predict"])
def predict_all(
    payload: dict[str, Any] = Body(...),
    db: Session = Depends(get_session),
) -> PredictAllResponse:
    raw = payload.get("features", payload)
    features: dict[str, float] = {}
    for k, v in raw.items():
        if k in {"features", "client_context"}:
            continue
        try:
            features[k] = float(v)
        except (TypeError, ValueError):
            continue

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
