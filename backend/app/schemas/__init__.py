"""Pydantic request/response schemas."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from backend.app.schemas.diabetes import DiabetesFeatures

# disease key -> strict request model
REQUEST_MODELS: dict[str, type[BaseModel]] = {
    "diabetes": DiabetesFeatures,
}


class ImportantFeature(BaseModel):
    feature: str
    value: float
    population_median: float | None = None
    direction: str
    impact: str
    importance: float


class PredictionResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    disease: str
    disease_key: str
    risk_score: float
    risk_level: str
    calibrated: bool
    model_version: str
    model_algorithm: str | None = None
    model_performance: dict[str, Any]
    important_features: list[ImportantFeature]
    threshold_policy: str | None = None
    disclaimer: str
    session_id: int | None = None


class SkippedDisease(BaseModel):
    disease: str
    disease_key: str
    reason: str
    missing_features: list[str]


class PredictAllResponse(BaseModel):
    results: list[PredictionResponse]
    skipped: list[SkippedDisease]
    disclaimer: str
    session_id: int | None = None


class PredictAllRequest(BaseModel):
    """Free-form feature bag for /predict/all.

    Only keys matching a model's schema are used; unknown keys are ignored;
    missing keys cause that disease to be skipped (never defaulted).
    """

    model_config = {"extra": "allow"}
    features: dict[str, float] = Field(default_factory=dict)
