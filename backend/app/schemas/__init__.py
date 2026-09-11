"""Pydantic request/response schemas.

Request models are built dynamically from each disease's
``ml/<disease>/feature_schema.json`` so the API, the training code and the
frontend all share one definition of a valid input.
"""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, create_model, field_validator

from backend.app.config import REPO_ROOT
from ml.common.utilities import read_json

_SCHEMA_DIR = REPO_ROOT / "ml"


def _zero_or_range_validator(name: str, lo: float | None, hi: float | None):
    """0 is an explicit "not measured" sentinel for this feature: the trained
    pipeline's ZeroToNaN step converts an exact 0 to NaN and median-imputes it
    (see ml/common/preprocessing.py), so 0 must be accepted even though it
    falls outside the feature's otherwise-plausible range. Any other value
    still has to fall inside [lo, hi]."""

    def _check(cls: type, v: float) -> float:  # noqa: ARG001
        if v == 0:
            return v
        if lo is not None and v < lo:
            raise ValueError(f"{name} must be 0 (treated as missing) or >= {lo}")
        if hi is not None and v > hi:
            raise ValueError(f"{name} must be 0 (treated as missing) or <= {hi}")
        return v

    return field_validator(name)(classmethod(_check))


def _build_request_model(disease: str) -> type[BaseModel] | None:
    path = _SCHEMA_DIR / disease / "feature_schema.json"
    if not path.exists():
        return None
    schema = read_json(path)
    fields: dict[str, tuple] = {}
    validators: dict[str, Any] = {}
    for name, detail in schema["feature_details"].items():
        if detail.get("type") == "categorical":
            options = [str(o) for o in detail.get("options", [])]
            if options:
                from typing import Literal  # noqa: PLC0415

                typ = Literal[tuple(options)]  # type: ignore[valid-type]
            else:
                typ = str
            fields[name] = (typ, Field(..., description=detail.get("unit", "")))
        else:
            lo = detail.get("min")
            hi = detail.get("max")
            if detail.get("zero_is_missing"):
                fields[name] = (float, Field(..., description=detail.get("unit", "")))
                validators[f"validate_{name}"] = _zero_or_range_validator(name, lo, hi)
            else:
                fields[name] = (
                    float,
                    Field(..., ge=lo, le=hi, description=detail.get("unit", "")),
                )
    return create_model(  # type: ignore[call-overload]
        f"{disease.title().replace('_', '')}Features",
        __config__=type("Cfg", (), {"extra": "forbid"}),
        __validators__=validators,
        **fields,
    )


def load_request_models() -> dict[str, type[BaseModel]]:
    models: dict[str, type[BaseModel]] = {}
    for child in sorted(_SCHEMA_DIR.iterdir()):
        if child.is_dir() and (child / "feature_schema.json").exists():
            m = _build_request_model(child.name)
            if m is not None:
                models[child.name] = m
    return models


REQUEST_MODELS: dict[str, type[BaseModel]] = load_request_models()


class ImportantFeature(BaseModel):
    feature: str
    value: Any
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
    risk_thresholds: dict[str, float | None] | None = None
    decision_threshold: float | None = None
    predicted_class: int | None = None
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
