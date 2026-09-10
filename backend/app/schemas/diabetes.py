"""Strict request schema for the diabetes model.

Field bounds come from ``ml.diabetes.load.FEATURE_RANGES`` so the API and the
training code share one definition of a plausible value.
"""
from __future__ import annotations

from pydantic import BaseModel, Field, create_model

from ml.diabetes.load import FEATURE_RANGES, FEATURE_UNITS, FEATURES

_fields: dict[str, tuple] = {}
for _name in FEATURES:
    _lo, _hi = FEATURE_RANGES[_name]
    _fields[_name] = (
        float,
        Field(..., ge=_lo, le=_hi, description=FEATURE_UNITS[_name]),
    )

DiabetesFeatures: type[BaseModel] = create_model(  # type: ignore[call-overload]
    "DiabetesFeatures",
    __config__=type("Cfg", (), {"extra": "forbid"}),
    **_fields,
)
