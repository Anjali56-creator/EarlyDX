"""Explanation of a single prediction.

Method (documented, model-supported):
  * feature ranking comes from the model's own permutation importance, computed
    on the held-out test set during training and stored in model_metadata.json;
  * direction ("above typical" / "below typical") compares the submitted value
    with the training-set median stored alongside the model.

This does not claim causation. It describes which inputs the trained model relies
on and where this input sits relative to the training population.
"""
from __future__ import annotations

from typing import Any

from backend.app.services.model_loader import ModelBundle

_IMPACT_BANDS = [(0.05, "high"), (0.01, "medium")]


def _impact(importance: float) -> str:
    for cutoff, label in _IMPACT_BANDS:
        if importance >= cutoff:
            return label
    return "low"


def explain(bundle: ModelBundle, features: dict[str, float], top_k: int = 4) -> list[dict[str, Any]]:
    importance = bundle.metadata.get("permutation_importance", [])
    reference = bundle.metadata.get("feature_reference", {})
    out: list[dict[str, Any]] = []

    for item in importance[:top_k]:
        name = item["feature"]
        if name not in features:
            continue
        value = features[name]
        ref = reference.get(name, {})
        median = ref.get("median")
        direction = "unknown"
        if median is not None:
            if value > ref.get("p75", median):
                direction = "above typical"
            elif value < ref.get("p25", median):
                direction = "below typical"
            else:
                direction = "near typical"
        out.append(
            {
                "feature": name,
                "value": value,
                "population_median": median,
                "direction": direction,
                "impact": _impact(float(item.get("importance", 0.0))),
                "importance": round(float(item.get("importance", 0.0)), 4),
            }
        )
    return out
