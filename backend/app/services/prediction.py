"""Turn a validated feature payload into a risk result using a real model."""
from __future__ import annotations

from typing import Any

import pandas as pd

from backend.app.explainability import explain
from backend.app.services.catalog import display_name
from backend.app.services.model_loader import STORE, ModelBundle
from ml.common.config import DISCLAIMER


class PredictionError(RuntimeError):
    pass


class MissingFeatures(PredictionError):
    def __init__(self, disease: str, missing: list[str]):
        self.disease = disease
        self.missing = missing
        super().__init__(f"{disease}: missing required features {missing}")


def _score(bundle: ModelBundle, features: dict[str, float]) -> float:
    row = pd.DataFrame([[features[f] for f in bundle.features]], columns=bundle.features)
    proba = bundle.estimator.predict_proba(row)[0, 1]
    return float(proba)


def predict_one(disease: str, features: dict[str, float]) -> dict[str, Any]:
    bundle = STORE.get(disease)
    if bundle is None:
        raise PredictionError(f"no model available for '{disease}'")

    missing = [f for f in bundle.features if features.get(f) is None]
    if missing:
        raise MissingFeatures(disease, missing)

    score = _score(bundle, features)
    level = bundle.risk_level(score)
    score_field = "risk_score" if not bundle.calibrated else "calibrated_risk_score"

    metrics = bundle.registry_entry.get("metrics", {})
    # standard classification metrics from the ONE held-out test evaluation
    # recorded in model_metadata.json (same run the registry summarises)
    test = bundle.metadata.get("test_metrics", {}) or {}
    return {
        "disease": display_name(disease),
        "disease_key": disease,
        score_field: round(score, 4),
        "risk_score": round(score, 4),
        "risk_level": level,
        "calibrated": bundle.calibrated,
        "model_version": bundle.model_id,
        "model_algorithm": bundle.registry_entry.get("algorithm"),
        "model_performance": {
            "test_roc_auc": metrics.get("test_roc_auc"),
            "test_pr_auc": metrics.get("test_pr_auc"),
            "test_recall_sensitivity": metrics.get("test_recall_sensitivity"),
            "test_specificity": metrics.get("test_specificity"),
            "test_accuracy": metrics.get("test_accuracy", test.get("accuracy")),
            "test_precision": test.get("precision"),
            "test_f1": test.get("f1"),
            "cv_roc_auc_mean": metrics.get("cv_roc_auc_mean"),
        },
        "important_features": explain(bundle, features),
        "threshold_policy": bundle.metadata.get("risk_thresholds", {}).get("policy"),
        # the actual validation-derived cut points behind risk_level, so the UI
        # can draw a real zone bar instead of inventing round numbers.
        "risk_thresholds": {
            "low_cut": bundle.thresholds.get("low_cut"),
            "high_cut": bundle.thresholds.get("high_cut"),
        },
        "decision_threshold": 0.5,
        "predicted_class": int(score >= 0.5),
        "disclaimer": DISCLAIMER,
    }


def predict_all(features: dict[str, float]) -> dict[str, Any]:
    """Run only the models whose full feature schema is satisfied by ``features``."""
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for disease in STORE.available():
        bundle = STORE.get(disease)
        assert bundle is not None
        missing = [f for f in bundle.features if features.get(f) is None]
        if missing:
            skipped.append(
                {
                    "disease": display_name(disease),
                    "disease_key": disease,
                    "reason": "missing required features",
                    "missing_features": missing,
                }
            )
            continue
        results.append(predict_one(disease, features))

    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    results.sort(key=lambda r: (order.get(r["risk_level"], 3), -r["risk_score"]))
    return {"results": results, "skipped": skipped, "disclaimer": DISCLAIMER}
