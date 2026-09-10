"""Static catalogue of the 12 in-scope conditions.

Model availability is decided at runtime by cross-referencing
``models/model_registry.json`` (see model_loader). Conditions with no model
carry an explicit reason.
"""
from __future__ import annotations

DISEASES: dict[str, dict[str, str]] = {
    "diabetes": {"display_name": "Diabetes", "group": "metabolic"},
    "heart_disease": {"display_name": "Heart Disease", "group": "cardiovascular"},
    "kidney_disease": {"display_name": "Chronic Kidney Disease", "group": "renal"},
    "liver_disease": {"display_name": "Liver Disease", "group": "hepatic"},
    "breast_cancer": {"display_name": "Breast Cancer", "group": "oncology"},
    "parkinsons": {"display_name": "Parkinson's Disease", "group": "neurological"},
    "heart_failure": {"display_name": "Heart Failure", "group": "cardiovascular"},
    "stroke": {"display_name": "Stroke", "group": "cardiovascular"},
    "hypertension": {"display_name": "Hypertension", "group": "cardiovascular"},
    "thyroid": {"display_name": "Thyroid Disease", "group": "endocrine"},
    "obesity": {"display_name": "Obesity", "group": "metabolic"},
    "maternal_health": {"display_name": "Maternal Health Risk", "group": "obstetric"},
}

# Reasons a condition has no model in this version.
UNAVAILABLE_REASONS: dict[str, str] = {
    "stroke": (
        "No public dataset with acceptable provenance and licensing was "
        "identified without personal-account credentials."
    ),
    "hypertension": (
        "No rigorously defined public target was established for V1; shipping a "
        "model here would require an unreliable target or synthetic data."
    ),
}


def display_name(disease: str) -> str:
    return DISEASES.get(disease, {}).get("display_name", disease)
