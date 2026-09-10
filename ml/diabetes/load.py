"""Load and validate the real Pima Indians Diabetes dataset."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.config import RAW_DIR

RAW_FILE: Path = RAW_DIR / "diabetes" / "pima_indians_diabetes.csv"

FEATURES: list[str] = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age",
]
TARGET: str = "Outcome"
ZERO_AS_MISSING: list[str] = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

# Plausible input ranges, used for the API feature schema and validation.
FEATURE_RANGES: dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 25),
    "Glucose": (30, 400),
    "BloodPressure": (20, 200),
    "SkinThickness": (5, 110),
    "Insulin": (5, 900),
    "BMI": (10, 80),
    "DiabetesPedigreeFunction": (0.0, 3.0),
    "Age": (18, 120),
}
FEATURE_UNITS: dict[str, str] = {
    "Pregnancies": "count",
    "Glucose": "mg/dL (2-hour OGTT plasma glucose)",
    "BloodPressure": "mm Hg (diastolic)",
    "SkinThickness": "mm (triceps skinfold)",
    "Insulin": "mu U/mL (2-hour serum insulin)",
    "BMI": "kg/m^2",
    "DiabetesPedigreeFunction": "unitless pedigree score",
    "Age": "years",
}


class DatasetError(RuntimeError):
    """Raised when the raw dataset is missing or fails validation."""


def load_raw() -> pd.DataFrame:
    if not RAW_FILE.exists():
        raise DatasetError(
            f"{RAW_FILE} not found. Run:  python scripts/download_diabetes.py"
        )
    return pd.read_csv(RAW_FILE)


def validate(df: pd.DataFrame) -> dict:
    """Structural validation. Raises ``DatasetError`` on any hard failure."""
    problems: list[str] = []

    expected_cols = FEATURES + [TARGET]
    if list(df.columns) != expected_cols:
        problems.append(f"columns {list(df.columns)} != expected {expected_cols}")

    if df.empty:
        problems.append("dataframe is empty")

    if TARGET in df.columns:
        bad = set(df[TARGET].dropna().unique()) - {0, 1}
        if bad:
            problems.append(f"target has non-binary values: {sorted(bad)}")

    for col in FEATURES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            problems.append(f"feature {col} is not numeric")

    if problems:
        raise DatasetError("dataset validation failed: " + "; ".join(problems))

    return {
        "rows": int(len(df)),
        "columns": list(df.columns),
        "target_distribution": {
            str(k): int(v) for k, v in df[TARGET].value_counts().sort_index().items()
        },
        "duplicate_rows": int(df.duplicated().sum()),
    }
