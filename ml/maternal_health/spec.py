"""Maternal health risk spec — UCI Maternal Health Risk (1014 rows).

Target RiskLevel has three classes (low / mid / high risk). Modelled as
high risk vs not-high-risk. Exact-duplicate rows are dropped because the raw
file contains many identical rows that would otherwise leak across the split.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_zip_member
from ml.common.framework import DiseaseSpec, FeatureMeta

ZIP_URL = "https://archive.ics.uci.edu/static/public/863/maternal+health+risk.zip"


def download() -> Path:
    return download_zip_member("maternal_health", ZIP_URL, ".csv", "maternal_health_risk.csv")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path)
    df["target"] = (df["RiskLevel"].str.strip().str.lower() == "high risk").astype(int)
    return df.drop(columns=["RiskLevel"])


FEATURES = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
UNITS = {
    "Age": "years",
    "SystolicBP": "mm Hg",
    "DiastolicBP": "mm Hg",
    "BS": "mmol/L (blood glucose)",
    "BodyTemp": "degrees Fahrenheit",
    "HeartRate": "bpm (resting)",
}

SPEC = DiseaseSpec(
    key="maternal_health",
    display_name="Maternal Health Risk",
    version="v1",
    dataset_name="UCI Maternal Health Risk",
    dataset_registry_key="maternal_health",
    load=_load,
    numeric_features=FEATURES,
    drop_exact_duplicates=True,
    feature_meta={f: FeatureMeta(f, "number", UNITS[f]) for f in FEATURES},
    positive_label="high risk during pregnancy (RiskLevel = 'high risk')",
    limitations=(
        "collected via IoT devices from rural Bangladesh clinics; the raw file "
        "contains many exact-duplicate rows (dropped here); the upstream "
        "risk-level labelling method is not fully specified"
    ),
)
