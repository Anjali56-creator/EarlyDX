"""Heart failure spec — UCI Heart Failure Clinical Records (299 patients).

Target: death during the follow-up period among patients already diagnosed with
heart failure. This is an adverse-outcome model, NOT a "will you develop heart
failure" model. The `time` (follow-up days) column is dropped because it leaks
the outcome.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_from_mirrors
from ml.common.framework import DiseaseSpec, FeatureMeta

MIRRORS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00519/heart_failure_clinical_records_dataset.csv",
    "https://archive.ics.uci.edu/static/public/519/heart+failure+clinical+records.zip",
]


def download() -> Path:
    return download_from_mirrors("heart_failure", MIRRORS[:1], "heart_failure_clinical_records.csv")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in ["time"] if c in df.columns])
    return df.rename(columns={"DEATH_EVENT": "target"})


NUMERIC = [
    "age", "creatinine_phosphokinase", "ejection_fraction", "platelets",
    "serum_creatinine", "serum_sodium",
]
BINARY = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]

UNITS = {
    "age": "years",
    "creatinine_phosphokinase": "mcg/L (CPK enzyme)",
    "ejection_fraction": "% blood leaving the heart per contraction",
    "platelets": "kiloplatelets/mL",
    "serum_creatinine": "mg/dL",
    "serum_sodium": "mEq/L",
    "anaemia": "0/1 (decrease of red blood cells)",
    "diabetes": "0/1",
    "high_blood_pressure": "0/1 (hypertension)",
    "sex": "0 = woman, 1 = man",
    "smoking": "0/1",
}

SPEC = DiseaseSpec(
    key="heart_failure",
    display_name="Heart Failure",
    version="v1",
    dataset_name="UCI Heart Failure Clinical Records",
    dataset_registry_key="heart_failure",
    load=_load,
    numeric_features=NUMERIC + BINARY,
    feature_meta={
        **{f: FeatureMeta(f, "number", UNITS[f]) for f in NUMERIC},
        **{f: FeatureMeta(f, "number", UNITS[f], 0, 1) for f in BINARY},
    },
    positive_label="death during the follow-up period (DEATH_EVENT = 1)",
    limitations=(
        "299 patients from one hospital in Faisalabad, Pakistan (2015); target is "
        "mortality among people already diagnosed with heart failure, not "
        "incidence; `time` column dropped to avoid leakage; very small sample"
    ),
)
