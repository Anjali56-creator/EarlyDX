"""Heart disease spec — UCI Heart Disease, Cleveland subset (303 rows)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.common.datasets import download_from_mirrors
from ml.common.framework import DiseaseSpec, FeatureMeta

COLUMNS = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "num",
]
MIRRORS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data",
]


def download() -> Path:
    return download_from_mirrors("heart_disease", MIRRORS, "processed.cleveland.data")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path, header=None, names=COLUMNS, na_values="?")
    # num: 0 = no disease, 1-4 = disease present -> binary
    df["target"] = (df["num"].fillna(0) > 0).astype(int)
    df = df.drop(columns=["num"])
    for c in ["cp", "restecg", "slope", "thal"]:
        df[c] = df[c].astype("Int64").astype(str).replace("<NA>", np.nan)
    return df


NUMERIC = ["age", "sex", "trestbps", "chol", "fbs", "thalach", "exang", "oldpeak", "ca"]
CATEGORICAL = ["cp", "restecg", "slope", "thal"]
UNITS = {
    "age": "years",
    "sex": "0 = female, 1 = male",
    "trestbps": "mm Hg (resting blood pressure)",
    "chol": "mg/dL (serum cholesterol)",
    "fbs": "0/1 (fasting blood sugar > 120 mg/dL)",
    "thalach": "bpm (max heart rate achieved)",
    "exang": "0/1 (exercise-induced angina)",
    "oldpeak": "ST depression induced by exercise",
    "ca": "count (major vessels coloured by fluoroscopy, 0-3)",
}

SPEC = DiseaseSpec(
    key="heart_disease",
    display_name="Heart Disease",
    version="v1",
    dataset_name="UCI Heart Disease (Cleveland)",
    dataset_registry_key="heart_disease",
    load=_load,
    numeric_features=NUMERIC,
    categorical_features=CATEGORICAL,
    feature_meta={
        **{f: FeatureMeta(f, "number", UNITS[f]) for f in NUMERIC},
        "cp": FeatureMeta("cp", "categorical", options=["1", "2", "3", "4"]),
        "restecg": FeatureMeta("restecg", "categorical", options=["0", "1", "2"]),
        "slope": FeatureMeta("slope", "categorical", options=["1", "2", "3"]),
        "thal": FeatureMeta("thal", "categorical", options=["3", "6", "7"]),
    },
    positive_label="angiographic heart disease present (num > 0)",
    limitations=(
        "~303 Cleveland Clinic records; `ca` and `thal` have missing values; "
        "avoid the 1025-row Kaggle heart.csv which contains duplicated rows"
    ),
)
