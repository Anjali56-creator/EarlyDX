"""Liver disease spec — UCI Indian Liver Patient Dataset (ILPD), 583 rows."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_from_mirrors
from ml.common.framework import DiseaseSpec, FeatureMeta

COLUMNS = [
    "Age", "Gender", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio", "Selector",
]
MIRRORS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00225/Indian%20Liver%20Patient%20Dataset%20(ILPD).csv",
]


def download() -> Path:
    return download_from_mirrors("liver_disease", MIRRORS, "ilpd.csv")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path, header=None, names=COLUMNS)
    # Selector: 1 = liver patient, 2 = not. Model "liver patient" as positive.
    df["target"] = (df["Selector"] == 1).astype(int)
    df = df.drop(columns=["Selector"])
    return df


NUMERIC = [
    "Age", "Total_Bilirubin", "Direct_Bilirubin", "Alkaline_Phosphotase",
    "Alamine_Aminotransferase", "Aspartate_Aminotransferase", "Total_Protiens",
    "Albumin", "Albumin_and_Globulin_Ratio",
]
UNITS = {
    "Age": "years",
    "Total_Bilirubin": "mg/dL",
    "Direct_Bilirubin": "mg/dL",
    "Alkaline_Phosphotase": "IU/L",
    "Alamine_Aminotransferase": "IU/L (ALT)",
    "Aspartate_Aminotransferase": "IU/L (AST)",
    "Total_Protiens": "g/dL",
    "Albumin": "g/dL",
    "Albumin_and_Globulin_Ratio": "ratio",
}

SPEC = DiseaseSpec(
    key="liver_disease",
    display_name="Liver Disease",
    version="v1",
    dataset_name="UCI Indian Liver Patient Dataset (ILPD)",
    dataset_registry_key="liver_disease",
    load=_load,
    numeric_features=NUMERIC,
    categorical_features=["Gender"],
    feature_meta={
        **{f: FeatureMeta(f, "number", UNITS[f]) for f in NUMERIC},
        "Gender": FeatureMeta("Gender", "categorical", options=["Female", "Male"]),
    },
    positive_label="liver patient (Selector = 1)",
    limitations=(
        "583 records from one region of Andhra Pradesh, India; class imbalance "
        "(~71% patients); strong sex imbalance; a few missing "
        "Albumin_and_Globulin_Ratio values"
    ),
)
