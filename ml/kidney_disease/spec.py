"""Chronic kidney disease spec — UCI Chronic Kidney Disease (#336, 400 rows).

Fetched from UCI's parsed ``data.csv`` endpoint. Heavy missingness across most
columns; the pipeline imputes (median / most-frequent) inside the training fold.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_from_mirrors
from ml.common.framework import DiseaseSpec, FeatureMeta

MIRRORS = ["https://archive.ics.uci.edu/static/public/336/data.csv"]

NUMERIC = ["age", "bp", "sg", "al", "su", "bgr", "bu", "sc", "sod", "pot",
           "hemo", "pcv", "wbcc", "rbcc"]
CATEGORICAL = ["rbc", "pc", "pcc", "ba", "htn", "dm", "cad", "appet", "pe", "ane"]

UNITS = {
    "age": "years", "bp": "mm Hg (blood pressure)", "sg": "urine specific gravity",
    "al": "albumin (0-5)", "su": "sugar (0-5)", "bgr": "mg/dL (random glucose)",
    "bu": "mg/dL (blood urea)", "sc": "mg/dL (serum creatinine)",
    "sod": "mEq/L (sodium)", "pot": "mEq/L (potassium)", "hemo": "g/dL (haemoglobin)",
    "pcv": "packed cell volume", "wbcc": "cells/cmm (white blood cells)",
    "rbcc": "millions/cmm (red blood cells)",
}


def download() -> Path:
    return download_from_mirrors("kidney_disease", MIRRORS, "ckd_uci.csv")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path)
    for c in NUMERIC:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in CATEGORICAL:
        df[c] = df[c].astype(str).str.strip().replace({"nan": None, "": None})
    df["target"] = (df["class"].astype(str).str.strip().str.lower() == "ckd").astype(int)
    return df.drop(columns=["class"])


SPEC = DiseaseSpec(
    key="kidney_disease",
    display_name="Chronic Kidney Disease",
    version="v1",
    dataset_name="UCI Chronic Kidney Disease",
    dataset_registry_key="kidney_disease",
    load=_load,
    numeric_features=NUMERIC,
    categorical_features=CATEGORICAL,
    feature_meta={
        **{f: FeatureMeta(f, "number", UNITS[f]) for f in NUMERIC},
        "rbc": FeatureMeta("rbc", "categorical", options=["normal", "abnormal"]),
        "pc": FeatureMeta("pc", "categorical", options=["normal", "abnormal"]),
        "pcc": FeatureMeta("pcc", "categorical", options=["present", "notpresent"]),
        "ba": FeatureMeta("ba", "categorical", options=["present", "notpresent"]),
        "htn": FeatureMeta("htn", "categorical", options=["yes", "no"]),
        "dm": FeatureMeta("dm", "categorical", options=["yes", "no"]),
        "cad": FeatureMeta("cad", "categorical", options=["yes", "no"]),
        "appet": FeatureMeta("appet", "categorical", options=["good", "poor"]),
        "pe": FeatureMeta("pe", "categorical", options=["yes", "no"]),
        "ane": FeatureMeta("ane", "categorical", options=["yes", "no"]),
    },
    positive_label="chronic kidney disease (class = ckd)",
    notes='The UCI CKD dataset is close to linearly separable (haemoglobin, serum creatinine, specific gravity and albumin are highly discriminative), so test metrics are near-perfect. This reflects the dataset, not real-world screening performance on undiagnosed patients.',
    limitations=(
        "400 records from a single hospital in India collected over ~2 months; "
        "heavy missingness in most columns; small sample"
    ),
)
