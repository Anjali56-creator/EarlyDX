"""Obesity spec — UCI 'Estimation of Obesity Levels' (#544, 2111 rows).

Important caveats (see LIMITATIONS.md):
  * 77% of rows were generated with SMOTE (Weka); only 23% are real responses.
  * The obesity class is essentially a function of BMI (height & weight), so
    Height and Weight are DELIBERATELY EXCLUDED to stop the model re-deriving the
    label. The model predicts obesity from lifestyle / demographic features only.

Target: NObeyesdad in {Obesity_Type_I, Obesity_Type_II, Obesity_Type_III} -> 1.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_zip_member
from ml.common.framework import DiseaseSpec, FeatureMeta

ZIP_URL = (
    "https://archive.ics.uci.edu/static/public/544/"
    "estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition.zip"
)


def download() -> Path:
    return download_zip_member("obesity", ZIP_URL, ".csv", "obesity_levels.csv")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path)
    obese = {"Obesity_Type_I", "Obesity_Type_II", "Obesity_Type_III"}
    df["target"] = df["NObeyesdad"].isin(obese).astype(int)
    df = df.drop(columns=["NObeyesdad", "Height", "Weight"])
    return df


NUMERIC = ["Age", "FCVC", "NCP", "CH2O", "FAF", "TUE"]
CATEGORICAL = ["Gender", "family_history_with_overweight", "FAVC", "CAEC", "SMOKE", "SCC", "CALC", "MTRANS"]
UNITS = {
    "Age": "years",
    "FCVC": "frequency of vegetable consumption (1-3)",
    "NCP": "number of main meals per day",
    "CH2O": "daily water intake (litres, 1-3)",
    "FAF": "physical activity frequency (days/week, 0-3)",
    "TUE": "time using technology devices (hours, 0-2)",
}

SPEC = DiseaseSpec(
    key="obesity",
    display_name="Obesity",
    version="v1",
    dataset_name="UCI Estimation of Obesity Levels Based on Eating Habits and Physical Condition",
    dataset_registry_key="obesity",
    load=_load,
    numeric_features=NUMERIC,
    categorical_features=CATEGORICAL,
    feature_meta={
        **{f: FeatureMeta(f, "number", UNITS[f]) for f in NUMERIC},
        **{f: FeatureMeta(f, "categorical") for f in CATEGORICAL},
    },
    positive_label="obesity (NObeyesdad in Obesity_Type_I/II/III)",
    limitations=(
        "77% of rows are SMOTE-synthetic, so metrics are weak evidence; population "
        "is Mexico/Peru/Colombia; Height and Weight excluded on purpose because "
        "the label is BMI-derived; treat this as a demonstration model"
    ),
    notes=(
        "Because most of the dataset is synthetic, this model must not be used to "
        "claim real-world obesity-prediction performance."
    ),
)
