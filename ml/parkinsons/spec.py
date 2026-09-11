"""Parkinson's spec — UCI Oxford Parkinson's Disease Detection (voice).

195 sustained-phonation recordings from 31 subjects. Because each subject
contributes several rows, the split and cross-validation are grouped by subject
to avoid subject leakage.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_from_mirrors
from ml.common.framework import DiseaseSpec, FeatureMeta

MIRRORS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data",
]


def download() -> Path:
    return download_from_mirrors("parkinsons", MIRRORS, "parkinsons.data")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path)
    # name like "phon_R01_S01_1" -> subject "phon_R01_S01"
    df["subject"] = df["name"].str.rsplit("_", n=1).str[0]
    df = df.drop(columns=["name"])
    return df.rename(columns={"status": "target"})


FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "MDVP:Jitter(%)",
    "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ", "Jitter:DDP", "MDVP:Shimmer",
    "MDVP:Shimmer(dB)", "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

SPEC = DiseaseSpec(
    key="parkinsons",
    display_name="Parkinson's Disease",
    version="v1",
    dataset_name="UCI Oxford Parkinson's Disease Detection Dataset",
    dataset_registry_key="parkinsons",
    load=_load,
    numeric_features=FEATURES,
    group_column="subject",
    feature_meta={f: FeatureMeta(f, "number", "acoustic voice measure") for f in FEATURES},
    positive_label="Parkinson's disease (status = 1)",
    limitations=(
        "only 31 subjects (23 with Parkinson's); inputs are voice-signal features, "
        "not user-enterable; split and CV are grouped by subject; tiny sample so "
        "metrics have wide confidence intervals"
    ),
)
