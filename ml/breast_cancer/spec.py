"""Breast cancer spec — UCI Breast Cancer Wisconsin (Diagnostic), WDBC.

The dataset ships inside scikit-learn (`sklearn.datasets.load_breast_cancer`),
which is the exact UCI WDBC data (569 rows, 30 features). We still write a CSV
cache under data/raw/ for transparency.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.config import RAW_DIR
from ml.common.framework import DiseaseSpec, FeatureMeta

CACHE = RAW_DIR / "breast_cancer" / "wdbc_from_sklearn.csv"


def download() -> Path:
    from sklearn.datasets import load_breast_cancer

    CACHE.parent.mkdir(parents=True, exist_ok=True)
    if CACHE.exists() and CACHE.stat().st_size > 0:
        return CACHE
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    # sklearn: target 0 = malignant, 1 = benign. We model malignant as positive.
    df["diagnosis_malignant"] = (df["target"] == 0).astype(int)
    df = df.drop(columns=["target"])
    df.to_csv(CACHE, index=False)
    return CACHE


def _load() -> pd.DataFrame:
    if not CACHE.exists():
        download()
    df = pd.read_csv(CACHE)
    return df.rename(columns={"diagnosis_malignant": "target"})


FEATURES = [
    "mean radius", "mean texture", "mean perimeter", "mean area", "mean smoothness",
    "mean compactness", "mean concavity", "mean concave points", "mean symmetry",
    "mean fractal dimension", "radius error", "texture error", "perimeter error",
    "area error", "smoothness error", "compactness error", "concavity error",
    "concave points error", "symmetry error", "fractal dimension error",
    "worst radius", "worst texture", "worst perimeter", "worst area",
    "worst smoothness", "worst compactness", "worst concavity",
    "worst concave points", "worst symmetry", "worst fractal dimension",
]

SPEC = DiseaseSpec(
    key="breast_cancer",
    display_name="Breast Cancer",
    version="v1",
    dataset_name="UCI Breast Cancer Wisconsin (Diagnostic) / WDBC (via scikit-learn)",
    dataset_registry_key="breast_cancer",
    load=_load,
    numeric_features=FEATURES,
    feature_meta={f: FeatureMeta(f, "number", "computed from digitised FNA image") for f in FEATURES},
    positive_label="malignant diagnosis",
    limitations=(
        "features are computed from digitised fine-needle-aspirate images, not "
        "routine intake data; single-institution sample; this is a pipeline "
        "demonstration, not a self-service risk tool"
    ),
)
