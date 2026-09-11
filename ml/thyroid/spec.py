"""Thyroid spec — UCI 'new-thyroid' dataset (215 rows, 5 assays, 3 classes).

The 'new-thyroid' subset is small but clean and fully numeric. Classes:
1 = normal, 2 = hyperthyroid, 3 = hypothyroid. Modelled as
abnormal (hyper or hypo) vs normal.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.common.datasets import download_from_mirrors
from ml.common.framework import DiseaseSpec, FeatureMeta

COLUMNS = ["thyroid_class", "T3_resin", "total_thyroxin", "total_triiodothyronine",
           "TSH", "TSH_diff_after_TRH"]
MIRRORS = [
    "https://archive.ics.uci.edu/ml/machine-learning-databases/thyroid-disease/new-thyroid.data",
]


def download() -> Path:
    return download_from_mirrors("thyroid", MIRRORS, "new-thyroid.data")


def _load() -> pd.DataFrame:
    path = download()
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["target"] = (df["thyroid_class"] != 1).astype(int)
    return df.drop(columns=["thyroid_class"])


FEATURES = ["T3_resin", "total_thyroxin", "total_triiodothyronine", "TSH", "TSH_diff_after_TRH"]
UNITS = {
    "T3_resin": "T3-resin uptake test (%)",
    "total_thyroxin": "total serum thyroxin (T4)",
    "total_triiodothyronine": "total serum triiodothyronine (T3)",
    "TSH": "basal thyroid-stimulating hormone",
    "TSH_diff_after_TRH": "max absolute TSH change after TRH injection",
}

SPEC = DiseaseSpec(
    key="thyroid",
    display_name="Thyroid Disease",
    version="v1",
    dataset_name="UCI Thyroid Disease ('new-thyroid' subset)",
    dataset_registry_key="thyroid",
    load=_load,
    numeric_features=FEATURES,
    feature_meta={f: FeatureMeta(f, "number", UNITS[f]) for f in FEATURES},
    positive_label="abnormal thyroid function (hyper- or hypothyroid)",
    notes='The small new-thyroid subset is close to separable, so test metrics are near-perfect; this reflects the dataset, not real-world performance.',
    limitations=(
        "only 215 samples; historical (Garvan Institute); the larger UCI thyroid "
        "subsets (sick, allhypo, thyroid0387) were not used in V1; five assay "
        "values, not routine intake data"
    ),
)
