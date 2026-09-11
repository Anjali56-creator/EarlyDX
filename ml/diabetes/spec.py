"""Diabetes disease spec (Pima Indians Diabetes Database)."""
from __future__ import annotations

import pandas as pd

from ml.common.framework import DiseaseSpec, FeatureMeta
from ml.diabetes.load import (
    FEATURE_RANGES,
    FEATURE_UNITS,
    FEATURES,
    TARGET as RAW_TARGET,
    load_raw,
    validate,
)


def _load() -> pd.DataFrame:
    df = load_raw()
    validate(df)
    return df.rename(columns={RAW_TARGET: "target"})


SPEC = DiseaseSpec(
    key="diabetes",
    display_name="Diabetes",
    version="v1",
    dataset_name="Pima Indians Diabetes Database",
    dataset_registry_key="diabetes",
    load=_load,
    numeric_features=list(FEATURES),
    zero_as_missing=["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    feature_meta={
        f: FeatureMeta(f, "number", FEATURE_UNITS[f], FEATURE_RANGES[f][0], FEATURE_RANGES[f][1])
        for f in FEATURES
    },
    positive_label="tested positive for diabetes (Outcome = 1)",
    limitations=(
        "female Pima (Akimel O'odham) patients aged 21+ only; small sample; "
        "encoded-missing zeros; not externally validated"
    ),
    ethical_note=(
        "the Pima (Akimel O'odham) community has raised concerns about research "
        "use of this dataset; see LIMITATIONS.md"
    ),
)
