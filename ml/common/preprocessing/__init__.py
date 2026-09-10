"""Reusable, leakage-safe preprocessing building blocks.

Every transformer here learns its parameters only inside ``Pipeline.fit`` on the
training split. Nothing touches the target.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


class ZeroToNaN(BaseEstimator, TransformerMixin):
    """Replace exact zeros with NaN in the named columns.

    Used for datasets (e.g. Pima diabetes) where a physiologically impossible
    zero is really an encoded missing value. Stateless: it learns nothing, so it
    is safe either side of the split, but it lives in the pipeline for clarity.
    """

    def __init__(self, columns: Sequence[str]):
        self.columns = columns

    def fit(self, X, y=None):  # noqa: D102
        return self

    def transform(self, X):  # noqa: D102
        X = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in list(self.columns):
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X

    def get_feature_names_out(self, input_features=None):  # noqa: D102
        return np.asarray(input_features)


def build_preprocessor(
    numeric_features: Sequence[str],
    categorical_features: Sequence[str] = (),
    *,
    scale: bool = True,
    numeric_impute: str = "median",
    categorical_impute: str = "most_frequent",
) -> ColumnTransformer:
    """A ``ColumnTransformer`` with a numeric and (optional) categorical branch.

    - numeric: impute -> optional standard scale
    - categorical: impute -> one-hot (unknowns ignored at predict time)
    """
    numeric_steps: list = [("impute", SimpleImputer(strategy=numeric_impute))]
    if scale:
        numeric_steps.append(("scale", StandardScaler()))
    numeric_pipe = Pipeline(numeric_steps)

    transformers: list = [("num", numeric_pipe, list(numeric_features))]

    if categorical_features:
        categorical_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy=categorical_impute)),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("cat", categorical_pipe, list(categorical_features)))

    return ColumnTransformer(transformers=transformers, remainder="drop")
