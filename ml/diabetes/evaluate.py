"""Evaluate a fitted diabetes pipeline on a held-out set.

Kept separate from training so the same metric code runs in tests and in the
training script, and so no metric is ever written by hand.
"""
from __future__ import annotations

from typing import Any

import pandas as pd

from ml.common.evaluation import binary_metrics, reliability
from ml.diabetes.load import FEATURES, TARGET


def predict_proba(model, df: pd.DataFrame):
    return model.predict_proba(df[FEATURES])[:, 1]


def evaluate(model, df: pd.DataFrame, *, threshold: float = 0.5) -> dict[str, Any]:
    y_true = df[TARGET].astype(int).to_numpy()
    y_prob = predict_proba(model, df)
    report = binary_metrics(y_true, y_prob, threshold=threshold)
    report["calibration"] = reliability(y_true, y_prob)
    return report
