"""Leakage-safe scikit-learn pipeline factory for the diabetes model.

The returned object is a single ``Pipeline`` that owns every learned parameter
(zeros->NaN, median imputation, scaling, the estimator). Fitting it on the
training split is therefore the only thing a caller needs to do to stay
leakage-free.
"""
from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ml.common.config import RANDOM_STATE
from ml.common.preprocessing import ZeroToNaN, build_preprocessor
from ml.diabetes.load import FEATURES, ZERO_AS_MISSING


def _prefix() -> list:
    return [
        ("zeros", ZeroToNaN(ZERO_AS_MISSING)),
        ("pre", build_preprocessor(FEATURES, scale=True)),
    ]


def candidate_estimators() -> dict[str, object]:
    """Baseline first, then stronger candidates. Keys are algorithm names."""
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


BASELINE_NAME = "LogisticRegression"


def build_pipeline(estimator) -> Pipeline:
    return Pipeline(_prefix() + [("clf", estimator)])
