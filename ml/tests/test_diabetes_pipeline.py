"""Diabetes data loading, validation, and the trained artifact."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

from ml.common.evaluation import binary_metrics, threshold_for_sensitivity
from ml.diabetes.load import FEATURES, RAW_FILE, DatasetError, load_raw, validate

ARTIFACT = Path(__file__).resolve().parents[2] / "models" / "diabetes-v1.joblib"
requires_model = pytest.mark.skipif(
    not ARTIFACT.exists(), reason="run `python -m ml.diabetes.train` first"
)
requires_data = pytest.mark.skipif(
    not RAW_FILE.exists(), reason="run `python scripts/download_diabetes.py` first"
)


@requires_data
def test_load_and_validate_real_dataset():
    df = load_raw()
    report = validate(df)
    assert report["rows"] == 768
    assert report["columns"][-1] == "Outcome"
    assert set(report["target_distribution"]) == {"0", "1"}


def test_validate_rejects_non_binary_target():
    bad = pd.DataFrame(
        [[0] * 8 + [3]], columns=FEATURES + ["Outcome"]
    )
    with pytest.raises(DatasetError):
        validate(bad)


def test_validate_rejects_wrong_columns():
    with pytest.raises(DatasetError):
        validate(pd.DataFrame({"foo": [1], "Outcome": [0]}))


@requires_model
def test_artifact_structure():
    payload = joblib.load(ARTIFACT)
    assert payload["model_id"] == "diabetes-v1"
    assert payload["features"] == FEATURES
    assert {"low_cut", "high_cut"} <= payload["thresholds"].keys()
    assert "feature_reference" in payload


@requires_model
def test_artifact_predicts_probabilities():
    payload = joblib.load(ARTIFACT)
    model = payload["model"]
    row = pd.DataFrame(
        [[6, 148, 72, 35, 120, 33.6, 0.627, 50]], columns=FEATURES
    )
    prob = model.predict_proba(row)[0, 1]
    assert 0.0 <= prob <= 1.0


@requires_model
def test_artifact_handles_encoded_missing_zeros():
    payload = joblib.load(ARTIFACT)
    model = payload["model"]
    # zeros in Insulin/SkinThickness are the dataset's encoded-missing; the
    # pipeline must not crash and must still return a valid probability
    row = pd.DataFrame([[2, 120, 70, 0, 0, 30.0, 0.4, 35]], columns=FEATURES)
    prob = model.predict_proba(row)[0, 1]
    assert np.isfinite(prob)


@requires_model
@requires_data
def test_reported_test_metrics_are_reproducible():
    """Re-run the exact split + evaluation and confirm the recorded numbers."""
    from ml.common.evaluation import binary_metrics
    from ml.common.framework import _split
    from ml.common.utilities import read_json
    from ml.diabetes.spec import SPEC

    meta = read_json(Path(__file__).resolve().parents[1] / "diabetes" / "model_metadata.json")
    payload = joblib.load(ARTIFACT)
    _, _, test, _ = _split(SPEC, SPEC.load())
    prob = payload["model"].predict_proba(test[SPEC.features])[:, 1]
    report = binary_metrics(test["target"].astype(int).to_numpy(), prob, threshold=0.5)
    assert report["roc_auc"] == pytest.approx(meta["test_metrics"]["roc_auc"], abs=1e-9)
    assert report["confusion_matrix"] == meta["test_metrics"]["confusion_matrix"]
