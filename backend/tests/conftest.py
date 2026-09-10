"""Shared test fixtures.

Tests run against the real trained diabetes model (they are skipped if it has
not been trained yet) and a throwaway SQLite database.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = REPO_ROOT / "models" / "diabetes-v1.joblib"

_tmp_db = Path(tempfile.gettempdir()) / "earlydx_test.db"
os.environ.setdefault("EARLYDX_DATABASE_URL", f"sqlite:///{_tmp_db.as_posix()}")


requires_model = pytest.mark.skipif(
    not ARTIFACT.exists(),
    reason="diabetes-v1.joblib not found - run `python -m ml.diabetes.train`",
)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    with TestClient(app) as c:
        yield c


VALID_DIABETES_PAYLOAD = {
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "SkinThickness": 35,
    "Insulin": 120,
    "BMI": 33.6,
    "DiabetesPedigreeFunction": 0.627,
    "Age": 50,
}
