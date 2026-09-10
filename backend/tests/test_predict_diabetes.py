"""POST /predict/diabetes and POST /predict/all."""
from __future__ import annotations

import copy

from backend.tests.conftest import VALID_DIABETES_PAYLOAD, requires_model


@requires_model
def test_predict_diabetes_ok(client):
    r = client.post("/predict/diabetes", json=VALID_DIABETES_PAYLOAD)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["disease"] == "Diabetes"
    assert body["model_version"] == "diabetes-v1"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    assert body["important_features"]
    assert body["important_features"][0]["feature"] == "Glucose"
    assert body["disclaimer"]


@requires_model
def test_predict_diabetes_is_deterministic(client):
    a = client.post("/predict/diabetes", json=VALID_DIABETES_PAYLOAD).json()
    b = client.post("/predict/diabetes", json=VALID_DIABETES_PAYLOAD).json()
    assert a["risk_score"] == b["risk_score"]


@requires_model
def test_low_and_high_inputs_move_the_score(client):
    low = {
        "Pregnancies": 1, "Glucose": 85, "BloodPressure": 66, "SkinThickness": 29,
        "Insulin": 90, "BMI": 26.6, "DiabetesPedigreeFunction": 0.351, "Age": 22,
    }
    high = {
        "Pregnancies": 8, "Glucose": 197, "BloodPressure": 74, "SkinThickness": 40,
        "Insulin": 300, "BMI": 45.0, "DiabetesPedigreeFunction": 1.4, "Age": 60,
    }
    s_low = client.post("/predict/diabetes", json=low).json()["risk_score"]
    s_high = client.post("/predict/diabetes", json=high).json()["risk_score"]
    assert s_high > s_low


def test_predict_diabetes_out_of_range_422(client):
    bad = copy.deepcopy(VALID_DIABETES_PAYLOAD)
    bad["Glucose"] = 5000
    assert client.post("/predict/diabetes", json=bad).status_code == 422


def test_predict_diabetes_missing_feature_422(client):
    bad = copy.deepcopy(VALID_DIABETES_PAYLOAD)
    del bad["Glucose"]
    assert client.post("/predict/diabetes", json=bad).status_code == 422


def test_predict_diabetes_extra_field_422(client):
    bad = copy.deepcopy(VALID_DIABETES_PAYLOAD)
    bad["smoker"] = 1
    assert client.post("/predict/diabetes", json=bad).status_code == 422


def test_predict_unknown_disease_404(client):
    assert client.post("/predict/nonsense", json={}).status_code == 404


def test_predict_unavailable_disease_409(client):
    r = client.post("/predict/stroke", json=VALID_DIABETES_PAYLOAD)
    assert r.status_code == 409
    assert "stroke" in r.text.lower() or "model" in r.text.lower()


@requires_model
def test_predict_all_runs_supported_and_skips_rest(client):
    payload = {"features": {**VALID_DIABETES_PAYLOAD, "some_unrelated_lab": 12.3}}
    r = client.post("/predict/all", json=payload)
    assert r.status_code == 200, r.text
    body = r.json()
    diseases_run = {x["disease"] for x in body["results"]}
    assert "Diabetes" in diseases_run
    assert isinstance(body["skipped"], list)


@requires_model
def test_predict_all_skips_diabetes_when_features_missing(client):
    r = client.post("/predict/all", json={"features": {"Age": 40}})
    assert r.status_code == 200
    body = r.json()
    assert body["results"] == []
    skipped_keys = {s["disease_key"] for s in body["skipped"]}
    assert "diabetes" in skipped_keys
    diab_skip = next(s for s in body["skipped"] if s["disease_key"] == "diabetes")
    assert "Glucose" in diab_skip["missing_features"]
