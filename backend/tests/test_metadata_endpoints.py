"""GET /health, /diseases, /models, /datasets, /schema/{disease}."""
from __future__ import annotations

from backend.tests.conftest import requires_model


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "models_loaded" in body
    assert "database" in body


def test_diseases_lists_all_twelve(client):
    r = client.get("/diseases")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 12
    keys = {d["key"] for d in body["diseases"]}
    assert {"diabetes", "stroke", "hypertension"} <= keys
    by_key = {d["key"]: d for d in body["diseases"]}
    assert by_key["stroke"]["model_available"] is False
    assert by_key["stroke"]["unavailable_reason"]
    assert by_key["hypertension"]["model_available"] is False


@requires_model
def test_diseases_marks_diabetes_available(client):
    body = client.get("/diseases").json()
    by_key = {d["key"]: d for d in body["diseases"]}
    assert by_key["diabetes"]["model_available"] is True
    assert "Glucose" in by_key["diabetes"]["required_features"]


@requires_model
def test_models_endpoint(client):
    body = client.get("/models").json()
    ids = {m["model_id"] for m in body["models"]}
    assert "diabetes-v1" in ids
    diab = next(m for m in body["models"] if m["model_id"] == "diabetes-v1")
    assert diab["algorithm"]
    assert "test_roc_auc" in diab["metrics"]
    assert diab["loaded"] is True


def test_datasets_endpoint(client):
    body = client.get("/datasets").json()
    assert body["count"] >= 12
    diab = next(d for d in body["datasets"] if d["disease"] == "diabetes")
    assert diab["records"] == 768
    assert diab["target"] == "Outcome"


@requires_model
def test_schema_endpoint(client):
    r = client.get("/schema/diabetes")
    assert r.status_code == 200
    schema = r.json()
    assert schema["required_features"] == [
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
    ]
    assert schema["feature_details"]["Glucose"]["min"] == 30


def test_schema_unknown_disease_404(client):
    assert client.get("/schema/nonsense").status_code == 404


def test_schema_unavailable_disease_404(client):
    assert client.get("/schema/stroke").status_code == 404
