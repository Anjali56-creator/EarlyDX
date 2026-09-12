"""GET /evaluation, /evaluation/{disease}, /assessments/recent."""
from __future__ import annotations

from backend.tests.conftest import VALID_DIABETES_PAYLOAD, requires_model


@requires_model
def test_evaluation_summary_lists_loaded_models(client):
    r = client.get("/evaluation")
    assert r.status_code == 200
    body = r.json()
    assert body["count"] >= 1
    by_key = {m["disease_key"]: m for m in body["models"]}
    d = by_key["diabetes"]
    assert d["model_version"] == "diabetes-v1"
    assert d["selected_algorithm"]
    assert "LogisticRegression" in d["candidates_compared"]
    # test metrics are the standard classification set, read from metadata
    for k in ("accuracy", "precision", "recall_sensitivity", "f1", "roc_auc"):
        assert isinstance(d["test_metrics"][k], float)


@requires_model
def test_evaluation_detail_matches_model_metadata(client):
    r = client.get("/evaluation/diabetes")
    assert r.status_code == 200
    ev = r.json()
    assert ev["disease_key"] == "diabetes"
    # candidate comparison: baseline plus at least one challenger
    assert ev["baseline_algorithm"] in ev["candidates"]
    assert len(ev["candidates"]) >= 2
    assert ev["best_validation_candidate_by_roc_auc"] in ev["candidates"]
    # held-out test evaluation with a real confusion matrix
    cm = ev["test_metrics"]["confusion_matrix"]
    assert set(cm) == {"tn", "fp", "fn", "tp"}
    assert sum(cm.values()) == ev["test_metrics"]["n"]
    assert ev["cross_validation"]["folds"]
    assert ev["calibration"]["method"] in {"none", "sigmoid", "isotonic"}
    # honesty flag: ROC curve points are not stored
    assert "roc_curve_points" in ev["unavailable"]

    # numbers must agree with the registry's rounded metrics
    reg = {m["model_id"]: m for m in client.get("/models").json()["models"]}
    assert round(ev["test_metrics"]["roc_auc"], 4) == reg["diabetes-v1"]["metrics"]["test_roc_auc"]


def test_evaluation_detail_unknown_disease_404(client):
    assert client.get("/evaluation/stroke").status_code == 404
    assert client.get("/evaluation/not-a-disease").status_code == 404


@requires_model
def test_recent_assessments_reflects_a_prediction(client):
    before = client.get("/assessments/recent").json()
    assert set(before) >= {"available", "total_sessions", "results"}
    if not before["available"]:
        return  # no DB in this environment; endpoint must still answer
    client.post("/predict/diabetes", json=VALID_DIABETES_PAYLOAD)
    after = client.get("/assessments/recent?limit=5").json()
    assert after["total_sessions"] == before["total_sessions"] + 1
    assert after["results"][0]["disease_key"] == "diabetes"
    assert after["results"][0]["model_id"] == "diabetes-v1"
    assert after["results"][0]["risk_level"] in {"LOW", "MODERATE", "HIGH"}
    # no input features are ever persisted or returned
    assert "features" not in after["results"][0]
    assert len(after["results"]) <= 5
