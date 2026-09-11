"""Generic prediction coverage for every model the API has loaded.

For each loaded model this builds a schema-valid payload (numeric midpoints,
first categorical option), calls /predict/{disease}, and checks the contract.
"""
from __future__ import annotations

import math

import pytest


def _loaded_diseases(client) -> list[str]:
    body = client.get("/diseases").json()
    return [d["key"] for d in body["diseases"] if d["model_available"]]


def _sample_payload(schema: dict) -> dict:
    payload: dict[str, object] = {}
    for name, d in schema["feature_details"].items():
        if d["type"] == "categorical":
            payload[name] = d["options"][0]
        else:
            lo, hi = float(d.get("min", 0.0)), float(d.get("max", 1.0))
            mid = (lo + hi) / 2
            # Round to a fixed number of significant figures so tiny-magnitude
            # ranges (e.g. acoustic jitter measures ~1e-5) don't get rounded
            # away to 0.0 and fall outside the schema's own min bound.
            if mid == 0:
                payload[name] = mid
            else:
                decimals = max(3, 3 - math.floor(math.log10(abs(mid))))
                payload[name] = round(mid, decimals)
    return payload


def test_at_least_one_model_loaded(client):
    assert client.get("/health").json()["models_loaded"] >= 1


def test_every_loaded_model_predicts(client):
    diseases = _loaded_diseases(client)
    assert diseases, "no models loaded"
    for disease in diseases:
        schema = client.get(f"/schema/{disease}").json()
        payload = _sample_payload(schema)
        r = client.post(f"/predict/{disease}", json=payload)
        assert r.status_code == 200, f"{disease}: {r.text}"
        body = r.json()
        assert 0.0 <= body["risk_score"] <= 1.0, disease
        assert body["risk_level"] in {"LOW", "MODERATE", "HIGH"}, disease
        assert body["model_version"].startswith(disease), disease
        assert body["important_features"], disease
        assert body["disclaimer"]


def test_every_loaded_model_rejects_missing_feature(client):
    for disease in _loaded_diseases(client):
        schema = client.get(f"/schema/{disease}").json()
        payload = _sample_payload(schema)
        payload.pop(next(iter(payload)))
        r = client.post(f"/predict/{disease}", json=payload)
        assert r.status_code == 422, f"{disease} should 422 on missing feature, got {r.status_code}"


def test_predict_all_ranks_and_reports_skipped(client):
    diseases = _loaded_diseases(client)
    # union of all features from a couple of models, so some run and some skip
    feats: dict[str, object] = {}
    for disease in diseases[:2]:
        feats.update(_sample_payload(client.get(f"/schema/{disease}").json()))
    r = client.post("/predict/all", json={"features": feats})
    assert r.status_code == 200, r.text
    body = r.json()
    levels = [x["risk_level"] for x in body["results"]]
    order = {"HIGH": 0, "MODERATE": 1, "LOW": 2}
    assert levels == sorted(levels, key=lambda l: order[l])
    assert isinstance(body["skipped"], list)


@pytest.mark.parametrize("disease", ["stroke", "hypertension"])
def test_unavailable_diseases_stay_unavailable(client, disease):
    assert client.post(f"/predict/{disease}", json={}).status_code == 409
    assert client.get(f"/schema/{disease}").status_code == 404
