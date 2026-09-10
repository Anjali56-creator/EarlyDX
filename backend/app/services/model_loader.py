"""Load model artifacts listed in the model registry and cache them in memory.

The backend never fabricates a prediction: if an artifact is missing or fails to
load, that disease is simply reported as unavailable.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from backend.app.config import REPO_ROOT, get_settings
from ml.common.utilities import read_json


@dataclass
class ModelBundle:
    model_id: str
    disease: str
    estimator: Any
    features: list[str]
    thresholds: dict[str, float]
    calibrated: bool
    metadata: dict[str, Any]
    registry_entry: dict[str, Any]

    def risk_level(self, score: float) -> str:
        low = self.thresholds.get("low_cut", 1 / 3)
        high = self.thresholds.get("high_cut", 2 / 3)
        if score >= high:
            return "HIGH"
        if score < low:
            return "LOW"
        return "MODERATE"


class ModelStore:
    def __init__(self) -> None:
        self._bundles: dict[str, ModelBundle] = {}
        self._errors: dict[str, str] = {}
        self._lock = threading.Lock()

    # ── loading ──────────────────────────────────────────────────────────
    def load_all(self) -> None:
        settings = get_settings()
        registry_path = settings.models_path / "model_registry.json"
        with self._lock:
            self._bundles.clear()
            self._errors.clear()
            if not registry_path.exists():
                return
            registry = read_json(registry_path).get("models", {})
            for model_id, entry in registry.items():
                if entry.get("status") not in {"active", "experimental"}:
                    continue
                try:
                    self._bundles[entry["disease"]] = self._load_one(entry)
                except Exception as exc:  # noqa: BLE001
                    self._errors[entry["disease"]] = f"{type(exc).__name__}: {exc}"

    def _load_one(self, entry: dict[str, Any]) -> ModelBundle:
        artifact_path = Path(entry["artifact_path"])
        if not artifact_path.is_absolute():
            artifact_path = REPO_ROOT / artifact_path
        payload = joblib.load(artifact_path)

        disease = entry["disease"]
        meta_path = REPO_ROOT / "ml" / disease / "model_metadata.json"
        metadata = read_json(meta_path) if meta_path.exists() else {}

        return ModelBundle(
            model_id=entry["model_id"],
            disease=disease,
            estimator=payload["model"],
            features=list(payload["features"]),
            thresholds=dict(payload.get("thresholds", {})),
            calibrated=bool(payload.get("calibrated", False)),
            metadata=metadata,
            registry_entry=entry,
        )

    # ── access ───────────────────────────────────────────────────────────
    def get(self, disease: str) -> ModelBundle | None:
        return self._bundles.get(disease)

    def available(self) -> list[str]:
        return sorted(self._bundles)

    def errors(self) -> dict[str, str]:
        return dict(self._errors)

    def count(self) -> int:
        return len(self._bundles)


STORE = ModelStore()
