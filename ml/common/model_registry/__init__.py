"""Read/write helpers for ``models/model_registry.json``.

The registry is the single source of truth the backend uses to decide which
models exist and which artifact produced a prediction.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from ml.common.config import MODEL_REGISTRY
from ml.common.utilities import read_json, write_json


def load_registry(path: str | Path = MODEL_REGISTRY) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {"_meta": {"description": "EarlyDX model registry."}, "models": {}}
    return read_json(path)


def upsert_model(entry: dict[str, Any], path: str | Path = MODEL_REGISTRY) -> None:
    """Insert or replace a model entry keyed by ``model_id``."""
    required = {
        "model_id", "disease", "version", "algorithm", "dataset",
        "training_date", "features", "metrics", "status", "artifact_path",
    }
    missing = required - entry.keys()
    if missing:
        raise ValueError(f"model registry entry missing fields: {sorted(missing)}")

    reg = load_registry(path)
    reg.setdefault("models", {})[entry["model_id"]] = entry
    write_json(path, reg)


def get_model(model_id: str, path: str | Path = MODEL_REGISTRY) -> dict[str, Any] | None:
    return load_registry(path).get("models", {}).get(model_id)
