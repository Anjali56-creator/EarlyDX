"""Global configuration for EarlyDX ML pipelines.

Everything here is deliberately static so that runs are reproducible.
"""
from __future__ import annotations

from pathlib import Path

# ── Reproducibility ────────────────────────────────────────────────────────────
RANDOM_STATE: int = 42

# ── Splits ────────────────────────────────────────────────────────────────────
# Stratified train / validation / test.
TEST_SIZE: float = 0.20
VAL_SIZE: float = 0.20  # fraction of the *full* dataset (not of the train part)
CV_FOLDS: int = 5

# ── Risk-level threshold policy ───────────────────────────────────────────────
# HIGH  cut = smallest score whose validation sensitivity >= TARGET_SENSITIVITY
# LOW   cut = largest  score whose validation specificity >= TARGET_SPECIFICITY
# scores between the two cuts are MODERATE.
TARGET_SENSITIVITY: float = 0.85
TARGET_SPECIFICITY: float = 0.85

# ── Paths ─────────────────────────────────────────────────────────────────────
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = REPO_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
METADATA_DIR: Path = DATA_DIR / "metadata"
DATASET_REGISTRY: Path = DATA_DIR / "dataset_registry.json"

MODELS_DIR: Path = REPO_ROOT / "models"
MODEL_REGISTRY: Path = MODELS_DIR / "model_registry.json"

DOC_MODEL_CARD: Path = REPO_ROOT / "MODEL_CARD.md"

DISCLAIMER: str = "This is a risk assessment and not a medical diagnosis."
