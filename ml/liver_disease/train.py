"""Train the liver_disease model via the shared framework.

Run:  python -m ml.liver_disease.train
"""
from __future__ import annotations

import json

from ml.common.framework import run_training
from ml.liver_disease.spec import SPEC


def main() -> int:
    meta = run_training(SPEC)
    tm = meta["test_metrics"]
    print(json.dumps({
        "model_id": meta["model_id"],
        "algorithm": meta["algorithm"],
        "test": {k: round(tm[k], 4) for k in
                 ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")},
        "confusion_matrix": tm["confusion_matrix"],
        "calibration": meta["calibration"]["method"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
