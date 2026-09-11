"""Train the diabetes model via the shared framework.

Run:  python -m ml.diabetes.train
"""
from __future__ import annotations

import json

from ml.common.framework import run_training
from ml.diabetes.spec import SPEC


def main() -> int:
    meta = run_training(SPEC)
    tm = meta["test_metrics"]
    print("\n=== diabetes-v1 test-set summary ===")
    print(json.dumps(
        {k: tm[k] for k in
         ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")},
        indent=2,
    ))
    print("confusion_matrix:", tm["confusion_matrix"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
