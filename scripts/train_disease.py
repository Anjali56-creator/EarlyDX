"""Train one disease model by key:  python scripts/train_disease.py <key>

<key> is one of the ml/<key>/ packages that defines spec.SPEC.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

KEYS = [
    "diabetes", "heart_disease", "kidney_disease", "liver_disease", "breast_cancer",
    "parkinsons", "heart_failure", "thyroid", "obesity", "maternal_health",
]


def main() -> int:
    if len(sys.argv) != 2 or sys.argv[1] not in KEYS:
        raise SystemExit(f"usage: python scripts/train_disease.py <{'|'.join(KEYS)}>")
    key = sys.argv[1]
    spec = importlib.import_module(f"ml.{key}.spec").SPEC
    from ml.common.framework import run_training

    meta = run_training(spec)
    m = meta["test_metrics"]
    print(f"{meta['model_id']}: algo={meta['algorithm']} "
          f"ROC-AUC={m['roc_auc']:.4f} PR-AUC={m['pr_auc']:.4f} "
          f"recall={m['recall_sensitivity']:.4f} calib={meta['calibration']['method']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
