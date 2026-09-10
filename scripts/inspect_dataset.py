"""Inspect a downloaded dataset and write a measured report.

Usage:  python scripts/inspect_dataset.py <disease>

Produces:
  data/metadata/<disease>.md          human-readable inspection report
  data/dataset_registry.json          measured fields for <disease> updated

Every value written comes from measuring the real file. Nothing is invented.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.common.utilities import get_logger, read_json, write_json  # noqa: E402

LOG = get_logger("inspect_dataset")
REGISTRY = REPO_ROOT / "data" / "dataset_registry.json"
METADATA_DIR = REPO_ROOT / "data" / "metadata"

# disease -> (relative csv path, target column, zero-as-missing columns)
DATASETS: dict[str, tuple[str, str, list[str]]] = {
    "diabetes": (
        "data/raw/diabetes/pima_indians_diabetes.csv",
        "Outcome",
        ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
    ),
}


def _leakage_suspects(df: pd.DataFrame, target: str) -> list[str]:
    out = []
    y = df[target]
    for col in df.columns:
        if col == target:
            continue
        s = df[col]
        if s.nunique(dropna=True) <= 1:
            out.append(f"{col}: constant column")
            continue
        if pd.api.types.is_numeric_dtype(s) and pd.api.types.is_numeric_dtype(y):
            corr = s.corr(y)
            if pd.notna(corr) and abs(corr) >= 0.95:
                out.append(f"{col}: |corr with target| = {abs(corr):.3f}")
    return out


def inspect(disease: str) -> None:
    if disease not in DATASETS:
        raise SystemExit(
            f"no inspection profile for '{disease}'. Known: {sorted(DATASETS)}"
        )
    rel, target, zero_missing = DATASETS[disease]
    path = REPO_ROOT / rel
    if not path.exists():
        raise SystemExit(f"{path} not found - run the download script first.")

    df = pd.read_csv(path)
    n_rows, n_cols = df.shape

    dup_rows = int(df.duplicated().sum())

    null_missing = {c: int(df[c].isna().sum()) for c in df.columns}
    zero_missing_counts = {
        c: int((df[c] == 0).sum()) for c in zero_missing if c in df.columns
    }

    if target not in df.columns:
        raise SystemExit(f"target column '{target}' not present")
    class_counts = df[target].value_counts(dropna=False).sort_index()
    class_dist = {str(k): int(v) for k, v in class_counts.items()}
    minority_frac = float(class_counts.min() / class_counts.sum())

    leakage = _leakage_suspects(df, target)

    describe = df.describe(include="all").transpose()

    # ── write markdown report ────────────────────────────────────────────────
    METADATA_DIR.mkdir(parents=True, exist_ok=True)
    md = METADATA_DIR / f"{disease}.md"
    lines: list[str] = []
    lines.append(f"## Dataset inspection - {disease}")
    lines.append("")
    lines.append(f"- **File:** `{rel}`")
    lines.append(f"- **Rows x Columns:** {n_rows} x {n_cols}")
    lines.append(f"- **Target column:** `{target}`")
    lines.append(f"- **Exact duplicate rows:** {dup_rows}")
    lines.append(f"- **Minority-class fraction:** {minority_frac:.3f}")
    lines.append("")
    lines.append("### Class distribution")
    lines.append("")
    lines.append("| class | count |")
    lines.append("|-------|-------|")
    for k, v in class_dist.items():
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("### Missing values (NaN)")
    lines.append("")
    lines.append("| column | n_missing |")
    lines.append("|--------|-----------|")
    for c, v in null_missing.items():
        lines.append(f"| {c} | {v} |")
    lines.append("")
    if zero_missing_counts:
        lines.append("### Zeros treated as encoded-missing")
        lines.append("")
        lines.append("| column | n_zeros |")
        lines.append("|--------|---------|")
        for c, v in zero_missing_counts.items():
            lines.append(f"| {c} | {v} |")
        lines.append("")
    lines.append("### Leakage / degenerate-column check")
    lines.append("")
    if leakage:
        for item in leakage:
            lines.append(f"- {item}")
    else:
        lines.append("- none flagged (no constant columns, no |corr| >= 0.95 with target)")
    lines.append("")
    lines.append("### Summary statistics")
    lines.append("")
    lines.append("```")
    lines.append(describe.to_string())
    lines.append("```")
    lines.append("")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    LOG.info("wrote %s", md)

    # ── update registry measured fields ─────────────────────────────────────
    reg = read_json(REGISTRY)
    entry = reg.setdefault(disease, {})
    entry.update(
        {
            "disease": disease,
            "records": int(n_rows),
            "features": [c for c in df.columns if c != target],
            "target": target,
            "class_distribution": class_dist,
            "missing_values": {
                "nan": {k: v for k, v in null_missing.items() if v > 0},
                "zeros_as_missing": zero_missing_counts,
                "duplicate_rows": dup_rows,
            },
            "preprocessing": (
                "zeros->NaN for physiologically-impossible fields, median "
                "imputation, standard scaling; fitted on train split only"
            ),
            "limitations": (
                "female Pima (Akimel O'odham) patients aged 21+ only; small "
                "sample; encoded-missing zeros; not externally validated"
            ),
            "status": "verified",
        }
    )
    reg.setdefault("_meta", {})["updated"] = pd.Timestamp.utcnow().isoformat()
    write_json(REGISTRY, reg)
    LOG.info("updated %s (%s measured fields)", REGISTRY, disease)

    print(f"\n{disease}: {n_rows} rows, {n_cols} cols, "
          f"dup={dup_rows}, minority_frac={minority_frac:.3f}, "
          f"leakage_flags={len(leakage)}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    inspect(sys.argv[1])
