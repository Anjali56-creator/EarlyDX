"""Download the real Pima Indians Diabetes Database into data/raw/diabetes/.

CRITICAL: this fetches a real public dataset. It never generates data. If every
mirror is unreachable it exits non-zero and writes nothing.

Provenance
----------
Originating institution : US National Institute of Diabetes and Digestive and
                          Kidney Diseases (NIDDK).
Historically distributed : UCI Machine Learning Repository ("Pima Indians
                          Diabetes Database"); the UCI entry has since been
                          retired from the active listing.
Fetched here from        : long-standing public mirrors that reproduce the exact
                          768-row file (verified by row count and column names).
Licence                  : commonly redistributed as CC0 / public-domain on
                          public mirrors; treat as "public, non-commercial
                          research use" and see DATASETS.md / LIMITATIONS.md for
                          the ethical context around the Pima (Akimel O'odham)
                          community.
"""
from __future__ import annotations

import io
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from ml.common.utilities import get_logger, read_json, sha256_file, write_json  # noqa: E402

LOG = get_logger("download_diabetes")

OUT_DIR = REPO_ROOT / "data" / "raw" / "diabetes"
OUT_FILE = OUT_DIR / "pima_indians_diabetes.csv"
REGISTRY = REPO_ROOT / "data" / "dataset_registry.json"

COLUMNS = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
    "BMI", "DiabetesPedigreeFunction", "Age", "Outcome",
]
EXPECTED_ROWS = 768

# (url, has_header)
MIRRORS = [
    ("https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv", True),
    ("https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv", False),
]


def _fetch(url: str) -> str:
    LOG.info("GET %s", url)
    with urllib.request.urlopen(url, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _normalise(raw_text: str, has_header: bool) -> str:
    import csv

    rows = list(csv.reader(io.StringIO(raw_text)))
    rows = [r for r in rows if r and any(c.strip() for c in r)]
    if has_header:
        header, data = rows[0], rows[1:]
        if [h.strip() for h in header] != COLUMNS:
            raise ValueError(f"unexpected header: {header}")
    else:
        data = rows
    if len(data) != EXPECTED_ROWS:
        raise ValueError(f"expected {EXPECTED_ROWS} rows, got {len(data)}")
    for r in data:
        if len(r) != len(COLUMNS):
            raise ValueError(f"row has {len(r)} fields, expected {len(COLUMNS)}")
    out = io.StringIO()
    w = csv.writer(out, lineterminator="\n")
    w.writerow(COLUMNS)
    w.writerows(data)
    return out.getvalue()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    last_err: Exception | None = None
    used_url = None
    for url, has_header in MIRRORS:
        try:
            text = _normalise(_fetch(url), has_header)
            used_url = url
            break
        except Exception as exc:  # noqa: BLE001
            LOG.warning("mirror failed (%s): %s", url, exc)
            last_err = exc
    else:
        LOG.error("all mirrors failed; wrote nothing")
        LOG.error("Dataset unavailable - awaiting user-provided dataset.")
        raise SystemExit(1) from last_err

    OUT_FILE.write_text(text, encoding="utf-8")
    digest = sha256_file(OUT_FILE)
    LOG.info("wrote %s (%d bytes) sha256=%s", OUT_FILE, OUT_FILE.stat().st_size, digest)

    # Record source facts in the registry. Measured facts (class distribution,
    # missing values) are filled in separately by scripts/inspect_dataset.py.
    reg = read_json(REGISTRY)
    reg["diabetes"].update(
        {
            "dataset_name": "Pima Indians Diabetes Database",
            "source": "NIDDK (US); historically UCI ML Repository; fetched from public mirror",
            "source_url": used_url,
            "license": "public / research use (CC0 on common mirrors); see DATASETS.md for ethical context",
            "records": EXPECTED_ROWS,
            "features": COLUMNS[:-1],
            "target": "Outcome",
            "sha256": digest,
            "status": "needs_review",
        }
    )
    write_json(REGISTRY, reg)
    LOG.info("updated %s (diabetes source fields)", REGISTRY)
    LOG.info("next: python scripts/inspect_dataset.py diabetes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
