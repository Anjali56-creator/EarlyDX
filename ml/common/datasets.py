"""Helpers for fetching real public datasets into ``data/raw/<disease>/``.

These functions only download and cache files. They never synthesise data. If a
source is unreachable the caller gets an exception and nothing is written.
"""
from __future__ import annotations

import io
import urllib.request
import zipfile
from pathlib import Path

from ml.common.config import RAW_DIR
from ml.common.utilities import get_logger, sha256_file

LOG = get_logger("ml.datasets")

_UA = {"User-Agent": "EarlyDX/0.1 (research prototype; dataset fetch)"}


def raw_dir(disease: str) -> Path:
    d = RAW_DIR / disease
    d.mkdir(parents=True, exist_ok=True)
    return d


def _get(url: str, timeout: int = 60) -> bytes:
    LOG.info("GET %s", url)
    req = urllib.request.Request(url, headers=_UA)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def download_file(disease: str, url: str, filename: str, *, timeout: int = 60) -> Path:
    """Download ``url`` to ``data/raw/<disease>/<filename>`` unless already present."""
    out = raw_dir(disease) / filename
    if out.exists() and out.stat().st_size > 0:
        LOG.info("cached %s (%d bytes)", out, out.stat().st_size)
        return out
    data = _get(url, timeout=timeout)
    out.write_bytes(data)
    LOG.info("wrote %s (%d bytes) sha256=%s", out, len(data), sha256_file(out)[:16])
    return out


def download_from_mirrors(disease: str, mirrors: list[str], filename: str) -> Path:
    out = raw_dir(disease) / filename
    if out.exists() and out.stat().st_size > 0:
        return out
    last: Exception | None = None
    for url in mirrors:
        try:
            data = _get(url)
            out.write_bytes(data)
            LOG.info("wrote %s from %s", out, url)
            return out
        except Exception as exc:  # noqa: BLE001
            LOG.warning("mirror failed %s: %s", url, exc)
            last = exc
    raise RuntimeError(
        f"[{disease}] all mirrors failed for {filename}. "
        "Dataset unavailable - awaiting user-provided dataset."
    ) from last


def download_zip_member(
    disease: str, url: str, member_suffix: str, out_name: str, *, timeout: int = 90
) -> Path:
    """Download a zip and extract the first member whose name ends with
    ``member_suffix`` to ``data/raw/<disease>/<out_name>``."""
    out = raw_dir(disease) / out_name
    if out.exists() and out.stat().st_size > 0:
        return out
    blob = _get(url, timeout=timeout)
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(member_suffix.lower())]
        if not names:
            raise RuntimeError(
                f"[{disease}] no member ending {member_suffix!r} in {url}; "
                f"members: {zf.namelist()}"
            )
        data = zf.read(names[0])
    out.write_bytes(data)
    LOG.info("extracted %s (%s) -> %s (%d bytes)", names[0], url, out, len(data))
    return out
