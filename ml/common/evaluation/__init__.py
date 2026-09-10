"""Metric computation, cross-validation and calibration helpers.

All metrics are computed from real model predictions passed in by the caller.
Nothing here fabricates or defaults a score.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold


def binary_metrics(y_true, y_prob, threshold: float = 0.5) -> dict[str, Any]:
    """Full binary-classification metric set at a given decision threshold."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")

    return {
        "threshold": float(threshold),
        "n": int(y_true.size),
        "class_distribution": {
            "0": int((y_true == 0).sum()),
            "1": int((y_true == 1).sum()),
        },
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall_sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(specificity),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def cross_validate_auc(estimator, X, y, *, folds: int, seed: int) -> dict[str, Any]:
    """Stratified k-fold CV reporting per-fold ROC-AUC, PR-AUC, recall, F1."""
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    y = np.asarray(y).astype(int)
    per_fold: list[dict[str, float]] = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else X[train_idx]
        X_te = X.iloc[test_idx] if hasattr(X, "iloc") else X[test_idx]
        est = clone(estimator)
        est.fit(X_tr, y[train_idx])
        prob = est.predict_proba(X_te)[:, 1]
        pred = (prob >= 0.5).astype(int)
        per_fold.append(
            {
                "roc_auc": float(roc_auc_score(y[test_idx], prob)),
                "pr_auc": float(average_precision_score(y[test_idx], prob)),
                "recall_sensitivity": float(
                    recall_score(y[test_idx], pred, zero_division=0)
                ),
                "f1": float(f1_score(y[test_idx], pred, zero_division=0)),
            }
        )

    keys = per_fold[0].keys()
    summary = {
        k: {
            "mean": float(np.mean([f[k] for f in per_fold])),
            "std": float(np.std([f[k] for f in per_fold])),
        }
        for k in keys
    }
    return {"folds": folds, "per_fold": per_fold, "summary": summary}


def reliability(y_true, y_prob, *, n_bins: int = 10) -> dict[str, Any]:
    """Reliability-curve bins plus the Brier score."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi if hi < 1.0 else y_prob <= hi)
        if mask.sum() == 0:
            continue
        bins.append(
            {
                "bin": [float(lo), float(hi)],
                "count": int(mask.sum()),
                "mean_predicted": float(y_prob[mask].mean()),
                "observed_frequency": float(y_true[mask].mean()),
            }
        )
    return {"brier": float(brier_score_loss(y_true, y_prob)), "bins": bins}


def threshold_for_sensitivity(y_true, y_prob, target: float) -> float:
    """Smallest threshold whose sensitivity (recall) is still >= ``target``."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], y_prob, [1.0]]))
    best = 0.0
    for t in candidates:
        pred = (y_prob >= t).astype(int)
        sens = recall_score(y_true, pred, zero_division=0)
        if sens >= target:
            best = float(t)
    return best


def threshold_for_specificity(y_true, y_prob, target: float) -> float:
    """Largest threshold whose specificity is still >= ``target``."""
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    candidates = np.unique(np.concatenate([[0.0], y_prob, [1.0]]))
    best = 1.0
    for t in sorted(candidates, reverse=True):
        pred = (y_prob >= t).astype(int)
        tn, fp, _, _ = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        if spec >= target:
            best = float(t)
    return best
