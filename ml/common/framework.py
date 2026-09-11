"""Reusable, disease-agnostic training framework.

A disease pipeline is described by a :class:`DiseaseSpec`. ``run_training(spec)``
then performs the identical sequence for every disease:

    load -> validate -> split -> baseline vs candidates -> CV -> calibration
    -> validation-derived thresholds -> single held-out test evaluation
    -> permutation importance -> serialise artifact + schema + metadata
    -> model registry entry -> MODEL_CARD.md section

Nothing here fabricates data or metrics: the spec's ``load`` must return a real
DataFrame, and every number written comes from this executed run.
"""
from __future__ import annotations

import datetime as dt
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import (
    GridSearchCV,
    GroupShuffleSplit,
    StratifiedGroupKFold,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline

from ml.common.config import (
    CV_FOLDS,
    DATASET_REGISTRY,
    DOC_MODEL_CARD,
    MODELS_DIR,
    RANDOM_STATE,
    REPO_ROOT,
    TARGET_SENSITIVITY,
    TARGET_SPECIFICITY,
    TEST_SIZE,
    VAL_SIZE,
    DISCLAIMER,
)
from ml.common.evaluation import (
    binary_metrics,
    reliability,
    threshold_for_sensitivity,
    threshold_for_specificity,
)
from ml.common.model_registry import upsert_model
from ml.common.preprocessing import ZeroToNaN, build_preprocessor
from ml.common.utilities import (
    get_logger,
    read_json,
    replace_between_markers,
    set_global_seed,
    sha256_file,
    write_json,
)

LOG = get_logger("ml.framework")

MIN_AUC_GAIN_OVER_BASELINE = 0.01
TARGET = "target"


@dataclass
class FeatureMeta:
    name: str
    kind: str = "number"  # "number" | "categorical"
    unit: str = ""
    min: float | None = None
    max: float | None = None
    options: list[str] | None = None


@dataclass
class DiseaseSpec:
    key: str
    display_name: str
    version: str
    dataset_name: str
    dataset_registry_key: str
    # load() -> DataFrame with feature columns + an integer 0/1 column named TARGET
    load: Callable[[], pd.DataFrame]
    numeric_features: list[str]
    categorical_features: list[str] = field(default_factory=list)
    zero_as_missing: list[str] = field(default_factory=list)
    feature_meta: dict[str, FeatureMeta] = field(default_factory=dict)
    group_column: str | None = None  # for grouped split/CV (e.g. Parkinson's subject)
    drop_exact_duplicates: bool = False
    min_roc_auc: float = 0.65  # below this test ROC-AUC the model is not promoted
    positive_label: str = "1"  # human description of what target==1 means
    limitations: str = ""
    ethical_note: str = ""
    notes: str = ""

    @property
    def model_id(self) -> str:
        return f"{self.key}-{self.version}"

    @property
    def features(self) -> list[str]:
        return list(self.numeric_features) + list(self.categorical_features)

    def artifact_path(self) -> Path:
        return MODELS_DIR / f"{self.model_id}.joblib"

    def dir(self) -> Path:
        return REPO_ROOT / "ml" / self.key


# ── pipeline construction ────────────────────────────────────────────────────
def build_pipeline(spec: DiseaseSpec, estimator) -> Pipeline:
    steps: list = []
    if spec.zero_as_missing:
        steps.append(("zeros", ZeroToNaN(spec.zero_as_missing)))
    pre: ColumnTransformer = build_preprocessor(
        spec.numeric_features, spec.categorical_features, scale=True
    )
    steps.append(("pre", pre))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def candidate_estimators() -> dict[str, Any]:
    return {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            class_weight="balanced",
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "GradientBoostingClassifier": GradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            random_state=RANDOM_STATE
        ),
    }


BASELINE_NAME = "LogisticRegression"

# Small, deliberately bounded grids: this is a screening-prototype dataset
# (hundreds to low-thousands of rows per disease), not a setting that
# benefits from an expensive search, and every grid is scored with
# cross-validation on the *training* split only (see ``_tune_candidate``) so
# tuning never sees the validation or test split.
HYPERPARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "LogisticRegression": {"clf__C": [0.01, 0.1, 1.0, 10.0]},
    "RandomForestClassifier": {
        "clf__n_estimators": [200, 300, 500],
        "clf__max_depth": [4, 8, 12, None],
        "clf__min_samples_leaf": [1, 3, 5],
    },
    "GradientBoostingClassifier": {
        "clf__n_estimators": [100, 200],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [2, 3],
    },
    "HistGradientBoostingClassifier": {
        "clf__max_iter": [100, 200],
        "clf__learning_rate": [0.03, 0.1],
        "clf__max_depth": [3, 5, None],
    },
}


def _tune_candidate(
    spec: DiseaseSpec, name: str, estimator, train: pd.DataFrame, feats: list[str]
) -> tuple[Any, dict[str, Any]]:
    """Hyperparameter-search ``estimator`` with stratified CV on ``train`` only.

    Returns an *unfitted* estimator carrying the winning hyperparameters (so
    callers can still ``clone()`` it downstream exactly like the untuned
    candidates), plus a small report of what was searched and found. Neither
    the validation nor the test split is touched here.
    """
    grid = HYPERPARAM_GRIDS.get(name)
    if not grid:
        return clone(estimator), {"searched": False}

    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    search = GridSearchCV(
        build_pipeline(spec, clone(estimator)),
        grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    search.fit(train[feats], train[TARGET].astype(int))

    clf_params = {
        k.split("__", 1)[1]: v for k, v in search.best_params_.items() if k.startswith("clf__")
    }
    tuned = clone(estimator)
    tuned.set_params(**clf_params)
    report = {
        "searched": True,
        "cv_folds": CV_FOLDS,
        "best_params": clf_params,
        "cv_roc_auc": float(search.best_score_),
    }
    return tuned, report


# ── splitting ───────────────────────────────────────────────────────────────
def _split(spec: DiseaseSpec, df: pd.DataFrame):
    y = df[TARGET].astype(int)
    if spec.group_column and spec.group_column in df.columns:
        groups = df[spec.group_column]
        gss1 = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_STATE)
        tv_idx, te_idx = next(gss1.split(df, y, groups))
        train_val, test = df.iloc[tv_idx], df.iloc[te_idx]
        val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
        gss2 = GroupShuffleSplit(
            n_splits=1, test_size=val_fraction, random_state=RANDOM_STATE
        )
        tr_idx, va_idx = next(
            gss2.split(train_val, train_val[TARGET], train_val[spec.group_column])
        )
        train, val = train_val.iloc[tr_idx], train_val.iloc[va_idx]
        return train, val, test, train_val
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[TARGET],
        random_state=RANDOM_STATE,
    )
    return train, val, test, train_val


def _cv_splitter(spec: DiseaseSpec):
    if spec.group_column:
        # fewer folds when grouping, to reduce single-class folds on small data
        return StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=RANDOM_STATE)
    return StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)


def _cross_validate(spec: DiseaseSpec, estimator, frame: pd.DataFrame) -> dict[str, Any]:
    from sklearn.metrics import (
        average_precision_score,
        f1_score,
        recall_score,
        roc_auc_score,
    )

    X = frame[spec.features]
    y = frame[TARGET].astype(int).to_numpy()
    groups = frame[spec.group_column].to_numpy() if spec.group_column else None
    splitter = _cv_splitter(spec)
    per_fold: list[dict[str, float]] = []
    skipped_folds = 0
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[te])) < 2:
            # a grouped fold can be single-class on tiny datasets; ranking
            # metrics are undefined there, so that fold is not scored.
            skipped_folds += 1
            continue
        est = clone(estimator)
        est.fit(X.iloc[tr], y[tr])
        prob = est.predict_proba(X.iloc[te])[:, 1]
        pred = (prob >= 0.5).astype(int)
        per_fold.append(
            {
                "roc_auc": float(roc_auc_score(y[te], prob)),
                "pr_auc": float(average_precision_score(y[te], prob)),
                "recall_sensitivity": float(recall_score(y[te], pred, zero_division=0)),
                "f1": float(f1_score(y[te], pred, zero_division=0)),
            }
        )
    if not per_fold:
        raise ValueError(f"[{spec.key}] every CV fold was single-class; cannot cross-validate")
    keys = per_fold[0].keys()
    summary = {
        k: {
            "mean": float(np.mean([f[k] for f in per_fold])),
            "std": float(np.std([f[k] for f in per_fold])),
        }
        for k in keys
    }
    return {
        "folds_requested": CV_FOLDS,
        "folds_scored": len(per_fold),
        "folds_skipped_single_class": skipped_folds,
        "per_fold": per_fold,
        "summary": summary,
    }


# ── main entry point ────────────────────────────────────────────────────────
def run_training(spec: DiseaseSpec) -> dict[str, Any]:
    set_global_seed(RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = spec.load()
    _validate(spec, df)
    if spec.drop_exact_duplicates:
        before = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        LOG.info("[%s] dropped %d exact-duplicate rows", spec.key, before - len(df))
    LOG.info(
        "[%s] loaded %d rows, target positives=%d (%.1f%%)",
        spec.key, len(df), int(df[TARGET].sum()), 100 * df[TARGET].mean(),
    )
    _update_dataset_registry(spec, df)

    train, val, test, train_val = _split(spec, df)
    LOG.info(
        "[%s] split train=%d val=%d test=%d", spec.key, len(train), len(val), len(test)
    )

    feats = spec.features
    val_y = val[TARGET].astype(int).to_numpy()
    test_y = test[TARGET].astype(int).to_numpy()

    # 1. hyperparameter-tune each candidate with CV on the training split only,
    #    then evaluate the tuned candidate once on validation for comparison.
    candidate_val: dict[str, dict[str, Any]] = {}
    tuned_estimators: dict[str, Any] = {}
    tuning_report: dict[str, Any] = {}
    for name, est in candidate_estimators().items():
        tuned, report = _tune_candidate(spec, name, est, train, feats)
        tuned_estimators[name] = tuned
        tuning_report[name] = report

        pipe = build_pipeline(spec, clone(tuned))
        pipe.fit(train[feats], train[TARGET].astype(int))
        prob = pipe.predict_proba(val[feats])[:, 1]
        candidate_val[name] = binary_metrics(val_y, prob, threshold=0.5)
        LOG.info(
            "[%s]   %-28s val ROC-AUC=%.4f recall=%.4f%s",
            spec.key, name, candidate_val[name]["roc_auc"],
            candidate_val[name]["recall_sensitivity"],
            f" tuned={report['best_params']}" if report.get("searched") else " (no grid; default params)",
        )

    # 2. selection: keep the interpretable baseline unless clearly beaten
    baseline_auc = candidate_val[BASELINE_NAME]["roc_auc"]
    best_name, best_auc = BASELINE_NAME, baseline_auc
    for name, rep in candidate_val.items():
        if name == BASELINE_NAME:
            continue
        if rep["roc_auc"] >= baseline_auc + MIN_AUC_GAIN_OVER_BASELINE and rep["roc_auc"] > best_auc:
            best_name, best_auc = name, rep["roc_auc"]
    LOG.info("[%s] selected %s (val ROC-AUC=%.4f, baseline=%.4f)",
             spec.key, best_name, best_auc, baseline_auc)
    chosen = tuned_estimators[best_name]

    # 3. cross-validation on train+val
    cv = _cross_validate(spec, build_pipeline(spec, clone(chosen)), train_val)
    LOG.info("[%s] CV ROC-AUC=%.4f +/- %.4f", spec.key,
             cv["summary"]["roc_auc"]["mean"], cv["summary"]["roc_auc"]["std"])

    # 4. calibration decided on validation Brier
    uncal = build_pipeline(spec, clone(chosen))
    uncal.fit(train[feats], train[TARGET].astype(int))
    brier_uncal = float(brier_score_loss(val_y, uncal.predict_proba(val[feats])[:, 1]))
    calib_results = {"none": brier_uncal}
    for method in ("sigmoid", "isotonic"):
        try:
            cc = CalibratedClassifierCV(
                build_pipeline(spec, clone(chosen)), method=method, cv=CV_FOLDS
            )
            cc.fit(train[feats], train[TARGET].astype(int))
            calib_results[method] = float(
                brier_score_loss(val_y, cc.predict_proba(val[feats])[:, 1])
            )
        except Exception as exc:  # noqa: BLE001
            LOG.warning("[%s] calibration %s failed: %s", spec.key, method, exc)
    best_calib = min(calib_results, key=calib_results.get)
    calibration_method = (
        "none" if best_calib == "none" or calib_results[best_calib] >= brier_uncal - 1e-4
        else best_calib
    )
    LOG.info("[%s] calibration=%s brier=%s", spec.key, calibration_method, calib_results)

    def _make_final(frame: pd.DataFrame):
        if calibration_method == "none":
            m = build_pipeline(spec, clone(chosen))
        else:
            m = CalibratedClassifierCV(
                build_pipeline(spec, clone(chosen)), method=calibration_method, cv=CV_FOLDS
            )
        m.fit(frame[feats], frame[TARGET].astype(int))
        return m

    # 5. thresholds from validation probabilities
    thr_model = _make_final(train)
    val_prob = thr_model.predict_proba(val[feats])[:, 1]
    low_cut = threshold_for_sensitivity(val_y, val_prob, TARGET_SENSITIVITY)
    high_cut = threshold_for_specificity(val_y, val_prob, TARGET_SPECIFICITY)
    policy = (
        f"HIGH if score >= {high_cut:.4f} (largest validation threshold with "
        f"specificity >= {TARGET_SPECIFICITY}); LOW if score < {low_cut:.4f} "
        f"(smallest validation threshold with sensitivity >= {TARGET_SENSITIVITY}); "
        "MODERATE otherwise."
    )
    if low_cut > high_cut:
        mid = float(np.median(val_prob))
        if mid <= 0.02 or mid >= 0.98:
            # near-separable data: the score distribution is bimodal at 0/1, so a
            # median cut is meaningless. Use a plain 0.5 decision boundary.
            low_cut = high_cut = 0.5
            policy = (
                "the validation score distribution is near-separable (bimodal at "
                "0/1), so sensitivity/specificity target thresholds are not "
                "informative; using a plain 0.5 cut (>= is HIGH, otherwise LOW; "
                "MODERATE unused). Treat the near-perfect metrics with caution."
            )
        else:
            low_cut = high_cut = mid
            policy = (
                "validation sensitivity and specificity targets could not both be "
                f"satisfied without the bands crossing; single cut at the validation "
                f"median score {mid:.4f} (>= is HIGH, otherwise LOW; MODERATE unused)."
            )
    LOG.info("[%s] thresholds low=%.4f high=%.4f", spec.key, low_cut, high_cut)

    # documentation only: how sensitivity/specificity/precision/FN/FP trade off
    # across thresholds, computed on the validation split (never test).
    threshold_grid = sorted({round(t, 2) for t in np.arange(0.05, 1.0, 0.05)} | {0.5, low_cut, high_cut})
    threshold_analysis = []
    for t in threshold_grid:
        m = binary_metrics(val_y, val_prob, threshold=t)
        threshold_analysis.append(
            {
                "threshold": round(t, 4),
                "sensitivity": m["recall_sensitivity"],
                "specificity": m["specificity"],
                "precision": m["precision"],
                "false_negatives": m["confusion_matrix"]["fn"],
                "false_positives": m["confusion_matrix"]["fp"],
            }
        )

    # 6. refit on train+val, evaluate ONCE on test
    final_model = _make_final(train_val)
    test_prob = final_model.predict_proba(test[feats])[:, 1]
    test_metrics = binary_metrics(test_y, test_prob, threshold=0.5)
    test_metrics["calibration"] = reliability(test_y, test_prob)
    test_at_low = binary_metrics(test_y, test_prob, threshold=low_cut)
    test_at_high = binary_metrics(test_y, test_prob, threshold=high_cut)
    LOG.info(
        "[%s] TEST ROC-AUC=%.4f PR-AUC=%.4f screening_recall@low=%.4f",
        spec.key, test_metrics["roc_auc"], test_metrics["pr_auc"],
        test_at_low["recall_sensitivity"],
    )

    # a bounded sample of held-out TEST rows (never train/val) with the
    # model's own prediction alongside the real label, for a research
    # "validation mode" that lets a user compare a known-labelled record
    # against what the model actually output — this is not used for any
    # metric above, only stored for later display.
    def _risk_level(score: float) -> str:
        if score >= high_cut:
            return "HIGH"
        if score < low_cut:
            return "LOW"
        return "MODERATE"

    sample_n = min(20, len(test))
    sample_idx = test.index[:sample_n]
    validation_samples = [
        {
            "features": {f: (float(v) if isinstance(v, (int, float, np.floating, np.integer)) else str(v))
                         for f, v in test.loc[i, feats].items()},
            "actual_outcome": int(test.loc[i, TARGET]),
            "predicted_probability": round(float(p), 4),
            "predicted_outcome": int(p >= 0.5),
            "predicted_risk_level": _risk_level(float(p)),
        }
        for i, p in zip(sample_idx, test_prob[: sample_n])
    ]

    # 7. permutation importance (model-agnostic) on the test set
    pim = permutation_importance(
        final_model, test[feats], test_y,
        scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance = sorted(
        (
            {"feature": f, "importance": float(m), "std": float(s)}
            for f, m, s in zip(feats, pim.importances_mean, pim.importances_std)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )

    # reference values for the explainability service
    feature_reference: dict[str, Any] = {}
    for f in spec.numeric_features:
        feature_reference[f] = {
            "median": float(train_val[f].median()),
            "p25": float(train_val[f].quantile(0.25)),
            "p75": float(train_val[f].quantile(0.75)),
        }
    for f in spec.categorical_features:
        feature_reference[f] = {"mode": str(train_val[f].mode().iloc[0])}

    # 8. serialise artifact
    artifact_path = spec.artifact_path()
    joblib.dump(
        {
            "model": final_model,
            "features": feats,
            "model_id": spec.model_id,
            "calibrated": calibration_method != "none",
            "thresholds": {"low_cut": low_cut, "high_cut": high_cut},
            "feature_reference": feature_reference,
        },
        artifact_path,
    )
    artifact_sha = sha256_file(artifact_path)

    # 9. feature schema
    schema = {
        "disease": spec.key,
        "model_id": spec.model_id,
        "target": TARGET,
        "positive_label": spec.positive_label,
        "required_features": feats,
        "feature_details": {},
    }
    for f in feats:
        meta = spec.feature_meta.get(f, FeatureMeta(f))
        detail: dict[str, Any] = {"type": meta.kind, "unit": meta.unit, "required": True}
        if meta.kind == "categorical":
            opts = meta.options or sorted(map(str, df[f].dropna().unique().tolist()))
            detail["options"] = opts
        else:
            detail["min"] = meta.min if meta.min is not None else float(df[f].min())
            detail["max"] = meta.max if meta.max is not None else float(df[f].max())
            if f in spec.zero_as_missing:
                # The trained pipeline's ZeroToNaN step (see ml/common/preprocessing)
                # treats an exact 0 in this column as an encoded-missing value and
                # median-imputes it, so 0 is a valid input regardless of the
                # plausible min/max below — it is never fed to the model as a raw 0.
                detail["zero_is_missing"] = True
        schema["feature_details"][f] = detail
    write_json(spec.dir() / "feature_schema.json", schema)

    # 10. model metadata
    training_date = dt.date.today().isoformat()
    metadata = {
        "model_id": spec.model_id,
        "disease": spec.key,
        "display_name": spec.display_name,
        "version": spec.version,
        "algorithm": best_name,
        "baseline_algorithm": BASELINE_NAME,
        "selection_rule": (
            f"prefer {BASELINE_NAME}; a candidate is chosen only if its validation "
            f"ROC-AUC exceeds the baseline by >= {MIN_AUC_GAIN_OVER_BASELINE}"
        ),
        "random_state": RANDOM_STATE,
        "training_date": training_date,
        "dataset": {
            "name": spec.dataset_name,
            "registry_key": spec.dataset_registry_key,
        },
        "split": {
            "scheme": "grouped" if spec.group_column else "stratified",
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        },
        "target_positive_rate": float(df[TARGET].mean()),
        "hyperparameter_tuning": {
            "method": "GridSearchCV, scoring=roc_auc, fit on the training split only",
            "cv_folds": CV_FOLDS,
            "by_candidate": tuning_report,
        },
        "validation_metrics_by_candidate": candidate_val,
        "cross_validation": cv,
        "calibration": {"method": calibration_method, "validation_brier": calib_results},
        "risk_thresholds": {
            "low_cut": low_cut,
            "high_cut": high_cut,
            "policy": policy,
            "target_sensitivity": TARGET_SENSITIVITY,
            "target_specificity": TARGET_SPECIFICITY,
        },
        "threshold_analysis": {
            "computed_on": "validation split",
            "decision_threshold_used_for_test_metrics": 0.5,
            "sweep": threshold_analysis,
        },
        "test_metrics": test_metrics,
        "test_metrics_at_low_threshold_screening": test_at_low,
        "test_metrics_at_high_threshold": test_at_high,
        "feature_reference": feature_reference,
        "permutation_importance": importance,
        "validation_samples": {
            "note": "a bounded sample of held-out TEST rows, never used in training/tuning/"
                    "calibration/threshold selection, for research validation display only.",
            "samples": validation_samples,
        },
        "artifact": {
            "path": artifact_path.relative_to(REPO_ROOT).as_posix(),
            "sha256": artifact_sha,
        },
        "output_field": "risk_score",
        "disclaimer": DISCLAIMER,
        "limitations": spec.limitations,
        "ethical_note": spec.ethical_note,
        "notes": spec.notes or (
            "Score is a model output called risk_score; it is a calibrated "
            "probability only when calibration.method != 'none'."
        ),
    }
    write_json(spec.dir() / "model_metadata.json", metadata)

    # 11. model registry — a weak model is registered but not promoted to active
    test_auc = test_metrics["roc_auc"]
    if test_auc < spec.min_roc_auc:
        status = "experimental"
        status_reason = (
            f"test ROC-AUC {test_auc:.3f} is below the promotion bar "
            f"{spec.min_roc_auc:.2f}; not served for prediction"
        )
        LOG.warning("[%s] %s", spec.key, status_reason)
    else:
        status = "active"
        status_reason = ""
    metadata["status"] = status
    metadata["status_reason"] = status_reason
    write_json(spec.dir() / "model_metadata.json", metadata)
    upsert_model(
        {
            "model_id": spec.model_id,
            "disease": spec.key,
            "version": spec.version,
            "algorithm": best_name,
            "dataset": spec.dataset_registry_key,
            "training_date": training_date,
            "features": feats,
            "metrics": {
                "test_roc_auc": round(test_metrics["roc_auc"], 4),
                "test_pr_auc": round(test_metrics["pr_auc"], 4),
                "test_recall_sensitivity": round(test_metrics["recall_sensitivity"], 4),
                "test_specificity": round(test_metrics["specificity"], 4),
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "cv_roc_auc_mean": round(cv["summary"]["roc_auc"]["mean"], 4),
            },
            "calibration": calibration_method,
            "status": status,
            "status_reason": status_reason,
            "artifact_path": artifact_path.relative_to(REPO_ROOT).as_posix(),
        }
    )

    # 12. MODEL_CARD.md section (idempotent, marker-bounded)
    _write_model_card_section(spec, metadata)

    LOG.info("[%s] done: %s", spec.key, spec.model_id)
    return metadata


def _update_dataset_registry(spec: DiseaseSpec, df: pd.DataFrame) -> None:
    """Write measured facts (not assumptions) for this dataset into the registry.

    Source / licence / URL fields are left to the download scripts; here we only
    record what can be measured from the real file.
    """
    if not DATASET_REGISTRY.exists():
        return
    reg = read_json(DATASET_REGISTRY)
    key = spec.dataset_registry_key
    entry = reg.setdefault(key, {"disease": key})
    counts = df[TARGET].value_counts().sort_index()
    nan_missing = {c: int(df[c].isna().sum()) for c in spec.features if c in df.columns}
    entry.update(
        {
            "disease": key,
            "dataset_name": entry.get("dataset_name") or spec.dataset_name,
            "records": int(len(df)),
            "features": spec.features,
            "target": f"{TARGET} (1 = {spec.positive_label})",
            "class_distribution": {str(k): int(v) for k, v in counts.items()},
            "missing_values": {c: v for c, v in nan_missing.items() if v > 0},
            "limitations": spec.limitations or entry.get("limitations", "unknown"),
            "status": "verified",
        }
    )
    reg.setdefault("_meta", {})["updated"] = dt.datetime.now(dt.timezone.utc).isoformat()
    write_json(DATASET_REGISTRY, reg)


def _validate(spec: DiseaseSpec, df: pd.DataFrame) -> None:
    problems: list[str] = []
    if df.empty:
        problems.append("empty dataframe")
    if TARGET not in df.columns:
        problems.append(f"missing '{TARGET}' column")
    else:
        bad = set(pd.unique(df[TARGET].dropna())) - {0, 1}
        if bad:
            problems.append(f"target not binary 0/1: found {sorted(bad)}")
    for f in spec.features:
        if f not in df.columns:
            problems.append(f"missing feature column '{f}'")
    if problems:
        raise ValueError(f"[{spec.key}] dataset validation failed: " + "; ".join(problems))


def _ensure_card_section(spec: DiseaseSpec) -> None:
    begin, end = f"<!-- BEGIN {spec.model_id} -->", f"<!-- END {spec.model_id} -->"
    text = DOC_MODEL_CARD.read_text(encoding="utf-8")
    if begin in text:
        return
    anchor = "## Diseases without a model in V1"
    block = f"\n## {spec.model_id}\n\n{begin}\n_pending first training run._\n{end}\n\n---\n\n"
    if anchor in text:
        text = text.replace(anchor, block + anchor, 1)
    else:
        text = text.rstrip() + "\n\n" + block
    DOC_MODEL_CARD.write_text(text, encoding="utf-8")


def _write_model_card_section(spec: DiseaseSpec, meta: dict[str, Any]) -> None:
    _ensure_card_section(spec)
    tm = meta["test_metrics"]
    cm = tm["confusion_matrix"]
    cv = meta["cross_validation"]["summary"]
    imp = meta["permutation_importance"]
    low = meta["test_metrics_at_low_threshold_screening"]
    body = [
        f"- **Status:** trained {meta['training_date']} (`random_state={meta['random_state']}`).",
        f"- **Disease / target:** {meta['display_name']} — positive class = {spec.positive_label}.",
        f"- **Dataset:** {meta['dataset']['name']} (registry key `{meta['dataset']['registry_key']}`). "
        "See `DATASETS.md` and `data/metadata/`.",
        f"- **Algorithm:** {meta['algorithm']} (baseline {meta['baseline_algorithm']}; {meta['selection_rule']}).",
        f"- **Split:** {meta['split']['scheme']} "
        f"{meta['split']['sizes']['train']}/{meta['split']['sizes']['val']}/{meta['split']['sizes']['test']} "
        f"train/validation/test; dataset positive rate {meta['target_positive_rate']:.3f}.",
        f"- **Cross-validation ({meta['cross_validation'].get('folds_scored', '?')} folds scored):** "
        f"ROC-AUC {cv['roc_auc']['mean']:.3f} +/- {cv['roc_auc']['std']:.3f}; "
        f"PR-AUC {cv['pr_auc']['mean']:.3f} +/- {cv['pr_auc']['std']:.3f}; "
        f"recall {cv['recall_sensitivity']['mean']:.3f} +/- {cv['recall_sensitivity']['std']:.3f}.",
        f"- **Test metrics (threshold 0.5):** accuracy {tm['accuracy']:.3f}, precision {tm['precision']:.3f}, "
        f"recall/sensitivity {tm['recall_sensitivity']:.3f}, specificity {tm['specificity']:.3f}, "
        f"F1 {tm['f1']:.3f}, ROC-AUC {tm['roc_auc']:.3f}, PR-AUC {tm['pr_auc']:.3f}.",
        f"- **Test confusion matrix:** TN {cm['tn']}, FP {cm['fp']}, FN {cm['fn']}, TP {cm['tp']} (n={tm['n']}).",
        f"- **Test at screening cut ({meta['risk_thresholds']['low_cut']:.3f}):** "
        f"recall {low['recall_sensitivity']:.3f}, specificity {low['specificity']:.3f}.",
        f"- **Calibration:** {meta['calibration']['method']} "
        f"(validation Brier {json.dumps({k: round(v, 4) for k, v in meta['calibration']['validation_brier'].items()})}).",
        f"- **Risk-level thresholds:** {meta['risk_thresholds']['policy']}",
        "- **Top permutation-importance features:** "
        + ", ".join(f"{i['feature']} ({i['importance']:.3f})" for i in imp[:4]) + ".",
        f"- **Limitations:** {meta['limitations'] or 'see DATASETS.md / LIMITATIONS.md'}",
    ]
    if meta.get("ethical_note"):
        body.append(f"- **Ethical note:** {meta['ethical_note']}")
    begin, end = f"<!-- BEGIN {spec.model_id} -->", f"<!-- END {spec.model_id} -->"
    text = DOC_MODEL_CARD.read_text(encoding="utf-8")
    DOC_MODEL_CARD.write_text(
        replace_between_markers(text, begin, end, "\n".join(body)), encoding="utf-8"
    )
