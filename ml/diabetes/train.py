"""Train, evaluate, calibrate and serialise the diabetes risk model.

Run:  python -m ml.diabetes.train

Produces (all from this executed run - nothing hand-written):
  models/diabetes-v1.joblib
  ml/diabetes/feature_schema.json
  ml/diabetes/model_metadata.json
  models/model_registry.json           (diabetes-v1 entry upserted)
  MODEL_CARD.md                        (diabetes-v1 section regenerated)
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.inspection import permutation_importance
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import train_test_split

from ml.common.config import (
    CV_FOLDS,
    DOC_MODEL_CARD,
    MODELS_DIR,
    RANDOM_STATE,
    TARGET_SENSITIVITY,
    TARGET_SPECIFICITY,
    TEST_SIZE,
    VAL_SIZE,
    DISCLAIMER,
)
from ml.common.evaluation import (
    binary_metrics,
    cross_validate_auc,
    reliability,
    threshold_for_sensitivity,
    threshold_for_specificity,
)
from ml.common.model_registry import upsert_model
from ml.common.utilities import (
    get_logger,
    read_json,
    replace_between_markers,
    set_global_seed,
    sha256_file,
    write_json,
)
from ml.diabetes import DISEASE, DISPLAY_NAME, MODEL_ID
from ml.diabetes.evaluate import evaluate, predict_proba
from ml.diabetes.load import (
    FEATURE_RANGES,
    FEATURE_UNITS,
    FEATURES,
    RAW_FILE,
    TARGET,
    load_raw,
    validate,
)
from ml.diabetes.pipeline import BASELINE_NAME, build_pipeline, candidate_estimators

LOG = get_logger("diabetes.train")

ARTIFACT_PATH = MODELS_DIR / f"{MODEL_ID}.joblib"
SCHEMA_PATH = __file__.replace("train.py", "feature_schema.json")
META_PATH = __file__.replace("train.py", "model_metadata.json")

# A candidate must beat the interpretable baseline by at least this much
# validation ROC-AUC to be preferred.
MIN_AUC_GAIN_OVER_BASELINE = 0.01


def _split(df: pd.DataFrame):
    strat = df[TARGET]
    train_val, test = train_test_split(
        df, test_size=TEST_SIZE, stratify=strat, random_state=RANDOM_STATE
    )
    val_fraction = VAL_SIZE / (1.0 - TEST_SIZE)
    train, val = train_test_split(
        train_val,
        test_size=val_fraction,
        stratify=train_val[TARGET],
        random_state=RANDOM_STATE,
    )
    return train, val, test, train_val


def _fit(pipe, frame: pd.DataFrame):
    return pipe.fit(frame[FEATURES], frame[TARGET].astype(int))


def main() -> int:
    set_global_seed(RANDOM_STATE)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_raw()
    val_report = validate(df)
    LOG.info("dataset ok: %s", val_report)

    train, val, test, train_val = _split(df)
    LOG.info(
        "split sizes -> train=%d val=%d test=%d (positives: %d/%d/%d)",
        len(train), len(val), len(test),
        int(train[TARGET].sum()), int(val[TARGET].sum()), int(test[TARGET].sum()),
    )

    # ── 1. candidates on validation ────────────────────────────────────────
    candidate_val: dict[str, dict[str, Any]] = {}
    fitted_on_train: dict[str, Any] = {}
    for name, est in candidate_estimators().items():
        pipe = build_pipeline(clone(est))
        _fit(pipe, train)
        fitted_on_train[name] = pipe
        prob = predict_proba(pipe, val)
        candidate_val[name] = binary_metrics(
            val[TARGET].astype(int).to_numpy(), prob, threshold=0.5
        )
        LOG.info(
            "  %-28s val ROC-AUC=%.4f  PR-AUC=%.4f  recall=%.4f",
            name,
            candidate_val[name]["roc_auc"],
            candidate_val[name]["pr_auc"],
            candidate_val[name]["recall_sensitivity"],
        )

    # ── 2. model selection (favour interpretable baseline) ─────────────────
    baseline_auc = candidate_val[BASELINE_NAME]["roc_auc"]
    best_name = BASELINE_NAME
    best_auc = baseline_auc
    for name, rep in candidate_val.items():
        if name == BASELINE_NAME:
            continue
        if rep["roc_auc"] >= baseline_auc + MIN_AUC_GAIN_OVER_BASELINE and rep["roc_auc"] > best_auc:
            best_name, best_auc = name, rep["roc_auc"]
    LOG.info("selected: %s (val ROC-AUC=%.4f, baseline=%.4f)", best_name, best_auc, baseline_auc)

    chosen_estimator = candidate_estimators()[best_name]

    # ── 3. cross-validation on train+val ──────────────────────────────────
    cv = cross_validate_auc(
        build_pipeline(clone(chosen_estimator)),
        train_val[FEATURES],
        train_val[TARGET].astype(int).to_numpy(),
        folds=CV_FOLDS,
        seed=RANDOM_STATE,
    )
    LOG.info(
        "CV ROC-AUC=%.4f +/- %.4f | PR-AUC=%.4f +/- %.4f | recall=%.4f +/- %.4f",
        cv["summary"]["roc_auc"]["mean"], cv["summary"]["roc_auc"]["std"],
        cv["summary"]["pr_auc"]["mean"], cv["summary"]["pr_auc"]["std"],
        cv["summary"]["recall_sensitivity"]["mean"], cv["summary"]["recall_sensitivity"]["std"],
    )

    # ── 4. calibration decision (chosen on validation Brier) ──────────────
    uncal = build_pipeline(clone(chosen_estimator))
    _fit(uncal, train)
    val_y = val[TARGET].astype(int).to_numpy()
    brier_uncal = float(brier_score_loss(val_y, predict_proba(uncal, val)))

    calib_results: dict[str, float] = {"none": brier_uncal}
    calibrated_models: dict[str, Any] = {}
    for method in ("sigmoid", "isotonic"):
        cc = CalibratedClassifierCV(
            build_pipeline(clone(chosen_estimator)), method=method, cv=CV_FOLDS
        )
        cc.fit(train[FEATURES], train[TARGET].astype(int))
        b = float(brier_score_loss(val_y, cc.predict_proba(val[FEATURES])[:, 1]))
        calib_results[method] = b
        calibrated_models[method] = cc
        LOG.info("  calibration %-9s val Brier=%.4f", method, b)

    best_calib = min(calib_results, key=calib_results.get)
    if best_calib == "none" or calib_results[best_calib] >= brier_uncal - 1e-4:
        calibration_method = "none"
    else:
        calibration_method = best_calib
    LOG.info("calibration selected: %s (val Brier: uncal=%.4f)", calibration_method, brier_uncal)

    def _make_final(train_frame: pd.DataFrame):
        if calibration_method == "none":
            m = build_pipeline(clone(chosen_estimator))
            m.fit(train_frame[FEATURES], train_frame[TARGET].astype(int))
            return m
        m = CalibratedClassifierCV(
            build_pipeline(clone(chosen_estimator)), method=calibration_method, cv=CV_FOLDS
        )
        m.fit(train_frame[FEATURES], train_frame[TARGET].astype(int))
        return m

    # ── 5. risk-level thresholds from validation probabilities ────────────
    final_for_thresholds = _make_final(train)
    val_prob = (
        final_for_thresholds.predict_proba(val[FEATURES])[:, 1]
        if calibration_method != "none"
        else predict_proba(final_for_thresholds, val)
    )
    # low_cut: below it -> LOW  (smallest threshold whose val sensitivity >= target,
    #          i.e. we still catch >=85% of positives when we call everything below LOW)
    # high_cut: at/above it -> HIGH (largest threshold whose val specificity >= target)
    low_cut = threshold_for_sensitivity(val_y, val_prob, TARGET_SENSITIVITY)
    high_cut = threshold_for_specificity(val_y, val_prob, TARGET_SPECIFICITY)
    threshold_policy = (
        f"HIGH if score >= {high_cut:.4f} (largest validation threshold with "
        f"specificity >= {TARGET_SPECIFICITY}); LOW if score < {low_cut:.4f} "
        f"(smallest validation threshold with sensitivity >= {TARGET_SENSITIVITY}); "
        f"MODERATE otherwise."
    )
    if low_cut > high_cut:
        mid = float(np.median(val_prob))
        low_cut = high_cut = mid
        threshold_policy = (
            "validation sensitivity and specificity targets could not both be "
            f"satisfied without the bands crossing; fell back to a single cut at "
            f"the validation median score {mid:.4f} (score >= cut is HIGH, "
            "otherwise LOW; MODERATE is unused for this model version)."
        )
    LOG.info("thresholds -> low_cut=%.4f high_cut=%.4f", low_cut, high_cut)

    # ── 6. refit final model on train+val, evaluate ONCE on test ──────────
    final_model = _make_final(train_val)
    test_metrics = evaluate(final_model, test, threshold=0.5)
    test_y = test[TARGET].astype(int).to_numpy()
    test_prob = (
        final_model.predict_proba(test[FEATURES])[:, 1]
        if calibration_method != "none"
        else predict_proba(final_model, test)
    )
    test_at_high = binary_metrics(test_y, test_prob, threshold=high_cut)
    test_at_low = binary_metrics(test_y, test_prob, threshold=low_cut)
    LOG.info(
        "TEST ROC-AUC=%.4f PR-AUC=%.4f | @0.5 recall=%.4f | screening @low(%.3f) recall=%.4f spec=%.4f | @high(%.3f) prec=%.4f",
        test_metrics["roc_auc"], test_metrics["pr_auc"], test_metrics["recall_sensitivity"],
        low_cut, test_at_low["recall_sensitivity"], test_at_low["specificity"],
        high_cut, test_at_high["precision"],
    )

    # Reference values for the explainability service (direction of each feature
    # relative to the training-set centre). Measured, not assumed.
    feature_reference = {
        f: {
            "median": float(train_val[f].median()),
            "p25": float(train_val[f].quantile(0.25)),
            "p75": float(train_val[f].quantile(0.75)),
        }
        for f in FEATURES
    }

    # ── 7. permutation importance (model-agnostic) on the test set ────────
    pim = permutation_importance(
        final_model, test[FEATURES], test_y,
        scoring="roc_auc", n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1,
    )
    importance = sorted(
        (
            {"feature": f, "importance": float(m), "std": float(s)}
            for f, m, s in zip(FEATURES, pim.importances_mean, pim.importances_std)
        ),
        key=lambda d: d["importance"],
        reverse=True,
    )
    LOG.info("top features: %s", [i["feature"] for i in importance[:4]])

    # ── 8. serialise artifact ────────────────────────────────────────────
    joblib.dump(
        {
            "model": final_model,
            "features": FEATURES,
            "model_id": MODEL_ID,
            "calibrated": calibration_method != "none",
            "thresholds": {"low_cut": low_cut, "high_cut": high_cut},
            "feature_reference": feature_reference,
        },
        ARTIFACT_PATH,
    )
    artifact_sha = sha256_file(ARTIFACT_PATH)
    LOG.info("wrote %s sha256=%s", ARTIFACT_PATH, artifact_sha[:16])

    # ── 9. feature schema ───────────────────────────────────────────────
    schema = {
        "disease": DISEASE,
        "model_id": MODEL_ID,
        "target": TARGET,
        "required_features": FEATURES,
        "feature_details": {
            f: {
                "type": "number",
                "unit": FEATURE_UNITS[f],
                "min": FEATURE_RANGES[f][0],
                "max": FEATURE_RANGES[f][1],
                "required": True,
            }
            for f in FEATURES
        },
    }
    write_json(SCHEMA_PATH, schema)

    # ── 10. model metadata ─────────────────────────────────────────────
    training_date = dt.date.today().isoformat()
    metadata = {
        "model_id": MODEL_ID,
        "disease": DISEASE,
        "display_name": DISPLAY_NAME,
        "version": "v1",
        "algorithm": best_name,
        "baseline_algorithm": BASELINE_NAME,
        "selection_rule": (
            f"prefer {BASELINE_NAME}; a candidate is chosen only if its "
            f"validation ROC-AUC exceeds the baseline by >= {MIN_AUC_GAIN_OVER_BASELINE}"
        ),
        "random_state": RANDOM_STATE,
        "training_date": training_date,
        "dataset": {
            "name": "Pima Indians Diabetes Database",
            "file": RAW_FILE.relative_to(RAW_FILE.parents[3]).as_posix(),
            "sha256": sha256_file(RAW_FILE),
            "registry_key": DISEASE,
        },
        "split": {
            "scheme": "stratified train/val/test",
            "sizes": {"train": len(train), "val": len(val), "test": len(test)},
            "test_size": TEST_SIZE,
            "val_size": VAL_SIZE,
        },
        "validation_metrics_by_candidate": candidate_val,
        "cross_validation": cv,
        "calibration": {
            "method": calibration_method,
            "validation_brier": calib_results,
        },
        "risk_thresholds": {
            "low_cut": low_cut,
            "high_cut": high_cut,
            "policy": threshold_policy,
            "target_sensitivity": TARGET_SENSITIVITY,
            "target_specificity": TARGET_SPECIFICITY,
        },
        "test_metrics": test_metrics,
        "test_metrics_at_high_threshold": test_at_high,
        "test_metrics_at_low_threshold_screening": test_at_low,
        "feature_reference": feature_reference,
        "permutation_importance": importance,
        "artifact": {"path": ARTIFACT_PATH.relative_to(MODELS_DIR.parent).as_posix(), "sha256": artifact_sha},
        "output_field": "risk_score" if calibration_method == "none" else "calibrated_risk_score",
        "disclaimer": DISCLAIMER,
        "notes": (
            "Score is a model output. It is called risk_score and is NOT a "
            "calibrated probability unless calibration.method != 'none'."
        ),
    }
    write_json(META_PATH, metadata)
    LOG.info("wrote %s", META_PATH)

    # ── 11. model registry ─────────────────────────────────────────────
    upsert_model(
        {
            "model_id": MODEL_ID,
            "disease": DISEASE,
            "version": "v1",
            "algorithm": best_name,
            "dataset": "pima-indians-diabetes",
            "training_date": training_date,
            "features": FEATURES,
            "metrics": {
                "test_roc_auc": round(test_metrics["roc_auc"], 4),
                "test_pr_auc": round(test_metrics["pr_auc"], 4),
                "test_recall_sensitivity": round(test_metrics["recall_sensitivity"], 4),
                "test_specificity": round(test_metrics["specificity"], 4),
                "test_accuracy": round(test_metrics["accuracy"], 4),
                "cv_roc_auc_mean": round(cv["summary"]["roc_auc"]["mean"], 4),
            },
            "calibration": calibration_method,
            "status": "active",
            "artifact_path": ARTIFACT_PATH.relative_to(MODELS_DIR.parent).as_posix(),
        }
    )
    LOG.info("registry updated")

    # ── 12. regenerate the MODEL_CARD.md section ──────────────────────
    _write_model_card(metadata)

    print("\n=== diabetes-v1 test-set summary ===")
    print(json.dumps(
        {k: test_metrics[k] for k in
         ("accuracy", "precision", "recall_sensitivity", "specificity", "f1", "roc_auc", "pr_auc")},
        indent=2,
    ))
    print("confusion_matrix:", test_metrics["confusion_matrix"])
    return 0


def _write_model_card(meta: dict[str, Any]) -> None:
    tm = meta["test_metrics"]
    cm = tm["confusion_matrix"]
    cv = meta["cross_validation"]["summary"]
    imp = meta["permutation_importance"]
    body = [
        f"- **Status:** trained {meta['training_date']} (`random_state="
        f"{meta['random_state']}`).",
        f"- **Disease:** {meta['display_name']} (type 2 risk proxy)",
        "- **Dataset:** Pima Indians Diabetes Database "
        f"(sha256 `{meta['dataset']['sha256'][:16]}...`). See `DATASETS.md` and "
        "`data/metadata/diabetes.md`.",
        f"- **Algorithm:** {meta['algorithm']} "
        f"(baseline {meta['baseline_algorithm']}; {meta['selection_rule']}).",
        f"- **Features used:** {', '.join(meta['permutation_importance'][i]['feature'] for i in range(len(imp)))}.",
        "- **Preprocessing:** zeros->NaN for Glucose/BloodPressure/SkinThickness/"
        "Insulin/BMI, median imputation, standard scaling; fitted on the training "
        "split only (single sklearn Pipeline).",
        f"- **Split:** stratified {meta['split']['sizes']['train']}/"
        f"{meta['split']['sizes']['val']}/{meta['split']['sizes']['test']} "
        "train/validation/test.",
        f"- **Cross-validation (5-fold, train+val):** ROC-AUC "
        f"{cv['roc_auc']['mean']:.3f} +/- {cv['roc_auc']['std']:.3f}; PR-AUC "
        f"{cv['pr_auc']['mean']:.3f} +/- {cv['pr_auc']['std']:.3f}; recall "
        f"{cv['recall_sensitivity']['mean']:.3f} +/- {cv['recall_sensitivity']['std']:.3f}.",
        "- **Test-set metrics (threshold 0.5):** "
        f"accuracy {tm['accuracy']:.3f}, precision {tm['precision']:.3f}, "
        f"recall/sensitivity {tm['recall_sensitivity']:.3f}, specificity "
        f"{tm['specificity']:.3f}, F1 {tm['f1']:.3f}, ROC-AUC {tm['roc_auc']:.3f}, "
        f"PR-AUC {tm['pr_auc']:.3f}.",
        f"- **Test confusion matrix:** TN {cm['tn']}, FP {cm['fp']}, FN {cm['fn']}, "
        f"TP {cm['tp']} (n={tm['n']}).",
        "- **Test set at the LOW/screening cut "
        f"({meta['risk_thresholds']['low_cut']:.3f}, score >= cut counted as a "
        "positive screen):** recall/sensitivity "
        f"{meta['test_metrics_at_low_threshold_screening']['recall_sensitivity']:.3f}, "
        f"specificity {meta['test_metrics_at_low_threshold_screening']['specificity']:.3f}.",
        "- **Test set at the HIGH cut "
        f"({meta['risk_thresholds']['high_cut']:.3f}):** precision "
        f"{meta['test_metrics_at_high_threshold']['precision']:.3f}, specificity "
        f"{meta['test_metrics_at_high_threshold']['specificity']:.3f}.",
        f"- **Calibration:** {meta['calibration']['method']} "
        f"(validation Brier: {json.dumps(meta['calibration']['validation_brier'])}).",
        f"- **Risk-level thresholds:** {meta['risk_thresholds']['policy']}",
        f"- **Top permutation-importance features:** "
        + ", ".join(f"{i['feature']} ({i['importance']:.3f})" for i in imp[:4]) + ".",
        "- **Intended use:** educational risk-assessment demonstration.",
        "- **Out-of-scope use:** clinical decision-making; populations other than "
        "adult women of Pima ancestry; anyone expecting a calibrated probability "
        "when calibration is 'none'.",
        "- **Ethical note:** the Pima (Akimel O'odham) community has raised "
        "concerns about research use of this dataset; see `LIMITATIONS.md`.",
    ]
    begin, end = "<!-- BEGIN diabetes-v1 -->", "<!-- END diabetes-v1 -->"
    text = DOC_MODEL_CARD.read_text(encoding="utf-8")
    DOC_MODEL_CARD.write_text(
        replace_between_markers(text, begin, end, "\n".join(body)), encoding="utf-8"
    )
    LOG.info("updated %s", DOC_MODEL_CARD)


if __name__ == "__main__":
    raise SystemExit(main())
