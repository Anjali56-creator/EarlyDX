## EarlyDX — Model Card

One section per trained model. Every number in this file is copied from an
executed evaluation run (`ml/<disease>/evaluate.py`) and mirrored in
`ml/<disease>/model_metadata.json` and `models/model_registry.json`. Nothing here
is hand-authored or estimated.

Sections are added only after the corresponding model has actually been trained
and evaluated. Diseases without a model are listed at the end.

---

## diabetes-v1

<!-- BEGIN diabetes-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease:** Diabetes (type 2 risk proxy)
- **Dataset:** Pima Indians Diabetes Database (sha256 `d765aa828a47e8d3...`). See `DATASETS.md` and `data/metadata/diabetes.md`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Features used:** Glucose, BMI, Age, DiabetesPedigreeFunction, Pregnancies, Insulin, SkinThickness, BloodPressure.
- **Preprocessing:** zeros->NaN for Glucose/BloodPressure/SkinThickness/Insulin/BMI, median imputation, standard scaling; fitted on the training split only (single sklearn Pipeline).
- **Split:** stratified 460/154/154 train/validation/test.
- **Cross-validation (5-fold, train+val):** ROC-AUC 0.832 +/- 0.024; PR-AUC 0.726 +/- 0.044; recall 0.682 +/- 0.035.
- **Test-set metrics (threshold 0.5):** accuracy 0.747, precision 0.653, recall/sensitivity 0.593, specificity 0.830, F1 0.621, ROC-AUC 0.825, PR-AUC 0.723.
- **Test confusion matrix:** TN 83, FP 17, FN 22, TP 32 (n=154).
- **Test set at the LOW/screening cut (0.293, score >= cut counted as a positive screen):** recall/sensitivity 0.833, specificity 0.670.
- **Test set at the HIGH cut (0.472):** precision 0.635, specificity 0.810.
- **Calibration:** isotonic (validation Brier: {"none": 0.1421677053055917, "sigmoid": 0.14082324437004584, "isotonic": 0.1397381388440477}).
- **Risk-level thresholds:** HIGH if score >= 0.4719 (largest validation threshold with specificity >= 0.85); LOW if score < 0.2933 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** Glucose (0.107), BMI (0.043), Age (0.026), DiabetesPedigreeFunction (0.011).
- **Intended use:** educational risk-assessment demonstration.
- **Out-of-scope use:** clinical decision-making; populations other than adult women of Pima ancestry; anyone expecting a calibrated probability when calibration is 'none'.
- **Ethical note:** the Pima (Akimel O'odham) community has raised concerns about research use of this dataset; see `LIMITATIONS.md`.
<!-- END diabetes-v1 -->

---

## Diseases without a model in V1

| Disease | Reason |
|---------|--------|
| Stroke | No public dataset with acceptable provenance/licensing without personal credentials. |
| Hypertension | No rigorously defined public target established for V1. |

Remaining diseases (Heart Disease, CKD, Liver, Breast Cancer, Parkinson's, Heart
Failure, Thyroid, Obesity, Maternal Health) get their sections here as their
pipelines are implemented in Phase 5.
