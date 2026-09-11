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
- **Disease / target:** Diabetes — positive class = tested positive for diabetes (Outcome = 1).
- **Dataset:** Pima Indians Diabetes Database (registry key `diabetes`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 460/154/154 train/validation/test; dataset positive rate 0.349.
- **Cross-validation (5-fold):** ROC-AUC 0.832 +/- 0.024; PR-AUC 0.726 +/- 0.044; recall 0.682 +/- 0.035.
- **Test metrics (threshold 0.5):** accuracy 0.747, precision 0.653, recall/sensitivity 0.593, specificity 0.830, F1 0.621, ROC-AUC 0.825, PR-AUC 0.723.
- **Test confusion matrix:** TN 83, FP 17, FN 22, TP 32 (n=154).
- **Test at screening cut (0.293):** recall 0.833, specificity 0.670.
- **Calibration:** isotonic (validation Brier {"none": 0.1422, "sigmoid": 0.1408, "isotonic": 0.1397}).
- **Risk-level thresholds:** HIGH if score >= 0.4719 (largest validation threshold with specificity >= 0.85); LOW if score < 0.2933 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** Glucose (0.107), BMI (0.043), Age (0.026), DiabetesPedigreeFunction (0.011).
- **Limitations:** female Pima (Akimel O'odham) patients aged 21+ only; small sample; encoded-missing zeros; not externally validated
- **Ethical note:** the Pima (Akimel O'odham) community has raised concerns about research use of this dataset; see LIMITATIONS.md
<!-- END diabetes-v1 -->

---


## breast_cancer-v1

<!-- BEGIN breast_cancer-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Breast Cancer — positive class = malignant diagnosis.
- **Dataset:** UCI Breast Cancer Wisconsin (Diagnostic) / WDBC (via scikit-learn) (registry key `breast_cancer`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** LogisticRegression (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 341/114/114 train/validation/test; dataset positive rate 0.373.
- **Cross-validation (5 folds scored):** ROC-AUC 0.995 +/- 0.005; PR-AUC 0.994 +/- 0.005; recall 0.959 +/- 0.030.
- **Test metrics (threshold 0.5):** accuracy 0.974, precision 0.976, recall/sensitivity 0.952, specificity 0.986, F1 0.964, ROC-AUC 0.994, PR-AUC 0.990.
- **Test confusion matrix:** TN 71, FP 1, FN 2, TP 40 (n=114).
- **Test at screening cut (0.078):** recall 0.976, specificity 0.944.
- **Calibration:** isotonic (validation Brier {"none": 0.0258, "sigmoid": 0.0263, "isotonic": 0.0224}).
- **Risk-level thresholds:** validation sensitivity and specificity targets could not both be satisfied without the bands crossing; single cut at the validation median score 0.0780 (>= is HIGH, otherwise LOW; MODERATE unused).
- **Top permutation-importance features:** worst concavity (0.010), worst symmetry (0.010), worst texture (0.008), mean concave points (0.005).
- **Limitations:** features are computed from digitised fine-needle-aspirate images, not routine intake data; single-institution sample; this is a pipeline demonstration, not a self-service risk tool
<!-- END breast_cancer-v1 -->

---


## heart_failure-v1

<!-- BEGIN heart_failure-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Heart Failure — positive class = death during the follow-up period (DEATH_EVENT = 1).
- **Dataset:** UCI Heart Failure Clinical Records (registry key `heart_failure`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 179/60/60 train/validation/test; dataset positive rate 0.321.
- **Cross-validation (5-fold):** ROC-AUC 0.789 +/- 0.045; PR-AUC 0.681 +/- 0.042; recall 0.595 +/- 0.135.
- **Test metrics (threshold 0.5):** accuracy 0.733, precision 0.600, recall/sensitivity 0.474, specificity 0.854, F1 0.529, ROC-AUC 0.789, PR-AUC 0.576.
- **Test confusion matrix:** TN 35, FP 6, FN 10, TP 9 (n=60).
- **Test at screening cut (0.307):** recall 0.684, specificity 0.707.
- **Calibration:** sigmoid (validation Brier {"none": 0.151, "sigmoid": 0.1483, "isotonic": 0.1494}).
- **Risk-level thresholds:** HIGH if score >= 0.4957 (largest validation threshold with specificity >= 0.85); LOW if score < 0.3071 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** serum_creatinine (0.082), age (0.078), ejection_fraction (0.077), serum_sodium (0.010).
- **Limitations:** 299 patients from one hospital in Faisalabad, Pakistan (2015); target is mortality among people already diagnosed with heart failure, not incidence; `time` column dropped to avoid leakage; very small sample
<!-- END heart_failure-v1 -->

---


## thyroid-v1

<!-- BEGIN thyroid-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Thyroid Disease — positive class = abnormal thyroid function (hyper- or hypothyroid).
- **Dataset:** UCI Thyroid Disease ('new-thyroid' subset) (registry key `thyroid`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 129/43/43 train/validation/test; dataset positive rate 0.302.
- **Cross-validation (5 folds scored):** ROC-AUC 0.995 +/- 0.008; PR-AUC 0.990 +/- 0.015; recall 0.884 +/- 0.042.
- **Test metrics (threshold 0.5):** accuracy 1.000, precision 1.000, recall/sensitivity 1.000, specificity 1.000, F1 1.000, ROC-AUC 1.000, PR-AUC 1.000.
- **Test confusion matrix:** TN 30, FP 0, FN 0, TP 13 (n=43).
- **Test at screening cut (0.050):** recall 1.000, specificity 0.833.
- **Calibration:** isotonic (validation Brier {"none": 0.0466, "sigmoid": 0.0547, "isotonic": 0.0398}).
- **Risk-level thresholds:** validation sensitivity and specificity targets could not both be satisfied without the bands crossing; single cut at the validation median score 0.0500 (>= is HIGH, otherwise LOW; MODERATE unused).
- **Top permutation-importance features:** total_thyroxin (0.091), TSH_diff_after_TRH (0.016), total_triiodothyronine (0.005), TSH (0.003).
- **Limitations:** only 215 samples; historical (Garvan Institute); the larger UCI thyroid subsets (sick, allhypo, thyroid0387) were not used in V1; five assay values, not routine intake data
<!-- END thyroid-v1 -->

---


## maternal_health-v1

<!-- BEGIN maternal_health-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Maternal Health Risk — positive class = high risk during pregnancy (RiskLevel = 'high risk').
- **Dataset:** UCI Maternal Health Risk (registry key `maternal_health`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 255/86/86 train/validation/test; dataset positive rate 0.262.
- **Cross-validation (5-fold):** ROC-AUC 0.926 +/- 0.044; PR-AUC 0.827 +/- 0.090; recall 0.829 +/- 0.117.
- **Test metrics (threshold 0.5):** accuracy 0.884, precision 0.741, recall/sensitivity 0.870, specificity 0.889, F1 0.800, ROC-AUC 0.949, PR-AUC 0.895.
- **Test confusion matrix:** TN 56, FP 7, FN 3, TP 20 (n=86).
- **Test at screening cut (0.100):** recall 0.957, specificity 0.635.
- **Calibration:** none (validation Brier {"none": 0.0772, "sigmoid": 0.0856, "isotonic": 0.0844}).
- **Risk-level thresholds:** validation sensitivity and specificity targets could not both be satisfied without the bands crossing; single cut at the validation median score 0.0998 (>= is HIGH, otherwise LOW; MODERATE unused).
- **Top permutation-importance features:** BS (0.128), SystolicBP (0.069), BodyTemp (0.032), Age (0.023).
- **Limitations:** collected via IoT devices from rural Bangladesh clinics; the raw file contains many exact-duplicate rows (dropped here); the upstream risk-level labelling method is not fully specified
<!-- END maternal_health-v1 -->

---


## heart_disease-v1

<!-- BEGIN heart_disease-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Heart Disease — positive class = angiographic heart disease present (num > 0).
- **Dataset:** UCI Heart Disease (Cleveland) (registry key `heart_disease`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** LogisticRegression (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 181/61/61 train/validation/test; dataset positive rate 0.459.
- **Cross-validation (5-fold):** ROC-AUC 0.907 +/- 0.018; PR-AUC 0.899 +/- 0.028; recall 0.819 +/- 0.051.
- **Test metrics (threshold 0.5):** accuracy 0.869, precision 0.812, recall/sensitivity 0.929, specificity 0.818, F1 0.867, ROC-AUC 0.958, PR-AUC 0.941.
- **Test confusion matrix:** TN 27, FP 6, FN 2, TP 26 (n=61).
- **Test at screening cut (0.497):** recall 0.929, specificity 0.818.
- **Calibration:** none (validation Brier {"none": 0.1086, "sigmoid": 0.1096, "isotonic": 0.1109}).
- **Risk-level thresholds:** HIGH if score >= 0.6010 (largest validation threshold with specificity >= 0.85); LOW if score < 0.4966 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** ca (0.105), cp (0.038), slope (0.016), exang (0.014).
- **Limitations:** ~303 Cleveland Clinic records; `ca` and `thal` have missing values; avoid the 1025-row Kaggle heart.csv which contains duplicated rows
<!-- END heart_disease-v1 -->

---


## liver_disease-v1

<!-- BEGIN liver_disease-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Liver Disease — positive class = liver patient (Selector = 1).
- **Dataset:** UCI Indian Liver Patient Dataset (ILPD) (registry key `liver_disease`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 349/117/117 train/validation/test; dataset positive rate 0.714.
- **Cross-validation (5-fold):** ROC-AUC 0.738 +/- 0.025; PR-AUC 0.884 +/- 0.013; recall 0.793 +/- 0.043.
- **Test metrics (threshold 0.5):** accuracy 0.735, precision 0.795, recall/sensitivity 0.843, specificity 0.471, F1 0.819, ROC-AUC 0.794, PR-AUC 0.916.
- **Test confusion matrix:** TN 16, FP 18, FN 13, TP 70 (n=117).
- **Test at screening cut (0.501):** recall 0.843, specificity 0.471.
- **Calibration:** none (validation Brier {"none": 0.1545, "sigmoid": 0.1659, "isotonic": 0.1599}).
- **Risk-level thresholds:** HIGH if score >= 0.7304 (largest validation threshold with specificity >= 0.85); LOW if score < 0.5012 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** Total_Bilirubin (0.039), Alkaline_Phosphotase (0.036), Aspartate_Aminotransferase (0.035), Direct_Bilirubin (0.028).
- **Limitations:** 583 records from one region of Andhra Pradesh, India; class imbalance (~71% patients); strong sex imbalance; a few missing Albumin_and_Globulin_Ratio values
<!-- END liver_disease-v1 -->

---


## obesity-v1

<!-- BEGIN obesity-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Obesity — positive class = obesity (NObeyesdad in Obesity_Type_I/II/III).
- **Dataset:** UCI Estimation of Obesity Levels Based on Eating Habits and Physical Condition (registry key `obesity`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 1266/422/423 train/validation/test; dataset positive rate 0.460.
- **Cross-validation (5-fold):** ROC-AUC 0.971 +/- 0.007; PR-AUC 0.971 +/- 0.005; recall 0.928 +/- 0.010.
- **Test metrics (threshold 0.5):** accuracy 0.910, precision 0.920, recall/sensitivity 0.882, specificity 0.934, F1 0.901, ROC-AUC 0.964, PR-AUC 0.967.
- **Test confusion matrix:** TN 213, FP 15, FN 23, TP 172 (n=423).
- **Test at screening cut (0.419):** recall 0.903, specificity 0.930.
- **Calibration:** isotonic (validation Brier {"none": 0.0884, "sigmoid": 0.0787, "isotonic": 0.0779}).
- **Risk-level thresholds:** validation sensitivity and specificity targets could not both be satisfied without the bands crossing; single cut at the validation median score 0.4187 (>= is HIGH, otherwise LOW; MODERATE unused).
- **Top permutation-importance features:** family_history_with_overweight (0.111), CAEC (0.052), Age (0.050), NCP (0.030).
- **Limitations:** 77% of rows are SMOTE-synthetic, so metrics are weak evidence; population is Mexico/Peru/Colombia; Height and Weight excluded on purpose because the label is BMI-derived; treat this as a demonstration model
<!-- END obesity-v1 -->

---


## kidney_disease-v1

<!-- BEGIN kidney_disease-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Chronic Kidney Disease — positive class = chronic kidney disease (class = ckd).
- **Dataset:** UCI Chronic Kidney Disease (registry key `kidney_disease`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** LogisticRegression (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 240/80/80 train/validation/test; dataset positive rate 0.625.
- **Cross-validation (5 folds scored):** ROC-AUC 1.000 +/- 0.000; PR-AUC 1.000 +/- 0.000; recall 1.000 +/- 0.000.
- **Test metrics (threshold 0.5):** accuracy 0.988, precision 1.000, recall/sensitivity 0.980, specificity 1.000, F1 0.990, ROC-AUC 1.000, PR-AUC 1.000.
- **Test confusion matrix:** TN 30, FP 0, FN 1, TP 49 (n=80).
- **Test at screening cut (0.500):** recall 0.980, specificity 1.000.
- **Calibration:** isotonic (validation Brier {"none": 0.0076, "sigmoid": 0.011, "isotonic": 0.0046}).
- **Risk-level thresholds:** the validation score distribution is near-separable (bimodal at 0/1), so sensitivity/specificity target thresholds are not informative; using a plain 0.5 cut (>= is HIGH, otherwise LOW; MODERATE unused). Treat the near-perfect metrics with caution.
- **Top permutation-importance features:** sg (0.007), rbc (0.005), al (0.003), pot (0.003).
- **Limitations:** 400 records from a single hospital in India collected over ~2 months; heavy missingness in most columns; small sample
<!-- END kidney_disease-v1 -->

---


## parkinsons-v1

<!-- BEGIN parkinsons-v1 -->
- **Status:** trained 2026-09-10 (`random_state=42`).
- **Disease / target:** Parkinson's Disease — positive class = Parkinson's disease (status = 1).
- **Dataset:** UCI Oxford Parkinson's Disease Detection Dataset (registry key `parkinsons`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** GradientBoostingClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** grouped 110/42/43 train/validation/test; dataset positive rate 0.754.
- **Cross-validation (4 folds scored):** ROC-AUC 0.713 +/- 0.263; PR-AUC 0.869 +/- 0.134; recall 0.861 +/- 0.170.
- **Test metrics (threshold 0.5):** accuracy 0.721, precision 0.721, recall/sensitivity 1.000, specificity 0.000, F1 0.838, ROC-AUC 0.573, PR-AUC 0.823.
- **Test confusion matrix:** TN 0, FP 12, FN 0, TP 31 (n=43).
- **Test at screening cut (0.741):** recall 0.806, specificity 0.250.
- **Calibration:** sigmoid (validation Brier {"none": 0.1667, "sigmoid": 0.1469, "isotonic": 0.164}).
- **Risk-level thresholds:** HIGH if score >= 0.9146 (largest validation threshold with specificity >= 0.85); LOW if score < 0.7412 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** PPE (0.118), spread2 (0.058), MDVP:Jitter(%) (0.051), spread1 (0.043).
- **Limitations:** only 31 subjects (23 with Parkinson's); inputs are voice-signal features, not user-enterable; split and CV are grouped by subject; tiny sample so metrics have wide confidence intervals
<!-- END parkinsons-v1 -->

---


## diabetes-v2

<!-- BEGIN diabetes-v2 -->
- **Status:** trained 2026-09-11 (`random_state=42`).
- **Disease / target:** Diabetes — positive class = tested positive for diabetes (Outcome = 1).
- **Dataset:** Pima Indians Diabetes Database (registry key `diabetes`). See `DATASETS.md` and `data/metadata/`.
- **Algorithm:** RandomForestClassifier (baseline LogisticRegression; prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01).
- **Split:** stratified 460/154/154 train/validation/test; dataset positive rate 0.349.
- **Cross-validation (5 folds scored):** ROC-AUC 0.839 +/- 0.020; PR-AUC 0.731 +/- 0.057; recall 0.748 +/- 0.027.
- **Test metrics (threshold 0.5):** accuracy 0.727, precision 0.620, recall/sensitivity 0.574, specificity 0.810, F1 0.596, ROC-AUC 0.820, PR-AUC 0.701.
- **Test confusion matrix:** TN 81, FP 19, FN 23, TP 31 (n=154).
- **Test at screening cut (0.296):** recall 0.815, specificity 0.680.
- **Calibration:** isotonic (validation Brier {"none": 0.1484, "sigmoid": 0.1371, "isotonic": 0.1368}).
- **Risk-level thresholds:** HIGH if score >= 0.4649 (largest validation threshold with specificity >= 0.85); LOW if score < 0.2958 (smallest validation threshold with sensitivity >= 0.85); MODERATE otherwise.
- **Top permutation-importance features:** Glucose (0.089), BMI (0.045), Age (0.028), DiabetesPedigreeFunction (0.012).
- **Limitations:** female Pima (Akimel O'odham) patients aged 21+ only; small sample; encoded-missing zeros; not externally validated
- **Ethical note:** the Pima (Akimel O'odham) community has raised concerns about research use of this dataset; see LIMITATIONS.md
<!-- END diabetes-v2 -->

---

## Diseases without a model in V1

| Disease | Reason |
|---------|--------|
| Stroke | No public dataset with acceptable provenance/licensing without personal credentials. |
| Hypertension | No rigorously defined public target established for V1. |

Remaining diseases (Heart Disease, CKD, Liver, Breast Cancer, Parkinson's, Heart
Failure, Thyroid, Obesity, Maternal Health) get their sections here as their
pipelines are implemented in Phase 5.
