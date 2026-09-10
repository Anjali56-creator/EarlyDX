## EarlyDX — Datasets

This document records the **candidate public datasets** for each of the 12
conditions in scope.

> **Status of this document.** No dataset has been downloaded, loaded, or
> inspected yet. Every record count, feature count, and class-distribution figure
> below is taken from the dataset's own published documentation and is marked
> *pending verification*. It will be replaced with values measured by
> `scripts/inspect_dataset.py` once the file is in `data/raw/<disease>/`, and the
> machine-readable registry at `data/dataset_registry.json` will be filled in at
> that point. Unknown fields will be recorded as the literal string `"unknown"`,
> never guessed.

No fabricated datasets are or will be used. Synthetic data is used only for
frontend/demo/testing and is always labelled `DEMO / SYNTHETIC DATA — NOT REAL
PATIENT DATA`.

---

## 1. Comparison table

| # | Disease | Candidate dataset | Source | Approx. records | Target | Licence (per source page, to verify) | Acquisition | Status |
|---|---------|-------------------|--------|-----------------|--------|--------------------------------------|-------------|--------|
| 1 | Diabetes | Pima Indians Diabetes (primary) / CDC Diabetes Health Indicators, BRFSS 2015 (secondary) | Kaggle / UCI #891; originally NIDDK & CDC | ~768 / ~253,680 | `Outcome` (0/1) / `Diabetes_binary` | Pima: CC0 on Kaggle (orig. NIDDK). UCI #891: "see linked dataset", CDC BRFSS public-use | Direct download | Candidate — not inspected |
| 2 | Heart Disease | Heart Disease (Cleveland subset) | UCI #45 | ~303 | `num` (0 vs 1-4 -> binary) | CC BY 4.0 | Direct download | Candidate — not inspected |
| 3 | Chronic Kidney Disease | Chronic Kidney Disease | UCI #336 | ~400 | `class` (ckd / notckd) | CC BY 4.0 | Direct download | Candidate — not inspected |
| 4 | Liver Disease | Indian Liver Patient Dataset (ILPD) | UCI #225 | ~583 | `Selector` (1 patient / 2 non-patient) | CC BY 4.0 | Direct download | Candidate — not inspected |
| 5 | Breast Cancer | Breast Cancer Wisconsin (Diagnostic), WDBC | UCI #17 | ~569 | `diagnosis` (M / B) | CC BY 4.0 | Direct download | Candidate — not inspected |
| 6 | Parkinson's Disease | Oxford Parkinson's Disease Detection (voice) | UCI #174 | ~195 recordings, 31 subjects | `status` (0/1) | CC BY 4.0 | Direct download | Candidate — not inspected |
| 7 | Heart Failure | Heart Failure Clinical Records | UCI #519 | ~299 | `DEATH_EVENT` (0/1) | CC BY 4.0 | Direct download | Candidate — not inspected |
| 8 | Stroke | Stroke Prediction Dataset | Kaggle (`fedesoriano`) | ~5,110 | `stroke` (0/1) | "Data files (c) original authors" — permissive, redistribution unclear | Kaggle account | Candidate — provenance concern |
| 9 | Hypertension | NHANES-derived hypertension cohort (to be assembled) | CDC NHANES public-use files | varies by cycle | derived `hypertension` (BP >= 130/80 or on medication) | US public-domain government data | Build from NHANES; no single canonical file | Needs work — no ready-made file |
| 10 | Thyroid Disease | Thyroid Disease (`allhypo` / `sick` / `thyroid0387`) | UCI #102 | ~2,800-9,000 depending on subset | hypo/hyper/negative class | CC BY 4.0 | Direct download | Candidate — not inspected |
| 11 | Obesity | Estimation of Obesity Levels from Eating Habits & Physical Condition | UCI #544 | ~2,111 | `NObeyesdad` (7 classes) | CC BY 4.0 | Direct download | Candidate — 77% SMOTE-synthetic |
| 12 | Maternal Health Risk | Maternal Health Risk | UCI #863 | ~1,014 | `RiskLevel` (low / mid / high) | CC BY 4.0 | Direct download | Candidate — not inspected |

"Approx. records" and licence text must all be re-checked against the live
source page at download time.

---

## 2. Immediately available vs needs user action

**Downloadable now without an account** (UCI direct download, Kaggle CC0 mirror):
Diabetes (Pima), Heart Disease, Chronic Kidney Disease, Liver Disease, Breast
Cancer, Parkinson's, Heart Failure, Thyroid, Obesity, Maternal Health Risk.

**Needs a Kaggle account / API token:** Stroke (`fedesoriano/stroke-prediction-dataset`),
CDC Diabetes Health Indicators (also on UCI, so optional).

**Needs assembly, not a download:** Hypertension — build a cohort from CDC
**NHANES** public-use files (demographics + examination BP + questionnaire on
antihypertensive medication), target BP >= 130/80 mmHg or current antihypertensive
use.

### V1 decisions (approved)

- **Stroke — `model: unavailable` in V1.** The widely used Kaggle stroke file
  (`fedesoriano`) has no documented primary source; the concern is noted in the
  literature (medRxiv 2026.02.24.26347028). No license-clean public alternative
  with acceptable provenance was found that does not require personal Kaggle
  credentials. EarlyDX will not train on a low-provenance file. The disease is
  listed with `model_available: false` and this reason.
- **Hypertension — `model: unavailable` in V1 unless the NHANES target is
  rigorous.** A model ships only if the NHANES-derived target (BP threshold or
  documented antihypertensive use) can be defined and reproduced from the raw
  public files with every derivation step scripted. If that bar is not met,
  hypertension ships as `model: unavailable` rather than using an unreliable
  target or any synthetic data. No synthetic data will be used for either.

---

## 3. Per-dataset detail

Each entry will get a full inspection report at `data/metadata/<disease>.md`
after download, covering: exact row/column counts, dtypes, missing-value map,
class distribution, duplicate rows, candidate leakage columns, and known
population/collection limitations. Summary of what is expected:

### 1. Diabetes — Pima Indians Diabetes
- **Source:** Kaggle mirror of the NIDDK dataset; UCI historical. Population is
  female Pima (Akimel O'odham) patients aged 21+.
- **Known issues:** biologically impossible zeros in `Glucose`, `BloodPressure`,
  `SkinThickness`, `Insulin`, `BMI` (encoded missing); small size (~768); narrow
  demographic — poor external validity. CDC BRFSS indicators dataset is the
  larger, more general alternative but is self-reported survey data.

### 2. Heart Disease — UCI Cleveland
- **Source:** UCI #45, Hungarian Institute of Cardiology / Cleveland Clinic et al.
- **Known issues:** only ~303 usable Cleveland rows; `ca` and `thal` have missing
  values; target `num` is 0-4 and is commonly binarised to 0 vs >0. Avoid the
  1,025-row Kaggle `heart.csv` — it contains large numbers of duplicated rows.

### 3. Chronic Kidney Disease — UCI #336
- **Known issues:** ~400 rows, heavy missingness across most columns; mixed
  numeric/categorical; collected at a single hospital in India over ~2 months.

### 4. Liver Disease — UCI ILPD #225
- **Known issues:** ~583 rows, class imbalance (~71% patient), gender imbalance,
  a handful of missing `Albumin_and_Globulin_Ratio` values; single-region (Andhra
  Pradesh) sample.

### 5. Breast Cancer — WDBC UCI #17
- **Known issues:** clean, ~569 rows, no missing values, ~63% benign. Features are
  computed from digitised fine-needle-aspirate images, not routine intake data —
  so this model is a demonstration of the pipeline, not something a general user
  can fill in.

### 6. Parkinson's — UCI #174
- **Known issues:** only 31 subjects, multiple voice recordings each (~195 rows) —
  grouping by subject is required for any honest CV to avoid subject leakage.
  Input is acoustic features, not user-enterable.

### 7. Heart Failure — UCI #519
- **Known issues:** ~299 patients from Faisalabad, Pakistan, 2015; target is
  death during follow-up, not incidence of heart failure — framing in the UI must
  reflect "risk of adverse outcome in diagnosed patients", not "risk of
  developing heart failure". `time` (follow-up days) is a leakage risk and will
  be excluded from the risk model.

### 8. Stroke — Kaggle `fedesoriano`
- **Known issues:** ~5,110 rows; ~5% positive class (severe imbalance); ~200
  missing `bmi`; `smoking_status` has an explicit "Unknown". **Provenance is not
  documented** — the uploader lists no primary source, and this has been flagged
  in the literature. Usable for a prototype with a prominent caveat; not suitable
  for any clinical claim.

### 9. Hypertension — NHANES-derived (to build)
- **Plan:** join NHANES `DEMO`, `BPX`/`BPXO` (blood pressure), `BMX` (body
  measures), `BPQ` (blood-pressure questionnaire / medication), optionally `SMQ`,
  `DIQ`, `ALQ`. Target = mean SBP >= 130 or mean DBP >= 80 or "told has
  hypertension" + on medication. Every step scripted and documented.
- **Known issues:** cross-sectional (prevalent, not incident, hypertension);
  measurement protocol differs across cycles; US-only population.

### 10. Thyroid — UCI #102
- **Known issues:** several sub-datasets (`sick`, `allhypo`, `allhyper`,
  `thyroid0387`) with different targets; many columns are "measured?" booleans
  paired with a value; `TBG` almost entirely missing; historical (Garvan
  Institute, 1980s). Choice of sub-dataset and target will be documented before
  training.

### 11. Obesity — UCI #544
- **Known issues:** **77% of rows were generated with SMOTE** (Weka) and only 23%
  are real survey responses; population is Mexico/Peru/Colombia. Because obesity
  class is essentially derived from BMI (height/weight), this is close to a
  definitional target — the model risks learning the BMI formula. EarlyDX will
  either predict a non-trivial target (e.g. obesity from lifestyle features with
  height/weight excluded) or clearly present this as a demonstration model. The
  synthetic majority makes any reported metric weak evidence.

### 12. Maternal Health Risk — UCI #863
- **Known issues:** ~1,014 rows, 6 features, 3-class target; collected via IoT
  devices from rural Bangladesh clinics; contains duplicate rows that must be
  handled; risk-level labelling methodology is not fully specified upstream.

---

## 4. Datasets that are NOT available

None are being substituted. Where a legitimate dataset cannot be obtained, the
disease will be reported as:

> Dataset unavailable — awaiting user-provided dataset.

and its model will be absent from the registry and marked `unavailable` in the
API and UI. Hypertension is the current risk for this outcome.
