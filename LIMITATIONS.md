## EarlyDX — Limitations

EarlyDX is a **portfolio and research prototype**. Read this document before
drawing any conclusion from its output.

---

## 1. Non-clinical status

- EarlyDX is **not** a medical device, **not** clinically validated, and **not**
  reviewed or cleared by any regulator.
- It performs **no external validation** on prospective or site-independent data.
- It makes **no compliance claim** — it is not HIPAA compliant, not GDPR
  compliant, and not medically certified.
- Output is a **risk score**, not a diagnosis, a probability of disease, or a
  recommendation. The correct reading is: *"a model trained on dataset X
  estimated an elevated score for this input"*.

---

## 2. Dataset limitations (general)

- Every model is trained on a **single public dataset** collected in one place,
  time, and population. None reflect a general or global patient population.
- Several datasets are **small** (Heart Disease ~303, Heart Failure ~299,
  Parkinson's 31 subjects, CKD ~400). Metrics from small test sets have wide
  confidence intervals.
- Some datasets encode **missing values as zero** (Pima diabetes) or have heavy
  missingness (CKD, Thyroid). Imputation choices affect results.
- Class **imbalance** is common (Liver ~71% positive, Stroke ~5% positive). We
  report PR-AUC and recall alongside accuracy for this reason.

Per-dataset detail lives in `DATASETS.md` and, after inspection, in
`data/metadata/<disease>.md`.

---

## 3. Demographic bias

- **Diabetes (Pima):** female Akimel O'odham (Pima) patients aged 21+ only. The
  model will not generalise to men, other ancestries, or younger people. The Pima
  community has also raised concerns about research use of this data; EarlyDX
  uses it as a widely studied teaching dataset and flags this context here.
- **Liver (ILPD):** single region of Andhra Pradesh, India; strong sex imbalance.
- **Heart Failure:** one hospital in Faisalabad, Pakistan, 2015.
- **Maternal Health:** rural Bangladesh clinics via IoT devices.
- **Obesity:** Mexico, Peru, Colombia — and 77% of rows are SMOTE-synthetic.

Predictions for people outside these populations are extrapolation.

---

## 4. Target-definition limitations

- **Heart Failure:** the dataset target is *death during follow-up in patients
  already diagnosed with heart failure*, not *risk of developing heart failure*.
  The UI frames it accordingly. The `time` column is excluded to avoid leakage.
- **Obesity:** obesity class is largely a function of BMI (height and weight). A
  model given height and weight is close to re-deriving the label. EarlyDX treats
  this as a demonstration model and/or excludes the definitional inputs.
- **Breast Cancer / Parkinson's:** inputs are features computed from medical
  imaging and voice signal processing, not values a user can supply. These are
  pipeline demonstrations, not self-service risk tools.

---

## 5. False positives and false negatives

- For early-detection screening, a **false negative** (missing real risk) is
  generally more harmful than a false positive. Model selection favours
  sensitivity/recall, which **increases false positives**.
- A high score does not mean disease is present. A low score does not mean it is
  absent. Base rates in these datasets differ from the general population, so the
  scores are **not** population-calibrated probabilities unless a model's
  `model_metadata.json` explicitly records a calibration method.

---

## 6. Conditions with no model in V1

- **Stroke:** no public dataset with acceptable provenance and licensing was
  identified without personal-account credentials. Shipped as
  `model: unavailable`.
- **Hypertension:** no rigorously defined public target was established for V1.
  Shipped as `model: unavailable` rather than inventing a target or using
  synthetic data.

The API and UI report these as unavailable and do not return a score for them.

---

## 7. Explainability limitations

- Feature attributions (SHAP, coefficients, permutation importance) describe **the
  model's behaviour on its training distribution**, not biological causation.
- Attributions can be unstable for correlated features and small datasets.

---

## 8. Engineering limitations

- No authentication, rate limiting, or multi-tenant isolation.
- Persistence stores assessment inputs and results with no PII fields, but the
  inputs are still health-related and the local database is not encrypted at
  rest by default.
- Demo/synthetic sessions are flagged but share the same tables as real ones.
