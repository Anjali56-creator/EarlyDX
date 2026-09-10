export function About() {
  return (
    <>
      <h1>About EarlyDX</h1>
      <p className="sub">Purpose, method, and what this system is not.</p>

      <h2>Purpose</h2>
      <p>
        EarlyDX is a portfolio and research prototype for multi-disease early risk assessment. Each
        condition has its own independent pipeline trained on a real, publicly available,
        de-identified dataset.
      </p>

      <h2>Method</h2>
      <ul>
        <li>One leakage-safe scikit-learn pipeline per disease; all preprocessing is fitted on the training split only.</li>
        <li>An interpretable baseline (logistic regression) is compared against stronger candidates; the baseline is kept unless clearly beaten.</li>
        <li>Metrics come from a held-out test set and stratified cross-validation. Model selection favours sensitivity for early detection.</li>
        <li>Probability calibration is applied only when it improves the validation Brier score, and is labelled as such.</li>
        <li>Explanations use the model's own permutation importance.</li>
      </ul>

      <h2>What EarlyDX is not</h2>
      <ul>
        <li>Not a medical device, not clinically validated, not externally validated.</li>
        <li>Not HIPAA compliant, not GDPR compliant, not medically certified.</li>
        <li>Not a diagnosis. A risk score is a model output, not a statement about your health.</li>
      </ul>

      <h2>Documentation</h2>
      <p>
        See <code>ARCHITECTURE.md</code>, <code>DATASETS.md</code>, <code>MODEL_CARD.md</code> and{" "}
        <code>LIMITATIONS.md</code> in the repository.
      </p>
    </>
  );
}
