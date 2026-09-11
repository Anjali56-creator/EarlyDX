export function About() {
  return (
    <>
      <h1>About EarlyDX</h1>
      <p className="sub">Purpose, method, and what this system is not.</p>

      <h2>What is EarlyDX?</h2>
      <p>
        EarlyDX is a portfolio and research prototype for multi-disease early risk assessment. Each
        condition has its own independent pipeline trained on a real, publicly available,
        de-identified dataset.
      </p>

      <h2>How it works</h2>
      <ol>
        <li>Select a supported condition on the Assessment page.</li>
        <li>Enter the values that condition's model requires.</li>
        <li>EarlyDX sends those values to the corresponding trained model.</li>
        <li>The model returns a risk estimate and, where defined, a risk level.</li>
        <li>The result is presented together with the model's own performance information.</li>
      </ol>

      <h2>Method</h2>
      <ul>
        <li>One leakage-safe scikit-learn pipeline per disease; all preprocessing is fitted on the training split only.</li>
        <li>An interpretable baseline (logistic regression) is compared against stronger candidates; the baseline is kept unless clearly beaten.</li>
        <li>Metrics come from a held-out test set and stratified cross-validation. Model selection favours sensitivity for early detection.</li>
        <li>Probability calibration is applied only when it improves the validation Brier score, and is labelled as such.</li>
        <li>Explanations use the model's own permutation importance.</li>
      </ul>

      <h2>Research limitations</h2>
      <ul>
        <li>Models are research prototypes, not medical devices, and are not clinically validated.</li>
        <li>Datasets vary in size, and some come from a single hospital, region, or demographic group — see the Datasets page for each dataset's documented limitations.</li>
        <li>Some datasets are small or contain missing/imputed values.</li>
        <li>None of the models have been externally validated on a separate population.</li>
        <li>A risk score is a model output, not a statement about your health, and is not a diagnosis.</li>
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
