import { render, screen } from "@testing-library/react";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Validation } from "../pages/Validation";

const DISEASES = {
  count: 1,
  diseases: [
    { key: "diabetes", display_name: "Diabetes", group: "metabolic", model_available: true, unavailable_reason: null, required_features: ["Glucose"] },
  ],
};

const metrics = (over: Record<string, number>) => ({
  n: 154, threshold: 0.5, accuracy: 0.7, precision: 0.6, recall_sensitivity: 0.6, specificity: 0.8, f1: 0.6, roc_auc: 0.8, pr_auc: 0.7,
  confusion_matrix: { tn: 83, fp: 17, fn: 22, tp: 32 },
  ...over,
});

const EVALUATION = {
  disease: "Diabetes",
  disease_key: "diabetes",
  model_version: "diabetes-v1",
  selected_algorithm: "RandomForestClassifier",
  baseline_algorithm: "LogisticRegression",
  selection_rule: "prefer LogisticRegression; a candidate is chosen only if its validation ROC-AUC exceeds the baseline by >= 0.01",
  best_validation_candidate_by_roc_auc: "RandomForestClassifier",
  training_date: "2026-09-10",
  dataset: { name: "Pima" },
  split: { scheme: "stratified", sizes: { train: 460, val: 154, test: 154 } },
  target_positive_rate: 0.35,
  candidates: {
    LogisticRegression: metrics({ roc_auc: 0.81 }),
    RandomForestClassifier: metrics({ roc_auc: 0.84 }),
    GradientBoostingClassifier: metrics({ roc_auc: 0.79 }),
  },
  test_metrics: { ...metrics({ roc_auc: 0.8255, accuracy: 0.7468, f1: 0.6214 }), class_distribution: { "0": 100, "1": 54 }, brier: 0.16, calibration_bins: [] },
  test_metrics_at_screening_threshold: metrics({ threshold: 0.29, recall_sensitivity: 0.83, specificity: 0.67, confusion_matrix: { tn: 67, fp: 33, fn: 9, tp: 45 } } as never),
  cross_validation: { folds: 5, summary: { roc_auc: { mean: 0.832, std: 0.024 } } },
  calibration: { method: "isotonic", validation_brier: { none: 0.142, sigmoid: 0.141, isotonic: 0.14 } },
  risk_thresholds: { low_cut: 0.293, high_cut: 0.472 },
  permutation_importance: [{ feature: "Glucose", importance: 0.11, std: 0.02 }],
  unavailable: { roc_curve_points: "not stored by the training run; only the scalar ROC-AUC is recorded" },
  limitations: "small sample",
  ethical_note: null,
  has_validation_samples: false,
};

function jsonRes(data: unknown, status = 200) {
  return { ok: status < 400, status, statusText: "OK", text: async () => JSON.stringify(data) } as Response;
}

beforeEach(() => {
  vi.stubGlobal("fetch", vi.fn(async (url: string) => {
    const u = String(url);
    if (u.endsWith("/diseases")) return jsonRes(DISEASES);
    if (u.endsWith("/evaluation/diabetes")) return jsonRes(EVALUATION);
    if (u.endsWith("/validation/diabetes")) return jsonRes({ detail: "none" }, 404);
    throw new Error(`unexpected fetch: ${u}`);
  }));
});
afterEach(() => {
  vi.unstubAllGlobals();
});

test("validation page shows candidate comparison, verdict, test metrics and confusion matrix", async () => {
  const router = createMemoryRouter([{ path: "/validation", element: <Validation /> }], { initialEntries: ["/validation"] });
  render(<RouterProvider router={router} />);

  await screen.findByText(/Verdict for Diabetes/);
  // all three candidates are listed and the deployed one is flagged
  expect(screen.getAllByText("LogisticRegression").length).toBeGreaterThan(0);
  expect(screen.getAllByText("GradientBoostingClassifier").length).toBeGreaterThan(0);
  expect(screen.getByText(/selected · deployed/)).toBeInTheDocument();
  // standard metrics + confusion matrix cells from the stored test evaluation
  expect(screen.getByText("0.826")).toBeInTheDocument(); // test ROC-AUC 0.8255
  expect(screen.getByText("TN 83")).toBeInTheDocument();
  expect(screen.getByText("FN 22")).toBeInTheDocument();
  // honesty: ROC curve unavailability is stated, and missing per-row samples are explained
  expect(screen.getByText(/only the scalar ROC-AUC is recorded/)).toBeInTheDocument();
  expect(screen.getByText(/Individual test rows were not captured/)).toBeInTheDocument();
});
