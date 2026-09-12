import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { createMemoryRouter, RouterProvider } from "react-router-dom";
import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { Layout } from "../components/Layout";
import { Assessment } from "../pages/Assessment";
import { Results } from "../pages/Results";

const DISEASES = {
  count: 2,
  diseases: [
    {
      key: "diabetes",
      display_name: "Diabetes",
      group: "metabolic",
      model_available: true,
      unavailable_reason: null,
      required_features: ["Glucose", "BMI", "Age"],
    },
    {
      key: "stroke",
      display_name: "Stroke",
      group: "cardiovascular",
      model_available: false,
      unavailable_reason: "no dataset",
      required_features: [],
    },
  ],
};

const SCHEMA = {
  disease: "diabetes",
  model_id: "diabetes-v1",
  target: "Outcome",
  required_features: ["Glucose", "BMI", "Age"],
  feature_details: {
    Glucose: { type: "number", unit: "mg/dL", min: 30, max: 400, required: true },
    BMI: { type: "number", unit: "kg/m^2", min: 10, max: 80, required: true },
    Age: { type: "number", unit: "years", min: 18, max: 120, required: true },
  },
};

const PREDICTION = {
  disease: "Diabetes",
  disease_key: "diabetes",
  risk_score: 0.82,
  risk_level: "HIGH",
  calibrated: true,
  model_version: "diabetes-v1",
  model_algorithm: "RandomForestClassifier",
  model_performance: { test_roc_auc: 0.83, test_pr_auc: 0.72, test_recall_sensitivity: 0.59, test_specificity: 0.83, cv_roc_auc_mean: 0.83 },
  important_features: [
    { feature: "Glucose", value: 197, population_median: 117, direction: "above typical", impact: "high", importance: 0.11 },
  ],
  threshold_policy: "HIGH if score >= 0.47…",
  disclaimer: "This is a risk assessment and not a medical diagnosis.",
  session_id: 1,
};

beforeEach(() => {
  sessionStorage.clear();
  vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
    const u = String(url);
    if (u.endsWith("/diseases")) return jsonRes(DISEASES);
    if (u.endsWith("/schema/diabetes")) return jsonRes(SCHEMA);
    if (u.endsWith("/predict/diabetes")) {
      const body = JSON.parse(String(init?.body ?? "{}"));
      expect(body).toHaveProperty("Glucose", 197);
      expect(body).toHaveProperty("BMI", 45);
      expect(body).toHaveProperty("Age", 60);
      return jsonRes(PREDICTION);
    }
    throw new Error(`unexpected fetch: ${u}`);
  }));
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function jsonRes(data: unknown) {
  return { ok: true, status: 200, statusText: "OK", text: async () => JSON.stringify(data) } as Response;
}

function renderApp() {
  const router = createMemoryRouter(
    [
      {
        path: "/",
        element: <Layout />,
        children: [
          { path: "assessment", element: <Assessment /> },
          { path: "results", element: <Results /> },
        ],
      },
    ],
    { initialEntries: ["/assessment"] },
  );
  return render(<RouterProvider router={router} />);
}

test("select disease → enter data → submit → see prediction and explanation", async () => {
  const user = userEvent.setup();
  renderApp();

  // form fields appear from the schema
  const glucose = await screen.findByLabelText(/Glucose/);
  await user.type(glucose, "197");
  await user.type(screen.getByLabelText(/BMI/), "45");
  await user.type(screen.getByLabelText(/Age/), "60");

  await user.click(screen.getByRole("button", { name: /submit assessment/i }));

  // results page renders the risk level and the contributing factor
  await waitFor(() => expect(screen.getByText(/HIGH RISK ESTIMATE/i)).toBeInTheDocument());
  expect(screen.getAllByText(/Diabetes/).length).toBeGreaterThan(0);
  expect(screen.getByText("82%")).toBeInTheDocument();
  // submitted inputs are echoed back, and the top factor is listed with its method labelled honestly
  expect(screen.getAllByText(/Glucose/).length).toBeGreaterThanOrEqual(2);
  expect(screen.getByText(/Your inputs/)).toBeInTheDocument();
  expect(screen.getByText(/permutation importance/i)).toBeInTheDocument();
  expect(screen.getByText(/not a medical diagnosis/i)).toBeInTheDocument();
  expect(screen.getAllByText(/diabetes-v1/).length).toBeGreaterThan(0);
});

test("conditions without a model are not offered as options", async () => {
  renderApp();
  await screen.findByLabelText(/Condition/);
  expect(screen.queryByRole("option", { name: "Stroke" })).not.toBeInTheDocument();
  expect(screen.getByText(/No trained model available: Stroke/)).toBeInTheDocument();
});
