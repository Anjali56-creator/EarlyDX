import type {
  DatasetEntry,
  DiseaseInfo,
  FeatureSchema,
  Health,
  ModelEntry,
  PredictAllResponse,
  PredictionResponse,
  ValidationSamples,
} from "../types";

// Production: VITE_API_BASE is the deployed backend origin (set at build time on
// Vercel). Local dev: unset, so requests go to /api and the Vite proxy forwards
// them to localhost:8000. Trailing slashes are stripped so a value like
// "https://host/" doesn't produce "https://host//health" (a 404).
const BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/+$/, "");

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  const text = await res.text();
  const body = text ? JSON.parse(text) : null;
  if (!res.ok) {
    const detail = body?.detail ?? res.statusText;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
  }
  return body as T;
}

export class ApiError extends Error {
  status: number;
  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }
}

export const api = {
  health: () => req<Health>("/health"),
  diseases: () => req<{ count: number; diseases: DiseaseInfo[] }>("/diseases"),
  models: () => req<{ count: number; models: ModelEntry[] }>("/models"),
  datasets: () => req<{ count: number; datasets: DatasetEntry[]; meta: Record<string, unknown> }>("/datasets"),
  schema: (disease: string) => req<FeatureSchema>(`/schema/${disease}`),
  predict: (disease: string, features: Record<string, number | string>) =>
    req<PredictionResponse>(`/predict/${disease}`, {
      method: "POST",
      body: JSON.stringify(features),
    }),
  predictAll: (features: Record<string, number | string>) =>
    req<PredictAllResponse>("/predict/all", {
      method: "POST",
      body: JSON.stringify({ features }),
    }),
  validationSamples: (disease: string) => req<ValidationSamples>(`/validation/${disease}`),
};
