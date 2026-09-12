import type {
  DatasetEntry,
  DiseaseInfo,
  EvaluationDetail,
  EvaluationSummary,
  FeatureSchema,
  Health,
  ModelEntry,
  PredictAllResponse,
  PredictionResponse,
  RecentAssessments,
  ValidationSamples,
} from "../types";

// Production: VITE_API_BASE is the deployed backend origin (set at build time on
// Vercel). Local dev: unset, so requests go to /api and the Vite proxy forwards
// them to localhost:8000. Trailing slashes are stripped so a value like
// "https://host/" doesn't produce "https://host//health" (a 404).
const BASE = (import.meta.env.VITE_API_BASE ?? "/api").replace(/\/+$/, "");

// The backend runs on a free Render instance that sleeps when idle. The first
// request after idle can hang for up to ~60 s while it boots, and Render's edge
// may answer 502/503 for a moment before the app is listening. GETs are
// therefore retried a few times with a short back-off; each attempt has its
// own timeout so a stalled connection cannot leave a page in "Loading…" forever.
const ATTEMPT_TIMEOUT_MS = 75_000;
const GET_RETRIES = 2;
const RETRY_DELAY_MS = 2_500;
const RETRYABLE_STATUS = new Set([502, 503, 504]);

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

async function attempt<T>(url: string, init?: RequestInit): Promise<T> {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), ATTEMPT_TIMEOUT_MS);
  let res: Response;
  try {
    res = await fetch(url, {
      headers: { "Content-Type": "application/json" },
      signal: ctrl.signal,
      ...init,
    });
  } catch (e) {
    // network failure, CORS rejection, or our own timeout — all surface here
    const aborted = (e as { name?: string })?.name === "AbortError";
    throw new ApiError(
      aborted
        ? "the API did not respond within 75 s (it may be waking from idle)"
        : `network error — ${(e as Error).message}`,
      0,
    );
  } finally {
    clearTimeout(timer);
  }
  const text = await res.text();
  let body: unknown = null;
  if (text) {
    try {
      body = JSON.parse(text);
    } catch {
      // Render's edge returns an HTML error page while the service is starting
      throw new ApiError(`unexpected non-JSON response (HTTP ${res.status})`, res.status);
    }
  }
  if (!res.ok) {
    const detail = (body as { detail?: unknown } | null)?.detail ?? res.statusText;
    throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status);
  }
  return body as T;
}

async function req<T>(path: string, init?: RequestInit): Promise<T> {
  const url = `${BASE}${path}`;
  const method = (init?.method ?? "GET").toUpperCase();
  // Only idempotent GETs are retried; a POST /predict is sent exactly once so
  // an assessment is never recorded twice.
  const retries = method === "GET" ? GET_RETRIES : 0;
  for (let i = 0; ; i++) {
    try {
      return await attempt<T>(url, init);
    } catch (e) {
      const err = e as ApiError;
      const retryable = err.status === 0 || RETRYABLE_STATUS.has(err.status);
      if (i >= retries || !retryable) throw err;
      await sleep(RETRY_DELAY_MS * (i + 1));
    }
  }
}

/** Fire-and-forget ping so the API starts waking as soon as the app opens,
 * rather than when the user reaches a page that needs data. */
export function warmUp(): void {
  req<Health>("/health").catch(() => undefined);
}

export class ApiError extends Error {
  /** HTTP status; 0 means the request never got a response (network/timeout). */
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
  evaluationSummary: () => req<{ count: number; models: EvaluationSummary[] }>("/evaluation"),
  evaluation: (disease: string) => req<EvaluationDetail>(`/evaluation/${disease}`),
  recentAssessments: (limit = 10) => req<RecentAssessments>(`/assessments/recent?limit=${limit}`),
};
