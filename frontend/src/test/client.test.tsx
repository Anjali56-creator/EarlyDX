import { afterEach, beforeEach, expect, test, vi } from "vitest";
import { api, ApiError } from "../api/client";

function res(status: number, body: unknown) {
  return { ok: status < 400, status, statusText: String(status), text: async () => (typeof body === "string" ? body : JSON.stringify(body)) } as Response;
}

beforeEach(() => {
  vi.useFakeTimers();
});
afterEach(() => { vi.useRealTimers(); vi.unstubAllGlobals(); });

test("GET retries through a cold-start 503 + HTML error page, then returns the real payload", async () => {
  const payload = { count: 1, diseases: [{ key: "diabetes", display_name: "Diabetes", group: "metabolic", model_available: true, unavailable_reason: null, required_features: [] }] };
  const fetchMock = vi.fn()
    .mockResolvedValueOnce(res(503, "<html>Service Unavailable</html>"))
    .mockRejectedValueOnce(new TypeError("Failed to fetch"))
    .mockResolvedValueOnce(res(200, payload));
  vi.stubGlobal("fetch", fetchMock);

  const p = api.diseases();
  await vi.runAllTimersAsync(); // advance the back-off sleeps
  await expect(p).resolves.toEqual(payload);
  expect(fetchMock).toHaveBeenCalledTimes(3);
  expect(String(fetchMock.mock.calls[0][0])).toMatch(/\/diseases$/);
});

test("GET gives up after the retry budget with a status-0 ApiError", async () => {
  vi.stubGlobal("fetch", vi.fn().mockRejectedValue(new TypeError("Failed to fetch")));
  const p = api.diseases();
  p.catch(() => undefined); // attach a handler before timers run so no unhandled rejection is logged
  await vi.runAllTimersAsync();
  await expect(p).rejects.toBeInstanceOf(ApiError);
  await expect(p).rejects.toMatchObject({ status: 0 });
});

test("POST /predict is never retried and 4xx is never retried", async () => {
  const post = vi.fn().mockRejectedValue(new TypeError("Failed to fetch"));
  vi.stubGlobal("fetch", post);
  const p1 = api.predict("diabetes", { Glucose: 1 });
  p1.catch(() => undefined);
  await vi.runAllTimersAsync();
  await expect(p1).rejects.toBeInstanceOf(ApiError);
  expect(post).toHaveBeenCalledTimes(1);

  const notFound = vi.fn().mockResolvedValue(res(404, { detail: "no feature schema for 'stroke'" }));
  vi.stubGlobal("fetch", notFound);
  const p2 = api.schema("stroke");
  p2.catch(() => undefined);
  await vi.runAllTimersAsync();
  await expect(p2).rejects.toMatchObject({ status: 404, message: "no feature schema for 'stroke'" });
  expect(notFound).toHaveBeenCalledTimes(1);
});
