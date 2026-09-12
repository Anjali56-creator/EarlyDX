import { useEffect, useState } from "react";

/** Shared loading / API-error states so every page behaves the same while
 * data is in flight. The backend runs on a free Render instance that sleeps
 * when idle; after a few seconds the loading copy explains the cold start so
 * a slow first request is not mistaken for a broken app. */
export function Loading({ what = "data" }: { what?: string }) {
  const [slow, setSlow] = useState(false);
  useEffect(() => {
    const t = setTimeout(() => setSlow(true), 4000);
    return () => clearTimeout(t);
  }, []);
  return (
    <div className={`state-line ${slow ? "state-slow" : "muted"}`} role="status" aria-live="polite">
      <span className="spinner" aria-hidden="true" /> Loading {what}…
      {slow && (
        <div className="muted" style={{ marginTop: 4 }}>
          Still waiting — the API runs on a free instance that goes to sleep when idle and can take up to a
          minute to wake and load its 10 models. It retries automatically; no action needed.
        </div>
      )}
    </div>
  );
}

export function ApiErrorBox({ message, onRetry }: { message: string; onRetry?: () => void }) {
  return (
    <div className="err-box" role="alert">
      <div><strong>Could not reach the EarlyDX API.</strong> {message}</div>
      <div className="muted" style={{ marginTop: 6 }}>
        If the backend was idle it may still be starting — retry in a few seconds.
      </div>
      {onRetry && (
        <button type="button" className="btn-secondary" onClick={onRetry} style={{ marginTop: 10 }}>
          Retry
        </button>
      )}
    </div>
  );
}
