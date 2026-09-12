/** Shared loading / API-error states so every page behaves the same while
 * data is in flight. The backend runs on a free Render instance that sleeps
 * when idle, so the loading copy explains a possible cold-start delay. */
export function Loading({ what = "data" }: { what?: string }) {
  return (
    <p className="muted state-line" role="status" aria-live="polite">
      Loading {what}… (the API may take up to a minute to wake if it has been idle)
    </p>
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
