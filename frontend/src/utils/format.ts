// Presentation-only helpers. These never change the underlying feature key
// that gets sent to the backend — they only affect what the user sees.

/** Turn a raw model feature key (e.g. "BloodPressure", "T3_resin", "bp")
 * into a more readable label, without inventing any new information.
 * Acronym-like tokens (all uppercase, e.g. "BMI", "TSH") are left as-is. */
export function humanizeLabel(key: string): string {
  let s = key.replace(/[:_]/g, " ");
  s = s.replace(/([a-z0-9])([A-Z])/g, "$1 $2");
  s = s.replace(/\s+/g, " ").trim();
  if (!s) return key;
  return s
    .split(" ")
    .map((word) => {
      if (/^[A-Z0-9()%.]+$/.test(word) && word.length > 1) return word;
      return word.charAt(0).toUpperCase() + word.slice(1);
    })
    .join(" ");
}

/** Turn a raw categorical option value (e.g. "Public_Transportation") into a
 * readable option label. The value sent to the backend is unchanged. */
export function humanizeOption(value: string): string {
  const s = value.replace(/_/g, " ").trim();
  return s.length ? s.charAt(0).toUpperCase() + s.slice(1) : s;
}
