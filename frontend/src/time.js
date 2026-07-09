// Time formatting helpers.
//
// The backend stores and serializes every datetime as *naive UTC* (e.g.
// "2026-07-09T01:30:00") — it comes from datetime.utcnow() /
// datetime.utcfromtimestamp() and .isoformat() emits no timezone suffix.
//
// A datetime string with no timezone is parsed by the browser as LOCAL
// time, so `new Date("2026-07-09T01:30:00")` reads the *UTC clock* as if it
// were local — showing 1:30 AM for an event that actually happened at 9:30
// PM EDT. We tag tz-less strings as UTC so the browser converts them to the
// viewer's real local time.

// True when the ISO string already carries a timezone (trailing Z or ±hh:mm).
const HAS_TZ = /[zZ]$|[+-]\d{2}:?\d{2}$/;

// Parse an API datetime string into a Date, treating a tz-less value as UTC.
// Returns null for empty/invalid input. Passes a Date through unchanged.
export function parseApiDate(iso) {
  if (!iso) return null;
  if (iso instanceof Date) return iso;
  const s = String(iso);
  const d = new Date(HAS_TZ.test(s) ? s : s + "Z");
  return isNaN(d.getTime()) ? null : d;
}

// Format an API datetime in the viewer's local timezone. Extra args are
// forwarded to Date.prototype.toLocaleString (locale, options). Returns an
// em dash for missing/invalid values.
export function fmtDateTime(iso, ...args) {
  const d = parseApiDate(iso);
  if (!d) return "—";
  return d.toLocaleString(...args);
}
