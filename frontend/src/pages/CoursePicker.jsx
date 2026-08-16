import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

/**
 * Does this course match what has been typed so far?
 *
 * Prefix matching, not "contains": typing s-n-e narrows to Snee Farm and
 * keeps it until a letter stops following, which is how someone hunting
 * for a name they already know expects a list to behave. "contains"
 * would surface Pebble Beach for "e" and make early keystrokes useless.
 *
 * Any WORD may start the match, so "farm" still finds Snee Farm Country
 * Club — a golfer thinks of it as Snee Farm, not as its full name — and
 * the town matches too, since "carrollton" is a fair way to look for a
 * course you know by where it is rather than what it is called.
 */
function matches(course, q) {
  if (!q) return true;
  const fields = [course.name || "", course.location || ""];
  return fields.some((f) => {
    const t = f.toLowerCase();
    if (t.startsWith(q)) return true;
    // Split on whitespace AND punctuation, so "country" matches in
    // "Snee Farm (Country) Club" and "st" finds "St. Andrews".
    return t.split(/[^a-z0-9]+/).some((w) => w && w.startsWith(q));
  });
}

export default function CoursePicker() {
  const [courses, setCourses] = useState(null);
  const [query, setQuery] = useState("");

  useEffect(() => {
    api.listPublicCourses().then(setCourses).catch(() => setCourses([]));
  }, []);

  const q = query.trim().toLowerCase();
  const shown = (courses || []).filter((c) => matches(c, q));

  return (
    <div className="wrap">
      <Brand subtitle="Pick a course" />

      <div className="card">
        <h2 style={{ marginBottom: 4 }}>Where are you playing?</h2>
        <p className="small muted" style={{ marginBottom: 16 }}>
          Tap a course to start your registration.
        </p>

        {courses !== null && courses.length > 3 && (
          <div className="field">
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Search by course or town"
              aria-label="Search courses"
            />
          </div>
        )}

        {courses === null ? (
          <>
            <div className="shimmer" style={{ height: 64, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 64, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 64 }} />
          </>
        ) : courses.length === 0 ? (
          <div className="muted small">No courses available yet.</div>
        ) : shown.length === 0 ? (
          <div className="center" style={{ padding: "18px 0" }}>
            <p className="muted small" style={{ marginBottom: 10 }}>
              No courses match &ldquo;{query}&rdquo;.
            </p>
            <button
              className="ghost small"
              style={{ width: "auto" }}
              onClick={() => setQuery("")}
            >
              Clear search
            </button>
          </div>
        ) : (
          <div className="stack" style={{ gap: 10 }}>
            {shown.map((c) => (
              <Link
                key={c.id}
                to={`/r/${c.qr_token}`}
                className="card"
                style={{
                  margin: 0,
                  display: "flex",
                  alignItems: "center",
                  gap: 14,
                  textDecoration: "none",
                  color: "inherit",
                  padding: 14,
                }}
              >
                <div
                  className="logo"
                  style={{ background: "var(--primary-soft)", color: "var(--emerald-700)", flexShrink: 0 }}
                  aria-hidden="true"
                >
                  <Icon name="flag" />
                </div>
                <div style={{ flex: 1, minWidth: 0 }}>
                  <b style={{ display: "block" }}>{c.name}</b>
                  <div className="small muted">{c.location || "—"}</div>
                </div>
                <span style={{ color: "var(--ink-soft)" }}>→</span>
              </Link>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
