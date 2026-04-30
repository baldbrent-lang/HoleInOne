import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

export default function CoursePicker() {
  const [courses, setCourses] = useState(null);

  useEffect(() => {
    api.listPublicCourses().then(setCourses).catch(() => setCourses([]));
  }, []);

  return (
    <div className="wrap">
      <Brand subtitle="Pick a course" />

      <div className="card">
        <h2 style={{ marginBottom: 4 }}>Where are you playing?</h2>
        <p className="small muted" style={{ marginBottom: 16 }}>
          Tap a course to start your registration.
        </p>

        {courses === null ? (
          <>
            <div className="shimmer" style={{ height: 64, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 64, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 64 }} />
          </>
        ) : courses.length === 0 ? (
          <div className="muted small">No courses available yet.</div>
        ) : (
          <div className="stack" style={{ gap: 10 }}>
            {courses.map((c) => (
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
