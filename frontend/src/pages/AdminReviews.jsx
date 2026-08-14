import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

/**
 * What golfers said, newest first. Read-only apart from the publish
 * toggle, which is what marks a review as OK to show on the site.
 */
export default function AdminReviews() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    try {
      setData(await api.listReviews(adminPassword));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (adminPassword) load();
  }, [adminPassword]);

  async function togglePublish(r) {
    try {
      await api.setReviewPublished(adminPassword, r.id, !r.published);
      load();
    } catch (e) {
      setError(e.message);
    }
  }

  return (
    <div className="wrap">
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants">Players</Link>
        <Link to="/admin/reviews" className="active">Reviews</Link>
        <Link to="/admin/courses">Courses</Link>
      </div>

      <div className="card">
        <h2>Reviews</h2>
        {error && <p className="small err-text">{error}</p>}
        {!data && !error && <div className="shimmer" style={{ height: 80 }} />}

        {data && (
          <p className="small muted">
            {data.count === 0
              ? "No reviews yet. They arrive via the thank-you email, a few hours after each gallery goes out."
              : `${data.count} review${data.count === 1 ? "" : "s"} · average ${data.average} ★`}
          </p>
        )}

        {data?.reviews.map((r) => (
          <div
            key={r.id}
            style={{
              borderTop: "1px solid var(--border)",
              padding: "12px 0",
            }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <div>
                <b>{"★".repeat(r.rating)}</b>
                <span className="small muted">{"★".repeat(5 - r.rating)}</span>
                {"  "}
                <b>{r.name}</b>
                {r.course_name && <span className="small muted"> · {r.course_name}</span>}
              </div>
              <div className="small muted" style={{ whiteSpace: "nowrap" }}>
                {new Date(r.created_at + "Z").toLocaleDateString()}
              </div>
            </div>
            {r.comment && <p style={{ margin: "6px 0 0" }}>{r.comment}</p>}
            <button
              className="ghost small"
              onClick={() => togglePublish(r)}
              style={{ marginTop: 8 }}
            >
              {r.published ? "Unpublish" : "Publish"}
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
