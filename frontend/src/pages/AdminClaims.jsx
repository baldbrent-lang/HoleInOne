import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

const NEXT = { new: "contacted", contacted: "fulfilled", fulfilled: "new" };

/**
 * Prize claims waiting to be worked. Everything needed to act on one is
 * on the row — name, hole, contact details, address — so paying a winner
 * doesn't mean opening three other screens.
 */
export default function AdminClaims() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    try {
      setData(await api.listClaims(adminPassword));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (adminPassword) load();
  }, [adminPassword]);

  async function advance(c) {
    try {
      await api.setClaimStatus(adminPassword, c.id, NEXT[c.status] || "new");
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
        <Link to="/admin/claims" className="active">Claims</Link>
        <Link to="/admin/reviews">Reviews</Link>
      </div>

      <div className="card">
        <h2>Prize claims</h2>
        {error && <p className="small err-text">{error}</p>}
        {!data && !error && <div className="shimmer" style={{ height: 80 }} />}

        {data && (
          <p className="small muted">
            {data.count === 0
              ? "No claims yet. They arrive when a golfer follows the link in a confirmed hole-in-one email."
              : `${data.count} claim${data.count === 1 ? "" : "s"} · ${data.open} open`}
          </p>
        )}

        {data?.claims.map((c) => (
          <div
            key={c.id}
            style={{ borderTop: "1px solid var(--border)", padding: "12px 0" }}
          >
            <div style={{ display: "flex", justifyContent: "space-between", gap: 12 }}>
              <div>
                <b>{c.name}</b>
                {c.hole_number && (
                  <span className="small muted"> · hole {c.hole_number}</span>
                )}
                {c.course_name && (
                  <span className="small muted"> · {c.course_name}</span>
                )}
              </div>
              <div className="small muted" style={{ whiteSpace: "nowrap" }}>
                {new Date(c.created_at + "Z").toLocaleDateString()}
              </div>
            </div>

            <div className="small" style={{ marginTop: 4 }}>
              {c.email && <span>{c.email}</span>}
              {c.email && c.mobile && <span className="muted"> · </span>}
              {c.mobile && <span>{c.mobile}</span>}
            </div>
            {c.mailing_address && (
              <div className="small muted" style={{ whiteSpace: "pre-wrap", marginTop: 4 }}>
                {c.mailing_address}
              </div>
            )}
            {c.note && <p style={{ margin: "6px 0 0" }}>{c.note}</p>}

            <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
              <span className={`pill ${c.status === "fulfilled" ? "" : "warn"}`}>
                {c.status}
              </span>
              <button className="ghost small" onClick={() => advance(c)}>
                Mark {NEXT[c.status] || "new"}
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
