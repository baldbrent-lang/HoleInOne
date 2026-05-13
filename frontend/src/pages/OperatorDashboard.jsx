import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, getOperatorToken, setOperatorToken } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

export default function OperatorDashboard() {
  const nav = useNavigate();
  const [data, setData] = useState(null);
  const [participants, setParticipants] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!getOperatorToken()) {
      nav("/operator/login", { replace: true });
      return;
    }
    api.operatorDashboard().then(setData).catch((e) => {
      if (String(e.message).startsWith("401")) {
        setOperatorToken("");
        nav("/operator/login", { replace: true });
      } else {
        setError(e.message);
      }
    });
    api.operatorParticipants().then(setParticipants).catch(() => setParticipants([]));
  }, [nav]);

  function logout() {
    setOperatorToken("");
    nav("/operator/login", { replace: true });
  }

  if (error) {
    return (
      <div className="wrap"><Brand /><div className="card err-text">{error}</div></div>
    );
  }
  if (!data) {
    return (
      <div className="wrap wide">
        <Brand subtitle="Operator portal" />
        <div className="card"><div className="shimmer" style={{ height: 200 }} /></div>
      </div>
    );
  }

  const s = data.stats;

  return (
    <div className="wrap wide">
      <div className="brand" style={{ justifyContent: "space-between", width: "100%" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 14 }}>
          <div
            style={{
              height: 60,
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
            }}
          >
            <img
              src="/golfreelz-logo.png"
              alt="GolfReelz"
              style={{ height: 100, width: "auto", display: "block" }}
            />
          </div>
          <div className="tag">{data.course.name} · operator</div>
        </div>
        <button className="ghost small" onClick={logout} style={{ width: "auto" }}>Sign out</button>
      </div>

      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="users" size={14} /> Today at {data.course.name}</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3vw, 2.1rem)" }}>
          {s.participants_today} golfer{s.participants_today === 1 ? "" : "s"} registered today
        </h1>
        <p>{s.aces_pending > 0 ? `${s.aces_pending} ace claim${s.aces_pending === 1 ? "" : "s"} under review` : "Quiet on the cup-cam — no aces pending review."}</p>
      </div>

      <div className="stat-grid">
        <div className="stat">
          <div className="icon-bg"><Icon name="users" size={16} /></div>
          <div className="label">Golfers</div>
          <div className="value">{s.participants_today}</div>
          <div className="sub">today · {s.participants_week} this week · {s.participants_month} this month</div>
        </div>
        <div className="stat">
          <div className="icon-bg"><Icon name="dollar" size={16} /></div>
          <div className="label">Revenue (lifetime)</div>
          <div className="value">${(s.revenue_cents / 100).toFixed(2)}</div>
          <div className="sub">{s.participants_paid_total} paid registrations</div>
        </div>
        <div className="stat">
          <div className="icon-bg"><Icon name="chart" size={16} /></div>
          <div className="label">Clips delivered</div>
          <div className="value">{s.clips_total}</div>
          <div className="sub">{s.aces_total} ace{s.aces_total === 1 ? "" : "s"} confirmed lifetime</div>
        </div>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 10 }}>Recent participants</h3>
        {participants === null ? (
          <div className="shimmer" style={{ height: 80 }} />
        ) : participants.length === 0 ? (
          <div className="muted small">No registrations yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Golfer</th>
                <th>Tee time</th>
                <th>Status</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {participants.slice(0, 25).map((p) => (
                <tr key={p.id}>
                  <td>
                    <div className="inline" style={{ gap: 10 }}>
                      {p.selfie_url ? (
                        <img src={p.selfie_url} alt="" style={{ width: 32, height: 42, objectFit: "cover", borderRadius: 6, border: "1px solid var(--border)" }} />
                      ) : (
                        <div style={{ width: 32, height: 42, borderRadius: 6, background: "var(--surface-alt)" }} />
                      )}
                      <div>
                        <b>{p.name}</b>
                        <div className="small muted">{p.mobile || p.email}</div>
                      </div>
                    </div>
                  </td>
                  <td className="small">
                    {new Date(p.tee_time).toLocaleString([], {
                      weekday: "short", month: "short", day: "numeric",
                      hour: "numeric", minute: "2-digit",
                    })}
                  </td>
                  <td>
                    {p.paid ? <span className="pill ok">Paid</span>
                      : p.refunded_at ? <span className="pill warn">Refunded</span>
                      : <span className="pill err">Unpaid</span>}
                  </td>
                  <td>
                    <a className="btn secondary small" href={p.gallery_url} target="_blank" rel="noreferrer">
                      Gallery ↗
                    </a>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
