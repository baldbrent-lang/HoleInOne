import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, API_BASE } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import { fmtDateTime } from "../time.js";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function readStoredPassword() {
  const v = localStorage.getItem(ADMIN_PW_STORAGE);
  if (v) return v;
  const legacy = localStorage.getItem(LEGACY_ADMIN_PW_STORAGE);
  if (legacy) {
    localStorage.setItem(ADMIN_PW_STORAGE, legacy);
    localStorage.removeItem(LEGACY_ADMIN_PW_STORAGE);
    return legacy;
  }
  return "";
}

function today() {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
}

export default function Admin() {
  const [adminPassword, setAdminPassword] = useState(() => readStoredPassword());
  const [authed, setAuthed] = useState(false);
  const [courses, setCourses] = useState([]);
  const [stats, setStats] = useState(null);
  const [flagged, setFlagged] = useState([]);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);

  async function load(key = adminPassword) {
    try {
      const [c, s, f] = await Promise.all([api.listCourses(key), api.stats(key), api.flaggedClips(key)]);
      setCourses(c); setStats(s); setFlagged(f);
      setAuthed(true);
      localStorage.setItem(ADMIN_PW_STORAGE, key);
      setError(null);
    } catch (e) {
      setError(e.message); setAuthed(false);
    }
  }

  useEffect(() => { if (adminPassword) load(adminPassword); /* eslint-disable-next-line */ }, []);

  function showToast(msg) { setToast(msg); setTimeout(() => setToast(null), 2500); }

  if (!authed) {
    return (
      <div className="wrap">
        <Brand subtitle="Operator Console" />
        <div className="card" style={{ maxWidth: 420, margin: "40px auto 0" }}>
          <h2 style={{ marginBottom: 4 }}>Sign in</h2>
          <p className="small muted" style={{ marginBottom: 18 }}>Admin password required.</p>
          <div className="field">
            <label>Password</label>
            <input
              type="password"
              value={adminPassword}
              onChange={(e) => setAdminPassword(e.target.value)}
              placeholder="Enter admin password"
              onKeyDown={(e) => e.key === "Enter" && load(adminPassword)}
              autoFocus
            />
          </div>
          {error && <p className="err-text small">{error}</p>}
          <button onClick={() => load(adminPassword)}>Sign in</button>
        </div>
      </div>
    );
  }

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
          <div className="tag">Operator Console</div>
        </div>
      </div>

      <div className="nav">
        <Link to="/admin" className="active">Dashboard</Link>
        <Link to="/admin/participants">Players</Link>
        <Link to="/admin/courses">Courses</Link>
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production">Production</Link>
        <Link to="/admin/produced-clips">Produced Clips</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
        <button
          className="ghost"
          onClick={() => { localStorage.removeItem(ADMIN_PW_STORAGE); window.location.reload(); }}
          style={{ marginLeft: "auto" }}
        >
          Sign out
        </button>
      </div>

      {stats && (
        <div className="stat-grid">
          <Link to="/admin/participants" className="stat" style={{ textDecoration: "none", color: "inherit" }}>
            <div className="icon-bg"><Icon name="users" size={16} /></div>
            <div className="label">Players</div>
            <div className="value">{stats.participants.total}</div>
            <div className="sub">
              <Link to={`/admin/participants?date=${today()}`}>Today {stats.participants.day}</Link>
              {" · "}
              <Link to={`/admin/participants`}>Week {stats.participants.week}</Link>
              {" · Month "}{stats.participants.month}
            </div>
          </Link>
          <div className="stat">
            <div className="icon-bg"><Icon name="dollar" size={16} /></div>
            <div className="label">Revenue</div>
            <div className="value">${(stats.revenue_cents / 100).toFixed(2)}</div>
            <div className="sub">Lifetime gross</div>
          </div>
          <div className="stat">
            <div className="icon-bg"><Icon name="chart" size={16} /></div>
            <div className="label">Clips</div>
            <div className="value">
              {Object.values(stats.clips_by_status).reduce((a, b) => a + b, 0) || 0}
            </div>
            <div className="sub">
              {Object.entries(stats.clips_by_status).length === 0
                ? "No clips yet"
                : Object.entries(stats.clips_by_status).map(([k, v]) => `${k}:${v}`).join("  ·  ")}
            </div>
          </div>
        </div>
      )}

      {stats?.by_course?.length > 0 && (
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>Players by course</h3>
          <div className="chip-row">
            {stats.by_course.map((row) => {
              const course = courses.find((c) => c.name === row.course);
              return (
                <Link
                  key={row.course}
                  to={course ? `/admin/participants?course_id=${course.id}` : "/admin/participants"}
                  className="chip"
                >
                  {row.course} <b style={{ marginLeft: 6 }}>{row.participants}</b>
                </Link>
              );
            })}
          </div>
        </div>
      )}

      <TestEmailCard adminPassword={adminPassword} onToast={showToast} />

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>Manual review queue</h3>
        <p className="small muted" style={{ marginBottom: 14 }}>
          Clips the auto-matcher couldn't assign with confidence. Tap a candidate to assign.
        </p>
        {flagged.length === 0 && <div className="muted small">Nothing flagged. ✨</div>}
        {flagged.map((c) => (
          <div key={c.id} className="clip" style={{ alignItems: "flex-start", flexWrap: "wrap" }}>
            <div className="thumb" style={c.thumbnail_url ? { backgroundImage: `url(${c.thumbnail_url})` } : {}} />
            <div className="meta">
              <div className="inline" style={{ gap: 8 }}>
                <b>Hole {c.hole_number}</b>
                <span className="pill dark" style={{ textTransform: "lowercase" }}>{c.camera_type.replace("_", " ")}</span>
                <span className={`pill ${c.status === "flagged" ? "err" : "warn"}`}>{c.status}</span>
              </div>
              <div className="small muted" style={{ marginTop: 4 }}>
                {fmtDateTime(c.captured_at)}{c.note ? ` · ${c.note}` : ""}
              </div>
              {c.candidates?.length > 0 && (
                <div style={{ marginTop: 10 }}>
                  <div className="tiny upper muted" style={{ marginBottom: 6 }}>Candidates in window</div>
                  <div className="chip-row">
                    {c.candidates.map((cand) => (
                      <button
                        key={cand.id}
                        type="button"
                        className="chip"
                        title={`Assign to ${cand.name}`}
                        onClick={async () => {
                          try {
                            await fetch(
                              `${API_BASE}/api/admin/clips/${c.id}/assign?participant_id=${cand.id}`,
                              { method: "POST", headers: { "X-Admin-Password": adminPassword } },
                            );
                            showToast(`Assigned to ${cand.name}`);
                            load();
                          } catch (e) {
                            showToast(`Error: ${e.message}`);
                          }
                        }}
                      >
                        {cand.selfie_url && (
                          <img src={cand.selfie_url} alt="" style={{ width: 22, height: 22, borderRadius: "50%", objectFit: "cover" }} />
                        )}
                        {cand.name}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function TestEmailCard({ adminPassword, onToast }) {
  const [to, setTo] = useState("");
  const [sending, setSending] = useState(false);
  const [result, setResult] = useState(null);

  async function send() {
    setSending(true);
    setResult(null);
    try {
      const r = await api.sendTestEmail(adminPassword, { to });
      setResult({ ok: true, provider: r.provider, to: r.to });
      onToast?.(`Test sent via ${r.provider}`);
    } catch (e) {
      setResult({ ok: false, error: e.message });
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="card">
      <h3 style={{ marginBottom: 6 }}>Test email delivery</h3>
      <p className="small muted" style={{ marginBottom: 10 }}>
        Drops a single message at the address below. Uses SMTP if
        <code> SMTP_HOST / SMTP_USER / SMTP_PASSWORD</code> are set in Secrets,
        else SendGrid, else mock-log.
      </p>
      <div className="row">
        <div className="field" style={{ flex: 2 }}>
          <label>Send test to</label>
          <input
            type="email"
            value={to}
            onChange={(e) => setTo(e.target.value)}
            placeholder="you@example.com"
          />
        </div>
        <div style={{ alignSelf: "end" }}>
          <button type="button" disabled={sending || !to} onClick={send}>
            {sending ? "Sending…" : "Send test"}
          </button>
        </div>
      </div>
      {result?.ok && (
        <p className="small" style={{ color: "var(--emerald-700)" }}>
          ✓ Sent via <b>{result.provider}</b> to {result.to}.
          {result.provider === "mock" && " (Set SMTP secrets to actually deliver.)"}
        </p>
      )}
      {result && !result.ok && <p className="err-text small">{result.error}</p>}
    </div>
  );
}
