import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, API_BASE } from "../api.js";

const ADMIN_PW_STORAGE = "parone.adminPassword";

export default function Admin() {
  const [adminPassword, setAdminPassword] = useState(() => localStorage.getItem(ADMIN_PW_STORAGE) || "");
  const [authed, setAuthed] = useState(false);
  const [courses, setCourses] = useState([]);
  const [stats, setStats] = useState(null);
  const [flagged, setFlagged] = useState([]);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);

  const [newName, setNewName] = useState("");
  const [newLocation, setNewLocation] = useState("");
  const [newPar3s, setNewPar3s] = useState("3,7,12,16");
  const [newMinutes, setNewMinutes] = useState(14);

  async function load(key = adminPassword) {
    try {
      const [c, s, f] = await Promise.all([api.listCourses(key), api.stats(key), api.flaggedClips(key)]);
      setCourses(c);
      setStats(s);
      setFlagged(f);
      setAuthed(true);
      localStorage.setItem(ADMIN_PW_STORAGE, key);
      setError(null);
    } catch (e) {
      setError(e.message);
      setAuthed(false);
    }
  }

  useEffect(() => {
    if (adminPassword) load(adminPassword);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function createCourse(e) {
    e.preventDefault();
    try {
      await api.createCourse(adminPassword, {
        name: newName,
        location: newLocation,
        par3_holes: newPar3s.split(",").map((s) => Number(s.trim())).filter(Boolean),
        minutes_per_hole: Number(newMinutes),
        tee_sheet_provider: "mock",
        tee_sheet_config: {},
      });
      setNewName("");
      setNewLocation("");
      load();
    } catch (e) {
      setError(e.message);
    }
  }

  async function simulate(courseToken) {
    try {
      const tts = await api.teeTimes(courseToken);
      const first = tts[0];
      if (!first) throw new Error("no tee times for today");
      await api.simulateRound(adminPassword, first.id, false);
      setToast(`Injected synthetic round on tee time #${first.id}`);
      setTimeout(() => setToast(null), 2500);
      load();
    } catch (e) {
      setToast(`Error: ${e.message}`);
      setTimeout(() => setToast(null), 3000);
    }
  }

  async function simulateHIO(courseToken) {
    try {
      const tts = await api.teeTimes(courseToken);
      const first = tts[0];
      if (!first) throw new Error("no tee times for today");
      await api.simulateRound(adminPassword, first.id, true);
      setToast(`Injected round with hole-in-one`);
      setTimeout(() => setToast(null), 2500);
      load();
    } catch (e) {
      setToast(`Error: ${e.message}`);
      setTimeout(() => setToast(null), 3000);
    }
  }

  if (!authed) {
    return (
      <div className="wrap">
        <div className="brand"><div className="dot" /><h1>Par One — Admin</h1></div>
        <div className="card">
          <div className="field">
            <label>Password</label>
            <input
              type="password"
              value={adminPassword}
              onChange={(e) => setAdminPassword(e.target.value)}
              placeholder="enter admin password"
              autoFocus
            />
          </div>
          {error && <p className="small" style={{ color: "var(--danger)" }}>{error}</p>}
          <button onClick={() => load(adminPassword)}>Sign in</button>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap wide">
      <div className="brand"><div className="dot" /><h1>Par One — Admin</h1></div>
      <div className="nav">
        <Link to="/admin" className="active">Dashboard</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
        <a onClick={() => { localStorage.removeItem(ADMIN_PW_STORAGE); window.location.reload(); }} href="#">Sign out</a>
      </div>

      {stats && (
        <div className="grid">
          <div className="card">
            <div className="muted small">Participants</div>
            <h2>{stats.participants.total}</h2>
            <div className="small muted">
              Today {stats.participants.day} · Week {stats.participants.week} · Month {stats.participants.month}
            </div>
          </div>
          <div className="card">
            <div className="muted small">Revenue</div>
            <h2>${(stats.revenue_cents / 100).toFixed(2)}</h2>
          </div>
          <div className="card">
            <div className="muted small">Clip status</div>
            <div className="small">
              {Object.entries(stats.clips_by_status).length === 0 && <span className="muted">No clips yet</span>}
              {Object.entries(stats.clips_by_status).map(([k, v]) => (
                <div key={k}>{k}: <b>{v}</b></div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="card">
        <h2>Courses</h2>
        <table>
          <thead>
            <tr><th>Name</th><th>Location</th><th>Par-3s</th><th>QR</th><th>Tools</th></tr>
          </thead>
          <tbody>
            {courses.map((c) => (
              <tr key={c.id}>
                <td><b>{c.name}</b></td>
                <td className="muted small">{c.location}</td>
                <td className="small">{(c.par3_holes || []).join(", ")}</td>
                <td>
                  <a href={api.courseQrUrl(c.qr_token)} target="_blank" rel="noreferrer">
                    <img
                      src={api.courseQrUrl(c.qr_token)}
                      alt="qr"
                      width={72}
                      height={72}
                      style={{ background: "white", padding: 4, borderRadius: 6 }}
                    />
                  </a>
                  <div style={{ marginTop: 6 }}>
                    <a
                      className="btn secondary small"
                      href={api.courseQrUrl(c.qr_token)}
                      download={`parone-${c.name.replace(/\s+/g, "_")}.png`}
                      style={{ textAlign: "center", display: "block", padding: "6px 8px", fontSize: "0.8rem" }}
                    >
                      Download QR
                    </a>
                  </div>
                  <div className="small muted" style={{ marginTop: 4 }}>
                    /r/{c.qr_token.slice(0, 10)}…
                  </div>
                </td>
                <td>
                  <button className="secondary small" onClick={() => simulate(c.qr_token)}>Simulate round</button>
                  <div style={{ height: 6 }} />
                  <button className="warn small" onClick={() => simulateHIO(c.qr_token)}>Simulate HIO</button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <form className="card" onSubmit={createCourse}>
        <h3>Add course</h3>
        <div className="row">
          <div className="field"><label>Name</label><input value={newName} onChange={(e) => setNewName(e.target.value)} required /></div>
          <div className="field"><label>Location</label><input value={newLocation} onChange={(e) => setNewLocation(e.target.value)} /></div>
        </div>
        <div className="row">
          <div className="field"><label>Par-3 holes (comma-separated)</label><input value={newPar3s} onChange={(e) => setNewPar3s(e.target.value)} /></div>
          <div className="field"><label>Minutes per hole</label><input type="number" value={newMinutes} onChange={(e) => setNewMinutes(e.target.value)} /></div>
        </div>
        <button>Create course</button>
      </form>

      <div className="card">
        <h3>Flagged / unassigned clips</h3>
        {flagged.length === 0 && <div className="muted small">None 🎉</div>}
        {flagged.map((c) => (
          <div key={c.id} className="clip" style={{ alignItems: "flex-start", flexWrap: "wrap" }}>
            <div
              className="thumb"
              style={c.thumbnail_url ? { backgroundImage: `url(${c.thumbnail_url})` } : {}}
            />
            <div className="meta">
              <b>Hole {c.hole_number}</b> · {c.camera_type}
              <div className="stats">
                <span className={`pill ${c.status === "flagged" ? "err" : "warn"}`}>{c.status}</span>
                {" "}{new Date(c.captured_at).toLocaleString()}
              </div>
              <div className="small muted">{c.note || "—"}</div>
              {c.candidates?.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  <div className="small muted" style={{ marginBottom: 4 }}>
                    Candidates in window:
                  </div>
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
                              `${import.meta.env.VITE_API_BASE || ""}/api/admin/clips/${c.id}/assign?participant_id=${cand.id}`,
                              { method: "POST", headers: { "X-Admin-Password": adminPassword } },
                            );
                            load();
                          } catch (e) {
                            setToast(`Error: ${e.message}`);
                            setTimeout(() => setToast(null), 3000);
                          }
                        }}
                      >
                        {cand.selfie_url && (
                          <img
                            src={cand.selfie_url}
                            alt=""
                            style={{ width: 24, height: 24, borderRadius: "50%", objectFit: "cover", verticalAlign: "middle", marginRight: 6 }}
                          />
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
