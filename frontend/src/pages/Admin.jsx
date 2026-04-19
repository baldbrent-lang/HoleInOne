import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, API_BASE } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

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

  const [newName, setNewName] = useState("");
  const [newLocation, setNewLocation] = useState("");
  const [newPar3s, setNewPar3s] = useState("3:173, 7:165, 12:205, 16:192");
  const [newMinutes, setNewMinutes] = useState(14);
  const [newLivestream, setNewLivestream] = useState("");

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

  async function createCourse(e) {
    e.preventDefault();
    try {
      const parsed = parseHolesAndYards(newPar3s);
      await api.createCourse(adminPassword, {
        name: newName,
        location: newLocation,
        par3_holes: parsed.holes,
        hole_yardages: parsed.yardages,
        minutes_per_hole: Number(newMinutes),
        tee_sheet_provider: "mock",
        tee_sheet_config: {},
        livestream_url: newLivestream.trim() || null,
      });
      setNewName(""); setNewLocation(""); setNewLivestream("");
      showToast("Course added");
      load();
    } catch (e) { setError(e.message); }
  }

  async function runSimulate(courseToken, withHio) {
    try {
      const tts = await api.teeTimes(courseToken);
      const first = tts[0];
      if (!first) throw new Error("no tee times for today");
      await api.simulateRound(adminPassword, first.id, withHio);
      showToast(withHio ? "Injected round with a hole-in-one" : "Injected synthetic round");
      load();
    } catch (e) { showToast(`Error: ${e.message}`); }
  }

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
        <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
          <div className="logo"><Icon name="flag" /></div>
          <div>
            <h1>GolfReelz</h1>
            <div className="tag">Operator Console</div>
          </div>
        </div>
      </div>

      <div className="nav">
        <Link to="/admin" className="active">Dashboard</Link>
        <Link to="/admin/participants">Participants</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
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
            <div className="label">Participants</div>
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
          <h3 style={{ marginBottom: 10 }}>Participants by course</h3>
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

      <div className="card">
        <h3 style={{ marginBottom: 12 }}>Courses</h3>
        {courses.length === 0 && <div className="muted small">No courses yet. Add your first below.</div>}
        {courses.length > 0 && (
          <table>
            <thead>
              <tr><th>Course</th><th>Par-3s</th><th>QR</th><th>Demo tools</th></tr>
            </thead>
            <tbody>
              {courses.map((c) => (
                <tr key={c.id}>
                  <td>
                    <b>{c.name}</b>
                    <div className="small muted">{c.location || "—"}</div>
                  </td>
                  <td className="small">
                    <ParThreeEditor course={c} adminPassword={adminPassword} onSaved={(msg) => { showToast(msg); load(); }} />
                  </td>
                  <td>
                    <div className="qr-block">
                      <a href={api.courseQrUrl(c.qr_token)} target="_blank" rel="noreferrer">
                        <img src={api.courseQrUrl(c.qr_token)} alt="qr" width={84} height={84} />
                      </a>
                      <a
                        className="btn secondary small"
                        href={api.courseQrUrl(c.qr_token)}
                        download={`golfreelz-${c.name.replace(/\s+/g, "_")}.png`}
                        style={{ textAlign: "center" }}
                      >
                        <Icon name="download" size={14} /> Download
                      </a>
                      <CopyLink url={`${window.location.origin}/r/${c.qr_token}`} onCopy={showToast} />
                    </div>
                  </td>
                  <td>
                    <div className="stack" style={{ gap: 6 }}>
                      <LivestreamEditor course={c} adminPassword={adminPassword} onSaved={(msg) => { showToast(msg); load(); }} />
                      <Link className="btn secondary small" to={`/admin/participants?course_id=${c.id}`} style={{ textAlign: "center" }}>
                        <Icon name="users" size={14} /> View participants
                      </Link>
                      <button className="secondary small" onClick={() => runSimulate(c.qr_token, false)}>
                        Simulate round
                      </button>
                      <button className="small" onClick={() => runSimulate(c.qr_token, true)}>
                        <Icon name="sparkle" size={14} /> Simulate hole-in-one
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <form className="card" onSubmit={createCourse}>
        <h3>Add a course</h3>
        <p className="small muted" style={{ marginBottom: 14 }}>
          Generates a unique QR token and populates 60 mock tee times for the day.
        </p>
        <div className="row">
          <div className="field"><label>Name</label><input value={newName} onChange={(e) => setNewName(e.target.value)} placeholder="Pebble Beach" required /></div>
          <div className="field"><label>Location</label><input value={newLocation} onChange={(e) => setNewLocation(e.target.value)} placeholder="Pebble Beach, CA" /></div>
        </div>
        <div className="row">
          <div className="field">
            <label>Par-3 holes + yardages</label>
            <input value={newPar3s} onChange={(e) => setNewPar3s(e.target.value)} placeholder="3:173, 7:165, 12:205, 16:192" />
            <div className="hint small muted">Format: <code>hole:yards</code>, comma-separated. Yards optional.</div>
          </div>
          <div className="field"><label>Minutes per hole</label><input type="number" value={newMinutes} onChange={(e) => setNewMinutes(e.target.value)} /></div>
        </div>
        <div className="field">
          <label>Livestream URL <span className="muted small">(optional)</span></label>
          <input
            type="url"
            value={newLivestream}
            onChange={(e) => setNewLivestream(e.target.value)}
            placeholder="https://youtube.com/live/... or https://twitch.tv/..."
          />
        </div>
        <button>Create course</button>
      </form>

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
                {new Date(c.captured_at).toLocaleString()}{c.note ? ` · ${c.note}` : ""}
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

function parseHolesAndYards(text) {
  const holes = [];
  const yardages = {};
  for (const raw of text.split(",")) {
    const piece = raw.trim();
    if (!piece) continue;
    const [h, y] = piece.split(":").map((s) => s.trim());
    const hole = Number(h);
    if (!hole) continue;
    holes.push(hole);
    if (y) {
      const yards = Number(y);
      if (yards) yardages[String(hole)] = yards;
    }
  }
  return { holes, yardages };
}

function formatHolesAndYards(holes = [], yardages = {}) {
  return holes
    .map((h) => {
      const y = yardages[String(h)];
      return y ? `${h}:${y}` : String(h);
    })
    .join(", ");
}

function ParThreeEditor({ course, adminPassword, onSaved }) {
  const [editing, setEditing] = useState(false);
  const initial = formatHolesAndYards(course.par3_holes, course.hole_yardages);
  const [value, setValue] = useState(initial);
  const [saving, setSaving] = useState(false);

  if (!editing) {
    return (
      <button
        className="ghost small"
        onClick={() => { setValue(initial); setEditing(true); }}
        style={{ width: "auto", padding: 0 }}
        title="Edit par-3 holes and yardages"
      >
        {initial || <span className="muted">none</span>} ✎
      </button>
    );
  }

  async function save() {
    setSaving(true);
    try {
      const parsed = parseHolesAndYards(value);
      await api.updateCourse(adminPassword, course.id, {
        par3_holes: parsed.holes,
        hole_yardages: parsed.yardages,
      });
      setEditing(false);
      onSaved?.("Par-3s updated");
    } catch (e) {
      onSaved?.(`Error: ${e.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="stack" style={{ gap: 4 }}>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="3:173, 7:165, 12:205"
        style={{ fontSize: "0.8rem", padding: "6px 8px" }}
        autoFocus
      />
      <div style={{ display: "flex", gap: 4 }}>
        <button className="small" onClick={save} disabled={saving} style={{ flex: 1 }}>
          {saving ? "Saving…" : "Save"}
        </button>
        <button className="ghost small" onClick={() => setEditing(false)}>Cancel</button>
      </div>
    </div>
  );
}

function LivestreamEditor({ course, adminPassword, onSaved }) {
  const [editing, setEditing] = useState(false);
  const [value, setValue] = useState(course.livestream_url || "");
  const [saving, setSaving] = useState(false);

  if (!editing) {
    if (course.livestream_url) {
      return (
        <div style={{ display: "flex", gap: 4 }}>
          <a
            className="btn secondary small"
            href={course.livestream_url}
            target="_blank"
            rel="noreferrer"
            style={{ flex: 1, textAlign: "center" }}
          >
            <LiveDot /> Watch live
          </a>
          <button className="ghost small" onClick={() => setEditing(true)} title="Edit livestream">
            ✎
          </button>
        </div>
      );
    }
    return (
      <button className="ghost small" onClick={() => setEditing(true)}>
        + Add livestream URL
      </button>
    );
  }

  async function save() {
    setSaving(true);
    try {
      await api.updateCourse(adminPassword, course.id, { livestream_url: value.trim() || null });
      setEditing(false);
      onSaved?.(value.trim() ? "Livestream saved" : "Livestream cleared");
    } catch (e) {
      onSaved?.(`Error: ${e.message}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="stack" style={{ gap: 4 }}>
      <input
        value={value}
        onChange={(e) => setValue(e.target.value)}
        placeholder="https://..."
        style={{ fontSize: "0.8rem", padding: "6px 8px" }}
        autoFocus
      />
      <div style={{ display: "flex", gap: 4 }}>
        <button className="small" onClick={save} disabled={saving} style={{ flex: 1 }}>
          {saving ? "Saving…" : "Save"}
        </button>
        <button className="ghost small" onClick={() => { setEditing(false); setValue(course.livestream_url || ""); }}>
          Cancel
        </button>
      </div>
    </div>
  );
}

function LiveDot() {
  return (
    <span
      style={{
        display: "inline-block",
        width: 8, height: 8, borderRadius: "50%",
        background: "var(--red-500)",
        marginRight: 6,
        animation: "pulse 1.6s ease-in-out infinite",
        boxShadow: "0 0 0 0 rgba(239, 68, 68, 0.4)",
      }}
    />
  );
}

function CopyLink({ url, onCopy }) {
  // Display a compact version like "….replit.dev/r/c_9mDNc…" but copy the full URL.
  const MAX = 28;
  const noScheme = url.replace(/^https?:\/\//, "");
  const slash = noScheme.indexOf("/r/");
  const host = slash >= 0 ? noScheme.slice(0, slash) : noScheme;
  const path = slash >= 0 ? noScheme.slice(slash) : "";
  const compactHost = host.length > 14 ? `…${host.slice(-12)}` : host;
  const compactPath = path.length > 14 ? `${path.slice(0, 6)}…${path.slice(-4)}` : path;
  let label = `${compactHost}${compactPath}`;
  if (label.length > MAX) label = label.slice(0, MAX - 1) + "…";

  async function copy() {
    try {
      await navigator.clipboard.writeText(url);
      onCopy?.("Registration link copied");
    } catch {
      window.prompt("Copy the registration link:", url);
    }
  }

  return (
    <button
      type="button"
      className="secondary small"
      onClick={copy}
      title={`Copy ${url}`}
      style={{ width: "100%", fontFamily: "JetBrains Mono, monospace", fontSize: "0.78rem" }}
    >
      <Icon name="share" size={13} /> {label}
    </button>
  );
}
