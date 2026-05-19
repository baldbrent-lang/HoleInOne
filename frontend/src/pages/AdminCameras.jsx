/**
 * Camera management page — register / pair / rotate-token / disable
 * the on-course capture devices. Phase 1 of the always-on hardware
 * integration; the event-trigger + upload-event endpoints the Pis
 * actually talk to land in phase 2.
 */
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function tsRel(iso) {
  if (!iso) return "—";
  // Backend serializes timestamps as naive UTC (no Z suffix). Without
  // a timezone marker, the browser parses them as local time, and
  // the diff against Date.now() comes out off by the user's UTC
  // offset (e.g. -5 hours in Central time = "-17985s ago"). Force
  // UTC interpretation by appending Z when no offset is present.
  const utcIso = /[Zz]$|[+-]\d{2}:?\d{2}$/.test(iso) ? iso : iso + "Z";
  const d = new Date(utcIso);
  const sec = Math.round((Date.now() - d.getTime()) / 1000);
  if (sec < 60) return `${sec}s ago`;
  if (sec < 3600) return `${Math.round(sec / 60)}m ago`;
  if (sec < 86400) return `${Math.round(sec / 3600)}h ago`;
  return `${Math.round(sec / 86400)}d ago`;
}

export default function AdminCameras() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  const [cameras, setCameras] = useState(null);
  const [courses, setCourses] = useState([]);
  const [error, setError] = useState(null);
  const [busy, setBusy] = useState({}); // {camera_id: true}
  const [revealedToken, setRevealedToken] = useState({}); // {camera_id: true}
  const [movingCam, setMovingCam] = useState(null); // camera_id whose move form is open
  const [moveDraft, setMoveDraft] = useState({ courseId: "", hole: "", role: "" });

  // Live-watch state: one camera at a time. `frameTick` increments every
  // ~100ms while watching, busting the <img> cache so we get the latest
  // JPEG. `frameAvailable` flips true once the Pi has uploaded its
  // first frame so we can hide the "Waiting…" placeholder.
  const [watchingCamId, setWatchingCamId] = useState(null);
  const [frameTick, setFrameTick] = useState(0);
  const [frameAvailable, setFrameAvailable] = useState(false);
  const watchingCamIdRef = useRef(null);

  // New-camera form state
  const [newCourseId, setNewCourseId] = useState("");
  const [newHole, setNewHole] = useState("");
  const [newRole, setNewRole] = useState("tee");
  const [newName, setNewName] = useState("");
  const [creating, setCreating] = useState(false);

  async function load() {
    try {
      const list = await api.listCameras(adminPassword);
      setCameras(list);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (!adminPassword) return;
    api.listCourses(adminPassword).then(setCourses).catch((e) => setError(e.message));
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminPassword]);

  // Tick the live-frame URL while watching so the <img> refetches.
  useEffect(() => {
    if (watchingCamId == null) return;
    const t = setInterval(() => setFrameTick((x) => x + 1), 100);
    return () => clearInterval(t);
  }, [watchingCamId]);

  // Best-effort: tell the backend to stop watching when the page is
  // unmounted (otherwise it'd hold the 10s TTL until it naturally
  // expires, which is fine but wastes Pi cycles).
  useEffect(() => {
    watchingCamIdRef.current = watchingCamId;
  }, [watchingCamId]);
  useEffect(() => {
    return () => {
      const id = watchingCamIdRef.current;
      if (id != null) api.stopWatchingCamera(adminPassword, id).catch(() => {});
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function startWatch(cam) {
    if (watchingCamId === cam.id) return;
    // If we were watching another camera, ask the backend to release
    // it before claiming a new one (fire-and-forget — don't block).
    if (watchingCamId != null) {
      api.stopWatchingCamera(adminPassword, watchingCamId).catch(() => {});
    }
    setBusy((b) => ({ ...b, [cam.id]: true }));
    setFrameAvailable(false);
    setFrameTick(0);
    try {
      await api.startWatchingCamera(adminPassword, cam.id);
      setWatchingCamId(cam.id);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  async function stopWatch() {
    const id = watchingCamId;
    setWatchingCamId(null);
    setFrameAvailable(false);
    if (id != null) {
      try {
        await api.stopWatchingCamera(adminPassword, id);
      } catch (e) {
        setError(e.message);
      }
    }
  }

  async function createCamera(e) {
    e.preventDefault();
    setError(null);
    if (!newCourseId || !newHole) {
      setError("Course and hole are required.");
      return;
    }
    setCreating(true);
    try {
      await api.createCamera(adminPassword, {
        courseId: parseInt(newCourseId, 10),
        assignedHole: parseInt(newHole, 10),
        assignedRole: newRole,
        name: newName.trim(),
      });
      setNewHole("");
      setNewName("");
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setCreating(false);
    }
  }

  async function pairWith(cam, partnerId) {
    setBusy((b) => ({ ...b, [cam.id]: true }));
    try {
      await api.pairCameras(adminPassword, cam.id, partnerId);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  async function unpair(cam) {
    setBusy((b) => ({ ...b, [cam.id]: true }));
    try {
      await api.unpairCamera(adminPassword, cam.id);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  async function rotateToken(cam) {
    if (!window.confirm(
      "Rotate this camera's auth token? The Pi will stop authenticating with the old token immediately — you'll need to re-provision the SD card.",
    )) return;
    setBusy((b) => ({ ...b, [cam.id]: true }));
    try {
      await api.rotateCameraToken(adminPassword, cam.id);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  async function toggleEnabled(cam) {
    setBusy((b) => ({ ...b, [cam.id]: true }));
    try {
      await api.updateCamera(adminPassword, cam.id, { enabled: !cam.enabled });
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  function openMove(cam) {
    setMoveDraft({
      courseId: String(cam.course_id),
      hole: String(cam.assigned_hole),
      role: cam.assigned_role,
    });
    setMovingCam(cam.id);
  }

  function closeMove() {
    setMovingCam(null);
  }

  async function submitMove(cam) {
    const hole = parseInt(moveDraft.hole, 10);
    const courseId = parseInt(moveDraft.courseId, 10);
    if (!Number.isFinite(courseId) || !Number.isFinite(hole)) {
      setError("Course and hole are required.");
      return;
    }
    setBusy((b) => ({ ...b, [cam.id]: true }));
    try {
      const updated = await api.updateCamera(adminPassword, cam.id, {
        courseId,
        assignedHole: hole,
        assignedRole: moveDraft.role,
      });
      if (updated && updated.auto_unpaired) {
        window.alert(
          "Move applied. This camera was auto-unpaired because the new placement no longer matched its previous partner's course / hole / role.",
        );
      }
      closeMove();
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  async function deleteCamera(cam) {
    if (!window.confirm(
      `Delete camera #${cam.id} (${cam.name || `hole ${cam.assigned_hole} ${cam.assigned_role}`})? This can't be undone.`,
    )) return;
    setBusy((b) => ({ ...b, [cam.id]: true }));
    try {
      await api.deleteCamera(adminPassword, cam.id);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy((b) => ({ ...b, [cam.id]: false }));
    }
  }

  async function copyToken(token) {
    try {
      await navigator.clipboard.writeText(token);
    } catch (e) {
      window.prompt("Copy this token:", token);
    }
  }

  if (!adminPassword) {
    return (
      <div className="wrap">
        <Brand subtitle="Operator Console" />
        <div className="card center">
          <h2>Admin password required</h2>
          <Link to="/admin"><button style={{ marginTop: 10 }}>Sign in</button></Link>
        </div>
      </div>
    );
  }

  // Pair candidates per camera: same course + same hole + opposite
  // role + currently unpaired (or paired with this camera).
  function pairCandidates(cam) {
    if (!cameras) return [];
    const oppositeRole = cam.assigned_role === "tee" ? "green" : "tee";
    return cameras.filter((c) =>
      c.id !== cam.id
      && c.course_id === cam.course_id
      && c.assigned_hole === cam.assigned_hole
      && c.assigned_role === oppositeRole
      && (!c.paired_with_camera_id || c.paired_with_camera_id === cam.id),
    );
  }

  function findById(id) {
    return cameras?.find((c) => c.id === id);
  }

  return (
    <div className="wrap wide">
      <Brand subtitle="Operator Console" />
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants">Players</Link>
        <Link to="/admin/courses">Courses</Link>
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production">Production</Link>
        <Link to="/admin/produced-clips">Produced Clips</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras" className="active">Cameras</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 6 }}>Cameras</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Register each on-course capture device (one tee + one green per
          par-3 hole). Each row's <code>auth_token</code> is the secret the
          Pi uses to call <code>/api/cameras/&#123;token&#125;/...</code> — keep
          it private and rotate it if a device is lost.
        </p>
      </div>

      {error && <div className="card err-text small">{error}</div>}

      <div className="card">
        <h4 style={{ marginBottom: 8 }}>Register new camera</h4>
        <form onSubmit={createCamera}>
          <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
            <div className="field" style={{ flex: 2, minWidth: 200 }}>
              <label className="small">Course</label>
              <select
                value={newCourseId}
                onChange={(e) => setNewCourseId(e.target.value)}
                disabled={creating}
              >
                <option value="">Pick a course…</option>
                {courses.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
            <div className="field" style={{ flex: 1, minWidth: 80 }}>
              <label className="small">Hole</label>
              <input
                type="number" min="1" max="18"
                value={newHole}
                onChange={(e) => setNewHole(e.target.value)}
                disabled={creating}
              />
            </div>
            <div className="field" style={{ flex: 1, minWidth: 120 }}>
              <label className="small">Role</label>
              <select value={newRole} onChange={(e) => setNewRole(e.target.value)} disabled={creating}>
                <option value="tee">tee</option>
                <option value="green">green</option>
              </select>
            </div>
            <div className="field" style={{ flex: 2, minWidth: 200 }}>
              <label className="small">Name (optional)</label>
              <input
                type="text" placeholder="Hole 3 tee — east tree"
                value={newName}
                onChange={(e) => setNewName(e.target.value)}
                disabled={creating}
              />
            </div>
            <div>
              <button type="submit" disabled={creating}>
                {creating ? "Creating…" : "Register"}
              </button>
            </div>
          </div>
        </form>
      </div>

      {cameras === null && (
        <div className="card"><div className="shimmer" style={{ height: 80 }} /></div>
      )}

      {cameras?.length === 0 && (
        <div className="card muted center" style={{ padding: 40 }}>
          No cameras registered yet. Use the form above to register the first one.
        </div>
      )}

      <div className="stack" style={{ gap: 10 }}>
        {cameras?.map((cam) => {
          const partner = cam.paired_with_camera_id
            ? findById(cam.paired_with_camera_id) : null;
          const candidates = pairCandidates(cam);
          const isBusy = !!busy[cam.id];
          const tokenVisible = !!revealedToken[cam.id];
          return (
            <div key={cam.id} className="card tight" style={{ margin: 0, padding: 12 }}>
              <div className="inline" style={{ justifyContent: "space-between", flexWrap: "wrap", gap: 8 }}>
                <div className="small">
                  <b>#{cam.id}</b>{" "}
                  <span className={`pill small ${cam.enabled ? "ok" : "warn"}`}>
                    {cam.enabled ? "enabled" : "disabled"}
                  </span>{" "}
                  <span className={`pill small ${cam.assigned_role === "tee" ? "" : "ok"}`}>
                    {cam.assigned_role}
                  </span>{" "}
                  <span className="muted">
                    · hole {cam.assigned_hole}
                    {cam.name && <> · {cam.name}</>}
                  </span>
                  <div className="tiny muted" style={{ marginTop: 2 }}>
                    last seen: {tsRel(cam.last_seen_at)}
                    {cam.last_event_at && <> · last event: {tsRel(cam.last_event_at)} ({cam.last_event_status})</>}
                    {cam.firmware_version && <> · fw {cam.firmware_version}</>}
                  </div>
                  <div className="tiny" style={{ marginTop: 6, fontFamily: "monospace" }}>
                    auth_token:{" "}
                    {tokenVisible ? (
                      <>
                        <code>{cam.auth_token}</code>
                        <button
                          type="button" className="ghost small"
                          onClick={() => copyToken(cam.auth_token)}
                          style={{ marginLeft: 8 }}
                        >
                          Copy
                        </button>
                      </>
                    ) : (
                      <button
                        type="button" className="ghost small"
                        onClick={() => setRevealedToken((r) => ({ ...r, [cam.id]: true }))}
                      >
                        Show
                      </button>
                    )}
                  </div>
                </div>
                <div className="inline" style={{ gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
                  {partner ? (
                    <span className="small muted">
                      paired with{" "}
                      <code>#{partner.id} ({partner.assigned_role})</code>
                      {" "}
                      <button
                        type="button" className="ghost small"
                        onClick={() => unpair(cam)} disabled={isBusy}
                      >
                        Unpair
                      </button>
                    </span>
                  ) : candidates.length > 0 ? (
                    <select
                      defaultValue=""
                      disabled={isBusy}
                      onChange={(e) => {
                        const v = parseInt(e.target.value, 10);
                        if (Number.isFinite(v)) pairWith(cam, v);
                      }}
                      className="small"
                      style={{ fontSize: 12 }}
                    >
                      <option value="">Pair with…</option>
                      {candidates.map((p) => (
                        <option key={p.id} value={p.id}>
                          #{p.id} ({p.assigned_role}) {p.name && `— ${p.name}`}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span className="tiny muted">no eligible partner</span>
                  )}
                  <button
                    type="button"
                    className={watchingCamId === cam.id ? "small" : "secondary small"}
                    onClick={() => (watchingCamId === cam.id ? stopWatch() : startWatch(cam))}
                    disabled={isBusy}
                    title="Open a live JPEG view from this camera (~10 fps)"
                  >
                    {watchingCamId === cam.id ? "Stop watching" : "Watch"}
                  </button>
                  <button
                    type="button" className="secondary small"
                    onClick={() => toggleEnabled(cam)} disabled={isBusy}
                  >
                    {cam.enabled ? "Disable" : "Enable"}
                  </button>
                  <button
                    type="button" className="secondary small"
                    onClick={() => openMove(cam)} disabled={isBusy}
                    title="Move this camera to a different course / hole / role"
                  >
                    Move
                  </button>
                  <button
                    type="button" className="secondary small"
                    onClick={() => rotateToken(cam)} disabled={isBusy}
                    title="Mint a new auth_token for this camera"
                  >
                    Rotate token
                  </button>
                  <button
                    type="button" className="ghost small err-text"
                    onClick={() => deleteCamera(cam)} disabled={isBusy}
                  >
                    Delete
                  </button>
                </div>
              </div>

              {watchingCamId === cam.id && (
                <div
                  className="card tight"
                  style={{ margin: "10px 0 0", padding: 10, background: "#000" }}
                >
                  <div
                    className="inline"
                    style={{ justifyContent: "space-between", marginBottom: 6 }}
                  >
                    <div className="small" style={{ color: "#bbb" }}>
                      Live · #{cam.id}
                      {cam.name && <> — {cam.name}</>}
                      {" · "}hole {cam.assigned_hole} {cam.assigned_role}
                    </div>
                    <button type="button" className="ghost small" onClick={stopWatch}>
                      Close
                    </button>
                  </div>
                  <div
                    style={{
                      position: "relative",
                      background: "#000",
                      minHeight: 240,
                    }}
                  >
                    <img
                      src={`${api.cameraLiveFrameUrl(cam.id)}?t=${frameTick}`}
                      alt=""
                      onLoad={(e) => {
                        if (e.target.naturalWidth > 0) setFrameAvailable(true);
                      }}
                      style={{
                        display: "block",
                        width: "100%",
                        height: "auto",
                        visibility: frameAvailable ? "visible" : "hidden",
                      }}
                    />
                    {!frameAvailable && (
                      <div
                        style={{
                          position: "absolute",
                          inset: 0,
                          display: "flex",
                          alignItems: "center",
                          justifyContent: "center",
                          color: "#bbb",
                          fontSize: 13,
                        }}
                      >
                        Waiting for live frame from Pi…
                      </div>
                    )}
                  </div>
                </div>
              )}

              {movingCam === cam.id && (
                <div
                  className="card tight"
                  style={{ margin: "10px 0 0", padding: 10, background: "var(--surface-alt)" }}
                >
                  <div className="small" style={{ marginBottom: 6 }}>
                    <b>Move camera #{cam.id}</b>{" "}
                    <span className="tiny muted">
                      · changing course / hole / role will auto-unpair if the
                      existing partner no longer fits
                    </span>
                  </div>
                  <div className="row" style={{ gap: 8, flexWrap: "wrap", alignItems: "flex-end" }}>
                    <div className="field" style={{ flex: 2, minWidth: 180 }}>
                      <label className="small">Course</label>
                      <select
                        value={moveDraft.courseId}
                        onChange={(e) => setMoveDraft((d) => ({ ...d, courseId: e.target.value }))}
                        disabled={isBusy}
                      >
                        {courses.map((c) => (
                          <option key={c.id} value={c.id}>{c.name}</option>
                        ))}
                      </select>
                    </div>
                    <div className="field" style={{ flex: 1, minWidth: 80 }}>
                      <label className="small">Hole</label>
                      <input
                        type="number" min="1" max="18"
                        value={moveDraft.hole}
                        onChange={(e) => setMoveDraft((d) => ({ ...d, hole: e.target.value }))}
                        disabled={isBusy}
                      />
                    </div>
                    <div className="field" style={{ flex: 1, minWidth: 100 }}>
                      <label className="small">Role</label>
                      <select
                        value={moveDraft.role}
                        onChange={(e) => setMoveDraft((d) => ({ ...d, role: e.target.value }))}
                        disabled={isBusy}
                      >
                        <option value="tee">tee</option>
                        <option value="green">green</option>
                      </select>
                    </div>
                    <div className="inline" style={{ gap: 6 }}>
                      <button
                        type="button"
                        onClick={() => submitMove(cam)}
                        disabled={isBusy}
                      >
                        {isBusy ? "Saving…" : "Apply move"}
                      </button>
                      <button
                        type="button" className="ghost"
                        onClick={closeMove} disabled={isBusy}
                      >
                        Cancel
                      </button>
                    </div>
                  </div>
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
