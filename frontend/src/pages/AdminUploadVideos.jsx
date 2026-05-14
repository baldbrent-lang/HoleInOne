/**
 * Simplified upload page — the "Upload" link in the admin nav.
 *
 * Operator picks a course, picks a date played, attaches a tee
 * video (and optionally a green-side video), hits Upload Video(s).
 * The files land in the production queue (LongVideoUpload row,
 * status=pending) and a notice tells them they're waiting for
 * production.
 *
 * No segments, no auto-detect knobs, no AI tracer model picker —
 * those still live on /admin/long-upload for power-user runs.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function todayLocalDate() {
  const d = new Date();
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}`;
}

export default function AdminUploadVideos() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  const [courses, setCourses] = useState([]);
  const [courseId, setCourseId] = useState("");
  const [datePlayed, setDatePlayed] = useState(todayLocalDate());
  const [teeFile, setTeeFile] = useState(null);
  const [greenFile, setGreenFile] = useState(null);
  const [dontAutoProduce, setDontAutoProduce] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(null);

  useEffect(() => {
    if (!adminPassword) return;
    api.listCourses(adminPassword).then((rows) => {
      setCourses(rows);
      if (rows.length === 1) setCourseId(String(rows[0].id));
    }).catch((e) => setError(e.message));
  }, [adminPassword]);

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setSuccess(null);
    if (!courseId) { setError("Pick a course."); return; }
    if (!datePlayed) { setError("Pick a date played."); return; }
    if (!teeFile) { setError("Attach a tee-angle video."); return; }

    // The backend wants an ISO 8601 timestamp; the date picker gives
    // us YYYY-MM-DD. Default the time to noon so participant matching
    // has a reasonable timestamp to work with — operator can fine-tune
    // via /admin/long-upload before kicking off processing.
    const baseCapturedAt = new Date(`${datePlayed}T12:00:00`).toISOString();

    const fd = new FormData();
    fd.append("course_id", String(parseInt(courseId, 10)));
    fd.append("base_captured_at", baseCapturedAt);
    fd.append("video", teeFile, teeFile.name);
    if (greenFile) fd.append("video_green", greenFile, greenFile.name);
    if (dontAutoProduce) fd.append("queue_only", "true");

    setUploading(true);
    setProgress(0);
    try {
      const data = await api.quickUploadVideos(adminPassword, fd, setProgress);
      setSuccess(data);
      // Reset form for the next upload.
      setTeeFile(null);
      setGreenFile(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setUploading(false);
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

  return (
    <div className="wrap">
      <Brand subtitle="Operator Console" />
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants">Participants</Link>
        <Link to="/admin/upload-videos" className="active">Upload</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 6 }}>Upload videos</h3>
        <p className="small muted" style={{ marginBottom: 14 }}>
          Drop a tee-angle video (and an optional green-side video) for
          a round. Submit kicks off processing in the background —
          produced clips appear on{" "}
          <Link to="/admin/broadcast-clips">Broadcast</Link> when ready.
          Check <i>Don't Auto Produce</i> below to queue the video for
          editing on{" "}
          <Link to="/admin/long-upload">Long upload</Link> instead.
        </p>

        {success && (
          <div
            className="card"
            style={{
              background: "rgba(40, 168, 92, 0.08)",
              border: "1px solid rgba(40, 168, 92, 0.35)",
              margin: "0 0 14px",
              padding: 12,
            }}
          >
            <b>Upload #{success.upload_id}{" "}
              {success.auto_processing ? "started" : "queued"}.</b>{" "}
            <span className="small muted">{success.message}</span>
            <div style={{ marginTop: 8 }}>
              <Link to={success.auto_processing ? "/admin/broadcast-clips" : "/admin/long-upload"}>
                <button className="small" style={{ width: "auto" }}>
                  {success.auto_processing
                    ? "Go to Broadcast →"
                    : "Go to Long upload →"}
                </button>
              </Link>
            </div>
          </div>
        )}

        {error && (
          <div className="card err-text small" style={{ margin: "0 0 14px" }}>
            {error}
          </div>
        )}

        <form onSubmit={submit}>
          <div className="row" style={{ gap: 12, flexWrap: "wrap", alignItems: "flex-end" }}>
            <div className="field" style={{ flex: 2, minWidth: 220 }}>
              <label className="small">Course</label>
              <select
                value={courseId}
                onChange={(e) => setCourseId(e.target.value)}
                disabled={uploading}
              >
                <option value="">Pick a course…</option>
                {courses.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
            </div>
            <div className="field" style={{ flex: 1, minWidth: 180 }}>
              <label className="small">Date played</label>
              <input
                type="date"
                value={datePlayed}
                onChange={(e) => setDatePlayed(e.target.value)}
                disabled={uploading}
              />
            </div>
          </div>

          <div className="row" style={{ gap: 12, flexWrap: "wrap", marginTop: 14 }}>
            <div className="field" style={{ flex: 1, minWidth: 260 }}>
              <label className="small">Tee-angle video</label>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setTeeFile(e.target.files?.[0] || null)}
                disabled={uploading}
              />
              {teeFile && (
                <div className="tiny muted" style={{ marginTop: 4 }}>
                  {teeFile.name} · {(teeFile.size / (1024 * 1024)).toFixed(1)} MB
                </div>
              )}
            </div>
            <div className="field" style={{ flex: 1, minWidth: 260 }}>
              <label className="small">
                Green-side video <span className="muted">(optional)</span>
              </label>
              <input
                type="file"
                accept="video/*"
                onChange={(e) => setGreenFile(e.target.files?.[0] || null)}
                disabled={uploading}
              />
              {greenFile && (
                <div className="tiny muted" style={{ marginTop: 4 }}>
                  {greenFile.name} · {(greenFile.size / (1024 * 1024)).toFixed(1)} MB
                </div>
              )}
            </div>
          </div>

          <div style={{ marginTop: 14 }}>
            <label
              className="small"
              style={{ display: "inline-flex", alignItems: "center", gap: 8, cursor: "pointer" }}
            >
              <input
                type="checkbox"
                checked={dontAutoProduce}
                onChange={(e) => setDontAutoProduce(e.target.checked)}
                disabled={uploading}
                style={{ margin: 0 }}
              />
              <span>
                <b>Don't Auto Produce.</b>{" "}
                <span className="muted">
                  Video will go in queue for editing prior to production.
                </span>
              </span>
            </label>
          </div>

          <div className="row" style={{ marginTop: 18, gap: 10, alignItems: "center" }}>
            <button type="submit" disabled={uploading}>
              {uploading
                ? `Uploading… ${progress}%`
                : "Submit"}
            </button>
            {uploading && progress > 0 && progress < 100 && (
              <div
                className="small muted"
                style={{
                  flex: 1, height: 8, background: "var(--border)",
                  borderRadius: 4, overflow: "hidden", maxWidth: 360,
                }}
              >
                <div
                  style={{
                    width: `${progress}%`, height: "100%",
                    background: "var(--primary, #28a85c)",
                    transition: "width 200ms ease",
                  }}
                />
              </div>
            )}
          </div>
        </form>
      </div>
    </div>
  );
}
