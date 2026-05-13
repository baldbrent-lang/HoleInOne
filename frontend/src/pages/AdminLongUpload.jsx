import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function nowLocal() {
  const d = new Date();
  d.setSeconds(0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

const EMPTY_SEG = {
  hole_number: "",
  start_sec: "",
  end_sec: "",
  distance_from_pin_feet: "",
  carry_yards: "",
  ball_speed_mph: "",
  ball_in_cup: false,
};

export default function AdminLongUpload() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  const [courses, setCourses] = useState([]);
  const [courseId, setCourseId] = useState("");
  const [cameraType, setCameraType] = useState("tee");
  const [baseCapturedAt, setBaseCapturedAt] = useState(nowLocal());
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [fileGreen, setFileGreen] = useState(null);
  const [aiTracerModel, setAiTracerModel] = useState("claude-opus-4-7");
  const [autoDetectSwings, setAutoDetectSwings] = useState(true);
  const [startingHole, setStartingHole] = useState(1);
  const [segments, setSegments] = useState([{ ...EMPTY_SEG }]);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [results, setResults] = useState(null);
  const [autoDetectInfo, setAutoDetectInfo] = useState(null);
  const [error, setError] = useState(null);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!adminPassword) return;
    api.listCourses(adminPassword).then(setCourses).catch((e) => setError(e.message));
  }, [adminPassword]);

  useEffect(() => {
    if (!file) { setVideoUrl(null); return; }
    const url = URL.createObjectURL(file);
    setVideoUrl(url);
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const selectedCourse = courses.find((c) => String(c.id) === String(courseId));
  const par3Hint = selectedCourse?.par3_holes?.join(", ");

  function setSeg(idx, patch) {
    setSegments((cur) => cur.map((s, i) => (i === idx ? { ...s, ...patch } : s)));
  }
  function addSeg() {
    setSegments((cur) => [...cur, { ...EMPTY_SEG }]);
  }
  function removeSeg(idx) {
    setSegments((cur) => cur.length > 1 ? cur.filter((_, i) => i !== idx) : cur);
  }
  function captureCurrentTime(idx, field) {
    const v = videoRef.current;
    if (!v) return;
    setSeg(idx, { [field]: v.currentTime.toFixed(2) });
  }

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResults(null);
    setAutoDetectInfo(null);

    if (!file) { setError("Pick a video file."); return; }
    let cleaned = [];
    if (!autoDetectSwings) {
      cleaned = segments
        .map((s) => ({
          hole_number: parseInt(s.hole_number, 10),
          start_sec: parseFloat(s.start_sec),
          end_sec: parseFloat(s.end_sec),
          distance_from_pin_feet: s.distance_from_pin_feet ? parseInt(s.distance_from_pin_feet, 10) : null,
          carry_yards: s.carry_yards ? parseInt(s.carry_yards, 10) : null,
          ball_speed_mph: s.ball_speed_mph ? parseInt(s.ball_speed_mph, 10) : null,
          ball_in_cup: !!s.ball_in_cup,
        }))
        .filter((s) => Number.isFinite(s.hole_number) && Number.isFinite(s.start_sec) && Number.isFinite(s.end_sec));

      if (cleaned.length === 0) {
        setError("At least one segment with hole / start / end is required (or enable Auto-detect).");
        return;
      }
      for (const s of cleaned) {
        if (s.end_sec <= s.start_sec) {
          setError(`Segment for hole ${s.hole_number}: end must be greater than start.`);
          return;
        }
      }
    }

    setUploading(true);
    setProgress(0);

    const fd = new FormData();
    fd.append("course_id", courseId);
    fd.append("camera_type", cameraType);
    fd.append("base_captured_at", new Date(baseCapturedAt).toISOString());
    fd.append("segments", JSON.stringify(cleaned));
    fd.append("auto_detect_swings", autoDetectSwings ? "true" : "false");
    fd.append("starting_hole", String(parseInt(startingHole, 10) || 1));
    fd.append("video", file, file.name);
    if (fileGreen) {
      fd.append("video_green", fileGreen, fileGreen.name);
      fd.append("ai_tracer_model", aiTracerModel);
    }

    try {
      const data = await api.longUploadClips(adminPassword, fd, setProgress);
      setResults(data.results || []);
      setAutoDetectInfo(data.auto_detect || null);
    } catch (e) {
      setError(e.message);
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
    <div className="wrap wide">
      <Brand subtitle="Operator Console" />
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants">Participants</Link>
        <Link to="/admin/upload">Upload clip</Link>
        <Link to="/admin/long-upload" className="active">Long upload</Link>
        <Link to="/admin/showcase">Home videos</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 6 }}>Long video upload</h3>
        <p className="small muted" style={{ marginBottom: 14 }}>
          Drop one continuous tee-side video and mark the start/end seconds
          for each swing. We'll cut each segment via ffmpeg, run it through
          the matcher, and deliver per-clip just like the regular upload
          flow. Captured-at for each clip is set to your base time + the
          segment's start.
          {" "}
          <b>Optional dual-camera:</b> add a green-side long video that's
          wall-clock-synced to the tee video and we'll run the full AI
          tracer on the tee cut, then composite tee-with-tracer until 1 s
          after the tracer ends, then hard-cut to the green clip for the
          ball landing.
        </p>

        <form onSubmit={submit}>
          <div className="row">
            <div className="field" style={{ flex: 2 }}>
              <label>Course</label>
              <select required value={courseId} onChange={(e) => setCourseId(e.target.value)}>
                <option value="">Select a course…</option>
                {courses.map((c) => (
                  <option key={c.id} value={c.id}>{c.name}</option>
                ))}
              </select>
              {par3Hint && <div className="hint small muted">Par-3s: {par3Hint}</div>}
            </div>
            <div className="field">
              <label>Camera angle</label>
              <select value={cameraType} onChange={(e) => setCameraType(e.target.value)}>
                <option value="tee">Tee</option>
                <option value="wide_green">Wide green</option>
                <option value="hole">Hole / cup</option>
              </select>
            </div>
            <div className="field">
              <label>Recording started at</label>
              <input type="datetime-local" required value={baseCapturedAt} onChange={(e) => setBaseCapturedAt(e.target.value)} />
            </div>
          </div>

          <div className="field">
            <label>Long video file (tee side)</label>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file && (
              <div className="small muted" style={{ marginTop: 4 }}>
                {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
              </div>
            )}
          </div>

          <div className="field">
            <label>
              Green-side long video (optional, dual-camera composite)
            </label>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setFileGreen(e.target.files?.[0] || null)}
            />
            <div className="small muted" style={{ marginTop: 4 }}>
              Must be wall-clock-synced to the tee video (both cameras
              started recording at the same moment). Same segment
              start/end times are applied to both files.
              {fileGreen && (
                <> · <b>{fileGreen.name}</b> · {(fileGreen.size / 1024 / 1024).toFixed(1)} MB</>
              )}
            </div>
            {fileGreen && (
              <div className="row" style={{ marginTop: 8, alignItems: "center", gap: 8 }}>
                <label className="small muted" style={{ display: "inline-flex", alignItems: "center", gap: 4 }}>
                  AI tracer model:
                  <select
                    value={aiTracerModel}
                    onChange={(e) => setAiTracerModel(e.target.value)}
                    disabled={uploading}
                    style={{ fontSize: 13 }}
                  >
                    <option value="claude-opus-4-7">Opus 4.7 (best)</option>
                    <option value="claude-haiku-4-5">Haiku 4.5 (5× cheaper, faster)</option>
                  </select>
                </label>
              </div>
            )}
          </div>

          {videoUrl && (
            <div className="field">
              <label>Preview / scrub to find swings</label>
              <video
                ref={videoRef}
                src={videoUrl}
                controls
                playsInline
                style={{ width: "100%", maxHeight: 360, borderRadius: 8, background: "#000" }}
              />
              <div className="hint small muted">
                Click <code>Set start</code> / <code>Set end</code> on a segment row
                to capture the player's current playback time.
              </div>
            </div>
          )}

          <div className="card" style={{ background: "var(--surface-alt)", margin: "0 0 16px" }}>
            <label className="inline" style={{ gap: 8, cursor: "pointer", marginBottom: 4 }}>
              <input
                type="checkbox"
                checked={autoDetectSwings}
                onChange={(e) => setAutoDetectSwings(e.target.checked)}
              />
              <span>
                <b>Auto-detect swings from audio</b>{" "}
                <span className="small muted">
                  (server scans the tee video's audio track for impact transients
                  and segments each swing automatically — no manual marking)
                </span>
              </span>
            </label>
            {autoDetectSwings && (
              <div className="row" style={{ marginTop: 8, alignItems: "center", gap: 8 }}>
                <label className="small muted" style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
                  Starting hole #:
                  <input
                    type="number"
                    min="1"
                    value={startingHole}
                    onChange={(e) => setStartingHole(e.target.value)}
                    style={{ width: 70, fontSize: 13 }}
                  />
                </label>
                <span className="tiny muted">
                  Each detected swing gets a sequential hole number from here.
                  You can re-assign individual clips from /admin/clips after upload.
                </span>
              </div>
            )}
          </div>

          {!autoDetectSwings && (
          <div className="card" style={{ background: "var(--surface-alt)", margin: "0 0 16px" }}>
            <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 8 }}>
              <h4>Segments</h4>
              <button type="button" className="secondary small" onClick={addSeg} style={{ width: "auto" }}>
                + Add segment
              </button>
            </div>
            {segments.map((s, i) => (
              <div
                key={i}
                style={{
                  marginTop: 10, paddingTop: 10,
                  borderTop: i === 0 ? "none" : "1px solid var(--border)",
                }}
              >
                <div className="tiny upper muted" style={{ marginBottom: 6 }}>
                  Swing {i + 1}
                  {segments.length > 1 && (
                    <button
                      type="button"
                      className="ghost small"
                      style={{ marginLeft: 12, width: "auto", padding: 0 }}
                      onClick={() => removeSeg(i)}
                    >
                      remove
                    </button>
                  )}
                </div>
                <div className="row">
                  <div className="field" style={{ flex: 0.6, marginBottom: 0 }}>
                    <label>Hole #</label>
                    <input
                      type="number" min="1" max="18"
                      value={s.hole_number}
                      onChange={(e) => setSeg(i, { hole_number: e.target.value })}
                      placeholder="3"
                    />
                  </div>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <label>Start (sec)</label>
                    <div style={{ display: "flex", gap: 4 }}>
                      <input
                        type="number" step="0.1" min="0"
                        value={s.start_sec}
                        onChange={(e) => setSeg(i, { start_sec: e.target.value })}
                        placeholder="12.5"
                      />
                      {videoUrl && (
                        <button
                          type="button" className="ghost small"
                          onClick={() => captureCurrentTime(i, "start_sec")}
                          style={{ width: "auto", whiteSpace: "nowrap" }}
                          title="Capture current playback time"
                        >
                          Set start
                        </button>
                      )}
                    </div>
                  </div>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <label>End (sec)</label>
                    <div style={{ display: "flex", gap: 4 }}>
                      <input
                        type="number" step="0.1" min="0"
                        value={s.end_sec}
                        onChange={(e) => setSeg(i, { end_sec: e.target.value })}
                        placeholder="22"
                      />
                      {videoUrl && (
                        <button
                          type="button" className="ghost small"
                          onClick={() => captureCurrentTime(i, "end_sec")}
                          style={{ width: "auto", whiteSpace: "nowrap" }}
                          title="Capture current playback time"
                        >
                          Set end
                        </button>
                      )}
                    </div>
                  </div>
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <label>Carry (yds)</label>
                    <input type="number" value={s.carry_yards} onChange={(e) => setSeg(i, { carry_yards: e.target.value })} placeholder="opt" />
                  </div>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <label>Ball speed (mph)</label>
                    <input type="number" value={s.ball_speed_mph} onChange={(e) => setSeg(i, { ball_speed_mph: e.target.value })} placeholder="opt" />
                  </div>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <label>Distance from pin (ft)</label>
                    <input type="number" value={s.distance_from_pin_feet} onChange={(e) => setSeg(i, { distance_from_pin_feet: e.target.value })} placeholder="for CTP" />
                  </div>
                  <div className="field" style={{ marginBottom: 0, alignSelf: "end", flex: 0 }}>
                    <label className="inline" style={{ gap: 8 }}>
                      <input
                        type="checkbox"
                        checked={!!s.ball_in_cup}
                        onChange={(e) => setSeg(i, { ball_in_cup: e.target.checked })}
                        style={{ width: "auto" }}
                      />
                      Ace
                    </label>
                  </div>
                </div>
              </div>
            ))}
          </div>
          )}

          {error && <p className="err-text small">{error}</p>}
          {uploading && progress > 0 && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ height: 8, background: "var(--surface-alt)", borderRadius: 4, overflow: "hidden" }}>
                <div style={{ width: `${progress}%`, height: "100%", background: "var(--primary)" }} />
              </div>
              <div className="small muted" style={{ marginTop: 4 }}>{progress}% uploaded — ffmpeg cutting starts as soon as upload finishes</div>
            </div>
          )}

          <button disabled={uploading || !file || !courseId}>
            {uploading
              ? `Uploading… ${progress}%`
              : (autoDetectSwings
                ? (fileGreen
                  ? "Auto-detect swings + run AI tracer + composite (dual-cam)"
                  : "Auto-detect swings + run matcher")
                : (fileGreen
                  ? `Cut ${segments.length} swing${segments.length === 1 ? "" : "s"} + run AI tracer + composite (dual-cam)`
                  : `Cut ${segments.length} swing${segments.length === 1 ? "" : "s"} + run matcher`))}
          </button>
        </form>
      </div>

      {results && (
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>Results ({results.filter((r) => r.ok).length}/{results.length} succeeded)</h3>
          {autoDetectInfo && (
            <p className="small muted" style={{ marginBottom: 12 }}>
              Auto-detected <b>{autoDetectInfo.n_detected}</b> swing
              {autoDetectInfo.n_detected === 1 ? "" : "s"} from tee-video audio
              {Array.isArray(autoDetectInfo.peaks) && autoDetectInfo.peaks.length > 0 && (
                <> at: {autoDetectInfo.peaks.map((p, i) => (
                  <code key={i} style={{ marginRight: 6 }}>
                    {p.peak_time_sec}s{p.ratio != null ? ` (×${p.ratio})` : ""}
                  </code>
                ))}</>
              )}
            </p>
          )}
          <div className="stack">
            {results.map((r, i) => (
              <div key={i} className="card tight" style={{ margin: 0 }}>
                <div className="inline" style={{ justifyContent: "space-between", width: "100%" }}>
                  <div>
                    <b>Swing {(r.index ?? i) + 1}{r.hole_number ? ` · Hole ${r.hole_number}` : ""}</b>
                    {r.dual_camera && (
                      <span className="pill small" style={{ marginLeft: 6 }}>dual-cam</span>
                    )}
                  </div>
                  {r.ok ? (
                    <span className={`pill ${r.status === "assigned" ? "ok" : "warn"}`}>{r.status}</span>
                  ) : (
                    <span className="pill err">failed</span>
                  )}
                </div>
                {r.ok ? (
                  <>
                    <div className="small muted" style={{ marginTop: 4 }}>
                      {r.participant_name ? `Matched to ${r.participant_name}` : "No match — appears in manual review queue."}
                    </div>
                    {r.composite && (
                      <div className="tiny muted" style={{ marginTop: 2 }}>
                        Composite: tee→green cut at <code>{r.composite.switch_sec}s</code>, end <code>{r.composite.end_sec}s</code>
                        {r.composite.method && <> · impact via <code>{r.composite.method}</code></>}
                      </div>
                    )}
                    {r.ai_tracer_error && (
                      <div className="tiny err-text" style={{ marginTop: 2 }}>
                        AI tracer issue: <code>{r.ai_tracer_error}</code> (fell back to raw cut)
                      </div>
                    )}
                    {r.source_url && (
                      <video
                        src={r.source_url}
                        controls
                        playsInline
                        preload="metadata"
                        style={{ width: "100%", maxWidth: 480, marginTop: 8, borderRadius: 8, background: "#000" }}
                      />
                    )}
                  </>
                ) : (
                  <div className="err-text small" style={{ marginTop: 4 }}>{r.error}</div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
