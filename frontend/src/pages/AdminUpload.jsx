import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, API_BASE } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function nowLocal() {
  const d = new Date();
  d.setSeconds(0, 0);
  const pad = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())}T${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

export default function AdminUpload() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  const [courses, setCourses] = useState([]);
  const [courseId, setCourseId] = useState("");
  const [hole, setHole] = useState("");
  const [cameraType, setCameraType] = useState("tee");
  const [capturedAt, setCapturedAt] = useState(nowLocal());
  const [carryYards, setCarryYards] = useState("");
  const [ballSpeed, setBallSpeed] = useState("");
  const [distanceFromPin, setDistanceFromPin] = useState("");
  const [ballInCup, setBallInCup] = useState(false);
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [overrideCandidates, setOverrideCandidates] = useState([]);
  const [assigning, setAssigning] = useState(false);

  useEffect(() => {
    if (!adminPassword) return;
    api.listCourses(adminPassword).then(setCourses).catch((e) => setError(e.message));
  }, [adminPassword]);

  // Pre-fill par-3 holes for selected course as a hint
  const selectedCourse = courses.find((c) => String(c.id) === String(courseId));
  const par3Hint = selectedCourse?.par3_holes?.join(", ");

  async function submit(e) {
    e.preventDefault();
    setError(null);
    setResult(null);

    if (!file) { setError("Pick a video file."); return; }
    setUploading(true);
    setProgress(0);

    const fd = new FormData();
    fd.append("course_id", courseId);
    fd.append("hole_number", hole);
    fd.append("camera_type", cameraType);
    fd.append("captured_at", new Date(capturedAt).toISOString());
    if (carryYards) fd.append("carry_yards", carryYards);
    if (ballSpeed) fd.append("ball_speed_mph", ballSpeed);
    if (distanceFromPin) fd.append("distance_from_pin_feet", distanceFromPin);
    fd.append("ball_in_cup", ballInCup ? "true" : "false");
    fd.append("video", file, file.name);

    try {
      const data = await api.uploadClip(adminPassword, fd, setProgress);
      setResult(data);
      setFile(null);
      // If unmatched, load all of this course's recent registrants for quick-assign
      if (data.status !== "assigned") {
        try {
          const list = await api.listParticipants(adminPassword, { course_id: courseId });
          setOverrideCandidates(list);
        } catch {
          setOverrideCandidates([]);
        }
      } else {
        setOverrideCandidates([]);
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setUploading(false);
    }
  }

  async function forceAssign(participantId, participantName) {
    if (!result?.clip_id) return;
    setAssigning(true);
    try {
      await api.assignClip(adminPassword, result.clip_id, participantId);
      setResult({
        ...result,
        status: "assigned",
        participant_id: participantId,
        participant_name: participantName,
        issue_note: null,
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setAssigning(false);
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
        <Link to="/admin/upload" className="active">Upload clip</Link>
        <Link to="/admin/long-upload">Long upload</Link>
        <Link to="/admin/showcase">Home videos</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 6 }}>Manual clip upload</h3>
        <p className="small muted" style={{ marginBottom: 16 }}>
          Drop an MP4 from your phone, GoPro, or any camera. We treat it as if Shot
          Tracer had pushed it via webhook — runs the matcher and lands in the right
          golfer's gallery (or the manual-review queue if no confident match).
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
            </div>
            <div className="field">
              <label>Hole #</label>
              <input
                type="number"
                required
                value={hole}
                onChange={(e) => setHole(e.target.value)}
                placeholder="3"
                min="1"
                max="18"
              />
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
          </div>

          <div className="field">
            <label>Captured at</label>
            <input
              type="datetime-local"
              required
              value={capturedAt}
              onChange={(e) => setCapturedAt(e.target.value)}
            />
            <div className="hint small muted">
              When the swing happened. Time-of-day is what the matcher uses to find the right tee time.
            </div>
          </div>

          <div className="row">
            <div className="field">
              <label>Carry (yds)</label>
              <input type="number" value={carryYards} onChange={(e) => setCarryYards(e.target.value)} placeholder="optional" />
            </div>
            <div className="field">
              <label>Ball speed (mph)</label>
              <input type="number" value={ballSpeed} onChange={(e) => setBallSpeed(e.target.value)} placeholder="optional" />
            </div>
            <div className="field">
              <label>Distance from pin (ft)</label>
              <input
                type="number"
                value={distanceFromPin}
                onChange={(e) => setDistanceFromPin(e.target.value)}
                placeholder="for closest-to-pin"
              />
              <div className="hint small muted">Powers the daily Closest-to-Pin contest.</div>
            </div>
            <div className="field" style={{ alignSelf: "end", flex: 0 }}>
              <label className="inline" style={{ gap: 8 }}>
                <input
                  type="checkbox"
                  checked={ballInCup}
                  onChange={(e) => setBallInCup(e.target.checked)}
                  style={{ width: "auto" }}
                />
                Ball in cup
              </label>
            </div>
          </div>

          <div className="field">
            <label>Video file</label>
            <input
              type="file"
              accept="video/*"
              onChange={(e) => setFile(e.target.files?.[0] || null)}
            />
            {file && (
              <div className="small muted" style={{ marginTop: 6 }}>
                {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
              </div>
            )}
          </div>

          {error && <p className="err-text small">{error}</p>}
          {uploading && (
            <div style={{ marginBottom: 12 }}>
              <div style={{ height: 8, background: "var(--surface-alt)", borderRadius: 4, overflow: "hidden" }}>
                <div
                  style={{
                    width: `${progress}%`, height: "100%",
                    background: "var(--primary)", transition: "width .15s",
                  }}
                />
              </div>
              <div className="small muted" style={{ marginTop: 4 }}>{progress}% uploaded</div>
            </div>
          )}

          <button disabled={uploading || !file || !courseId || !hole}>
            {uploading ? `Uploading… ${progress}%` : "Upload + run matcher"}
          </button>
        </form>
      </div>

      {result && (
        <div
          className="card"
          style={{
            background: result.status === "assigned" ? "var(--primary-soft)" : undefined,
            border: result.status === "assigned" ? "1px solid var(--emerald-200)" : undefined,
          }}
        >
          <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 10 }}>
            <h3>Result</h3>
            <span className={`pill ${result.status === "assigned" ? "ok" : result.status === "flagged" ? "err" : "warn"}`}>
              {result.status}
            </span>
          </div>
          {result.participant_name ? (
            <p>
              Matched to <b>{result.participant_name}</b>{" "}
              <span className="small muted">(participant #{result.participant_id})</span>
            </p>
          ) : (
            <p className="muted">
              No match — appears in the manual-review queue.
              {result.issue_note && <> Reason: <code>{result.issue_note}</code></>}
            </p>
          )}
          <video
            src={result.source_url}
            controls
            style={{ width: "100%", maxWidth: 640, marginTop: 10, borderRadius: 8, background: "#000" }}
          />

          {result.status !== "assigned" && overrideCandidates.length > 0 && (
            <div style={{ marginTop: 16 }}>
              <div className="tiny upper muted" style={{ marginBottom: 8 }}>
                Force-assign to a registered golfer
              </div>
              <p className="small muted" style={{ marginBottom: 10 }}>
                The auto-matcher needs <code>captured_at</code> within 5 hours of a registered tee time.
                If you're testing or back-filling, just pick a golfer below.
              </p>
              <div className="chip-row">
                {overrideCandidates.map((cand) => (
                  <button
                    key={cand.id}
                    type="button"
                    className="chip"
                    disabled={assigning}
                    title={`Assign to ${cand.name}`}
                    onClick={() => forceAssign(cand.id, cand.name)}
                  >
                    {cand.selfie_url && (
                      <img
                        src={cand.selfie_url}
                        alt=""
                        style={{ width: 22, height: 22, borderRadius: "50%", objectFit: "cover" }}
                      />
                    )}
                    {cand.name}
                    <span className="small muted" style={{ marginLeft: 6 }}>
                      · {new Date(cand.tee_time.starts_at).toLocaleString([], {
                        month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
                      })}
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}

          {result.participant_id && (
            <div style={{ marginTop: 12 }}>
              <Link to={`/admin/participants?course_id=${courseId}`} className="btn secondary small" style={{ marginRight: 8 }}>
                <Icon name="users" size={14} /> See golfer's gallery
              </Link>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
