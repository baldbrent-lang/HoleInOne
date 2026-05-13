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

function fmtDuration(sec) {
  if (sec == null || !Number.isFinite(sec)) return null;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = Math.round(sec - m * 60);
  return `${m}:${String(s).padStart(2, "0")}`;
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
  const [queuedUploadId, setQueuedUploadId] = useState(null);
  const [error, setError] = useState(null);
  const [priorUploads, setPriorUploads] = useState(null);
  const [reprocessing, setReprocessing] = useState({}); // {id: true}
  const [testCutting, setTestCutting] = useState({}); // {id: true}
  const [testCutResults, setTestCutResults] = useState({}); // {id: {detector, cuts, ...}}
  const [testCutDetector, setTestCutDetector] = useState({}); // {id: "motion"|"audio"}
  const [audioMinRatio, setAudioMinRatio] = useState({}); // {id: number}
  const [motionRatio, setMotionRatio] = useState({}); // {id: number}
  const [pairWindow, setPairWindow] = useState({}); // {id: number} -> combined-mode pair window
  const [detectOnly, setDetectOnly] = useState({}); // {id: true} -> skip cutting
  const [showSource, setShowSource] = useState({}); // {id: true}
  const videoRef = useRef(null);

  const anyProcessing =
    Array.isArray(priorUploads) &&
    priorUploads.some(
      (u) => u.processing_status === "pending" || u.processing_status === "processing",
    );

  useEffect(() => {
    if (!adminPassword) return;
    api.listCourses(adminPassword).then(setCourses).catch((e) => setError(e.message));
    refreshPriorUploads();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminPassword]);

  // Poll the long-uploads listing while any row is pending/processing so the
  // UI flips to "completed" without a manual refresh. Polling stops once
  // everything has settled.
  useEffect(() => {
    if (!adminPassword || !anyProcessing) return;
    const id = setInterval(refreshPriorUploads, 4000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminPassword, anyProcessing]);

  async function refreshPriorUploads() {
    try {
      const list = await api.listLongUploads(adminPassword);
      setPriorUploads(list);
    } catch (e) {
      // Non-fatal — fall back to the upload form alone.
      setPriorUploads([]);
    }
  }

  async function reprocessPrior(uploadId) {
    setReprocessing((r) => ({ ...r, [uploadId]: true }));
    setError(null);
    const fd = new FormData();
    fd.append("segments", "[]");
    fd.append("auto_detect_swings", "true");
    fd.append("starting_hole", String(parseInt(startingHole, 10) || 1));
    fd.append("ai_tracer_model", aiTracerModel);
    try {
      await api.reprocessLongUpload(adminPassword, uploadId, fd);
      setQueuedUploadId(uploadId);
      refreshPriorUploads();
    } catch (e) {
      setError(e.message);
    } finally {
      setReprocessing((r) => ({ ...r, [uploadId]: false }));
    }
  }

  async function testCutPrior(uploadId) {
    const detector = testCutDetector[uploadId] || "motion";
    const opts = {};
    if (detector === "audio") {
      const ratio = parseFloat(audioMinRatio[uploadId]);
      if (Number.isFinite(ratio) && ratio > 0) opts.audioMinPeakRatio = ratio;
    } else if (detector === "combined") {
      const aRatio = parseFloat(audioMinRatio[uploadId] ?? 5);
      const mRatio = parseFloat(motionRatio[uploadId] ?? 2);
      const win = parseFloat(pairWindow[uploadId] ?? 3);
      if (Number.isFinite(aRatio) && aRatio > 0) opts.audioMinPeakRatio = aRatio;
      if (Number.isFinite(mRatio) && mRatio > 0) opts.motionRatio = mRatio;
      if (Number.isFinite(win) && win > 0) opts.combinedPairWindowSec = win;
    } else {
      const ratio = parseFloat(motionRatio[uploadId]);
      if (Number.isFinite(ratio) && ratio > 0) opts.motionRatio = ratio;
    }
    if (detectOnly[uploadId]) opts.cutClips = false;
    setTestCutting((t) => ({ ...t, [uploadId]: true }));
    setError(null);
    try {
      const data = await api.testCutLongUpload(adminPassword, uploadId, detector, opts);
      setTestCutResults((r) => ({ ...r, [uploadId]: data }));
    } catch (e) {
      setError(e.message);
    } finally {
      setTestCutting((t) => ({ ...t, [uploadId]: false }));
    }
  }

  async function deletePrior(uploadId) {
    if (!window.confirm("Delete this stored long upload + its source files? Per-swing clips already produced from it are kept.")) {
      return;
    }
    try {
      await api.deleteLongUpload(adminPassword, uploadId);
      refreshPriorUploads();
    } catch (e) {
      setError(e.message);
    }
  }

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
    setQueuedUploadId(null);

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
      setQueuedUploadId(data?.upload_id ?? null);
      refreshPriorUploads();
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
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/showcase">Home videos</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
      </div>

      {queuedUploadId != null && (
        <div className="card" style={{ borderLeft: "4px solid var(--primary)" }}>
          <b>Upload #{queuedUploadId} queued.</b>{" "}
          <span className="small muted">
            Cutting, AI-tracer, and splicing run in the background — usually
            5–10 minutes. You can close this page; finished clips appear on{" "}
            <Link to="/admin/broadcast-clips">Broadcast</Link> when ready, and
            this list updates automatically.
          </span>
        </div>
      )}

      {priorUploads && priorUploads.length > 0 && (
        <div className="card">
          <h3 style={{ marginBottom: 6 }}>Previous long uploads</h3>
          <p className="small muted" style={{ marginBottom: 10 }}>
            Re-edit a stored long upload without re-uploading the source(s).
            Reprocess uses the current <b>Auto-detect</b>, <b>Starting hole</b>,
            and <b>AI tracer model</b> settings from the form below.
          </p>
          <div className="stack" style={{ gap: 8 }}>
            {priorUploads.map((u) => {
              const missing = u.tee_missing || u.green_missing;
              const status = u.processing_status || "completed";
              const busy = status === "pending" || status === "processing";
              const statusPill = (
                status === "processing" ? <span className="pill warn small">processing…</span> :
                status === "pending"    ? <span className="pill warn small">queued</span> :
                status === "failed"     ? <span className="pill err small">failed</span> :
                                          <span className="pill ok small">done</span>
              );
              return (
                <div key={u.id} className="card tight" style={{ margin: 0, padding: 10 }}>
                  <div className="inline" style={{ justifyContent: "space-between", width: "100%" }}>
                    <div className="small">
                      <b>#{u.id}</b>{" "}
                      <span style={{ marginLeft: 4 }}>{statusPill}</span>{" "}
                      <span className="muted">
                        · {u.course_name || `course #${u.course_id}`}{" "}
                        · {u.base_captured_at ? new Date(u.base_captured_at).toLocaleString() : "—"}
                        {u.dual_camera && <> · <span className="pill small">dual-cam</span></>}
                      </span>
                      <div className="tiny muted" style={{ marginTop: 2 }}>
                        tee: <code>{u.tee_original_filename || u.tee_filename}</code>
                        {fmtDuration(u.tee_duration_sec) && <> · {fmtDuration(u.tee_duration_sec)}</>}
                        {u.tee_size_mb != null && <> · {u.tee_size_mb} MB</>}
                        {u.tee_missing && <> · <span className="err-text">missing on disk</span></>}
                        {u.dual_camera && (
                          <>
                            <br />
                            green: <code>{u.green_original_filename || u.green_filename}</code>
                            {fmtDuration(u.green_duration_sec) && <> · {fmtDuration(u.green_duration_sec)}</>}
                            {u.green_size_mb != null && <> · {u.green_size_mb} MB</>}
                            {u.green_missing && <> · <span className="err-text">missing on disk</span></>}
                          </>
                        )}
                        {u.last_n_segments != null && status !== "processing" && (
                          <> · last run: {u.last_n_succeeded ?? "?"}/{u.last_n_segments} succeeded</>
                        )}
                      </div>
                      {status === "failed" && u.last_error && (
                        <div className="tiny err-text" style={{ marginTop: 2 }}>
                          error: <code>{u.last_error}</code>
                        </div>
                      )}
                      {busy && (
                        <div className="tiny muted" style={{ marginTop: 2 }}>
                          {status === "pending" ? (
                            "Queued — will start shortly."
                          ) : u.last_n_segments == null ? (
                            "Detecting swings (audio + motion)…"
                          ) : (
                            <>
                              <b>{u.last_n_segments}</b> swings identified ·
                              {" "}processed <b>{u.last_n_succeeded ?? 0}</b> /{" "}
                              {u.last_n_segments}
                            </>
                          )}
                        </div>
                      )}
                    </div>
                    <div className="inline" style={{ gap: 6, flexWrap: "wrap", justifyContent: "flex-end" }}>
                      <select
                        value={testCutDetector[u.id] || "motion"}
                        onChange={(e) =>
                          setTestCutDetector((d) => ({ ...d, [u.id]: e.target.value }))
                        }
                        disabled={!!testCutting[u.id] || missing || busy}
                        className="small"
                        style={{ fontSize: 12, padding: "2px 6px" }}
                        title="Swing detector used for the test cut. 'combined' AND's audio + motion: a swing only counts when an audio peak and a motion peak land within the pair window."
                      >
                        <option value="motion">motion</option>
                        <option value="audio">audio</option>
                        <option value="combined">combined</option>
                      </select>
                      {(testCutDetector[u.id] || "motion") === "audio" && (
                        <input
                          type="number"
                          min="1"
                          step="0.5"
                          value={audioMinRatio[u.id] ?? 10}
                          onChange={(e) =>
                            setAudioMinRatio((r) => ({ ...r, [u.id]: e.target.value }))
                          }
                          disabled={!!testCutting[u.id] || missing || busy}
                          className="small"
                          style={{ width: 60, fontSize: 12, padding: "2px 6px" }}
                          title="Minimum peak/median ratio. Higher = stricter (only the loudest thwacks). Default 10."
                        />
                      )}
                      {(testCutDetector[u.id] || "motion") === "motion" && (
                        <input
                          type="number"
                          min="1"
                          step="0.5"
                          value={motionRatio[u.id] ?? 4}
                          onChange={(e) =>
                            setMotionRatio((r) => ({ ...r, [u.id]: e.target.value }))
                          }
                          disabled={!!testCutting[u.id] || missing || busy}
                          className="small"
                          style={{ width: 60, fontSize: 12, padding: "2px 6px" }}
                          title="Motion threshold = median frame-diff × this ratio. Higher = stricter. Default 4."
                        />
                      )}
                      {(testCutDetector[u.id] || "motion") === "combined" && (
                        <span
                          className="tiny"
                          style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
                        >
                          a:
                          <input
                            type="number"
                            min="1"
                            step="0.5"
                            value={audioMinRatio[u.id] ?? 5}
                            onChange={(e) =>
                              setAudioMinRatio((r) => ({ ...r, [u.id]: e.target.value }))
                            }
                            disabled={!!testCutting[u.id] || missing || busy}
                            className="small"
                            style={{ width: 50, fontSize: 12, padding: "2px 6px" }}
                            title="Audio peak/median ratio for combined mode. Default 5."
                          />
                          m:
                          <input
                            type="number"
                            min="1"
                            step="0.5"
                            value={motionRatio[u.id] ?? 2}
                            onChange={(e) =>
                              setMotionRatio((r) => ({ ...r, [u.id]: e.target.value }))
                            }
                            disabled={!!testCutting[u.id] || missing || busy}
                            className="small"
                            style={{ width: 50, fontSize: 12, padding: "2px 6px" }}
                            title="Motion threshold ratio for combined mode. Default 2."
                          />
                          ±
                          <input
                            type="number"
                            min="0.5"
                            step="0.5"
                            value={pairWindow[u.id] ?? 3}
                            onChange={(e) =>
                              setPairWindow((r) => ({ ...r, [u.id]: e.target.value }))
                            }
                            disabled={!!testCutting[u.id] || missing || busy}
                            className="small"
                            style={{ width: 50, fontSize: 12, padding: "2px 6px" }}
                            title="Max seconds between an audio peak and a motion peak to count them as the same swing. Default 3."
                          />
                          s
                        </span>
                      )}
                      <label
                        className="tiny"
                        style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
                        title="Skip ffmpeg cutting and just list the detected swing times. Fast — useful while tuning the detector."
                      >
                        <input
                          type="checkbox"
                          checked={!!detectOnly[u.id]}
                          onChange={(e) =>
                            setDetectOnly((d) => ({ ...d, [u.id]: e.target.checked }))
                          }
                          disabled={!!testCutting[u.id] || missing || busy}
                          style={{ margin: 0 }}
                        />
                        detect only
                      </label>
                      <button
                        type="button"
                        className="secondary small"
                        onClick={() =>
                          setShowSource((s) => ({ ...s, [u.id]: !s[u.id] }))
                        }
                        disabled={missing}
                        title={
                          missing
                            ? "Source file missing on disk"
                            : "Show the raw long video upload(s) inline so you can scrub and verify swings"
                        }
                      >
                        {showSource[u.id] ? "Hide source" : "Show source"}
                      </button>
                      <button
                        type="button"
                        className="secondary small"
                        onClick={() => testCutPrior(u.id)}
                        disabled={!!testCutting[u.id] || missing || busy}
                        title="Cut the tee video into proposed swing clips (no AI tracer, no matcher) so you can review the segmentation"
                      >
                        {testCutting[u.id] ? "Cutting…" : "Test cut"}
                      </button>
                      <button
                        type="button"
                        className="small"
                        onClick={() => reprocessPrior(u.id)}
                        disabled={!!reprocessing[u.id] || missing || busy}
                        title={
                          busy
                            ? "Already running"
                            : missing
                            ? "Source file missing on disk"
                            : "Re-cut + re-process this upload with current settings"
                        }
                      >
                        {reprocessing[u.id] ? "Queuing…" : busy ? "Running…" : "Reprocess"}
                      </button>
                      <button
                        type="button"
                        className="ghost small err-text"
                        onClick={() => deletePrior(u.id)}
                        disabled={!!reprocessing[u.id] || busy}
                        title={busy ? "Wait for processing to finish" : "Delete this stored upload + its source files"}
                      >
                        Delete
                      </button>
                    </div>
                  </div>

                  {showSource[u.id] && !missing && (u.tee_url || u.green_url) && (
                    <div className="card tight" style={{ margin: "10px 0 0", background: "var(--surface-alt)" }}>
                      <div className="small" style={{ marginBottom: 6 }}>
                        <b>Source preview{u.dual_camera ? "s" : ""}</b>{" "}
                        <span className="tiny muted">
                          · first-frame thumbnails so you can verify the tee/green labels match the footage
                        </span>
                      </div>
                      <div
                        style={{
                          display: "grid",
                          gridTemplateColumns: "repeat(auto-fill, minmax(220px, max-content))",
                          gap: 10,
                          alignItems: "start",
                        }}
                      >
                        {u.tee_url && (
                          <div style={{ width: 240 }}>
                            <div className="tiny muted" style={{ marginBottom: 2 }}>
                              <b>tee</b> · <code>{u.tee_original_filename || u.tee_filename}</code>
                            </div>
                            <video
                              src={u.tee_url}
                              preload="metadata"
                              muted
                              playsInline
                              style={{ width: 240, height: 135, borderRadius: 6, background: "#000", objectFit: "cover" }}
                            />
                          </div>
                        )}
                        {u.dual_camera && u.green_url && (
                          <div style={{ width: 240 }}>
                            <div className="tiny muted" style={{ marginBottom: 2 }}>
                              <b>green</b> · <code>{u.green_original_filename || u.green_filename}</code>
                            </div>
                            <video
                              src={u.green_url}
                              preload="metadata"
                              muted
                              playsInline
                              style={{ width: 240, height: 135, borderRadius: 6, background: "#000", objectFit: "cover" }}
                            />
                          </div>
                        )}
                      </div>
                    </div>
                  )}

                  {testCutResults[u.id] && (
                    <div className="card tight" style={{ margin: "10px 0 0", background: "var(--surface-alt)" }}>
                      <div className="small" style={{ marginBottom: 6 }}>
                        <b>Test cut</b>{" "}
                        <span className="muted">
                          · detector: <code>{testCutResults[u.id].detector}</code>
                          {" "}· {testCutResults[u.id].n_windows} window
                          {testCutResults[u.id].n_windows === 1 ? "" : "s"}
                          {testCutResults[u.id].tee_fps != null && (
                            <> · {testCutResults[u.id].tee_fps} fps</>
                          )}
                          {testCutResults[u.id].dual_camera && <> · dual-cam</>}
                        </span>
                      </div>
                      {testCutResults[u.id].debug && (
                        <div className="tiny muted" style={{ marginBottom: 8, fontFamily: "monospace", lineHeight: 1.4 }}>
                          {testCutResults[u.id].debug.reason ? (
                            <div className="err-text">reason: {testCutResults[u.id].debug.reason}</div>
                          ) : testCutResults[u.id].detector === "combined" ? (
                            <>
                              {(() => {
                                const c = testCutResults[u.id].debug.combined || {};
                                const a = testCutResults[u.id].debug.audio || {};
                                const m = testCutResults[u.id].debug.motion || {};
                                return (
                                  <>
                                    combined: audio_peaks={c.n_audio_windows ?? "?"}
                                    {" "}· motion_peaks={c.n_motion_windows ?? "?"}
                                    {" "}· paired={c.n_paired ?? "?"}
                                    {" "}· window=±{c.pair_window_sec}s
                                    <br />
                                    audio: median={a.median_envelope?.toFixed?.(4) ?? "?"}
                                    {" "}· threshold={a.threshold?.toFixed?.(4) ?? "?"}
                                    {" "}(×{a.min_peak_ratio_used})
                                    {" "}· raw_peaks={a.n_raw_peaks ?? "?"}
                                    <br />
                                    motion: median={m.median_motion?.toFixed?.(4) ?? "?"}
                                    {" "}· threshold={m.threshold?.toFixed?.(4) ?? "?"}
                                    {" "}(×{m.motion_ratio_used})
                                    {" "}· raw_bursts={m.n_raw_bursts ?? "?"}
                                    {" "}· duration_ok={m.n_duration_ok ?? "?"}
                                    {Array.isArray(c.pairs) && c.pairs.length > 0 && (
                                      <>
                                        <br />
                                        pairs (audio_sec, motion_sec, Δt, ×audio, ×motion):
                                        <br />
                                        {c.pairs.map((p, i) => (
                                          <span key={i} style={{ marginRight: 10 }}>
                                            [{p.audio_peak_sec}s ↔ {p.motion_peak_sec}s]
                                            {" "}Δ{p.dt_sec}s
                                            {" "}×{p.audio_ratio}/{p.motion_ratio}
                                          </span>
                                        ))}
                                      </>
                                    )}
                                  </>
                                );
                              })()}
                            </>
                          ) : testCutResults[u.id].detector === "audio" ? (
                            <>
                              audio: median={testCutResults[u.id].debug.median_envelope?.toFixed(4)}{" "}
                              · threshold={testCutResults[u.id].debug.threshold?.toFixed(4)}{" "}
                              (×{testCutResults[u.id].debug.min_peak_ratio_used})
                              {testCutResults[u.id].debug.threshold_floor_hit && (
                                <> · <span className="warn-text">floor 0.02 active</span></>
                              )}
                              <br />
                              peaks: raw={testCutResults[u.id].debug.n_raw_peaks}
                              {" "}· kept_after_nms={testCutResults[u.id].debug.n_after_nms}
                              {" "}· min_sep={testCutResults[u.id].debug.min_separation_sec}s
                              {" "}over {testCutResults[u.id].debug.duration_sec?.toFixed(1)}s
                              {Array.isArray(testCutResults[u.id].debug.top_peaks) &&
                                testCutResults[u.id].debug.top_peaks.length > 0 && (
                                <>
                                  <br />
                                  top peaks (sec, ×ratio, kept-by-nms) — raise the
                                  ratio above the loudest unwanted peak to filter it out:
                                  <br />
                                  {testCutResults[u.id].debug.top_peaks.map((p, i) => (
                                    <span key={i} style={{ marginRight: 10 }}>
                                      {p.peak_sec}s ×{p.ratio} {p.kept ? "✓" : "✗"}
                                    </span>
                                  ))}
                                </>
                              )}
                            </>
                          ) : (
                            <>
                              motion: median={testCutResults[u.id].debug.median_motion?.toFixed(4)}{" "}
                              · threshold={testCutResults[u.id].debug.threshold?.toFixed(4)}{" "}
                              (×{testCutResults[u.id].debug.motion_ratio_used})
                              {" "}· max={testCutResults[u.id].debug.max_motion?.toFixed(4)}
                              <br />
                              bursts: raw={testCutResults[u.id].debug.n_raw_bursts}
                              {" "}· duration_ok={testCutResults[u.id].debug.n_duration_ok}
                              {" "}· final={testCutResults[u.id].debug.n_final}
                              {" "}· sampled @ {testCutResults[u.id].debug.effective_hz?.toFixed(1)} Hz
                              {" "}over {testCutResults[u.id].debug.duration_sec?.toFixed(1)}s
                              {Array.isArray(testCutResults[u.id].debug.top_raw_bursts) &&
                                testCutResults[u.id].debug.top_raw_bursts.length > 0 && (
                                <>
                                  <br />
                                  top raw bursts (sec, dur, ×ratio, passes-duration):
                                  <br />
                                  {testCutResults[u.id].debug.top_raw_bursts.map((b, i) => (
                                    <span key={i} style={{ marginRight: 10 }}>
                                      [{b.start_sec}-{b.end_sec}] {b.duration_sec}s ×{b.peak_ratio}{" "}
                                      {b.passes_duration ? "✓" : "✗"}
                                    </span>
                                  ))}
                                </>
                              )}
                            </>
                          )}
                        </div>
                      )}
                      {testCutResults[u.id].cuts.length === 0 && (
                        <div className="tiny muted">
                          No swings detected. The diagnostic line above shows
                          whether the median/threshold was too high or the
                          duration filter rejected everything.
                        </div>
                      )}
                      {testCutResults[u.id].cuts_skipped ? (
                        <div className="tiny" style={{ fontFamily: "monospace", lineHeight: 1.6 }}>
                          <div className="muted" style={{ marginBottom: 4 }}>
                            <b>{testCutResults[u.id].cuts.length}</b> swing
                            {testCutResults[u.id].cuts.length === 1 ? "" : "s"} detected
                            · cutting skipped
                          </div>
                          <table style={{ borderCollapse: "collapse", fontSize: 12 }}>
                            <thead>
                              <tr style={{ textAlign: "left" }}>
                                <th style={{ padding: "2px 10px 2px 0" }}>#</th>
                                <th style={{ padding: "2px 10px 2px 0" }}>peak</th>
                                <th style={{ padding: "2px 10px 2px 0" }}>×ratio</th>
                                <th style={{ padding: "2px 10px 2px 0" }}>window</th>
                                <th style={{ padding: "2px 10px 2px 0" }}>conf</th>
                              </tr>
                            </thead>
                            <tbody>
                              {testCutResults[u.id].cuts.map((c) => (
                                <tr key={c.index}>
                                  <td style={{ padding: "1px 10px 1px 0" }}>{c.index + 1}</td>
                                  <td style={{ padding: "1px 10px 1px 0" }}>
                                    {c.peak_time_sec != null ? `${c.peak_time_sec.toFixed(2)}s` : "—"}
                                  </td>
                                  <td style={{ padding: "1px 10px 1px 0" }}>
                                    {c.ratio != null ? `×${c.ratio.toFixed(1)}` : "—"}
                                  </td>
                                  <td style={{ padding: "1px 10px 1px 0" }}>
                                    {c.start_sec.toFixed(1)}–{c.end_sec.toFixed(1)}s
                                  </td>
                                  <td style={{ padding: "1px 10px 1px 0" }}>
                                    {c.confidence || "—"}
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </div>
                      ) : (
                        <div
                          style={{
                            display: "grid",
                            gridTemplateColumns: testCutResults[u.id].dual_camera
                              ? "repeat(auto-fill, minmax(360px, 1fr))"
                              : "repeat(auto-fill, minmax(220px, 1fr))",
                            gap: 8,
                          }}
                        >
                          {testCutResults[u.id].cuts.map((c) => (
                            <div key={c.index} style={{ minWidth: 0 }}>
                              <div className="tiny muted" style={{ marginBottom: 2 }}>
                                #{c.index + 1} · {c.start_sec.toFixed(1)}–{c.end_sec.toFixed(1)}s
                                {" "}({c.duration_sec}s)
                                {c.peak_time_sec != null && (
                                  <> · peak {c.peak_time_sec.toFixed(2)}s</>
                                )}
                                {c.ratio != null && <> · ×{c.ratio.toFixed(1)}</>}
                                {c.confidence && <> · {c.confidence}</>}
                              </div>
                              <div
                                style={{
                                  display: "grid",
                                  gridTemplateColumns: testCutResults[u.id].dual_camera ? "1fr 1fr" : "1fr",
                                  gap: 4,
                                }}
                              >
                                {c.ok && c.url ? (
                                  <video
                                    src={c.url}
                                    controls
                                    playsInline
                                    preload="metadata"
                                    style={{ width: "100%", borderRadius: 6, background: "#000" }}
                                  />
                                ) : (
                                  <div className="tiny err-text">tee cut failed</div>
                                )}
                                {testCutResults[u.id].dual_camera && (
                                  c.green_url ? (
                                    <video
                                      src={c.green_url}
                                      controls
                                      playsInline
                                      preload="metadata"
                                      style={{ width: "100%", borderRadius: 6, background: "#000" }}
                                    />
                                  ) : c.green_ok === false ? (
                                    <div className="tiny err-text">green cut failed</div>
                                  ) : null
                                )}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}

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
              <div className="small muted" style={{ marginTop: 4 }}>
                {progress < 100
                  ? `${progress}% uploaded — once the upload finishes, AI cutting + splicing runs in the background.`
                  : "Upload complete — queuing background job…"}
              </div>
            </div>
          )}

          <button disabled={uploading || !file || !courseId}>
            {uploading
              ? `Uploading… ${progress}%`
              : (autoDetectSwings
                ? (fileGreen
                  ? "Upload + queue auto-detect + AI tracer + composite"
                  : "Upload + queue auto-detect + matcher")
                : (fileGreen
                  ? `Upload + queue ${segments.length} swing${segments.length === 1 ? "" : "s"} + AI tracer + composite`
                  : `Upload + queue ${segments.length} swing${segments.length === 1 ? "" : "s"} + matcher`))}
          </button>
        </form>
      </div>
    </div>
  );
}
