/**
 * Production queue — operator view of every uploaded long video.
 *
 * Each upload becomes one card with thumbnails of the tee + green
 * source videos plus probe metadata (duration / frames / fps / size)
 * and the captured timestamp window. Action buttons depend on state:
 *
 *   - swing_count='multiple' AND processing in flight
 *       → everything greyed out, "Production in Progress" badge
 *   - already produced (last_n_succeeded > 0 OR status='completed')
 *       → Edit · Re-Produce · Delete
 *   - swing_count='single' not yet produced
 *       → Edit · Produce · Delete
 *
 * Action handlers are stubs for now — the user will hand over the
 * functional spec next.
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function fmtDateTime(iso) {
  if (!iso) return "—";
  try {
    const d = new Date(iso);
    return d.toLocaleString();
  } catch {
    return iso;
  }
}

function fmtDuration(sec) {
  if (sec == null) return "—";
  const s = Math.round(sec);
  const m = Math.floor(s / 60);
  const r = s % 60;
  return m ? `${m}m ${String(r).padStart(2, "0")}s` : `${r}s`;
}

function addSeconds(iso, sec) {
  if (!iso || sec == null) return null;
  try {
    return new Date(new Date(iso).getTime() + sec * 1000).toISOString();
  } catch {
    return null;
  }
}

function uploadState(row) {
  // Anything mid-production blocks every action until it finishes.
  if (row.processing_status === "processing") return "processing";
  // Anything that finished a run (clips emitted) is "produced".
  if (
    row.processing_status === "completed" &&
    (row.last_n_succeeded || 0) > 0
  ) {
    return "produced";
  }
  if ((row.last_n_succeeded || 0) > 0) return "produced";
  return "queued";
}

function MetaRow({ k, v }) {
  // Empty string / null / undefined → blank value (no em-dash). The label
  // stays so the rows in adjacent tiles still line up vertically.
  const display = v === null || v === undefined || v === "" ? "" : v;
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "70px 1fr",
        gap: 6,
        fontSize: "0.78rem",
        lineHeight: 1.5,
      }}
    >
      <span className="muted">{k}</span>
      <span style={{ wordBreak: "break-word" }}>{display}</span>
    </div>
  );
}

function qualityText(qualityLabel, width, height) {
  const dims = width && height ? `${width}×${height}` : null;
  if (qualityLabel && dims) return `${qualityLabel} · ${dims}`;
  if (qualityLabel) return qualityLabel;
  if (dims) return dims;
  return "";
}

function Thumb({ src, alt, missing, placeholder, onClick }) {
  // Shared thumbnail box for all three Production tiles. Clicking opens
  // the video viewer when an onClick handler is provided. Width is
  // 100% of the parent tile so the three tiles spread evenly via the
  // outer flex container.
  const clickable = !!onClick;
  return (
    <div
      role={clickable ? "button" : undefined}
      tabIndex={clickable ? 0 : undefined}
      onClick={clickable ? onClick : undefined}
      onKeyDown={clickable
        ? (e) => {
            if (e.key === "Enter" || e.key === " ") {
              e.preventDefault();
              onClick();
            }
          }
        : undefined}
      style={{
        width: "100%",
        aspectRatio: "16 / 9",
        background: "var(--border, #222)",
        borderRadius: 6,
        overflow: "hidden",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        marginBottom: 8,
        cursor: clickable ? "pointer" : "default",
        position: "relative",
      }}
    >
      {src ? (
        <img
          src={src}
          alt={alt}
          style={{ width: "100%", height: "100%", objectFit: "cover" }}
        />
      ) : (
        <span className="small muted">
          {missing ? "File missing" : (placeholder || "No preview")}
        </span>
      )}
      {clickable && src && (
        <span
          aria-hidden
          style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            background: "rgba(0,0,0,0.0)",
            transition: "background 120ms ease",
            color: "#fff",
            fontSize: 36,
            textShadow: "0 2px 8px rgba(0,0,0,0.6)",
            pointerEvents: "none",
          }}
        >
          ▶
        </span>
      )}
    </div>
  );
}

function VideoTile({ label, thumb, durationSec, nbFrames, fps, sizeMb,
                     startsAt, missing, notUploaded, qualityLabel, width, height,
                     videoUrl, onOpenViewer }) {
  // When the tile has no underlying source (file missing or never
  // uploaded), every meta row renders blank — the labels stay so the
  // tiles in the row line up visually.
  const hasSource = !!(thumb || videoUrl || durationSec != null);
  return (
    <div style={{ flex: "1 1 0", minWidth: 200, maxWidth: 340 }}>
      <div className="tiny upper muted" style={{ marginBottom: 4 }}>{label}</div>
      <Thumb
        src={thumb}
        alt={`${label} thumbnail`}
        missing={missing}
        placeholder={notUploaded ? "Not Uploaded" : "No preview"}
        onClick={videoUrl ? () => onOpenViewer(videoUrl, label) : undefined}
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <MetaRow k="Quality" v={hasSource ? qualityText(qualityLabel, width, height) : ""} />
        <MetaRow k="Length" v={durationSec != null ? fmtDuration(durationSec) : ""} />
        <MetaRow k="Frames" v={nbFrames != null ? nbFrames : ""} />
        <MetaRow k="Frame rate" v={fps != null ? `${fps} fps` : ""} />
        <MetaRow k="Size" v={sizeMb != null ? `${sizeMb} MB` : ""} />
        <MetaRow k="Starts" v={hasSource ? fmtDateTime(startsAt) : ""} />
        <MetaRow
          k="Ends"
          v={hasSource && durationSec != null
            ? fmtDateTime(addSeconds(startsAt, durationSec))
            : ""}
        />
      </div>
    </div>
  );
}

function ProducedTile({ clips, onOpenViewer }) {
  // Right-most tile on the Production card: thumbnail + summary of every
  // produced clip cut from this upload. Falls back to a "Not produced"
  // placeholder when the worker hasn't emitted anything.
  const has = clips && clips.length > 0;
  const first = has ? clips[0] : null;
  const aces = has ? clips.filter((c) => c.ball_in_cup).length : 0;
  const holes = has
    ? clips.map((c) => c.hole_number).filter((h, i, a) => h != null && a.indexOf(h) === i)
    : [];
  return (
    <div style={{ flex: "1 1 0", minWidth: 200, maxWidth: 340 }}>
      <div className="tiny upper muted" style={{ marginBottom: 4 }}>
        Produced Video
      </div>
      <Thumb
        src={first?.thumbnail_url}
        alt="Produced clip thumbnail"
        placeholder={has ? "No preview" : "Not produced"}
        onClick={first?.video_url ? () => onOpenViewer(first.video_url, "Produced Video") : undefined}
      />
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        <MetaRow k="Clips" v={has ? clips.length : ""} />
        <MetaRow k="Aces" v={has ? aces : ""} />
        <MetaRow k="Holes" v={holes.length ? holes.join(", ") : ""} />
        {has && clips.length > 1 && (
          <div style={{ marginTop: 4 }}>
            <div className="tiny upper muted" style={{ marginBottom: 2 }}>
              Open clip
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 4 }}>
              {clips.map((c) => (
                <button
                  key={c.id}
                  type="button"
                  className="ghost"
                  onClick={() => onOpenViewer(c.video_url, `Produced — hole ${c.hole_number}`)}
                  style={{
                    fontSize: "0.7rem",
                    padding: "2px 8px",
                    width: "auto",
                  }}
                  title={`Play produced clip for hole ${c.hole_number}`}
                >
                  #{c.id}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

function EditWizard({ row, adminPassword, onClose }) {
  // Auto-detection wizard for single-swing uploads. On mount, hits the
  // /long-uploads/{id}/auto-detect endpoint which runs the cheap part
  // of the AI tracer pipeline (audio impact → address frame → Claude
  // handedness call) and returns enough landmarks to seed manual
  // tweaks: handedness, address frame JPG, ball-at-rest position,
  // ball detection ROI, and a target estimate.
  const [detection, setDetection] = useState(null);
  const [running, setRunning] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!row) return;
    let cancelled = false;
    setRunning(true);
    setError(null);
    api
      .autoDetectLongUpload(adminPassword, row.id)
      .then((data) => { if (!cancelled) setDetection(data); })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setRunning(false); });
    return () => { cancelled = true; };
  }, [row, adminPassword]);

  if (!row) return null;

  const fw = detection?.frame_width;
  const fh = detection?.frame_height;

  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Edit wizard for upload ${row.id}`}
      onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.85)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 1000, padding: 24, cursor: "zoom-out",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="card"
        style={{
          maxWidth: "min(960px, 95vw)", width: "100%",
          maxHeight: "90vh", overflow: "auto",
          cursor: "default", margin: 0,
        }}
      >
        <div
          className="row"
          style={{ alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}
        >
          <div>
            <h3 style={{ margin: 0 }}>Edit wizard</h3>
            <div className="small muted">
              Upload #{row.id} · {row.course_name || `course ${row.course_id}`} · single swing
            </div>
          </div>
          <button
            type="button"
            className="ghost"
            onClick={onClose}
            style={{ width: "auto" }}
          >
            Close ✕
          </button>
        </div>

        <div
          className="card"
          style={{
            background: "rgba(255,255,255,0.04)",
            border: "1px solid var(--border)",
            margin: "0 0 12px",
            padding: 12,
          }}
        >
          {running && (
            <div className="row" style={{ alignItems: "center", gap: 12 }}>
              <div
                className="shimmer"
                style={{ width: 18, height: 18, borderRadius: "50%" }}
              />
              <span className="small">
                Auto-detecting handedness, address frame, and ball position…
              </span>
            </div>
          )}
          {error && (
            <div className="err-text small">
              Auto-detect failed: {error}
            </div>
          )}
          {!running && !error && detection && (
            <DetectionPreview detection={detection} frameW={fw} frameH={fh} />
          )}
        </div>

        <div
          className="row"
          style={{ gap: 8, justifyContent: "flex-end", marginTop: 6 }}
        >
          <button
            type="button"
            className="ghost"
            onClick={onClose}
            style={{ width: "auto" }}
          >
            Cancel
          </button>
          <button
            type="button"
            disabled={running || !!error || !detection}
            onClick={onClose}
            style={{ width: "auto" }}
            title="Save & continue — wiring lands next"
          >
            Save (stub)
          </button>
        </div>
      </div>
    </div>
  );
}

function DetectionPreview({ detection, frameW, frameH }) {
  const addressUrl = detection.address?.image_url;
  const ball = detection.ball_at_rest;
  const roi = detection.ball_detection_area;
  const target = detection.target;

  // Convert native pixel coords to overlay percentages so the markers
  // sit correctly on the address-frame JPG regardless of its on-screen
  // size. Skip the overlay entirely if we don't have frame dimensions.
  const hasDims = !!(frameW && frameH);
  const pct = (v, span) => (hasDims ? `${(v / span) * 100}%` : "0%");

  return (
    <div>
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "minmax(260px, 1.4fr) minmax(220px, 1fr)",
          gap: 16,
          alignItems: "flex-start",
        }}
      >
        <div>
          <div className="tiny upper muted" style={{ marginBottom: 4 }}>
            Address frame{" "}
            {detection.address?.frame != null && (
              <span style={{ textTransform: "none" }}>
                · frame {detection.address.frame}
              </span>
            )}
          </div>
          <div
            style={{
              position: "relative",
              width: "100%",
              aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
              background: "var(--border, #222)",
              borderRadius: 6,
              overflow: "hidden",
            }}
          >
            {addressUrl ? (
              <img
                src={addressUrl}
                alt="Detected address frame"
                style={{ width: "100%", height: "100%", objectFit: "cover" }}
              />
            ) : (
              <div
                className="muted small"
                style={{
                  position: "absolute", inset: 0,
                  display: "flex", alignItems: "center", justifyContent: "center",
                }}
              >
                No address frame
              </div>
            )}

            {hasDims && roi && (
              <div
                style={{
                  position: "absolute",
                  left: pct(roi.x, frameW),
                  top: pct(roi.y, frameH),
                  width: pct(roi.w, frameW),
                  height: pct(roi.h, frameH),
                  border: "2px solid #22c55e",
                  borderRadius: 4,
                  pointerEvents: "none",
                  boxShadow: "0 0 0 1px rgba(34,197,94,0.4)",
                }}
                title="Ball detection area"
              />
            )}

            {hasDims && ball && (
              <div
                style={{
                  position: "absolute",
                  left: pct(ball.x, frameW),
                  top: pct(ball.y, frameH),
                  width: 12, height: 12,
                  borderRadius: "50%",
                  background: "#22c55e",
                  border: "2px solid #fff",
                  transform: "translate(-50%, -50%)",
                  pointerEvents: "none",
                  boxShadow: "0 0 6px rgba(0,0,0,0.6)",
                }}
                title={`Ball at rest (${ball.x}, ${ball.y})`}
              />
            )}

            {hasDims && ball && target && (
              <svg
                aria-hidden
                viewBox={`0 0 ${frameW} ${frameH}`}
                preserveAspectRatio="none"
                style={{
                  position: "absolute", inset: 0,
                  width: "100%", height: "100%",
                  pointerEvents: "none",
                }}
              >
                <defs>
                  <marker
                    id="targetArrow"
                    viewBox="0 0 10 10"
                    refX="8" refY="5"
                    markerWidth="6" markerHeight="6"
                    orient="auto-start-reverse"
                  >
                    <path d="M 0 0 L 10 5 L 0 10 z" fill="#fbbf24" />
                  </marker>
                </defs>
                <line
                  x1={ball.x} y1={ball.y}
                  x2={target.x} y2={target.y}
                  stroke="#fbbf24"
                  strokeWidth={Math.max(3, Math.round((frameW || 1280) / 320))}
                  strokeLinecap="round"
                  markerEnd="url(#targetArrow)"
                />
              </svg>
            )}
          </div>
          <div className="tiny muted" style={{ marginTop: 6 }}>
            Green dot = ball at rest · Green box = ball detection area ·
            Yellow arrow = target direction.
          </div>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
          <DetectField
            label="Handedness"
            value={detection.handedness?.value
              ? detection.handedness.value.charAt(0).toUpperCase() + detection.handedness.value.slice(1)
              : "Unknown"}
            sub={detection.handedness?.confidence
              ? `Confidence: ${detection.handedness.confidence}`
              : null}
          />
          <DetectField
            label="Address frame"
            value={detection.address?.frame != null
              ? `Frame ${detection.address.frame}`
              : "Not detected"}
          />
          <DetectField
            label="Impact frame"
            value={detection.impact?.frame != null
              ? `Frame ${detection.impact.frame}`
              : "Not detected"}
            sub={detection.impact?.method ? `via ${detection.impact.method}` : null}
          />
          <DetectField
            label="Resting ball"
            value={ball ? `${ball.x}, ${ball.y} px` : "Not detected"}
          />
          <DetectField
            label="Detection area"
            value={roi
              ? `${roi.w} × ${roi.h} px @ (${roi.x}, ${roi.y})`
              : "Not detected"}
          />
          <DetectField
            label="Target"
            value={target ? `${target.x}, ${target.y} px` : "Not detected"}
            sub={target?.method || null}
          />
          <DetectField
            label="Frame size"
            value={frameW && frameH ? `${frameW} × ${frameH} px` : "Unknown"}
            sub={detection.fps ? `${detection.fps} fps` : null}
          />
        </div>
      </div>
    </div>
  );
}

function DetectField({ label, value, sub }) {
  return (
    <div>
      <div className="tiny upper muted">{label}</div>
      <div style={{ fontSize: "0.95rem" }}>{value}</div>
      {sub && <div className="tiny muted">{sub}</div>}
    </div>
  );
}

function VideoLightbox({ url, title, onClose }) {
  if (!url) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title || "Video viewer"}
      onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.85)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 1000, padding: 24, cursor: "zoom-out",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: "min(1280px, 95vw)", width: "100%",
          cursor: "default",
        }}
      >
        <div
          className="row"
          style={{
            color: "#fff", marginBottom: 8, alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <b style={{ fontSize: "0.9rem" }}>{title || "Video"}</b>
          <button
            type="button"
            className="ghost"
            onClick={onClose}
            style={{ width: "auto" }}
          >
            Close ✕
          </button>
        </div>
        <video
          src={url}
          controls
          autoPlay
          style={{
            width: "100%", maxHeight: "80vh",
            background: "#000", borderRadius: 6,
          }}
        />
      </div>
    </div>
  );
}

export default function AdminProduction() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  const [rows, setRows] = useState(null);
  const [error, setError] = useState(null);
  const [busyId, setBusyId] = useState(null);
  const [viewer, setViewer] = useState(null); // {url, title}
  const [editingRow, setEditingRow] = useState(null);

  function openViewer(url, title) {
    if (!url) return;
    setViewer({ url, title });
  }

  async function load() {
    setError(null);
    try {
      const data = await api.listLongUploads(adminPassword);
      setRows(data);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (!adminPassword) return;
    load();
    // Poll while anything is actively producing so the badge clears
    // automatically when the background worker finishes.
    const id = setInterval(load, 8000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [adminPassword]);

  async function handleDelete(row) {
    if (!confirm(`Delete upload #${row.id}? This removes the source video(s).`)) return;
    setBusyId(row.id);
    try {
      await api.deleteLongUpload(adminPassword, row.id);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  }

  function handleEdit(row) {
    // Single-swing uploads open the EditWizard modal — auto-detection
    // of handedness / address frame / ball position / ROI / target.
    // Multiple-swing uploads still need the per-segment editor, so
    // those bounce to the existing long-upload page.
    if (row.swing_count === "single") {
      setEditingRow(row);
    } else {
      window.location.href = `/admin/long-upload?upload_id=${row.id}`;
    }
  }

  async function handleProduce(row) {
    // Stub: kicks off a default reprocess on the existing row. Matches
    // the auto-produce defaults used by /clips/quick-upload.
    setBusyId(row.id);
    try {
      const fd = new FormData();
      fd.append("segments", "[]");
      fd.append("auto_detect_swings", "true");
      fd.append("starting_hole", "1");
      await api.reprocessLongUpload(adminPassword, row.id, fd);
      await load();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  }

  if (!adminPassword) {
    return (
      <div className="wrap">
        <Brand subtitle="Operator Console" />
        <div className="card center">
          <h2>Admin password required</h2>
          <Link to="/admin">
            <button style={{ marginTop: 10 }}>Sign in</button>
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap wide">
      <Brand subtitle="Operator Console" />
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants">Players</Link>
        <Link to="/admin/courses">Courses</Link>
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production" className="active">Production</Link>
        <Link to="/admin/produced-clips">Produced Clips</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>Production queue</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Every uploaded long video, newest first. Multi-swing rounds
          auto-produce on upload; one-swing clips wait here until you
          hit <b>Produce</b>. Produced clips land on{" "}
          <Link to="/admin/broadcast-clips">Broadcast</Link>.
        </p>
      </div>

      {error && <div className="card err-text small">{error}</div>}

      {rows === null && (
        <div className="card">
          <div className="shimmer" style={{ height: 200 }} />
        </div>
      )}

      {rows?.length === 0 && (
        <div className="card muted center" style={{ padding: 40 }}>
          Nothing in the production queue.{" "}
          <Link to="/admin/upload-videos">Upload a video →</Link>
        </div>
      )}

      {rows?.map((row) => {
        const state = uploadState(row);
        const greyed = state === "processing";
        const busy = busyId === row.id;
        return (
          <div
            key={row.id}
            className="card"
            style={{
              marginBottom: 12,
              opacity: greyed ? 0.6 : 1,
              position: "relative",
            }}
          >
            <div
              className="row"
              style={{
                gap: 10, flexWrap: "wrap", alignItems: "baseline",
                marginBottom: 10,
              }}
            >
              <h4 style={{ margin: 0 }}>
                #{row.id} · {row.course_name || `course ${row.course_id}`}
              </h4>
              <span className="small muted">
                {row.swing_count === "single" ? "One swing" : "Multiple swings"}
              </span>
              <span className="small muted">·</span>
              <span className="small muted">
                Uploaded {fmtDateTime(row.created_at)}
              </span>
            </div>

            <div
              style={{
                display: "flex",
                gap: 20,
                alignItems: "flex-start",
                flexWrap: "wrap",
              }}
            >
              <div
                style={{
                  flex: "1 1 600px",
                  display: "flex",
                  gap: 16,
                  flexWrap: "wrap",
                  alignItems: "flex-start",
                }}
              >
                <VideoTile
                  label="Tee Angle (Raw Video)"
                  thumb={row.tee_thumbnail_url}
                  durationSec={row.tee_duration_sec}
                  nbFrames={row.tee_nb_frames}
                  fps={row.tee_fps}
                  sizeMb={row.tee_size_mb}
                  startsAt={row.base_captured_at}
                  missing={row.tee_missing}
                  qualityLabel={row.tee_quality_label}
                  width={row.tee_width}
                  height={row.tee_height}
                  videoUrl={row.tee_url}
                  onOpenViewer={openViewer}
                />
                <VideoTile
                  label="Green Angle (Raw Video)"
                  thumb={row.dual_camera ? row.green_thumbnail_url : null}
                  durationSec={row.dual_camera ? row.green_duration_sec : null}
                  nbFrames={row.dual_camera ? row.green_nb_frames : null}
                  fps={row.dual_camera ? row.green_fps : null}
                  sizeMb={row.dual_camera ? row.green_size_mb : null}
                  startsAt={row.dual_camera ? row.base_captured_at : null}
                  missing={row.dual_camera ? row.green_missing : false}
                  notUploaded={!row.dual_camera}
                  qualityLabel={row.dual_camera ? row.green_quality_label : null}
                  width={row.dual_camera ? row.green_width : null}
                  height={row.dual_camera ? row.green_height : null}
                  videoUrl={row.dual_camera ? row.green_url : null}
                  onOpenViewer={openViewer}
                />
                <ProducedTile
                  clips={row.produced_clips}
                  onOpenViewer={openViewer}
                />
              </div>

              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 8,
                  alignItems: "stretch",
                  minWidth: 160,
                  flexShrink: 0,
                }}
              >
                {state === "processing" && (
                  <span
                    className="small"
                    style={{
                      padding: "4px 10px",
                      borderRadius: 999,
                      background: "rgba(255, 176, 0, 0.15)",
                      border: "1px solid rgba(255, 176, 0, 0.5)",
                      textAlign: "center",
                    }}
                  >
                    Production in Progress
                  </span>
                )}
                {state === "produced" && (
                  <span
                    className="small"
                    style={{
                      padding: "4px 10px",
                      borderRadius: 999,
                      background: "rgba(40, 168, 92, 0.15)",
                      border: "1px solid rgba(40, 168, 92, 0.5)",
                      textAlign: "center",
                    }}
                  >
                    Produced · {row.last_n_succeeded}/{row.last_n_segments || row.last_n_succeeded} clips
                  </span>
                )}
                {state === "queued" && (
                  <span
                    className="small"
                    style={{
                      padding: "4px 10px",
                      borderRadius: 999,
                      background: "rgba(120, 120, 120, 0.15)",
                      border: "1px solid rgba(120, 120, 120, 0.5)",
                      textAlign: "center",
                    }}
                  >
                    Queued
                  </span>
                )}
                <button
                  className="small"
                  onClick={() => handleEdit(row)}
                  disabled={greyed || busy}
                >
                  Edit
                </button>
                {state === "produced" ? (
                  <button
                    className="small"
                    onClick={() => handleProduce(row)}
                    disabled={greyed || busy}
                  >
                    Re-Produce
                  </button>
                ) : (
                  state === "queued" && (
                    <button
                      className="small"
                      onClick={() => handleProduce(row)}
                      disabled={greyed || busy}
                    >
                      Produce
                    </button>
                  )
                )}
                <button
                  className="small danger"
                  onClick={() => handleDelete(row)}
                  disabled={greyed || busy}
                >
                  Delete
                </button>
              </div>
            </div>

            {row.last_error && (
              <div className="err-text small" style={{ marginTop: 10 }}>
                {row.last_error}
              </div>
            )}
          </div>
        );
      })}

      <VideoLightbox
        url={viewer?.url}
        title={viewer?.title}
        onClose={() => setViewer(null)}
      />

      {editingRow && (
        <EditWizard
          row={editingRow}
          adminPassword={adminPassword}
          onClose={() => setEditingRow(null)}
        />
      )}
    </div>
  );
}
