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
    // Stub — the user will spec the editor surface. For now bounce to
    // the existing long-upload page which already has segment editing.
    window.location.href = `/admin/long-upload?upload_id=${row.id}`;
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
    </div>
  );
}
