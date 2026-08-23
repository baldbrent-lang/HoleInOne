/**
 * Production queue — operator view of every uploaded long video.
 *
 * Each upload becomes one card with thumbnails of the tee + green
 * source videos plus probe metadata (duration / frames / fps / size)
 * and the captured timestamp window. Action buttons depend on state:
 *
 *   - producing (or waiting its turn in the produce queue)
 *       → everything greyed out, status overlay naming the stage
 *   - already produced (last_n_succeeded > 0 OR status='completed')
 *       → Edit · Re-Produce · Delete
 *   - uploaded, nothing produced yet
 *       → Edit · Produce · Delete
 *
 * Every upload auto-produces on arrival now, so the third state is
 * mostly a produce that found no swings, or one an operator deleted the
 * clips from. `swing_count` survives only as the Edit wizard's shape
 * switch (see isMulti); it is no longer an operator choice.
 */
import { Component, useCallback, useEffect, useMemo, useRef, useState }
  from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";
import { useInfiniteList } from "../hooks/useInfiniteList.js";

/**
 * A CRASH IN ONE MODAL MUST NOT TAKE THE PAGE WITH IT.
 *
 * React unmounts the whole tree when a render throws and nothing
 * catches it, so a single bad value inside a dialog left the operator
 * looking at a blank white browser window -- no error, no page, nothing
 * to report but "it went white". That is the worst possible failure
 * mode: the one piece of information needed to fix it is the one thing
 * destroyed by it.
 *
 * This keeps the page, shows what actually threw, and logs the stack to
 * the console so it can be copied.
 */
class Boundary extends Component {
  constructor(props) {
    super(props);
    this.state = { err: null };
  }

  static getDerivedStateFromError(err) {
    return { err };
  }

  componentDidCatch(err, info) {
    // eslint-disable-next-line no-console
    console.error(`[${this.props.name || "panel"}] render failed`, err, info);
  }

  render() {
    if (!this.state.err) return this.props.children;
    const msg = this.state.err?.message || String(this.state.err);
    return (
      <div
        role="dialog"
        onClick={this.props.onClose}
        style={{
          position: "fixed", inset: 0, zIndex: 1400,
          background: "rgba(0,0,0,0.75)", display: "flex",
          alignItems: "center", justifyContent: "center", padding: 16,
        }}
      >
        <div className="card" onClick={(e) => e.stopPropagation()}
             style={{ margin: 0, padding: 18, maxWidth: 620 }}>
          <div className="row" style={{ justifyContent: "space-between",
                                        gap: 12 }}>
            <b>{this.props.name || "This panel"} could not be drawn</b>
            {this.props.onClose && (
              <button className="btn ghost" style={{ width: "auto" }}
                      onClick={this.props.onClose}>Close ✕</button>
            )}
          </div>
          <div className="err-text small" style={{ marginTop: 10 }}>{msg}</div>
          <div className="tiny muted" style={{ marginTop: 8 }}>
            The rest of the page is fine — close this and carry on. The
            full stack is in the browser console.
          </div>
          <pre className="tiny" style={{
            marginTop: 8, maxHeight: 220, overflow: "auto",
            whiteSpace: "pre-wrap", opacity: 0.75,
          }}>{this.state.err?.stack || ""}</pre>
        </div>
      </div>
    );
  }
}
import { parseApiDate } from "../time.js";
import { ViewMapModal } from "../components/ViewMapModal.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function fmtDateTime(iso) {
  const d = parseApiDate(iso);
  return d ? d.toLocaleString() : "—";
}

// Server errors sometimes carry a whole HTML page (Replit's proxy 502
// when a render outlives the request timeout). Strip markup and shorten
// so the wizard shows a readable one-liner instead of a wall of HTML.
function sanitizeErr(msg) {
  const s = String(msg || "");
  if (/<!DOCTYPE|<html|couldn&#39;t reach|couldn't reach/i.test(s)) {
    return (
      "the server took too long and the connection timed out (502). " +
      "Try a tighter start/end window on Step 1, then re-run."
    );
  }
  const plain = s.replace(/<[^>]*>/g, " ").replace(/\s+/g, " ").trim();
  return plain.length > 220 ? `${plain.slice(0, 220)}…` : plain;
}

// Signature of the three clip frames the tracer render depends on
// (start / impact / end). When any of them changes on Step 1, the
// cached Step-2 tracer is stale and must be re-rendered.
function frameSig(d) {
  if (!d) return "";
  return `${d.startFrame ?? "s"}|${d.impactFrame ?? "i"}|${d.endFrame ?? "e"}`;
}

// A TIME OF DAY, to the millisecond. The two raw clips share nothing
// else -- different frame rates, different start moments, so a frame
// number means nothing across them -- and the whole reason the pair can
// be lined up at all is that both cameras stamped when they started
// rolling. Milliseconds because a frame is 20-33ms and the question
// being asked of this readout is whether two pictures are the same
// instant.
function fmtWallClock(ms) {
  if (ms == null || !Number.isFinite(ms)) return "—";
  const d = new Date(ms);
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
    + `.${p(d.getMilliseconds(), 3)}`;
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
  const d = parseApiDate(iso);
  if (!d) return null;
  return new Date(d.getTime() + sec * 1000).toISOString();
}

// How long the clip runs past the landing. Long enough to see the ball
// settle, short enough that it does not sit on an empty green. Mirrors
// LANDING_TAIL_SEC on the backend, which is what actually cuts.
const LANDING_TAIL_SEC = 1.5;

function uploadState(row, busy) {
  // The optimistic flag first: the POST returns before the worker flips
  // processing_status, and without this the card un-greys in that gap.
  if (busy) return "processing";
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

// What a producing card says about itself. Sits on top of the greyed
// card (outside its opacity, so it stays legible) and names the Debug3
// stage the backend is actually on — "Finding swing candidates",
// "Swing 2 of 3: finding the ball at impact", "Building the clip".
//
// Three states, in priority order: waiting its turn in the produce
// queue, running with a named stage, running before the first stage has
// been reported. Never shown on an idle card.
/* The wrist-speed trace stage 1 works from. A count of zero candidates
   cannot tell you whether pose never saw the golfer, whether the hands
   never moved fast enough, or whether the spine-bend gate rejected a
   real swing — and those need three different fixes. The shape can.
   Bars are per-sample wrist speed; the dashed line is the threshold;
   markers under the axis are the bursts, green if they became a
   candidate and amber if they were rejected. */
function PoseTrace({ series, threshold, bursts, durationSec }) {
  const W = 720, H = 90, pad = 4;
  const n = series.length;
  if (!n) return null;
  const max = Math.max(threshold || 0, ...series) || 1;
  const y = (v) => H - pad - (v / max) * (H - 2 * pad);
  const x = (i) => pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
  const dur = durationSec || n;
  const pts = series.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  return (
    <div style={{ marginTop: 6, overflowX: "auto" }}>
      <svg width={W} height={H + 16} style={{ maxWidth: "100%" }}
        role="img" aria-label="wrist speed over the clip">
        <polyline points={pts.join(" ")} fill="none"
          stroke="var(--emerald-700, #1a9d55)" strokeWidth="1" />
        {threshold != null && (
          <line x1={pad} x2={W - pad} y1={y(threshold)} y2={y(threshold)}
            stroke="#b7791f" strokeDasharray="4 3" strokeWidth="1" />
        )}
        {bursts.map((b, i) => {
          const px = pad + (Math.min(1, (b.t || 0) / Math.max(1e-6, dur)))
            * (W - 2 * pad);
          return (
            <g key={i}>
              <line x1={px} x2={px} y1={pad} y2={H - pad} strokeWidth="1"
                stroke={b.status === "swing" ? "#1a9d55" : "#b7791f"}
                strokeOpacity="0.45" />
              <circle cx={px} cy={H + 6} r="3"
                fill={b.status === "swing" ? "#1a9d55" : "#b7791f"} />
            </g>
          );
        })}
      </svg>
      <div className="muted">
        wrist speed across the clip · dashed = burst threshold · dots =
        bursts (green kept, amber rejected)
      </div>
    </div>
  );
}

function ProduceStatusOverlay({ row, greyed, override }) {
  const queued = !override && row.queue_state === "queued";
  if (!greyed && !queued) return null;

  const stage = row.produce_stage;
  const total = row.produce_total || 0;
  const done = row.produce_done || 0;
  // Only a per-candidate stage carries a meaningful total; the one-off
  // stages report 0 and get an indeterminate bar rather than a fake 0%.
  // An override ("Deleting…") is not a produce run and has no progress.
  const pct =
    !override && total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null;

  // The card is greyed for something OTHER than a produce — say the
  // delete — and has to name it. The produce stages are stale in that
  // case: they describe the run that made the clip, not what is
  // happening now.
  // A ROW CLAIMING TO PRODUCE WITH NOTHING BEHIND IT. The server can
  // tell -- it knows what is actually running in it -- and says so, so
  // the card can stop looking busy and say what to do instead. Half an
  // hour of "Producing…" is the same picture whether the run is slow or
  // dead, and only one of those is worth waiting out.
  const label = override ? override : row.produce_stalled
    ? "Stalled — nothing is producing this. Press Produce to start again."
    : queued
      ? `Waiting to produce${
          row.queue_position ? ` · ${row.queue_position} of ${row.queue_depth}` : ""
        }`
      : stage && stage !== "done" && stage !== "failed"
        ? stage
        : "Producing…";

  return (
    <div
      style={{
        position: "absolute",
        top: 10,
        left: 12,
        right: 12,
        zIndex: 5,
        pointerEvents: "none",
        display: "flex",
        justifyContent: "center",
      }}
    >
      <div
        style={{
          display: "flex",
          alignItems: "center",
          gap: 10,
          maxWidth: "100%",
          padding: "7px 14px",
          borderRadius: 999,
          background: queued
            ? "rgba(234, 179, 8, 0.95)"
            : "rgba(17, 24, 39, 0.92)",
          color: queued ? "#3f2d00" : "#f9fafb",
          border: "1px solid rgba(255,255,255,0.18)",
          boxShadow: "0 2px 10px rgba(0,0,0,0.28)",
          fontSize: 13,
          fontWeight: 600,
        }}
      >
        {!queued && (
          <span
            aria-hidden="true"
            style={{
              width: 12,
              height: 12,
              flexShrink: 0,
              borderRadius: "50%",
              border: "2px solid rgba(255,255,255,0.35)",
              borderTopColor: "#fff",
              animation: "gr-spin 0.8s linear infinite",
            }}
          />
        )}
        <span
          style={{
            overflow: "hidden",
            textOverflow: "ellipsis",
            whiteSpace: "nowrap",
          }}
        >
          {label}
        </span>
        {pct !== null && (
          <span
            style={{
              flexShrink: 0,
              width: 54,
              height: 5,
              borderRadius: 999,
              background: "rgba(255,255,255,0.25)",
              overflow: "hidden",
            }}
          >
            <span
              style={{
                display: "block",
                width: `${pct}%`,
                height: "100%",
                background: "#22c55e",
                transition: "width 0.4s ease",
              }}
            />
          </span>
        )}
      </div>
    </div>
  );
}

/* A centred "are you sure?" — window.confirm pins its box to the top of
   the browser chrome, which on a wide operator screen is nowhere near
   what the operator just clicked. This sits in the middle of the
   viewport, and, unlike window.confirm, it can be dismissed the instant
   the operator says yes so the greyed card underneath is visible while
   the work runs. */
function ConfirmDialog({ open, title, body, confirmLabel, onConfirm, onCancel }) {
  useEffect(() => {
    if (!open) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") onCancel?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onCancel]);
  if (!open) return null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onCancel}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 120,
        background: "rgba(15, 23, 42, 0.55)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 16,
      }}
    >
      <div
        className="card"
        onClick={(e) => e.stopPropagation()}
        style={{
          maxWidth: 460,
          width: "100%",
          margin: 0,
          textAlign: "center",
          boxShadow: "0 12px 40px rgba(0,0,0,0.35)",
        }}
      >
        <h4 style={{ marginTop: 0 }}>{title}</h4>
        {body && (
          <p className="muted" style={{ fontSize: 14, marginBottom: 18 }}>
            {body}
          </p>
        )}
        <div style={{ display: "flex", gap: 10, justifyContent: "center" }}>
          <button
            type="button"
            className="ghost"
            style={{ width: "auto", minWidth: 110 }}
            onClick={onCancel}
          >
            Cancel
          </button>
          <button
            type="button"
            className="danger"
            style={{ width: "auto", minWidth: 110 }}
            autoFocus
            onClick={onConfirm}
          >
            {confirmLabel || "Delete"}
          </button>
        </div>
      </div>
    </div>
  );
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

/* Stamp a run's completion into a clip's URLs so the browser cannot
   serve the previous run's video or thumbnail from cache. Returns the
   clips untouched when there is nothing to stamp — the ids and every
   other field are preserved, since the card matches clips by id. */
function bustUrl(url, v) {
  if (!url || v == null) return url;
  return `${url}${url.includes("?") ? "&" : "?"}v=${v}`;
}

function bustClips(clips, completedAt) {
  const v = completedAt ? Date.parse(completedAt) : null;
  if (!clips || !Number.isFinite(v)) return clips;
  return clips.map((c) => ({
    ...c,
    video_url: bustUrl(c.video_url, v),
    thumbnail_url: bustUrl(c.thumbnail_url, v),
  }));
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
                     videoUrl, recordingStartedAt, onOpenViewer, footer }) {
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
        onClick={videoUrl
          ? () => onOpenViewer(videoUrl, label,
                               recordingStartedAt || startsAt, fps,
                               // TIME OF DAY ON EVERY RAW VIDEO, not just
                               // the ones whose Pi reported a first-frame
                               // stamp. Where it did not, the clip still
                               // has a start -- the trigger, or the
                               // upload's base capture time -- which is
                               // right to a fraction of a second and far
                               // more use than no clock at all. It is
                               // marked approximate so it is never
                               // mistaken for the measured one the
                               // camera sync is reckoned from.
                               !recordingStartedAt && !!startsAt)
          : undefined}
      />
      {/* DIRECTLY UNDER THE PICTURE, above the meta rows. The control
          acts on what is in the frame above it; six rows of file
          statistics in between made it read as a footnote to the
          statistics instead. */}
      {footer && <div style={{ margin: "6px 0" }}>{footer}</div>}
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

function ProducedTile({ clips, swings, onOpenViewer,
                        onDeleteClip, onEditClip, onAddClip }) {
  // Right-most tile on the Production card: thumbnail + summary of every
  // produced clip cut from this upload. With multiple swings, toggle
  // through each produced clip (◀/▶); the thumbnail + play follow the
  // selection. Falls back to a "Not produced" placeholder when empty.
  const has = clips && clips.length > 0;
  const [sel, setSel] = useState(0);
  const idx = has ? Math.min(sel, clips.length - 1) : 0;
  const cur = has ? clips[idx] : null;
  // MOG2 layer-in evidence for the selected clip: produce persists a
  // per-swing overlay (raw motion heat + AI picks + MOG2 chain/added
  // points) + the timed heat dots into edit_metrics.swings — clip
  // order matches swing order.
  // A swing has something to plot if ANY of these carry points. Gating on
  // the MOG2 overlay and timed points alone hid the button on every clip
  // the debug3 engine produced, because that engine does not run the MOG2
  // layer -- and ball_track_frames is exactly the track the editor draws.
  const hasEvidence = (s) =>
    s?.mog2_overlay_url
    || (s?.timed_points || []).length > 0
    || (s?.ball_track_frames || []).length > 0;
  const withOverlay = (swings || []).filter(hasEvidence);
  // Match the SELECTED CLIP to its swing by the stored clip_id first —
  // positional (idx) matching breaks the moment a clip is deleted (the
  // remaining clips shift position but the swings keep their idx, so
  // clip 1 showed the DELETED swing's click-to-plot). Positional match
  // stays as the fallback for rows produced before clip_id stamping.
  const curSwing =
    (cur &&
      (swings || []).find(
        (s) => s?.clip_id != null && s.clip_id === cur.id && hasEvidence(s),
      )) ||
    (withOverlay.length === 1 && (clips || []).length <= 1
      ? withOverlay[0]
      : (swings || []).find((s) => s?.idx === idx && hasEvidence(s)));
  const aces = has ? clips.filter((c) => c.ball_in_cup).length : 0;
  const holes = has
    ? clips.map((c) => c.hole_number).filter((h, i, a) => h != null && a.indexOf(h) === i)
    : [];
  const play = (c) =>
    c?.video_url &&
    onOpenViewer(
      c.video_url,
      c.hole_number != null ? `Produced — hole ${c.hole_number}` : "Produced Video",
    );
  const nav = (delta) =>
    setSel((s) => {
      const n = clips.length;
      return ((Math.min(s, n - 1) + delta) % n + n) % n;
    });
  return (
    <div style={{ flex: "1 1 0", minWidth: 200, maxWidth: 340 }}>
      <div className="tiny upper muted" style={{ marginBottom: 4 }}>
        Produced Video
      </div>
      <Thumb
        src={cur?.thumbnail_url}
        alt="Produced clip thumbnail"
        placeholder={has ? "No preview" : "Not produced"}
        onClick={cur?.video_url ? () => play(cur) : undefined}
      />
      {has && clips.length > 1 && (
        <div
          style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: 8, marginTop: 4,
          }}
        >
          <button
            type="button"
            className="ghost"
            style={{ width: "auto", padding: "1px 8px", fontSize: "0.9rem" }}
            onClick={() => nav(-1)}
            title="Previous clip"
          >
            ◀
          </button>
          <span className="tiny">
            clip {idx + 1}/{clips.length}
            {cur?.hole_number != null ? ` · hole ${cur.hole_number}` : ""}
          </span>
          <button
            type="button"
            className="ghost"
            style={{ width: "auto", padding: "1px 8px", fontSize: "0.9rem" }}
            onClick={() => nav(1)}
            title="Next clip"
          >
            ▶
          </button>
        </div>
      )}
      {has && clips.length === 1 && cur?.hole_number != null && (
        <div className="tiny" style={{ textAlign: "center", marginTop: 4 }}>
          hole {cur.hole_number}
        </div>
      )}
      {/* ONE BAND, THREE BUTTONS, ALL THE SAME SHAPE. Delete used to be
          a bare 🗑 wedged into the clip pager, where it was both a
          different size from everything else and one slip away from the
          ▶ that only changes which clip is on screen. It is an action on
          the selected clip, exactly like Edit -- so it sits with Edit,
          and only its colour says it is the dangerous one. */}
      {(onEditClip || onAddClip || onDeleteClip) && (
        <div style={{ display: "flex", gap: 6, marginTop: 6 }}>
          {onEditClip && (
            <button
              type="button"
              className="small ghost"
              style={{ flex: 1, width: "auto" }}
              onClick={() => onEditClip(cur, curSwing, idx)}
              disabled={!has}
              title={has
                ? `Edit clip ${idx + 1}${
                    cur?.hole_number != null ? ` (hole ${cur.hole_number})` : ""
                  } — opens the plot map: ball, impact frame, landing, flag and both tracers for THIS clip.`
                : "Nothing produced yet to edit"}
            >
              ✎ Edit
            </button>
          )}
          {onAddClip && (
            <button
              type="button"
              className="small ghost"
              style={{ flex: 1, width: "auto" }}
              onClick={() => onAddClip()}
              title="Add a clip the detector missed — opens the edit wizard on a new, blank swing spanning the upload, ready for its start, impact and end frames."
            >
              ＋ Add
            </button>
          )}
          {onDeleteClip && (
            <button
              type="button"
              className="small ghost"
              style={{ flex: 1, width: "auto", color: "var(--danger)",
                       borderColor: "var(--danger)" }}
              onClick={() => cur && onDeleteClip(cur, idx)}
              disabled={!has}
              title={has
                ? `Delete clip ${idx + 1}${
                    cur?.hole_number != null ? ` · hole ${cur.hole_number}` : ""
                  }, its files and the swing it was cut from. The raw upload and the other clips stay.`
                : "Nothing produced yet to delete"}
            >
              🗑 Delete
            </button>
          )}
        </div>
      )}
      {/* THE CLICK-TO-PLOT BUTTON IS GONE, because Edit is it. The
          two dialogs were merged onto the plot map, so a second button
          opening the same screen for the same clip was one button too
          many. */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 2 }}>
        <MetaRow k="Clips" v={has ? clips.length : ""} />
        <MetaRow k="Aces" v={has ? aces : ""} />
        <MetaRow k="Holes" v={holes.length ? holes.join(", ") : ""} />
      </div>
    </div>
  );
}

/**
 * The five things a swing IS, down the right of the plot map.
 *
 * This is the edit wizard's field list, moved onto the screen where the
 * pictures are. The two used to be separate: the wizard listed the
 * numbers and showed one still, click-to-plot showed the motion and
 * hid the numbers in a toolbar. Every real edit needed both, so every
 * real edit meant opening one, closing it and opening the other.
 *
 * Each row says what it is, what it is set to, and how to set it. A row
 * with nothing in it says "Not set" rather than a zero, because a zero
 * is a value and the difference decides whether produce believes it.
 */
function PlotField({ label, value, hint, children, accent, icon }) {
  return (
    <div style={{
      border: "1px solid var(--line)", borderRadius: 8, padding: "6px 9px",
      background: "var(--card, rgba(255,255,255,0.02))",
    }}>
      {/* LABEL AND VALUE ON ONE LINE. Stacked, five fields were taller
          than the panel and the last of them needed a scroll to reach —
          on a screen whose whole point is that everything about the
          swing is in front of you. */}
      <div className="row" style={{ justifyContent: "space-between",
                                    alignItems: "baseline", gap: 8 }}>
        <span className="tiny upper muted" style={{ whiteSpace: "nowrap" }}>
          {icon ? `${icon} ` : ""}{label}
        </span>
        <span style={{ fontWeight: 700, fontSize: "0.92rem",
                       color: value ? (accent || "var(--ink)")
                                    : "var(--muted)" }}>
          {value || "not set"}
        </span>
      </div>
      {hint && <div className="tiny muted" style={{ marginTop: 1 }}>{hint}</div>}
      {children && <div style={{ marginTop: 4 }}>{children}</div>}
    </div>
  );
}

// Frames of clickable detections either side of impact. A few frames of
// lead-in covers an impact frame estimated slightly late (the assumed-
// impact path pins it to the pose peak, which can sit a frame or two off
// the strike).
//
// FORTY AFTER, NOT A HUNDRED, and the number comes from what MOG2
// actually produces rather than from how long a ball is in the air. The
// detector realistically yields no more than about forty usable points
// past the strike; everything beyond that is the golfer walking off, a
// cart, or wind in the trees. At a hundred, sixty frames of that noise
// were on the map -- every one of them a dot that can only ever be a
// wrong pick, and all of them competing for the cursor with the handful
// of real ones.
const PLOT_WINDOW_PRE = 5;
const PLOT_WINDOW_POST = 40;

/**
 * BOTH RAW CAMERAS UNDER ONE TRANSPORT, on the wall clock.
 *
 * The two Pis start recording at different moments, so "frame 900" means
 * nothing across them and scrubbing the two players independently is a
 * guessing game. The only thing they share is the time of day, so that is
 * what drives them: `deltaSec` is green_start − tee_start, which makes
 * the green video's position `teeTime − deltaSec` at every instant. One
 * play button moves both; a rAF loop hauls green back whenever it drifts
 * more than a frame's worth away.
 *
 * Rewind is a timer rather than a negative playbackRate, which no browser
 * implements — stepping currentTime backwards on an interval is the only
 * way to actually see the footage run backwards.
 */
function RawSyncPlayer({
  teeUrl, greenUrl, teeFps, greenFps, deltaSec, teeStartedAt,
  greenStartedAt, deltaSource, baseCapturedAt,
  // WHERE BOTH CAMERAS ARE STOPPED, reported up so the impact and
  // landing frames can be set from here. Only ever called with a pair
  // while paused, and with null the rest of the time: a moving picture
  // has no frame worth committing to, and reporting one 30 times a
  // second would re-render the whole modal for the length of the video.
  onFrames,
}) {
  const teeRef = useRef(null);
  const greenRef = useRef(null);
  const [playing, setPlaying] = useState(false);
  const [rate, setRate] = useState(1);      // 1 or 3, forward only
  const [rewinding, setRewinding] = useState(false);
  const [teeTime, setTeeTime] = useState(0);
  // THE GREEN PLAYER'S OWN POSITION, read back rather than derived. The
  // whole claim this view makes is that the two cameras are showing the
  // same instant, and a clock computed from the tee's position cannot
  // test that claim -- it would read correct while the green picture
  // was seconds adrift. Read where green ACTUALLY is, turn it into a
  // time of day with green's OWN start stamp, and the two clocks agree
  // only when the pictures do.
  const [greenTimeRaw, setGreenTimeRaw] = useState(0);
  const [dur, setDur] = useState(0);
  const tfps = Number(teeFps) || 30;
  const gfps = Number(greenFps) || tfps;
  // WHEN EACH CAMERA STARTED ROLLING, as a time of day. This is the only
  // thing the two clips share -- frame 900 means nothing across them --
  // so it is what the sync is built on and what the readout shows.
  const teeStamp = parseApiDate(teeStartedAt)?.getTime() ?? null;
  const greenStamp = parseApiDate(greenStartedAt)?.getTime() ?? null;
  const bothStamped = teeStamp != null && greenStamp != null;
  // The offset IS the difference between the two start times, whenever
  // both cameras stamped one. Computed here rather than only passed in,
  // so the number driving the pictures and the number on screen are the
  // same arithmetic rather than two things that ought to agree.
  const delta = bothStamped
    ? (greenStamp - teeStamp) / 1000
    : (Number(deltaSec) || 0);
  // THERE IS ALWAYS A TIME OF DAY, because there is always something
  // that says when this capture happened -- if not each camera's own
  // start, then the upload's. What changes is how much the two clocks
  // are worth: on real stamps they are measured independently and can
  // disagree, which is what makes them evidence. Anchored on the
  // upload's capture time they are locked together by the offset and
  // agree by construction, so they tell you the time and nothing about
  // the sync. The footer says which of those you are looking at.
  const baseEpoch = parseApiDate(baseCapturedAt)?.getTime() ?? null;
  const teeEpoch = teeStamp ?? baseEpoch;
  const greenEpoch = greenStamp
    ?? (teeEpoch != null ? teeEpoch + delta * 1000 : null);
  const haveClock = teeEpoch != null && greenEpoch != null;

  // GREEN FOLLOWS TEE, always. Two <video>s decode independently and
  // wander apart within a few seconds even when started together, so the
  // offset is re-asserted continuously rather than once at play. A third
  // of a frame is the tolerance: tighter and the correction itself
  // becomes visible as a stutter.
  useEffect(() => {
    let raf = 0;
    const tick = () => {
      const t = teeRef.current;
      const g = greenRef.current;
      if (t) {
        setTeeTime(t.currentTime);
        if (g) setGreenTimeRaw(g.currentTime);
        if (g && g.readyState >= 2) {
          const want = t.currentTime - delta;
          const clamped = Math.max(0, Math.min(g.duration || 0, want));
          if (Math.abs(g.currentTime - clamped) > (1 / gfps) / 3) {
            g.currentTime = clamped;
          }
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [delta, gfps]);

  // REWIND, hand-cranked. Held at 3x real time by stepping the tee clock
  // back a fixed slice every interval; green is dragged along by the sync
  // loop above, so there is nothing camera-specific here.
  useEffect(() => {
    if (!rewinding) return undefined;
    const step = 0.1;
    const id = setInterval(() => {
      const t = teeRef.current;
      if (!t) return;
      const next = t.currentTime - step * 3;
      if (next <= 0) {
        t.currentTime = 0;
        setRewinding(false);
      } else {
        t.currentTime = next;
      }
    }, step * 1000);
    return () => clearInterval(id);
  }, [rewinding]);

  function stopRewind() {
    setRewinding(false);
  }

  async function play(r = 1) {
    stopRewind();
    const t = teeRef.current;
    const g = greenRef.current;
    if (!t) return;
    t.playbackRate = r;
    setRate(r);
    // GREEN IS MUTED AND PLAYED TOO, so its decoder keeps running and the
    // sync loop only has to nudge. Seeking a paused video every frame
    // instead is what made this judder.
    if (g) {
      g.playbackRate = r;
      g.currentTime = Math.max(0, Math.min(g.duration || 0,
                                           t.currentTime - delta));
      g.play().catch(() => {});
    }
    try { await t.play(); setPlaying(true); } catch { /* autoplay block */ }
  }

  function pause() {
    stopRewind();
    teeRef.current?.pause();
    greenRef.current?.pause();
    setPlaying(false);
  }

  // ONE FRAME, on the TEE's frame rate — the tee camera is the clock
  // everything else is expressed against, so a "frame" here is one of
  // its frames even when the green camera runs at a different rate.
  function stepFrames(n) {
    pause();
    const t = teeRef.current;
    if (!t) return;
    t.currentTime = Math.max(
      0, Math.min(t.duration || 0, t.currentTime + n / tfps));
  }

  const teeFrame = Math.round(teeTime * tfps);
  // Where green SHOULD be (for "has it started yet") and where it IS
  // (for the frame number and the clock).
  const greenTime = teeTime - delta;
  const greenFrame = Math.round(greenTimeRaw * gfps);
  const teeWall = teeEpoch == null ? null : teeEpoch + teeTime * 1000;
  const greenWall = greenEpoch == null
    ? null : greenEpoch + greenTimeRaw * 1000;
  // The clocks are computed independently; a gap between them is real
  // drift, so say so rather than letting it pass as a rounding wobble.
  const skewMs = (teeWall != null && greenWall != null && greenUrl
                  && greenTime >= 0)
    ? Math.abs(teeWall - greenWall) : null;
  const paused = !playing && !rewinding;
  const btn = { width: "auto", padding: "2px 9px" };

  // `null` while it runs, the pair when it stops. Passing the same null
  // repeatedly is free -- React bails out on an unchanged state value --
  // so this can sit on deps that tick with the video without costing a
  // render per frame.
  useEffect(() => {
    onFrames?.(paused
      ? { teeFrame, greenFrame: greenUrl && greenTime >= 0 ? greenFrame : null }
      : null);
  }, [paused, teeFrame, greenFrame, greenTime, greenUrl, onFrames]);

  return (
    <div style={{
      flex: 1, minWidth: 0, minHeight: 0, display: "flex",
      flexDirection: "column", gap: 8,
    }}>
      <div style={{
        flex: 1, minHeight: 0, display: "flex", gap: 8,
        alignItems: "stretch",
      }}>
        <div style={{ flex: 1, minWidth: 0, display: "flex",
                      flexDirection: "column", gap: 3 }}>
          <video
            ref={teeRef}
            src={teeUrl}
            muted
            playsInline
            preload="auto"
            onLoadedMetadata={(e) => setDur(e.currentTarget.duration || 0)}
            onEnded={() => setPlaying(false)}
            style={{ width: "100%", flex: 1, minHeight: 0,
                     objectFit: "contain", background: "#000",
                     borderRadius: 6 }}
          />
          <div className="small" style={{ textAlign: "center" }}>
            <b style={{ fontVariantNumeric: "tabular-nums" }}>
              {haveClock ? fmtWallClock(teeWall) : `${teeTime.toFixed(2)}s`}
            </b>
            <span className="muted"> · tee f{teeFrame}</span>
          </div>
        </div>
        {greenUrl && (
          <div style={{ flex: 1, minWidth: 0, display: "flex",
                        flexDirection: "column", gap: 3 }}>
            <video
              ref={greenRef}
              src={greenUrl}
              muted
              playsInline
              preload="auto"
              style={{ width: "100%", flex: 1, minHeight: 0,
                       objectFit: "contain", background: "#000",
                       borderRadius: 6 }}
            />
            <div className="small" style={{ textAlign: "center" }}>
              {greenTime < 0 ? (
                <span className="muted">
                  green was not recording yet at this instant
                </span>
              ) : (
                <>
                  <b style={{ fontVariantNumeric: "tabular-nums" }}>
                    {haveClock
                      ? fmtWallClock(greenWall)
                      : `${greenTimeRaw.toFixed(2)}s`}
                  </b>
                  <span className="muted"> · green f{greenFrame}</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 6, alignItems: "center",
                    justifyContent: "center", flexWrap: "wrap" }}>
        <button type="button" className="ghost small" style={btn}
                onClick={() => stepFrames(-1)} title="Back one tee frame">
          ◀|
        </button>
        <button type="button"
                className={rewinding ? "small" : "ghost small"} style={btn}
                onClick={() => {
                  pause();
                  setRewinding(true);
                }}
                title="Rewind at 3x">
          ◀◀ 3x
        </button>
        {/* ONE BUTTON. Play and pause are the two states of a single
            control, and a pair of them means the operator has to read
            which one is live before pressing either. */}
        <button type="button"
                className={playing && rate === 1 ? "small" : "ghost small"}
                style={btn}
                onClick={() => (paused ? play(1) : pause())}
                title={paused ? "Play both" : "Pause both"}>
          {paused ? "▶" : "❙❙"}
        </button>
        {/* PRESS AGAIN TO GO FASTER. 2x, then 4x, then 6x, then back
            to 2x -- one button that steps through the speeds rather
            than one speed you can only take or leave. */}
        <button type="button"
                className={playing && rate > 1 ? "small" : "ghost small"}
                style={btn}
                onClick={() => play(
                  !playing || rate < 2 ? 2 : (rate >= 6 ? 2 : rate + 2))}
                title="Fast forward — press again for the next speed up">
          ▶▶ {playing && rate > 1 ? rate : 2}x
        </button>
        <button type="button" className="ghost small" style={btn}
                onClick={() => stepFrames(1)} title="Forward one tee frame">
          |▶
        </button>
        <input
          type="range"
          min={0}
          max={Math.max(0.01, dur)}
          step={1 / tfps}
          value={Math.min(teeTime, dur || 0)}
          onChange={(e) => {
            pause();
            const t = teeRef.current;
            if (t) t.currentTime = Number(e.target.value);
          }}
          style={{ flex: 1, minWidth: 140, maxWidth: 420 }}
        />
      </div>
      <div className="small muted" style={{ textAlign: "center" }}>
        {/* THE OFFSET, AND HOW MUCH IT IS WORTH. "Nobody knows" and
            "they started together" are both zero, and an operator
            looking at two videos that will not line up needs to be able
            to tell which one they are looking at. */}
        {bothStamped ? (
          <>
            tee started {fmtWallClock(teeEpoch)} · green started{" "}
            {fmtWallClock(greenEpoch)} · offset{" "}
            {delta >= 0 ? "+" : ""}{delta.toFixed(3)}s, measured from the
            cameras&rsquo; own start stamps
          </>
        ) : haveClock ? (
          <>
            clock from the upload&rsquo;s capture time{" "}
            {fmtWallClock(teeEpoch)} — neither camera stamped its own
            start, so the green clip is placed by an offset of{" "}
            {delta >= 0 ? "+" : ""}{delta.toFixed(3)}s
            {deltaSource === "saved"
              ? " that was saved for this upload"
              : " nobody has established"}
            . The two clocks cannot disagree, so they say when, not
            whether the pictures line up.
          </>
        ) : (
          <>
            nothing on this upload says when either camera started, so
            there is no time of day to show — the numbers above are
            seconds into each file, and the clips are lined up on an
            offset of {delta >= 0 ? "+" : ""}{delta.toFixed(3)}s
          </>
        )}
        {bothStamped && skewMs != null && skewMs > 120 && (
          <div style={{ color: "#fbbf24" }}>
            the two clocks are {Math.round(skewMs)}ms apart — the pictures
            are drifting, not showing the same instant
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Standalone click-to-plot modal, opened from a production card's
 * 🖱 Click-to-plot button. Big zoomable heat view with every timed dot
 * clickable; Save & close bakes the queued picks into the swing's
 * tracer (cv2 fast render, no AI), re-finalizes the video with the
 * saved graphics, and commits it to Produced Clips — the same
 * pipeline as the wizard's Produce, minus the wizard.
 */
function ClickToPlotModal({
  row, swingPos, addNew, adminPassword, onClose, onBackground, onDone,
}) {
  const swings = row.edit_metrics?.swings || [];
  // ADDING A SWING IS THE SAME SCREEN WITH EMPTY FIELDS. A swing the
  // detector missed needs exactly what this map already collects -- tee
  // spot, impact frame, flight points, landing -- so a new swing reads
  // every field off an empty object. The only things that differ are at
  // save: the swing is APPENDED rather than patched, and it needs a
  // frame window, which is derived from the impact frame the operator
  // set.
  //
  // ADD IS A STATED INTENT, NOT A POSITION PAST THE END. It used to be
  // the latter -- open at `swings.length` and call anything at or past
  // that new -- and the length was read when the button was clicked
  // while the check ran when the modal rendered. Add a swing, save it,
  // press Add again before the row has come back from the server: the
  // position is still N, the array now HAS an Nth swing, and the second
  // Add silently opened the first one. Everything on it came along --
  // its tee points, its comet, its landing -- and the two sets of
  // points plotted one tracer through both, which is what "the tracer
  // goes wacky" looked like. A flag cannot go stale between the click
  // and the render.
  const isNew = !!addNew || swingPos == null || swingPos >= swings.length;
  const swing = isNew ? {} : (swings[swingPos] || {});
  // FLIGHT WINDOW. Pre-swing motion (waggle, address, shadow) is noise on
  // this map, and so is everything long after the ball has gone — the
  // golfer walking off, a cart, wind in the trees. Both crowd the map with
  // dots that can only ever be wrong picks. Show impact-5 (a few frames of
  // lead-in, in case impact is estimated a touch late) through impact+40,
  // which is about as many points as MOG2 realistically yields past the
  // strike -- so impact f500 shows f495-f540 and nothing after.
  // IMPACT FRAME, editable here. It drives the flight window below AND
  // where the rendered tracer line starts, so when it is wrong (pinned to
  // a waggle rather than the strike) the map hides the real flight and the
  // line is drawn across frames nothing was detected in. Fixing it in the
  // wizard meant leaving this screen and losing the plot in progress.
  const [impactFrame, setImpactFrame] = useState(swing.impact_frame ?? null);
  const impactF = impactFrame;
  // BALL AT IMPACT = the tracer's starting point. The renderer anchors
  // the fitted curve on it, so when it is wrong the line begins in the
  // wrong place no matter how good the flight points are. Editable here
  // because this is the screen where you can actually SEE where the ball
  // was, against the motion heat.
  const [ballAtRest, setBallAtRest] = useState(swing.ball ?? null);
  // One tee frame as a picture, for the map's frame stepper. Stable
  // across renders so the canvas's fetch effect is not re-run every time
  // a dot is clicked.
  const loadTeeFrame = useCallback(
    async (f) => {
      const r = await api.getLongUploadFrame(adminPassword, row.id, f, "tee");
      return r?.image_url;
    },
    [adminPassword, row.id],
  );
  const winLo = impactF == null ? null : impactF - PLOT_WINDOW_PRE;
  const winHi = impactF == null ? null : impactF + PLOT_WINDOW_POST;
  const inWindow = (arr) =>
    impactF == null
      ? arr
      : arr.filter((p) => p.frame >= winLo && p.frame <= winHi);
  const dots = inWindow(swing.timed_points || []);
  const denseDots = inWindow(swing.cand_points || []);
  // Dots that are ALREADY in the swing's saved ball track (from an
  // earlier Save here, or from the tracer itself) start out green —
  // so reopening the modal shows what's plotted, and Save only sends
  // the diff (new clicks as manual points, un-clicks as cleared).
  const [baked] = useState(() => {
    // EVERY point in the saved ball track, not just the ones that happen
    // to sit on a detection dot. Production adds points from the AI launch
    // plot, the launch tracker and arc completion, none of which coincide
    // with a clickable dot — so intersecting the two sets left those
    // points invisible to this editor and impossible to remove. They are
    // exactly the ones worth removing when the tracer goes wrong.
    const init = {};
    for (const r of swing.ball_track_frames || []) {
      if (r.found && r.x != null && r.y != null && init[r.frame] === undefined) {
        init[r.frame] = { x: r.x, y: r.y };
      }
    }
    return init;
  });
  const [marks, setMarks] = useState(() => ({ ...baked }));
  // No busy state: Save & close hands the run to the production card
  // and closes, so there is never a moment where this modal is waiting.
  // THE PIXEL SPACE THE DOTS ARE IN — this swing's own, when produce
  // recorded it. Every dot is placed by `p.x / frameW`, so frameW has to
  // be the width the points were MEASURED at, not merely the width of
  // the tee source. Those were assumed identical and are not: the
  // pipeline measures on the cut segment, and a cut that has been
  // re-encoded through compress_for_email is capped at 1280 on the long
  // edge, which parks every dot at 1280/1920 of its true x — the whole
  // plot shifted left, worse the further right the point. Produce now
  // scales detections back to native and stamps the space it used;
  // prefer that stamp, and fall back to the upload-level width for rows
  // persisted before it existed.
  const frameW =
    swing.track_frame_width
    ?? row.edit_metrics?.frame_width ?? row.tee_width ?? null;
  const frameH =
    swing.track_frame_height
    ?? row.edit_metrics?.frame_height ?? row.tee_height ?? null;
  const bgUrl = swing.tracer_raw_motion_url || swing.mog2_overlay_url;
  // Whether the tee picture can be stepped at all. On a swing with no
  // produce behind it this is the ONLY thing that can be shown, so it
  // is what decides whether the map renders rather than the presence of
  // dots -- which a new swing has none of, by definition.
  const canStepTee = !!row.tee_nb_frames;

  // WHICH VIEW IS UP: the tee's motion heat, the green camera, or the
  // tracer's own line. Declared here because everything below keys off
  // it — the frame that loads, the dots that are shown, what a click
  // means.
  const [cam, setCam] = useState("tee");

  // ── THE TRACER ITSELF ─────────────────────────────────────────────
  // A third view of the same swing: not the dots, but the LINE they
  // produce. Two things about it are judgement rather than
  // measurement — where the ball finished in this picture, and how
  // high the arc goes — and both were previously only adjustable by
  // producing a clip and looking at it. Here they are two handles.
  //
  // The curve drawn is the renderer's own: the server is asked for the
  // tail rather than the browser modelling one, because two models
  // would drift and the operator would end up shaping the wrong one.
  const [shape, setShape] = useState(null);   // {tail, kind, track}
  const [shapeErr, setShapeErr] = useState(null);
  const [tracerEnd, setTracerEnd] = useState(() => {
    const e = swing.tracer_end
      || swing.tracer_tail?.target
      || (swing.target ? [swing.target.x, swing.target.y] : null);
    return e ? { x: Math.round(e[0]), y: Math.round(e[1]) } : null;
  });
  const [shapeBase] = useState(
    () => JSON.stringify(swing.tracer_end || null));
  // A PREDICTED AIM POINT IS NOT AN EDIT. Opening the tab on a swing
  // with no landing marked fills one in from the flight so there is an
  // arc to look at and a handle to take hold of — but until the
  // operator actually moves it, nothing has been decided, and Save must
  // not light up as though it had.
  const [endGuessed, setEndGuessed] = useState(false);
  // HOW LONG THE BALL IS IN THE AIR, which is what paces the drawn
  // continuation. Measured off the two cameras' clocks when the landing
  // was marked on the green; otherwise the hole's yardage through the
  // carry-time table, and editable either way — a 150-yard hole played
  // into the wind is not a 150-yard shot.
  const [flightSec, setFlightSec] = useState(
    () => (swing.tracer_flight_sec != null
      ? Number(swing.tracer_flight_sec) : null));
  const [flightBase] = useState(
    () => (swing.tracer_flight_sec != null
      ? Number(swing.tracer_flight_sec) : null));
  const shapeChanged =
    (!endGuessed
     && JSON.stringify(tracerEnd ? [tracerEnd.x, tracerEnd.y] : null)
       !== shapeBase)
    || flightSec !== flightBase;
  // ONE HANDLE. The curve is the tracked flight's own parabola carried
  // on to where the ball finished, so the landing is the only thing
  // there is to say about it: the direction it leaves at, how fast and
  // how hard it falls were all measured off the MOG2 points and are
  // not the operator's to invent. Dragging the end re-solves the whole
  // arc, which is what the shot-tracer apps do.
  const tracerPath = shape?.tail || [];

  async function fetchShape(end, force = false) {
    const at = end || tracerEnd;
    try {
      setShapeErr(null);
      const out = await api.tracerShape(adminPassword, row.id, {
        // The tail is anchored on the tracked ball, so it is sent
        // rather than re-detected: this is the same track the map is
        // showing and the same one Save will render from.
        track_frames: (swing.ball_track_frames || []).filter(
          (r) => r.found && r.x != null),
        ball: ballAtRest || swing.ball || null,
        impact_frame: impactFrame ?? swing.impact_frame ?? null,
        // No aim point yet: the server guesses one off the flight and
        // hands it back, so the tab opens on an arc rather than on an
        // instruction to go and mark something somewhere else.
        end: at ? [at.x, at.y] : null,
        flight_sec: flightSec,
        land_frame: swing.tracer_tail?.land_frame ?? null,
        width: frameW, height: frameH,
        fps: row.tee_fps || null,
      });
      setShape(out);
      if (flightSec == null && out?.flight_sec != null) {
        setFlightSec(out.flight_sec);
      }
      if (!at && out?.predicted) {
        setTracerEnd({ x: Math.round(out.predicted[0]),
                       y: Math.round(out.predicted[1]) });
        setEndGuessed(true);
      }
      if (!out?.tail?.length && !force) {
        setShapeErr(out?.reason || "the model could not draw a tail here");
      }
    } catch (e) {
      setShapeErr(e?.message || String(e));
    }
  }

  // Asked for on entering the tab, and again whenever the aim moves —
  // the tail is re-solved for the new end, which is the whole point of
  // dragging it.
  useEffect(() => {
    if (cam !== "tracer") return undefined;
    // DEBOUNCED, because the things that change here change fast: a
    // drag moves the aim point on every pointer event and typing a
    // flight time moves it on every keystroke, and each of those was a
    // round trip. A quarter of a second after the operator stops is
    // still instant to them and is one request instead of forty.
    const id = setTimeout(() => fetchShape(), 250);
    return () => clearTimeout(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cam, tracerEnd?.x, tracerEnd?.y, flightSec, row.id]);

  // ── THE OTHER CAMERA ──────────────────────────────────────────────
  // The tee picture answers "where did the tracer go". The green one
  // answers "where did it come down", and on this screen that is the
  // same gesture: toggle, and the picture becomes the green camera's
  // own frame, the dots become a frame-diff scan of the green window,
  // and a click means "it landed HERE" — the one mark the comet is
  // walked backwards from, so marking the landing and seeing the comet
  // are a single action rather than a trip through the wizard.
  const [green, setGreen] = useState(null); // /frame payload + defaultEnd
  const [greenErr, setGreenErr] = useState(null);
  const [greenDots, setGreenDots] = useState([]);
  const [greenScanning, setGreenScanning] = useState(false);
  const [greenNote, setGreenNote] = useState(null);
  const [greenLevel, setGreenLevel] = useState(2);
  // Whatever landing is already on record FOR THIS SWING — from the
  // wizard, or from a produce.
  //
  // THE UPLOAD-LEVEL ONE IS NOT A FALLBACK on a row with more than one
  // swing. It exists because a single-swing row keeps its answer at the
  // top level, and reading it as a fallback handed swing 2 whatever
  // swing 1 came down on -- which then got saved back to the top level
  // and passed on again. A landing is one ball coming down once: it is
  // set by a produce, set by hand, or not set. It is never inherited.
  const soloRow = !isNew && swings.length <= 1;
  const [savedLanding] = useState(() => {
    const f = swing.landing_frame
      ?? (soloRow ? row.edit_metrics?.landing_frame : null) ?? null;
    const s = swing.landing_spot
      || (soloRow ? row.edit_metrics?.landing_spot : null) || null;
    return f != null && s
      ? { frame: f, x: Math.round(s.x ?? s[0]), y: Math.round(s.y ?? s[1]) }
      : null;
  });
  // THE COMET'S POINTS, PLOTTED THE SAME WAY THE TRACER'S ARE. One
  // landing was not enough: a click marked the landing and the NEXT
  // click moved it, so the map could only ever hold one green dot and
  // nothing accumulated. Here as on the tee side, every click adds a
  // point and clicking it again takes it away — the difference is only
  // that these points are a descent rather than a flight, and that the
  // search can propose them. The last one in time is the landing.
  const [greenMarks, setGreenMarks] = useState(() => {
    const init = {};
    for (const p of swing.green_track || []) {
      init[p.frame] = { x: Math.round(p.x), y: Math.round(p.y) };
    }
    if (!Object.keys(init).length && savedLanding) {
      init[savedLanding.frame] = { x: savedLanding.x, y: savedLanding.y };
    }
    return init;
  });
  const [greenBase] = useState(() => JSON.stringify(
    swing.green_track?.length ? swing.green_track : null));
  // Why the search said no, when it did. The points it FOUND go into
  // the marks rather than being kept apart, so the operator can drop a
  // wrong one or add a missed one instead of taking the chain or
  // leaving it.
  const [cometReason, setCometReason] = useState(null);
  const [cometBusy, setCometBusy] = useState(false);
  // MARK THE LANDING ANYWHERE, not only on a dot the scan happened to
  // find. Clicking dots is the fast path when the descent was detected;
  // when it was not -- a ball landing in shadow, against the trees, or
  // simply missed -- there was no way to say where it came down at all,
  // and the landing is what gives the tracer its aim AND its flight
  // time. Same gesture as the tee side's ball placement.
  // THE FLAG STICK, WHICH BELONGS TO THE HOLE, NOT THE SWING.
  //
  // It is stored once against the hole's tee<->green mapping, so it
  // carries from swing to swing and from clip to clip until somebody
  // moves it. Kept in GREEN pixels because that is the camera that can
  // see the base of the stick; where it sits in the TEE frame is read
  // through the calibration rather than stored, so re-calibrating the
  // hole corrects the flag too instead of leaving a stale coordinate.
  const [pinGreen, setPinGreen] = useState(null);   // {x, y}
  const [pinTee, setPinTee] = useState(null);       // {x, y}, derived
  const [pinNote, setPinNote] = useState(null);
  // ONE-CLICK PLACEMENT, only for what is not there yet. Once a thing
  // exists in the picture it is dragged; arming a mode to move
  // something you can already see is a button standing between the
  // operator and the obvious gesture.
  const [placeOnTee, setPlaceOnTee] = useState(null);    // 'tee' | 'pin'
  const [placeOnGreen, setPlaceOnGreen] = useState(null); // 'landing' | 'pin'
  const [teeViewFrame, setTeeViewFrame] = useState(null);
  // WHERE THE RAW PLAYER IS STOPPED, or null while it is running. The
  // raw tab is the only place both cameras are on screen at once, so it
  // is where the impact and landing frames are easiest to actually
  // FIND -- scrub until the ball leaves, read the tee frame; scrub until
  // it lands, read the green frame.
  const [rawFrames, setRawFrames] = useState(null);
  // THE OFFSET BETWEEN THE TWO CAMERAS, in the same order the produce
  // cut uses it: each Pi's own first-frame stamp, then whatever a
  // previous run established, then an assumption of zero. Worked out
  // here rather than fetched because both stamps are already on the row
  // — and `measured` is carried separately, because "they started
  // together" and "nobody knows" are the same number and must not look
  // the same on screen.
  const rawDelta = useMemo(() => {
    const a_ = row.tee_recording_started_at;
    const b_ = row.green_recording_started_at;
    if (a_ && b_) {
      const d = (new Date(b_).getTime() - new Date(a_).getTime()) / 1000;
      if (Number.isFinite(d)) return { sec: d, source: "stamps" };
    }
    // A SAVED OFFSET IS NOT A CAMERA STAMP. This used to report itself
    // as measured, so an upload with no per-camera stamps at all said
    // "from the cameras' own start stamps" under two clocks it had no
    // way to place in the day. It is a real number and better than
    // nothing -- an operator typed it, or a previous run worked it out
    // -- but it says how far apart the clips are, not when either began.
    const saved = row.edit_metrics?.tee_green_delta_sec;
    if (saved != null && Number.isFinite(Number(saved))) {
      return { sec: Number(saved), source: "saved" };
    }
    return { sec: 0, source: "assumed" };
  }, [row.tee_recording_started_at, row.green_recording_started_at,
      row.edit_metrics?.tee_green_delta_sec]);
  // THE LANDING, IN THE TEE PICTURE. Stored in green pixels because
  // that is the camera that sees it land, but the tee view is where the
  // tracer is drawn and so where "the ball finished THERE" is easiest
  // to say. Dragged here, it is read back through the hole's homography
  // and the green-side landing follows.
  const [teeLandingXY, setTeeLandingXY] = useState(null);
  const [teeLandingNote, setTeeLandingNote] = useState(null);
  // Which green frame the map is showing, so a placed landing carries
  // the instant it was seen at and not just the pixel.
  const [greenViewFrame, setGreenViewFrame] = useState(null);
  // Load the hole's flag once, and put it in the tee frame as well.
  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const vm = await api.getViewMap(adminPassword, row.id);
        // THE PIN FOR THIS SWING, which the server works out from the
        // upload's capture time. `view_map.pin_green` is the newest one
        // on record -- right for a swing shot today, wrong for one shot
        // before the last time the flag moved.
        const g = vm?.pin_green ?? vm?.view_map?.pin_green;
        if (!live || !g) return;
        setPinGreen({ x: Math.round(g[0]), y: Math.round(g[1]) });
        if (vm?.pin_note) setPinNote(vm.pin_note);
      } catch { /* no mapping yet: the flag simply has nowhere to be */ }
    })();
    return () => { live = false; };
  }, [adminPassword, row.id]);

  useEffect(() => {
    if (!pinGreen) { setPinTee(null); return undefined; }
    let live = true;
    (async () => {
      try {
        const out = await api.mapLandingToTee(adminPassword, row.id, {
          green: [pinGreen.x, pinGreen.y],
        });
        if (!live) return;
        setPinTee(out?.tee
          ? { x: Math.round(out.tee[0]), y: Math.round(out.tee[1]) } : null);
      } catch (e) {
        if (live) { setPinTee(null); setPinNote(e?.message || String(e)); }
      }
    })();
    return () => { live = false; };
  }, [pinGreen?.x, pinGreen?.y, adminPassword, row.id]);  // eslint-disable-line react-hooks/exhaustive-deps

  // MOVED ON THE GREEN: that is the flag, and the tee follows it through
  // the calibration. This is the authoritative direction -- the green
  // camera can see the base of the stick.
  async function setPinFromGreen(pt) {
    setPinGreen(pt);
    setPlaceOnGreen(null);
    try {
      const out = await api.saveHolePin(adminPassword, row.id,
                                        { green: [pt.x, pt.y] });
      setPinNote(out?.tee_xy
        ? `flag saved for this hole — ${Math.round(out.tee_xy[0])}, `
          + `${Math.round(out.tee_xy[1])} in the tee frame`
        : (out?.reason || "flag saved for this hole"));
    } catch (e) {
      setPinNote(e?.message || String(e));
    }
  }

  // MOVED ON THE TEE: the flag has not moved, the MAPPING is off. The
  // green camera saw where the stick is; if it lands somewhere else in
  // the tee picture, that gap is the hole's calibration error at the
  // flag, which is the one point on the green anybody can identify in
  // both views. So this reports the correction and does NOT write it
  // back as a new flag position -- a coarse view must not overwrite the
  // precise one, and a calibration that silently re-fits from a drag is
  // one nobody can reason about.
  async function setPinFromTee(pt) {
    setPinTee(pt);
    setPlaceOnTee(null);
    if (!pinGreen) {
      setPinNote("mark the flag on the green camera first — that is the "
        + "view that can see the base of the stick");
      return;
    }
    try {
      const out = await api.mapLandingToTee(adminPassword, row.id, {
        green: [pinGreen.x, pinGreen.y],
      });
      const t = out?.tee;
      if (!t) { setPinNote("this hole has no tee ↔ green mapping yet"); return; }
      const off = Math.hypot(t[0] - pt.x, t[1] - pt.y);
      setPinNote(
        `the calibration puts the flag at ${Math.round(t[0])}, `
        + `${Math.round(t[1])} — ${off.toFixed(0)}px from where you put it`
        + (off > 8
          ? ". Re-calibrate tee ↔ green to close that gap; the flag itself "
            + "is unchanged."
          : ". That is close — the mapping agrees with you."));
    } catch (e) {
      setPinNote(e?.message || String(e));
    }
  }

  const loadGreenFrame = useCallback(
    async (f) => {
      const r = await api.getLongUploadFrame(adminPassword, row.id, f, "green");
      return r?.image_url;
    },
    [adminPassword, row.id],
  );
  const cometPoints = Object.entries(greenMarks)
    .map(([f, pt]) => ({ frame: parseInt(f, 10), x: pt.x, y: pt.y }))
    .sort((a, b) => a.frame - b.frame);
  const landing = cometPoints.length
    ? cometPoints[cometPoints.length - 1] : null;
  const comet = cometPoints.length > 1 ? { points: cometPoints } : null;

  // AFTER `landing` IS DECLARED, and that is the whole point of it
  // being here rather than up with the other landing state. A
  // dependency array is evaluated DURING RENDER, so an effect that
  // lists `landing?.x` above the `const landing` line reads a
  // const in its temporal dead zone -- "Cannot access 'X' before
  // initialization", thrown on every render of this modal, which
  // with no error boundary took the whole page white.
  // Where the current landing sits in the tee frame, asked of the
  // server because the homography lives there. Re-asked whenever the
  // landing moves on the green side, so the two pictures agree.
  useEffect(() => {
    if (!landing) {
      setTeeLandingXY(tracerEnd ? { x: tracerEnd.x, y: tracerEnd.y } : null);
      return undefined;
    }
    let live = true;
    (async () => {
      try {
        const out = await api.mapLandingToTee(adminPassword, row.id, {
          green: [landing.x, landing.y],
        });
        if (!live) return;
        // AN AIM ALREADY DRAGGED WINS over the mapped position. Without
        // this, re-opening the modal snapped the handle back to where
        // the homography puts the landing and silently discarded the
        // operator's own answer -- the one produce is actually using.
        setTeeLandingXY(
          tracerEnd
            ? { x: tracerEnd.x, y: tracerEnd.y }
            : (out?.tee ? { x: out.tee[0], y: out.tee[1] } : null));
        setTeeLandingNote(out?.tee ? null : (out?.reason || null));
      } catch (e) {
        if (live) {
          setTeeLandingXY(null);
          setTeeLandingNote(e?.message || String(e));
        }
      }
    })();
    return () => { live = false; };
    // tracerEnd is read but deliberately NOT a dependency: it is set BY
    // the drag this effect would otherwise fight, and listing it would
    // re-run the mapping on every drop.
  }, [landing?.x, landing?.y, adminPassword, row.id]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Dragged in the tee view: this is the TEE-SIDE AIM, and nothing else.
  //
  // It was moving the green landing, which is backwards. The landing is
  // a fact the green camera measured -- it saw the ball come down -- and
  // a drag on the tee picture is a much coarser statement about the same
  // place: the tee camera sees the whole green as a sliver a couple of
  // hundred pixels wide, so a few pixels here is many yards there.
  // Letting the coarse view overwrite the precise one loses the
  // measurement.
  //
  // What a drag here IS good for is the thing the tee picture actually
  // decides: where the predictive tracer is aimed. That is `tracer_end`,
  // which produce already prefers over the mapped landing and which is
  // saved with the swing.
  //
  // And the pair is worth something on its own. The operator has just
  // said "the green camera's landing, in MY picture, is HERE" -- a
  // correspondence between the two views, which is exactly what the
  // calibrator collects by clicking a bunker corner in both. The
  // distance between where the map put it and where they dragged it is
  // how wrong the map is at the one point that matters, so it is
  // measured and shown rather than quietly absorbed.
  async function dropTeeLanding(pt) {
    if (!pt) return;
    setTeeLandingXY(pt);
    setTracerEnd(pt);
    setEndGuessed(false);
    if (!landing) {
      setTeeLandingNote(
        `tracer aimed at ${pt.x}, ${pt.y} — mark the landing on the green `
        + "tab too and this also measures the hole's calibration");
      return;
    }
    try {
      const out = await api.mapLandingToTee(adminPassword, row.id, {
        green: [landing.x, landing.y],
      });
      const t = out?.tee;
      if (!t) { setTeeLandingNote(`tracer aimed at ${pt.x}, ${pt.y}`); return; }
      const off = Math.hypot(t[0] - pt.x, t[1] - pt.y);
      setTeeLandingNote(
        `tracer aimed at ${pt.x}, ${pt.y} · the green landing maps to `
        + `${Math.round(t[0])}, ${Math.round(t[1])} — ${off.toFixed(0)}px away`
        + (off > 8
          ? ". That gap is this hole's calibration error at the landing —"
            + " worth re-calibrating tee ↔ green."
          : ". The mapping agrees with you here."));
    } catch (e) {
      setTeeLandingNote(e?.message || String(e));
    }
  }

  // What the green half has to say, as one line. It lives over the
  // picture rather than on the toolbar: on the toolbar a sentence this
  // long is squeezed into a narrow column, wraps to a dozen lines, and
  // every one of them comes off the height of the map.
  const cometStatus = cometBusy
    ? "☄ walking back from the landing…"
    : cometPoints.length > 1
      ? `☄ ${cometPoints.length} frames (f${cometPoints[0].frame}→f${
          landing.frame}) — Save & close draws it on the clip`
      : cometPoints.length === 1
        ? `landing marked at f${landing.frame}`
          + (cometReason ? ` — ${cometReason}. ` : " — ")
          + "click more dots along the descent to draw the comet by hand"
        : greenErr || greenNote || "click the dots the ball comes down on";
  const greenChanged = JSON.stringify(
    cometPoints.length > 1 ? cometPoints : null) !== greenBase
    || (!!landing !== !!savedLanding)
    || (!!landing && !!savedLanding
        && (landing.frame !== savedLanding.frame
            || landing.x !== savedLanding.x
            || landing.y !== savedLanding.y));

  // THE GREEN PICTURE. Loaded on the first toggle, and again whenever
  // the landing moves: the frame worth looking at is the one the ball
  // is on. Before there is a landing that is where the produced clip
  // stops — its last green frame — which is the picture the operator
  // asked for.
  useEffect(() => {
    if (cam !== "green") return undefined;
    let dead = false;
    (async () => {
      try {
        setGreenErr(null);
        const want = landing?.frame ?? green?.defaultEnd ?? null;
        const out = await api.getLongUploadFrame(
          adminPassword, row.id,
          // No idea yet which frame that is: ask for one past the end
          // and let the server clamp, which also answers where produce
          // would stop.
          want ?? 1e9, "green", impactFrame ?? null,
        );
        if (dead) return;
        const end = out.default_end_frame ?? null;
        if (want == null && end != null && end !== out.frame) {
          const at = await api.getLongUploadFrame(
            adminPassword, row.id, end, "green", impactFrame ?? null,
          );
          if (!dead) setGreen({ ...at, defaultEnd: end });
          return;
        }
        setGreen({ ...out, defaultEnd: end ?? out.frame });
      } catch (e) {
        if (!dead) setGreenErr(e?.message || String(e));
      }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cam, landing?.frame, row.id]);

  // The heat over the green window — the same scan the wizard's landing
  // step runs. Fired once on the first toggle so the toggle itself is
  // the whole gesture; the button re-runs it a level deeper.
  async function scanGreen(level) {
    if (greenScanning) return;
    setGreenScanning(true);
    setGreenNote(null);
    try {
      const out = await api.scanPlotRegion(adminPassword, row.id, {
        which: "green",
        impact_frame: impactFrame ?? null,
        sensitivity: level,
      });
      const found = out.dots || [];
      setGreenDots(found);
      setGreenLevel(level);
      setGreenNote(
        found.length
          ? `${found.length} dots over f${out.start_frame}–f${out.end_frame} `
            + `(level ${level}) — click the one where it lands`
          : level >= 3
            ? "no motion at all in the green window — the ball did not "
              + "land in this camera's view"
            : `nothing at level ${level} — press Deeper`,
      );
    } catch (e) {
      setGreenNote(e?.message || String(e));
    } finally {
      setGreenScanning(false);
    }
  }
  const greenScanned = useRef(false);
  useEffect(() => {
    if (cam !== "green" || greenScanned.current) return;
    greenScanned.current = true;
    scanGreen(2);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cam]);

  // A click adds a point; clicking it again takes it away. Nothing is
  // replaced — that was the bug.
  function toggleGreenDot(p, clear = false) {
    setGreenMarks((m) => {
      const cur = m[p.frame];
      if (clear || (cur && Math.abs(cur.x - p.x) <= 2
                    && Math.abs(cur.y - p.y) <= 2)) {
        const next = { ...m };
        delete next[p.frame];
        return next;
      }
      return { ...m, [p.frame]: { x: p.x, y: p.y } };
    });
  }

  // Walk the descent back from the latest marked point. Whatever it
  // finds is merged INTO the marks, so the found chain arrives as
  // ordinary plotted points the operator can edit rather than as a
  // separate thing to accept or reject. Run automatically off the first
  // click, and by the button after that.
  async function findComet(from) {
    const at = from || landing;
    if (!at || cometBusy) return;
    setCometBusy(true);
    setCometReason(null);
    try {
      const out = await api.greenFlight(adminPassword, row.id, {
        landing_frame: at.frame,
        landing_spot: [at.x, at.y],
      });
      if (out?.points?.length) {
        setGreenMarks((m) => {
          const next = { ...m };
          for (const q of out.points) {
            next[q.frame] = { x: Math.round(q.x), y: Math.round(q.y) };
          }
          return next;
        });
      } else {
        setCometReason(out?.reason || "no obvious path");
      }
    } catch (e) {
      setCometReason(e?.message || String(e));
    } finally {
      setCometBusy(false);
    }
  }

  // Resolve this swing's produced clip by IDENTITY (clip_id) first —
  // positional lookup goes stale as soon as a clip is deleted.
  const clipForSwing =
    (row.produced_clips || []).find(
      (c) => swing.clip_id != null && c.id === swing.clip_id,
    ) ?? (isNew || swingPos == null
      ? null : row.produced_clips?.[swingPos]);
  const holeNumber = Number(
    swing.finalized_hole_number
      ?? clipForSwing?.hole_number
      ?? 1,
  ) || 1;

  function toggleDot(p, forceClear = false) {
    const f = p.frame;
    setMarks((m) => {
      const cur = m[f];
      // Clicking the SAME dot again un-marks it (as before). Alt/right
      // click clears the frame outright, which matters when the mark and
      // the dot are a few px apart — the position test would otherwise
      // treat the click as a move and there was no way to remove it.
      if (forceClear || (cur && Math.abs(cur.x - p.x) <= 2 && Math.abs(cur.y - p.y) <= 2)) {
        const next = { ...m };
        delete next[f];
        return next;
      }
      return { ...m, [f]: { x: p.x, y: p.y } };
    });
  }

  // Drop every mark. Baked points become cleared frames on save, so this
  // is "remove the whole plotted track", not just "forget my edits".
  function clearAllMarks() {
    setMarks({});
  }

  // Put it back to whatever was saved before this modal was opened.
  function resetMarks() {
    setMarks({ ...baked });
    setGreenMarks(JSON.parse(greenBase || "null")?.reduce(
      (acc, q) => ({ ...acc, [q.frame]: { x: Math.round(q.x),
                                          y: Math.round(q.y) } }), {})
      || (savedLanding
        ? { [savedLanding.frame]: { x: savedLanding.x, y: savedLanding.y } }
        : {}));
    setCometReason(null);
    setImpactFrame(swing.impact_frame ?? null);
    setBallAtRest(swing.ball ?? null);
    setPlaceOnTee(null);
    setPlaceOnGreen(null);
  }

  // Diff vs the baked state: new/moved picks become manual points,
  // un-clicked baked dots become cleared frames.
  function pendingChanges() {
    const overrides = Object.entries(marks)
      .filter(([f, p]) => {
        const b = baked[f];
        return !(b && b.x === p.x && b.y === p.y);
      })
      .map(([f, p]) => ({ frame: parseInt(f, 10), x: p.x, y: p.y }));
    const cleared = Object.keys(baked)
      .filter((f) => !marks[f])
      .map((f) => parseInt(f, 10));
    return { overrides, cleared };
  }

  async function saveAndClose() {
    const { overrides, cleared } = pendingChanges();
    const movedImpact =
      impactFrame != null && impactFrame !== (swing.impact_frame ?? null);
    const movedBall =
      !!ballAtRest &&
      (ballAtRest.x !== (swing.ball?.x ?? null) ||
        ballAtRest.y !== (swing.ball?.y ?? null));
    if (
      !isNew
      && overrides.length === 0 && cleared.length === 0
      && !movedImpact && !movedBall && !greenChanged
      && !shapeChanged
    ) {
      onClose();
      return;
    }
    // OUT OF THE OPERATOR'S WAY, IMMEDIATELY. This is four server calls
    // and a video render -- tens of seconds -- and the modal used to sit
    // there greyed for all of it, then un-grey on a failure with a line
    // of red text at the bottom of a full-screen editor. Same shape as
    // the wizard's Produce: hand the run to the production card, which
    // is where a minutes-long job belongs, and close.
    // NOTHING CHANGED ON THE TEE SIDE means nothing about the tracer
    // changed, and re-rendering it is minutes of work to arrive back at
    // the same overlay. A landing marked on the green camera only
    // affects the green half, so skip straight to the finalize that
    // draws it.
    // ...but only when the tracer finalize will pick up is already THIS
    // swing's. finalize composites the upload-level tracer_url, which
    // the fast render is what sets — on a multi-swing upload where the
    // last render was another swing's, skipping it would finalize the
    // wrong tracer.
    const teeChanged =
      overrides.length > 0 || cleared.length > 0 || movedImpact || movedBall
      // A shaped arc IS a different tracer, so it has to be re-drawn.
      || shapeChanged
      || !swing.tracer_url
      || row.edit_metrics?.tracer_url !== swing.tracer_url;
    const stage = (msg) => onBackground?.(msg);
    stage(teeChanged ? "Re-rendering the tracer…" : "Drawing the comet…");
    onClose();
    try {
      // 1. Bake the picks into the swing's track (cv2 only, no AI).
      // A NEW SWING HAS NO WINDOW YET, and the renderer needs one to know
      // which frames to draw over. The impact frame is the only anchor
      // the operator has given, so the window is hung off it: a few
      // seconds of lead-in and enough after for a full flight, clipped to
      // the footage that actually exists.
      const fps_ = Number(row.tee_fps) || 30;
      const lastF_ = row.tee_nb_frames ? row.tee_nb_frames - 1 : null;
      const newWindow =
        isNew && impactFrame != null
          ? {
            start_frame: Math.max(0, Math.round(impactFrame - 3 * fps_)),
            end_frame: (() => {
              const e = Math.round(impactFrame + 10 * fps_);
              return lastF_ == null ? e : Math.min(e, lastF_);
            })(),
          }
          : null;
      const hasWindow =
        swing.start_frame != null && swing.end_frame != null;
      const renderWindow = newWindow
        || (hasWindow
          ? { start_frame: swing.start_frame, end_frame: swing.end_frame }
          : null);
      const fast = teeChanged
        ? await api.renderWizardTracerFast(adminPassword, row.id, {
          manual_positions: overrides,
          cleared_frames: cleared,
          base_track_frames: swing.ball_track_frames || [],
          impact_frame: impactFrame ?? null,
          ball_at_rest: ballAtRest || null,
          // THE ARC AS SHAPED. The aim point the operator dragged wins
          // over the wizard's target — on this screen they are looking
          // at the picture the line is drawn on, which the target was
          // only ever a proxy for.
          target: tracerEnd || swing.target || null,
          flight_sec: flightSec,
          render_window: renderWindow,
        })
        : {
          tracer_url: swing.tracer_url,
          ball_track_frames: swing.ball_track_frames || [],
        };
      // The swing this save writes: a fresh one appended to the list when
      // the map was opened on a swing that does not exist yet, otherwise
      // the existing one patched in place. `newIdx` follows the wizard's
      // rule -- one past the highest idx on the row, not the array length,
      // so a deleted swing cannot make two swings share a number.
      const newIdx = isNew
        ? (swings.length
          ? Math.max(...swings.map((s) => s.idx ?? 0)) + 1 : 0)
        : (swing.idx ?? swingPos);
      const patch = (s) => ({
              ...s,
              impact_frame: impactFrame ?? s.impact_frame,
              // The landing, and the descent found from it. Kept on the
              // swing so a later produce and a later re-open both start
              // from what was marked here.
              // Kept on the swing so a re-produce draws the same arc
              // rather than reverting to the model's.
              ...(tracerEnd
                ? { tracer_end: [tracerEnd.x, tracerEnd.y] } : {}),
              ...(flightSec != null
                ? { tracer_flight_sec: flightSec } : {}),
              ...(landing
                ? {
                    landing_frame: landing.frame,
                    landing_spot: { x: landing.x, y: landing.y },
                    green_track: cometPoints.length > 1
                      ? cometPoints : null,
                  }
                : {}),
              ...(ballAtRest
                // ball_manual marks it operator-placed; the produce
                // worker checks that flag before writing a detected rest
                // position, so a re-produce cannot move it back.
                ? { ball: ballAtRest, ball_manual: true }
                : {}),
              tracer_url: fast.tracer_url,
              ball_track_frames: fast.ball_track_frames || [],
      });
      const nextSwings = isNew
        ? [...swings, patch({
          idx: newIdx,
          fps: fps_,
          address_frame: renderWindow?.start_frame ?? 0,
          ...(renderWindow || {}),
        })]
        : swings.map((s, i) => (i === swingPos ? patch(s) : s));
      await api.saveEditMetrics(adminPassword, row.id, {
        swings: nextSwings,
        // Mirrored to the top level ONLY on a single-swing row, which is
        // where the wizard puts it and where such a row reads it back
        // from. On a multi-swing row the top level is a slot every swing
        // shares, so writing this swing's landing into it is how one
        // clip's landing became every later clip's.
        ...(landing && soloRow
          ? {
              landing_frame: landing.frame,
              landing_spot: { x: landing.x, y: landing.y },
            }
          : {}),
      });
      // 2. PRODUCE, the same way the edit wizard's Produce does.
      //
      // This used to be three legacy calls -- render-tracer-fast, then
      // finalize, then commit-clip -- while the wizard's Produce went
      // through find_flight + _d3_fast_produce. Two paths, and they had
      // drifted: the wizard's tracer flew and held for the whole tee
      // half while a clip saved from here came back with a short stub.
      // One renderer, so they cannot disagree again.
      //
      // The plotted points go WITH the request. That is the whole
      // difference between the two callers: produce normally finds its
      // own flight, and here the operator has just overruled it by
      // hand, so their line is handed over and find_flight is skipped.
      stage("Producing the clip…");
      // ONE SWING'S POINTS, AND NOTHING ELSE'S. A ball is in the air
      // for a few seconds, so a point thirty seconds either side of
      // impact is not a late tail of this flight -- it is another
      // swing's, and two clusters that far apart get fitted into one
      // curve that passes through neither. Bounded generously, at twice
      // the longest flight the renderer will cut: this exists to catch
      // points from a different swing, not to second-guess the tracker.
      const _fps2 = Number(row.tee_fps) || 30;
      const _impF = impactFrame ?? swing.impact_frame ?? null;
      const _lo2 = _impF == null ? null : _impF - 2 * _fps2;
      const _hi2 = _impF == null ? null : _impF + 22 * _fps2;
      const _all2 = (fast.ball_track_frames || [])
        .filter((p) => p && p.found !== false
          && p.x != null && p.y != null && p.frame != null)
        .map((p) => ({ frame: p.frame, x: p.x, y: p.y }))
        .sort((a, b) => a.frame - b.frame);
      const plotted = _lo2 == null
        ? _all2
        : _all2.filter((p) => p.frame >= _lo2 && p.frame <= _hi2);
      if (plotted.length !== _all2.length) {
        console.warn(
          `click-to-plot: dropped ${_all2.length - plotted.length} track `
          + `point(s) outside f${_lo2}-f${_hi2} around impact f${_impF} `
          + "- they belong to a different swing");
      }
      await api.wizardProduce(adminPassword, row.id, {
        // NO BALL AT REST IS NORMAL HERE. Click-to-plot is often used on
        // exactly the swings where the rest ball was never found -- the
        // operator plots the flight off the motion map instead. The
        // first plotted point is where the line starts, so it stands in;
        // sending null rejected the save with a 400 and the whole plot
        // went nowhere.
        ball: ballAtRest
          ? [ballAtRest.x, ballAtRest.y]
          : (swing.ball
            ? [swing.ball.x, swing.ball.y]
            : (plotted.length ? [plotted[0].x, plotted[0].y] : null)),
        impact_frame: impactFrame ?? swing.impact_frame
          ?? (plotted.length ? plotted[0].frame : null),
        landing_frame: landing?.frame ?? null,
        landing_spot: landing ? [landing.x, landing.y] : null,
        hole_number: holeNumber,
        // THIS SWING ONLY. Click-to-plot is opened on one clip, so a
        // save from it must not clear the upload's other clips.
        solo: true,
        swing_idx: newIdx,
        points: plotted.length >= 2 ? plotted : null,
        launch_frame: plotted.length ? plotted[0].frame : null,
      });
      onDone?.(true, null);
    } catch (e) {
      // The modal is gone, so the failure has to surface on the card.
      onDone?.(false, e.message);
    }
  }

  const { overrides: pendAdd, cleared: pendClear } = pendingChanges();
  const impactMoved =
    impactFrame != null && impactFrame !== (swing.impact_frame ?? null);
  const ballMoved =
    !!ballAtRest &&
    (ballAtRest.x !== (swing.ball?.x ?? null) ||
      ballAtRest.y !== (swing.ball?.y ?? null));
  // WHATEVER PICTURE IS IN FRONT OF THE OPERATOR. On the tee map that
  // is the frame stepper; on the raw tab it is the tee player, but only
  // while it is stopped -- "current" has to mean a frame you are looking
  // at, not one that has already gone past.
  const teeCurrent = cam === "raw" ? (rawFrames?.teeFrame ?? null)
    : teeViewFrame;
  const greenCurrent = cam === "raw" ? (rawFrames?.greenFrame ?? null)
    : greenViewFrame;
  // MOVE THE LANDING TO ANOTHER FRAME, keeping where it is on the green.
  // The landing frame is not stored: it IS the frame the last green
  // point is filed under, so setting it means re-keying that point.
  // Nothing to re-key without a landing spot, which is why this asks for
  // one rather than inventing a position.
  function setLandingFrame(f) {
    if (f == null || !landing || f === landing.frame) return;
    const from = landing.frame;
    const xy = { x: landing.x, y: landing.y };
    setGreenMarks((m) => {
      const next = { ...m };
      delete next[from];
      next[f] = xy;
      return next;
    });
    setCometReason(null);
  }
  const nChanged =
    pendAdd.length + pendClear.length + (impactMoved ? 1 : 0)
    + (ballMoved ? 1 : 0) + (greenChanged ? 1 : 0)
    + (shapeChanged ? 1 : 0);
  // A NEW SWING NEEDS AN IMPACT FRAME AND NOTHING ELSE. Every other
  // field on this panel is optional -- a clip can be a tee tracer with
  // no landing, or a landing with no tee spot -- but without impact
  // there is no window to render and no moment to cut on, so that is
  // the one thing Save waits for. `nChanged` measures edits against a
  // saved swing, which a brand new one has none of, so it cannot gate
  // the button here.
  const canSave = isNew ? impactFrame != null : nChanged > 0;
  // The earliest point actually in the saved track — with a wrong impact
  // frame this is the honest answer to "when does the ball leave", so it
  // is offered as a one-click fix.
  const firstTrackF = (swing.ball_track_frames || [])
    .filter((r) => r.found && r.x != null && r.y != null)
    .reduce((m, r) => (m == null || r.frame < m ? r.frame : m), null);
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={`Click-to-plot for upload ${row.id}`}
      onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.85)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 1000, padding: 6, cursor: "zoom-out",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="card"
        style={{
          maxWidth: "min(2200px, 99.5vw)", width: "100%",
          maxHeight: "99vh", height: "99vh", overflow: "hidden",
          cursor: "default", margin: 0,
          // .card's 20px is a page card's padding; here it is 40px of
          // picture in each direction.
          padding: 8,
          display: "flex", flexDirection: "column",
        }}
      >
        <div
          className="row"
          style={{
            alignItems: "center", justifyContent: "space-between",
            gap: 8, marginBottom: 4,
            // One row, always. A long label that wraps here turns the
            // toolbar into a column and takes a third of the map with
            // it -- which is exactly what the green status line did.
            flexWrap: "nowrap", overflow: "hidden",
          }}
        >
          <div
            className="small"
            style={{
              whiteSpace: "nowrap", overflow: "hidden",
              textOverflow: "ellipsis", minWidth: 0,
            }}
          >
            <b>🖱 Click-to-plot</b>
            <span className="muted">
              {" "}· #{row.id} ·{" "}
              {isNew ? "new swing" : `swing ${(swing.idx ?? swingPos) + 1}`}
              {" "}· hole {holeNumber}
              {cam === "tracer"
                ? " · the tracer's line"
                  + (shape?.kind ? ` · ${shape.kind}` : "")
                : cam === "green"
                ? ` · green camera · ${greenDots.length} dots`
                  + (green?.frame != null ? ` · f${green.frame}` : "")
                : ` · ${dots.length} dots`
                  + (denseDots.length > 0
                    ? ` · ${denseDots.length} candidates` : "")
                  + (winLo != null ? ` · showing f${winLo}–f${winHi}` : "")}
            </span>
          </div>
          {/* WHICH CAMERA. Two pictures of the same shot: the tee, where
              the tracer is drawn, and the green, where it comes down.
              The tee half is plotted point by point; the green half
              needs one click — the landing — and finds its own path
              back from it. */}
          <div className="row" style={{ gap: 4, alignItems: "center" }}>
            {["tee", "green", "raw"].map((w) => (
              <button
                key={w}
                type="button"
                className={cam === w ? "small" : "ghost small"}
                style={{ width: "auto", padding: "0 10px" }}
                onClick={() => setCam(w)}
                disabled={w === "green" && !row.green_filename}
                title={
                  w === "tee"
                    ? "The tee camera's motion heat — plot the tracer's ball points"
                    : w === "raw"
                      ? "Both raw videos, locked to the wall clock and driven by one set of transport buttons"
                      : row.green_filename
                        ? "The green camera at the end of the clip — click where the ball lands and the comet is found from there"
                        : "This upload has no green video"
                }
              >
                {w === "tee" ? "tee view"
                  : w === "green" ? "☄ green view" : "▶ raw video"}
              </button>
            ))}
          </div>
          <div
            className="row"
            style={{
              gap: 8, alignItems: "center", flexWrap: "nowrap",
              flexShrink: 0,
            }}
          >
            {nChanged > 0 && (
              <span className="small" style={{ color: "var(--emerald-700)" }}>
                {pendAdd.length > 0 && `${pendAdd.length} new`}
                {pendAdd.length > 0 && pendClear.length > 0 && " · "}
                {pendClear.length > 0 && `${pendClear.length} removed`}
              </span>
            )}
            {/* TEE-ONLY CONTROLS. The impact frame, the tracer's start
                and the plotted points are all things about the tee
                picture; on the green camera they are noise, and worse,
                they are noise that wraps the toolbar onto three lines
                and takes that height off the map. */}
            {/* THE TEE'S NUMBERS MOVED RIGHT. Impact, the tracer's
                start and the plotted-point count all used to live here,
                which put five controls and three readouts on one line
                above the picture — and every one of them is a fact
                about the swing, which is what the panel down the right
                is for. What is left on the toolbar is what acts on the
                DIALOG rather than on the swing. */}
            {/* THE COMET. In green mode this is the live state of the
                click just made — searching, found, or the sentence
                saying why there is no path. In tee mode it is what the
                last produce drew, which is otherwise only answerable by
                watching the clip to the end. */}
            {cam === "tracer" ? (
              <>
                <span className="tiny muted" style={{ whiteSpace: "nowrap" }}>
                  {tracerEnd
                    ? `${endGuessed ? "guessed" : "ends"} `
                      + `${tracerEnd.x},${tracerEnd.y}`
                    : "no aim point"}
                </span>
                <span
                  className="small"
                  style={{
                    display: "inline-flex", alignItems: "center", gap: 4,
                    whiteSpace: "nowrap",
                  }}
                  title={
                    "How long the ball is in the air, which is what paces "
                    + "the drawn continuation. Measured off the two "
                    + "cameras' clocks when the landing is marked on the "
                    + "green; otherwise from the hole's yardage. Change it "
                    + "and the tracer re-times."
                  }
                >
                  <span className="muted">flight</span>
                  <button
                    type="button"
                    className="ghost small"
                    style={{ width: "auto", padding: "0 6px" }}
                    disabled={flightSec == null}
                    onClick={() => setFlightSec(
                      (v) => Math.max(0.5, Math.round((v - 0.2) * 10) / 10))}
                  >
                    −
                  </button>
                  <input
                    type="number"
                    step="0.1"
                    min="0.5"
                    max="20"
                    value={flightSec ?? ""}
                    onChange={(e) => {
                      const n = parseFloat(e.target.value);
                      setFlightSec(Number.isFinite(n)
                        ? Math.max(0.5, Math.min(20, n)) : null);
                    }}
                    style={{ width: 62, textAlign: "center", padding: "1px 4px" }}
                  />
                  <span className="muted">s</span>
                  <button
                    type="button"
                    className="ghost small"
                    style={{ width: "auto", padding: "0 6px" }}
                    disabled={flightSec == null}
                    onClick={() => setFlightSec(
                      (v) => Math.min(20, Math.round((v + 0.2) * 10) / 10))}
                  >
                    +
                  </button>
                </span>
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto" }}
                  disabled={!shapeChanged}
                  onClick={() => {
                    const e = JSON.parse(shapeBase);
                    setTracerEnd(e
                      ? { x: Math.round(e[0]), y: Math.round(e[1]) }
                      : null);
                    setEndGuessed(false);
                    setFlightSec(flightBase);
                    if (!e) fetchShape();
                  }}
                  title="Put the arc back to the shape that was saved"
                >
                  Reset arc
                </button>
              </>
            ) : cam === "green" ? (
              <>
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto" }}
                  disabled={greenScanning}
                  onClick={() => scanGreen(Math.min(3, greenLevel + 1))}
                  title="Frame-diff the green window again, one level deeper: more dots, including leaves in the wind. You are the filter."
                >
                  {greenScanning
                    ? "Scanning…"
                    : greenLevel >= 3 ? "🔍 Rescan" : "🔍 Deeper"}
                </button>
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto" }}
                  disabled={!landing || cometBusy}
                  onClick={() => findComet()}
                  title="Walk the ball's descent backwards from the last marked point and plot every frame it finds. Whatever comes back is added to the marks, so it can be corrected by hand."
                >
                  {cometBusy ? "Looking…" : "☄ Find descent"}
                </button>
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto" }}
                  disabled={!cometPoints.length}
                  onClick={() => { setGreenMarks({}); setCometReason(null); }}
                  title="Remove every plotted comet point"
                >
                  Clear ({cometPoints.length})
                </button>
              </>
            ) : swing?.green_track ? (
              <span className="tiny" style={{ color: "#3ee37a" }}>
                {/* MIN AND MAX, not first and last. A saved track is not
                    promised to be in frame order, and reading its ends
                    positionally reported a span it does not have. */}
                ☄ green comet · {swing.green_track.length} frames
                {" "}(f{Math.min(...swing.green_track.map((q) => q.frame))}→
                f{Math.max(...swing.green_track.map((q) => q.frame))})
              </span>
            ) : null}
            <button
              type="button"
              className="ghost small"
              onClick={onClose}
              style={{ width: "auto" }}
            >
              Cancel
            </button>
            <button
              type="button"
              className="small"
              onClick={saveAndClose}
              style={{ width: "auto" }}
              disabled={!canSave}
              title={
                canSave
                  ? "Re-render the tracer with the changes, re-apply graphics, and update Produced Clips"
                  : (isNew
                    ? "Set the impact frame first — a new clip is cut around it"
                    : "Click dots on the heat to add/remove ball points first — green dots are already in the saved track")
              }
            >
              {isNew ? "Add clip" : "Save & close"}
            </button>
          </div>
        </div>

        <div style={{
          flex: 1, minHeight: 0, display: "flex",
          alignItems: "stretch", gap: 10,
        }}>
        <div style={{
          flex: 1, minWidth: 0, minHeight: 0, display: "flex",
          alignItems: "stretch", justifyContent: "center",
        }}>
          {cam === "raw" ? (
            <RawSyncPlayer
              teeUrl={row.tee_url}
              greenUrl={row.dual_camera ? row.green_url : null}
              teeFps={row.tee_fps}
              greenFps={row.green_fps}
              deltaSec={rawDelta?.sec ?? 0}
              // NO FALLBACK TO base_captured_at. That is one timestamp
              // for the upload, so handing it to both cameras makes them
              // look like they started in the same millisecond and turns
              // "nobody stamped this" into a confident, wrong time of
              // day. Missing is missing; the player says so.
              teeStartedAt={row.tee_recording_started_at}
              greenStartedAt={row.green_recording_started_at}
              deltaSource={rawDelta?.source || "assumed"}
              baseCapturedAt={row.base_captured_at}
              onFrames={setRawFrames}
            />
          ) : cam === "tracer" ? (
            bgUrl ? (
              <PlotHeatCanvas
                // ONE INSTANCE PER CAMERA. All three tabs render this
                // same component in the same slot of one ternary, so
                // without a key React reuses a single instance across a
                // tab switch -- and it carries its loaded frame image and
                // its frame cache with it. That cache is keyed by frame
                // number alone, so tee f900 would be served the GREEN
                // camera's f900. What it looked like was the green
                // picture with the tee's dots drawn on it.
                key={`plot-${cam}`}
                // THE SAME PICTURE AS THE TEE MAP, because the tracer
                // is drawn on the tee camera and the motion heat is
                // where its points came from. What changes is that the
                // LINE is the subject: the dots step back to a faint
                // layer and two handles appear on the curve.
                bgUrl={bgUrl}
                dots={[]}
                denseDots={[]}
                frameW={frameW}
                frameH={frameH}
                marks={{}}
                track={(swing.ball_track_frames || []).filter(
                  (r) => r.found && r.x != null && r.y != null,
                )}
                ballXY={ballAtRest}
                shape={tracerPath}
                handles={tracerEnd ? [{
                  id: "end", x: tracerEnd.x, y: tracerEnd.y,
                  colour: "#f472b6",
                  title: "Where the ball finishes IN THIS PICTURE. Drag "
                    + "it and the whole arc re-solves to reach it — the "
                    + "same parabola the tracked ball is already on, "
                    + "carried on to here. This is the tee view only, "
                    + "and does not move the landing marked on the "
                    + "green camera.",
                }] : []}
                onHandleDrag={(id, pt) => {
                  if (id !== "end") return;
                  setTracerEnd(pt);
                  setEndGuessed(false);
                }}
                onHandleDrop={() => fetchShape()}
                note={shapeErr
                  || (tracerEnd
                    ? `${shape?.kind || "…"} · ${shape?.n || 0} frames of `
                      + "flight — "
                      + (endGuessed
                        ? `this landing is a GUESS from ${
                            shape?.predicted_from || "the flight"}; drag `
                          + "the pink handle onto the real one"
                        : "drag the pink handle to move where it lands")
                      + (shape?.flight_from
                        ? ` · flight from ${shape.flight_from}` : "")
                    : "nothing tracked to work from on this swing")}
                noteColour={shapeErr ? "#f59e0b" : "#67e8f9"}
              />
            ) : (
              <div className="muted small" style={{ textAlign: "center", padding: 24 }}>
                No motion heat saved for this swing — produce it once and
                the tracer can be shaped here.
              </div>
            )
          ) : cam === "green" ? (
            green?.image_url ? (
              <PlotHeatCanvas
                // ONE INSTANCE PER CAMERA. All three tabs render this
                // same component in the same slot of one ternary, so
                // without a key React reuses a single instance across a
                // tab switch -- and it carries its loaded frame image and
                // its frame cache with it. That cache is keyed by frame
                // number alone, so tee f900 would be served the GREEN
                // camera's f900. What it looked like was the green
                // picture with the tee's dots drawn on it.
                key={`plot-${cam}`}
                bgUrl={green.image_url}
                // THE SAME MAP, POINTED AT THE OTHER CAMERA. Dots are
                // the green window's motion, and every click plots one
                // of the comet's points — same gesture as the tee side,
                // so they accumulate. There is no tracer line or start
                // marker out here; the path IS the marks.
                dots={greenDots}
                denseDots={[]}
                frameW={green.width}
                frameH={green.height}
                marks={greenMarks}
                // Step the green camera the same way the tee steps, over
                // the whole green clip — the descent is a second or two
                // and the frame it starts on is not knowable in advance.
                loadFrame={loadGreenFrame}
                frameLo={0}
                frameHi={Math.max(0, (green.total_frames || 1) - 1)}
                // Where the operator left off, then the landing --
                // which is what `set to current` on the raw tab moves
                // this to, so setting the landing frame there lands on
                // it here.
                startFrame={greenViewFrame ?? landing?.frame
                  ?? green.frame ?? 0}
                onViewFrame={setGreenViewFrame}
                // PLACE THE LANDING ANYWHERE. It lands in `greenMarks`
                // like a clicked dot does, so everything downstream —
                // the comet, the saved landing_spot/landing_frame, the
                // tracer's aim and its flight time — is fed by the one
                // mechanism and cannot disagree with itself.
                // DRAG THEM, DO NOT ARM A MODE. The ball and the flag
                // are things in the picture; picking one up and putting
                // it down is what an operator means, and it works
                // whenever they look at it rather than only after
                // pressing a button first.
                handles={[
                  ...(landing ? [{
                    id: "landing", x: landing.x, y: landing.y, icon: "ball",
                    title: "Where the ball finished. Drag it onto the ball.",
                  }] : []),
                  ...(pinGreen ? [{
                    id: "pin", x: pinGreen.x, y: pinGreen.y, icon: "flag",
                    title: "The flagstick. Drag it to the BASE of the stick — the tee view follows through the calibration, and it stays put for every swing on this hole.",
                  }] : []),
                ]}
                onHandleDrag={(id, pt) => {
                  if (!pt) return;
                  if (id === "pin") { setPinGreen(pt); return; }
                  const f = landing?.frame ?? greenViewFrame
                    ?? green.frame ?? 0;
                  setGreenMarks((m) => ({ ...m, [f]: { x: pt.x, y: pt.y } }));
                }}
                onHandleDrop={(id, pt) => {
                  if (id === "pin" && pt) setPinFromGreen(pt);
                  if (id === "landing") setCometReason(null);
                }}
                // Nothing placed yet? One click puts it down; after that
                // it is dragged like everything else.
                placingBall={placeOnGreen != null}
                onPlaceBall={(pt) => {
                  if (placeOnGreen === "pin") { setPinFromGreen(pt); }
                  else {
                    const f = greenViewFrame ?? green.frame ?? 0;
                    setGreenMarks((m) => ({ ...m, [f]: { x: pt.x, y: pt.y } }));
                    setCometReason(null);
                  }
                  setPlaceOnGreen(null);
                }}
                track={[]}
                comet={comet}
                note={cometStatus}
                noteColour={cometBusy
                  ? "#fde047"
                  : comet
                    ? "#3ee37a"
                    : cometReason ? "#f59e0b" : "#fde047"}
                onToggleDot={(p, clear) => {
                  const first = Object.keys(greenMarks).length === 0;
                  toggleGreenDot({ frame: p.frame, x: p.x, y: p.y }, clear);
                  // The first mark is the landing, and the landing is
                  // the one thing the search needs — so offer the chain
                  // straight away instead of making the operator plot a
                  // descent the machine can usually see.
                  if (first && !clear) {
                    findComet({ frame: p.frame, x: p.x, y: p.y });
                  }
                }}
                scanRegion={async (region, sensitivity) => {
                  const out = await api.scanPlotRegion(adminPassword, row.id, {
                    ...region,
                    which: "green",
                    sensitivity,
                    impact_frame: impactFrame ?? null,
                  });
                  return out.dots || [];
                }}
              />
            ) : (
              <div className="muted small" style={{ textAlign: "center", padding: 24 }}>
                {greenErr
                  ? `Could not load the green frame: ${greenErr}`
                  : "Loading the green camera…"}
              </div>
            )
          ) : (dots.length > 0 || denseDots.length > 0 || canStepTee) ? (
            <PlotHeatCanvas
              key={`plot-${cam}`}
              // NO HEAT COMPOSITE ON A NEW SWING, and that is fine --
              // there has been no produce, so there is nothing to
              // composite. The map opens on the real video frame
              // instead, which is what it opens on anyway; `bgUrl` is
              // only the fallback for the instant before the first
              // frame arrives.
              bgUrl={bgUrl || null}
              dots={dots}
              denseDots={denseDots}
              frameW={frameW}
              frameH={frameH}
              marks={marks}
              // THE SWING'S OWN BOUNDS, not the dot window. The dot
              // window is impact−5…impact+40, and the commonest reason
              // to want the video is that the impact frame is WRONG --
              // bounding the search by a window derived from it would
              // make finding the right one circular.
              loadFrame={loadTeeFrame}
              frameLo={swing.start_frame ?? (isNew ? 1 : 0)}
              frameHi={swing.end_frame
                ?? (row.tee_nb_frames ? row.tee_nb_frames - 1 : (winHi ?? 0))}
              // WHERE THE OPERATOR LEFT OFF, first. Keying this per
              // camera remounts it on a tab switch, which is what stops
              // the two cameras sharing a picture -- but a remount also
              // forgets the frame, and walking back to f3017 after a
              // glance at the green is not a thing to ask twice. The
              // parent already tracks the tee frame for `set to
              // current`, so it is handed back on the way in.
              //
              // Failing that: a new clip opens at frame 1 and the
              // operator walks forward, because on a swing nobody has
              // produced yet there is no impact frame to open at and
              // finding the strike IS the first job. An existing swing
              // opens on its impact frame.
              startFrame={teeViewFrame ?? impactF ?? (isNew ? 1 : undefined)}
              track={(swing.ball_track_frames || []).filter(
                (r) => r.found && r.x != null && r.y != null,
              )}
              onToggleDot={toggleDot}
              ballXY={ballAtRest}
              placingBall={placeOnTee != null}
              onPlaceBall={(pt) => {
                if (placeOnTee === "pin") setPinFromTee(pt);
                else setBallAtRest(pt);
                setPlaceOnTee(null);
              }}
              // THE LANDING, GRABBABLE IN THE TEE PICTURE. It is stored
              // in green pixels, but this is the picture the tracer is
              // drawn on, so it is the one where the operator can see
              // whether the line finishes where the ball did.
              handles={[
                ...(ballAtRest ? [{
                  id: "tee", x: ballAtRest.x, y: ballAtRest.y, icon: "tee",
                  title: "The tee — where the ball was hit from, and where the tracer starts. Drag it onto the ball.",
                }] : []),
                ...(pinTee ? [{
                  id: "pin", x: pinTee.x, y: pinTee.y, icon: "flag",
                  title: "The flagstick, put here by this hole's calibration rather than by a click. Dragging it does NOT move the flag — the green camera decides that — it measures how far the calibration is out at the one point both cameras can identify.",
                }] : []),
                ...(teeLandingXY ? [{
                  id: "aim", x: teeLandingXY.x, y: teeLandingXY.y,
                  colour: "#f472b6",
                  title: "Where the tracer is aimed. Drag it onto the spot you can actually see; the landing on the green camera is not changed.",
                }] : []),
              ]}
              onHandleDrag={(id, pt) => {
                if (!pt) return;
                if (id === "tee") setBallAtRest(pt);
                if (id === "pin") setPinTee(pt);
                if (id === "aim") setTeeLandingXY(pt);
              }}
              onHandleDrop={(id, pt) => {
                if (id === "pin" && pt) setPinFromTee(pt);
                if (id === "aim") dropTeeLanding(pt);
              }}
              onViewFrame={setTeeViewFrame}
              scanRegion={async (region, sensitivity) => {
                const out = await api.scanPlotRegion(adminPassword, row.id, {
                  ...region,
                  sensitivity,
                  start_frame:
                    winLo != null
                      ? Math.max(swing.start_frame ?? 0, winLo)
                      : swing.start_frame ?? 0,
                  end_frame:
                    winHi != null
                      ? Math.min(swing.end_frame ?? winHi, winHi)
                      : swing.end_frame ?? null,
                });
                return inWindow(out.dots || []);
              }}
            />
          ) : (
            <div className="muted small" style={{ textAlign: "center", padding: 24 }}>
              No timed motion dots saved for this swing yet.
              <br />
              Re-Produce the upload (or run a Classical render in the Edit
              wizard) to populate them.
            </div>
          )}
        </div>

        {/* WHAT THIS SWING IS, down the right. The edit wizard's field
            list, on the screen that has the pictures — because every
            real edit needed both and they used to be two dialogs. */}
        <div style={{
          width: 250, flexShrink: 0, display: "flex",
          flexDirection: "column", gap: 7, overflowY: "auto", minHeight: 0,
        }}>
          <PlotField
            label="Tee spot" icon="⛳"
            value={ballAtRest ? `${ballAtRest.x}, ${ballAtRest.y}` : null}
            accent="#fde68a"
            hint={ballAtRest
              ? "Drag the tee on the tee view onto the ball."
              : "Click the tee view where the ball was hit from."}
          >
            {!ballAtRest && (
              <button type="button"
                      className={placeOnTee === "tee" ? "small" : "ghost small"}
                      style={{ width: "100%" }}
                      onClick={() => {
                        setCam("tee");
                        setPlaceOnTee((v) => (v === "tee" ? null : "tee"));
                      }}>
                {placeOnTee === "tee" ? "click the tee view…" : "place it"}
              </button>
            )}
          </PlotField>

          <PlotField
            label="Impact frame" icon="⧗"
            value={impactF != null ? `f${impactF}` : null}
            hint={impactF != null
              ? `motion points f${winLo}–f${winHi}`
              : "Set this and the motion points appear."}
          >
            <button type="button" className="ghost small"
                    style={{ width: "100%" }}
                    disabled={teeCurrent == null}
                    onClick={() => {
                      setImpactFrame(teeCurrent);
                      // AND TAKE THE TEE MAP THERE. Set from the raw
                      // tab, the tee map is still parked wherever it
                      // was last looked at -- which is not the frame
                      // just chosen, and is the one place the operator
                      // is about to go to check the choice.
                      if (cam === "raw") setTeeViewFrame(teeCurrent);
                    }}
                    title={cam === "raw"
                      ? "Use the tee frame the raw player is stopped on. Pause it on the strike, then press this."
                      : "Use the frame the tee view is showing. Step to the strike with the frame buttons on the picture, then press this."}>
              set to current{teeCurrent != null ? ` (f${teeCurrent})` : ""}
              {cam === "raw" && teeCurrent == null ? " — pause first" : ""}
            </button>
            <button type="button" className="ghost small"
                    style={{ width: "100%", marginTop: 3,
                             color: "var(--danger)" }}
                    disabled={Object.keys(marks).length === 0}
                    onClick={clearAllMarks}
                    title="Remove every plotted tee point for this swing.">
              ✕ clear tee tracer ({Object.keys(marks).length})
            </button>
          </PlotField>

          <PlotField
            label="Ball landing spot" icon="◍"
            value={landing ? `${landing.x}, ${landing.y}` : null}
            accent="#e2e8f0"
            hint={!row.green_filename
              ? "no green camera on this upload"
              : landing
                ? "Drag the ball on the green view."
                : "Click the green view where it came down."}
          >
            {!landing && row.green_filename && (
              <button type="button"
                      className={placeOnGreen === "landing"
                        ? "small" : "ghost small"}
                      style={{ width: "100%" }}
                      onClick={() => {
                        setCam("green");
                        setPlaceOnGreen((v) => (v === "landing"
                          ? null : "landing"));
                      }}>
                {placeOnGreen === "landing"
                  ? "click the green view…" : "place it"}
              </button>
            )}
          </PlotField>

          <PlotField
            label="Landing frame" icon="⧖"
            value={landing ? `f${landing.frame}` : null}
            hint={cometPoints.length > 1
              ? `${cometPoints.length} descent points`
              : "The frame it comes down on."}
          >
            <button type="button" className="ghost small"
                    style={{ width: "100%" }}
                    disabled={greenCurrent == null || !landing
                      || greenCurrent === landing.frame}
                    onClick={() => {
                      setLandingFrame(greenCurrent);
                      if (cam === "raw") setGreenViewFrame(greenCurrent);
                    }}
                    title={!landing
                      ? "Mark the ball landing spot first — this moves that point to another frame, and there is nothing to move without it."
                      : (cam === "raw"
                        ? "Use the green frame the raw player is stopped on. Pause it on the bounce, then press this."
                        : "Use the frame the green view is showing.")}>
              set to current{greenCurrent != null ? ` (f${greenCurrent})` : ""}
              {cam === "raw" && greenCurrent == null ? " — pause first" : ""}
            </button>
            <button type="button" className="ghost small"
                    style={{ width: "100%", marginTop: 3,
                             color: "var(--danger)" }}
                    disabled={!cometPoints.length}
                    onClick={() => { setGreenMarks({}); setCometReason(null); }}
                    title="Remove every plotted descent point for this swing.">
              ✕ clear descent ({cometPoints.length})
            </button>
          </PlotField>

          <PlotField
            label="Flag stick" icon="⚑"
            value={pinGreen ? `${pinGreen.x}, ${pinGreen.y}` : null}
            accent="#ef4444"
            hint={pinTee
              ? `tee frame ${pinTee.x}, ${pinTee.y} — belongs to the hole`
              : "Belongs to the hole; it stays put from swing to swing."}
          >
            {!pinGreen && (
              <button type="button"
                      className={placeOnGreen === "pin" ? "small" : "ghost small"}
                      style={{ width: "100%" }}
                      disabled={!row.green_filename}
                      onClick={() => {
                        setCam("green");
                        setPlaceOnGreen((v) => (v === "pin" ? null : "pin"));
                      }}
                      title="Click the BASE of the flagstick on the green view.">
                {placeOnGreen === "pin"
                  ? "click the flag base…" : "place it on green"}
              </button>
            )}
            {pinNote && (
              <div className="tiny" style={{ marginTop: 3, color: "#fbbf24" }}>
                {pinNote}
              </div>
            )}
          </PlotField>
        </div>
        </div>

        {/* The legend is a dozen lines of prose that was pushing the map
            up the screen on every open, long after the operator had read
            it once. Collapsed by default; the map gets the height. */}
        <details style={{ marginTop: 6 }}>
        <summary className="tiny muted" style={{ cursor: "pointer" }}>
          How this map works (legend &amp; shortcuts)
        </summary>
        <div className="tiny muted" style={{ marginTop: 4 }}>
          {cam === "tracer" && (
            <p style={{ margin: "0 0 6px" }}>
              <b>The tracer&apos;s line.</b> Solid green is the tracked
              ball — measured, and not editable here. The dashed cyan is
              the model&apos;s continuation, and it has one handle: the
              pink one, where the ball finishes IN THIS PICTURE. Drag it
              and the whole arc re-solves to reach it. It is the tee
              view only and does not move the landing marked on the
              green camera, and with no landing marked anywhere it
              starts as a GUESS — the measured flight carried on to
              where it comes back down to the height the ball was last
              seen at — so there is always an arc to look at and
              something to take hold of.
              What you see is what the renderer draws: the curve
              comes back from the same function that draws the clip,
              and it is the tracked ball&apos;s own parabola carried on
              to the landing, so it leaves the green track without a
              corner however far the end is dragged. Save &amp; close
              re-renders with it.
            </p>
          )}
          {cam === "green" && (
            <p style={{ margin: "0 0 6px" }}>
              <b>Green camera.</b> The picture is the frame the produced
              clip ends on, or the landing frame once one is marked; the
              dots are every scrap of motion in the green window. Click
              the dot where the ball touches down and the ball&apos;s
              descent is walked backwards from it — the pale path drawn
              head-to-tail is exactly what produce draws as a comet.
              Click the same dot again to unmark it. No path found means
              no comet: the reason is printed above. Save &amp; close
              keeps the landing and burns the comet into the green half
              of the clip.
            </p>
          )}
          The green line is the swing&apos;s CURRENT saved tracer path —
          where the rendered tracer actually sits. Green dots are already
          in the saved ball track; amber are unused detections. Click a
          dot to add the ball at exactly that spot for that frame; a
          different dot on the same frame replaces the pick. The green
          ringed green marker is where the tracer line STARTS (the ball at
          impact) — hit <b>set tracer start</b> and click the map to move
          it. The green
          dots ARE the produced tracer&apos;s points — <b>click one to
          remove it</b> (it turns into a red dashed ring; click again to
          put it back). Alt-click or right-click removes without needing
          an exact hit, the per-frame list below removes by frame number,
          and <b>Clear all</b> drops every point at once. Zoom past
          {" "}{DENSE_DOT_ZOOM}× to reveal the denser candidate layer.
          Save &amp; close re-renders the tracer with the changes (no AI
          calls) and updates the produced clip.
        </div>
        </details>

        {cam === "tee" && Object.keys(marks).length > 0 && (
          <div
            className="tiny"
            style={{
              marginTop: 4, display: "flex", flexWrap: "wrap",
              gap: 4, alignItems: "center",
            }}
          >
            <span className="muted">plotted:</span>
            {Object.keys(marks)
              .map((f) => parseInt(f, 10))
              .sort((a, b) => a - b)
              .map((f) => (
                <button
                  key={`mk-${f}`}
                  type="button"
                  className="ghost small"
                    onClick={() =>
                    setMarks((m) => {
                      const next = { ...m };
                      delete next[f];
                      return next;
                    })
                  }
                  title={`Remove the plotted point on frame ${f}`}
                  style={{
                    width: "auto", padding: "0 6px", lineHeight: "18px",
                    fontSize: 11,
                  }}
                >
                  f{f} ✕
                </button>
              ))}
          </div>
        )}

      </div>
    </div>
  );
}

/**
 * Edit wizard for single-swing uploads.
 *
 * On mount we auto-detect handedness / address / impact / ball /
 * ROI / target via the /auto-detect endpoint. Every field on the
 * right is then click-to-edit:
 *   - Handedness: Right ⇄ Left toggle.
 *   - Address / Impact frame: scrub through the upload's frames
 *     with ±1 / ±10 step buttons.
 *   - Resting ball: drag a green dot on the address frame.
 *   - Detection area: drag/resize a green rectangle on the address
 *     frame.
 *   - Target: drag a red flag on the address frame.
 *
 * Each editor has a local Apply button that commits the change to
 * the wizard's draft. The outer Save button at the bottom is still
 * a stub — wiring the draft back to the production pipeline lands
 * next.
 */
function EditWizard({
  row, adminPassword, onClose, onSaved, onProducing, onProduceError,
  // ONE CLIP AT A TIME, when the operator asked for one clip. Opened
  // from a produced clip's own ✎ Edit, `focusClipId` names the clip they
  // were looking at: the wizard lands on that clip's swing and hides the
  // swing selector, because a picker that offers the other swings is
  // exactly the thing they had already navigated past.
  focusClipId = null,
  // Opened from ＋ Add clip: seed a new blank swing on mount and land on
  // it. Same `addSwing` the selector bar calls -- this only spares the
  // operator having to go and find the button inside the wizard.
  startNewSwing = false,
}) {
  // Hydrate from whatever was already persisted: only auto-detect on
  // the very first Edit. Subsequent re-opens skip the AI call and
  // pre-fill the wizard from row.edit_metrics.
  const saved = row?.edit_metrics || null;

  const [draft, setDraft] = useState(null);
  const [frameDims, setFrameDims] = useState({
    width: null, height: null, totalFrames: null,
  });
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(null);
  // 'metrics' = step 1 (handedness/frames/ball/ROI/target);
  // 'tracer'  = step 2 (rendered tracer + per-frame ball editor).
  const [step, setStep] = useState("metrics");
  const [tracer, setTracer] = useState(null); // { url, frames }
  const [renderingTracer, setRenderingTracer] = useState(false);
  // True only between the click and the server accepting the job — the
  // produce itself runs long after this component is gone.
  const [producing, setProducing] = useState(false);
  const [tracerError, setTracerError] = useState(null);
  // start|impact|end signature the current tracer was rendered against.
  // If Step 1 edits any of these frames, Next re-renders instead of
  // reusing the now-stale ball track.
  const [renderedFrameSig, setRenderedFrameSig] = useState(null);
  const [tracerStats, setTracerStats] = useState(null); // {engine,n_points,n_candidates}
  const [finalUrl, setFinalUrl] = useState(null);
  const [finalizing, setFinalizing] = useState(false);
  const [finalError, setFinalError] = useState(null);
  const [committing, setCommitting] = useState(false);
  // Queued per-frame ball edits, hoisted from TracerStep so Step 3
  // can show a red note and pass them to /render-tracer-fast.
  const [manualPositions, setManualPositions] = useState({});
  // Multi-swing state. swings = [{idx, start_frame, end_frame, ...}, ...]
  // selectedSwing = index of the swing currently being edited in the
  // wizard. Both unused for single-swing rows.
  const isMulti = row?.swing_count === "multiple";
  const [swings, setSwings] = useState([]);
  // WHICH SWING TO OPEN ON, decided before the first render rather than
  // in an effect afterwards -- hydration reads this to pick the draft it
  // fills in, so a later correction would show the wrong clip's numbers
  // for a beat and re-fire the auto-render on the wrong swing.
  const [selectedSwing, setSelectedSwing] = useState(() => {
    if (focusClipId == null) return 0;
    const arr = row?.edit_metrics?.swings || [];
    const p = arr.findIndex(
      (s) => s?.clip_id != null && s.clip_id === focusClipId);
    return p >= 0 ? p : 0;
  });
  // Single-clip edit: no selector bar, and Produce is about this swing.
  // ONE CLIP ON SCREEN, whichever way the wizard was opened for one.
  // ＋ Add is the clearer case: an operator who asked to add a clip is
  // shown a blank one, not a rank of eleven existing swings -- three of
  // them duplicates of each other -- with the new one hidden off the
  // right-hand end of a scrollbar.
  const soloClip = focusClipId != null || startNewSwing;
  // Mirror of selectedSwing readable inside async callbacks without
  // re-creating them, so a render that finishes after the operator
  // switched tabs only updates the display if they're still on that swing.
  const selectedSwingRef = useRef(selectedSwing);
  useEffect(() => { selectedSwingRef.current = selectedSwing; }, [selectedSwing]);
  // What the last produce's tracer continuation actually did, recorded
  // on the swing by the renderer. Three numbers decide the shape --
  // duration, aim, and which model drew it -- and with none of them on
  // screen every "still looks wrong" had to be answered by guessing.
  const lastTail = (row?.edit_metrics?.swings || [])[selectedSwing]
    ?.tracer_tail || null;
  // Swings we've already kicked an auto-detect render for this session, so
  // selecting a swing fires the AI ball-track at most once (no re-fire /
  // loop when it re-renders after the result is saved).
  const autoRenderAttempted = useRef(new Set());
  // Editable on-screen graphics for Step 3. Hydrated from saved
  // edit_metrics; default yardage comes from the course's hole_yardages
  // for the chosen hole_number.
  const [graphics, setGraphics] = useState({
    player_name: "Brent Baldwin",
    hole_number: 1,
    yardage: 101,
  });
  // Last-finalized snapshot so we can tell when the graphics are
  // dirty (and thus need a re-finalize on Next).
  const [finalizedGraphics, setFinalizedGraphics] = useState(null);

  function applySaved(s) {
    setDraft({
      handedness: s.handedness || "right",
      addressFrame: s.address_frame ?? 0,
      addressImageUrl: s.address_image_url || null,
      impactFrame: s.impact_frame ?? 0,
      startFrame: s.start_frame ?? null,
      endFrame: s.end_frame ?? null,
      cutFrame: s.cut_frame ?? null,
      ball: s.ball || null,
      ballManual: !!s.ball_manual,
      landingFrame: s.landing_frame ?? null,
      landingSpot: s.landing_spot || null,
      roi: s.roi || null,
      target: s.target || null,
    });
    // ALWAYS reset the tracer to THIS swing's saved render — empty when the
    // swing hasn't been rendered yet. Without this, switching to an
    // un-rendered swing kept showing the PREVIOUS swing's ball-track (e.g.
    // Swing 1's frames while editing Swing 3).
    const hasSavedTracer = !!(
      s.tracer_url || (s.ball_track_frames && s.ball_track_frames.length)
    );
    setTracer({
      url: s.tracer_url || null,
      frames: s.ball_track_frames || [],
      // Classical engine's all-detections composite (green = rising-arc
      // chain, yellow = other motion candidates) — shown on Step 2.
      debugUrl: s.tracer_debug_url || null,
      // Raw-motion heatmap — total unfiltered motion (body/clouds/ball).
      rawMotionUrl: s.tracer_raw_motion_url || null,
      rawMotionArcUrl: s.tracer_raw_motion_arc_url || null,
      rawMotionFramesUrl: s.tracer_raw_motion_frames_url || null,
      // Produce's MOG2 layer-in evidence: raw heat + AI picks (yellow)
      // + MOG2 chain (white) + points added to the arc (red).
      mog2OverlayUrl: s.mog2_overlay_url || null,
      // Clickable MOG2 candidate dots — hydrated from the persisted
      // (capped) pool; a fresh classical/KNN render replaces them with
      // its full session set.
      candidates: s.cand_points || [],
      // Timed heat dots ARE persisted — by produce's MOG2 layer and by
      // wizard classical renders — so click-to-plot works on open.
      timedPoints: s.timed_points || [],
    });
    // A saved tracer corresponds to the saved start/impact/end frames;
    // seed the signature so Next only re-renders after a real edit. No
    // saved tracer → null so the first Next renders fresh.
    setRenderedFrameSig(
      hasSavedTracer
        ? frameSig({
            startFrame: s.start_frame ?? null,
            impactFrame: s.impact_frame ?? 0,
            endFrame: s.end_frame ?? null,
          })
        : null,
    );
    if (s.finalized_video_url) setFinalUrl(s.finalized_video_url);
    const hole = s.finalized_hole_number ?? 1;
    const courseYards = row?.course_hole_yardages?.[String(hole)];
    const yards = s.finalized_yardage
      ?? (courseYards != null ? Number(courseYards) : 101);
    const g = {
      player_name: s.finalized_player_name || "Brent Baldwin",
      hole_number: hole,
      yardage: yards,
    };
    setGraphics(g);
    if (s.finalized_video_url) setFinalizedGraphics(g);
    // Restore any queued-but-not-yet-rendered manual ball marks so the
    // operator's plotted points survive closing + reopening the wizard.
    setManualPositions(s.pending_manual_positions || {});
  }

  useEffect(() => {
    if (!row) return;

    // THE WIZARD NEVER MAKES THE OPERATOR WAIT. It used to open into a
    // spinner -- "Detecting swings (audio + motion)…" on a multi-swing
    // row, "waiting for upload-time auto-detect" on a single -- and the
    // second could poll for two minutes before giving up. On a course
    // link that is dead time in front of a golfer, and the numbers it
    // was waiting for are exactly the ones the operator is about to
    // overrule by hand. So: if edit_metrics already has something, use
    // it; otherwise open on the first frame with nothing filled in and
    // let the operator work.
    const dims = {
      width: saved?.frame_width ?? row.tee_width ?? null,
      height: saved?.frame_height ?? row.tee_height ?? null,
      totalFrames: row.tee_nb_frames || null,
    };

    if (isMulti) {
      const cached = saved?.swings;
      if (Array.isArray(cached) && cached.length > 0) {
        setSwings(cached);
        applySaved(cached[selectedSwing] || cached[0] || {});
        setFrameDims(dims);
        return;
      }
      // One empty swing, ready to be filled in. No detect call.
      setSwings([{ idx: 0, fps: row.tee_fps || 30 }]);
      setFrameDims({
        width: row.tee_width || null,
        height: row.tee_height || null,
        totalFrames: row.tee_nb_frames || null,
      });
      applySaved({});
      return;
    }

    if (saved && (saved.address_frame != null || saved.ball)) {
      applySaved(saved);
      setFrameDims(dims);
      return;
    }

    applySaved({});
    setFrameDims({
      width: row.tee_width || null,
      height: row.tee_height || null,
      totalFrames: row.tee_nb_frames || null,
    });
  }, [row, adminPassword]);  // eslint-disable-line react-hooks/exhaustive-deps

  // ＋ Add clip: seed the new swing once the existing ones are loaded.
  // It has to wait for hydration -- `addSwing` appends to `swings`, and
  // running it against the empty initial array would throw away every
  // swing already on the row.
  const seededNewSwing = useRef(false);
  useEffect(() => {
    if (!startNewSwing || seededNewSwing.current) return;
    if (!isMulti || swings.length === 0) return;
    seededNewSwing.current = true;
    addSwing();
  }, [startNewSwing, isMulti, swings.length]);  // eslint-disable-line react-hooks/exhaustive-deps

  // Multi-swing: re-hydrate the draft whenever the operator picks a
  // different swing from the selector bar.
  useEffect(() => {
    if (!isMulti) return;
    const sw = swings[selectedSwing];
    if (!sw) return;
    applySaved(sw);

    // Step 3 shows the produced clip for the SELECTED swing. applySaved
    // only sets finalUrl when the swing has a wizard-finalized video,
    // so without this an unfinalized swing would keep showing the
    // previous swing's clip. Prefer the wizard-finalized URL; fall back
    // to the auto-produced clip matched by swing order; else clear it.
    const producedForSwing = row.produced_clips?.[selectedSwing]?.video_url || null;
    setFinalUrl(sw.finalized_video_url || producedForSwing || null);

    // Auto-detect: the first time a swing is opened with no rendered
    // ball-track, kick off the AI tracer for it so the operator lands on
    // plotted points for THIS swing instead of an empty (or previous
    // swing's) track. Guarded to fire at most once per swing per session.
    // Only on Step 2 (the tracer step) — firing on Step 1 would make the
    // subsequent Step 1 → Next reuse this render and ignore the operator's
    // address/impact/ball edits.
    const hasTracer = !!(sw.ball_track_frames && sw.ball_track_frames.length);
    if (
      step === "tracer" &&
      !hasTracer &&
      !autoRenderAttempted.current.has(selectedSwing) &&
      !renderingTracer
    ) {
      autoRenderAttempted.current.add(selectedSwing);
      renderTracerForSwing(selectedSwing);
    }

    // /detect-swings only returns frame indices, not JPGs. The
    // preview shows the address frame; lazy-fetch it on first
    // selection of each swing, then cache the URL back into
    // edit_metrics.swings so re-selecting is instant.
    if (!sw.address_image_url && sw.address_frame != null) {
      api
        .getLongUploadFrame(adminPassword, row.id, sw.address_frame)
        .then((data) => {
          if (!data?.image_url) return;
          setDraft((d) => ({ ...d, addressImageUrl: data.image_url }));
          setSwings((prev) => {
            const next = prev.map((s, i) =>
              i === selectedSwing
                ? { ...s, address_image_url: data.image_url }
                : s
            );
            api
              .saveEditMetrics(adminPassword, row.id, { swings: next })
              .catch((e) => console.warn("cache addr url failed", e));
            return next;
          });
        })
        .catch((e) => console.warn("addr frame fetch failed", e));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedSwing, swings.length]);

  // Auto-save queued manual ball marks (debounced) so plotted points
  // survive closing the wizard before a re-render. Persisted under
  // pending_manual_positions and restored by applySaved on reopen.
  const manualSaveMounted = useRef(false);
  useEffect(() => {
    if (!row) return;
    // Skip the first run so we don't immediately re-save what we just
    // hydrated on open.
    if (!manualSaveMounted.current) {
      manualSaveMounted.current = true;
      return;
    }
    const t = setTimeout(() => {
      persistPatch({ pending_manual_positions: manualPositions });
    }, 700);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [manualPositions]);

  if (!row) return null;

  async function persistPatch(patch) {
    try {
      if (isMulti) {
        // Multi-swing: each patch is for the currently-selected swing.
        // Merge into swings[selectedSwing] and persist the whole array.
        const next = swings.map((sw, i) =>
          i === selectedSwing ? { ...sw, ...patch } : sw
        );
        setSwings(next);
        const r = await api.saveEditMetrics(adminPassword, row.id, { swings: next });
        onSaved?.(r);
        return;
      }
      const r = await api.saveEditMetrics(adminPassword, row.id, patch);
      onSaved?.(r);
    } catch (e) {
      console.warn("save metrics failed", e);
    }
  }

  async function deleteSwing(index) {
    // Drop a false-positive swing the detector picked up. Removes it
    // from the swing list and persists. Re-hydrates the draft for
    // whatever swing ends up selected afterward.
    if (!isMulti) return;
    if (swings.length <= 1) {
      window.alert(
        "Can't delete the only swing. Close the wizard and use Delete " +
        "on the production card to remove the whole upload."
      );
      return;
    }
    if (!window.confirm(`Delete Swing ${index + 1}? It won't be produced.`)) {
      return;
    }
    const next = swings.filter((_, i) => i !== index);
    setSwings(next);
    // Keep the selection pointing at a sane swing.
    let nextSelected = selectedSwing;
    if (index === selectedSwing) nextSelected = Math.max(0, index - 1);
    else if (index < selectedSwing) nextSelected = selectedSwing - 1;
    setSelectedSwing(nextSelected);
    applySaved(next[nextSelected] || next[0] || {});
    try {
      const r = await api.saveEditMetrics(adminPassword, row.id, { swings: next });
      onSaved?.(r);
    } catch (e) {
      setError(e.message);
    }
  }

  async function addSwing() {
    // Add a swing the auto-detector missed. Seeds a blank window
    // spanning the clip; the operator then sets start/address/impact/
    // end (and cut) via the frame pickers. Selects the new swing.
    if (!isMulti) return;
    const newIdx = swings.length
      ? Math.max(...swings.map((s) => s.idx ?? 0)) + 1
      : 0;
    const newSwing = {
      idx: newIdx,
      start_frame: 0,
      end_frame: row.tee_nb_frames ? row.tee_nb_frames - 1 : null,
      address_frame: 0,
      impact_frame: 0,
      fps: row.tee_fps || 30,
    };
    const next = [...swings, newSwing];
    setSwings(next);
    const nextSelected = next.length - 1;
    setSelectedSwing(nextSelected);
    applySaved(newSwing);
    setStep("metrics");
    try {
      const r = await api.saveEditMetrics(adminPassword, row.id, { swings: next });
      onSaved?.(r);
    } catch (e) {
      setError(e.message);
    }
  }

  async function renderTracerForSwing(swIndex) {
    // Full AI ball-track render for ONE swing, addressed by index so the
    // result is persisted to the right swing even if the operator clicked
    // to another tab while it ran. Used for auto-detect on swing select.
    const sw = swings[swIndex];
    if (!sw || renderingTracer) return;
    setRenderingTracer(true);
    setTracerError(null);
    try {
      const out = await api.renderWizardTracer(adminPassword, row.id, {
        handedness: sw.handedness || draft.handedness || "right",
        impact_frame: sw.impact_frame,
        start_frame: sw.start_frame ?? null,
        end_frame: sw.end_frame ?? null,
        ball_at_rest: sw.ball || null,
        ball_manual: !!sw.ball_manual,
      });
      const frames = out.ball_track_frames || [];
      // Only reflect into the visible tracer if we're still on this swing.
      if (selectedSwingRef.current === swIndex) {
        setTracer({
          url: out.tracer_url, frames,
          debugUrl: out.debug_url || null,
          rawMotionUrl: out.raw_motion_url || null,
          rawMotionArcUrl: out.raw_motion_arc_url || null,
          rawMotionFramesUrl: out.raw_motion_frames_url || null,
          mog2OverlayUrl: out.mog2_overlay_url || null,
          candidates: out.candidates || [],
          timedPoints: out.timed_points || [],
        });
        setRenderedFrameSig(frameSig(draft));
        setTracerStats({
          engine: out.engine || "ai",
          n_points: out.n_points,
          n_candidates: out.n_candidates,
          n_backfilled: out.n_backfilled,
          n_ai_anchors: out.n_ai_anchors ?? null,
          rest_lock: out.rest_lock || null,
          anchor_check: out.anchor_check || out.mog2_stats?.anchor_check || null,
        });
      }
      // Persist to the captured swing index regardless of current tab.
      setSwings((prev) => {
        const next = prev.map((s, i) =>
          i === swIndex
            ? {
                ...s,
                tracer_url: out.tracer_url,
                ball_track_frames: frames,
                tracer_engine: out.engine || "ai",
                tracer_debug_url: out.debug_url || null,
                tracer_raw_motion_url: out.raw_motion_url || null,
                tracer_raw_motion_arc_url: out.raw_motion_arc_url || null,
                tracer_raw_motion_frames_url: out.raw_motion_frames_url || null,
                mog2_overlay_url: out.mog2_overlay_url || null,
                mog2_stats: out.mog2_stats || null,
                timed_points: out.timed_points || [],
                cand_points: (out.candidates || []).slice(0, 1500),
                ...(out.ball_at_rest && !out.ball_manual
                  ? { ball: out.ball_at_rest }
                  : {}),
              }
            : s
        );
        api
          .saveEditMetrics(adminPassword, row.id, { swings: next })
          .then(() => onSaved?.())
          .catch((e) => console.warn("save tracer failed", e));
        return next;
      });
    } catch (e) {
      setTracerError(sanitizeErr(e.message));
    } finally {
      setRenderingTracer(false);
    }
  }

  async function persistDraftMetrics() {
    await persistPatch({
      handedness: draft.handedness,
      address_frame: draft.addressFrame,
      address_image_url: draft.addressImageUrl,
      impact_frame: draft.impactFrame,
      ball: draft.ball,
      // PRESSING PRODUCE IS AN ENDORSEMENT. ball_manual is what stops a
      // later produce re-detecting over the top, and it used to be set
      // only by the ball editor's Done button -- so a ball that looked
      // right and was left alone got overwritten on the next run. If the
      // operator ships this ball, they mean it.
      ball_manual: draft.ball ? true : !!draft.ballManual,
      landing_frame: draft.landingFrame ?? null,
      landing_spot: draft.landingSpot ?? null,
      roi: draft.roi,
      target: draft.target,
    });
  }

  async function handleProduce() {
    // Save what the operator set, hand the ball + impact frame to the
    // server, and close. Deliberately NOT awaited to completion: the job
    // takes minutes and belongs on the production card, not in a modal
    // the operator has to sit in front of.
    if (!draft?.ball || draft.impactFrame == null) return;
    const payload = {
      ball: [draft.ball.x, draft.ball.y],
      impact_frame: draft.impactFrame,
      // The landing, on the GREEN camera. The clip ends a beat after
      // it, and the tracer is drawn to arrive at the spot — so these
      // two travel together and neither is much use alone.
      landing_frame: draft.landingFrame ?? null,
      landing_spot: draft.landingSpot
        ? [draft.landingSpot.x, draft.landingSpot.y]
        : null,
      // THIS CLIP, NOT THIS UPLOAD. The wizard opened on one clip (✎
      // Edit) or on a new one (＋ Add), so Produce must leave the
      // upload's other clips alone. Without it, adding a clip to an
      // upload that had ten left it with one.
      solo: soloClip,
      swing_idx: swings[selectedSwing]?.idx ?? selectedSwing,
    };
    setProducing(true);
    // GREY THE CARD ON THE CLICK, not two round trips later...
    onProducing?.(true);
    // ...and GET OUT OF ITS WAY. This is a full-screen modal sitting on
    // top of the card, so greying the card early bought nothing while
    // the wizard was still up: the operator clicked Produce and watched
    // the modal sit there for two server round trips before it closed
    // and revealed the state that had been set all along. Dismiss on the
    // click, then save and queue in the background. Neither call needs a
    // modal to live in, and a failure belongs on the page anyway.
    onClose?.();
    try {
      await persistDraftMetrics();
      await api.wizardProduce(adminPassword, row.id, payload);
      onSaved?.();
    } catch (e) {
      // Nothing was queued, so put the card back rather than leaving it
      // greyed for a run that never started. The wizard is gone by now,
      // so the message has to go to the page.
      onProducing?.(false);
      onProduceError?.(e?.message || String(e));
    }
  }

  async function handleNext() {
    // Persist current draft, then either reuse the cached tracer or
    // render a fresh one.
    await persistDraftMetrics();
    // NEVER re-render an unchanged swing. If it already has a rendered
    // tracer OR plotted ball points, just advance — re-rendering would
    // wipe the existing points (the operator's plots AND the ones carried
    // over from the original production). Only render when:
    //   - there's no existing trace at all (first time through), or
    //   - the start / impact / end frames changed (points are anchored to
    //     the old window, so they're stale).
    // Switching tracer engines does NOT re-render on Next — that's what
    // the explicit ↻ Render button is for.
    // Crucially the reuse no longer requires a tracer_url: a swing can
    // carry ball_track_frames without a (still-valid) rendered video, and
    // those points must survive Next.
    const framesChanged =
      renderedFrameSig !== null && frameSig(draft) !== renderedFrameSig;
    const hasExistingTrace =
      !!(tracer?.url) || (tracer?.frames?.length || 0) > 0;
    if (hasExistingTrace && !framesChanged) {
      setStep("tracer");
      return;
    }
    // A frame edit invalidates any queued manual marks — clear them so
    // they aren't baked into the fresh render at the wrong positions.
    if (framesChanged && Object.keys(manualPositions).length > 0) {
      setManualPositions({});
    }
    await renderFreshTracer();
  }

  async function handleForceRender() {
    // Explicit "↻ Render" on Step 1: always render fresh with the
    // currently-selected engine (the only way an engine switch takes
    // effect now that Next reuses the existing track). Warns before
    // discarding manually plotted points.
    const nManual =
      Object.keys(manualPositions).length +
      (tracer?.frames || []).filter((f) => f && f.manual).length;
    if (
      nManual > 0 &&
      !window.confirm(
        `Re-rendering replaces the current ball track and discards ` +
          `${nManual} manually plotted point(s). Continue?`
      )
    ) {
      return;
    }
    await persistDraftMetrics();
    if (Object.keys(manualPositions).length > 0) setManualPositions({});
    await renderFreshTracer();
  }

  async function renderFreshTracer() {
    setRenderingTracer(true);
    setTracerError(null);
    try {
      const out = await api.renderWizardTracer(adminPassword, row.id, {
        handedness: draft.handedness,
        impact_frame: draft.impactFrame,
        start_frame: draft.startFrame ?? null,
        end_frame: draft.endFrame ?? null,
        ball_at_rest: draft.ball,
        ball_manual: !!draft.ballManual,
      });
      setTracer({
        url: out.tracer_url,
        frames: out.ball_track_frames || [],
        debugUrl: out.debug_url || null,
        rawMotionUrl: out.raw_motion_url || null,
        rawMotionArcUrl: out.raw_motion_arc_url || null,
        rawMotionFramesUrl: out.raw_motion_frames_url || null,
        mog2OverlayUrl: out.mog2_overlay_url || null,
        candidates: out.candidates || [],
        timedPoints: out.timed_points || [],
      });
      // Adopt the flight-derived rest position (never over an operator-set
      // one) so the Step-2 rest card starts where the render anchored.
      if (out.ball_at_rest && !out.ball_manual) {
        setDraft((d) => ({ ...d, ball: out.ball_at_rest, ballManual: false }));
      }
      setRenderedFrameSig(frameSig(draft));
      setTracerStats({
        engine: out.engine || "ai",
        n_points: out.n_points,
        n_candidates: out.n_candidates,
        n_backfilled: out.n_backfilled,
        n_ai_anchors: out.n_ai_anchors ?? null,
        rest_lock: out.rest_lock || null,
        anchor_check: out.anchor_check || out.mog2_stats?.anchor_check || null,
      });
      // Cache the run into the swing so re-opens hydrate the tracer
      // instead of re-running. Records which engine produced it. The
      // backend also writes these to top-level edit_metrics; multi-swing
      // rows additionally need them inside swings[selectedSwing].
      await persistPatch({
        tracer_url: out.tracer_url,
        ball_track_frames: out.ball_track_frames || [],
        tracer_engine: out.engine || "ai",
        tracer_debug_url: out.debug_url || null,
        tracer_raw_motion_url: out.raw_motion_url || null,
        tracer_raw_motion_arc_url: out.raw_motion_arc_url || null,
        tracer_raw_motion_frames_url: out.raw_motion_frames_url || null,
        mog2_overlay_url: out.mog2_overlay_url || null,
        mog2_stats: out.mog2_stats || null,
        timed_points: out.timed_points || [],
        cand_points: (out.candidates || []).slice(0, 1500),
        ...(out.ball_at_rest && !out.ball_manual
          ? { ball: out.ball_at_rest }
          : {}),
      });
      onSaved?.();
      setStep("tracer");
    } catch (e) {
      setTracerError(sanitizeErr(e.message));
    } finally {
      setRenderingTracer(false);
    }
  }

  async function handleAdvanceToFinalize() {
    // Step 2 → Step 3. Reuse the cached final video when present;
    // otherwise apply the intro overlay on top of the rendered tracer.
    // Manual ball-position edits are NOT applied here — Step 3 Next
    // does that (cv2 fast render only, never Claude calls).
    if (finalUrl) {
      setStep("finalize");
      return;
    }
    if (!tracer?.url) {
      setFinalError("No tracer video yet — re-run Step 2 first.");
      setStep("finalize");
      return;
    }
    setFinalizing(true);
    setFinalError(null);
    try {
      const out = await api.finalizeWizardVideo(adminPassword, row.id, {
        player_name: graphics.player_name,
        hole_number: Number(graphics.hole_number) || 1,
        yardage: Number(graphics.yardage) || null,
        start_frame: draft.startFrame ?? null,
        end_frame: draft.endFrame ?? null,
        cut_frame: draft.cutFrame ?? null,
        // Per-swing final file on multi-swing rows so finalizing one
        // swing can't replace the video behind another swing's clip.
        ...(isMulti
          ? { swing: swings[selectedSwing]?.idx ?? selectedSwing }
          : {}),
      });
      setFinalUrl(out.final_video_url);
      setFinalizedGraphics({ ...graphics });
      // Cache the finalized output per-swing so re-opens land on
      // Step 3 with the saved final video pre-loaded.
      await persistPatch({
        finalized_video_url: out.final_video_url,
        finalized_hole_number: Number(graphics.hole_number) || 1,
        finalized_yardage: Number(graphics.yardage) || null,
        finalized_player_name: graphics.player_name,
      });
      onSaved?.();
    } catch (e) {
      setFinalError(e.message);
    } finally {
      setFinalizing(false);
      setStep("finalize");
    }
  }

  function graphicsDirty() {
    if (!finalizedGraphics) return true; // never finalized yet
    return (
      (finalizedGraphics.player_name || "") !== (graphics.player_name || "")
      || Number(finalizedGraphics.hole_number) !== Number(graphics.hole_number)
      || Number(finalizedGraphics.yardage) !== Number(graphics.yardage)
    );
  }

  async function handleSaveToProduced() {
    // Step 3 Produce. If the operator queued ball edits on Step 2,
    // bake them in first via the cv2-only fast render (no Claude
    // calls). If the on-screen graphics changed, re-finalize with
    // the new values. Either way commit the clip to Produced Clips
    // so it's broadcastable right after the run. Leaves the wizard
    // open so the operator can review the result; Finish closes.
    setCommitting(true);
    setFinalError(null);
    try {
      const overrides = Object.entries(manualPositions).map(
        ([f, p]) => ({ frame: parseInt(f, 10), x: p.x, y: p.y })
      );
      if (overrides.length > 0) {
        setFinalizing(true);
        try {
          const fast = await api.renderWizardTracerFast(adminPassword, row.id, {
            manual_positions: overrides,
          });
          setTracer((t) => ({
            url: fast.tracer_url,
            frames: fast.ball_track_frames || [],
            debugUrl: t?.debugUrl || null,
            rawMotionUrl: t?.rawMotionUrl || null,
            rawMotionArcUrl: t?.rawMotionArcUrl || null,
            rawMotionFramesUrl: t?.rawMotionFramesUrl || null,
            mog2OverlayUrl: t?.mog2OverlayUrl || null,
          }));
          setManualPositions({});
          // Persist the merged tracer (operator marks baked in) per
          // swing so re-opens skip the AI re-run AND keep the manual
          // anchor points.
          await persistPatch({
            tracer_url: fast.tracer_url,
            ball_track_frames: fast.ball_track_frames || [],
          });
        } finally {
          setFinalizing(false);
        }
      }
      // Always re-finalize on Produce. Skipping when nothing in
      // the local state changed meant a cached finalUrl from before
      // a backend fix would never get refreshed; the explicit click
      // is the operator's signal to re-render regardless.
      {
        setFinalizing(true);
        try {
          const fin = await api.finalizeWizardVideo(adminPassword, row.id, {
            player_name: graphics.player_name,
            hole_number: Number(graphics.hole_number) || 1,
            yardage: Number(graphics.yardage) || null,
            start_frame: draft.startFrame ?? null,
            end_frame: draft.endFrame ?? null,
            cut_frame: draft.cutFrame ?? null,
            ...(isMulti
              ? { swing: swings[selectedSwing]?.idx ?? selectedSwing }
              : {}),
          });
          setFinalUrl(fin.final_video_url);
          setFinalizedGraphics({ ...graphics });
          await persistPatch({
            finalized_video_url: fin.final_video_url,
            finalized_hole_number: Number(graphics.hole_number) || 1,
            finalized_yardage: Number(graphics.yardage) || null,
            finalized_player_name: graphics.player_name,
          });
        } finally {
          setFinalizing(false);
        }
      }
      // Multi-swing: commit into THIS swing's produced clip. Prefer the
      // clip id recorded on the swing (survives clip deletions that
      // shift positions); fall back to position — without clip_id the
      // backend updates the upload's most recent clip, i.e. some other
      // swing's video.
      const _clipId = isMulti
        ? swings[selectedSwing]?.clip_id
          ?? row.produced_clips?.[selectedSwing]?.id
          ?? null
        : null;
      const _committed = await api.commitWizardClip(
        adminPassword, row.id,
        _clipId != null ? { clip_id: _clipId } : {},
      );
      if (isMulti && _committed?.clip_id != null) {
        await persistPatch({ clip_id: _committed.clip_id });
      }
      onSaved?.();
    } catch (e) {
      setFinalError(e.message);
    } finally {
      setCommitting(false);
    }
  }

  const fw = frameDims.width;
  const fh = frameDims.height;
  const totalFrames = frameDims.totalFrames;

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
        zIndex: 1000, padding: 16, cursor: "zoom-out",
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="card"
        style={{
          maxWidth: "min(1400px, 98vw)", width: "100%",
          maxHeight: "96vh", height: "96vh", overflow: "hidden",
          cursor: "default", margin: 0,
          display: "flex", flexDirection: "column",
        }}
      >
        <div
          className="row"
          style={{ alignItems: "center", justifyContent: "space-between", marginBottom: 8 }}
        >
          <div>
            {/* One step now: set the ball and the impact frame, then
                Produce. The old Tracer / Final-video steps existed to pick
                between tracer engines and hand-plot a track; there is one
                engine and Debug3's pipeline plots it. */}
            <h3 style={{ margin: 0 }}>Edit wizard</h3>
            <div className="small muted">
              Upload #{row.id} · {row.course_name || `course ${row.course_id}`} ·{" "}
              {startNewSwing
                ? "new clip"
                : soloClip
                  ? "editing one clip"
                  : isMulti
                    ? `multi-swing${swings.length ? ` (${swings.length})` : ""}`
                    : "single swing"}
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
            margin: "0 0 10px",
            padding: 10,
            flex: 1,
            overflow: "auto",
            minHeight: 0,
          }}
        >
          {error && (
            <div className="err-text small">{error}</div>
          )}
          {tracerError && step === "metrics" && (
            <div className="err-text small" style={{ marginBottom: 8 }}>
              Tracer render failed: {tracerError} — hit Next to retry, or
              switch the Tracer engine below.
            </div>
          )}
          {isMulti && !soloClip && swings.length > 0 && (
            <SwingSelectorBar
              swings={swings}
              selectedSwing={selectedSwing}
              setSelectedSwing={setSelectedSwing}
              onDeleteSwing={deleteSwing}
              onAddSwing={addSwing}
            />
          )}
          {!error && draft && step === "metrics" && (
            <WizardBody
              row={row}
              adminPassword={adminPassword}
              draft={draft}
              setDraft={setDraft}
              editing={editing}
              setEditing={setEditing}
              frameW={fw}
              frameH={fh}
              totalFrames={totalFrames}
              persistPatch={persistPatch}
            />
          )}
          {!error && draft && step === "tracer" && tracerStats && (
            <div
              className="tiny"
              style={{
                marginBottom: 8, padding: "4px 8px", borderRadius: 4,
                background: "rgba(34,197,94,0.08)",
                border: "1px solid var(--border, #2a2a2a)",
              }}
            >
              Engine:{" "}
              <b>
                {tracerStats.engine === "classical"
                  ? "Classical CV (MOG2)"
                  : tracerStats.engine === "knn"
                    ? "Classical CV (KNN)"
                    : tracerStats.engine === "hybrid"
                      ? "MOG2 + AI verify"
                      : tracerStats.engine === "ai_mog2"
                        ? "AI + MOG2 trail (produce's engine)"
                        : "AI"}
              </b>
              {tracerStats.n_ai_anchors != null && (
                <> · {tracerStats.n_ai_anchors} AI anchors</>
              )}
              {tracerStats.rest_lock?.locked && (
                <> · 🔒 rest-lock @ f{(tracerStats.rest_lock.seed_frames || []).join(",")}</>
              )}
              {tracerStats.anchor_check && (
                <>
                  {" · \u2693 "}
                  {tracerStats.anchor_check.verified
                    ? `anchors verified (rest ${
                        tracerStats.anchor_check.snapped
                          ? `snapped ${tracerStats.anchor_check.snap_px}px`
                          : "exact"
                      }, impact ${
                        tracerStats.anchor_check.impact_delta >= 0 ? "+" : ""
                      }${tracerStats.anchor_check.impact_delta}f by departure)`
                    : `anchors unverified: ${tracerStats.anchor_check.reason}`}
                  {tracerStats.anchor_check.image_url && (
                    <>
                      {" "}
                      <a
                        href={tracerStats.anchor_check.image_url}
                        target="_blank"
                        rel="noreferrer"
                      >
                        film-strip
                      </a>
                    </>
                  )}
                  {tracerStats.anchor_check.image_mog2_url && (
                    <>
                      {" · "}
                      <a
                        href={tracerStats.anchor_check.image_mog2_url}
                        target="_blank"
                        rel="noreferrer"
                        title="The same anchor tiles as MOG2 frame-diff heat"
                      >
                        🔥 mog2
                      </a>
                    </>
                  )}
                </>
              )}
              {" · "}{tracerStats.n_points ?? "—"} points plotted
              {tracerStats.n_candidates != null && (
                <> · {tracerStats.n_candidates} candidates</>
              )}
              {tracerStats.n_backfilled ? (
                <> · {tracerStats.n_backfilled} backfilled</>
              ) : null}
              {" — switch the Tracer toggle on Step 1 and hit ↻ Render to compare."}
            </div>
          )}
          {!error && draft && step === "tracer" && (
            <TracerStep
              row={row}
              adminPassword={adminPassword}
              draft={draft}
              setDraft={setDraft}
              tracer={tracer}
              setTracer={setTracer}
              rendering={renderingTracer}
              setRendering={setRenderingTracer}
              error={tracerError}
              setError={setTracerError}
              frameW={fw}
              frameH={fh}
              totalFrames={totalFrames}
              onSaved={onSaved}
              persistPatch={persistPatch}
              manualPositions={manualPositions}
              setManualPositions={setManualPositions}
            />
          )}
          {!error && draft && step === "finalize" && (
            <FinalizeStep
              row={row}
              finalUrl={finalUrl}
              finalizing={finalizing}
              committing={committing}
              error={finalError}
              frameW={fw}
              frameH={fh}
              pendingEdits={Object.keys(manualPositions).length}
              graphics={graphics}
              setGraphics={setGraphics}
              graphicsDirty={graphicsDirty()}
              alreadyProduced={!!finalUrl}
              onProduce={handleSaveToProduced}
            />
          )}
        </div>

        <div className="row" style={{ gap: 8, justifyContent: "flex-end", alignItems: "center" }}>
          {step !== "metrics" && (
            <button
              type="button"
              className="ghost"
              onClick={() => setStep(step === "finalize" ? "tracer" : "metrics")}
              style={{ width: "auto", marginRight: "auto" }}
              disabled={committing}
            >
              ← Back
            </button>
          )}
          <button
            type="button"
            className="ghost"
            onClick={onClose}
            style={{ width: "auto" }}
            disabled={committing}
          >
            Cancel
          </button>
          {/* WHAT THE TRACER WILL DO, said before Produce is pressed.
              The flight time between the last tracked point and the
              landing is arithmetic when BOTH the landing frame and the
              landing spot are set -- both Pis stamp wall-clock. With
              either missing there is no clock to read, the duration
              gets estimated from the track alone, and the tail comes
              out too short and too bent. That was invisible: produce
              ran, the clip was remade, and the tracer looked untouched
              with nothing on screen explaining why. */}
          <span className="tiny" style={{
            marginRight: "auto", maxWidth: 520,
            color: (draft?.landingFrame != null && draft?.landingSpot)
              ? "#3ee37a" : "#f59e0b",
          }}>
            {/* AND WHAT THE LAST ONE DID. The shape is decided by three
                numbers -- the duration, the aim, and which model drew
                it -- and until they were on screen every "still looks
                wrong" was answered with a guess. */}
            {lastTail && (
              <span className="muted" style={{ display: "block" }}>
                last produce: {lastTail.kind || "no tail"}
                {lastTail.frames
                  ? ` · ${lastTail.frames} frames (f${lastTail.from_frame}`
                    + `→f${lastTail.to_frame})` : ""}
                {lastTail.land_frame != null
                  ? ` · clocks say it lands at tee f${lastTail.land_frame}`
                  : " · flight time ESTIMATED (no clock)"}
                {lastTail.target
                  ? ` · aimed (${lastTail.target[0]}, ${lastTail.target[1]})`
                  : ""}
                {lastTail.apex_y != null ? ` · apex y${lastTail.apex_y}` : ""}
              </span>
            )}
            {draft?.landingFrame != null && draft?.landingSpot
              ? "Tracer will fly to the landing spot, timed off the two "
                + "camera clocks."
              : draft?.target
                ? "No landing frame + spot — the tracer will aim at the "
                  + "flag with an ESTIMATED flight time. Set both landing "
                  + "fields for the real arc."
                : "No landing spot and no target — the tracer will stop "
                  + "where the ball was last detected."}
          </span>
          {/* ONE BUTTON. Stages 1-3 exist to find the ball and the impact
              frame; by the time the wizard is open the operator has found
              both by eye. So this hands those two numbers to the same
              pipeline produce uses and closes -- the work happens on the
              server and the production card shows its progress. */}
          <button
            type="button"
            disabled={producing || !draft || !draft.ball
                      || draft.impactFrame == null}
            onClick={handleProduce}
            style={{ width: "auto" }}
            title={
              !draft?.ball
                ? "Place the ball at rest first"
                : draft?.impactFrame == null
                  ? "Set the impact frame first"
                  : "Produce this swing from the ball and impact frame "
                    + "above, then close. Progress shows on the "
                    + "production card."
            }
          >
            {producing ? "Starting…" : "Produce"}
          </button>
        </div>
      </div>
    </div>
  );
}

function SwingSelectorBar({ swings, selectedSwing, setSelectedSwing, onDeleteSwing, onAddSwing }) {
  // Horizontal scroll of numbered swing chips. Sticky to the top of
  // the wizard body so the operator can switch between swings on
  // every step without losing context. Each chip shows the frame
  // window so it's obvious where in the source video that swing
  // lives, plus a × to drop a false-positive swing.
  const canDelete = swings.length > 1;
  return (
    <div
      style={{
        display: "flex",
        gap: 6,
        overflowX: "auto",
        marginBottom: 10,
        paddingBottom: 4,
        borderBottom: "1px solid var(--border, #2a2a2a)",
      }}
    >
      {swings.map((sw, i) => {
        const active = i === selectedSwing;
        return (
          <div
            key={sw.idx ?? i}
            className={active ? "" : "ghost"}
            style={{
              display: "inline-flex", alignItems: "center", gap: 4,
              padding: "4px 6px 4px 10px",
              flex: "0 0 auto", fontSize: "0.82rem",
              border: "1px solid var(--border, #2a2a2a)",
              borderRadius: 6,
              background: active ? "var(--primary-soft, rgba(34,197,94,0.12))" : "transparent",
              cursor: "pointer",
            }}
            title={`Frames ${sw.start_frame ?? "—"}–${sw.end_frame ?? "—"}`}
          >
            <button
              type="button"
              onClick={() => setSelectedSwing(i)}
              style={{
                width: "auto", padding: 0, background: "transparent",
                border: "none", color: "inherit", cursor: "pointer",
                fontWeight: active ? 600 : 400,
              }}
            >
              Swing {i + 1}
              <span className="tiny" style={{ marginLeft: 6, opacity: 0.75 }}>
                {sw.start_frame ?? "—"}–{sw.end_frame ?? "—"}
              </span>
            </button>
            {canDelete && (
              <button
                type="button"
                onClick={(e) => { e.stopPropagation(); onDeleteSwing?.(i); }}
                title={`Delete Swing ${i + 1}`}
                aria-label={`Delete Swing ${i + 1}`}
                style={{
                  width: "auto", padding: "0 4px", background: "transparent",
                  border: "none", color: "var(--err, #dc2626)",
                  cursor: "pointer", fontSize: "1rem", lineHeight: 1,
                }}
              >
                ×
              </button>
            )}
          </div>
        );
      })}
      {onAddSwing && (
        <button
          type="button"
          className="ghost"
          onClick={onAddSwing}
          title="Add a swing the detector missed"
          style={{
            width: "auto", padding: "4px 12px",
            flex: "0 0 auto", fontSize: "0.82rem",
            border: "1px dashed var(--border, #2a2a2a)",
            borderRadius: 6,
          }}
        >
          + Add swing
        </button>
      )}
    </div>
  );
}

function WizardBody({
  row, adminPassword, draft, setDraft, editing, setEditing,
  frameW, frameH, totalFrames, persistPatch,
}) {
  const hasDims = !!(frameW && frameH);

  // Frame-navigation modes ('address', 'impact') need a per-mode
  // working frame. Seed from draft and let ± step buttons mutate.
  const [navFrame, setNavFrame] = useState(null);
  const [navUrl, setNavUrl] = useState(null);
  const [navTotal, setNavTotal] = useState(totalFrames);
  // The size of the frame on screen, per camera. The green camera runs
  // at its own resolution, and a click on it means nothing without the
  // frame it was made in.
  const [navDims, setNavDims] = useState(null);
  const [navLoading, setNavLoading] = useState(false);
  // Re-detect runs a detector over the whole clip and then reloads the
  // page; the button has to hold the state itself.
  const [redetecting, setRedetecting] = useState(false);
  // Real time needs the SOURCE fps, and it differs per camera (the tee
  // runs ~50fps, the green its own rate) -- so it comes back with the
  // frame rather than being assumed.
  const [navFps, setNavFps] = useState(null);
  const [navWhich, setNavWhich] = useState("tee");
  // Real-world instant of the frame on screen, from the Pi's own stamp
  // of when its first frame was captured.
  const [navWallClock, setNavWallClock] = useState(null);
  // D3_GREEN_SEC, echoed by the backend so the wizard's default end
  // frame and produce's actual green coverage cannot drift apart.
  const [greenSeconds, setGreenSeconds] = useState(null);
  // Where produce would stop, in green frames — the server works it out
  // because the tee->green offset lives there.
  const [defaultEnd, setDefaultEnd] = useState(null);

  // MOTION ON THE GREEN, over the frames the produced clip covers plus
  // a couple of seconds. The landing frame is the hardest thing in this
  // wizard to find by stepping: the ball is a few pixels, it is in view
  // for a handful of frames, and it arrives from off-screen. Scanning
  // the window and drawing every blob at once turns that search into a
  // glance -- and answers the prior question, whether the ball landed
  // anywhere the green camera can see, without finding the frame at all.
  // THE LANDING, CARRIED ONTO THE TEE FRAME. The tracer has to finish
  // where the ball came down, and that is marked on the other camera --
  // so the green→tee mapping brings it here, and the operator sees it
  // before producing rather than after.
  const [teeLanding, setTeeLanding] = useState(null);   // {xy} | {reason}
  const [viewMap, setViewMap] = useState(null);
  const [calibrating, setCalibrating] = useState(null); // {tee, green}
  // The flagstick in GREEN pixels, remembered against the hole. Mark it
  // once and the rest of the session's swings can take it -- but it is
  // stamped with the day it was set, because pins move and a Tuesday
  // pin on Thursday's swing is worse than no pin at all.
  const [holePin, setHolePin] = useState(null);
  const [pinNote, setPinNote] = useState(null);
  // Whether this HOLE is already mapped, and how well. The calibration
  // is a property of two bolted-down cameras, so it is done once and
  // then never again -- but only if the wizard says so. Silence reads
  // as "not done", and the operator re-clicks eight pairs for nothing.
  const [viewMapInfo, setViewMapInfo] = useState(null);

  const _vmCal = viewMapInfo?.view_map;
  const calLine = _vmCal ? (
    `${viewMapInfo.course_name} · hole ${viewMapInfo.hole} mapped `
    + `${(_vmCal.calibrated_at || "").slice(0, 10)} · ${_vmCal.n_points} pairs`
    + (_vmCal.cv_px != null ? ` · ±${_vmCal.cv_px}px` : "")
  ) : null;
  // THE COMET'S PATH, PREVIEWED. Produce draws a comet on the green
  // half whenever a chain of blobs walks back from the marked landing.
  // Showing that chain here means the operator knows before producing
  // whether there will be one -- and when there is not, why.
  const [greenFlight, setGreenFlight] = useState(null);
  const [greenFlightBusy, setGreenFlightBusy] = useState(false);

  async function findGreenFlight() {
    if (greenFlightBusy) return;
    setGreenFlightBusy(true);
    try {
      const out = await api.greenFlight(adminPassword, row.id, {
        landing_frame: draft.landingFrame,
        landing_spot: draft.landingSpot
          ? [draft.landingSpot.x, draft.landingSpot.y] : null,
      });
      setGreenFlight(out);
    } catch (e) {
      setGreenFlight({ points: null, reason: e?.message || String(e) });
    } finally {
      setGreenFlightBusy(false);
    }
  }

  // Re-ask whenever the landing moves: the chain is anchored on it, so
  // a different mark is a different search.
  useEffect(() => { setGreenFlight(null); },
           [draft.landingFrame, draft.landingSpot?.x, draft.landingSpot?.y]);

  const [greenHeat, setGreenHeat] = useState(null);
  const [greenScanning, setGreenScanning] = useState(false);
  const [greenScanNote, setGreenScanNote] = useState(null);
  const [greenScanLevel, setGreenScanLevel] = useState(2);

  // Re-ask whenever the spot moves: the whole value of showing it is
  // that it tracks what the operator just clicked.
  useEffect(() => {
    const spot = draft.landingSpot;
    if (!spot) { setTeeLanding(null); return; }
    let dead = false;
    (async () => {
      try {
        const out = await api.mapLanding(adminPassword, row.id, {
          x: spot.x, y: spot.y,
          // The green click was made in the green source's own pixels,
          // and the answer is wanted in the tee source's.
          green_size: navWhich === "green" && navDims
            ? [navDims.w, navDims.h] : null,
          tee_size: frameW && frameH ? [frameW, frameH] : null,
        });
        if (!dead) setTeeLanding(out);
      } catch (e) {
        if (!dead) setTeeLanding({ reason: e?.message || String(e) });
      }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [draft.landingSpot?.x, draft.landingSpot?.y, row.id, viewMap]);

  // What the hole already knows: its mapping, and where the flag was
  // last marked. Loaded once so the Target row can offer the pin
  // without the operator asking for it.
  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const vm = await api.getViewMap(adminPassword, row.id);
        if (dead) return;
        setViewMapInfo(vm || null);
        const _pg = vm?.pin_green ?? vm?.view_map?.pin_green;
        setHolePin(_pg
          ? { pin_green: _pg,
              pin_set_at: vm?.pin_effective_at
                ?? vm?.view_map?.pin_set_at,
              pin_note: vm?.pin_note }
          : null);
      } catch { /* the Target row simply offers nothing */ }
    })();
    return () => { dead = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [row.id, viewMap]);

  // A click on the green camera in target mode: store the pin against
  // the hole and take back the tee-frame point it maps to.
  async function pickGreenPin(pt) {
    setPinNote(null);
    try {
      const out = await api.saveHolePin(adminPassword, row.id, {
        green: [pt.x, pt.y],
        green_size: navDims ? [navDims.w, navDims.h] : null,
        tee_size: frameW && frameH ? [frameW, frameH] : null,
      });
      setHolePin({ pin_green: out.pin_green, pin_set_at: out.pin_set_at });
      if (out.tee_xy) {
        const t = { x: Math.round(out.tee_xy[0]), y: Math.round(out.tee_xy[1]) };
        setDraft((d) => ({ ...d, target: t }));
        persistPatch({ target: t });
        // Straight back to the tee frame: the whole point is seeing
        // where the flag ended up in the picture the tracer is drawn on.
        setEditing("target");
      } else {
        setPinNote(out.reason || "could not map that to the tee view");
      }
    } catch (e) {
      setPinNote(e?.message || String(e));
    }
  }

  // Re-map the hole's stored pin into THIS swing's target. Mapped fresh
  // rather than copied: a re-calibration since it was marked should
  // correct it, not be ignored.
  async function applyHolePin() {
    if (!holePin?.pin_green) return;
    setPinNote(null);
    try {
      const out = await api.mapLanding(adminPassword, row.id, {
        spot: holePin.pin_green,
        tee_size: frameW && frameH ? [frameW, frameH] : null,
      });
      if (out.tee_xy) {
        const t = { x: Math.round(out.tee_xy[0]), y: Math.round(out.tee_xy[1]) };
        setDraft((d) => ({ ...d, target: t }));
        persistPatch({ target: t });
      } else {
        setPinNote(out.reason || "could not map the hole's flag");
      }
    } catch (e) {
      setPinNote(e?.message || String(e));
    }
  }

  // Open the calibrator on the two frames the operator is already
  // working with: impact on the tee, the landing on the green.
  async function openCalibrator() {
    setTeeLanding((t) => t);
    try {
      const [t, g, vm] = await Promise.all([
        api.getLongUploadFrame(adminPassword, row.id,
                               draft.impactFrame ?? 0, "tee"),
        api.getLongUploadFrame(adminPassword, row.id,
                               draft.landingFrame ?? defaultEnd ?? 0, "green",
                               draft.impactFrame ?? null),
        api.getViewMap(adminPassword, row.id).catch(() => null),
      ]);
      setViewMap(vm?.view_map || null);
      setCalibrating({
        tee: { ...t, frame: draft.impactFrame ?? 0 },
        green: { ...g, frame: draft.landingFrame ?? defaultEnd ?? 0 },
        existing: vm?.view_map || null,
        mismatch: vm?.mismatch || null,
        // What this calibration will apply to. Worth saying out loud:
        // it is not saved against this upload, so it is about to change
        // every swing those two cameras record.
        scope: vm?.course_name
          ? `${vm.course_name} · hole ${vm.hole}`
            + (vm.key_reason ? ` · filed by ${vm.key_reason}` : "")
          : null,
        reason: vm?.reason || null,
      });
    } catch (e) {
      setTeeLanding({ reason: e?.message || String(e) });
    }
  }

  async function scanGreen(level) {
    if (greenScanning) return;
    setGreenScanning(true);
    setGreenScanNote(null);
    try {
      const out = await api.scanPlotRegion(adminPassword, row.id, {
        which: "green",
        impact_frame: draft.impactFrame ?? null,
        sensitivity: level,
      });
      const dots = out.dots || [];
      const first = out.start_frame ?? 0;
      const last = out.end_frame ?? first;
      setGreenHeat({ dots, first, span: Math.max(1, last - first) });
      setGreenScanLevel(level);
      if (!dots.length) {
        // Nothing found is a real answer at level 3 and a shrug at
        // level 1, so say which one this was.
        setGreenScanNote(
          level >= 3
            ? "No motion at all in the green window — the ball did not "
              + "land in this camera's view."
            : `No motion at level ${level}. Try Deeper.`,
        );
      } else {
        setGreenScanNote(
          `${dots.length} dots over frames ${first}–${last} `
          + `(level ${level}). Click the one where it lands.`,
        );
      }
    } catch (e) {
      setGreenScanNote(e?.message || String(e));
    } finally {
      setGreenScanning(false);
    }
  }

  // Default cut frame (until manually set): 2.5 s after impact. The
  // produced clip cuts from the tee tracer to the green camera here.
  const fps = row?.tee_fps || 30;
  const autoCutFrame = draft.impactFrame != null
    ? Math.round(draft.impactFrame + 2.5 * fps)
    : null;
  const effectiveCutFrame = draft.cutFrame ?? autoCutFrame;
  // Real-world wall-clock time of the cut frame (tee start + cut/fps).
  // This is the same instant the green camera switches to, so the
  // operator can verify it against the clock overlay on each raw clip.
  const teeStartMs = row?.tee_recording_started_at
    ? parseApiDate(row.tee_recording_started_at)?.getTime() ?? null
    : null;
  const cutClockMs = (teeStartMs != null && effectiveCutFrame != null)
    ? teeStartMs + (effectiveCutFrame / fps) * 1000
    : null;

  // Frame-pick modes: address, impact, start, end, cut. Each seeds the
  // navigator from the corresponding draft frame index when entered.
  // No "start": the clip's lead-in is D3_PRE_ROLL_SEC before the strike,
  // decided by produce, not trimmed by hand. Four things go in here --
  // impact, end, ball, target -- and the first two are frames.
  const FRAME_PICK_MODES = new Set(["address", "impact", "landing", "cut"]);
  const frameForMode = {
    address: draft.addressFrame,
    impact: draft.impactFrame,
    landing: draft.landingFrame ?? defaultEnd ?? defaultEndFrame(),
    cut: effectiveCutFrame ?? 0,
  };

  // THE END FRAME IS A GREEN DECISION. It is where the produced clip
  // stops, and by then the cut is on the green camera -- asking the
  // operator to choose it from the tee view is asking about the wrong
  // picture. Everything else is a tee frame.
  // The landing is a green-camera call: the ball comes down in the green
  // view, and by then the cut has already moved there. The landing SPOT is
  // marked on the landing frame, so it stays on the green too.
  // ...and the flag is a green-camera call for the same reason the
  // landing is: on the tee frame the pin is a couple of pixels on the
  // horizon, and on the green camera it is metres across.
  const cameraForMode = (mode) =>
    (mode === "landing" || mode === "landing_spot" || mode === "target_green")
      ? "green" : "tee";

  // Where produce would end this clip if nobody said otherwise: the
  // green half runs `greenSeconds` past the strike (D3_GREEN_SEC on the
  // backend, echoed by the frame endpoint so the two cannot drift).
  function defaultEndFrame() {
    const fps = navFps || 30;
    const secs = greenSeconds || 6;
    const base = draft.impactFrame ?? 0;
    const n = Math.round(base + secs * fps);
    const max = (navTotal ?? totalFrames ?? 0) - 1;
    return max > 0 ? Math.min(max, n) : n;
  }

  // OPEN ON THE IMPACT FRAME. The wizard used to land on the address
  // frame, which is the least useful picture in the clip -- the swing has
  // not happened and nothing on this panel refers to it. With nothing
  // detected the wizard opens on the FIRST frame: no detection runs any
  // more, so there is no impact frame to open on and no reason to guess
  // at one -- the operator scrubs from the top.
  useEffect(() => {
    loadFrame(draft.impactFrame ?? 0, "tee");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // EVERY MODE LOADS ITS OWN PICTURE. This used to reload only for the
  // four frame-picking modes, plus two special cases -- so entering
  // Target, or Ball landing spot with no landing frame yet, left
  // whatever was already on screen. Come to Target straight from
  // Landing frame and you were marking the flag on a picture of the
  // green while the app believed you were on the tee, which is how a
  // tee-coordinate flag ended up drawn over the green view at the same
  // spot in both.
  const greenPickFrame = () =>
    draft.landingFrame ?? defaultEnd ?? defaultEndFrame();
  const frameForEditing = (mode) => {
    if (FRAME_PICK_MODES.has(mode)) return frameForMode[mode] ?? 0;
    // Marking a spot means looking at the frame it happened on.
    if (mode === "landing_spot" || mode === "target_green") {
      return greenPickFrame();
    }
    // PLACING THE BALL IS AN IMPACT-FRAME JOB. It used to show the
    // address frame, where the club is still behind the ball and the
    // golfer's stance hides the spot -- the operator was aiming at a
    // picture of a different moment. The impact frame is the one the
    // tracer starts from, so it is the one to point at. Everything else
    // tee-side wants the same frame.
    return draft.impactFrame ?? 0;
  };

  useEffect(() => {
    loadFrame(frameForEditing(editing), cameraForMode(editing));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing, draft.impactFrame, draft.landingFrame]);

  async function loadFrame(frameIdx, which = null) {
    const cam = which || cameraForMode(editing);
    setNavLoading(true);
    try {
      const data = await api.getLongUploadFrame(
        adminPassword, row.id, frameIdx, cam,
        cam === "green" ? draft.impactFrame ?? null : null,
      );
      setNavFrame(data.frame);
      setNavUrl(data.image_url);
      setNavWhich(data.which || cam);
      if (data.fps) setNavFps(data.fps);
      setNavWallClock(data.wall_clock || null);
      if (data.green_seconds) setGreenSeconds(data.green_seconds);
      if (data.total_frames) setNavTotal(data.total_frames);
      if (data.width && data.height) {
        setNavDims({ w: data.width, h: data.height });
      }
      // Where produce would stop, in green frames. Only arrives WITH a
      // green frame, since working it out needs the tee->green offset
      // the server holds. The "end" mode that used to self-correct to
      // it is gone -- the End frame row was replaced by Landing frame --
      // so this is now just recorded for the green modes to default to.
      if (data.default_end_frame != null) setDefaultEnd(data.default_end_frame);
    } catch (e) {
      console.warn("frame fetch failed", e);
      // A missing green half must not leave the operator staring at the
      // previous frame with no explanation.
      if (cam === "green") setNavUrl(null);
    } finally {
      setNavLoading(false);
    }
  }

  function clampedStep(delta) {
    const cur = navFrame ?? frameForMode[editing] ?? 0;
    const max = (navTotal ?? totalFrames ?? 1) - 1;
    return Math.max(0, Math.min(max, (cur || 0) + delta));
  }

  function clampedJump(absolute) {
    const max = (navTotal ?? totalFrames ?? 1) - 1;
    const n = Number.isFinite(absolute) ? absolute : 0;
    return Math.max(0, Math.min(max, n));
  }

  // TIME OF DAY, not an offset into the clip. "28.5s" describes a
  // position in a file; "9:08:30.02" is when the shot happened, which is
  // what matches a clip to a group on the tee sheet. Hundredths because
  // at 50fps a frame is 20ms -- tenths would make consecutive frames
  // read identically while stepping.
  const atTime = () => {
    const d = parseApiDate(navWallClock);
    if (!d) return "";
    const hh = d.getHours();
    const mm = String(d.getMinutes()).padStart(2, "0");
    const ss = String(d.getSeconds()).padStart(2, "0");
    const cs = String(Math.floor(d.getMilliseconds() / 10)).padStart(2, "0");
    return ` · ${hh}:${mm}:${ss}.${cs}`;
  };

  let leftImageUrl = navUrl || draft.addressImageUrl;
  let leftFrameLabel = `Frame ${navFrame ?? "—"}${atTime()}`;
  const showFrameNav = FRAME_PICK_MODES.has(editing);
  if (editing === "landing_spot") {
    leftImageUrl = navUrl || draft.addressImageUrl;
    leftFrameLabel =
      `Landing frame · ${draft.landingFrame ?? "—"}${atTime()}`
      + " · green camera — mark where it lands";
  } else if (editing === "target_green") {
    leftImageUrl = navUrl || draft.addressImageUrl;
    leftFrameLabel =
      `Frame ${navFrame ?? "—"}${atTime()}`
      + " · green camera — click the BASE of the flagstick";
  } else if (editing === "ball") {
    // The impact frame, with no frame-nav controls -- the operator is
    // placing a ball here, not choosing a frame.
    leftImageUrl = navUrl || draft.addressImageUrl;
    leftFrameLabel =
      `Impact frame · ${draft.impactFrame ?? "—"}${atTime()}`
      + " — place the ball";
  } else if (showFrameNav) {
    leftImageUrl = navUrl || draft.addressImageUrl;
    const total = navTotal != null ? ` / ${navTotal - 1}` : "";
    const labels = {
      address: "Address", impact: "Impact",
      landing: "Landing", cut: "Cut",
    };
    leftFrameLabel =
      `${labels[editing] || "Frame"} frame · ${navFrame ?? "—"}${total}`
      + atTime()
      + (navWhich === "green" ? " · green camera" : "");
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(320px, 1.5fr) minmax(260px, 1fr)",
        gap: 16,
        height: "100%",
        minHeight: 0,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div className="tiny upper muted" style={{ marginBottom: 4 }}>
          {leftFrameLabel}
        </div>
        <div
          style={{
            flex: 1, minHeight: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          <FramePreview
            imageUrl={leftImageUrl}
            // THE DIMENSIONS OF THE PICTURE ON SCREEN, which is not
            // always the tee's. The green camera runs at its own
            // resolution, and every overlay is placed by dividing
            // through these -- so handing it the tee's size while
            // showing a green frame puts each marker a fraction off,
            // and turns a click on the green into a tee-scaled
            // coordinate.
            frameW={navDims?.w ?? frameW}
            frameH={navDims?.h ?? frameH}
            editing={editing}
            draft={draft}
            setDraft={setDraft}
            loading={navLoading}
            onPickGreenPin={pickGreenPin}
            onGreen={navWhich === "green"}
            // The chain produce would draw its comet along, in the
            // green camera's own pixels — so it only belongs over a
            // green frame, same rule as the landing heat.
            comet={
              navWhich === "green" && greenFlight?.points
                ? { points: greenFlight.points, current: navFrame } : null
            }
            // THE FLAG, IN THE COORDINATES OF WHICHEVER CAMERA IS UP.
            // draft.target is a tee pixel; drawing it over a green
            // frame puts the flag at the same spot in both pictures,
            // which is what it was doing. The hole's pin is held in
            // GREEN pixels, so on a green frame that is the one to
            // draw -- the same fact, in the right space.
            greenTarget={
              navWhich === "green" && holePin?.pin_green
                ? { x: holePin.pin_green[0], y: holePin.pin_green[1] }
                : null
            }
            // The mirror image of the heat rule: the mapped landing is
            // in TEE coordinates, so it belongs only over a tee frame.
            mappedLanding={
              navWhich !== "green" && teeLanding?.tee_xy
                ? { x: teeLanding.tee_xy[0], y: teeLanding.tee_xy[1] }
                : null
            }
            // Only over the green camera: these are green-frame
            // coordinates and would land in the trees on a tee frame.
            heat={
              greenHeat && (editing === "landing" || editing === "landing_spot")
                ? {
                  ...greenHeat,
                  current: navFrame,
                  // A dot IS the answer to both questions this step
                  // asks -- which frame, and where in it -- so take
                  // both and save them. Stepping off the frame
                  // afterwards must not quietly discard the pick.
                  onPick: (d) => {
                    loadFrame(d.frame, "green");
                    setDraft((prev) => ({
                      ...prev,
                      landingFrame: d.frame,
                      landingSpot: { x: d.x, y: d.y },
                    }));
                    persistPatch({
                      landing_frame: d.frame,
                      landing_spot: { x: d.x, y: d.y },
                    });
                  },
                }
                : null
            }
            frameNav={showFrameNav ? {
              current: navFrame,
              total: navTotal,
              onJumpStart: () => loadFrame(0),
              onStepBack10: () => loadFrame(clampedStep(-10)),
              onStepBack1: () => loadFrame(clampedStep(-1)),
              onStepFwd1: () => loadFrame(clampedStep(1)),
              onStepFwd10: () => loadFrame(clampedStep(10)),
              onJumpEnd: () => {
                const t = navTotal ?? totalFrames;
                if (t) loadFrame(t - 1);
              },
            } : null}
          />
        </div>
        <div className="tiny muted" style={{ marginTop: 6 }}>
          Green dot = ball at rest · Green box = ball-tracer detection
          area · Red flag = target. Click a field on the right to edit —
          then drag the marker, or click anywhere on the frame to place it.
        </div>
        {calibrating && (
          calibrating.reason ? (
            <div className="err-text small" style={{ marginTop: 6 }}>
              {calibrating.reason}
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto", marginLeft: 8 }}
                onClick={() => setCalibrating(null)}
              >
                OK
              </button>
            </div>
          ) : (
            <ViewMapModal
              uploadId={row.id}
              adminPassword={adminPassword}
              teeFrame={calibrating.tee}
              greenFrame={calibrating.green}
              existing={calibrating.existing}
              mismatch={calibrating.mismatch}
              scope={calibrating.scope}
              onClose={() => setCalibrating(null)}
              onSaved={(out) => {
                setCalibrating(null);
                // Bump the dependency so the mapped landing is re-asked
                // with the calibration that was just saved.
                setViewMap({ saved_at: out?.rms_px ?? "exact", ...out });
              }}
            />
          )
        )}
      </div>

      <div
        style={{
          display: "flex", flexDirection: "column", gap: 10,
          overflowY: "auto", minHeight: 0, paddingRight: 4,
        }}
      >
        <EditableRow
          label="Impact frame"
          value={`Frame ${draft.impactFrame}`}
          active={editing === "impact"}
          onActivate={() => setEditing(editing === "impact" ? null : "impact")}
        >
          <FrameStepper
            current={navFrame}
            total={navTotal}
            loading={navLoading}
            onStep={(delta) => loadFrame(clampedStep(delta))}
            onJump={(n) => loadFrame(clampedJump(n))}
            onApply={() => {
              if (navFrame == null) return;
              setDraft((d) => ({ ...d, impactFrame: navFrame }));
              persistPatch({ impact_frame: navFrame });
              setEditing(null);
            }}
          />
        </EditableRow>

        <EditableRow
          label="Ball at rest"
          value={
            draft.ball
              ? `${draft.ball.x}, ${draft.ball.y} px${draft.ballManual ? " (placed by hand)" : ""}`
              : "Not set"
          }
          active={editing === "ball"}
          onActivate={() => setEditing(editing === "ball" ? null : "ball")}
        >
          <div className="tiny muted">
            Drag the green dot on the left, or click anywhere on the frame
            to drop it there. This is where the tracer line STARTS, so it
            wants to be on the ball itself, not near it.
          </div>
          <button
            type="button"
            style={{ width: "auto", marginTop: 6 }}
            onClick={() => {
              if (draft.ball) {
                // ball_manual marks it operator-placed. Production checks
                // that flag before writing a detected rest position, so a
                // re-produce can't quietly move it back.
                setDraft((d) => ({ ...d, ballManual: true }));
                persistPatch({ ball: draft.ball, ball_manual: true });
              }
              setEditing(null);
            }}
          >
            Done
          </button>
        </EditableRow>

        <EditableRow
          label="Landing frame"
          value={draft.landingFrame != null
            ? `Frame ${draft.landingFrame}`
            : "Not set"}
          active={editing === "landing"}
          onActivate={() => setEditing(editing === "landing" ? null : "landing")}
        >
          <div className="tiny muted" style={{ marginBottom: 6 }}>
            The GREEN camera. Step to the frame where the ball first
            touches down. The clip ends {LANDING_TAIL_SEC}s after this, and
            the tracer is drawn to arrive here.
          </div>
          <div className="row" style={{ gap: 6, marginBottom: 6,
                                        flexWrap: "wrap" }}>
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto" }}
              disabled={greenScanning}
              onClick={() => scanGreen(greenScanLevel)}
              title="Diff every frame of the green window the clip covers — impact through the end of the cut, plus 2s — and draw all the motion at once. The ball reads as a short arc ramping blue to orange; an empty picture means it did not land in this camera's view."
            >
              {greenScanning ? "Scanning…" : "🔍 Scan for landing"}
            </button>
            {greenHeat && (
              <>
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto" }}
                  disabled={greenScanning || greenScanLevel >= 3}
                  onClick={() => scanGreen(Math.min(3, greenScanLevel + 1))}
                  title="Lower the motion threshold and keep more blobs. Level 3 hands back wind in the trees too — you are the filter."
                >
                  Deeper
                </button>
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto" }}
                  onClick={() => { setGreenHeat(null); setGreenScanNote(null); }}
                >
                  Clear
                </button>
              </>
            )}
          </div>
          {greenScanNote && (
            <div className="tiny muted" style={{ marginBottom: 6 }}>
              {greenScanNote}
            </div>
          )}
          {/* THE COMET, PREVIEWED. Produce draws one on the green half
              whenever a chain of blobs walks back from the marked
              landing. Running the same search here means the operator
              knows before producing whether there will be one, and when
              there will not be, why. */}
          {draft.landingFrame != null && draft.landingSpot && (
            <div style={{ marginBottom: 6 }}>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto" }}
                disabled={greenFlightBusy}
                onClick={findGreenFlight}
                title="Walk backwards from the landing looking for the ball's own frames. Produce draws a comet along whatever this finds — and nothing at all if it finds no obvious path, because a fabricated one over grass is worse than none."
              >
                {greenFlightBusy ? "Looking…" : "☄ Find the ball's descent"}
              </button>
              {greenFlight && (
                <span className="tiny" style={{
                  marginLeft: 8,
                  color: greenFlight.points ? "#3ee37a" : "#f59e0b",
                }}>
                  {greenFlight.points
                    ? `${greenFlight.n} frames — produce will draw a comet `
                      + `along this`
                    : `no comet: ${greenFlight.reason}`}
                </span>
              )}
            </div>
          )}
          <FrameStepper
            current={navFrame}
            total={navTotal}
            loading={navLoading}
            onStep={(delta) => loadFrame(clampedStep(delta), "green")}
            onJump={(n) => loadFrame(clampedJump(n), "green")}
            onApply={() => {
              if (navFrame == null) return;
              setDraft((d) => ({ ...d, landingFrame: navFrame }));
              persistPatch({ landing_frame: navFrame });
              setEditing(null);
            }}
          />
        </EditableRow>

        <EditableRow
          label="Ball landing spot"
          value={draft.landingSpot
            ? `${draft.landingSpot.x}, ${draft.landingSpot.y} px`
            : "Not set"}
          active={editing === "landing_spot"}
          onActivate={() =>
            setEditing(editing === "landing_spot" ? null : "landing_spot")}
        >
          <div className="tiny muted">
            {draft.landingFrame == null
              ? "Set the landing frame first — this marks the spot on it."
              : "Click where the ball lands on the green. This is where "
                + "the tracer ENDS, so put it on the ball, not near it."}
          </div>
          {/* WHAT THIS BUYS, said plainly. The spot is marked on the
              green camera and the tracer is drawn on the tee one, so
              until the two views are mapped to each other this click
              cannot reach the line it is meant to finish. */}
          {draft.landingSpot && (
            <div className="tiny" style={{ marginTop: 6 }}>
              {teeLanding?.tee_xy ? (
                <span style={{ color: "#f97316" }}>
                  → tee frame {teeLanding.tee_xy[0]}, {teeLanding.tee_xy[1]} px
                  — the tracer will finish there (shown on the tee frame)
                </span>
              ) : (
                <span className="muted">
                  Not mapped to the tee view: {teeLanding?.reason || "…"}
                </span>
              )}
            </div>
          )}
          <div className="row" style={{ gap: 6, marginTop: 6,
                                        flexWrap: "wrap" }}>
            <button
              type="button"
              style={{ width: "auto" }}
              disabled={draft.landingFrame == null}
              onClick={() => {
                if (draft.landingSpot) {
                  persistPatch({ landing_spot: draft.landingSpot });
                }
                setEditing(null);
              }}
            >
              Done
            </button>
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto" }}
              onClick={openCalibrator}
              title="Map the green camera's view onto the tee camera's, by clicking the same ground features in both. Done ONCE per hole — every swing these two cameras record afterwards is aimed by it, so there is nothing to do here on a hole that already shows as mapped."
            >
              {_vmCal ? "⊹ Re-calibrate tee ↔ green" : "⊹ Calibrate tee ↔ green"}
            </button>
          </div>
          {calLine && (
            <div className="tiny" style={{ marginTop: 4, color: "#3ee37a" }}>
              ✓ {calLine} — nothing to do here, this hole is already
              mapped.
            </div>
          )}
        </EditableRow>

        <EditableRow
          label="Target"
          value={draft.target ? `${draft.target.x}, ${draft.target.y} px` : "Not set"}
          // The row stays open in either mode — they are two pictures of
          // the same field, not two different fields.
          active={editing === "target" || editing === "target_green"}
          // OPENS ON THE GREEN CAMERA. On the tee frame the pin is a
          // couple of pixels on the horizon and there is no way to be
          // accurate about it; on the green camera it is metres across.
          // So mark it where it is visible and let the mapping carry it
          // back to the tee, which is the only place it gets used.
          onActivate={() => setEditing(
            (editing === "target" || editing === "target_green")
              ? null : "target_green")}
        >
          <div className="tiny muted">
            {editing === "target_green"
              ? "Click the BASE of the flagstick — where it meets the "
                + "grass. The mapping describes the ground, so the top of "
                + "the stick maps as if it were lying on it, which lands "
                + "the target well past the hole."
              : "The flag, carried back to the tee frame by this hole's "
                + "mapping. Mark on green camera to set it where you can "
                + "see it; the tee frame is only for nudging the result."}
          </div>
          {holePin?.pin_green && editing !== "target_green" && (
            <div className="tiny" style={{ marginTop: 6 }}>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto" }}
                onClick={applyHolePin}
              >
                Use this hole&apos;s flag
              </button>
              <span className="muted" style={{ marginLeft: 6 }}>
                marked {holePin.pin_set_at?.slice(0, 10)}
                {/* PINS MOVE. Same position on a different day is a
                    different hole location, so the date is the point of
                    this line, not decoration. */}
              </span>
            </div>
          )}
          <div className="row" style={{ gap: 6, marginTop: 6,
                                        flexWrap: "wrap" }}>
            <button
              type="button"
              style={{ width: "auto" }}
              onClick={() => {
                if (draft.target) persistPatch({ target: draft.target });
                setEditing(null);
              }}
            >
              Done
            </button>
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto" }}
              onClick={() => setEditing(
                editing === "target_green" ? "target" : "target_green")}
              title="Mark the flagstick on the GREEN camera, where it is metres away rather than a few pixels on the horizon, and carry it to the tee frame through the same mapping the landing uses."
            >
              {editing === "target_green"
                ? "← nudge it on the tee frame"
                : "⚑ Mark on green camera"}
            </button>
          </div>
          {pinNote && (
            <div className="err-text tiny" style={{ marginTop: 6 }}>
              {pinNote}
            </div>
          )}
        </EditableRow>

        <div className="tiny muted" style={{ marginTop: 4 }}>
          Frame size:{" "}
          {frameW && frameH ? `${frameW} × ${frameH} px` : "unknown"}
          {totalFrames ? ` · ${totalFrames} frames` : ""}
        </div>

        <button
          type="button"
          className="ghost"
          style={{ width: "100%", marginTop: 6 }}
          disabled={redetecting}
          onClick={async () => {
            if (!confirm(
              "Re-run auto-detect from the source video? This replaces "
              + "the current handedness / address / impact / ball / "
              + "ROI / target with a fresh detection."
            )) return;
            // This one runs a detector over the whole clip and then
            // reloads the page -- tens of seconds with nothing on
            // screen unless the button says so itself.
            setRedetecting(true);
            try {
              // Persist into edit_metrics directly; the wizard reads
              // from there on next reload.
              await api.autoDetectLongUpload(adminPassword, row.id);
              window.location.reload();
            } catch (e) {
              setRedetecting(false);
              alert(`Re-detect failed: ${e.message}`);
            }
          }}
          title="Re-run auto-detect from the source video and replace the current metrics"
        >
          {redetecting ? "Re-detecting…" : "Re-detect from source"}
        </button>
      </div>
    </div>
  );
}

function EditableRow({ label, value, active, onActivate, children }) {
  return (
    <div
      style={{
        border: "1px solid var(--border, #2a2a2a)",
        borderRadius: 6,
        padding: 6,
        background: active ? "rgba(34,197,94,0.06)" : "transparent",
      }}
    >
      <button
        type="button"
        onClick={onActivate}
        style={{
          width: "100%", padding: 0, background: "transparent",
          border: "none", textAlign: "left", cursor: "pointer", color: "inherit",
        }}
      >
        <div className="tiny upper muted">{label}</div>
        <div style={{ fontSize: "0.8rem" }}>{value}</div>
      </button>
      {active && (
        <div style={{ marginTop: 8 }}>{children}</div>
      )}
    </div>
  );
}

function FrameStepper({ current, total, loading, onStep, onJump, onApply }) {
  const disabled = loading || current == null;
  const [jumpVal, setJumpVal] = useState("");
  const maxFrame = total != null ? total - 1 : null;

  function commitJump() {
    if (jumpVal === "") return;
    const n = parseInt(jumpVal, 10);
    if (!Number.isFinite(n)) return;
    onJump?.(n);
    setJumpVal("");
  }

  return (
    <div>
      <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
        <button type="button" className="ghost" style={{ width: "auto" }}
          disabled={disabled} onClick={() => onStep(-10)}>−10</button>
        <button type="button" className="ghost" style={{ width: "auto" }}
          disabled={disabled} onClick={() => onStep(-1)}>−1</button>
        <span className="small" style={{ alignSelf: "center", minWidth: 70, textAlign: "center" }}>
          {current ?? "—"}{total != null ? ` / ${total - 1}` : ""}
        </span>
        <button type="button" className="ghost" style={{ width: "auto" }}
          disabled={disabled} onClick={() => onStep(1)}>+1</button>
        <button type="button" className="ghost" style={{ width: "auto" }}
          disabled={disabled} onClick={() => onStep(10)}>+10</button>
      </div>
      {onJump && (
        <div className="row" style={{ gap: 6, marginTop: 8, alignItems: "center" }}>
          <input
            type="number"
            min="0"
            max={maxFrame ?? undefined}
            value={jumpVal}
            onChange={(e) => setJumpVal(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter") { e.preventDefault(); commitJump(); } }}
            placeholder={maxFrame != null ? `Jump to frame (0–${maxFrame})` : "Jump to frame"}
            disabled={loading}
            style={{ flex: 1, minWidth: 0 }}
          />
          <button
            type="button"
            className="ghost"
            style={{ width: "auto" }}
            disabled={loading || jumpVal === ""}
            onClick={commitJump}
          >
            Go
          </button>
        </div>
      )}
      <button
        type="button"
        style={{ width: "100%", marginTop: 8 }}
        disabled={disabled}
        onClick={onApply}
      >
        Save this frame
      </button>
    </div>
  );
}

function scaleRoi(roi, factor, frameW, frameH) {
  if (!roi) return roi;
  const cx = roi.x + roi.w / 2;
  const cy = roi.y + roi.h / 2;
  const w = Math.max(20, Math.round(roi.w * factor));
  const h = Math.max(20, Math.round(roi.h * factor));
  let x = Math.round(cx - w / 2);
  let y = Math.round(cy - h / 2);
  x = Math.max(0, Math.min(frameW - w, x));
  y = Math.max(0, Math.min(frameH - h, y));
  return { x, y, w, h };
}


function FramePreview({ imageUrl, frameW, frameH, editing, draft, setDraft,
  loading, frameNav, heat, mappedLanding, onPickGreenPin,
  greenTarget, onGreen, comet }) {
  const hasDims = !!(frameW && frameH);
  const containerRef = useRef(null);

  // Convert a pointer event to native pixel coords on the frame.
  function eventToFrame(e) {
    if (!containerRef.current || !hasDims) return null;
    const r = containerRef.current.getBoundingClientRect();
    const xPct = (e.clientX - r.left) / r.width;
    const yPct = (e.clientY - r.top) / r.height;
    return {
      x: Math.max(0, Math.min(frameW - 1, Math.round(xPct * frameW))),
      y: Math.max(0, Math.min(frameH - 1, Math.round(yPct * frameH))),
    };
  }

  // Drag-the-ball handler. Pointer capture so the drag survives leaving
  // the dot before releasing.
  function onBallPointerDown(e) {
    if (editing !== "ball") return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const pt = eventToFrame(ev);
      if (pt) setDraft((d) => ({ ...d, ball: pt }));
    };
    const up = () => {
      target.releasePointerCapture?.(e.pointerId);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  }

  // Clicking anywhere on the frame in ball/target mode also moves the
  // marker — easier than precision-grabbing the dot.
  function onFramePointerDown(e) {
    if (editing === "ball") {
      const pt = eventToFrame(e);
      if (pt) setDraft((d) => ({ ...d, ball: pt }));
    } else if (editing === "target") {
      const pt = eventToFrame(e);
      if (pt) setDraft((d) => ({ ...d, target: pt }));
    } else if (editing === "landing_spot") {
      const pt = eventToFrame(e);
      if (pt) setDraft((d) => ({ ...d, landingSpot: pt }));
    } else if (editing === "target_green") {
      // A GREEN-frame click. It cannot go into draft.target as it
      // stands -- that field is tee pixels -- so it goes up to be
      // mapped across, and comes back as the target.
      const pt = eventToFrame(e);
      if (pt) onPickGreenPin?.(pt);
    }
  }

  function onLandingPointerDown(e) {
    if (editing !== "landing_spot") return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const pt = eventToFrame(ev);
      if (pt) setDraft((d) => ({ ...d, landingSpot: pt }));
    };
    const up = () => {
      target.releasePointerCapture?.(e.pointerId);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  }

  function onTargetPointerDown(e) {
    if (editing !== "target") return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const move = (ev) => {
      const pt = eventToFrame(ev);
      if (pt) setDraft((d) => ({ ...d, target: pt }));
    };
    const up = () => {
      target.releasePointerCapture?.(e.pointerId);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  }

  // ROI body drag (move) — handlers below per-corner do resize.
  function onRoiBodyPointerDown(e) {
    if (editing !== "roi" || !draft.roi) return;
    e.preventDefault();
    e.stopPropagation();
    const target = e.currentTarget;
    target.setPointerCapture(e.pointerId);
    const start = eventToFrame(e);
    const origin = { ...draft.roi };
    const move = (ev) => {
      const pt = eventToFrame(ev);
      if (!pt || !start) return;
      const dx = pt.x - start.x;
      const dy = pt.y - start.y;
      const nx = Math.max(0, Math.min(frameW - origin.w, origin.x + dx));
      const ny = Math.max(0, Math.min(frameH - origin.h, origin.y + dy));
      setDraft((d) => ({ ...d, roi: { ...origin, x: nx, y: ny } }));
    };
    const up = () => {
      target.releasePointerCapture?.(e.pointerId);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  }

  function onRoiHandlePointerDown(corner) {
    return (e) => {
      if (editing !== "roi" || !draft.roi) return;
      e.preventDefault();
      e.stopPropagation();
      const target = e.currentTarget;
      target.setPointerCapture(e.pointerId);
      const origin = { ...draft.roi };
      const move = (ev) => {
        const pt = eventToFrame(ev);
        if (!pt) return;
        let { x, y, w, h } = origin;
        if (corner.includes("e")) w = Math.max(20, pt.x - x);
        if (corner.includes("s")) h = Math.max(20, pt.y - y);
        if (corner.includes("w")) {
          const right = x + w;
          x = Math.min(right - 20, Math.max(0, pt.x));
          w = right - x;
        }
        if (corner.includes("n")) {
          const bottom = y + h;
          y = Math.min(bottom - 20, Math.max(0, pt.y));
          h = bottom - y;
        }
        if (x + w > frameW) w = frameW - x;
        if (y + h > frameH) h = frameH - y;
        setDraft((d) => ({ ...d, roi: { x, y, w, h } }));
      };
      const up = () => {
        target.releasePointerCapture?.(e.pointerId);
        target.removeEventListener("pointermove", move);
        target.removeEventListener("pointerup", up);
      };
      target.addEventListener("pointermove", move);
      target.addEventListener("pointerup", up);
    };
  }

  const showRoi = !!draft.roi && (editing === null || editing === "roi"
    || editing === "ball" || editing === "target");
  // The ball at rest is a TEE-frame fact. While the operator is on a
  // green frame it is a green dot sitting on grass that will not move --
  // which reads as the landing marker being broken, because the actual
  // landing marker is the one that has not been placed yet.
  // WHAT IS ACTUALLY ON SCREEN, not what the mode implies. The two
  // agree once every mode loads its own camera, but only after the load
  // lands -- and in the gap between, a tee marker would be painted over
  // a green frame. `onGreen` is the camera the displayed image came
  // from, so it is the one to believe.
  const onGreenFrame = onGreen
    || editing === "landing_spot" || editing === "landing"
    || editing === "target_green";
  const showBall = !!draft.ball && !onGreenFrame;
  // The target is a TEE-frame point. Over a green frame it would be a
  // flag planted wherever that pixel happens to fall.
  const showTarget = !!draft.target && !onGreenFrame;
  // The landing spot belongs to the landing FRAME, on the green camera.
  // Showing it over a tee frame would put an orange dot in the trees.
  const showLandingSpot = !!draft.landingSpot && editing === "landing_spot";
  const landingEditable = editing === "landing_spot";
  const ballEditable = editing === "ball";
  const targetEditable = editing === "target";
  const roiEditable = editing === "roi";

  const pct = (v, span) => `${(v / span) * 100}%`;

  // ONE LABEL PER FRAME, not per dot. The scan keeps up to ten blobs a
  // frame, so numbering every dot would stamp the same number ten times
  // over one cluster and bury the picture underneath. The first dot of
  // each frame carries the label; the rest of that frame's cluster is
  // plainly the same colour beside it.
  const heatLabelAt = useMemo(() => {
    const seen = new Set();
    const out = new Set();
    (heat?.dots || []).forEach((d, i) => {
      if (seen.has(d.frame)) return;
      seen.add(d.frame);
      out.add(i);
    });
    return out;
  }, [heat]);

  return (
    <div
      ref={containerRef}
      onPointerDown={onFramePointerDown}
      style={{
        position: "relative",
        // Vertical-fit: aspect-ratio drives width when height fills the
        // parent flex slot. maxWidth caps it on wide screens; maxHeight
        // is the available flex space.
        height: "100%",
        maxHeight: "100%",
        maxWidth: "100%",
        aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
        background: "var(--border, #222)",
        borderRadius: 6,
        overflow: "hidden",
        cursor: (ballEditable || targetEditable || landingEditable
                 || editing === "target_green")
          ? "crosshair" : "default",
        userSelect: "none",
      }}
    >
      {imageUrl ? (
        <img
          src={imageUrl}
          alt="Frame preview"
          draggable={false}
          style={{
            width: "100%", height: "100%", objectFit: "cover",
            pointerEvents: "none",
          }}
        />
      ) : (
        <div
          className="muted small"
          style={{
            position: "absolute", inset: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          {loading ? "Loading frame…" : "No frame"}
        </div>
      )}
      {/* THE GREEN'S MOTION, ALL OF IT AT ONCE. One dot per blob per
          frame over the window the clip covers, ramped blue (early) to
          orange (late) so the ball reads as a short coloured arc ending
          in a scatter where it bounces -- and so an empty picture is an
          answer too: nothing moved out here, so it did not land in
          view. Clicking a dot goes to its frame and drops the landing
          spot on it, which is the whole reason to look. */}
      {hasDims && heat?.dots?.length > 0 && (
        <svg
          viewBox={`0 0 ${frameW} ${frameH}`}
          preserveAspectRatio="none"
          style={{ position: "absolute", inset: 0, width: "100%",
                   height: "100%" }}
        >
          {heat.dots.map((d, i) => {
            const t = heat.span > 0
              ? Math.max(0, Math.min(1, (d.frame - heat.first) / heat.span))
              : 0;
            const hue = 205 - 175 * t;   // blue -> orange
            const here = d.frame === heat.current;
            const colour = `hsl(${hue} 90% 60%)`;
            const pick = (e) => {
              e.preventDefault();
              e.stopPropagation();
              heat.onPick?.(d);
            };
            return (
              <g key={`${d.frame}-${d.x}-${d.y}-${i}`}>
                <circle
                  cx={d.x}
                  cy={d.y}
                  r={(here ? 5 : 2.6) * (frameW / 900)}
                  fill={colour}
                  fillOpacity={here ? 0.95 : 0.5}
                  stroke={here ? "#fff" : "none"}
                  strokeWidth={frameW / 1200}
                  style={{ cursor: "pointer" }}
                  onPointerDown={pick}
                >
                  <title>{`frame ${d.frame} · ${d.x}, ${d.y}`}</title>
                </circle>
                {/* THE FRAME NUMBER, because the arc is only half the
                    answer. Knowing the ball is that streak is no use
                    without knowing which frame to save, and reading it
                    off a hover tooltip one dot at a time is not
                    reading. Outlined in black: these sit over grass,
                    sand and shade in the same picture. */}
                {heatLabelAt.has(i) && (
                  <text
                    x={d.x + frameW / 180}
                    y={d.y - frameW / 260}
                    fontSize={frameW / (here ? 62 : 80)}
                    fontWeight={here ? 700 : 500}
                    fill={here ? "#fff" : colour}
                    fillOpacity={here ? 1 : 0.9}
                    stroke="#000"
                    strokeWidth={frameW / 900}
                    strokeOpacity={0.75}
                    paintOrder="stroke"
                    style={{ cursor: "pointer" }}
                    onPointerDown={pick}
                  >
                    {d.frame}
                  </text>
                )}
              </g>
            );
          })}
        </svg>
      )}
      {/* THE BALL'S DESCENT, the path produce will run its comet along.
          Drawn head-to-tail so the direction reads at a glance, and
          with the frame under the playhead picked out, because the
          question being asked here is "is that the ball or a shadow"
          and stepping through is how it gets answered. */}
      {hasDims && comet?.points?.length > 1 && (
        <svg
          viewBox={`0 0 ${frameW} ${frameH}`}
          preserveAspectRatio="none"
          style={{ position: "absolute", inset: 0, width: "100%",
                   height: "100%", pointerEvents: "none" }}
        >
          <polyline
            points={comet.points.map((q) => `${q.x},${q.y}`).join(" ")}
            fill="none" stroke="#fff" strokeOpacity={0.55}
            strokeWidth={frameW / 500}
          />
          {comet.points.map((q, i) => {
            const t = i / Math.max(1, comet.points.length - 1);
            const here = q.frame === comet.current;
            return (
              <circle
                key={`${q.frame}-${i}`}
                cx={q.x} cy={q.y}
                r={(here ? 6 : 2 + 3 * t) * (frameW / 900)}
                fill={here ? "#fff" : `hsl(${40 - 30 * t} 100% ${55 + 25 * t}%)`}
                fillOpacity={here ? 1 : 0.45 + 0.5 * t}
                stroke={here ? "#38bdf8" : "none"}
                strokeWidth={frameW / 700}
              >
                <title>{`frame ${q.frame}`}</title>
              </circle>
            );
          })}
        </svg>
      )}
      {/* THE FLAG ON THE GREEN CAMERA, at its own coordinates. Same
          marker as the tee-side one, a different space — which is the
          whole point: it should sit on the actual flagstick in both
          pictures, not at the same pixel in both. */}
      {hasDims && greenTarget && (
        <div
          style={{
            position: "absolute",
            left: `${(greenTarget.x / frameW) * 100}%`,
            top: `${(greenTarget.y / frameH) * 100}%`,
            transform: "translate(-1px, -100%)",
            pointerEvents: "none",
          }}
        >
          <div style={{
            width: 2, height: 26, background: "#fff",
            boxShadow: "0 0 2px rgba(0,0,0,0.9)",
          }} />
          <div style={{
            position: "absolute", left: 2, top: 0,
            width: 0, height: 0,
            borderTop: "9px solid transparent",
            borderBottom: "9px solid transparent",
            borderLeft: "16px solid #ef4444",
            filter: "drop-shadow(0 0 1px rgba(0,0,0,0.9))",
          }} />
        </div>
      )}
      {/* WHERE THE TRACER WILL FINISH, in this frame. The landing was
          marked on the green camera; this is the same spot carried
          across by the green→tee mapping. Shown before producing so a
          bad calibration is caught by eye — if this sits in the trees
          instead of on the fairway, the mapping is wrong, and watching
          a produced tracer end there is a slower way to learn it. */}
      {hasDims && mappedLanding && (
        <svg
          viewBox={`0 0 ${frameW} ${frameH}`}
          preserveAspectRatio="none"
          style={{ position: "absolute", inset: 0, width: "100%",
                   height: "100%", pointerEvents: "none" }}
        >
          <circle cx={mappedLanding.x} cy={mappedLanding.y}
                  r={frameW / 90} fill="none" stroke="#f97316"
                  strokeWidth={frameW / 380} strokeOpacity={0.95} />
          <circle cx={mappedLanding.x} cy={mappedLanding.y}
                  r={frameW / 420} fill="#f97316" />
          <text x={mappedLanding.x + frameW / 70}
                y={mappedLanding.y - frameW / 180}
                fontSize={frameW / 55} fontWeight={600}
                fill="#f97316" stroke="#000" strokeWidth={frameW / 500}
                paintOrder="stroke">
            lands here
          </text>
        </svg>
      )}
      {loading && imageUrl && (
        <div
          className="tiny"
          style={{
            position: "absolute", top: 6, right: 8,
            background: "rgba(0,0,0,0.6)", padding: "2px 6px",
            borderRadius: 4, color: "#fff",
          }}
        >
          Loading…
        </div>
      )}

      {hasDims && showRoi && draft.roi && (
        <div
          onPointerDown={onRoiBodyPointerDown}
          style={{
            position: "absolute",
            left: pct(draft.roi.x, frameW),
            top: pct(draft.roi.y, frameH),
            width: pct(draft.roi.w, frameW),
            height: pct(draft.roi.h, frameH),
            border: "2px solid #22c55e",
            borderRadius: 4,
            cursor: roiEditable ? "move" : "default",
            boxShadow: "0 0 0 1px rgba(34,197,94,0.4)",
            pointerEvents: roiEditable ? "auto" : "none",
            background: roiEditable ? "rgba(34,197,94,0.06)" : "transparent",
          }}
          title="Ball detection area"
        >
          {roiEditable && ["nw", "ne", "sw", "se"].map((corner) => (
            <div
              key={corner}
              onPointerDown={onRoiHandlePointerDown(corner)}
              style={{
                position: "absolute",
                width: 12, height: 12,
                background: "#22c55e",
                border: "2px solid #fff",
                borderRadius: 2,
                cursor: `${corner}-resize`,
                left: corner.includes("w") ? -7 : "auto",
                right: corner.includes("e") ? -7 : "auto",
                top: corner.includes("n") ? -7 : "auto",
                bottom: corner.includes("s") ? -7 : "auto",
              }}
            />
          ))}
        </div>
      )}

      {hasDims && showBall && draft.ball && (
        <div
          onPointerDown={onBallPointerDown}
          style={{
            position: "absolute",
            left: pct(draft.ball.x, frameW),
            top: pct(draft.ball.y, frameH),
            width: 16, height: 16,
            borderRadius: "50%",
            background: "#22c55e",
            border: "2px solid #fff",
            transform: "translate(-50%, -50%)",
            cursor: ballEditable ? "grab" : "default",
            boxShadow: "0 0 6px rgba(0,0,0,0.7)",
            pointerEvents: ballEditable ? "auto" : "none",
            touchAction: "none",
          }}
          title={`Ball at rest (${draft.ball.x}, ${draft.ball.y})`}
        />
      )}

      {hasDims && showLandingSpot && draft.landingSpot && (
        <div
          onPointerDown={onLandingPointerDown}
          style={{
            position: "absolute",
            left: pct(draft.landingSpot.x, frameW),
            top: pct(draft.landingSpot.y, frameH),
            width: 16, height: 16,
            borderRadius: "50%",
            // Amber, not the ball's green: this is a different point on a
            // different camera, and confusing the two would be easy.
            background: "#d97706",
            border: "2px solid #fff",
            transform: "translate(-50%, -50%)",
            cursor: landingEditable ? "grab" : "default",
            boxShadow: "0 0 6px rgba(0,0,0,0.7)",
            pointerEvents: landingEditable ? "auto" : "none",
            touchAction: "none",
          }}
          title={
            `Ball landing spot (${draft.landingSpot.x}, `
            + `${draft.landingSpot.y})`
          }
        />
      )}

      {hasDims && showTarget && draft.target && (
        <FlagMarker
          x={draft.target.x}
          y={draft.target.y}
          frameW={frameW}
          frameH={frameH}
          editable={targetEditable}
          onPointerDown={onTargetPointerDown}
        />
      )}

      {frameNav && (
        <FrameNavBar nav={frameNav} loading={loading} />
      )}
    </div>
  );
}

function FrameNavBar({ nav, loading }) {
  // Compact frame-step bar at the bottom of the preview. Only rendered
  // when the wizard is in address / impact frame-pick mode.
  const stop = (e) => e.stopPropagation();
  const btn = {
    background: "rgba(0,0,0,0.55)", color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)", borderRadius: 4,
    width: 36, height: 30, fontSize: 13, fontWeight: 600,
    cursor: "pointer", padding: 0,
  };
  return (
    <div
      onPointerDown={stop}
      onClick={stop}
      style={{
        position: "absolute",
        left: "50%", bottom: 10,
        transform: "translateX(-50%)",
        display: "flex", gap: 6, alignItems: "center",
        background: "rgba(0,0,0,0.4)", padding: "6px 8px",
        borderRadius: 6, backdropFilter: "blur(4px)",
      }}
    >
      <button type="button" style={btn} disabled={loading} onClick={nav.onJumpStart} title="First frame">⏮</button>
      <button type="button" style={btn} disabled={loading} onClick={nav.onStepBack10} title="−10 frames">−10</button>
      <button type="button" style={btn} disabled={loading} onClick={nav.onStepBack1} title="−1 frame">−1</button>
      <span style={{ color: "#fff", fontSize: 12, padding: "0 6px", minWidth: 70, textAlign: "center" }}>
        {nav.current ?? "—"}{nav.total != null ? ` / ${nav.total - 1}` : ""}
      </span>
      <button type="button" style={btn} disabled={loading} onClick={nav.onStepFwd1} title="+1 frame">+1</button>
      <button type="button" style={btn} disabled={loading} onClick={nav.onStepFwd10} title="+10 frames">+10</button>
      <button type="button" style={btn} disabled={loading} onClick={nav.onJumpEnd} title="Last frame">⏭</button>
    </div>
  );
}

function FlagMarker({ x, y, frameW, frameH, editable, onPointerDown }) {
  // Flag rendered as inline SVG so it scales cleanly and stays
  // visible against both grass and sky backgrounds.
  return (
    <div
      onPointerDown={onPointerDown}
      style={{
        position: "absolute",
        left: `${(x / frameW) * 100}%`,
        top: `${(y / frameH) * 100}%`,
        width: 36, height: 36,
        transform: "translate(-6px, -100%)",
        cursor: editable ? "grab" : "default",
        pointerEvents: editable ? "auto" : "none",
        filter: "drop-shadow(0 2px 4px rgba(0,0,0,0.7))",
        touchAction: "none",
      }}
      title={`Target (${x}, ${y})`}
    >
      <svg viewBox="0 0 36 36" width="36" height="36" aria-hidden>
        <line x1="6" y1="2" x2="6" y2="34"
          stroke="#fff" strokeWidth="2" strokeLinecap="round" />
        <path d="M6 4 L28 9 L6 16 Z"
          fill="#ef4444" stroke="#fff" strokeWidth="1.5"
          strokeLinejoin="round" />
        <circle cx="6" cy="34" r="2.5" fill="#fff" />
      </svg>
    </div>
  );
}

/**
 * The two swing detectors, side by side.
 *
 * The pipeline counts swings with POSE — a wrist-speed burst with the right
 * spine bend. That detects a MOTION, so it cannot tell a swing from a
 * practice swing, and it fires on a waggle.
 *
 * The other answer is the ball: it was on the tee, and then it was not.
 * Nothing but a struck ball does that. This panel shows what each one found
 * on this clip, where they agree, and what it cost — so the comparison can
 * be made on real footage before either one replaces the other.
 *
 * Nothing here changes what gets produced.
 */
export function SwingDetectPanel({ sd }) {
  const [open, setOpen] = useState(false);
  const [shot, setShot] = useState(null);
  if (sd.error) {
    return (
      <div className="tiny" style={{ marginTop: 6, color: "var(--danger,#c0392b)" }}>
        <b>Ball-departure detector:</b> failed — {sd.error}
      </div>
    );
  }
  const rows = sd.rows || [];
  const c = sd.counts || {};
  const secs = (sd.scan_sec || 0) + (sd.verify_sec || 0);
  const pill = (v) =>
    v === "both" ? "ok" : v === "ball only" ? "warn" : "warn";

  return (
    <div
      className="tiny"
      style={{
        marginTop: 8, padding: "6px 8px", borderRadius: 6,
        background: "var(--surface-2)", border: "1px solid var(--line)",
      }}
    >
      <div className="row" style={{ justifyContent: "space-between", gap: 8 }}>
        <b>Swing detection: pose vs the ball leaving</b>
        <button className="btn tiny ghost" onClick={() => setOpen((v) => !v)}>
          {open ? "hide detail" : "show detail"}
        </button>
      </div>
      <div style={{ marginTop: 3 }}>
        <span className="pill">{sd.n_pose} pose</span>{" "}
        <span className="pill">{sd.n_ball} ball departure
          {sd.n_ball === 1 ? "" : "s"}</span>{" "}
        <span className={`pill ${sd.n_matched ? "ok" : "warn"}`}>
          {sd.n_matched} agree
        </span>{" "}
        {sd.n_pose_only > 0 && (
          <span className="pill warn">{sd.n_pose_only} pose only</span>
        )}{" "}
        {sd.n_ball_only > 0 && (
          <span className="pill warn">{sd.n_ball_only} ball only</span>
        )}
        <span className="muted" style={{ marginLeft: 8 }}>
          {secs.toFixed(2)}s ({sd.scan_sec}s to scan the clip,{" "}
          {sd.verify_sec}s to pin the impact frames)
        </span>
      </div>
      <div className="muted" style={{ marginTop: 3 }}>
        {sd.reason}
      </div>
      <div className="muted" style={{ marginTop: 2 }}>
        <b>Looking in:</b>{" "}
        {sd.roi_source || "the whole frame"}
        {sd.roi && (
          <> — x {Math.round(sd.roi.x * 100)}% y {Math.round(sd.roi.y * 100)}%,{" "}
            {Math.round(sd.roi.w * 100)}%×{Math.round(sd.roi.h * 100)}%</>
        )}
      </div>
      {sd.roi_note && (
        <div style={{ marginTop: 2, color: "var(--danger,#c0392b)" }}>
          {sd.roi_note}
        </div>
      )}

      {rows.length === 0 && (
        <div className="muted" style={{ marginTop: 4 }}>
          Neither detector found anything in this clip.
        </div>
      )}
      {rows.length > 0 && (
        <table style={{ width: "100%", marginTop: 6, borderCollapse: "collapse" }}>
          <thead>
            <tr className="muted" style={{ textAlign: "left" }}>
              <th style={{ padding: "2px 4px" }}>at</th>
              <th style={{ padding: "2px 4px" }}>found by</th>
              <th style={{ padding: "2px 4px" }}>pose peak</th>
              <th style={{ padding: "2px 4px" }}>ball left</th>
              <th style={{ padding: "2px 4px" }}>impact frame</th>
              <th style={{ padding: "2px 4px" }}>ball at</th>
              <th style={{ padding: "2px 4px" }}>rest</th>
              <th style={{ padding: "2px 4px" }}>AI judge</th>
              <th style={{ padding: "2px 4px" }}>preview</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, k) => (
              <tr key={k} style={{ borderTop: "1px solid var(--line)" }}>
                <td style={{ padding: "2px 4px" }}>{r.t}s</td>
                <td style={{ padding: "2px 4px" }}>
                  <span className={`pill ${pill(r.verdict)}`}>{r.verdict}</span>
                  {r.dt != null && (
                    <span className="muted"> {r.dt}s apart</span>
                  )}
                  {r.proposed_by_ball && (
                    <div className="tiny muted"
                         title="pose never fired here — the ball leaving proposed this swing, and it went through every stage on that basis">
                      ran on the ball's evidence
                    </div>
                  )}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {r.pose ? (
                    <>
                      {r.pose.t}s
                      {r.pose.gate_ok === false && (
                        <span className="muted"> ({r.pose.gate})</span>
                      )}
                    </>
                  ) : (
                    <span className="muted">—</span>
                  )}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {r.ball ? `${r.ball.t}s` : <span className="muted">—</span>}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {!r.ball ? (
                    <span className="muted">—</span>
                  ) : r.ball.impact_frame != null ? (
                    <>
                      f{r.ball.impact_frame}
                      {r.ball.image && (
                        <button
                          className="btn tiny ghost"
                          style={{ marginLeft: 4 }}
                          onClick={() => setShot(r.ball)}
                        >
                          strip
                        </button>
                      )}
                    </>
                  ) : (
                    <span
                      className="pill warn"
                      title={r.ball.verify_reason || ""}
                    >
                      not pinned
                    </span>
                  )}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {/* Two different answers to the same question, shown
                      together: the departure detector's, and stage 2's
                      club-arc one. They disagree often enough that
                      collapsing them hides the disagreement. */}
                  {r.ball ? (
                    <span title="where the resting-ball detector saw it leave">
                      {r.ball.x},{r.ball.y}
                    </span>
                  ) : (
                    <span className="muted">—</span>
                  )}
                  {r.ball_hint && (
                    <div className="tiny muted"
                         title="where stage 2 put it, from the bottom of the club arc">
                      arc {Math.round(r.ball_hint[0])},{Math.round(r.ball_hint[1])}
                    </div>
                  )}
                  {r.ball_final && !r.ball_hint && (
                    <div className="tiny muted" title="the ball the tracer used">
                      used {Math.round(r.ball_final[0])},{Math.round(r.ball_final[1])}
                    </div>
                  )}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {r.ball?.rest_sec != null
                    ? `${r.ball.rest_sec}s`
                    : <span className="muted">—</span>}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {r.not_processed ? (
                    <span className="pill warn" title={r.not_processed}>
                      never judged
                    </span>
                  ) : !r.judge ? (
                    <span className="muted">—</span>
                  ) : r.judge.ai_judge == null ? (
                    <span className="muted" title={r.judge.ai_reason || r.judge.verdict || ""}>
                      {r.judge.verdict || "not judged"}
                    </span>
                  ) : (
                    <span
                      className={`pill ${r.judge.ai_judge ? "ok" : (r.judge.unsure ? "warn" : "bad")}`}
                      title={r.judge.ai_reason || ""}
                    >
                      {r.judge.ai_judge ? "swing" : "not a swing"}
                      {r.judge.ai_confidence ? ` (${r.judge.ai_confidence})` : ""}
                      {r.judge.dropped ? " · dropped" : ""}
                      {!r.judge.dropped && r.judge.unsure ? " · unsure, kept" : ""}
                    </span>
                  )}
                </td>
                <td style={{ padding: "2px 4px" }}>
                  {r.not_processed ? (
                    <span className="tiny muted" title={r.not_processed}>
                      pose never fired here
                    </span>
                  ) : !r.preview ? (
                    <span className="muted">—</span>
                  ) : r.preview.clip_url ? (
                    <a href={r.preview.clip_url} target="_blank" rel="noreferrer">
                      clip
                    </a>
                  ) : (
                    <span className="muted" title={r.preview.error || ""}>
                      {r.preview.error ? "failed" : "none"}
                    </span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      {open && (
        <div className="muted" style={{ marginTop: 6 }}>
          {/* Why it found nothing is a different question from how many it
              found, and these four numbers are the only things that answer
              it: no ball-like blob at all, plenty but none in the box,
              plenty in the box but none that held still, or all of the
              above and none of them ever left. */}
          <div>
            <b>Scan:</b> {c.n_cand_total ?? 0} ball-like blob
            {c.n_cand_total === 1 ? "" : "s"} in the frame ·{" "}
            {c.n_cand_in_roi ?? 0} inside the box · {c.n_tracks ?? 0} track
            {c.n_tracks === 1 ? "" : "s"} · {c.n_rested ?? 0} held still for{" "}
            {c.min_rest_sec ?? "?"}s · {c.n_raw_departures ?? 0} departure
            {c.n_raw_departures === 1 ? "" : "s"} before merging ·{" "}
            sampled at {Math.round(c.eff_hz || 0)}Hz over{" "}
            {Math.round(c.duration_sec || 0)}s
          </div>
          {rows.filter((r) => r.ball).map((r, k) => (
            <div key={k} style={{ marginTop: 3 }}>
              <b>{r.ball.t}s:</b> {r.ball.verify_reason || "not checked"}
              {r.ball.present_ratio_pre != null && (
                <> · ball on the spot in{" "}
                  {Math.round(r.ball.present_ratio_pre * 100)}% of the frames
                  before impact</>
              )}
              {r.ball.snap_px != null && (
                <> · snapped {r.ball.snap_px}px</>
              )}
            </div>
          ))}
        </div>
      )}

      {shot?.image && (
        <div
          onClick={() => setShot(null)}
          style={{
            position: "fixed", inset: 0, zIndex: 1200, cursor: "zoom-out",
            background: "rgba(0,0,0,0.82)", display: "flex",
            alignItems: "center", justifyContent: "center", padding: 20,
          }}
        >
          <div onClick={(e) => e.stopPropagation()} style={{ maxWidth: "96vw" }}>
            <div className="tiny" style={{ color: "#fff", marginBottom: 6 }}>
              Departure at {shot.t}s — each tile is the tee spot on one frame,
              green while the ball is there, red once it is gone.
            </div>
            <img
              src={shot.image}
              alt="ball departure film-strip"
              style={{ maxWidth: "96vw", maxHeight: "82vh", display: "block" }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * Swing test — one question, isolated: did a ball sit somewhere in this
 * clip and then leave?
 *
 * Three answers, in the order they can fail: WHERE it looked (the tee-box
 * search area on a real frame, with every ball-like blob it saw), WHETHER
 * it found a ball (a blob that held still, ringed where it sat), and
 * WHETHER that ball left and on WHICH FRAME (the 15Hz scan re-watched at
 * full rate, with the film-strip that shows the call).
 */
// An ISO instant as a readable wall clock. The cameras are synced on real
// time, so real time is what the panel has to show -- seconds-into-the-clip
// look identical for two files that started seconds apart.
function clockOf(iso) {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  const p = (n, w = 2) => String(n).padStart(w, "0");
  return `${p(d.getHours())}:${p(d.getMinutes())}:${p(d.getSeconds())}`
    + `.${p(d.getMilliseconds(), 3)}`;
}

function SwingTestModal({ state, onClose, adminPassword, onRerun }) {
  // Hooks before the early return: this component re-renders on every
  // poll tick, and a conditional hook order would blow up mid-run.
  const [draft, setDraft] = useState(null);      // {x,y,w,h} fractions
  const [calibrating, setCalibrating] = useState(false);
  const [calibrated, setCalibrated] = useState(null);  // measured px
  // Show the frame with nothing drawn on it, to check the picture
  // rather than the detector's account of it.
  const [bareFrame, setBareFrame] = useState(false);
  const [diag, setDiag] = useState(null);              // why-not-found
  const [producing, setProducing] = useState(false);
  const [produced, setProduced] = useState(null);
  const [produceErr, setProduceErr] = useState(null);
  const [dragFrom, setDragFrom] = useState(null);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [saveErr, setSaveErr] = useState(null);
  const imgRef = useRef(null);

  const rep = state?.report;
  const deps = rep?.departures || [];
  const c = rep?.counts || {};
  const canSaveBox = !!(rep?.course_id && rep?.hole && rep?.day);
  const expectR = rep?.expect_radius_px;

  // Pointer position as a fraction of the displayed image, clamped so a
  // drag that leaves the picture still yields a box inside the frame.
  function frac(e) {
    const r = imgRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
  }
  function onDown(e) {
    const p = frac(e);
    if (!p) return;
    e.preventDefault();
    setDragFrom(p);
    setDraft({ x: p.x, y: p.y, w: 0, h: 0 });
    setSaved(false);
    setSaveErr(null);
  }
  function onMove(e) {
    if (!dragFrom) return;
    const p = frac(e);
    if (!p) return;
    setDraft({
      x: Math.min(dragFrom.x, p.x),
      y: Math.min(dragFrom.y, p.y),
      w: Math.abs(p.x - dragFrom.x),
      h: Math.abs(p.y - dragFrom.y),
    });
  }
  function onUp(e) {
    setDragFrom(null);
    // DRAG sets the search box. CLICK calibrates the ball SIZE -- it
    // does not pin a position, because the ball moves: every golfer
    // tees it somewhere else. The click only says "this blob is a
    // ball", and the server measures how many pixels across it is.
    // A 0x0 "box" is what a click looks like to a drag handler.
    setDraft((d) => {
      if (d && d.w > 0.01 && d.h > 0.01) return d;
      const p = e ? frac(e) : null;
      if (p) calibrate(p);
      return null;
    });
  }

  function onCancel() {
    setDragFrom(null);
    setDraft((d) => (d && d.w > 0.01 && d.h > 0.01 ? d : null));
  }

  // A click does BOTH: measures the ball for calibration, and asks the
  // detector why it did not find one there. The second is the useful
  // half when the panel says it found nothing.
  async function produce(idx) {
    setProducing(true);
    setProduceErr(null);
    setProduced(null);
    try {
      setProduced(await api.swingTestProduce(adminPassword, state.uploadId, idx));
    } catch (e) {
      setProduceErr(e.message);
    } finally {
      setProducing(false);
    }
  }

  async function calibrate(p) {
    setCalibrating(true);
    setSaveErr(null);
    setCalibrated(null);
    setDiag(null);
    try {
      const r = await api.calibrateBall(adminPassword, state.uploadId, p);
      setCalibrated(r);
      setSaved(true);
    } catch (e) {
      setSaveErr(e.message);
    }
    try {
      setDiag(await api.diagnoseBall(adminPassword, state.uploadId, p));
    } catch (e) {
      setDiag({ verdict: `Diagnosis failed: ${e.message}` });
    } finally {
      setCalibrating(false);
    }
  }

  async function save(patch) {
    setSaving(true);
    setSaveErr(null);
    try {
      await api.setTeeBox(adminPassword, rep.course_id, {
        hole: rep.hole, day: rep.day, ...patch,
      });
      setSaved(true);
      if (patch.roi === null) setDraft(null);
    } catch (e) {
      setSaveErr(e.message);
    } finally {
      setSaving(false);
    }
  }

  if (!state) return null;
  const Img = ({ url, cap }) =>
    url ? (
      <figure style={{ margin: "8px 0" }}>
        <a href={url} target="_blank" rel="noreferrer">
          <img src={url} alt={cap} style={{ width: "100%", borderRadius: 6 }} />
        </a>
        <figcaption className="tiny muted" style={{ marginTop: 4 }}>{cap}</figcaption>
      </figure>
    ) : null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)",
        display: "flex", alignItems: "flex-start", justifyContent: "center",
        zIndex: 1000, padding: 16, overflow: "auto",
      }}
    >
      <div
        className="card"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 1100, width: "100%", margin: 0 }}
      >
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>⛳ Swing test — upload #{state.uploadId}</b>
          <button type="button" className="ghost small"
            style={{ width: "auto" }} onClick={onClose}>
            Close ✕
          </button>
        </div>

        {state.running && (
          <div style={{ marginTop: 10 }}>
            <div className="small">
              <span
                className="shimmer"
                style={{
                  display: "inline-block", width: 12, height: 12,
                  borderRadius: "50%", marginRight: 8,
                  verticalAlign: "middle",
                }}
              />
              Running{state.stage ? ` — ${state.stage}` : ""}
              {state.total ? ` (${state.done}/${state.total})` : ""}. The scan
              reads the whole clip at 15Hz, then each departure is re-watched
              frame by frame.
            </div>
          </div>
        )}
        {state.error && (
          <div className="err-text small" style={{ marginTop: 8 }}>
            {state.error}
          </div>
        )}

        {rep && (
          <div style={{ marginTop: 12 }}>
            <div
              className="small"
              style={{
                padding: "8px 10px", borderRadius: 6, marginBottom: 10,
                background: deps.length ? "rgba(0,160,80,0.14)" : "rgba(200,120,0,0.16)",
              }}
            >
              <b>{rep.verdict}</b>
              <div className="tiny muted" style={{ marginTop: 3 }}>
                {rep.duration_sec != null && <>{rep.duration_sec}s clip · </>}
                {rep.fps} fps · scanned at {Math.round(rep.sample_hz || 0)}Hz ·
                {" "}scan {rep.scan_sec}s + frame check {rep.verify_sec}s
              </div>
            </div>

            {/* TIME PER STEP, the same shape Debug3 reports: which step
                cost the run is not guessable from the source, and it is
                what decides where an optimisation is worth anything. */}
            {rep.steps?.length > 0 && (
              <div style={{ marginBottom: 14 }}>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Time per step · {rep.total_sec}s total
                </div>
                {rep.steps.map((st) => (
                  <div key={st.n} className="small"
                    style={{
                      display: "flex", gap: 10, padding: "6px 0",
                      borderBottom: "1px solid var(--line)",
                      opacity: st.seconds > 0 ? 1 : 0.5,
                    }}
                  >
                    <b style={{ minWidth: 18 }}>{st.n}</b>
                    <div style={{ flex: 1 }}>
                      <b>{st.name}</b>
                      <div className="tiny muted">{st.detail}</div>
                    </div>
                    <div style={{ textAlign: "right", minWidth: 120 }}>
                      <b>{st.count}</b>
                      <div className="tiny muted">{st.counts}</div>
                    </div>
                    <div style={{ textAlign: "right", minWidth: 86 }}>
                      <b>{st.seconds > 0 ? `${st.seconds}s` : "—"}</b>
                      <div className="tiny muted">
                        {st.seconds > 0 ? `${st.pct}%` : "did not run"}
                      </div>
                      <div style={{
                        height: 3, borderRadius: 2, marginTop: 2,
                        background: "var(--line)",
                      }}>
                        <div style={{
                          height: "100%", borderRadius: 2,
                          width: `${Math.min(100, st.pct || 0)}%`,
                          background: (st.pct || 0) >= 40
                            ? "var(--danger, #c0392b)"
                            : "var(--emerald-700, #16a34a)",
                        }} />
                      </div>
                    </div>
                  </div>
                ))}

                {/* 6 · PRODUCE. Analysis steps 1-5 describe the shot; this
                    is the one that makes the clip, and it belongs in the
                    same list rather than buried under the green images
                    where it was. It is an action, so it carries its own
                    button and reports what it built. */}
                {(() => {
                  // A missed green is still a shot worth a clip: produce
                  // needs a confirmed impact, not a landing. Without one
                  // the clip is tee-only and the tracer runs on to an
                  // assumed landing.
                  const dep = (rep.departures || []).find((d) => d.green?.cut)
                    || (rep.departures || []).find((d) => d.impact_frame != null);
                  const teeOnly = dep && !dep.green?.cut;
                  return (
                    <div className="small"
                      style={{
                        display: "flex", gap: 10, padding: "6px 0",
                        borderBottom: "1px solid var(--line)",
                        opacity: dep ? 1 : 0.5,
                      }}
                    >
                      <b style={{ minWidth: 18 }}>6</b>
                      <div style={{ flex: 1 }}>
                        <b>Produce</b>
                        <div className="tiny muted">
                          {teeOnly
                            ? "no landing on the green camera — tee-only, tracer carried on to an assumed landing"
                            : "tee tracer → cut to the green camera 1s before the ball lands → 3s of green"}
                          {" "}→ the usual graphics
                        </div>
                        {produced?.url && (
                          <div className="tiny" style={{ marginTop: 3 }}>
                            <a href={produced.url} target="_blank" rel="noreferrer">
                              <b>open the produced clip →</b>
                            </a>
                            {" "}tee {produced.tee_window_sec?.join("–")}s + green{" "}
                            {produced.green_window_sec?.join("–")}s
                            {produced.n_flight_points != null && (
                              <> · {produced.n_flight_points} tracer points</>
                            )}
                            {produced.assumed_landing && (
                              <> · assumed landing {produced.assumed_landing.join(", ")}</>
                            )}
                            {produced.reason && (
                              <div className="muted" style={{ marginTop: 2 }}>
                                {produced.reason}
                              </div>
                            )}
                          </div>
                        )}
                        {produceErr && (
                          <div className="err-text tiny" style={{ marginTop: 3 }}>
                            {produceErr}
                          </div>
                        )}
                      </div>
                      <div style={{ textAlign: "right", minWidth: 200 }}>
                        <button type="button" className="small"
                          style={{ width: "auto" }}
                          disabled={!dep || producing}
                          title={dep
                            ? "Build the clip this analysis describes"
                            : "Needs a departure with a confirmed impact frame"}
                          onClick={() => produce(dep?.idx ?? 0)}>
                          {producing ? "Producing…" : "🎬 Produce"}
                        </button>
                        <div className="tiny muted" style={{ marginTop: 2 }}>
                          {produced?.url
                            ? (produced.tee_only ? "built · tee only" : "built")
                            : dep ? (teeOnly ? "ready · tee only" : "ready")
                              : "no confirmed impact"}
                        </div>
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}

            {/* 1. WHERE IT LOOKED */}
            <h4 style={{ margin: "12px 0 4px" }}>1 · Where it looked</h4>
            <div className="tiny muted">
              {rep.whole_frame ? (
                <>
                  No tee box is set, so the search area is the{" "}
                  <b>whole frame</b> — shoes, cups and sky glints all compete
                  with the ball.
                </>
              ) : (
                <>
                  Search area from <b>{rep.roi_source || "an ROI"}</b>
                  {rep.roi_px && (
                    <> — {rep.roi_px.w}×{rep.roi_px.h}px at {rep.roi_px.x},{rep.roi_px.y}</>
                  )}
                  {rep.frame_w ? <> of a {rep.frame_w}×{rep.frame_h} frame</> : null}.
                </>
              )}
              {rep.roi_note && <div style={{ marginTop: 3 }}>{rep.roi_note}</div>}
            </div>
            {/* DRAW THE BOX HERE. The tee markers move every morning and
                a course has several par-3s, so the search area is set per
                hole per day -- once, on the day's first clip of that hole,
                and then every other clip that day reads it for free. */}
            {rep.area_image && (
              <figure style={{ margin: "8px 0" }}>
                {/* EVERY MARK ON THE PICTURE IS AN ASSERTION. The box, the
                    dots and the caption are all the detector's claims about
                    the frame, and the question they most often provoke --
                    is the ball really there? -- cannot be answered while
                    they cover it. Same pixels, drawn on or not. */}
                {rep.area_image_clean && (
                  <button
                    type="button"
                    className="secondary small"
                    onClick={() => setBareFrame((v) => !v)}
                    style={{ marginBottom: 6 }}
                    title={
                      bareFrame
                        ? "Put the search box, the accepted blobs and the caption back"
                        : "Hide every overlay and show the raw frame underneath"
                    }
                  >
                    {bareFrame ? "Show the graphics" : "Clear the graphics"}
                  </button>
                )}
                <div
                  style={{
                    position: "relative", lineHeight: 0,
                    cursor: canSaveBox ? "crosshair" : "default",
                    userSelect: "none",
                  }}
                  onMouseDown={canSaveBox ? onDown : undefined}
                  onMouseMove={canSaveBox ? onMove : undefined}
                  onMouseUp={canSaveBox ? onUp : undefined}
                  // Leaving the picture ABANDONS the gesture: routing this to
                  // onUp would read the exit point as a click and calibrate
                  // the ball size against whatever is on the frame edge.
                  onMouseLeave={canSaveBox ? onCancel : undefined}
                >
                  <img
                    ref={imgRef}
                    src={
                      bareFrame && rep.area_image_clean
                        ? rep.area_image_clean
                        : rep.area_image
                    }
                    alt="ball search area"
                    draggable={false}
                    style={{ width: "100%", borderRadius: 6 }}
                  />
                  {draft && (
                    <div
                      style={{
                        position: "absolute",
                        left: `${draft.x * 100}%`, top: `${draft.y * 100}%`,
                        width: `${draft.w * 100}%`, height: `${draft.h * 100}%`,
                        border: "3px dashed #00e0ff",
                        background: "rgba(0,224,255,0.12)",
                        pointerEvents: "none",
                      }}
                    />
                  )}
                  {calibrated?.at && rep.frame_size && (
                    <div
                      style={{
                        position: "absolute",
                        left: `${(calibrated.at[0] / rep.frame_size[0]) * 100}%`,
                        top: `${(calibrated.at[1] / rep.frame_size[1]) * 100}%`,
                        width: 22, height: 22, marginLeft: -11, marginTop: -11,
                        border: "2px solid #ffe000", borderRadius: "50%",
                        boxShadow: "0 0 0 1px rgba(0,0,0,0.6)",
                        pointerEvents: "none",
                      }}
                    />
                  )}
                </div>
                <figcaption className="tiny muted" style={{ marginTop: 4 }}>
                  Orange = the search area actually used for this run. Green
                  dots = every white, round, ball-sized blob the scan accepted
                  (before the box gate). Yellow ring = a ball that rested here
                  and then left.
                  {canSaveBox && (
                    <> <b>Click a ball</b> to calibrate its size for hole{" "}
                    {rep.hole}, or <b>drag</b> to set the search box for{" "}
                    {rep.day}.</>
                  )}
                </figcaption>
              </figure>
            )}

            {canSaveBox && (
              <>
                <div className="tiny muted" style={{ marginBottom: 6 }}>
                  <b>Size is the thing that separates a ball from a shoe.</b>{" "}
                  The ball moves — every golfer tees it up somewhere else — so
                  its position is never assumed. Its size is: the camera does
                  not move, so a ball is nearly the same number of pixels
                  across in every clip of this hole. Click one ball, once, and
                  the detector finds every ball automatically from then on.
                  The box is separate, and only narrows where it looks.
                </div>
                <div className="small" style={{ marginBottom: 6 }}>
                  {expectR ? (
                    <>Calibrated: looking for a ball of radius <b>{expectR}px</b>.</>
                  ) : (
                    <span className="muted">
                      Not calibrated for hole {rep.hole} yet — using generic
                      size limits derived from the frame height.
                    </span>
                  )}
                  {c.accept_radius_px && (
                    <> Accepting radius{" "}
                    <b>{c.accept_radius_px[0]}–{c.accept_radius_px[1]}px</b>
                    {c.native_scan
                      ? " scanned at full resolution inside the box."
                      : " scanned on a downscaled whole frame — draw a box to scan at full resolution."}</>
                  )}
                  {calibrated && (
                    <> <b style={{ color: "var(--ok, #2a8)" }}>
                      Measured {calibrated.measured_px}px radius — saved for hole {calibrated.hole}.
                    </b></>
                  )}
                  {calibrating && <> Measuring…</>}
                </div>
                {diag && (
                  <div className="small" style={{
                    marginBottom: 8, padding: "8px 10px", borderRadius: 6,
                    background: "rgba(120,120,120,0.12)",
                  }}>
                    <b>Why it did not find a ball there:</b> {diag.verdict}
                    {diag.probe?.samples?.length > 0 && (
                      <div className="tiny muted" style={{ marginTop: 6 }}>
                        {diag.probe.samples.slice(0, 4).map((sm, i) => (
                          <div key={i}>
                            t={sm.t}s · radius {sm.radius_native_px}px ·
                            {" "}{sm.pixels}px · box {sm.box_px?.join("×")} ·
                            {" "}aspect {sm.aspect} · extent {sm.extent} ·{" "}
                            {sm.accepted ? "accepted" : `rejected: ${sm.reason}`}
                          </div>
                        ))}
                        {diag.accept_radius_px && (
                          <div>accepted radius window:{" "}
                            {diag.accept_radius_px[0]}–{diag.accept_radius_px[1]}px</div>
                        )}
                      </div>
                    )}
                    {diag.probe?.pixel?.length > 0 && (
                      <div className="tiny muted" style={{ marginTop: 6 }}>
                        pixel there: {diag.probe.pixel.slice(0, 3).map((px, i) => (
                          <span key={i}>V={px.v} S={px.s} top-hat={px.tophat}{" "}</span>
                        ))}
                      </div>
                    )}
                  </div>
                )}
                <div className="row" style={{ gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                  <button
                    type="button" className="secondary small" style={{ width: "auto" }}
                    disabled={!draft || saving}
                    onClick={() => save({ roi: draft })}
                  >
                    {saving ? "Saving…" : `Save search box for ${rep.day}`}
                  </button>
                  {draft && (
                    <button type="button" className="ghost small"
                      style={{ width: "auto" }}
                      onClick={() => setDraft(null)}>
                      Discard box
                    </button>
                  )}
                  {saved && onRerun && (
                    <button type="button" className="small" style={{ width: "auto" }}
                      onClick={() => onRerun(state.uploadId)}>
                      Re-run swing test →
                    </button>
                  )}
                  <button
                    type="button" className="ghost small"
                    style={{ width: "auto", marginLeft: "auto" }}
                    disabled={saving}
                    onClick={() => save({ roi: null })}
                    title={`Forget the search box for hole ${rep.hole} on ${rep.day}`}
                  >
                    Clear this day&apos;s box
                  </button>
                </div>
              </>
            )}
            {saved && (
              <div className="tiny" style={{ color: "var(--ok, #2a8)", marginTop: 4 }}>
                Saved. Every clip of hole {rep.hole} on {rep.day} uses it — set
                once, not per video.
              </div>
            )}
            {saveErr && (
              <div className="err-text tiny" style={{ marginTop: 4 }}>{saveErr}</div>
            )}

            {/* 2. WHAT IT FOUND */}
            <h4 style={{ margin: "12px 0 4px" }}>2 · What it found</h4>
            <div className="small" style={{ marginBottom: 6 }}>
                {c.n_cand_total ?? 0} ball-like blob(s) in the frame ·{" "}
                {c.n_cand_in_roi ?? 0} inside the search area ·{" "}
                {c.n_tracks ?? 0} tracked ·{" "}
                <b>{c.n_rested ?? 0}</b> held still for {c.min_rest_sec}s ·{" "}
                <b>{deps.length}</b> departure(s)
              {c.n_drop_shape > 0 && (
                <> · {c.n_drop_shape} rejected as the wrong shape (shoes)</>
              )}
              {c.n_drop_size > 0 && (
                <> · {c.n_drop_size} rejected as the wrong size</>
              )}
            </div>
            {!deps.length && (
              <div className="tiny muted">{rep.reason}</div>
            )}

            {/* 2b. REST + BURST — the only two things being measured */}
            {(rep.rest_and_burst || []).length > 0 && (
              <div style={{ margin: "12px 0" }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <h4 style={{ margin: "12px 0 4px" }}>
                    Rest &amp; burst — every candidate
                  </h4>
                  <span className={rep.n_swings_found ? "pill ok" : "pill warn"}>
                    {rep.n_swings_found} swing(s) of{" "}
                    {rep.rest_and_burst.length} candidate(s)
                  </span>
                </div>
                <div style={{ overflowX: "auto" }}>
                  <table className="tiny" style={{ borderCollapse: "collapse", width: "100%" }}>
                    <thead>
                      <tr style={{ textAlign: "left", opacity: 0.7 }}>
                        <th style={{ padding: "4px 8px" }}>spot</th>
                        <th style={{ padding: "4px 8px" }}>ball there before</th>
                        <th style={{ padding: "4px 8px" }}>left on</th>
                        <th style={{ padding: "4px 8px" }}>burst peak</th>
                        <th style={{ padding: "4px 8px" }}>fall</th>
                        <th style={{ padding: "4px 8px" }}>verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {rep.rest_and_burst.map((r, i) => {
                        const b = r.burst || {};
                        const good = r.verdict === "swing";
                        return (
                          <tr
                            key={i}
                            style={{
                              borderTop: "1px solid rgba(128,128,128,0.25)",
                              background: good ? "rgba(0,160,80,0.10)" : "none",
                            }}
                            title={r.reason || ""}
                          >
                            <td style={{ padding: "4px 8px" }}>
                              <b>({r.x}, {r.y})</b>
                            </td>
                            <td style={{ padding: "4px 8px" }}>
                              {r.present_ratio_pre == null
                                ? "—"
                                : `${Math.round(r.present_ratio_pre * 100)}% of frames`}
                            </td>
                            <td style={{ padding: "4px 8px" }}>
                              {r.impact_frame == null
                                ? (r.late_departure_frame != null
                                    ? `f${r.late_departure_frame} (too late to confirm)`
                                    : "—")
                                : `f${r.impact_frame} (${r.impact_sec}s)`}
                            </td>
                            <td style={{ padding: "4px 8px" }}>
                              {b.peak == null ? "—" : `${b.peak}%`}
                            </td>
                            <td style={{ padding: "4px 8px" }}>
                              {b.fall == null ? "—" : `${b.fall}x`}
                              {b.verdict === "quiet" && " (quiet)"}
                            </td>
                            <td style={{ padding: "4px 8px" }}>
                              <b style={{ color: good ? "#0a0" : undefined }}>
                                {r.verdict}
                              </b>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="tiny muted" style={{ marginTop: 6 }}>
                  "ball there before" is how much of the second before it left
                  actually showed a ball. "fall" is how far the motion around
                  the spot dropped back after the peak — a strike is 50x or
                  more, something walking away is under 4x. Hover a row for the
                  full sentence.
                </div>
              </div>
            )}

            {/* 2c. THE GREEN AS A CLOCK */}
            {rep.green_descents && (
              <div
                className="card"
                style={{ margin: "10px 0", padding: 10 }}
              >
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <b>Working backwards from the green</b>
                  <span
                    className={
                      rep.green_descents.n_descents ? "pill ok" : "pill warn"
                    }
                  >
                    {rep.green_descents.n_descents ?? 0} descent(s)
                  </span>
                </div>
                <div className="tiny muted" style={{ marginTop: 4 }}>
                  {rep.green_descents.reason}
                </div>
                {(rep.green_descents.swing_windows || []).map((w, i) => (
                  <div key={i} className="tiny" style={{ marginTop: 4 }}>
                    Landed{" "}
                    <b>{w.landed_at ? clockOf(w.landed_at) : `${w.green_sec}s in`}</b>
                    {w.landing_xy && (
                      <> at ({w.landing_xy[0]}, {w.landing_xy[1]})</>
                    )}
                    {" "}(green f{w.green_frame})
                    {w.swing_from_at ? (
                      <>
                        {" "}— so the swing is between{" "}
                        <b>{clockOf(w.swing_from_at)}</b> and{" "}
                        <b>{clockOf(w.swing_to_at)}</b>, which is{" "}
                        {w.swing_from_sec}s–{w.swing_to_sec}s (f
                        {w.swing_from_frame}–f{w.swing_to_frame}) on the tee
                        clip.
                      </>
                    ) : (
                      <> — no wall clock on this pair, so the landing cannot
                      be placed on the tee clip's timeline.</>
                    )}
                  </div>
                ))}
                <div className="tiny muted" style={{ marginTop: 4 }}>
                  Tee {rep.green_descents.tee_fps}fps started{" "}
                  {rep.green_descents.tee_started_at
                    ? clockOf(rep.green_descents.tee_started_at)
                    : "at an unknown time"}
                  {" · "}green {rep.green_descents.green_fps}fps started{" "}
                  {rep.green_descents.green_started_at
                    ? clockOf(rep.green_descents.green_started_at)
                    : "at an unknown time"}
                  {rep.green_descents.clock_ok ? (
                    <> — matched on the clock, not on clip position.</>
                  ) : (
                    <>
                      {" "}— <b>NO WALL CLOCK</b> on this pair. The two clips
                      cannot be lined up, because they neither start together
                      nor run at the same rate.
                    </>
                  )}
                </div>
              </div>
            )}

            {/* 3. DID IT LEAVE, AND WHEN */}
            {deps.length > 0 && (
              <>
                <h4 style={{ margin: "14px 0 4px" }}>3 · Did it leave, and when</h4>
                {deps.map((b) => (
                  <div
                    key={b.idx}
                    className="card"
                    style={{ margin: "8px 0", padding: 10 }}
                  >
                    <div className="row" style={{ justifyContent: "space-between" }}>
                      <b>
                        Ball {b.idx + 1} — rested {b.rest_sec}s at ({b.x}, {b.y})
                        {b.green_landing_sec != null && (
                          <span className="pill ok" style={{ marginLeft: 6 }}>
                            lands on the green at {b.green_landing_sec}s
                          </span>
                        )}
                        {b.club_rate != null && (
                          <span className="tiny muted" style={{ marginLeft: 6 }}>
                            · club at address on{" "}
                            {Math.round(b.club_rate * 100)}% of the frames
                            before it emptied
                          </span>
                        )}
                      </b>
                      <span className={b.departed ? "pill ok" : "pill warn"}>
                        {b.departed
                          ? `left on frame ${b.impact_frame} (${b.impact_sec}s)`
                          : "departure not confirmed"}
                      </span>
                    </div>
                    {b.spot_departures && (
                      <div className="tiny muted" style={{ marginTop: 4 }}>
                        <b>Watched this spot across the whole clip</b> at full
                        rate: {b.spot_departures.length} departure(s)
                        {b.spot_peak != null && (
                          <> · ball reads {b.spot_peak} against a measured
                          threshold of {b.spot_threshold}</>
                        )}
                        {b.spot_departures.length > 0 && (
                          <> — {b.spot_departures.map((d) => `f${d.frame} (sat ${d.rest_sec}s)`).join(", ")}</>
                        )}
                        {b.spot_reason && <> — {b.spot_reason}</>}
                      </div>
                    )}
                    <div className="tiny muted" style={{ marginTop: 4 }}>
                      Departure frame {b.frame} ({b.t_sec}s), taken from the
                      spot walk rather than the scan.{" "}
                      {b.departed
                        ? `The frame-by-frame check pins the disappearance at frame ${b.impact_frame}.`
                        : "The frame-by-frame check could not confirm it."}
                      {b.snap_px != null && <> Rest position snapped {b.snap_px}px.</>}
                      {b.verify_reason && <> — {b.verify_reason}</>}
                    </div>
                    <Img url={b.rest_image} cap={`Ball ${b.idx + 1} where it sat, before the strike.`} />
                    <Img
                      url={b.strip_image}
                      cap="Frame by frame at the rest spot. Green = ball present, red = gone, yellow box = the frame it disappeared."
                    />

                    {/* THE GREEN CAMERA. The tee stages watch the ball
                        leave; this watches it arrive on the other camera,
                        4-8s later, where the picture is far quieter. */}
                    {b.green && (
                      <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid var(--line)" }}>
                        <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                          Green camera · descent
                          {b.green.window && <> · frames {b.green.window.join("–")}</>}
                          {b.green.seconds != null && <> · {b.green.seconds}s</>}
                        </div>
                        <div className="tiny muted" style={{ marginBottom: 6 }}>
                          4–8s after impact, in real time. Tee impact f{b.impact_frame} →
                          {b.green.window_sec && <> green {b.green.window_sec[0]}–{b.green.window_sec[1]}s</>}
                          {b.green.green_fps && <> at {b.green.green_fps}fps</>}
                          {b.green.delta_sec != null && (
                            <> · offset {b.green.delta_sec}s{" "}
                              <b className={b.green.delta_source === "camera_event" ? "" : "err-text"}>
                                ({b.green.delta_source === "camera_event"
                                  ? "measured from the camera clocks"
                                  : b.green.delta_source === "edit_metrics"
                                    ? "from a saved offset"
                                    : "ASSUMED ZERO — the two files are treated as starting together"})
                              </b>
                            </>
                          )}
                        </div>
                        {b.green.stats?.frames != null && (
                          <div className="small" style={{ marginBottom: 4 }}>
                            {b.green.stats.components ?? 0} components ·{" "}
                            <b>{b.green.stats.kept ?? 0} kept</b> ·{" "}
                            {b.green.n_tracks ?? 0} tracks ·{" "}
                            <b>{b.green.descents?.length ?? 0} descending</b>
                          </div>
                        )}
                        {b.green.reason && (
                          <div className="tiny muted" style={{ marginBottom: 6 }}>{b.green.reason}</div>
                        )}
                        <Img url={b.green.frame_image}
                          cap="One green-camera frame: red = masked, green = ball-sized blobs kept." />
                        <Img url={b.green.dets_image}
                          cap="Every detection in the 4–8s window, coloured by time — blue early, orange late." />
                        <Img url={b.green.tracks_image}
                          cap="Descent candidates. Hollow ring = first frame, filled dot = last." />

                        {/* Where it pitched and where it finished, which is
                            the part a viewer actually cares about. */}
                        {b.green.landing_frame != null && (
                          <div className="small" style={{
                            padding: "8px 10px", borderRadius: 6, margin: "6px 0",
                            background: "rgba(0,160,80,0.14)",
                          }}>
                            {/* REAL CLOCK TIME. Seconds-into-the-clip is the
                                wrong unit for checking sync: two files that
                                start at different instants both begin at 0.0s,
                                so the numbers agree while the moments do not.
                                The wall clock is what the cameras are aligned
                                to, so it is what is shown. */}
                            <div className="tiny muted" style={{ marginBottom: 3 }}>
                              IMPACT · tee frame <b>{b.impact_frame}</b>
                              {b.green.impact_at
                                ? <> · <b>{clockOf(b.green.impact_at)}</b></>
                                : (b.green.impact_sec_tee != null
                                   && <> · {b.green.impact_sec_tee}s into the tee clip</>)}
                            </div>
                            <div>
                              <span style={{ color: "var(--danger, #c0392b)" }}>●</span>{" "}
                              <b>LANDED</b> · green frame <b>{b.green.landing_frame}</b>
                              {b.green.landing_at
                                ? <> · <b>{clockOf(b.green.landing_at)}</b></>
                                : (b.green.landing_sec != null && <> · {b.green.landing_sec}s into the green clip</>)}
                              {b.green.landing_xy && <> · at <b>{b.green.landing_xy.join(", ")}</b></>}
                            </div>
                            {b.green.rest_frame != null && (
                              <div style={{ marginTop: 3 }}>
                                <span style={{ color: "var(--emerald-700, #16a34a)" }}>●</span>{" "}
                                <b>AT REST</b> · green frame <b>{b.green.rest_frame}</b>
                                {b.green.rest_at
                                  ? <> · <b>{clockOf(b.green.rest_at)}</b></>
                                  : (b.green.rest_sec != null && <> · {b.green.rest_sec}s into the green clip</>)}
                                {b.green.rest_xy && <> · at <b>{b.green.rest_xy.join(", ")}</b></>}
                              </div>
                            )}
                            {/* THE SYNC CHECK. Flight time is landing minus
                                impact in ONE timeline. A golf shot hangs
                                4-8s; anything outside that is the camera
                                offset, not the detector. */}
                            {b.green.landing_after_impact != null && (
                              <div style={{
                                marginTop: 4, paddingTop: 4,
                                borderTop: "1px solid rgba(0,0,0,0.12)",
                              }}>
                                <b>Hang time {b.green.landing_after_impact}s</b>
                                {" "}
                                <span className={
                                  b.green.landing_after_impact >= 2
                                  && b.green.landing_after_impact <= 10
                                    ? "pill ok" : "pill err"}>
                                  {b.green.landing_after_impact >= 2
                                   && b.green.landing_after_impact <= 10
                                    ? "plausible"
                                    : "IMPLAUSIBLE — check the camera offset"}
                                </span>
                                {b.green.rest_after_impact != null && (
                                  <span className="muted">
                                    {" "}· at rest {b.green.rest_after_impact}s after impact
                                  </span>
                                )}
                              </div>
                            )}
                            {b.green.cut && (
                              <div style={{ marginTop: 6 }}>
                                <button type="button" className="small"
                                  style={{ width: "auto" }}
                                  disabled={producing}
                                  onClick={() => produce(b.idx)}>
                                  {producing ? "Producing…" : "🎬 Produce this swing"}
                                </button>
                                {produced?.url && (
                                  <div className="tiny" style={{ marginTop: 4 }}>
                                    <a href={produced.url} target="_blank" rel="noreferrer">
                                      produced clip
                                    </a>
                                    {" "}— tee {produced.tee_window_sec?.join("–")}s
                                    {" "}+ green {produced.green_window_sec?.join("–")}s
                                    {produced.n_flight_points != null && (
                                      <> · {produced.n_flight_points} tracer points</>
                                    )}
                                  </div>
                                )}
                                {produceErr && (
                                  <div className="err-text tiny" style={{ marginTop: 4 }}>
                                    {produceErr}
                                  </div>
                                )}
                              </div>
                            )}
                            {b.green.cut && (
                              <div className="tiny" style={{
                                marginTop: 5, paddingTop: 5,
                                borderTop: "1px solid rgba(0,0,0,0.12)",
                              }}>
                                <b>PRODUCTION CUT</b> — switch to the green camera
                                1s before the ball lands, hold it 3s:
                                <div style={{ marginTop: 2 }}>
                                  cut at <b>{b.green.cut.at ? clockOf(b.green.cut.at) : `${b.green.cut.at_green_sec}s green`}</b>
                                  {" "}· tee clip {b.green.cut.at_tee_sec}s
                                  {" "}· green clip {b.green.cut.at_green_sec}s
                                  {" "}→ {b.green.cut.green_window_sec?.[1]}s
                                </div>
                                {!b.green.green_started_at && (
                                  <div className="muted" style={{ marginTop: 2 }}>
                                    no camera clocks on this upload — the cut is
                                    computed with the offset assumed zero
                                  </div>
                                )}
                              </div>
                            )}
                            {b.green.ground_path?.length > 0 && (
                              <div className="tiny muted" style={{ marginTop: 3 }}>
                                {b.green.ground_path.length} frames of bounce and roll between them
                                {b.green.landing_xy && b.green.rest_xy && (
                                  <> · travelled{" "}
                                    {Math.round(Math.hypot(
                                      b.green.rest_xy[0] - b.green.landing_xy[0],
                                      b.green.rest_xy[1] - b.green.landing_xy[1],
                                    ))}px after pitching</>
                                )}
                              </div>
                            )}
                          </div>
                        )}
                        {b.green.follow_reason && b.green.landing_frame == null && (
                          <div className="tiny muted" style={{ marginBottom: 4 }}>
                            {b.green.follow_reason}
                          </div>
                        )}
                        <Img url={b.green.path_image}
                          cap="Orange = descent, red cross = landing, yellow = bounce and roll, green ring = at rest. Rest is where the detections stop: MOG2 sees motion, so a settled ball produces nothing." />
                        {b.green.descents?.length > 0 && (
                          <div style={{ overflowX: "auto" }}>
                            <table className="tiny" style={{ borderCollapse: "collapse", marginTop: 4 }}>
                              <thead><tr style={{ textAlign: "left" }}>
                                <th style={{ paddingRight: 10 }}>frames</th>
                                <th style={{ paddingRight: 10 }}>from → to</th>
                                <th style={{ paddingRight: 10 }}>fell</th>
                                <th style={{ paddingRight: 10 }}>span</th>
                                <th>points</th>
                              </tr></thead>
                              <tbody>
                                {b.green.descents.map((d, di) => (
                                  <tr key={di}>
                                    <td style={{ paddingRight: 10 }}>{d.frames?.join("–")}</td>
                                    <td style={{ paddingRight: 10 }}>{d.from?.join(",")} → {d.to?.join(",")}</td>
                                    <td style={{ paddingRight: 10 }}>{d.fall_px}px</td>
                                    <td style={{ paddingRight: 10 }}>{d.span_px}px</td>
                                    <td>{d.n_points}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </table>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Debug3's stages 4, 5 and 6, from one MOG2 pass over
                        the 3 seconds after impact -- the only stretch of
                        the clip with a ball in the air. */}
                    {b.stages && !b.stages.error && (
                      <div style={{ marginTop: 10, paddingTop: 8, borderTop: "1px solid var(--line)" }}>
                        <div className="tiny upper muted" style={{ marginBottom: 6 }}>
                          Flight pipeline · frames{" "}
                          {b.stages.window ? b.stages.window.join("–") : "?"} (3s after impact)
                          {b.stages.seconds != null && <> · {b.stages.seconds}s</>}
                        </div>

                        <div className="small"><b>4 · MOG2 + component + area filter</b></div>
                        <div className="tiny muted">
                          Big blobs become a golfer mask; only ball-sized off-body blobs survive.
                        </div>
                        {b.stages.detect?.stats && (
                          <div className="small" style={{ margin: "4px 0 2px" }}>
                            {b.stages.detect.stats.components ?? 0} components ·{" "}
                            {b.stages.detect.stats.on_golfer ?? 0} on the golfer ·{" "}
                            {b.stages.detect.stats.too_big ?? 0} too big ·{" "}
                            <b>{b.stages.detect.stats.kept ?? 0} kept</b>
                          </div>
                        )}
                        <Img url={b.stages.detect?.frame_image}
                          cap="One frame: red = golfer mask (excluded), green = ball-sized blobs kept." />
                        <Img url={b.stages.detect?.dets_image}
                          cap="Every kept detection, coloured by time — blue early, orange late." />

                        <div className="small" style={{ marginTop: 10 }}>
                          <b>5 · Nearest-neighbour tracking</b>{" "}
                          <span className="muted">— {b.stages.tracks?.n ?? 0} tracks built</span>
                        </div>
                        <div className="tiny muted">
                          Constant-velocity prediction with a gate that widens on a missed frame.
                        </div>
                        <Img url={b.stages.tracks?.image}
                          cap="The tracks it linked. One object should yield one track." />

                        <div className="small" style={{ marginTop: 10 }}>
                          <b>6 · RANSAC parabola + flight tests</b>{" "}
                          <span className={b.stages.flight?.ok ? "pill ok" : "pill warn"}>
                            {b.stages.flight?.ok ? "flight found" : "no flight"}
                          </span>
                        </div>
                        <div className="tiny muted">
                          x linear in t, y quadratic; must rise and must point back at the ball.
                        </div>
                        {b.stages.flight?.ok && (
                          <div className="small" style={{ margin: "4px 0 2px" }}>
                            {b.stages.flight.n_inliers} inliers ·{" "}
                            {b.stages.flight.rms_px}px rms ·{" "}
                            {b.stages.flight.n_points} tracer points · launch f
                            {b.stages.flight.launch_frame}
                          </div>
                        )}
                        {b.stages.flight?.reason && (
                          <div className="tiny muted" style={{ marginBottom: 4 }}>
                            {b.stages.flight.reason}
                          </div>
                        )}
                        <Img url={b.stages.flight?.image}
                          cap="The fitted flight over the frame it was measured on." />

                        {/* The measured flight carried on to the landing the
                            green camera found. */}
                        {b.bezier && (
                          <div style={{ marginTop: 8 }}>
                            <div className="small">
                              <b>Continuation to the landing</b>{" "}
                              <span className={b.bezier.ok ? "pill ok" : "pill warn"}>
                                {b.bezier.ok ? "projected" : "not projected"}
                              </span>
                            </div>
                            <div className="tiny muted">
                              A quadratic Bézier is a parabola: it leaves the last
                              measured point along the direction the ball was already
                              travelling, rises to an apex, and comes down to the
                              landing. Direction is a regression over the last several
                              points, not the final two — at 50fps a centroid jitters a
                              couple of pixels, and two points multiply that across the
                              whole projection.
                            </div>
                            {b.bezier.reason && (
                              <div className="tiny muted" style={{ marginTop: 3 }}>
                                {b.bezier.reason}
                              </div>
                            )}
                            {b.bezier.ok && (
                              <div className="small" style={{ marginTop: 3 }}>
                                P0 {b.bezier.p0?.join(",")} → apex {b.bezier.apex?.join(",")}
                                {" "}→ P2 {b.bezier.p2?.join(",")} · control {b.bezier.ctrl_px}px
                                {b.bezier.n_stalled_dropped > 0 && (
                                  <> · {b.bezier.n_stalled_dropped} stalled point(s) at the frame edge dropped</>
                                )}
                              </div>
                            )}
                            <Img url={b.bezier.image}
                              cap="Solid = measured flight. Dashed = projected continuation. The two are drawn differently on purpose: one is where the ball was seen, the other is where it must have gone." />
                          </div>
                        )}

                        {/* Why each candidate track was accepted or thrown
                            out -- the answer to "there was clearly a ball,
                            why no flight?" */}
                        {b.stages.flight?.tried?.length > 0 && (
                          <details style={{ marginTop: 6 }}>
                            <summary className="tiny muted" style={{ cursor: "pointer" }}>
                              Candidate tracks and their verdicts
                            </summary>
                            <div style={{ overflowX: "auto" }}>
                              <table className="tiny" style={{ borderCollapse: "collapse", marginTop: 6 }}>
                                <thead><tr style={{ textAlign: "left" }}>
                                  <th style={{ paddingRight: 10 }}>frames</th>
                                  <th style={{ paddingRight: 10 }}>from → to</th>
                                  <th style={{ paddingRight: 10 }}>rise</th>
                                  <th style={{ paddingRight: 10 }}>inliers</th>
                                  <th style={{ paddingRight: 10 }}>rms</th>
                                  <th>verdict</th>
                                </tr></thead>
                                <tbody>
                                  {b.stages.flight.tried.map((t, ti) => (
                                    <tr key={ti} style={{
                                      color: String(t.verdict || "").startsWith("accepted")
                                        ? "var(--ok, #2a8)" : undefined,
                                    }}>
                                      <td style={{ paddingRight: 10 }}>{t.frames?.join("–")}</td>
                                      <td style={{ paddingRight: 10 }}>
                                        {t.from?.join(",")} → {t.to?.join(",")}
                                      </td>
                                      <td style={{ paddingRight: 10 }}>{t.rise_px}px</td>
                                      <td style={{ paddingRight: 10 }}>{t.n_inliers}</td>
                                      <td style={{ paddingRight: 10 }}>{t.rms_px}</td>
                                      <td>{t.verdict}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </details>
                        )}
                      </div>
                    )}
                    {b.stages?.error && (
                      <div className="err-text tiny" style={{ marginTop: 8 }}>
                        Flight stages failed: {b.stages.error}
                      </div>
                    )}
                  </div>
                ))}
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/**
 * Debug3 report — the blob-and-track method. Where Debug2 reads the shape
 * the swing draws in a motion composite, this one never looks at a
 * composite: per-frame MOG2, drop the golfer, keep ball-sized blobs, link
 * them over time, fit a parabola. Read-only, so nothing to save.
 */
/** The tee box Debug3 searches in, drawn on a frame you can redraw it on.
 *
 * The ROI resolver has always written "draw one on the frame below" when
 * no box exists for the hole and day. That was true of the swing test and
 * has never been true here -- Debug3 printed the instruction with nothing
 * underneath it, so the one action it asked for was the one action it did
 * not offer.
 */
function D3TeeBox({ tb, uploadId, adminPassword, onRerun }) {
  // TWO pieces of state, not one. The first version used the box itself
  // as the "am I dragging" flag, so releasing the mouse never ended the
  // drag and a box could never be finished -- which is exactly what it
  // did: the hint stayed on "drag to set the box" and Save stayed
  // disabled however carefully you dragged. The anchor is its own thing.
  const [dragFrom, setDragFrom] = useState(null);
  const [draft, setDraft] = useState(null);   // {x,y,w,h} fractions
  const [angle, setAngle] = useState(tb?.roi?.angle || 0);
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState(null);
  const imgRef = useRef(null);
  if (!tb) return null;

  const box = draft || tb.roi || null;
  const canDraw = !!(tb.frame_url && tb.course_id && tb.hole && tb.day);

  function frac(e) {
    const r = imgRef.current?.getBoundingClientRect();
    if (!r || !r.width || !r.height) return null;
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
  }
  function onDown(e) {
    if (!canDraw) return;
    const p = frac(e);
    if (!p) return;
    e.preventDefault();
    setDragFrom(p);
    setDraft({ x: p.x, y: p.y, w: 0, h: 0 });
    setMsg(null);
  }
  function onMove(e) {
    if (!dragFrom) return;
    const p = frac(e);
    if (!p) return;
    setDraft({
      x: Math.min(dragFrom.x, p.x), y: Math.min(dragFrom.y, p.y),
      w: Math.abs(p.x - dragFrom.x), h: Math.abs(p.y - dragFrom.y),
    });
  }
  function onUp() {
    if (!dragFrom) return;
    setDragFrom(null);
    // A click rather than a drag: leave whatever was there alone.
    setDraft((d) => (d && (d.w < 0.01 || d.h < 0.01) ? null : d));
  }

  const save = async (roi) => {
    setBusy(true);
    setMsg(null);
    try {
      await api.setTeeBox(adminPassword, tb.course_id,
        { hole: tb.hole, day: tb.day, roi });
      setMsg(roi ? "Saved. Re-run to search inside it."
                 : "Cleared. Re-run to search the fallback box.");
      if (!roi) setDraft(null);
    } catch (err) {
      setMsg(`Could not save: ${err.message || err}`);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card" style={{ padding: 10, marginBottom: 10 }}>
      <div className="row" style={{ justifyContent: "space-between" }}>
        <b>Where stage 2 looks for the ball</b>
        <span className="tiny muted">{tb.source || "no box"}</span>
      </div>
      {tb.note && (
        <div className="tiny" style={{ color: "var(--warn, #b45309)", marginTop: 4 }}>
          {tb.note}
        </div>
      )}
      {canDraw ? (
        <>
          <div
            style={{ position: "relative", marginTop: 8, userSelect: "none" }}
            onMouseDown={onDown}
            onMouseMove={onMove}
            onMouseUp={onUp}
            onMouseLeave={onUp}
          >
            <img
              ref={imgRef}
              src={tb.frame_url}
              alt="tee box"
              draggable={false}
              style={{ width: "100%", borderRadius: 6, display: "block" }}
            />
            {box && (
              <div
                style={{
                  position: "absolute",
                  left: `${box.x * 100}%`, top: `${box.y * 100}%`,
                  width: `${box.w * 100}%`, height: `${box.h * 100}%`,
                  border: `2px solid ${draft ? "#22c55e" : "#3b82f6"}`,
                  background: "rgba(59,130,246,0.10)",
                  // A tee deck runs away from a camera set beside it, so
                  // the box that fits it is slanted. Rotated about its
                  // own centre, which is what the backend does too.
                  transform: `rotate(${angle}deg)`,
                  transformOrigin: "center center",
                  pointerEvents: "none",
                }}
              />
            )}
          </div>
          <div className="row" style={{ gap: 8, marginTop: 8, flexWrap: "wrap" }}>
            <span className="tiny muted">
              {draft
                ? `New box: x ${Math.round(draft.x * 100)}% y ${Math.round(draft.y * 100)}%, `
                  + `${Math.round(draft.w * 100)}%×${Math.round(draft.h * 100)}%`
                : "Drag on the frame to set the box for this hole today."}
            </span>
            <label className="tiny muted"
                   style={{ display: "flex", alignItems: "center", gap: 6 }}
                   title="Tilt the box to match the tee deck. An upright box over a slanted tee has to be tall enough to hold both ends, and every extra row of turf it covers is somewhere a false candidate can come from.">
              tilt
              <input
                type="range" min={-45} max={45} step={0.5} value={angle}
                onChange={(e) => setAngle(Number(e.target.value))}
                style={{ width: 110 }}
              />
              <input
                type="number" min={-45} max={45} step={0.5} value={angle}
                onChange={(e) => {
                  // Typed, so it can be an exact number rather than
                  // whatever a 130-pixel slider happened to land on.
                  const v = Number(e.target.value);
                  if (Number.isFinite(v)) {
                    setAngle(Math.max(-45, Math.min(45, v)));
                  }
                }}
                style={{ width: 62 }}
              />
              °
              {angle !== 0 && (
                <button className="btn tiny ghost"
                        onClick={() => setAngle(0)}>reset</button>
              )}
            </label>
            <button
              className="btn tiny"
              disabled={(!draft && angle === (tb?.roi?.angle || 0)) || busy}
              onClick={() => {
                const b = draft || tb.roi;
                save({ x: b.x, y: b.y, w: b.w, h: b.h, angle });
              }}
            >
              {busy ? "Saving…" : `Save box for hole ${tb.hole} on ${tb.day}`}
            </button>
            <button className="btn tiny ghost" disabled={busy}
                    onClick={() => save(null)}>
              Clear this day's box
            </button>
            {onRerun && (
              <button className="btn tiny" disabled={busy}
                      onClick={onRerun}>
                Re-run Debug3
              </button>
            )}
          </div>
          {msg && <div className="tiny" style={{ marginTop: 6 }}>{msg}</div>}
        </>
      ) : (
        <div className="tiny muted" style={{ marginTop: 6 }}>
          {tb.frame_url
            ? "No course/hole/day on this upload, so a box cannot be saved against it."
            : "This report predates the reference frame — run Debug3 again to draw a box."}
        </div>
      )}
    </div>
  );
}


/** Every resting-ball candidate in the tee box, with pictures.
 *
 * The one detector here that is not built on motion. A ball on a tee
 * does not move, so MOG2 calls it background and every motion panel in
 * this file fills with blobs on sleeves and hat brims while the ball
 * sits in plain sight. This asks what SAT still and looked like a ball.
 */
/** The hitting area on its own: view it, draw it, tilt it, save it.
 *
 * The drawer used to live inside the scan report, which meant the only
 * way to reach the control that decides where the scan looks was to
 * first sit through a two-minute scan looking in the wrong place.
 */
function HittingAreaModal({ state, onClose, adminPassword }) {
  const [tb, setTb] = useState(null);
  const [err, setErr] = useState(null);
  useEffect(() => {
    let live = true;
    api.getTeeBox(adminPassword, state.uploadId)
      .then((r) => live && setTb(r))
      .catch((e) => live && setErr(e.message));
    return () => { live = false; };
  }, [adminPassword, state.uploadId]);
  return (
    <div role="dialog" className="modal-back" onClick={onClose}
         style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
                  zIndex: 60, overflow: "auto", padding: 24 }}>
      <div className="card" onClick={(e) => e.stopPropagation()}
           style={{ maxWidth: 1100, margin: "0 auto", padding: 16 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>⬛ Hitting area — upload #{state.uploadId}</b>
          <button className="btn ghost" onClick={onClose}>Close ✕</button>
        </div>
        {err && <div className="err-text" style={{ marginTop: 10 }}>{err}</div>}
        {!tb && !err && (
          <div className="small" style={{ marginTop: 10 }}>Loading the frame…</div>
        )}
        {tb && (
          <>
            <div className="tiny muted" style={{ margin: "8px 0" }}>
              Everything that looks for a ball on the tee looks here and
              nowhere else. Drag to set it, tilt it to match the deck — a
              slanted tee inside an upright box means the box has to be tall
              enough to hold both ends, and every extra row of turf is
              somewhere a false candidate comes from.
            </div>
            <D3TeeBox tb={tb} adminPassword={adminPassword} />
          </>
        )}
      </div>
    </div>
  );
}


/**
 * WHERE THE RUN SPENT ITS TIME. Same shape Debug3 reports, so the two
 * can be compared directly rather than by eye across two layouts.
 *
 * The bar is the point. A column of seconds has to be read and ranked
 * before it says anything; the question being asked of this panel is
 * "which step is the problem", and a bar answers that at a glance.
 * `unattributed` is the other half of the answer: it is time no step
 * claimed, which is the only way this table can point at something
 * nobody thought to measure.
 */
function ScanTiming({ stages, timing, nested, nestedLabel, note }) {
  const list = stages || [];
  if (!list.length) return null;
  const total = timing?.total_sec || list.reduce(
    (a, st) => a + (st.seconds || 0), 0);
  const worst = list.reduce(
    (m, st) => ((st.seconds || 0) > (m?.seconds || 0) ? st : m), null);
  const bar = (sec) => (
    <div style={{ height: 4, borderRadius: 2, marginTop: 3,
                  background: "var(--line)" }}>
      <div style={{
        height: 4, borderRadius: 2, background: "var(--emerald-700)",
        width: `${total ? Math.min(100, (100 * (sec || 0)) / total) : 0}%`,
      }} />
    </div>
  );
  const rows = (arr, indent) => arr.map((st) => (
    <div key={`${indent}-${st.n}`} className="tiny"
         style={{ marginTop: 6, paddingLeft: 4 + indent,
                  borderLeft: "3px solid var(--line)" }}>
      <b>{st.n} · {st.title}</b>
      {st.count != null && (
        <span className="pill" style={{ marginLeft: 6 }}>
          {st.count} {st.counts}
        </span>
      )}
      <span className="muted">
        {" "}· {(st.seconds || 0).toFixed(2)}s
        {total ? ` · ${Math.round(100 * (st.seconds || 0) / total)}%` : ""}
      </span>
      {bar(st.seconds)}
      <div className="muted" style={{ marginTop: 2 }}>{st.detail}</div>
      {indent === 0 && st.n === 1 && nested?.length ? (
        <div style={{ marginTop: 4 }}>
          <div className="tiny muted"><i>{nestedLabel}</i></div>
          {rows(nested, 12)}
        </div>
      ) : null}
    </div>
  ));
  return (
    <div style={{ marginTop: 8 }}>
      <div className="row tiny" style={{ justifyContent: "space-between" }}>
        <b>Where the time went</b>
        <span className="muted">
          {total.toFixed(1)}s total
          {worst ? ` · slowest: ${worst.title} (${(worst.seconds || 0).toFixed(1)}s)` : ""}
          {timing?.per_candidate_sec != null
            ? ` · ${timing.per_candidate_sec}s per candidate` : ""}
        </span>
      </div>
      {note && <div className="tiny muted">{note}</div>}
      {rows(list, 0)}
      {timing?.build?.module_mtime && (
        <div className="tiny muted" style={{ marginTop: 4 }}>
          build {timing.build.module_mtime.replace("T", " ")}
          {timing.build.swing_detector
            ? ` · detector ${timing.build.swing_detector}` : ""}
          {" — two deployments giving different answers on one video is "}
          {"either a code difference or a data one; this line is the "}
          {"first half of telling them apart."}
        </div>
      )}
      {timing?.machine?.effective_cpus != null && (
        <div className="tiny muted" style={{ marginTop: 4 }}>
          ran on {timing.machine.effective_cpus} usable core(s)
          {timing.machine.host_cpus != null
            && timing.machine.host_cpus !== timing.machine.effective_cpus
            ? ` of ${timing.machine.host_cpus} the host reports`
            : ""}
          {timing.machine.cv2_threads != null
            ? ` · OpenCV on ${timing.machine.cv2_threads} thread(s)` : ""}
          {" — compare this line first when the same clip is slower in "}
          {"one place than another."}
        </div>
      )}
      {timing?.unattributed_sec > 0.05 && (
        <div className="tiny muted" style={{ marginTop: 6 }}>
          {timing.unattributed_sec}s unattributed — time no step above
          claimed. Small is bookkeeping; large means there is a step here
          nobody is measuring yet.
        </div>
      )}
    </div>
  );
}

/**
 * MARK THE FLAG, ON THE DAY IT WAS THERE.
 *
 * The calibration is a fact about two bolted-down cameras and is done
 * once, on the Cameras page. The flag is not: it is cut in a new place
 * each morning, so it belongs to a swing's DATE rather than to the hole
 * forever. Marking it here stamps it with this upload's capture time,
 * and every later swing takes it until somebody marks a later one.
 *
 * Which is why this shows the date it is about. "The flag is here" and
 * "the flag was here on Tuesday" are different claims, and only the
 * second one is true.
 */
function FlagstickModal({ row, adminPassword, onClose, onSaved }) {
  const [img, setImg] = useState(null);
  const [err, setErr] = useState(null);
  const [pin, setPin] = useState(null);       // {x, y} in green pixels
  const [note, setNote] = useState(null);
  const [saving, setSaving] = useState(false);
  const [dims, setDims] = useState(null);     // {w, h} of the green frame
  const wrapRef = useRef(null);
  const [drag, setDrag] = useState(false);

  useEffect(() => {
    let live = true;
    (async () => {
      try {
        const [f, vm] = await Promise.all([
          api.getLongUploadFrame(adminPassword, row.id, 0, "green"),
          api.getViewMap(adminPassword, row.id).catch(() => null),
        ]);
        if (!live) return;
        if (!f?.image_url) throw new Error("no green frame on this upload");
        setImg(f.image_url);
        setDims({ w: f.width || row.green_width || 1280,
                  h: f.height || row.green_height || 720 });
        const g = vm?.pin_green ?? vm?.view_map?.pin_green;
        if (g) setPin({ x: Math.round(g[0]), y: Math.round(g[1]) });
        setNote(vm?.pin_note || null);
        if (!vm?.view_map) {
          setErr("This hole has no green→tee mapping yet — calibrate the "
            + "camera pair on the Cameras page first, or the flag has "
            + "nowhere to be carried to.");
        }
      } catch (e) {
        if (live) setErr(e?.message || String(e));
      }
    })();
    return () => { live = false; };
  }, [adminPassword, row.id]);  // eslint-disable-line react-hooks/exhaustive-deps

  function at(ev) {
    const el = wrapRef.current;
    if (!el || !dims) return null;
    const r = el.getBoundingClientRect();
    const fx = (ev.clientX - r.left) / r.width;
    const fy = (ev.clientY - r.top) / r.height;
    return {
      x: Math.max(0, Math.min(dims.w - 1, Math.round(fx * dims.w))),
      y: Math.max(0, Math.min(dims.h - 1, Math.round(fy * dims.h))),
    };
  }

  async function save() {
    if (!pin) return;
    setSaving(true);
    try {
      const out = await api.saveHolePin(adminPassword, row.id,
                                        { green: [pin.x, pin.y] });
      onSaved?.(out);
      onClose();
    } catch (e) {
      setErr(e?.message || String(e));
      setSaving(false);
    }
  }

  const captured = row.base_captured_at
    ? fmtDateTime(row.base_captured_at) : "an unknown time";

  return (
    <div role="dialog" aria-modal="true" aria-label="Set the flagstick"
         onClick={onClose}
         style={{ position: "fixed", inset: 0, zIndex: 1200,
                  background: "rgba(0,0,0,0.75)", display: "flex",
                  alignItems: "center", justifyContent: "center",
                  padding: 16 }}>
      <div className="card" onClick={(e) => e.stopPropagation()}
           style={{ margin: 0, padding: 16, maxWidth: 980, width: "100%" }}>
        <div className="row" style={{ justifyContent: "space-between",
                                      gap: 12 }}>
          <b>⚑ Set flagstick — #{row.id}</b>
          <button className="btn ghost" style={{ width: "auto" }}
                  onClick={onClose}>Close ✕</button>
        </div>
        <div className="tiny muted" style={{ marginTop: 4 }}>
          Click or drag the flag to the BASE of the stick. This is where
          the flag was on <b>{captured}</b> — it carries to every later
          swing until it is marked again on a more recent one, and it does
          not move the swings before it.
          {note && <> Currently: {note}.</>}
        </div>
        {err && (
          <div className="err-text small" style={{ marginTop: 8 }}>{err}</div>
        )}
        {!img && !err && (
          <div className="small" style={{ marginTop: 10 }}>
            Fetching a frame from the green camera…
          </div>
        )}
        {img && (
          <div
            ref={wrapRef}
            onPointerDown={(e) => { setDrag(true); setPin(at(e)); }}
            onPointerMove={(e) => { if (drag) setPin(at(e)); }}
            onPointerUp={() => setDrag(false)}
            onPointerLeave={() => setDrag(false)}
            style={{ position: "relative", marginTop: 10, cursor: "crosshair",
                     borderRadius: 8, overflow: "hidden",
                     touchAction: "none", userSelect: "none" }}
          >
            <img src={img} alt="Green camera" draggable={false}
                 style={{ width: "100%", display: "block" }} />
            {pin && dims && (
              // Anchored at its BASE, because that is the point being
              // marked -- a pennant centred on the click would put the
              // stored coordinate somewhere up the pole.
              <div style={{
                position: "absolute",
                left: `${(pin.x / dims.w) * 100}%`,
                top: `${(pin.y / dims.h) * 100}%`,
                transform: "translate(-1px, -100%)",
                pointerEvents: "none",
              }}>
                <div style={{ width: 2, height: 34, background: "#fff",
                              boxShadow: "0 0 3px rgba(0,0,0,0.8)" }} />
                <div style={{
                  position: "absolute", left: 2, top: 0,
                  width: 0, height: 0,
                  borderLeft: "16px solid #ef4444",
                  borderTop: "6px solid transparent",
                  borderBottom: "6px solid transparent",
                }} />
              </div>
            )}
          </div>
        )}
        <div className="row" style={{ marginTop: 10, gap: 8,
                                      justifyContent: "flex-end" }}>
          <span className="tiny muted" style={{ marginRight: "auto" }}>
            {pin ? `${pin.x}, ${pin.y} in the green frame`
              : "nothing marked yet"}
          </span>
          <button className="btn ghost" style={{ width: "auto" }}
                  onClick={onClose}>Cancel</button>
          <button className="btn" style={{ width: "auto" }}
                  disabled={!pin || saving} onClick={save}>
            {saving ? "Saving…" : "Save flagstick"}
          </button>
        </div>
      </div>
    </div>
  );
}

function BallScanModal({ state, adminPassword, onClose }) {
  const rep = state?.report;
  const [why, setWhy] = useState(null);
  async function askWhy(fx, fy) {
    if (!state?.uploadId) return;
    setWhy({ loading: true });
    try {
      const r = await api.diagnoseBall(adminPassword, state.uploadId,
                                       { x: fx, y: fy });
      setWhy(r || { verdict: "no answer" });
    } catch (e) {
      setWhy({ error: e?.message || String(e) });
    }
  }
  // Scan-and-produce nests the scan under `scan`; a plain scan is flat.
  const spots = rep?.spots || rep?.scan?.spots || [];
  return (
    <div role="dialog" className="modal-back" onClick={onClose}
         style={{ position: "fixed", inset: 0, background: "rgba(0,0,0,0.55)",
                  zIndex: 60, overflow: "auto", padding: 24 }}>
      <div className="card" onClick={(e) => e.stopPropagation()}
           style={{ maxWidth: 1100, margin: "0 auto", padding: 16 }}>
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>⚪ Scan for ball — upload #{state.uploadId}</b>
          <button className="btn ghost" onClick={onClose}>Close ✕</button>
        </div>

        {state.running && (
          <div className="small" style={{ marginTop: 10 }}>
            {state.stage || "Scanning…"}
            {state.total > 0 && ` (${state.done}/${state.total})`}
          </div>
        )}
        {state.error && (
          <div className="err-text" style={{ marginTop: 10 }}>{state.error}</div>
        )}

        {/* A PLAIN SCAN GETS THE SAME TABLE. It is the step that
            dominates a produce, so the place to look for time in it is
            the run that does nothing else. Scan-and-produce nests this
            same list under its own stage 1 instead. */}
        {!rep?.clips && rep?.stages?.length > 0 && (
          <div className="card" style={{ margin: "10px 0", padding: 10 }}>
            <ScanTiming
              stages={rep.stages}
              timing={rep.timing}
              note={"Every candidate here cost the scan below; the video's "
                + "length sets stage 2 and the candidate count sets stage 4."}
            />
          </div>
        )}

        {rep?.clear_error && (
          <div className="err-text small" style={{ margin: "10px 0" }}>
            ⚠ {rep.clear_error}
          </div>
        )}
        {rep?.clips && (
          <div className="card" style={{ margin: "10px 0", padding: 10 }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>Traced from the ball, not the golfer</b>
              <span className={rep.n_traced ? "pill ok" : "pill warn"}>
                {rep.n_traced} of {rep.n_candidates} produced
              </span>
            </div>
            <div className="tiny muted" style={{ marginTop: 2 }}>
              {rep.reason} — a candidate qualifies by sitting{" "}
              {rep.min_held_sec}s or longer, which is what a ball waiting to
              be hit does and a speck does not.
            </div>
            <ScanTiming
              stages={rep.stages}
              timing={rep.timing}
              nested={rep.scan_stages}
              nestedLabel="inside stage 1, the scan's own steps:"
            />
            {rep.clips.map((c, i) => (
              <div key={i} className="tiny"
                   style={{ marginTop: 6, borderTop: "1px solid var(--line)",
                            paddingTop: 6 }}>
                <b>({c.spot[0]}, {c.spot[1]})</b>
                {c.impact_frame != null && (
                  <> · impact f{c.impact_frame} ({c.impact_sec}s)</>
                )}
                {c.held_sec != null && <> · sat {c.held_sec}s</>}
                {c.n_points != null && <> · {c.n_points} flight point(s)</>}
                {(rep.distances || []).filter((d) => d.swing === i).map((d) => (
                  <span key="d" className="pill ok" style={{ marginLeft: 6 }}
                        title={d.source === "descent"
                          ? `Measured from where the ball stopped FALLING — the pitch mark. It bounces and rolls from there, so this is where the shot arrived, not where it finished.`
                          : "Measured from the last point of the comet you plotted — where the ball came to rest."}>
                    📏 {d.text}
                    {d.source === "descent" ? " (pitch mark)" : ""}
                  </span>
                ))}
                {(rep.distance_notes || []).filter((n) => n.swing === i)
                  .map((n) => (
                    <div key="dn" className="tiny muted">
                      No distance: {n.reason}
                    </div>
                  ))}
                {c.clip_url ? (
                  <> · <a href={c.clip_url} target="_blank" rel="noreferrer">
                    clip</a></>
                ) : (
                  <span className="muted"> · {c.reason
                    || c.flight_reason || "no clip"}</span>
                )}

                {/* WHAT ELSE WAS ON OFFER. A produced clip is a claim
                    about where a ball went; the only way to disagree
                    with it is to see the paths it beat. */}
                {(c.tried || []).length > 0 && (
                  <details style={{ marginTop: 4 }}>
                    <summary className="muted">
                      {c.n_tracks} track(s) found, {c.tried.length} fitted —
                      every path considered and why it lost
                    </summary>
                    <div style={{ overflowX: "auto", marginTop: 4 }}>
                      <table style={{ borderCollapse: "collapse", fontSize: 11 }}>
                        <thead>
                          <tr className="muted" style={{ textAlign: "left" }}>
                            <th style={{ padding: "2px 6px" }}>#</th>
                            <th style={{ padding: "2px 6px" }}>frames</th>
                            <th style={{ padding: "2px 6px" }}>pts</th>
                            <th style={{ padding: "2px 6px" }}>inliers</th>
                            <th style={{ padding: "2px 6px" }}>rms</th>
                            <th style={{ padding: "2px 6px" }}>rise</th>
                            <th style={{ padding: "2px 6px" }}>aims</th>
                            <th style={{ padding: "2px 6px" }}>score</th>
                            <th style={{ padding: "2px 6px" }}>verdict</th>
                          </tr>
                        </thead>
                        <tbody>
                          {c.tried.map((t) => {
                            const won = /^accepted/.test(t.verdict || "");
                            return (
                              <tr key={t.idx}
                                  style={{
                                    borderTop: "1px solid var(--line)",
                                    background: won
                                      ? "rgba(0,160,80,0.12)" : "none",
                                  }}
                                  title={t.aim_basis || ""}>
                                <td style={{ padding: "2px 6px" }}>{t.idx}</td>
                                <td style={{ padding: "2px 6px" }}>
                                  f{t.frames?.[0]}–{t.frames?.[1]}
                                </td>
                                <td style={{ padding: "2px 6px" }}>{t.n_points}</td>
                                <td style={{ padding: "2px 6px" }}>{t.n_inliers}</td>
                                <td style={{ padding: "2px 6px" }}>{t.rms_px}</td>
                                <td style={{ padding: "2px 6px" }}>
                                  {Math.round(t.rise_px)}
                                </td>
                                <td style={{ padding: "2px 6px" }}>
                                  {t.aim_px != null ? `${Math.round(t.aim_px)}px` : "—"}
                                </td>
                                <td style={{ padding: "2px 6px" }}>
                                  {t.score != null ? <b>{t.score}</b> : "—"}
                                </td>
                                <td style={{ padding: "2px 6px" }}>
                                  {won ? <b>{t.verdict}</b> : t.verdict}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                    <div className="muted" style={{ marginTop: 4 }}>
                      <b>aims</b> is how far the path passes from the ball when
                      run down to its height — the test that separates a real
                      flight from a bird, a cart and a sleeve. Hover a row for
                      how that distance was measured.
                    </div>
                  </details>
                )}

                {(c.tracks_preview || []).length > 0 && (
                  <details style={{ marginTop: 4 }}>
                    <summary className="muted">
                      The {c.tracks_preview.length} track(s) drawn on the pictures
                    </summary>
                    <div style={{ overflowX: "auto", marginTop: 4 }}>
                      <table style={{ borderCollapse: "collapse" }}>
                        <thead>
                          <tr className="muted" style={{ textAlign: "left" }}>
                            <th style={{ padding: "2px 8px" }}>#</th>
                            <th style={{ padding: "2px 8px" }}>why</th>
                            <th style={{ padding: "2px 8px" }}>pts</th>
                            <th style={{ padding: "2px 8px" }}>frames</th>
                            <th style={{ padding: "2px 8px" }}>span</th>
                            <th style={{ padding: "2px 8px" }}>rise</th>
                            <th style={{ padding: "2px 8px" }}>density</th>
                            <th style={{ padding: "2px 8px" }}>from → to</th>
                          </tr>
                        </thead>
                        <tbody>
                          {c.tracks_preview.map((t) => (
                            <tr key={t.idx}
                                style={{ borderTop: "1px solid var(--line)" }}>
                              <td style={{ padding: "2px 8px" }}>
                                <span style={{
                                  display: "inline-block", width: 9, height: 9,
                                  borderRadius: 2, marginRight: 5,
                                  background: Array.isArray(t.color)
                                    ? `rgb(${t.color[2]},${t.color[1]},${t.color[0]})`
                                    : "#888",
                                }} />
                                {t.idx}
                              </td>
                              <td style={{ padding: "2px 8px" }}>{t.why}</td>
                              <td style={{ padding: "2px 8px" }}>{t.n}</td>
                              <td style={{ padding: "2px 8px" }}>
                                f{t.frames?.[0]}–{t.frames?.[1]}
                              </td>
                              <td style={{ padding: "2px 8px" }}>
                                {Math.round(t.span_px)}
                              </td>
                              <td style={{ padding: "2px 8px" }}>
                                {Math.round(t.rise_px)}
                              </td>
                              <td style={{ padding: "2px 8px" }}>
                                {t.density?.toFixed?.(2) ?? t.density}
                              </td>
                              <td style={{ padding: "2px 8px" }}>
                                ({t.from?.[0]},{t.from?.[1]}) → ({t.to?.[0]},{t.to?.[1]})
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                    {c.fit && (
                      <div className="muted" style={{ marginTop: 4 }}>
                        <b>Winner:</b> {c.fit.n_inliers} inlier(s), rms{" "}
                        {c.fit.rms_px}px, x degree {c.fit.x_degree}
                        {c.fit.aim_px != null && (
                          <> · run down to the ball&apos;s height it passes{" "}
                            <b>{c.fit.aim_px}px</b> from it
                            {c.fit.aim_basis ? ` (${c.fit.aim_basis})` : ""}</>
                        )}
                      </div>
                    )}
                    <div className="row" style={{ gap: 8, marginTop: 6,
                                                  flexWrap: "wrap" }}>
                      {[["the frame it worked from", c.frame_image_url],
                        ["every ball-sized blob kept", c.dets_image_url],
                        ["tracks, one colour each", c.tracks_image_url],
                        ["the chosen flight", c.flight_image_url],
                        ["the ball it started from", c.rest_image_url]].map(
                        ([lab, url]) => url ? (
                          <figure key={lab}
                                  style={{ margin: 0, flex: "1 1 260px" }}>
                            <a href={url} target="_blank" rel="noreferrer">
                              <img src={url} alt={lab}
                                   style={{ width: "100%", borderRadius: 6 }} />
                            </a>
                            <figcaption className="muted">{lab}</figcaption>
                          </figure>
                        ) : null,
                      )}
                    </div>
                  </details>
                )}

                {(c.flight_points || []).length > 0 && (
                  <details style={{ marginTop: 4 }}>
                    <summary className="muted">
                      {c.flight_points.length} flight point(s) — the path the
                      tracer drew
                    </summary>
                    <div className="muted"
                         style={{ marginTop: 4, fontFamily: "monospace",
                                  fontSize: 11, maxHeight: 160,
                                  overflow: "auto" }}>
                      {c.flight_points.map((q) => (
                        <div key={q.frame}>
                          f{q.frame}: {q.x}, {q.y}
                        </div>
                      ))}
                    </div>
                  </details>
                )}
              </div>
            ))}
          </div>
        )}

        {rep && (
          <>
            <div className="small" style={{ marginTop: 10 }}>
              <b>{spots.length}</b> resting-ball candidate(s) ·{" "}
              {rep.n_sampled} frame(s) sampled of {rep.n_frames} ·{" "}
              {rep.fps}fps
            </div>
            <div className="tiny muted" style={{ marginTop: 2 }}>
              Looking in: {(rep.roi_source ?? rep.scan?.roi_source)
                            || "the whole frame"}
              {(rep.roi || rep.scan?.roi) && (() => {
                const r = rep.roi || rep.scan.roi;
                return (
                  <> — x {Math.round(r.x * 100)}% y {Math.round(r.y * 100)}%,{" "}
                    {Math.round(r.w * 100)}%×{Math.round(r.h * 100)}%
                    {r.angle ? `, tilted ${r.angle}°` : ""}</>
                );
              })()}
            </div>
            {rep.roi_note && (
              <div className="tiny" style={{ color: "var(--warn,#b45309)", marginTop: 4 }}>
                {rep.roi_note}
              </div>
            )}
            {(rep.overview_url || rep.scan?.overview_url) && (
              <figure style={{ margin: "10px 0" }}>
                {/* CLICK THE BALL YOU CAN SEE. The server has always been
                    able to answer "why was the ball at this pixel not a
                    candidate" -- it walks the blob nearest the click
                    through the very same gates every other blob goes
                    through and names the one that turned it away. The
                    probe was only reachable from the dev-only swing
                    test, so the question kept getting answered by
                    squinting at screenshots instead, which is guessing
                    with extra steps. */}
                <div
                  onClick={(e) => {
                    const r = e.currentTarget.getBoundingClientRect();
                    askWhy((e.clientX - r.left) / r.width,
                           (e.clientY - r.top) / r.height);
                  }}
                  style={{ cursor: "crosshair", borderRadius: 6,
                           overflow: "hidden" }}
                >
                  <img src={rep.overview_url || rep.scan.overview_url}
                       alt="all candidates"
                       style={{ width: "100%", display: "block" }} />
                </div>
                <figcaption className="tiny muted">
                  Green = the tee box searched. Each candidate is ringed in
                  its own colour, matching its row in the table.
                  {" "}<b>Click a ball the scan did not ring</b> and it will
                  say which gate rejected it.
                </figcaption>
                {why && (
                  <div className="card tight"
                       style={{ margin: "8px 0 0", padding: 10 }}>
                    <div className="row"
                         style={{ justifyContent: "space-between" }}>
                      <b className="small">Why not a ball there?</b>
                      <button className="btn ghost small"
                              style={{ width: "auto" }}
                              onClick={() => setWhy(null)}>Close ✕</button>
                    </div>
                    {why.loading ? (
                      <div className="small">Asking the detector…</div>
                    ) : (
                      <>
                        <div className="small" style={{ marginTop: 4 }}>
                          {why.verdict || why.error || "no answer"}
                        </div>
                        {(why.expect_radius_px != null
                          || why.roi_source) && (
                          <div className="tiny muted" style={{ marginTop: 4 }}>
                            {why.expect_radius_px != null && (
                              <>expects a ball of r{why.expect_radius_px}px
                                {why.accept_radius_px
                                  ? ` (accepts ${why.accept_radius_px})`
                                  : ""}{" · "}</>
                            )}
                            {why.roi_source && <>box from {why.roi_source}</>}
                            {why.hole != null && <> · hole {why.hole}</>}
                          </div>
                        )}
                        {why.probe && (
                          <pre className="tiny muted" style={{
                            margin: "6px 0 0", whiteSpace: "pre-wrap",
                          }}>
                            {JSON.stringify(why.probe, null, 1)}
                          </pre>
                        )}
                      </>
                    )}
                  </div>
                )}
              </figure>
            )}
            {!spots.length && (
              <div className="tiny muted" style={{ marginTop: 8 }}>
                {rep.reason}
              </div>
            )}

            {/* WHAT THE DESCENT SEARCH SAW, all of it. The per-candidate
                pictures below show the chain that won; this shows the
                whole field, so a descent you can see that the scan did
                not take is either visibly here and marked REJECTED with
                its reason, or visibly absent — and those call for
                opposite responses. */}
            {rep.descents?.overview_url && (
              <figure style={{ margin: "10px 0 0" }}>
                <a href={rep.descents.overview_url} target="_blank"
                   rel="noreferrer">
                  <img src={rep.descents.overview_url} alt="descents found"
                       style={{ width: "100%", borderRadius: 6 }} />
                </a>
                <figcaption className="tiny muted">
                  Every descent found on the green camera —{" "}
                  {rep.descents.n_accepted} of {rep.descents.n_seen} strong
                  enough to confirm a swing. Accepted at{" "}
                  {rep.descents.gates?.min_points}+ points within{" "}
                  {rep.descents.gates?.max_bend_px}px of a straight line, or{" "}
                  {rep.descents.gates?.short_points} points within{" "}
                  {rep.descents.gates?.short_bend_px}px — a shorter chain has
                  to be straighter, because how many points get linked is
                  partly about the tree line behind the ball. A candidate
                  claims the one landing in its own flight window, and each
                  descent can only be claimed once.
                </figcaption>
              </figure>
            )}
            {rep.descents && !rep.descents.overview_url && (
              <div className="tiny muted" style={{ marginTop: 8 }}>
                Descent search: {rep.descents.reason}
              </div>
            )}

            {/* SUMMARY FIRST. The cards below are the evidence; this is
                the answer, and it is what you read to decide whether a
                clip is worth opening the pictures for. */}
            {spots.length > 0 && (
              <div style={{ overflowX: "auto", marginTop: 10 }}>
                <table className="tiny"
                       style={{ borderCollapse: "collapse", width: "100%" }}>
                  <thead>
                    <tr className="muted" style={{ textAlign: "left" }}>
                      <th style={{ padding: "3px 8px" }}>#</th>
                      <th style={{ padding: "3px 8px" }}>where</th>
                      <th style={{ padding: "3px 8px" }}>first seen</th>
                      <th style={{ padding: "3px 8px" }}>last seen</th>
                      <th style={{ padding: "3px 8px" }}>sat</th>
                      <th style={{ padding: "3px 8px" }}>gone</th>
                      <th style={{ padding: "3px 8px" }}>blocked</th>
                      <th style={{ padding: "3px 8px" }}>frames</th>
                      <th style={{ padding: "3px 8px" }}>ascent</th>
                      <th style={{ padding: "3px 8px" }}>descent</th>
                    </tr>
                  </thead>
                  <tbody>
                    {spots.map((sp, i) => (
                      <tr key={i}
                          style={{ borderTop: "1px solid var(--line)" }}>
                        <td style={{ padding: "3px 8px" }}>
                          <span style={{
                            display: "inline-block", width: 10, height: 10,
                            borderRadius: "50%", marginRight: 5,
                            background: sp.color || "#f59e0b",
                            verticalAlign: "middle",
                          }} />
                          {i + 1}
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          <b>{sp.x}, {sp.y}</b>
                          <span className="muted"> r{sp.radius}</span>
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          f{sp.first_frame}{" "}
                          <span className="muted">({sp.first_sec}s)</span>
                          {sp.merged_sightings > 1 && (
                            <div className="tiny muted"
                                 title="the sampler split this into separate sightings and the gap between them turned out to be short enough that they are one ball">
                              {sp.merged_sightings} sightings joined
                            </div>
                          )}
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          f{sp.last_frame}{" "}
                          <span className="muted">({sp.last_sec}s)</span>
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          {sp.held_sec}s
                          {/* HOW FAR THE SIGHTINGS SCATTERED. A teed ball
                              does not move: every sighting lands on the
                              same pixel bar the camera's own shake. A
                              shoe under a waiting player shuffles. Shown
                              rather than acted on, so the threshold can
                              be read off real balls and real shoes side
                              by side instead of guessed. */}
                          {sp.wobble_px != null && (
                            <div className="tiny muted"
                                 title="How far this spot's sightings scattered around their own average. A ball at rest should be a fraction of a pixel; anything that shuffles is not sitting still.">
                              ±{sp.wobble_px}px
                            </div>
                          )}
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          {sp.gone_frame != null ? (
                            <b>f{sp.gone_frame}{" "}
                              <span className="muted">({sp.gone_sec}s)</span>
                            </b>
                          ) : sp.still_blocked ? (
                            <span className="pill warn"
                                  title="the watch ran out with something still over the spot — it was never seen to be gone">
                              still covered
                            </span>
                          ) : (
                            <span className="muted">—</span>
                          )}
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          {sp.blocked_frames
                            ? `${sp.blocked_frames}f`
                            : <span className="muted">—</span>}
                        </td>
                        <td style={{ padding: "3px 8px" }}>{sp.votes}</td>
                        <td style={{ padding: "3px 8px" }}>
                          {/* DID A BALL LEAVE THIS SPOT. The tee camera's
                              own verdict, and the one a shoe cannot pass
                              even in principle: the shoe IS the thing
                              sitting there, so nothing rises away from
                              it. */}
                          {sp.ascent ? (
                            <span className="pill ok" title={sp.ascent_reason}>
                              ↑ {sp.ascent.n_points}pts · {sp.ascent.rise_px}px
                            </span>
                          ) : (
                            <span className="tiny muted"
                                  title={sp.ascent_reason || "not searched"}>
                              none
                            </span>
                          )}
                          {sp.gate && (
                            <div className="tiny muted">via {sp.gate}</div>
                          )}
                        </td>
                        <td style={{ padding: "3px 8px" }}>
                          {sp.descent ? (
                            <span className="pill ok"
                                  title={`The green camera saw a ball come down ${sp.descent.flight_sec}s after this spot emptied — ${sp.descent.n_points} points falling, bending only ${sp.descent.bend_px}px. That is this swing, seen from the other end.`}>
                              ✓ {sp.descent.n_points} pts ·{" "}
                              {sp.descent.flight_sec}s
                              {sp.descent.refined_from
                                ? ` (deep +${sp.descent.n_points
                                    - sp.descent.refined_from})` : ""}
                            </span>
                          ) : (sp.descent_near || []).length ? (
                            <span className="pill warn"
                                  title={sp.descent_reason}>
                              {sp.descent_near.length} near-miss
                            </span>
                          ) : (
                            <span className="muted"
                                  title={sp.descent_reason
                                    || "no green camera on this upload"}>
                              —
                            </span>
                          )}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
                <div className="tiny muted" style={{ marginTop: 6 }}>
                  <b>descent</b> is the green camera's verdict: a chain of at
                  least four points falling nearly straight down, landing in
                  the window a shot struck here would land in. A candidate
                  with one is a swing — it gets produced however briefly it
                  sat, and its descent becomes the clip's landing and its
                  green comet.
                  <br />
                  <b>last seen</b> is the last frame a ball was visible there.
                  <b> gone</b> is the first frame after that where the spot was
                  both clear of anything covering it and empty — a clubhead at
                  address hides a ball that is still sitting there, so the two
                  differ by however long the player stood over it.
                </div>
              </div>
            )}
            {spots.map((sp, i) => (
              <div key={i} className="card"
                   style={{ margin: "10px 0", padding: 10 }}>
                <div className="row" style={{ justifyContent: "space-between" }}>
                  <b>
                    <span style={{
                      display: "inline-block", width: 11, height: 11,
                      borderRadius: "50%", marginRight: 6,
                      background: sp.color || "#f59e0b",
                      verticalAlign: "middle",
                    }} />
                    {i + 1}. ({sp.x}, {sp.y}) · r {sp.radius}px
                  </b>
                  <span className="pill">{sp.votes} frame(s)</span>
                </div>
                <div className="tiny muted" style={{ marginTop: 2 }}>
                  First seen f{sp.first_frame} ({sp.first_sec}s) · last seen
                  f{sp.last_frame} ({sp.last_sec}s) · sat {sp.held_sec}s
                  {sp.gone_frame != null && (
                    <> · <b>gone by f{sp.gone_frame} ({sp.gone_sec}s)</b>
                      {sp.blocked_frames
                        ? ` after ${sp.blocked_frames} frame(s) covered`
                        : ""}</>
                  )}
                  {sp.gone_frame == null && sp.still_blocked && (
                    <> · still covered when the watch ran out</>
                  )}
                </div>
                {/* THE GREEN CAMERA'S VERDICT, AS A PICTURE. "Four
                    points falling nearly straight down" is a sentence;
                    whether THIS chain is a ball or the tree line is only
                    answerable by looking at it. */}
                {sp.descent ? (
                  <div style={{ marginTop: 8 }}>
                    <div className="tiny" style={{ color: "#3ee37a" }}>
                      ✓ Confirmed by the green camera — a ball came down{" "}
                      {sp.descent.flight_sec}s after this spot emptied,
                      landing at ({sp.descent.landing_xy[0]},{" "}
                      {sp.descent.landing_xy[1]}) on f
                      {sp.descent.landing_frame}. {sp.descent.n_points} points,
                      falling {sp.descent.drop_px}px and bending only{" "}
                      {sp.descent.bend_px}px off a straight line.
                      {sp.descent.refined_from ? (
                        <> The tracker linked only{" "}
                          {sp.descent.refined_from} of them; a deeper
                          re-scan of this window found the rest.</>
                      ) : null}
                    </div>
                    {sp.ascent_image && (
                      <figure style={{ margin: "6px 0 0" }}>
                        <a href={sp.ascent_image} target="_blank"
                           rel="noreferrer">
                          <img src={sp.ascent_image} alt="ascent"
                               style={{ width: "100%", borderRadius: 6 }} />
                        </a>
                        <figcaption className="tiny muted">
                          What rose after this spot emptied. Green = the
                          chain taken as the ascent, grey = every other
                          chain considered with the reason it lost. The
                          small ring is where the ball was sitting; the
                          wide one is how close a chain has to start to
                          count as leaving it.
                          {sp.ascent_reason && <> — {sp.ascent_reason}</>}
                        </figcaption>
                      </figure>
                    )}
                    {sp.descent_image_url && (
                      <figure style={{ margin: "6px 0 0" }}>
                        <a href={sp.descent_image_url} target="_blank"
                           rel="noreferrer">
                          <img src={sp.descent_image_url} alt="descent"
                               style={{ width: "100%", borderRadius: 6 }} />
                        </a>
                        <figcaption className="tiny muted">
                          The chain the detector linked, on the frame the
                          ball landed in. White ring = the landing; this
                          becomes the clip's green comet.
                        </figcaption>
                      </figure>
                    )}
                  </div>
                ) : sp.descent_reason ? (
                  <div className="tiny" style={{ marginTop: 8,
                        color: (sp.descent_near || []).length
                          ? "#f59e0b" : "var(--muted)" }}>
                    No descent: {sp.descent_reason}
                    {sp.descent_near_image_url && (
                      <figure style={{ margin: "6px 0 0" }}>
                        <a href={sp.descent_near_image_url} target="_blank"
                           rel="noreferrer">
                          <img src={sp.descent_near_image_url}
                               alt="descent turned away"
                               style={{ width: "100%", borderRadius: 6 }} />
                        </a>
                        <figcaption className="tiny muted">
                          What fell in this candidate&apos;s window and was
                          turned away. If that is a ball coming down, the
                          gates are wrong — the numbers on it say by how
                          much.
                        </figcaption>
                      </figure>
                    )}
                  </div>
                ) : null}
                <div className="row" style={{ gap: 10, marginTop: 8,
                                              flexWrap: "wrap" }}>
                  {[["first", sp.first_image, sp.first_frame],
                    ["last", sp.last_image, sp.last_frame]].map(
                    ([lab, url, fr]) => url ? (
                      <figure key={lab} style={{ margin: 0, flex: "1 1 300px" }}>
                        <a href={url} target="_blank" rel="noreferrer">
                          <img src={url} alt={lab}
                               style={{ width: "100%", borderRadius: 6 }} />
                        </a>
                        <figcaption className="tiny muted">
                          {lab === "first" ? "first frame it appears"
                                           : "last frame before it goes"} — f{fr}
                        </figcaption>
                      </figure>
                    ) : null,
                  )}
                </div>
              </div>
            ))}
          </>
        )}
      </div>
    </div>
  );
}


function Debug3Modal({ state, onClose }) {
  if (!state) return null;
  const rep = state.report;
  const Img = ({ url, cap }) =>
    url ? (
      <figure style={{ margin: "8px 0" }}>
        <a href={url} target="_blank" rel="noreferrer">
          <img src={url} alt={cap} style={{ width: "100%", borderRadius: 6 }} />
        </a>
        <figcaption className="tiny muted" style={{ marginTop: 4 }}>{cap}</figcaption>
      </figure>
    ) : null;
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)",
        display: "flex", alignItems: "flex-start", justifyContent: "center",
        zIndex: 1000, padding: 16, overflow: "auto",
      }}
    >
      <div
        className="card"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 1200, width: "100%", margin: 0 }}
      >
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>🧿 Debug3 — upload #{state.uploadId}</b>
          <button type="button" className="ghost small"
            style={{ width: "auto" }} onClick={onClose}>
            Close ✕
          </button>
        </div>

        {state.running && (
          <div style={{ marginTop: 10 }}>
            <div className="small">
              <span
                className="shimmer"
                style={{
                  display: "inline-block", width: 12, height: 12,
                  borderRadius: "50%", marginRight: 8,
                  verticalAlign: "middle",
                }}
              />
              Running in the background{state.stage ? ` — ${state.stage}` : ""}
              {state.total ? ` (${state.done}/${state.total})` : ""}. MOG2 runs
              on every frame of every candidate&apos;s flight window, so this
              takes a few minutes. You can leave this open.
            </div>
            <ol className="tiny muted" style={{ marginTop: 8, paddingLeft: 18 }}>
              <li>Pose candidates — wrist speed + spine bend</li>
              <li>Ball at impact — club-arc vertex on the ground line</li>
              <li>MOG2 → connected components → golfer masked → ball-sized blobs kept</li>
              <li>Link detections across frames (constant velocity + gate)</li>
              <li>RANSAC parabola, then: must rise, must point back at the ball</li>
            </ol>
          </div>
        )}
        {state.error && (
          <div className="err-text small" style={{ marginTop: 8 }}>
            {state.error}
          </div>
        )}

        {rep?.stages?.length > 0 && (
          <div style={{ marginTop: 12 }}>
            {/* First thing in the report, above the stage list: it is the
                question the stages cannot answer — how many swings are in
                this clip — and it is what someone opens Debug3 to check. */}
            <D3TeeBox
              tb={rep.tee_box}
              uploadId={state.uploadId ?? state.id}
              adminPassword={state.adminPassword}
              onRerun={state.onRerun}
            />
            {rep.swing_detect ? (
              <SwingDetectPanel sd={rep.swing_detect} />
            ) : (
              /* A report from before this existed has no swing_detect and
                 would just render nothing — which is indistinguishable
                 from the panel being broken. Say which it is. */
              <div className="tiny muted" style={{ marginBottom: 8 }}>
                <b>Swing detection comparison:</b> not in this report — it
                was produced before the ball-departure detector was added,
                or by a plain Re-Produce, which skips it. Run{" "}
                <b>Debug3</b> again to get it.
              </div>
            )}
            {rep.stages.map((st) => (
              <div key={st.n} className="small"
                style={{
                  display: "flex", gap: 10, padding: "6px 0",
                  borderBottom: "1px solid var(--line)",
                }}
              >
                <b style={{ minWidth: 18 }}>{st.n}</b>
                <div style={{ flex: 1 }}>
                  <b>{st.name}</b>
                  <div className="tiny muted">{st.detail}</div>
                </div>
                <div style={{ textAlign: "right", minWidth: 120 }}>
                  <b>{st.count}</b>
                  <div className="tiny muted">{st.counts}</div>
                </div>
                {/* Wall clock for the stage, with a bar so the expensive
                    one is obvious without reading the numbers. */}
                {st.seconds != null && (
                  <div style={{ textAlign: "right", minWidth: 86 }}>
                    <b>{st.seconds}s</b>
                    <div className="tiny muted">{st.pct}%</div>
                    <div
                      style={{
                        height: 3, borderRadius: 2, marginTop: 2,
                        background: "var(--line)",
                      }}
                    >
                      <div
                        style={{
                          height: "100%", borderRadius: 2,
                          width: `${Math.min(100, st.pct || 0)}%`,
                          background:
                            (st.pct || 0) >= 40
                              ? "var(--danger, #c0392b)"
                              : "var(--emerald-700, #16a34a)",
                        }}
                      />
                    </div>
                  </div>
                )}
              </div>
            ))}
            {rep.timing && (
              <div className="tiny muted" style={{ marginTop: 6 }}>
                <b>{rep.timing.total_sec}s total</b>
                {rep.timing.n_swings > 0 && (
                  <> · {rep.timing.per_swing_sec}s per swing to analyse
                    ({rep.timing.n_swings} swing
                    {rep.timing.n_swings === 1 ? "" : "s"})</>
                )}
                {Object.keys(rep.timing.outside_stages || {}).length > 0 && (
                  <>
                    {" "}· outside the 7 stages:{" "}
                    {Object.entries(rep.timing.outside_stages)
                      .map(([k, v]) => `${k} ${v}s`)
                      .join(", ")}
                  </>
                )}
                {rep.timing.unattributed_sec > 0.5 && (
                  <> · {rep.timing.unattributed_sec}s unaccounted</>
                )}
                {/* Stage 7 itemised — it is normally the bulk of the run,
                    and one opaque number is the wrong shape for the wait
                    an operator actually sits through. */}
                {Object.keys(rep.timing.produce_breakdown || {}).length > 0 && (
                  <div style={{ marginTop: 3 }}>
                    <b>stage 7 breakdown:</b>{" "}
                    {Object.entries(rep.timing.produce_breakdown)
                      .map(([k, v]) => `${k} ${v}s`)
                      .join(" · ")}
                  </div>
                )}
              </div>
            )}
            <div className="tiny muted" style={{ marginTop: 6 }}>
              {rep.frame?.[0]}×{rep.frame?.[1]} @ {rep.fps}fps · ball scale
              r = {rep.r_px}px
            </div>
            {/* Tee→green sync: say whether the offset was MEASURED from
                the cameras' clocks or ASSUMED, so a visibly wrong cut
                points at the offset rather than at the pipeline. */}
            {(rep.produced?.clips || []).some((c) => c.clip_id) && (
              <div className="tiny muted" style={{ marginTop: 4 }}>
                {rep.produced.clips
                  .filter((c) => c.clip_id)
                  .map((c, i) => (
                    <div key={`sync-${i}`}>
                      swing {i + 1}: tee {c.tee_window_sec?.[0]}–
                      {c.tee_window_sec?.[1]}s ({c.tee_video_dur_sec}s shown)
                      {c.green ? (
                        <>
                          {" "}· cut to green at Δ{c.green_delta_sec}s{" "}
                          <b style={{
                            color: c.green_delta_source === "camera_event"
                              ? "var(--emerald-700, #16a34a)"
                              : "var(--danger, #c0392b)",
                          }}>
                            {c.green_delta_source === "camera_event"
                              ? "measured from camera clocks"
                              : c.green_delta_source === "edit_metrics"
                                ? "saved offset"
                                : "ASSUMED 0 — no camera clocks"}
                          </b>
                        </>
                      ) : (
                        <> · <b>tee-only</b> (no green cut)</>
                      )}
                      {c.plot_background === false && (
                        <> · <b>no click-to-plot background</b></>
                      )}
                    </div>
                  ))}
              </div>
            )}
            {rep.produced && (
              <div
                className="small"
                style={{
                  marginTop: 8, padding: "8px 10px", borderRadius: 6,
                  background: rep.produced.ok
                    ? "var(--primary-soft)" : "rgba(192,57,43,0.08)",
                  border: `1px solid ${rep.produced.ok
                    ? "var(--emerald-200)" : "rgba(192,57,43,0.35)"}`,
                }}
              >
                <b style={{
                  color: rep.produced.ok
                    ? "var(--emerald-800)" : "#c0392b",
                }}>
                  {rep.produced.ok
                    ? "Re-produced — the card's clip and click-to-plot are updated"
                    : "Re-produce failed"}
                </b>
                <div className="muted">
                  {rep.produced.detail || rep.produced.error}
                </div>
              </div>
            )}
            {rep.pose_debug && (
              <div className="tiny" style={{ marginTop: 6 }}>
                <b>Stage 1 working (pose):</b>{" "}
                <span className="muted">
                  {rep.pose_debug.n_pose_frames ?? "?"} frame(s) with a
                  person of {rep.pose_debug.n_samples ?? "?"} sampled
                  {rep.pose_debug.coverage != null
                    && ` (${Math.round(rep.pose_debug.coverage * 100)}% coverage)`}
                  {" · "}wrist-speed median {rep.pose_debug.median?.toFixed?.(4)},
                  threshold {rep.pose_debug.threshold?.toFixed?.(4)}
                  {" · "}{rep.pose_debug.n_raw_bursts ?? 0} raw burst(s),
                  {" "}{rep.pose_debug.n_bend_rejected ?? 0} rejected as upright
                  {" · "}spine-bend gate{" "}
                  {rep.pose_debug.back_bend_min_deg}–
                  {rep.pose_debug.back_bend_max_deg}°
                  {rep.pose_debug.n_crop_frames != null
                    && ` · ${rep.pose_debug.n_crop_frames} frame(s) via the person crop`}
                  {rep.pose_debug.n_bootstrap_scans
                    ? ` · tiled bootstrap swept ${rep.pose_debug.n_bootstrap_scans}x`
                      + (rep.pose_debug.bootstrap_found_at != null
                         ? `, found the golfer at ${rep.pose_debug.bootstrap_found_at}s`
                         : ", never found a whole person")
                    : ""}
                </span>
                {rep.pose_debug.reason && (
                  <div style={{ color: "#c0392b" }}>
                    {rep.pose_debug.reason}
                  </div>
                )}
                {(rep.bursts || []).length > 0 ? (
                  <div style={{ marginTop: 4 }}>
                    <span className="muted">every burst it saw:</span>{" "}
                    {rep.bursts.map((b, i) => (
                      <span key={i} style={{
                        display: "inline-block", marginRight: 8,
                        color: b.status === "swing" ? "#1a9d55" : "#b7791f",
                      }}>
                        {b.t}s ×{b.ratio}
                        {b.bend != null && ` ${b.bend}°`} [{b.status}]
                      </span>
                    ))}
                  </div>
                ) : (
                  <div style={{ color: "#b7791f", marginTop: 4 }}>
                    No wrist-speed burst even reached the gates — nothing in
                    this clip moved the hands fast enough relative to the
                    clip's own median.
                  </div>
                )}
                {(rep.pose_series || []).length > 4 && (
                  <PoseTrace
                    series={rep.pose_series}
                    threshold={rep.pose_debug.threshold}
                    bursts={rep.bursts || []}
                    durationSec={rep.pose_debug.duration_sec}
                  />
                )}
              </div>
            )}
            <div className="tiny" style={{ marginTop: 4 }}>
              <b>Ball side:</b>{" "}
              {rep.ball_side ? (
                <span className="pill ok">{rep.ball_side} of the golfer</span>
              ) : (
                <span className="pill warn">
                  not set — searching BOTH sides
                </span>
              )}
              <span className="muted" style={{ marginLeft: 6 }}>
                {rep.ball_side_reason}
              </span>
              {!rep.ball_side && (
                <div className="muted">
                  A two-sided search puts the golfer's own shoes in the same
                  vote as the ball. Set the ball side on the tee camera
                  (/admin/cameras → Edit) and the wrong half of the search
                  disappears entirely.
                </div>
              )}
            </div>
            {rep.rest_ball && !rep.swing_detect && (
              <div className="tiny muted" style={{ marginTop: 4 }}>
                <b>Resting ball:</b> {rep.rest_ball.reason}
                {(rep.rest_ball.departures || []).map((d, k) => (
                  <span key={k} style={{ marginLeft: 6 }}>
                    [{d.t}s ({d.x},{d.y}) rest {d.rest_sec}s]
                  </span>
                ))}
              </div>
            )}
          </div>
        )}

        {(rep?.swings || []).map((sw) => (
          <div key={sw.idx} className="card"
            style={{ marginTop: 14, background: "var(--surface-2)" }}>
            <div className="row" style={{ justifyContent: "space-between" }}>
              <b>Candidate {sw.idx + 1} · {sw.peak_time_sec}s</b>
              <span className={`pill ${
                sw.dropped_by_judge ? "warn" : sw.flight?.length ? "ok" : "warn"
              }`}>
                {sw.dropped_by_judge
                  ? "dropped — not a swing (high confidence)"
                  : sw.flight?.length
                    ? `${sw.flight.length} tracer points`
                    : "no flight"}
              </span>
            </div>
            <div className="tiny muted" style={{ marginTop: 2 }}>
              impact frame {sw.impact_frame} · window f{sw.window?.[0]}–
              f{sw.window?.[1]}
              {/* Stage 1 labels now rather than eliminating. Say which
                  candidates only got here because its gates were overruled
                  — those are the ones worth eyeballing. */}
              {sw.pose_gate_ok === false && (
                <span
                  title={
                    "Stage 1's gates rejected this burst (" +
                    (sw.pose_gate || "rejected") +
                    "). It was carried through anyway so stages 2 and 3 " +
                    "could judge it on the ball and the club fan."
                  }
                  style={{
                    marginLeft: 8, padding: "1px 6px", borderRadius: 999,
                    background: "rgba(214,158,46,0.18)", color: "#8a6116",
                    fontWeight: 600,
                  }}
                >
                  rescued · stage 1 said {sw.pose_gate || "no"}
                </span>
              )}
            </div>


            <div className="small" style={{ marginTop: 10 }}>
              <b>Stage 2 — ball at rest:</b>{" "}
              {sw.ball_hint
                ? `(${sw.ball_hint[0]}, ${sw.ball_hint[1]})`
                : "not found"}
              <div className="tiny muted">{sw.ball_hint_reason}</div>
              <div className="tiny muted">
                {sw.ball_hint
                  ? "The judge centres on it and the aim gate is armed."
                  : "The judge falls back to the wrist and the aim gate is DISARMED for this candidate."}
              </div>
            </div>
            <Img url={sw.ball_hint_image_url}
              cap="Stage 2 — where it looked for the ball at rest, at the pose impact frame. Red = the search window (ground band, ball side only), yellow = the ground line at the feet, orange = the motion it found there. Drawn whether or not a ball was found." />

            {sw.judge && (
              <div className="small" style={{ marginTop: 10 }}>
                <b>Stage 3 — AI judge (club fan):</b>{" "}
                {sw.judge.ai_judge === true
                  ? "swing"
                  : sw.judge.ai_judge === false
                    ? "NOT a swing"
                    : sw.judge.verdict === "club_swing"
                      ? "swing (club-fan heuristic)"
                      : sw.judge.verdict === "no_swing"
                        ? "no swing (club-fan heuristic — advisory only)"
                        : "no verdict"}
                <span className="pill" style={{ marginLeft: 6 }}>
                  decided by {sw.judge.decided_by}
                </span>
                {sw.judge.ai_confidence != null && (
                  <span className={`pill ${sw.judge.confident ? "" : "warn"}`}
                    style={{ marginLeft: 6 }}>
                    confidence {sw.judge.ai_confidence}
                  </span>
                )}
                <div className="tiny muted">
                  {sw.judge.ai_reason || sw.judge.reason}
                </div>
                {sw.judge.fan != null && (
                  <div className="tiny muted">
                    fan {sw.judge.fan}° over {sw.judge.n_rays} rays /{" "}
                    {sw.judge.n_angles} angles
                  </div>
                )}
                {sw.judge_unsure && (
                  <div className="tiny" style={{ color: "#b7791f" }}>
                    Kept anyway — the judge said "not a swing" but only at{" "}
                    {sw.judge.ai_confidence} confidence. Only a HIGH-confidence
                    rejection drops a candidate; an ambiguous picture is not
                    grounds for throwing away a shot.
                  </div>
                )}
                {sw.judge.decided_by === "heuristic" && (
                  <div className="tiny muted">
                    Only a high-confidence AI verdict drops a candidate — the
                    heuristic is recorded but never vetoes.
                  </div>
                )}
              </div>
            )}
            <Img url={sw.heat_image_url}
              cap="Stage 3 — the motion-heat composite the judge was shown. The club's sweep through impact is the fan; a practice swing, a bag drop or someone bending to tee up do not draw one." />
            {sw.dropped_by_judge && (
              <div className="tiny muted" style={{ marginTop: 6 }}>
                Tracking, the flight fit and produce were all skipped for
                this candidate.
              </div>
            )}

            <div className="small" style={{ marginTop: 10 }}>
              <b>Ball at impact:</b>{" "}
              {sw.ball ? `(${sw.ball[0]}, ${sw.ball[1]})` : "not found"}
              {sw.ball_source && (
                <span className="pill" style={{ marginLeft: 6 }}>
                  {sw.ball_source}
                </span>
              )}
              <div className="tiny muted">{sw.ball_reason}</div>
              {sw.ball_alt && (
                <div className="tiny muted" style={{ marginTop: 2 }}>
                  {sw.ball_alt_source}: ({sw.ball_alt[0]}, {sw.ball_alt[1]})
                  {sw.ball_disagree_px != null && (
                    <> — the two disagree by <b>{sw.ball_disagree_px}px</b></>
                  )}
                  <div>{sw.ball_alt_reason}</div>
                </div>
              )}
            </div>
            <Img url={sw.ball_image_url}
              cap="Stage 2 (refined) — club arc re-measured at the flight's launch frame; green is the vertex" />

            <div className="small" style={{ marginTop: 10 }}>
              <b>Detections:</b> {sw.detect_reason}
              {sw.area_summary && (
                <div className="tiny muted">
                  blob areas: n={sw.area_summary.n} · median=
                  {sw.area_summary.median}px · p90={sw.area_summary.p90}px ·
                  max={sw.area_summary.max}px — cap in use {sw.max_area}px,
                  max side {sw.max_side}px
                  {sw.n_at_strict_cap != null && (
                    <> · a flat 30px cap would have kept {sw.n_at_strict_cap}</>
                  )}
                </div>
              )}
            </div>
            <Img url={sw.frame_image_url}
              cap="Stage 3 — one frame classified: red is the golfer mask (excluded), green are ball-sized blobs kept" />
            <Img url={sw.dets_image_url}
              cap="Stage 3 — every kept detection over the window, blue early to red late" />

            <Img url={sw.tracks_image_url}
              cap="Stage 4 — the track candidates: which detections the tracker decided belong to the same object. Numbers and colours match the table below; hollow ring = first frame of a track, filled dot = last." />

            <div className="small" style={{ marginTop: 10 }}>
              <b>Tracks:</b> {sw.n_tracks} built
              {(sw.tracks_preview || []).length > 0 && (
                <>
                  {sw.n_tracks > sw.tracks_preview.length && (
                    <span className="tiny muted" style={{ marginLeft: 6 }}>
                      — {sw.tracks_preview.length} drawn: the longest, plus
                      the ones that rise most. A branch in the wind outlasts
                      a struck ball, so length alone hides the shot.
                    </span>
                  )}
                  {sw.winner_not_shown && (
                    <div className="tiny" style={{ color: "#b7791f" }}>
                      The fit chose a track that is not in this list — see
                      "every track that was tested" below.
                    </div>
                  )}
                  <table className="tiny" style={{ width: "100%", marginTop: 4 }}>
                    <thead>
                      <tr>
                        <th align="left">#</th>
                        <th align="left">points</th>
                        <th align="left">frames</th>
                        <th align="right">span</th>
                        <th align="right">rise</th>
                        <th align="right" title="points per frame spanned — 1.0 means it was seen on every frame. Junk built from long-range links sits near 0.3.">seen</th>
                        <th align="left">from → to</th>
                        <th align="right" title="inliers / rms / how far its back-projection lands from the ball">fit</th>
                        <th align="left">shown because</th>
                        <th align="left">verdict</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sw.tracks_preview.map((t, k) => (
                        <tr key={k}
                          style={t.winner
                            ? { background: "var(--surface-3, rgba(0,0,0,.05))" }
                            : undefined}>
                          <td style={{ whiteSpace: "nowrap" }}>
                            <span style={{
                              display: "inline-block", width: 10, height: 10,
                              borderRadius: 2, marginRight: 6,
                              background: t.color || "#888",
                              verticalAlign: "middle",
                            }} />
                            {t.idx ?? k + 1}
                            {t.winner && (
                              <b title="the fit chose this track"> ← flight</b>
                            )}
                          </td>
                          <td>{t.n}</td>
                          <td>f{t.frames?.[0]}–{t.frames?.[1]}</td>
                          <td align="right">{t.span_px}px</td>
                          <td align="right">{t.rise_px}px</td>
                          <td align="right"
                            style={t.density != null && t.density < 0.5
                              ? { color: "#b7791f" } : undefined}>
                            {t.density ?? "–"}
                          </td>
                          <td className="muted" style={{ whiteSpace: "nowrap" }}>
                            {t.from ? `${t.from[0]},${t.from[1]}` : "–"} →{" "}
                            {t.to ? `${t.to[0]},${t.to[1]}` : "–"}
                          </td>
                          <td align="right" className="muted"
                            style={{ whiteSpace: "nowrap" }}>
                            {t.n_inliers != null
                              ? `${t.n_inliers}/${t.n} · ${t.rms_px ?? "–"}px`
                              : "–"}
                            {t.aim_px != null && (
                              <> · aims {Math.round(t.aim_px)}px
                                {t.aim_path_px != null && <> (path)</>}
                              </>
                            )}
                            {t.aim_at_impact_px != null
                              && t.aim_path_px != null
                              && Math.abs(t.aim_at_impact_px - t.aim_path_px) > 50 && (
                              <div style={{ color: "#b7791f" }}>
                                at the pose impact frame it would read{" "}
                                {Math.round(t.aim_at_impact_px)}px — pose has
                                the impact frame wrong
                              </div>
                            )}
                          </td>
                          <td className="muted">{t.why}</td>
                          <td style={String(t.verdict || "").startsWith("accepted")
                            ? { color: "var(--emerald-700)" }
                            : { color: "#b7791f" }}>
                            {t.verdict || "–"}
                            {t.score != null && <> · score {t.score}</>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </div>

            <div className="small" style={{ marginTop: 10 }}>
              <b>Flight:</b> {sw.flight_reason}
              {sw.find_flight_ok === false && (
                <div className="tiny" style={{ color: "#c0392b", marginTop: 4 }}>
                  <b>
                    {sw.find_flight_failed
                      ? "The flight stage threw an error"
                      : "A track was accepted but the stage produced nothing usable"}
                    :
                  </b>{" "}
                  {sw.find_flight_reason || "no reason recorded"}
                  <div>
                    Nothing was produced for this swing as a result.
                  </div>
                  {sw.find_flight_traceback && (
                    <details style={{ marginTop: 4 }}>
                      <summary>where it threw</summary>
                      <pre style={{
                        whiteSpace: "pre-wrap", overflowX: "auto",
                        fontSize: 11, margin: 0,
                      }}>{sw.find_flight_traceback}</pre>
                    </details>
                  )}
                </div>
              )}
              {sw.images_error && (
                <div className="tiny" style={{ color: "#b7791f", marginTop: 4 }}>
                  Panel images failed ({sw.images_error}) — the flight itself
                  is unaffected.
                  {sw.images_traceback && (
                    <details style={{ marginTop: 4 }}>
                      <summary>where it threw</summary>
                      <pre style={{
                        whiteSpace: "pre-wrap", overflowX: "auto",
                        fontSize: 11, margin: 0,
                      }}>{sw.images_traceback}</pre>
                    </details>
                  )}
                </div>
              )}
              {sw.fit && (
                <div className="tiny muted">
                  {sw.fit.n_inliers} inliers · rms {sw.fit.rms_px}px · aims{" "}
                  {sw.fit.aim_px}px from the ball · says impact was at (
                  {sw.fit.at_impact?.[0]}, {sw.fit.at_impact?.[1]})
                </div>
              )}
              {(sw.tried || []).length > 0 && (
                <details style={{ marginTop: 6 }}>
                  <summary className="tiny muted">
                    every track that was tested ({sw.tried.length})
                  </summary>
                  <table className="tiny" style={{ width: "100%", marginTop: 4 }}>
                    <thead>
                      <tr>
                        <th align="left">pts</th>
                        <th align="right">inliers</th>
                        <th align="right">span</th>
                        <th align="right">rise</th>
                        <th align="right" title="how far the flight passes from the ball, measured by running the path DOWN to the ball's height — not by evaluating it at pose's impact frame, which is routinely several frames out">aims</th>
                        {/* Shape: a ball rises, peaks once, falls. ↑↓ is
                            the apex count, ↓↑ the physically impossible
                            reversal. mono 1.0 = a clean profile. */}
                        <th align="right" title="rise→fall (apex) / fall→rise (impossible)">↑↓ / ↓↑</th>
                        <th align="right" title="1.0 is a clean rise-peak-fall profile">mono</th>
                        <th align="right" title="net displacement ÷ path length; ~0 wanders in place">direct</th>
                        <th align="right">score</th>
                        {/* What the old count-driven formula would have
                            picked — shown so a disagreement is visible
                            rather than silent. */}
                        <th align="right" title="the old score: 2×inliers + capped span − rms/10">was</th>
                        <th align="left">verdict</th>
                        <th align="left" title="the same number drawn on the tracks picture">#</th>
                        <th align="left">frames</th>
                        <th align="left" title="first and last detection — use this to find the track you can see on screen">from → to</th>
                        <th align="right" title="points per frame spanned">seen</th>
                      </tr>
                    </thead>
                    <tbody>
                      {sw.tried.map((t, k) => {
                        const accepted = String(t.verdict || "").startsWith("accepted");
                        // Flag rows the two formulas rank differently.
                        const best = sw.tried.reduce((a, b) =>
                          (b.score ?? -1e9) > (a.score ?? -1e9) ? b : a, sw.tried[0]);
                        const bestLegacy = sw.tried.reduce((a, b) =>
                          (b.score_legacy ?? -1e9) > (a.score_legacy ?? -1e9) ? b : a, sw.tried[0]);
                        const flipped = best !== bestLegacy && t === bestLegacy;
                        return (
                        <tr key={k}
                          style={{
                            color: accepted ? "var(--emerald-700)" : undefined,
                            background: flipped
                              ? "rgba(192,57,43,0.07)" : undefined,
                          }}
                          title={flipped
                            ? "the old score would have picked this one"
                            : undefined}
                        >
                          <td>{t.n_points}</td>
                          <td align="right">{t.n_inliers}</td>
                          <td align="right">{t.span_px}</td>
                          <td align="right">{t.rise_px}</td>
                          <td align="right" className="muted"
                            style={{ whiteSpace: "nowrap" }}
                            title={t.aim_basis || ""}>
                            {t.aim_px != null ? `${Math.round(t.aim_px)}px` : "–"}
                            {t.aim_frame != null && ` @f${t.aim_frame}`}
                          </td>
                          <td align="right">
                            {t.n_rise_to_fall ?? "–"} / {t.n_fall_to_rise ?? "–"}
                          </td>
                          <td align="right">{t.monotonicity ?? "–"}</td>
                          <td align="right">{t.directness ?? "–"}</td>
                          <td align="right"><b>{t.score ?? "–"}</b></td>
                          <td align="right" className="muted">{t.score_legacy ?? "–"}</td>
                          <td>{t.verdict}</td>
                          <td>{t.idx ?? "–"}</td>
                          <td style={{ whiteSpace: "nowrap" }}>
                            {t.frames ? `f${t.frames[0]}–${t.frames[1]}` : "–"}
                          </td>
                          <td className="muted" style={{ whiteSpace: "nowrap" }}>
                            {t.from ? `${t.from[0]},${t.from[1]}` : "–"} →{" "}
                            {t.to ? `${t.to[0]},${t.to[1]}` : "–"}
                          </td>
                          <td align="right"
                            style={t.density != null && t.density < 0.5
                              ? { color: "#b7791f" } : undefined}>
                            {t.density ?? "–"}
                          </td>
                        </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </details>
              )}
            </div>
            {sw.launch && (
              <div className="small" style={{ marginTop: 10 }}>
                <b>Launch, from the flight itself:</b>{" "}
                {sw.launch.ok
                  ? `(${sw.launch.xy[0]}, ${sw.launch.xy[1]}) at f${sw.launch.frame}`
                  : "not available"}
                <div className="tiny muted">{sw.launch.reason}</div>
                {sw.launch_vs_pose_frames != null && (
                  <div className="tiny muted">
                    that is {sw.launch_vs_pose_frames > 0 ? "+" : ""}
                    {sw.launch_vs_pose_frames} frames from the pose peak
                  </div>
                )}
              </div>
            )}
            <Img url={sw.rest_check_image_url}
              cap={`Ball at rest — frame ${sw.rest_check_frame} (5 before launch), our estimate ringed, with a 6x inset. This is the frame where the ball should still be sitting there.`} />
            {sw.club_arc_relocated && (
              <div className="tiny muted" style={{ marginTop: 6 }}>
                <b>Club arc at the launch frame</b> — this is the ball we
                use (f{sw.club_arc_relocated.frame}):{" "}
                {sw.club_arc_relocated.xy
                  ? `(${sw.club_arc_relocated.xy[0]}, ${sw.club_arc_relocated.xy[1]}) — ${sw.club_arc_relocated.vs_launch_px}px from the extrapolated launch`
                  : "not found"}
                <div>{sw.club_arc_relocated.reason}</div>
              </div>
            )}
            <Img url={sw.club_arc_relocated_image_url}
              cap="Club arc re-measured over the real downswing, not the pose peak's window" />
            {sw.refine && (
              <div className="tiny muted" style={{ marginTop: 6 }}>
                <b>Blob check (confirmation only):</b>{" "}
                {sw.refine.ok
                  ? `found at (${sw.refine.xy[0]}, ${sw.refine.xy[1]}), ` +
                    `${sw.refine.agrees_px}px from the extrapolated launch`
                  : "nothing conclusive"}
                <div>{sw.refine.reason}</div>
              </div>
            )}
            <Img url={sw.refine_image_url}
              cap="Ball search box at 6x — cyan is where the flight said to look, green is the stationary blob found there" />
            <Img url={sw.flight_image_url}
              cap="Stage 5 — green inliers, red × outliers, cyan the fitted parabola, magenta where the curve says impact was, grey the rejected tracks. Detections behind them ramp blue (early) to orange (late)." />

            {sw.produce && (
              <div style={{ marginTop: 12 }}>
                <div className="small">
                  <b>Stage 6 — preview clip</b>{" "}
                  <span className={`pill ${sw.produce.ok ? "ok" : "err"}`}>
                    {sw.produce.ok ? "rendered" : "failed"}
                  </span>
                  <div className="tiny muted">
                    {sw.produce.ok ? (
                      <>
                        {sw.produce.tracer_points} tracer point(s) from the
                        RANSAC inliers, ball ({sw.produce.ball?.[0]},{" "}
                        {sw.produce.ball?.[1]}), impact f
                        {sw.produce.impact_frame}
                        {sw.produce.frame_range && (
                          <> · frames {sw.produce.frame_range[0]}–
                            {sw.produce.frame_range[1]}</>
                        )}
                      </>
                    ) : (
                      sw.produce.error
                    )}
                  </div>
                </div>
                {sw.produce.clip_url && (
                  <video
                    src={sw.produce.clip_url}
                    controls
                    playsInline
                    style={{
                      width: "100%", borderRadius: 6, marginTop: 8,
                      background: "#000",
                    }}
                  />
                )}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * Debug2 report — the five stages, each showing its own work. Read-only:
 * the run writes evidence images and changes no swing data, so this
 * modal has nothing to save.
 */
function Debug2Modal({ state, onClose }) {
  if (!state) return null;
  const rep = state.report;
  return (
    <div
      role="dialog"
      aria-modal="true"
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.88)",
        display: "flex", alignItems: "flex-start", justifyContent: "center",
        zIndex: 1000, padding: 16, overflow: "auto",
      }}
    >
      <div
        className="card"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: 1200, width: "100%", margin: 0 }}
      >
        <div className="row" style={{ justifyContent: "space-between" }}>
          <b>🔬 Debug2 — upload #{state.uploadId}</b>
          <button type="button" className="ghost small"
            style={{ width: "auto" }} onClick={onClose}>
            Close ✕
          </button>
        </div>

        {state.running && (
          <div style={{ marginTop: 10 }}>
            <div className="small">
              <span
                className="shimmer"
                style={{
                  display: "inline-block", width: 12, height: 12,
                  borderRadius: "50%", marginRight: 8,
                  verticalAlign: "middle",
                }}
              />
              Running — this is one synchronous pass, so the stages below
              fill in only when it finishes. Minutes, not seconds.
            </div>
            {/* The stage list up front, so the wait is legible: you can
                see what it is doing even though the result arrives in one
                piece. */}
            <ol className="tiny muted" style={{ marginTop: 8, paddingLeft: 18 }}>
              <li>Pose candidates — wrist speed + spine bend</li>
              <li>Impact frame, and the ball from the bottom of the club arc</li>
              <li>AI judge on the motion-heat composite</li>
              <li>MOG2 heat over impact−5 … impact+40</li>
              <li>Chain walked upward from the ball</li>
            </ol>
          </div>
        )}
        {state.error && (
          <div className="err-text small" style={{ marginTop: 8 }}>
            {state.error}
          </div>
        )}

        {rep?.stages?.length > 0 && (
          <div style={{ marginTop: 10 }}>
            {rep.stages.map((st) => (
              <div key={st.n} className="small" style={{ marginBottom: 2 }}>
                <b>{st.n}. {st.name}</b>{" "}
                <span className="muted">— {st.detail}</span>{" "}
                <b style={{ color: st.count ? "var(--emerald-700)" : "#b7791f" }}>
                  {st.count}
                </b>
                {st.counts && <span className="muted"> {st.counts}</span>}
              </div>
            ))}
          </div>
        )}

        {rep?.bursts?.length > 0 && (
          <div className="tiny" style={{ marginTop: 8 }}>
            <span className="muted">
              every burst the pose detector saw (stage 1 working):
            </span>{" "}
            {rep.bursts.map((b, i) => (
              <span
                key={i}
                style={{
                  display: "inline-block", marginRight: 6,
                  color: b.status === "swing" ? "#1a9d55" : "#b7791f",
                }}
              >
                {b.t}s ×{b.ratio}
                {b.bend != null && ` ${b.bend}°`} [{b.status}]
              </span>
            ))}
          </div>
        )}

        {(rep?.swings || []).map((sw) => {
          const dropped = sw.verdict === "not_swing";
          return (
            <div
              key={sw.idx}
              style={{
                border: `1px solid ${dropped ? "rgba(192,57,43,0.5)" : "rgba(26,157,85,0.5)"}`,
                borderRadius: 8, padding: "8px 12px", marginTop: 12,
              }}
            >
              <div className="small" style={{ fontWeight: 700 }}>
                candidate {sw.idx + 1} @ {sw.peak_time_sec}s · impact f
                {sw.impact_frame}
                {sw.back_bend_deg != null && ` · bend ${sw.back_bend_deg}°`}
                {sw.ratio != null && ` · speed ×${sw.ratio}`}
                {" — "}
                <span style={{ color: dropped ? "#c0392b" : "#1a9d55" }}>
                  {dropped ? "❌ not a swing" : "✅ swing"}
                </span>
              </div>
              <div className="tiny muted">{sw.verdict_reason}</div>

              {sw.ai_path != null && (
                <div className="tiny" style={{ marginTop: 4 }}>
                  <b>AI traced the ball trail:</b>{" "}
                  {sw.ai_path.length > 0 ? (
                    <span style={{ color: "#1a9d55" }}>
                      {sw.ai_path.length} point(s)
                      {sw.ai_path_confidence && ` · ${sw.ai_path_confidence}`}
                      {sw.ai_path_source && ` · read off the ${sw.ai_path_source}`}
                      {sw.ai_path_start_px != null &&
                        ` · starts ${sw.ai_path_start_px}px from the club-arc ball`}
                    </span>
                  ) : (
                    <span style={{ color: "#b7791f" }}>no trail found</span>
                  )}
                  {sw.ai_path_note && (
                    <div className="muted">{sw.ai_path_note}</div>
                  )}
                </div>
              )}
              <div className="tiny" style={{ marginTop: 6 }}>
                <b>ball at impact:</b>{" "}
                {sw.ball ? `${sw.ball[0]}, ${sw.ball[1]}` : "not found"}
                {sw.ball_source ? ` (${sw.ball_source})` : ""}
                {" — "}{sw.ball_reason}
                {sw.ball_alt && (
                  <div className="muted">
                    {sw.ball_alt_source}: {sw.ball_alt[0]}, {sw.ball_alt[1]}
                    {sw.ball_disagree_px != null
                      ? ` — the two disagree by ${sw.ball_disagree_px}px`
                      : ""}
                  </div>
                )}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 4 }}>
                {[
                  ["club arc → ball", sw.ball_image_url],
                  ["heat composite (what the judge saw)", sw.heat_image_url],
                  [
                    sw.window
                      ? `windowed heat f${sw.window[0]}–f${sw.window[1]}`
                      : "windowed heat",
                    sw.heat_window_image_url,
                  ],
                  ["chain from the ball", sw.chain_image_url],
                ].map(([cap, url]) =>
                  url ? (
                    <a key={cap} href={url} target="_blank" rel="noreferrer"
                      style={{ width: 280, display: "block" }}>
                      <div className="tiny muted">{cap}</div>
                      <img src={url} alt={cap}
                        style={{ width: "100%", borderRadius: 6 }} />
                    </a>
                  ) : null,
                )}
              </div>
              {!dropped && (
                <div className="tiny" style={{ marginTop: 4 }}>
                  <b>chain:</b> {sw.chain_reason || "—"}
                  {sw.n_dots != null && ` · ${sw.n_dots} dots in window`}
                  {sw.n_band_new != null && (
                    <div className="muted">
                      band re-scan above the club fan: {sw.n_band_scan} dot(s)
                      found, <b>{sw.n_band_new} the tracer pool did not
                      have</b> — the pool's gates are tuned for the golfer's
                      body, not for open sky
                    </div>
                  )}
                  {sw.chain_method && (
                    <div className="muted">
                      method: {sw.chain_method}
                      <div style={{ opacity: 0.75 }}>
                        four are tried in order, first one that passes wins:
                        1 dots on the trail the AI traced · 2 straight runs
                        in the left/middle/right bands above the club fan ·
                        3 lock on above head height then walk back down ·
                        4 walk up from the ball. Every one must rise and
                        point back at the ball.
                      </div>
                    </div>
                  )}
                  {sw.chain_tries?.length > 1 && (
                    <div className="muted">
                      tried: {sw.chain_tries.join("  |  ")}
                    </div>
                  )}
                  {sw.bands?.length > 0 && (
                    <div className="muted">
                      bands above the club fan:{" "}
                      {sw.bands
                        .map(
                          (t) =>
                            `${t.zone} ${t.n_dots} dots` +
                            (t.chain?.length
                              ? ` → ${t.chain.length} in line, aims ${Math.round(
                                  t.aim_px ?? 0,
                                )}px from the ball`
                              : " → nothing"),
                        )
                        .join("  ·  ")}
                    </div>
                  )}
                  {sw.aim_px != null && (
                    <div
                      style={{
                        color: sw.aim_px <= 60 ? "#1a9d55" : "#b7791f",
                      }}
                    >
                      run back to impact, the chain lands{" "}
                      <b>{sw.aim_px}px</b> from the ball
                      {sw.aim_px <= 60
                        ? " — chain and ball agree"
                        : " — chain and ball DISAGREE; one of them is wrong"}
                    </div>
                  )}
                  {sw.rejected_why?.length > 0 && (
                    <div className="muted" style={{ marginTop: 2 }}>
                      rejects:{" "}
                      {sw.rejected_why
                        .map((r) => `${r.n}× ${r.why}`)
                        .join(" · ")}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function ImageLightbox({ url, title, onClose }) {
  // Full-screen zoom/pan viewer for a single image (the per-frame
  // detector-view JPGs). Wheel or +/− to zoom, drag to pan when zoomed,
  // double-click toggles 1x ⇄ 3x. Esc/backdrop/Close to dismiss.
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const drag = useRef(null);
  useEffect(() => {
    setZoom(1);
    setPan({ x: 0, y: 0 });
  }, [url]);
  useEffect(() => {
    if (!url) return undefined;
    const onKey = (e) => {
      if (e.key === "Escape") onClose?.();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [url, onClose]);
  if (!url) return null;
  const clampZoom = (z) => Math.max(1, Math.min(12, z));
  const zoomBy = (mult) =>
    setZoom((z) => {
      const nz = clampZoom(z * mult);
      if (nz === 1) setPan({ x: 0, y: 0 });
      return nz;
    });
  return (
    <div
      role="dialog"
      aria-modal="true"
      aria-label={title || "Image viewer"}
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.92)",
        zIndex: 1100, display: "flex", flexDirection: "column", padding: 14,
      }}
    >
      <div
        className="row"
        onClick={(e) => e.stopPropagation()}
        style={{
          color: "#fff", alignItems: "center",
          justifyContent: "space-between", marginBottom: 8, cursor: "default",
        }}
      >
        <b style={{ fontSize: "0.9rem" }}>{title || "Image"}</b>
        <div className="row" style={{ gap: 6, alignItems: "center" }}>
          <button type="button" className="ghost small" style={{ width: "auto" }}
            onClick={() => zoomBy(1 / 1.4)} title="Zoom out">−</button>
          <span className="small" style={{ minWidth: 44, textAlign: "center" }}>
            {zoom.toFixed(1)}×
          </span>
          <button type="button" className="ghost small" style={{ width: "auto" }}
            onClick={() => zoomBy(1.4)} title="Zoom in">+</button>
          <button type="button" className="ghost small" style={{ width: "auto" }}
            onClick={() => { setZoom(1); setPan({ x: 0, y: 0 }); }} title="Reset zoom">
            Reset
          </button>
          <button type="button" className="ghost small" style={{ width: "auto" }}
            onClick={onClose}>Close ✕</button>
        </div>
      </div>
      <div
        onClick={(e) => e.stopPropagation()}
        onWheel={(e) => zoomBy(e.deltaY < 0 ? 1.2 : 1 / 1.2)}
        onMouseDown={(e) => {
          e.preventDefault();
          drag.current = { sx: e.clientX, sy: e.clientY, px: pan.x, py: pan.y };
        }}
        onMouseMove={(e) => {
          if (!drag.current) return;
          setPan({
            x: drag.current.px + (e.clientX - drag.current.sx),
            y: drag.current.py + (e.clientY - drag.current.sy),
          });
        }}
        onMouseUp={() => { drag.current = null; }}
        onMouseLeave={() => { drag.current = null; }}
        onDoubleClick={() => {
          setPan({ x: 0, y: 0 });
          setZoom((z) => (z > 1 ? 1 : 3));
        }}
        style={{
          flex: 1, minHeight: 0, overflow: "hidden",
          display: "flex", alignItems: "center", justifyContent: "center",
          cursor: zoom > 1 ? "grab" : "zoom-in",
        }}
      >
        <img
          src={url}
          alt={title || "detector view"}
          draggable={false}
          style={{
            maxWidth: "100%", maxHeight: "100%",
            transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})`,
            transformOrigin: "center center",
            imageRendering: zoom >= 4 ? "pixelated" : "auto",
            userSelect: "none",
          }}
        />
      </div>
    </div>
  );
}

/**
 * Zoomable heat canvas with clickable timed dots — the interactive
 * frames-on-heat view. Self-contained zoom/pan; `marks` is a
 * {frame: {x,y}} map of queued picks, `onToggleDot(p)` queues/unqueues.
 * `denseDots` (optional) is the raw per-frame candidate pool — hidden
 * at low zoom, revealed as smaller clickable dots once zoomed past 2.5×
 * so a sparse timed chain can be filled in from the full detection set.
 * Used by the wizard's Step-2 plot view AND the standalone card modal.
 */
const DENSE_DOT_ZOOM = 2.5;

function PlotHeatCanvas({
  bgUrl, dots, denseDots, frameW, frameH, marks, onToggleDot, onClose,
  scanRegion, track, ballXY, placingBall, onPlaceBall, comet,
  note, noteColour, shape, handles, onHandleDrag, onHandleDrop,
  // STEP THE REAL VIDEO UNDER THE DOTS. The map's background is a
  // motion-heat composite: every frame of the window smeared into one
  // picture, which is what makes a flight legible as a streak and makes
  // a single instant impossible to see. `loadFrame(n)` fetches the
  // actual frame n, so the operator can walk the strike frame by frame
  // and see the ball itself rather than the trail it left.
  loadFrame, frameLo, frameHi, startFrame, onViewFrame,
  // The hole's flagstick, in this picture's pixels. Drawn, not
  // clickable: it is moved from the field panel, so that a click on the
  // map is never ambiguous between "the ball went here" and "the flag
  // is here".
  flag,
}) {
  const [zoom, setZoom] = useState(1);
  const [focus, setFocus] = useState({ x: 50, y: 50 });
  // Extra detections pulled by 🔍 Scan (frame-diff over the zoomed
  // region) — merged into the dense layer for this session.
  const [scanDots, setScanDots] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [scanNote, setScanNote] = useState(null);
  // How hard the next scan looks. Starts at the normal level and steps
  // up when a scan comes back with nothing new -- the operator can see
  // the ball, so "no motion found" means the gates were too tight, not
  // that there is nothing there. Capped at 3, which returns leaves in
  // the wind as well as the ball; that is the right trade here, since
  // the operator is the filter.
  const [scanLevel, setScanLevel] = useState(2);
  const hasDims = !!(frameW && frameH);

  // ── the video frame under the dots ────────────────────────────────
  // null means the heat composite, which stays the default: it is the
  // view that shows a whole flight at once.
  const canStep = typeof loadFrame === "function"
    && frameLo != null && frameHi != null && frameHi >= frameLo;
  // ALWAYS THE REAL FRAME. The motion-heat composite is every frame of
  // the window smeared into one picture -- useful for seeing a whole
  // flight at once, and useless for placing anything, because nothing
  // in it is a moment. Placing is what this map is for now, so it opens
  // on the video and the composite is not offered.
  const [viewFrame, setViewFrame] = useState(null);
  const [frameUrl, setFrameUrl] = useState(null);
  const [frameErr, setFrameErr] = useState(null);
  // Frames already fetched, so stepping back over ground already walked
  // is instant instead of a round trip per press. The server caches the
  // JPG on disk too; this saves the request as well as the seek.
  const frameCache = useRef(new Map());
  useEffect(() => {
    if (!canStep || viewFrame == null) { setFrameUrl(null); return undefined; }
    let live = true;
    const want = viewFrame;
    const cached = frameCache.current.get(want);
    if (cached) setFrameUrl(cached);
    (async () => {
      try {
        const url = cached || await loadFrame(want);
        if (!live) return;
        frameCache.current.set(want, url);
        setFrameUrl(url);
        setFrameErr(null);
        // Warm the next one so a run of ▶ presses does not stutter.
        // Failures here are silent on purpose: it is a guess about what
        // the operator will ask for next, and a wrong guess is not worth
        // an error message.
        const nxt = want + 1;
        if (nxt <= frameHi && !frameCache.current.has(nxt)) {
          loadFrame(nxt)
            .then((u) => frameCache.current.set(nxt, u))
            .catch(() => {});
        }
      } catch (e) {
        if (live) setFrameErr(e?.message || String(e));
      }
    })();
    return () => { live = false; };
  }, [canStep, viewFrame, loadFrame, frameHi]);
  // WHICH FRAME IS ON SCREEN, upward. A landing marked on the green
  // camera is a place AND an instant, and the instant is whichever
  // frame the operator was looking at when they clicked it.
  useEffect(() => { onViewFrame?.(viewFrame); }, [viewFrame, onViewFrame]);
  // Open on a frame rather than on the composite.
  useEffect(() => {
    if (!canStep) return;
    setViewFrame((f) => (f == null
      ? Math.max(frameLo, Math.min(frameHi, startFrame ?? frameLo))
      : f));
  }, [canStep, frameLo, frameHi, startFrame]);
  const stepFrame = (d) => setViewFrame((f) => {
    if (f == null) return null;
    return Math.max(frameLo, Math.min(frameHi, f + d));
  });

  // MOG2 dots, hideable. They are the point of this screen and they are
  // also what covers the picture: on a frame where the ball is a
  // three-pixel smudge, the dot marking it is bigger than it is. Being
  // able to take them away for a moment is the difference between
  // guessing which dot is the ball and seeing it.
  const [showDots, setShowDots] = useState(true);
  // HOW BIG THE PICTURE GETS, IN PIXELS, MEASURED.
  //
  // The dots are placed as a percentage of this box while the picture
  // is drawn object-fit: cover, so the instant the box stops being the
  // frame's shape the picture is cropped and every dot points at the
  // wrong grass. The old sizing -- flex:1 with an aspect-ratio and a
  // max-width -- let flex fix the height and the max-width clip the
  // width independently, which on a 2560x1440 screen made a 16:9 frame
  // into a 1.61 box: a tenth of the picture cropped away, and every dot
  // off with it. On a tall window it was 0.46.
  //
  // So the fit is computed rather than asked for: the largest box of
  // the frame's shape that fits the space available. Exact in every
  // window shape, and within 1% of the space in all of them.
  const areaRef = useRef(null);
  const [fit, setFit] = useState(null);
  useEffect(() => {
    const el = areaRef.current;
    if (!el || !hasDims || typeof ResizeObserver === "undefined") {
      return undefined;
    }
    const measure = () => {
      const r = el.getBoundingClientRect();
      if (r.width < 2 || r.height < 2) return;
      const s = Math.min(r.width / frameW, r.height / frameH);
      setFit({ w: Math.floor(frameW * s), h: Math.floor(frameH * s) });
    };
    measure();
    const ro = new ResizeObserver(measure);
    ro.observe(el);
    return () => ro.disconnect();
  }, [frameW, frameH, hasDims]);
  // Dense candidates + scanned dots that aren't already represented by
  // a timed dot (same frame within 2px) — no doubled-up targets.
  const extraDots = [...(denseDots || []), ...scanDots].filter(
    (c) =>
      !(dots || []).some(
        (p) =>
          p.frame === c.frame &&
          Math.abs(p.x - c.x) <= 2 &&
          Math.abs(p.y - c.y) <= 2,
      ),
  );
  const showDense = zoom >= DENSE_DOT_ZOOM;

  async function doScan() {
    if (!scanRegion || scanning || !hasDims || zoom < 1.99) return;
    // Current viewport in native pixels. transform is scale(zoom) with
    // origin (focus%), so the visible span is 1/zoom of the frame
    // starting at origin*(1 - 1/zoom).
    const x0 = Math.max(0, Math.round((focus.x / 100) * (1 - 1 / zoom) * frameW));
    const y0 = Math.max(0, Math.round((focus.y / 100) * (1 - 1 / zoom) * frameH));
    const region = {
      x: x0, y: y0,
      w: Math.round(frameW / zoom),
      h: Math.round(frameH / zoom),
    };
    setScanning(true);
    setScanNote(null);
    try {
      const found = await scanRegion(region, scanLevel);
      const gk = (p) => `${p.frame}:${Math.round(p.x / 3)}:${Math.round(p.y / 3)}`;
      setScanDots((prev) => {
        const seen = new Set(
          [...(dots || []), ...(denseDots || []), ...prev].map(gk),
        );
        const fresh = (found || []).filter((p) => {
          const k = gk(p);
          if (seen.has(k)) return false;
          seen.add(k);
          return true;
        });
        if (fresh.length) {
          setScanNote(
            `+${fresh.length} new detections in view`
            + (scanLevel > 2 ? " (deep scan)" : ""),
          );
        } else if (scanLevel < 3) {
          // Nothing new at this level. Say what pressing it again will
          // do rather than leaving the operator to conclude the area is
          // empty when they can see the ball in it.
          setScanLevel(scanLevel + 1);
          setScanNote("nothing new — press Scan again to look harder");
        } else {
          setScanNote("no new motion found, even on the deepest scan");
        }
        return [...prev, ...fresh];
      });
    } catch (e) {
      console.warn("region scan failed", e);
      setScanNote("scan failed — try again");
    } finally {
      setScanning(false);
    }
  }
  // HANDLE DRAGGING. Measured against the scaled element itself, whose
  // bounding rect already accounts for the zoom and the transform
  // origin — deriving frame coordinates from the outer box instead is
  // right at 1x and wrong everywhere else.
  const stageRef = useRef(null);
  const [dragId, setDragId] = useState(null);
  function stageXY(e) {
    const el = stageRef.current;
    if (!el || !hasDims) return null;
    const r = el.getBoundingClientRect();
    return {
      x: Math.max(0, Math.min(frameW - 1,
        Math.round(((e.clientX - r.left) / r.width) * frameW))),
      y: Math.max(0, Math.min(frameH - 1,
        Math.round(((e.clientY - r.top) / r.height) * frameH))),
    };
  }

  const zoomBtn = {
    background: "rgba(255,255,255,0.12)", color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)", borderRadius: 4,
    width: 28, height: 26, fontSize: 13, fontWeight: 600,
    cursor: "pointer", padding: 0,
  };
  const panBy = (dx, dy) => {
    const step = 30 / Math.max(1, zoom);
    setFocus((c) => ({
      x: Math.max(0, Math.min(100, c.x + dx * step)),
      y: Math.max(0, Math.min(100, c.y + dy * step)),
    }));
  };
  return (
    <div
      ref={areaRef}
      style={{
        display: "flex", flexDirection: "column", gap: 6,
        height: "100%", minHeight: 0, maxWidth: "100%", width: "100%",
        alignItems: "center", justifyContent: "center",
      }}
    >
      {/* Image area — takes the whole box. The zoom / scan / pan controls
          are absolutely positioned INSIDE it (see the overlay below)
          rather than stacked above, so nothing but the map competes for
          the modal's height. */}
      <div
        style={{
          position: "relative",
          // The measured fit, in pixels. Falls back to the aspect-ratio
          // box for the first paint (and for anything without a
          // ResizeObserver), which is right often enough to not flash.
          ...(fit
            ? { width: fit.w, height: fit.h, flex: "0 0 auto" }
            : {
              flex: 1, minHeight: 0, maxHeight: "100%", maxWidth: "100%",
              aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
            }),
          background: "var(--border, #222)",
          borderRadius: 6, overflow: "hidden",
          userSelect: "none",
        }}
      >
      <div
        onPointerDown={
          placingBall && hasDims
            ? (e) => {
                // Measure against THIS element: it is the scaled one, so
                // its bounding rect already accounts for zoom and the
                // transform origin. Deriving frame coords from the outer
                // box instead would be wrong at any zoom but 1x.
                const r = e.currentTarget.getBoundingClientRect();
                const fx = Math.round(
                  ((e.clientX - r.left) / r.width) * frameW,
                );
                const fy = Math.round(
                  ((e.clientY - r.top) / r.height) * frameH,
                );
                onPlaceBall?.({
                  x: Math.max(0, Math.min(frameW - 1, fx)),
                  y: Math.max(0, Math.min(frameH - 1, fy)),
                });
              }
            : undefined
        }
        ref={stageRef}
        style={{
          position: "absolute", inset: 0,
          transform: `scale(${zoom})`,
          transformOrigin: `${focus.x}% ${focus.y}%`,
          transition: dragId ? "none" : "transform 120ms ease",
          cursor: placingBall ? "crosshair" : undefined,
        }}
      >
        <img
          src={(viewFrame != null && frameUrl) ? frameUrl : bgUrl}
          alt={viewFrame != null ? `Frame ${viewFrame}` : "Raw motion heat"}
          draggable={false}
          style={{
            width: "100%", height: "100%", objectFit: "cover",
            pointerEvents: "none",
          }}
        />
        {/* Saved tracer path — the swing's CURRENT ball track drawn as a
            green line so the operator can see where the rendered tracer
            actually sits relative to the real motion streaks (and to the
            clickable detections). Non-interactive. */}
        {hasDims && (track || []).length > 0 && (
          <svg
            viewBox={`0 0 ${frameW} ${frameH}`}
            preserveAspectRatio="none"
            style={{
              position: "absolute", inset: 0, width: "100%", height: "100%",
              pointerEvents: "none",
            }}
          >
            <polyline
              points={track.map((t) => `${t.x},${t.y}`).join(" ")}
              fill="none"
              stroke="#22c55e"
              strokeWidth={Math.max(2, frameW / 700)}
              strokeOpacity={0.9}
            />
            {track.map((t, i) => (
              <circle
                key={`t-${t.frame}-${i}`}
                cx={t.x}
                cy={t.y}
                r={Math.max(2.5, frameW / 500)}
                fill="#22c55e"
                fillOpacity={0.85}
                stroke="#052e16"
                strokeWidth={1}
              />
            ))}
            {track.length > 0 && (
              <text
                x={track[0].x + frameW / 120}
                y={track[0].y}
                fontSize={Math.max(11, frameW / 110)}
                fill="#4ade80"
                stroke="#000"
                strokeWidth={0.6}
                paintOrder="stroke"
              >
                {`track f${track[0].frame}`}
              </text>
            )}
            {track.length > 1 && (
              <text
                x={track[track.length - 1].x + frameW / 120}
                y={track[track.length - 1].y}
                fontSize={Math.max(11, frameW / 110)}
                fill="#4ade80"
                stroke="#000"
                strokeWidth={0.6}
                paintOrder="stroke"
              >
                {`f${track[track.length - 1].frame}`}
              </text>
            )}
          </svg>
        )}

        {/* THE COMET'S PATH. The chain of frames the ball was found on
            coming down on the green camera — exactly what produce will
            draw a comet along, so seeing it here is seeing the clip.
            Head-to-tail: thick and bright at the landing, thin and dim
            where the ball entered the search, so its direction reads
            without a legend. Non-interactive. */}
        {hasDims && (comet?.points?.length || 0) > 1 && (
          <svg
            viewBox={`0 0 ${frameW} ${frameH}`}
            preserveAspectRatio="none"
            style={{
              position: "absolute", inset: 0, width: "100%", height: "100%",
              pointerEvents: "none",
            }}
          >
            {comet.points.slice(1).map((p, i) => {
              const a = comet.points[i];
              const t = (i + 1) / (comet.points.length - 1);
              return (
                <line
                  key={`cm-${p.frame}`}
                  x1={a.x} y1={a.y} x2={p.x} y2={p.y}
                  stroke="#fff7ed"
                  strokeOpacity={0.35 + 0.6 * t}
                  strokeWidth={Math.max(1.5, (frameW / 900) * (1 + 3 * t))}
                  strokeLinecap="round"
                />
              );
            })}
            <circle
              cx={comet.points[comet.points.length - 1].x}
              cy={comet.points[comet.points.length - 1].y}
              r={Math.max(4, frameW / 260)}
              fill="#fff"
              stroke="#3ea6ff"
              strokeWidth={Math.max(1, frameW / 900)}
            />
            <text
              x={comet.points[0].x + frameW / 120}
              y={comet.points[0].y}
              fontSize={Math.max(11, frameW / 110)}
              fill="#fed7aa"
              stroke="#000"
              strokeWidth={0.6}
              paintOrder="stroke"
            >
              {`comet f${comet.points[0].frame}→f${
                comet.points[comet.points.length - 1].frame}`}
            </text>
          </svg>
        )}

        {/* THE TRACER'S OWN LINE, as the renderer would draw it: the
            tracked ball solid, the modelled continuation dashed, so
            what is measured and what is predicted are never confused
            for one another. */}
        {hasDims && (shape?.length || 0) > 1 && (
          <svg
            viewBox={`0 0 ${frameW} ${frameH}`}
            preserveAspectRatio="none"
            style={{
              position: "absolute", inset: 0, width: "100%", height: "100%",
              pointerEvents: "none",
            }}
          >
            <polyline
              points={shape.map((q) => `${q[0]},${q[1]}`).join(" ")}
              fill="none"
              stroke="#0b1220"
              strokeOpacity={0.55}
              strokeWidth={Math.max(5, frameW / 260)}
              strokeLinecap="round"
            />
            <polyline
              points={shape.map((q) => `${q[0]},${q[1]}`).join(" ")}
              fill="none"
              stroke="#67e8f9"
              strokeWidth={Math.max(2.5, frameW / 500)}
              strokeDasharray={`${Math.max(8, frameW / 90)} ${
                Math.max(6, frameW / 130)}`}
              strokeLinecap="round"
            />
          </svg>
        )}
        {hasDims && (handles || []).map((h) => (
          <div
            key={h.id}
            onPointerDown={(e) => {
              e.stopPropagation();
              e.currentTarget.setPointerCapture(e.pointerId);
              setDragId(h.id);
            }}
            onPointerMove={(e) => {
              if (dragId !== h.id) return;
              const pt = stageXY(e);
              if (pt) onHandleDrag?.(h.id, pt);
            }}
            onPointerUp={(e) => {
              if (dragId !== h.id) return;
              setDragId(null);
              const pt = stageXY(e);
              onHandleDrop?.(h.id, pt);
            }}
            title={h.title}
            style={{
              position: "absolute",
              left: `${(h.x / frameW) * 100}%`,
              top: `${(h.y / frameH) * 100}%`,
              // AN ICON HANGS FROM ITS OWN ANCHOR. A flagstick is at
              // the BOTTOM of the stick and a tee is at the bottom of
              // the peg, so centring them would put both a dozen pixels
              // above the thing they mark. A plain handle still
              // centres.
              transform: h.icon === "flag" || h.icon === "tee"
                ? "translate(-50%, -100%)" : "translate(-50%, -50%)",
              width: h.icon ? "auto" : Math.max(26, 26 / zoom),
              height: h.icon ? "auto" : Math.max(26, 26 / zoom),
              borderRadius: h.icon ? 0 : "50%",
              border: h.icon
                ? "none" : `2px solid ${h.colour || "#67e8f9"}`,
              // TINTED WITH ITS OWN COLOUR. Hard-coded cyan made every
              // handle look like the tracer's end handle, which is the
              // one thing a second draggable point must not be mistaken
              // for.
              background: h.icon ? "transparent" : (h.colour
                ? `${h.colour}${dragId === h.id ? "99" : "33"}`
                : (dragId === h.id
                  ? "rgba(103,232,249,0.55)"
                  : "rgba(103,232,249,0.18)")),
              boxShadow: h.icon ? "none" : "0 0 0 2px rgba(0,0,0,0.55)",
              cursor: h.axis === "y" ? "ns-resize" : "move",
              touchAction: "none",
              opacity: dragId === h.id ? 0.75 : 1,
              zIndex: 8,
            }}
          >
            {h.icon === "flag" ? (
              /* The flagstick as it has always been drawn here: a white
                 pole with a red pennant, anchored at its base. */
              <div style={{ position: "relative" }}>
                <div style={{
                  width: 2, height: Math.max(22, 22 / zoom),
                  background: "#fff",
                  boxShadow: "0 0 2px rgba(0,0,0,0.9)",
                }} />
                <div style={{
                  position: "absolute", left: 2, top: 0,
                  width: 0, height: 0,
                  borderTop: `${Math.max(7, 7 / zoom)}px solid transparent`,
                  borderBottom: `${Math.max(7, 7 / zoom)}px solid transparent`,
                  borderLeft: `${Math.max(13, 13 / zoom)}px solid #ef4444`,
                  filter: "drop-shadow(0 0 1px rgba(0,0,0,0.9))",
                }} />
              </div>
            ) : h.icon === "tee" ? (
              /* A golf tee: the cup, the stem, the point. Anchored at
                 the point, which is where the ball sat. */
              <div style={{ position: "relative",
                            width: Math.max(14, 14 / zoom),
                            height: Math.max(20, 20 / zoom) }}>
                <div style={{
                  position: "absolute", top: 0, left: 0, right: 0,
                  height: Math.max(4, 4 / zoom), background: "#fde68a",
                  borderRadius: 2,
                  boxShadow: "0 0 2px rgba(0,0,0,0.9)",
                }} />
                <div style={{
                  position: "absolute", top: Math.max(3, 3 / zoom),
                  left: "50%", transform: "translateX(-50%)",
                  width: 0, height: 0,
                  borderLeft: `${Math.max(3, 3 / zoom)}px solid transparent`,
                  borderRight: `${Math.max(3, 3 / zoom)}px solid transparent`,
                  borderTop: `${Math.max(17, 17 / zoom)}px solid #fde68a`,
                  filter: "drop-shadow(0 0 1px rgba(0,0,0,0.9))",
                }} />
              </div>
            ) : h.icon === "ball" ? (
              /* A golf ball: white, ringed so it reads against turf. */
              <div style={{
                width: Math.max(15, 15 / zoom),
                height: Math.max(15, 15 / zoom),
                borderRadius: "50%",
                background: "radial-gradient(circle at 35% 30%, #fff, #cbd5e1)",
                border: "1px solid #0f172a",
                boxShadow: "0 0 0 2px rgba(0,0,0,0.5), 0 0 5px rgba(0,0,0,0.7)",
              }} />
            ) : null}
          </div>
        ))}

        {/* THE FLAG. Small, gold, and not interactive — it is a
            landmark rather than a control, and the one thing on this
            map that is about the HOLE rather than about this swing. */}
        {hasDims && flag && (
          <div
            style={{
              position: "absolute",
              left: `${(flag.x / frameW) * 100}%`,
              top: `${(flag.y / frameH) * 100}%`,
              transform: "translate(-10%, -100%)",
              fontSize: Math.max(15, 15 / zoom),
              lineHeight: 1,
              pointerEvents: "none",
              filter: "drop-shadow(0 0 3px #000) drop-shadow(0 0 3px #000)",
              zIndex: 7,
            }}
            title={`Flagstick — ${flag.x}, ${flag.y}`}
          >
            ⛳
          </div>
        )}
        {/* BALL AT IMPACT — where the tracer line STARTS. Not a track
            point: the renderer anchors the fitted curve here, so it is
            the one marker that decides where the line begins rather than
            where it passes through. Drawn as a ringed crosshair so it
            cannot be mistaken for a plotted flight point. */}
        {hasDims && ballXY && (
          <div
            style={{
              position: "absolute",
              left: `${(ballXY.x / frameW) * 100}%`,
              top: `${(ballXY.y / frameH) * 100}%`,
              width: Math.max(22, 22 / zoom),
              height: Math.max(22, 22 / zoom),
              marginLeft: -Math.max(11, 11 / zoom),
              marginTop: -Math.max(11, 11 / zoom),
              borderRadius: "50%",
              border: "2px solid #f0fdf4",
              boxShadow: "0 0 0 2px #16a34a, 0 0 6px rgba(0,0,0,0.8)",
              background: "rgba(22,163,74,0.25)",
              pointerEvents: "none",
              zIndex: 5,
            }}
            title={`Tracer start — ball at impact (${ballXY.x}, ${ballXY.y})`}
          />
        )}
        {/* PRODUCTION TRACK POINTS, clickable. The green line is drawn
            in an SVG layer with pointerEvents:none, so the points
            production put in the track could be SEEN but not touched —
            and most of them (AI launch picks, launch-tracker points, arc
            completion) have no detection dot underneath to click
            instead. These targets sit on top so a wrong point can just
            be clicked off. zIndex keeps them above the amber dots where
            the two overlap, and the hit area has a floor so it stays
            clickable when zoomed in. */}
        {hasDims && track.map((t, i) => {
          const still = marks[t.frame];
          const kept =
            !!still &&
            Math.abs(still.x - t.x) <= 2 &&
            Math.abs(still.y - t.y) <= 2;
          const hit = Math.max(15, 15 / zoom);
          return (
            <div
              key={`tp-${t.frame}-${i}`}
              onPointerDown={(e) => {
                e.stopPropagation();
                onToggleDot({ frame: t.frame, x: t.x, y: t.y }, kept);
              }}
              onContextMenu={(e) => {
                e.preventDefault();
                e.stopPropagation();
                onToggleDot({ frame: t.frame, x: t.x, y: t.y }, true);
              }}
              title={
                kept
                  ? `Frame ${t.frame} — in the produced tracer. Click to REMOVE it.`
                  : `Frame ${t.frame} — removed. Click to put it back.`
              }
              style={{
                position: "absolute",
                left: `${(t.x / frameW) * 100}%`,
                top: `${(t.y / frameH) * 100}%`,
                width: hit, height: hit,
                borderRadius: "50%",
                border: kept
                  ? "2px solid #22c55e"
                  : "2px dashed rgba(239,68,68,0.95)",
                background: kept
                  ? "rgba(34,197,94,0.30)"
                  : "rgba(239,68,68,0.10)",
                transform: "translate(-50%, -50%)",
                cursor: "pointer",
                zIndex: 4,
                boxShadow: "0 0 4px rgba(0,0,0,0.6)",
              }}
            />
          );
        })}
        {/* Dense candidate layer — smaller, dimmer targets that only
            appear once zoomed in, so the fit view stays readable. */}
        {hasDims && showDots && showDense &&
          extraDots.map((p, i) => {
            const q = marks[p.frame];
            const isQueued =
              !!q && Math.abs(q.x - p.x) <= 2 && Math.abs(q.y - p.y) <= 2;
            return (
              <div
                key={`d-${p.frame}-${i}`}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  onToggleDot(p, e.altKey || e.button === 2);
                }}
                onContextMenu={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onToggleDot(p, true);
                }}
                title={`Frame ${p.frame} · ${p.x}, ${p.y} (candidate) — ${
                  isQueued
                    ? "queued (click to un-queue)"
                    : "click to queue the ball here for this frame"
                } · alt/right-click clears this frame"`}
                style={{
                  position: "absolute",
                  left: `${(p.x / frameW) * 100}%`,
                  top: `${(p.y / frameH) * 100}%`,
                  width: 10, height: 10,
                  borderRadius: "50%",
                  border: isQueued
                    ? "2px solid #fff"
                    : "1px solid rgba(245,158,11,0.9)",
                  background: isQueued
                    ? "#22c55e"
                    : "rgba(245,158,11,0.18)",
                  transform: `translate(-50%, -50%) scale(${1 / zoom})`,
                  cursor: "pointer",
                  boxShadow: "0 0 4px rgba(0,0,0,0.6)",
                }}
              >
                <span
                  style={{
                    position: "absolute",
                    left: 10,
                    top: i % 2 === 0 ? -11 : 9,
                    fontSize: 9,
                    color: isQueued ? "#4ade80" : "rgba(253,224,71,0.85)",
                    textShadow: "0 0 3px #000, 0 0 3px #000",
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                  }}
                >
                  {p.frame}
                </span>
              </div>
            );
          })}
        {hasDims && showDots &&
          (dots || []).map((p, i) => {
            const q = marks[p.frame];
            const isQueued =
              !!q && Math.abs(q.x - p.x) <= 2 && Math.abs(q.y - p.y) <= 2;
            return (
              <div
                key={`${p.frame}-${i}`}
                onPointerDown={(e) => {
                  e.stopPropagation();
                  onToggleDot(p, e.altKey || e.button === 2);
                }}
                onContextMenu={(e) => {
                  e.preventDefault();
                  e.stopPropagation();
                  onToggleDot(p, true);
                }}
                title={`Frame ${p.frame} · ${p.x}, ${p.y} — ${
                  isQueued
                    ? "queued (click to un-queue)"
                    : "click to queue the ball here for this frame"
                } · alt/right-click clears this frame`}
                style={{
                  position: "absolute",
                  left: `${(p.x / frameW) * 100}%`,
                  top: `${(p.y / frameH) * 100}%`,
                  width: 14, height: 14,
                  borderRadius: "50%",
                  border: isQueued ? "2px solid #fff" : "2px solid #f59e0b",
                  background: isQueued ? "#22c55e" : "rgba(245,158,11,0.35)",
                  transform: `translate(-50%, -50%) scale(${1 / zoom})`,
                  cursor: "pointer",
                  boxShadow: "0 0 6px rgba(0,0,0,0.7)",
                }}
              >
                {/* Frame label — alternates above/below like the baked
                    frames-on-heat image, so dense stretches stay
                    readable. Inherits the counter-scale. */}
                <span
                  style={{
                    position: "absolute",
                    left: 13,
                    top: i % 2 === 0 ? -13 : 11,
                    fontSize: 10, fontWeight: 600,
                    color: isQueued ? "#4ade80" : "#fde047",
                    textShadow: "0 0 3px #000, 0 0 3px #000, 0 0 3px #000",
                    whiteSpace: "nowrap",
                    pointerEvents: "none",
                  }}
                >
                  {p.frame}
                </span>
              </div>
            );
          })}
      </div>
      {(note || scanNote || (extraDots.length > 0 && !showDense)) && (
        <div
          style={{
            position: "absolute", left: 8, bottom: 8,
            background: "rgba(0,0,0,0.55)",
            color: (note && noteColour) || "#fde047",
            padding: "3px 10px", borderRadius: 6, fontSize: 12,
            pointerEvents: "none", backdropFilter: "blur(4px)",
            // Stay clear of the control overlay in the opposite corner.
            maxWidth: "min(55%, 520px)",
          }}
        >
          {scanNote
            ? `🔍 ${scanNote}`
            : note
              || `🔍 zoom to ${DENSE_DOT_ZOOM}×+ to reveal ${extraDots.length} more clickable detections`}
        </div>
      )}
      {/* FLOATING CONTROLS. These used to be a SIBLING above the image
          (order:-1) to keep them off the map, which cost the map a whole
          toolbar's height on every screen — on a 16:9 frame in a 96vh
          modal that is the difference between a comfortable click target
          and a squint. Floating them bottom-right instead gives the map
          the full box: they sit over the one corner the ball flight has
          already left, they blur out what is behind them, and pointer
          events stop here so a click on a button is never also a click
          on the map. */}
      <div
        onPointerDown={(e) => e.stopPropagation()}
        onClick={(e) => e.stopPropagation()}
        style={{
          position: "absolute", right: 8, bottom: 8, zIndex: 30,
          display: "flex", gap: 6, flexWrap: "wrap",
          justifyContent: "flex-end", maxWidth: "calc(100% - 16px)",
          background: "rgba(0,0,0,0.6)", padding: "5px 7px",
          borderRadius: 8, backdropFilter: "blur(6px)",
          border: "1px solid rgba(255,255,255,0.16)",
          boxShadow: "0 4px 14px rgba(0,0,0,0.45)",
        }}
      >
        {/* FRAME STEPPER. Off by default: the heat composite is the
            view that shows a whole flight at once, and this trades that
            for being able to see one instant properly. */}
        {canStep && viewFrame != null && (
          <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
            <button type="button" style={{ ...zoomBtn, width: 28 }}
                    disabled={viewFrame <= frameLo}
                    onClick={() => stepFrame(-10)}
                    title="Back ten frames">‹‹</button>
            <button type="button" style={{ ...zoomBtn, width: 22 }}
                    disabled={viewFrame <= frameLo}
                    onClick={() => stepFrame(-1)}
                    title="Back one frame">‹</button>
            <span style={{
              color: frameErr ? "#f87171" : "#fff", fontSize: 12,
              padding: "0 3px", alignSelf: "center",
              minWidth: 54, textAlign: "center", fontWeight: 700,
            }}>
              {frameErr ? "error" : `f${viewFrame}`}
            </span>
            <button type="button" style={{ ...zoomBtn, width: 22 }}
                    disabled={viewFrame >= frameHi}
                    onClick={() => stepFrame(1)}
                    title="Forward one frame">›</button>
            <button type="button" style={{ ...zoomBtn, width: 28 }}
                    disabled={viewFrame >= frameHi}
                    onClick={() => stepFrame(10)}
                    title="Forward ten frames">››</button>
          </div>
        )}
        {/* MOG2 dots on/off. On a frame where the ball is a three-pixel
            smudge, the dot marking it is bigger than the ball. */}
        <button
          type="button"
          style={{
            ...zoomBtn, width: "auto", padding: "0 8px",
            background: showDots
              ? zoomBtn.background : "rgba(248,113,113,0.35)",
          }}
          onClick={() => setShowDots((v) => !v)}
          title={showDots
            ? "Hide the MOG2 dots — see the picture underneath"
            : "Show the MOG2 dots again"}
        >
          {showDots ? "⦿ MOG2" : "⦾ MOG2"}
        </button>
        <button
          type="button"
          style={zoomBtn}
          disabled={zoom <= 1.05}
          onClick={() => setZoom((z) => Math.max(1, z / 1.4))}
          title="Zoom out"
        >−</button>
        <span style={{ color: "#fff", fontSize: 12, padding: "0 6px", alignSelf: "center" }}>
          {zoom.toFixed(1)}×
        </span>
        <button
          type="button"
          style={zoomBtn}
          disabled={zoom >= 15.9}
          onClick={() => setZoom((z) => Math.min(16, z * 1.4))}
          title="Zoom in"
        >+</button>
        <button
          type="button"
          style={{ ...zoomBtn, width: 36 }}
          onClick={() => { setZoom(1); setFocus({ x: 50, y: 50 }); }}
          title="Fit full frame"
        >Fit</button>
        {scanRegion && (
          <button
            type="button"
            style={{
              ...zoomBtn, width: "auto", padding: "0 8px",
              background: scanning
                ? "rgba(245,158,11,0.7)"
                : zoomBtn.background,
            }}
            disabled={scanning || zoom < 1.99}
            onClick={doScan}
            title={
              zoom < 1.99
                ? "Zoom to at least 2× first, then Scan finds every motion blob in the visible area"
                : `Deep-scan the visible area (level ${scanLevel} of 3): frame-diff over the swing window with much looser gates than the tracer — every transient blob in view becomes a clickable dot. Finds nothing? Press again; it steps up a level. Takes a few seconds.`
            }
          >
            {scanning
              ? "Scanning…"
              : scanLevel > 2 ? "🔍 Scan harder" : "🔍 Scan"}
          </button>
        )}
        <div style={{
          display: "grid",
          gridTemplateColumns: "repeat(3, 22px)",
          gridTemplateRows: "22px 22px",
          gap: 2, marginLeft: 4,
        }}>
          <span />
          <button
            type="button"
            style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
            disabled={zoom <= 1.05 || focus.y <= 0.1}
            onClick={() => panBy(0, -1)}
            title="Pan up"
          >↑</button>
          <span />
          <button
            type="button"
            style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
            disabled={zoom <= 1.05 || focus.x <= 0.1}
            onClick={() => panBy(-1, 0)}
            title="Pan left"
          >←</button>
          <button
            type="button"
            style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
            disabled={zoom <= 1.05 || focus.y >= 99.9}
            onClick={() => panBy(0, 1)}
            title="Pan down"
          >↓</button>
          <button
            type="button"
            style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
            disabled={zoom <= 1.05 || focus.x >= 99.9}
            onClick={() => panBy(1, 0)}
            title="Pan right"
          >→</button>
        </div>
        {onClose && (
          <button
            type="button"
            style={{ ...zoomBtn, width: 48 }}
            onClick={onClose}
            title="Close this view"
          >Close</button>
        )}
      </div>
      </div>
    </div>
  );
}

function TracerStep({
  row, adminPassword, draft, setDraft, tracer, setTracer,
  rendering, setRendering, error, setError,
  frameW, frameH, totalFrames, onSaved, persistPatch,
  manualPositions, setManualPositions,
}) {
  // manualPositions / setManualPositions are hoisted to the wizard so
  // Step 3 can see the queued edits (red note) and pass them to
  // /render-tracer-fast on commit. Each entry is {frame: {x, y}}.
  const [selectedFrame, setSelectedFrame] = useState(null);
  // One tee frame as a picture, for the plot map's frame stepper.
  const loadStepFrame = useCallback(
    async (f) => {
      const r = await api.getLongUploadFrame(adminPassword, row.id, f, "tee");
      return r?.image_url;
    },
    [adminPassword, row.id],
  );
  const [editorBg, setEditorBg] = useState(null); // {url, frame}
  const [editorBall, setEditorBall] = useState(null); // {x, y}
  const [zoom, setZoom] = useState(1);
  // Full-screen zoomable viewer for a card's detector-view image.
  const [imgView, setImgView] = useState(null); // {url, title}
  // Editor background: raw frame vs the full-frame detector view (MOG2
  // mask + candidates baked in). Defaults ON so clicking a card lands on
  // the evidence; the 🔥 toolbar button toggles back to the clean frame.
  const [detectorView, setDetectorView] = useState(true);
  // Frames the operator explicitly cleared. Sent to the backend so
  // the renderer drops them from the ball track entirely (AI marks
  // included). Reset after a successful re-render.
  const [clearedFrames, setClearedFrames] = useState(() => new Set());
  // Clickable MOG2 candidate dots (classical/KNN renders): amber snap
  // targets on the editor — click one to mark the ball exactly there.
  const [showCandidates, setShowCandidates] = useState(true);
  // Click-to-plot: the frames-on-heat view rebuilt as an interactive
  // screen — every timed motion dot (labelled with its frame) drawn over
  // the raw-motion heat, clickable to queue the ball for that frame.
  const [plotAll, setPlotAll] = useState(false);
  const editorRef = useRef(null);

  // Default zoom level when dropping straight onto a ball position.
  // 3x is tight enough to nudge pixel-perfect, loose enough to keep
  // surrounding context visible if the AI was a couple frames off.
  const BALL_ZOOM = 3;

  const frames = tracer?.frames || [];
  const hasDims = !!(frameW && frameH);
  const maxFrame = totalFrames ? totalFrames - 1 : null;
  // MOG2 candidate detections on the frame being edited (classical/KNN
  // renders only; empty for AI or after a wizard reopen).
  const selectedFrameCands =
    selectedFrame != null
      ? (tracer?.candidates || []).filter((c) => c.frame === selectedFrame)
      : [];
  const visibleCands = showCandidates ? selectedFrameCands : [];

  // Synthetic "rest" entry shown as the first card in the grid: the
  // ball at its resting position two frames before impact. The
  // tracer renderer already anchors there (REST_ANCHOR_FRAMES_BEFORE_IMPACT
  // on the backend); this entry just gives the operator a clickable
  // card so they can verify or fine-tune the start of the line.
  // Skipped when the AI already produced a frame at the same index
  // or when we don't have impact/ball context yet.
  const restFrame =
    draft?.impactFrame != null
      ? Math.max(0, draft.impactFrame - 2)
      : null;
  const restEntry =
    draft?.ball && restFrame != null && !frames.some((f) => f.frame === restFrame)
      ? {
          frame: restFrame,
          found: true,
          x: draft.ball.x,
          y: draft.ball.y,
          manual: false,
          image_url: null,
          rest: true,
        }
      : null;
  const displayFrames = restEntry ? [restEntry, ...frames] : frames;

  // Pivot for the zoom transform — defaults to the ball detection
  // ROI from Step 1 so the operator drops straight into the ball
  // area. Falls back to the resting-ball point, then frame centre.
  // Pan override. Defaults to null so the auto-computed ROI focus
  // applies; arrow buttons in the zoom toolbar set it once the user
  // wants to shift the viewable area.
  const [focusOverride, setFocusOverride] = useState(null);

  const autoFocusPct = (() => {
    const roi = draft?.roi;
    if (roi && hasDims) {
      return {
        x: ((roi.x + roi.w / 2) / frameW) * 100,
        y: ((roi.y + roi.h / 2) / frameH) * 100,
      };
    }
    if (draft?.ball && hasDims) {
      return {
        x: (draft.ball.x / frameW) * 100,
        y: (draft.ball.y / frameH) * 100,
      };
    }
    return { x: 50, y: 50 };
  })();
  const focusPct = focusOverride || autoFocusPct;

  function panBy(dx, dy) {
    // Each press shifts the focus by ~30% of the visible region.
    // visible region in frame-% = 100/zoom, so step = 30/zoom.
    const step = 30 / Math.max(1, zoom);
    setFocusOverride((prev) => {
      const cur = prev || autoFocusPct;
      return {
        x: Math.max(0, Math.min(100, cur.x + dx * step)),
        y: Math.max(0, Math.min(100, cur.y + dy * step)),
      };
    });
  }

  // Auto-zoom level that makes the ROI fill ~70% of the editor.
  // Capped so the ball doesn't disappear off-screen at extreme ratios.
  const autoZoom = (() => {
    const roi = draft?.roi;
    if (!roi || !hasDims) return 1;
    const zx = (0.7 * frameW) / Math.max(1, roi.w);
    const zy = (0.7 * frameH) / Math.max(1, roi.h);
    return Math.max(1, Math.min(8, Math.min(zx, zy)));
  })();

  function mergedBallFor(f) {
    if (clearedFrames.has(f.frame)) return null;
    const m = manualPositions[f.frame];
    if (m) return { x: m.x, y: m.y, manual: true };
    if (f.found && f.x != null && f.y != null) return { x: f.x, y: f.y, manual: !!f.manual };
    return null;
  }

  // Returns the most useful ball position for centring the editor:
  // manual mark on `frameIdx`, then AI detection on `frameIdx`, then
  // walk backward to the most recent known position, then forward
  // looking for the next known one. Skips frames the operator cleared.
  function findCentringBall(frameIdx) {
    const ballAt = (f) => {
      if (clearedFrames.has(f)) return null;
      const m = manualPositions[f];
      if (m) return { x: m.x, y: m.y };
      const rec = (tracer?.frames || []).find((r) => r.frame === f);
      if (rec?.found && rec.x != null && rec.y != null) {
        return { x: rec.x, y: rec.y };
      }
      if (f === restFrame && draft?.ball) {
        return { x: draft.ball.x, y: draft.ball.y };
      }
      return null;
    };
    const here = ballAt(frameIdx);
    if (here) return here;
    for (let f = frameIdx - 1; f >= 0; f--) {
      const b = ballAt(f);
      if (b) return b;
    }
    const ahead = [...(tracer?.frames || [])]
      .map((r) => r.frame)
      .concat(Object.keys(manualPositions).map((k) => parseInt(k, 10)))
      .filter((f) => f > frameIdx)
      .sort((a, b) => a - b);
    for (const f of ahead) {
      const b = ballAt(f);
      if (b) return b;
    }
    return null;
  }

  async function loadEditorFrame(frameIdx) {
    setPlotAll(false);
    setSelectedFrame(frameIdx);
    setEditorBg(null);
    setEditorBall(null);
    // Centre + zoom on the ball (current frame, else last known). If
    // nothing's ever been marked, fall back to the ROI auto-zoom so
    // the operator still drops into the right neighbourhood.
    const centring = findCentringBall(frameIdx);
    if (centring && hasDims) {
      setZoom(BALL_ZOOM);
      setFocusOverride({
        x: (centring.x / frameW) * 100,
        y: (centring.y / frameH) * 100,
      });
    } else {
      setZoom(autoZoom);
      setFocusOverride(null);
    }
    try {
      const data = await api.getLongUploadFrame(adminPassword, row.id, frameIdx);
      // Full-frame detector-view overlay (MOG2 mask + candidates baked
      // in, same coordinate space as the raw frame) when the classical
      // engine produced one for this frame — the editor can toggle it.
      const trackEntry = (tracer?.frames || []).find((f) => f.frame === frameIdx);
      setEditorBg({
        url: data.image_url,
        frame: data.frame,
        overlayUrl: trackEntry?.overlay_image_url || null,
      });
      const existing = trackEntry;
      const m = manualPositions[frameIdx];
      if (m) setEditorBall({ x: m.x, y: m.y });
      else if (!clearedFrames.has(frameIdx) && existing?.found && existing.x != null) {
        setEditorBall({ x: existing.x, y: existing.y });
      } else if (frameIdx === restFrame && draft?.ball) {
        // Synthetic rest entry: show the rest position from Step 1.
        setEditorBall({ x: draft.ball.x, y: draft.ball.y });
      }
    } catch (e) {
      console.warn("frame fetch failed", e);
    }
  }

  function applyEditorBall() {
    if (selectedFrame == null || !editorBall) return;
    setManualPositions((m) => ({
      ...m,
      [selectedFrame]: { x: editorBall.x, y: editorBall.y },
    }));
  }

  function clearEditorBall() {
    if (selectedFrame == null) return;
    // Drop any queued manual mark for this frame.
    setManualPositions((m) => {
      const next = { ...m };
      delete next[selectedFrame];
      return next;
    });
    // If the AI had a detection here, mark the frame as explicitly
    // cleared so the backend renderer drops it from the merged track
    // on the next re-render.
    const existing = (tracer?.frames || []).find((f) => f.frame === selectedFrame);
    if (existing?.found && existing.x != null) {
      setClearedFrames((s) => {
        const next = new Set(s);
        next.add(selectedFrame);
        return next;
      });
    }
    setEditorBall(null);
  }

  function addFrame(delta) {
    // Advance from the operator's current view if they've already
    // walked past the AI-tracked frames; otherwise start from the
    // last AI-tracked frame. Without the selectedFrame fallback,
    // repeated "+5 frames" clicks always anchored to the same AI
    // max and stayed on the same target.
    const lastTracked = frames.length
      ? Math.max(...frames.map((f) => f.frame))
      : 0;
    const base = selectedFrame != null
      ? Math.max(selectedFrame, lastTracked)
      : lastTracked;
    const target = Math.max(0, Math.min(maxFrame ?? base + delta, base + delta));
    loadEditorFrame(target);
  }

  async function regenerate() {
    // cv2-only re-render: merges queued manual positions into the
    // existing ball-track and renders the tracer overlay. No Claude.
    setRendering(true);
    setError(null);
    try {
      const overrides = Object.entries(manualPositions).map(
        ([f, p]) => ({ frame: parseInt(f, 10), x: p.x, y: p.y })
      );
      const cleared = Array.from(clearedFrames);
      // Pass THIS swing's own data + frame window so the backend renders
      // only the selected swing's segment (not the whole clip) and anchors
      // to the right swing on a multi-swing upload.
      const hasWindow =
        draft?.startFrame != null && draft?.endFrame != null;
      const out = await api.renderWizardTracerFast(adminPassword, row.id, {
        manual_positions: overrides,
        cleared_frames: cleared,
        base_track_frames: tracer?.frames || [],
        impact_frame: draft?.impactFrame ?? null,
        ball_at_rest: draft?.ball || null,
        target: draft?.target || null,
        render_window: hasWindow
          ? { start_frame: draft.startFrame, end_frame: draft.endFrame }
          : null,
      });
      setTracer((t) => ({
        url: out.tracer_url,
        frames: out.ball_track_frames || [],
        debugUrl: t?.debugUrl || null,
        rawMotionUrl: t?.rawMotionUrl || null,
        rawMotionArcUrl: t?.rawMotionArcUrl || null,
        rawMotionFramesUrl: t?.rawMotionFramesUrl || null,
        mog2OverlayUrl: t?.mog2OverlayUrl || null,
        // Keep the clickable candidate/timed dots across the cv2 fast
        // re-render — the detections themselves haven't changed.
        candidates: t?.candidates || [],
        timedPoints: t?.timedPoints || [],
      }));
      setManualPositions({});
      setClearedFrames(new Set());
      setSelectedFrame(null);
      setEditorBg(null);
      setEditorBall(null);
      // Persist the merged track per swing so re-opens hydrate it
      // (with the operator's manual marks baked in) instead of
      // falling back to a fresh AI re-run on the next Step 1 → 2.
      await persistPatch?.({
        tracer_url: out.tracer_url,
        ball_track_frames: out.ball_track_frames || [],
      });
      onSaved?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setRendering(false);
    }
  }

  function editorEventToFrame(e) {
    if (!editorRef.current || !hasDims) return null;
    const r = editorRef.current.getBoundingClientRect();
    // Pointer fraction of the editor container (pre-transform).
    const xPct = (e.clientX - r.left) / r.width;
    const yPct = (e.clientY - r.top) / r.height;
    // Account for the CSS transform: scale(zoom) with transform-origin
    // = focusPct%. After scaling, a point originally at (a, b) in the
    // unscaled image (in editor-fraction coords) sits at
    //   (ox + (a - ox) * zoom, oy + (b - oy) * zoom)
    // in the editor. Invert to recover the image coord:
    const ox = focusPct.x / 100;
    const oy = focusPct.y / 100;
    const imageX = ox + (xPct - ox) / zoom;
    const imageY = oy + (yPct - oy) / zoom;
    return {
      x: Math.max(0, Math.min(frameW - 1, Math.round(imageX * frameW))),
      y: Math.max(0, Math.min(frameH - 1, Math.round(imageY * frameH))),
    };
  }

  function commitPoint(pt) {
    // Queue `pt` as the ball position for the selected frame — shared by
    // free clicks on the frame and clicks on a MOG2 candidate dot.
    setEditorBall(pt);
    if (selectedFrame == null) return;
    // Clicking on the REST card's frame moves the resting-ball anchor
    // itself (the start of the tracer line), not a flight point.
    if (selectedFrame === restFrame) {
      setDraft?.((d) => ({ ...d, ball: { x: pt.x, y: pt.y }, ballManual: true }));
      persistPatch?.({ ball: { x: pt.x, y: pt.y }, ball_manual: true });
      return;
    }
    setManualPositions((m) => ({ ...m, [selectedFrame]: pt }));
    // Marking a position un-clears the frame: the operator is
    // putting a ball back, so we shouldn't tell the backend to
    // drop it.
    if (clearedFrames.has(selectedFrame)) {
      setClearedFrames((s) => {
        const next = new Set(s);
        next.delete(selectedFrame);
        return next;
      });
    }
  }

  function onEditorPointerDown(e) {
    // Click auto-queues the position so navigating to another frame
    // doesn't drop the work. Add Frame button is just for explicit
    // confirmation now — clicking already commits.
    const pt = editorEventToFrame(e);
    if (!pt) return;
    commitPoint(pt);
  }

  function toggleTimedDot(p) {
    // Click-to-plot dot: queue the ball at this dot for the dot's frame.
    // Clicking the already-queued dot un-queues it; clicking a different
    // dot on the same frame replaces the pick (radio-button per frame).
    const f = p.frame;
    const cur = manualPositions[f];
    if (cur && Math.abs(cur.x - p.x) <= 2 && Math.abs(cur.y - p.y) <= 2) {
      setManualPositions((m) => {
        const next = { ...m };
        delete next[f];
        return next;
      });
      return;
    }
    setManualPositions((m) => ({ ...m, [f]: { x: p.x, y: p.y } }));
    if (clearedFrames.has(f)) {
      setClearedFrames((s) => {
        const next = new Set(s);
        next.delete(f);
        return next;
      });
    }
  }

  function openPlotAll() {
    setPlotAll(true);
    setSelectedFrame(null);
    setEditorBg(null);
    setEditorBall(null);
  }


  const zoomBtn = {
    background: "rgba(255,255,255,0.12)", color: "#fff",
    border: "1px solid rgba(255,255,255,0.3)", borderRadius: 4,
    width: 28, height: 26, fontSize: 13, fontWeight: 600,
    cursor: "pointer", padding: 0,
  };

  if (rendering && !tracer) {
    return (
      <div className="row" style={{ alignItems: "center", gap: 12 }}>
        <div className="shimmer" style={{ width: 18, height: 18, borderRadius: "50%" }} />
        <span className="small">Rendering tracer… this can take 30–60s.</span>
      </div>
    );
  }

  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(320px, 1.5fr) minmax(280px, 1fr)",
        gap: 16,
        height: "100%",
        minHeight: 0,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div className="tiny upper muted" style={{ marginBottom: 4 }}>
          {plotAll
            ? `Click-to-plot · ${(tracer?.timedPoints || []).length} timed dots · ${Object.keys(manualPositions).length} queued`
            : selectedFrame != null
              ? `Editing frame ${selectedFrame}`
              : "Rendered tracer"}
        </div>
        <div
          style={{
            flex: 1, minHeight: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          {plotAll ? (
            <PlotHeatCanvas
              bgUrl={
                tracer?.rawMotionUrl
                || tracer?.rawMotionFramesUrl
                || tracer?.mog2OverlayUrl
              }
              // THE SAME WINDOW THE STANDALONE PLOT USES. This one had
              // a floor at impact and no ceiling at all, so every dot
              // MOG2 found for the rest of the clip was on the map --
              // the golfer walking off, a cart, wind in the trees --
              // and none of it can ever be a right pick. Forty frames is
              // about as many points as the detector realistically
              // yields past the strike.
              dots={(tracer?.timedPoints || []).filter(
                (p) => draft?.impactFrame == null
                  || (p.frame >= draft.impactFrame
                      && p.frame <= draft.impactFrame + PLOT_WINDOW_POST),
              )}
              denseDots={(tracer?.candidates || []).filter(
                (p) => draft?.impactFrame == null
                  || (p.frame >= draft.impactFrame
                      && p.frame <= draft.impactFrame + PLOT_WINDOW_POST),
              )}
              frameW={frameW}
              frameH={frameH}
              marks={manualPositions}
              // Same stepper as the standalone plot: this is the same
              // map, and an operator who learns it in one place should
              // not find it missing in the other.
              loadFrame={loadStepFrame}
              frameLo={draft?.startFrame ?? 0}
              frameHi={draft?.endFrame
                ?? (totalFrames ? totalFrames - 1 : 0)}
              startFrame={draft?.impactFrame ?? undefined}
              onToggleDot={toggleTimedDot}
              onClose={() => setPlotAll(false)}
              scanRegion={async (region, sensitivity) => {
                const out = await api.scanPlotRegion(adminPassword, row.id, {
                  ...region,
                  sensitivity,
                  start_frame:
                    draft?.impactFrame != null
                      ? Math.max(
                          draft.startFrame ?? 0,
                          draft.impactFrame - 2,
                        )
                      : draft?.startFrame ?? 0,
                  end_frame: draft?.endFrame ?? null,
                });
                return out.dots || [];
              }}
            />
          ) : selectedFrame != null ? (
            <div
              ref={editorRef}
              onPointerDown={onEditorPointerDown}
              style={{
                position: "relative",
                height: "100%", maxHeight: "100%", maxWidth: "100%",
                aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
                background: "var(--border, #222)",
                borderRadius: 6, overflow: "hidden",
                cursor: "crosshair", userSelect: "none",
              }}
            >
              {/* Scaled scene so the ball detection ROI fills the
                  viewer when first opened. transform-origin pins the
                  ROI centre as the zoom pivot. Overlays positioned in
                  frame % live inside this div so they track the zoom. */}
              <div
                style={{
                  position: "absolute", inset: 0,
                  transform: `scale(${zoom})`,
                  transformOrigin: `${focusPct.x}% ${focusPct.y}%`,
                  transition: "transform 120ms ease",
                }}
              >
                {editorBg?.url ? (
                  <img
                    src={
                      detectorView && editorBg.overlayUrl
                        ? editorBg.overlayUrl
                        : editorBg.url
                    }
                    alt={`Frame ${selectedFrame}`}
                    draggable={false}
                    style={{
                      width: "100%", height: "100%", objectFit: "cover",
                      pointerEvents: "none",
                    }}
                  />
                ) : (
                  <div
                    className="muted small"
                    style={{
                      position: "absolute", inset: 0,
                      display: "flex", alignItems: "center", justifyContent: "center",
                    }}
                  >
                    Loading frame…
                  </div>
                )}
                {/* Clickable MOG2 candidate dots for THIS frame — click
                    one to snap the ball mark exactly onto the detection
                    instead of eyeballing it. Rendered before the green
                    marker so the chosen point paints on top. */}
                {hasDims &&
                  visibleCands.map((c, i) => (
                    <div
                      key={`${c.x}-${c.y}-${i}`}
                      onPointerDown={(e) => {
                        e.stopPropagation();
                        commitPoint({ x: c.x, y: c.y });
                      }}
                      title={`MOG2 candidate at ${c.x}, ${c.y} — click to mark the ball here for frame ${selectedFrame}`}
                      style={{
                        position: "absolute",
                        left: `${(c.x / frameW) * 100}%`,
                        top: `${(c.y / frameH) * 100}%`,
                        width: 16, height: 16,
                        borderRadius: "50%",
                        border: "2px solid #f59e0b",
                        background: "rgba(245,158,11,0.28)",
                        // Counter-scale so the target stays a constant
                        // visual size (and hit area) at any zoom level.
                        transform: `translate(-50%, -50%) scale(${1 / zoom})`,
                        cursor: "pointer",
                        boxShadow: "0 0 6px rgba(0,0,0,0.6)",
                      }}
                    />
                  ))}
                {hasDims && editorBall && (
                  <div
                    style={{
                      position: "absolute",
                      left: `${(editorBall.x / frameW) * 100}%`,
                      top: `${(editorBall.y / frameH) * 100}%`,
                      width: 18, height: 18,
                      borderRadius: "50%",
                      background: "#22c55e",
                      border: "3px solid #fff",
                      // Counter-scale so the marker stays a constant
                      // visual size at any zoom level.
                      transform: `translate(-50%, -50%) scale(${1 / zoom})`,
                      pointerEvents: "none",
                      boxShadow: "0 0 8px rgba(0,0,0,0.7)",
                    }}
                  />
                )}
              </div>

              {/* Zoom controls anchored to the editor (outside the
                  scaled scene so the controls stay at constant size). */}
              <div
                onPointerDown={(e) => e.stopPropagation()}
                onClick={(e) => e.stopPropagation()}
                style={{
                  position: "absolute", right: 8, top: 8,
                  display: "flex", gap: 6,
                  background: "rgba(0,0,0,0.45)", padding: "4px 6px",
                  borderRadius: 6, backdropFilter: "blur(4px)",
                }}
              >
                <button
                  type="button"
                  style={zoomBtn}
                  disabled={zoom <= 1.05}
                  onClick={() => setZoom((z) => Math.max(1, z / 1.4))}
                  title="Zoom out"
                >−</button>
                <span style={{ color: "#fff", fontSize: 12, padding: "0 6px", alignSelf: "center" }}>
                  {zoom.toFixed(1)}×
                </span>
                <button
                  type="button"
                  style={zoomBtn}
                  disabled={zoom >= 15.9}
                  onClick={() => setZoom((z) => Math.min(16, z * 1.4))}
                  title="Zoom in"
                >+</button>
                <button
                  type="button"
                  style={{ ...zoomBtn, width: 44 }}
                  onClick={() => { setZoom(autoZoom); setFocusOverride(null); }}
                  title="Auto zoom to ball detection area"
                >Auto</button>
                <button
                  type="button"
                  style={{ ...zoomBtn, width: 36 }}
                  onClick={() => { setZoom(1); setFocusOverride(null); }}
                  title="Fit full frame"
                >Fit</button>
                {editorBg?.overlayUrl && (
                  <button
                    type="button"
                    style={{
                      ...zoomBtn, width: 44,
                      background: detectorView
                        ? "rgba(220,60,60,0.85)"
                        : zoomBtn.background,
                    }}
                    onClick={() => setDetectorView((v) => !v)}
                    title="Toggle the detector view — the full frame with the MOG2 motion mask in red, candidates in yellow, chosen point ringed green. Zoom/pan/click work the same in both views."
                  >
                    {detectorView ? "🔥 on" : "🔥 off"}
                  </button>
                )}
                {selectedFrameCands.length > 0 && (
                  <button
                    type="button"
                    style={{
                      ...zoomBtn, width: 48,
                      background: showCandidates
                        ? "rgba(245,158,11,0.85)"
                        : zoomBtn.background,
                    }}
                    onClick={() => setShowCandidates((v) => !v)}
                    title={`${selectedFrameCands.length} MOG2 motion candidate(s) detected on this frame — the amber dots are clickable: click one to mark the ball exactly there. This button toggles the dots.`}
                  >
                    ◎ {selectedFrameCands.length}
                  </button>
                )}
                {/* Pan controls — only useful when zoomed past 1×.
                    Each press shifts the viewable area by ~30% of
                    the visible region in that direction. */}
                <div style={{
                  display: "grid",
                  gridTemplateColumns: "repeat(3, 22px)",
                  gridTemplateRows: "22px 22px",
                  gap: 2, marginLeft: 4,
                }}>
                  <span />
                  <button
                    type="button"
                    style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
                    disabled={zoom <= 1.05 || focusPct.y <= 0.1}
                    onClick={() => panBy(0, -1)}
                    title="Pan up"
                  >↑</button>
                  <span />
                  <button
                    type="button"
                    style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
                    disabled={zoom <= 1.05 || focusPct.x <= 0.1}
                    onClick={() => panBy(-1, 0)}
                    title="Pan left"
                  >←</button>
                  <button
                    type="button"
                    style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
                    disabled={zoom <= 1.05 || focusPct.y >= 99.9}
                    onClick={() => panBy(0, 1)}
                    title="Pan down"
                  >↓</button>
                  <button
                    type="button"
                    style={{ ...zoomBtn, width: 22, height: 22, fontSize: 12 }}
                    disabled={zoom <= 1.05 || focusPct.x >= 99.9}
                    onClick={() => panBy(1, 0)}
                    title="Pan right"
                  >→</button>
                </div>
              </div>
            </div>
          ) : tracer?.url ? (
            <video
              src={tracer.url}
              controls
              style={{
                height: "100%", maxHeight: "100%", maxWidth: "100%",
                aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
                background: "#000", borderRadius: 6,
              }}
            />
          ) : (
            <div
              className="muted small"
              style={{
                width: "100%", aspectRatio: "16 / 9",
                background: "var(--border, #222)", borderRadius: 6,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              No tracer rendered yet
            </div>
          )}
        </div>
        <div className="tiny muted" style={{ marginTop: 6 }}>
          {plotAll
            ? "Every dot is a timed motion detection labelled with its frame number. Click a dot to queue the ball at exactly that spot for that frame — click again to un-queue, or click a different dot on the same frame to replace the pick. Zoom in over the arc to separate dense stretches, then Re-generate tracer to apply."
            : selectedFrame != null
              ? `Click on the ball to queue this frame as a tracer point.${
                  selectedFrameCands.length
                    ? ` Amber ◎ dots are this frame's MOG2 motion candidates — click one to snap the mark exactly onto the detection.`
                    : ""
                } Navigate to other frames and add more — including past the AI's 12-frame stop, all the way to the green. Re-generate tracer re-renders here (no AI calls).`
              : "Click a frame card on the right to correct the AI's ball position."}
        </div>
      </div>

      <div
        style={{
          display: "flex", flexDirection: "column", gap: 10,
          overflowY: "auto", minHeight: 0, paddingRight: 4,
        }}
      >
        {error && <div className="err-text small">{error}</div>}

        {selectedFrame != null && (
          <div
            className="card"
            style={{ margin: 0, padding: 10, background: "rgba(34,197,94,0.08)" }}
          >
            <div className="tiny upper muted" style={{ marginBottom: 4 }}>
              Frame {selectedFrame} editor
            </div>
            <div className="small" style={{ marginBottom: 6 }}>
              Ball:{" "}
              {editorBall
                ? <b>{editorBall.x}, {editorBall.y} px</b>
                : <span className="muted">no position</span>}
              {manualPositions[selectedFrame] && (
                <span className="small" style={{ marginLeft: 6, color: "var(--emerald-700)" }}>
                  (queued)
                </span>
              )}
              {clearedFrames.has(selectedFrame) && (
                <span className="small" style={{ marginLeft: 6, color: "var(--danger)" }}>
                  (cleared)
                </span>
              )}
            </div>
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              <button
                type="button"
                className="ghost"
                style={{ width: "auto" }}
                onClick={clearEditorBall}
                disabled={
                  !manualPositions[selectedFrame]
                  && !(tracer?.frames || []).some(
                    (f) => f.frame === selectedFrame && f.found && f.x != null,
                  )
                }
                title="Remove the ball mark on this frame (manual or AI). The renderer will drop this frame from the tracer track."
              >
                Clear frame
              </button>
              <button
                type="button"
                className="ghost"
                style={{ width: "auto", marginLeft: "auto" }}
                onClick={() => { setSelectedFrame(null); setEditorBg(null); }}
              >
                Close
              </button>
            </div>
            <div className="row" style={{ gap: 6, marginTop: 8, flexWrap: "wrap" }}>
              <button type="button" className="ghost" style={{ width: "auto" }}
                onClick={() => loadEditorFrame(Math.max(0, selectedFrame - 5))}>−5</button>
              <button type="button" className="ghost" style={{ width: "auto" }}
                onClick={() => loadEditorFrame(Math.max(0, selectedFrame - 1))}>−1</button>
              <button type="button" className="ghost" style={{ width: "auto" }}
                onClick={() => loadEditorFrame(Math.min(maxFrame ?? selectedFrame + 1, selectedFrame + 1))}>+1</button>
              <button type="button" className="ghost" style={{ width: "auto" }}
                onClick={() => loadEditorFrame(Math.min(maxFrame ?? selectedFrame + 5, selectedFrame + 5))}>+5</button>
            </div>
          </div>
        )}

        <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
          <button
            type="button"
            className="ghost"
            style={{ width: "auto" }}
            onClick={() => addFrame(1)}
            title="Add the frame right after the last tracked one"
          >
            + 1 frame
          </button>
          <button
            type="button"
            className="ghost"
            style={{ width: "auto" }}
            onClick={() => addFrame(5)}
          >
            + 5 frames
          </button>
          <button
            type="button"
            style={{ width: "auto", marginLeft: "auto" }}
            disabled={
              rendering
              || (Object.keys(manualPositions).length === 0 && clearedFrames.size === 0)
            }
            onClick={regenerate}
            title="Re-render the tracer with the queued edits. cv2 only — no AI calls."
          >
            {rendering
              ? "Re-rendering…"
              : `Re-generate tracer${Object.keys(manualPositions).length + clearedFrames.size
                ? ` (${Object.keys(manualPositions).length + clearedFrames.size})`
                : ""}`}
          </button>
        </div>

        <div
          className="row"
          style={{ alignItems: "center", gap: 8, marginTop: 4 }}
        >
          <div className="tiny upper muted">
            Per-frame ball-track ({displayFrames.length}
            {restEntry ? " · incl. rest" : ""})
          </div>
          {tracer?.debugUrl && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto", padding: "1px 8px" }}
              onClick={() =>
                setImgView({
                  url: tracer.debugUrl,
                  title:
                    "All detections (whole clip) — green = rising-arc chain, " +
                    "yellow = other motion candidates",
                })
              }
              title="One image with every motion detection from the clip — the ball's arc reads as a chain of green dots turning yellow past the apex. Zoom/pan inside."
            >
              🗺 All-detections map
            </button>
          )}
          {tracer?.rawMotionUrl && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto", padding: "1px 8px" }}
              onClick={() =>
                setImgView({
                  url: tracer.rawMotionUrl,
                  title:
                    "Raw motion heat — TOTAL unfiltered motion over the " +
                    "window (body, club, clouds, water, ball). Blue = moved " +
                    "rarely, red = moved constantly.",
                })
              }
              title="Accumulated per-pixel motion from the background subtractor (MOG2/KNN), before any ball filtering — shows everything that moved, including body and swing."
            >
              🌡 Raw motion
            </button>
          )}
          {tracer?.rawMotionArcUrl && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto", padding: "1px 8px" }}
              onClick={() =>
                setImgView({
                  url: tracer.rawMotionArcUrl,
                  title:
                    "Detected ball path over raw motion — red curve = the " +
                    "fit the tracer rendered, white dots = tracked points. " +
                    "Compare against the heatmap's blue arc.",
                })
              }
              title="The raw-motion heatmap with the detected ball path drawn on top — eyeball the detection against the visible arc."
            >
              🎯 Path on heat
            </button>
          )}
          {tracer?.rawMotionFramesUrl && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto", padding: "1px 8px" }}
              onClick={() =>
                setImgView({
                  url: tracer.rawMotionFramesUrl,
                  title:
                    "Frames on heat — every timed motion dot labelled with " +
                    "its SOURCE frame number. Read a dot's frame, open that " +
                    "card, and plot the ball there.",
                })
              }
              title="The raw-motion heatmap with each transient dot labelled by the frame it fired in — tells you exactly which frame a descent dot belongs to."
            >
              🔢 Frames on heat
            </button>
          )}
          {(tracer?.timedPoints || []).length > 0 &&
            (tracer?.rawMotionUrl
              || tracer?.rawMotionFramesUrl
              || tracer?.mog2OverlayUrl) && (
            <button
              type="button"
              className={plotAll ? "small" : "ghost small"}
              style={{ width: "auto", padding: "1px 8px" }}
              onClick={() => (plotAll ? setPlotAll(false) : openPlotAll())}
              title="Interactive frames-on-heat: every timed motion dot drawn over the heat, labelled with its frame — click a dot to queue the ball there for that frame. One click per frame, click again to un-queue, Re-generate tracer applies."
            >
              🖱 Click-to-plot
            </button>
          )}
          {tracer?.mog2OverlayUrl && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto", padding: "1px 8px" }}
              onClick={() =>
                setImgView({
                  url: tracer.mog2OverlayUrl,
                  title:
                    "MOG2 vs AI (from produce) — yellow = AI picks, " +
                    "white = MOG2 chain, red = MOG2 points added to " +
                    "the arc.",
                })
              }
              title="Produce's MOG2 layer-in evidence: the raw motion heat with the AI tracer's picks (yellow), the MOG2 chain (white rings) and the points MOG2 added to the arc (red)."
            >
              🔥 MOG2 vs AI
            </button>
          )}
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
            gap: 6,
          }}
        >
          {displayFrames.map((f) => {
            const ball = mergedBallFor(f);
            const isQueued = !!manualPositions[f.frame];
            return (
              <button
                key={f.frame}
                type="button"
                onClick={() => loadEditorFrame(f.frame)}
                style={{
                  width: "100%", padding: 0,
                  border: selectedFrame === f.frame
                    ? "2px solid #22c55e"
                    : "1px solid var(--border)",
                  borderRadius: 4, overflow: "hidden",
                  background: "transparent", cursor: "pointer",
                  textAlign: "left",
                }}
                title={`Frame ${f.frame}${ball ? ` · ${ball.x}, ${ball.y}` : " · no ball"}`}
              >
                <div style={{ position: "relative", aspectRatio: "16 / 9", background: "#222" }}>
                  {f.image_url && (
                    <img
                      src={f.image_url}
                      alt={`Frame ${f.frame}`}
                      style={{ width: "100%", height: "100%", objectFit: "cover" }}
                    />
                  )}
                  {f.image_url && (
                    <span
                      role="button"
                      tabIndex={0}
                      onClick={(e) => {
                        e.stopPropagation();
                        setImgView({
                          url: f.image_url,
                          title: `Frame ${f.frame} — detector view (red=motion · yellow=candidates · green=chosen)`,
                        });
                      }}
                      onKeyDown={(e) => {
                        if (e.key === "Enter" || e.key === " ") {
                          e.preventDefault();
                          e.stopPropagation();
                          setImgView({
                            url: f.image_url,
                            title: `Frame ${f.frame} — detector view`,
                          });
                        }
                      }}
                      title="Enlarge detector view (zoom / pan)"
                      style={{
                        position: "absolute", top: 2, right: 2,
                        padding: "0 5px", fontSize: 13, lineHeight: "18px",
                        background: "rgba(0,0,0,0.55)", color: "#fff",
                        borderRadius: 4, cursor: "zoom-in",
                      }}
                    >
                      🔍
                    </span>
                  )}
                  {hasDims && ball && !(f.zoomed && f.image_url) && (
                    <div
                      style={{
                        position: "absolute",
                        left: `${(ball.x / frameW) * 100}%`,
                        top: `${(ball.y / frameH) * 100}%`,
                        width: 10, height: 10, borderRadius: "50%",
                        background: ball.manual || isQueued ? "#fbbf24" : "#22c55e",
                        border: "2px solid #fff",
                        transform: "translate(-50%, -50%)",
                      }}
                    />
                  )}
                </div>
                <div className="tiny" style={{ padding: "3px 4px" }}>
                  <b>f{f.frame}</b>
                  <span className="muted">
                    {" "}{f.rest ? "rest" : (f.found ? "found" : "no ball")}
                    {isQueued ? " · queued" : ""}
                  </span>
                </div>
              </button>
            );
          })}
          {!frames.length && (
            <div className="muted small">
              No ball-track yet. Click a frame to add points, then Next on Step 3.
            </div>
          )}
        </div>
      </div>
      <ImageLightbox
        url={imgView?.url}
        title={imgView?.title}
        onClose={() => setImgView(null)}
      />
    </div>
  );
}

function FinalizeStep({
  row, finalUrl, finalizing, committing, error, frameW, frameH,
  pendingEdits, graphics, setGraphics, graphicsDirty,
  alreadyProduced, onProduce,
}) {
  const hasDims = !!(frameW && frameH);
  const hasPending = pendingEdits > 0;
  const dirtyForRender = hasPending || graphicsDirty;
  const busy = !!(finalizing || committing);
  const produceLabel = busy
    ? (finalizing ? "Producing…" : "Committing…")
    : (alreadyProduced
      ? (dirtyForRender ? "Re-Produce" : "Re-Produce")
      : "Produce");

  function pickCourseYardage(hole) {
    const map = row?.course_hole_yardages || {};
    const v = map[String(hole)];
    if (v == null) return null;
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return (
    <div
      style={{
        display: "grid",
        gridTemplateColumns: "minmax(320px, 1.6fr) minmax(260px, 1fr)",
        gap: 16,
        height: "100%",
        minHeight: 0,
      }}
    >
      <div style={{ display: "flex", flexDirection: "column", minHeight: 0 }}>
        <div className="tiny upper muted" style={{ marginBottom: 4 }}>
          Final video — tracer + on-screen graphics
        </div>
        <div
          style={{
            flex: 1, minHeight: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
            position: "relative",
          }}
        >
          {finalizing ? (
            <div className="row" style={{ alignItems: "center", gap: 12 }}>
              <div className="shimmer" style={{ width: 18, height: 18, borderRadius: "50%" }} />
              <span className="small">Re-rendering tracer + graphics… 10–30s.</span>
            </div>
          ) : finalUrl ? (
            <video
              src={finalUrl}
              controls
              autoPlay
              style={{
                height: "100%", maxHeight: "100%", maxWidth: "100%",
                aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
                background: "#000", borderRadius: 6,
              }}
            />
          ) : (
            <div
              className="muted small"
              style={{
                width: "100%", aspectRatio: "16 / 9",
                background: "var(--border, #222)", borderRadius: 6,
                display: "flex", alignItems: "center", justifyContent: "center",
              }}
            >
              {error ? error : "Final video not rendered"}
            </div>
          )}
          {dirtyForRender && !finalizing && (
            <div
              role="note"
              style={{
                position: "absolute", left: 12, right: 12, bottom: 12,
                background: "rgba(239, 68, 68, 0.92)",
                color: "#fff",
                padding: "10px 14px",
                borderRadius: 6,
                fontSize: "0.92rem",
                boxShadow: "0 4px 14px rgba(0,0,0,0.4)",
                textAlign: "center",
              }}
            >
              Clicking <b>Produce</b> will re-render
              {hasPending && (
                <> with the updated tracer path ({pendingEdits}{" "}
                {pendingEdits === 1 ? "new point" : "new points"})</>
              )}
              {hasPending && graphicsDirty ? " and " : ""}
              {graphicsDirty && (
                <> with your updated graphics</>
              )}
              .
            </div>
          )}
        </div>
        <div className="tiny muted" style={{ marginTop: 6 }}>
          The player banner, course / hole / par / yardage are baked
          in. Click <b>Produce</b> on the right to (re-)render and
          send to Produced Clips. <b>Finish</b> closes the wizard.
        </div>
      </div>

      <div
        style={{
          display: "flex", flexDirection: "column", gap: 10,
          overflowY: "auto", minHeight: 0, paddingRight: 4,
        }}
      >
        {error && <div className="err-text small">{error}</div>}

        <div
          className="card"
          style={{ margin: 0, padding: 10, background: "rgba(34,197,94,0.06)" }}
        >
          <div className="tiny upper muted" style={{ marginBottom: 4 }}>
            Ready for Produced Clips
          </div>
          <div className="small">
            Upload #{row.id}
            <br />
            {row.course_name || `Course ${row.course_id}`}
          </div>
        </div>

        <div
          className="card"
          style={{
            margin: 0, padding: 10,
            border: graphicsDirty
              ? "1px solid rgba(239,68,68,0.5)"
              : "1px solid var(--border)",
          }}
        >
          <div className="tiny upper muted" style={{ marginBottom: 8 }}>
            On-screen graphics
          </div>
          <div className="field" style={{ marginBottom: 8 }}>
            <label className="small muted">Player</label>
            <input
              type="text"
              value={graphics.player_name || ""}
              onChange={(e) => setGraphics((g) => ({ ...g, player_name: e.target.value }))}
              placeholder="Player name"
              disabled={finalizing}
            />
          </div>
          <div className="row" style={{ gap: 8 }}>
            <div className="field" style={{ flex: 1 }}>
              <label className="small muted">Hole</label>
              <input
                type="number"
                min={1} max={18}
                value={graphics.hole_number ?? ""}
                onChange={(e) => {
                  const hole = Math.max(1, Math.min(18, parseInt(e.target.value, 10) || 1));
                  setGraphics((g) => {
                    // Auto-populate yardage from the course's
                    // hole_yardages map for the new hole, unless the
                    // user has already typed something. We treat the
                    // current yardage as "user-overridden" only when
                    // it differs from the previous hole's course value.
                    const prevCourse = pickCourseYardage(g.hole_number);
                    const newCourse = pickCourseYardage(hole);
                    const userEdited = prevCourse != null
                      && Number(g.yardage) !== prevCourse;
                    return {
                      ...g,
                      hole_number: hole,
                      yardage: userEdited
                        ? g.yardage
                        : (newCourse != null ? newCourse : g.yardage),
                    };
                  });
                }}
                disabled={finalizing}
              />
            </div>
            <div className="field" style={{ flex: 1 }}>
              <label className="small muted">Yards</label>
              <input
                type="number"
                min={0}
                value={graphics.yardage ?? ""}
                onChange={(e) => setGraphics((g) => ({
                  ...g,
                  yardage: e.target.value === "" ? "" : Math.max(0, parseInt(e.target.value, 10) || 0),
                }))}
                disabled={finalizing}
              />
            </div>
          </div>
          {graphicsDirty && (
            <div className="tiny" style={{ marginTop: 6, color: "#ef4444" }}>
              Graphics changed since last render. Produce will re-apply.
            </div>
          )}
        </div>

        <button
          type="button"
          onClick={onProduce}
          disabled={busy}
          style={{ width: "100%" }}
          title={alreadyProduced
            ? "Re-render the tracer (cv2 only, no AI) with current ball points + graphics and update Produced Clips"
            : "Render the final clip with current graphics and send to Produced Clips"}
        >
          {produceLabel}
        </button>

        {hasPending && (
          <div className="small" style={{ color: "#ef4444" }}>
            {pendingEdits} unsaved tracer point{pendingEdits === 1 ? "" : "s"} from Step 2.
            Clicking <b>Produce</b> will fold them into the tracer and
            re-apply graphics — no AI calls.
          </div>
        )}
      </div>
    </div>
  );
}

function eventStatusBadge(status) {
  // Inline-pill styling that mirrors the long-upload card's status
  // chip (Queued / Production in Progress / Produced) so the two
  // sections look like one family.
  switch (status) {
    case "processed":
      return {
        label: "Processed",
        bg: "rgba(40, 168, 92, 0.15)",
        border: "rgba(40, 168, 92, 0.5)",
      };
    case "failed":
      return {
        label: "Failed",
        bg: "rgba(220, 53, 69, 0.15)",
        border: "rgba(220, 53, 69, 0.5)",
      };
    case "paired_uploaded":
      return {
        label: "Ready",
        bg: "rgba(120, 120, 120, 0.15)",
        border: "rgba(120, 120, 120, 0.5)",
      };
    case "tee_uploaded":
      return {
        label: "Tee uploaded",
        bg: "rgba(255, 176, 0, 0.15)",
        border: "rgba(255, 176, 0, 0.5)",
      };
    case "triggered":
      return {
        label: "Triggered",
        bg: "rgba(255, 176, 0, 0.15)",
        border: "rgba(255, 176, 0, 0.5)",
      };
    default:
      return {
        label: status || "—",
        bg: "rgba(120, 120, 120, 0.15)",
        border: "rgba(120, 120, 120, 0.5)",
      };
  }
}

function CameraEventCard({
  ev, busy, onOpenViewer, onReproduce, onBroadcast, onDelete,
}) {
  // One row in the camera-event production list. Layout matches the
  // long-upload card exactly — tee tile / green tile / produced tile
  // in a flex row on the left, status pill + action buttons stacked
  // vertically on the right — so the operator sees a consistent
  // language across both kinds of queue items.
  // Show "processing" the instant the operator clicks, not when the POST
  // returns. ev.status is SERVER state, so it cannot change until the
  // request completes and the list refetches -- about three seconds, during
  // which the row looked untouched and the button looked unpressed.
  const badge = busy
    ? {
        label: "Production in Progress",
        bg: "rgba(214, 158, 46, 0.15)",
        border: "rgba(214, 158, 46, 0.5)",
      }
    : eventStatusBadge(ev.status);
  const triggeredAt = ev.triggered_at;
  // WHEN EACH CAMERA'S FIRST FRAME IS. The Pi reports it per camera and
  // the two differ by a fraction of a second, which is exactly what the
  // tee->green delta is made of -- so prefer each camera's own stamp and
  // fall back to the shared trigger time, which the clock overlay then
  // marks approximate rather than passing off as measured.
  const teeStartsAt = ev.tee_recording_started_at || triggeredAt;
  const greenStartsAt = ev.green_recording_started_at || triggeredAt;
  const producedClips = ev.produced_clip ? [ev.produced_clip] : [];
  const hasProduced = !!ev.produced_clip;
  const onBroadcastChannel = !!ev.produced_clip?.is_highlight;
  return (
    <div
      className="card"
      style={{
        marginBottom: 12,
        opacity: busy ? 0.6 : 1,
        position: "relative",
      }}
    >
      <div
        className="row"
        style={{
          gap: 10, flexWrap: "wrap", alignItems: "baseline", marginBottom: 10,
        }}
      >
        <h4 style={{ margin: 0 }}>
          Event #{ev.id} ·{" "}
          {ev.course_name || `course ${ev.course_id}`} · hole {ev.hole_number}
        </h4>
        <span className="small muted">
          {ev.dual_camera ? "Tee + Green" : "Tee only"}
        </span>
        <div style={{ flex: 1 }} />
        <span className="small muted">
          Captured {fmtDateTime(triggeredAt)}
        </span>
      </div>

      {ev.last_error && (
        <div className="err-text small" style={{ marginBottom: 10 }}>
          {ev.last_error}
        </div>
      )}

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
            label={`Tee · ${ev.tee_camera_name || `cam #${ev.tee_camera_id}`}`}
            thumb={ev.tee_thumbnail_url}
            durationSec={ev.tee_duration_sec}
            nbFrames={ev.tee_nb_frames}
            fps={ev.tee_fps}
            sizeMb={ev.tee_size_mb}
            startsAt={teeStartsAt}
            recordingStartedAt={ev.tee_recording_started_at}
            missing={ev.tee_missing}
            notUploaded={!ev.tee_clip_filename}
            qualityLabel={null}
            width={ev.tee_width}
            height={ev.tee_height}
            videoUrl={ev.tee_url}
            onOpenViewer={onOpenViewer}
          />
          <VideoTile
            label={
              ev.dual_camera
                ? `Green · ${ev.green_camera_name || `cam #${ev.green_camera_id}`}`
                : "Green · n/a"
            }
            thumb={ev.green_thumbnail_url}
            durationSec={ev.green_duration_sec}
            nbFrames={ev.green_nb_frames}
            fps={ev.green_fps}
            sizeMb={ev.green_size_mb}
            startsAt={greenStartsAt}
            recordingStartedAt={ev.green_recording_started_at}
            missing={ev.green_missing}
            notUploaded={!ev.green_clip_filename}
            qualityLabel={null}
            width={ev.green_width}
            height={ev.green_height}
            videoUrl={ev.green_url}
            onOpenViewer={onOpenViewer}
          />
          <ProducedTile clips={producedClips} onOpenViewer={onOpenViewer} />
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
          <span
            className="small"
            style={{
              padding: "4px 10px",
              borderRadius: 999,
              background: badge.bg,
              border: `1px solid ${badge.border}`,
              textAlign: "center",
            }}
          >
            {badge.label}
          </span>
          <button
            className="small"
            onClick={() => onReproduce(ev)}
            disabled={busy || !ev.tee_url}
            title={ev.tee_url
              ? "Re-run the production pipeline on the existing raw clips"
              : "No raw tee clip on disk — can't re-process"}
          >
            {busy ? "…" : "Re-Produce"}
          </button>
          <button
            className={onBroadcastChannel ? "small" : "small ghost"}
            onClick={() => onBroadcast(ev)}
            disabled={busy || !hasProduced}
            title={hasProduced
              ? (onBroadcastChannel
                ? "Remove this clip from the Broadcast channel"
                : "Send this clip to the Broadcast channel")
              : "Produce this event first to enable Broadcast"}
          >
            {onBroadcastChannel ? "On Broadcast" : "Broadcast"}
          </button>
          <button
            className="small ghost err-text"
            onClick={() => onDelete(ev)}
            disabled={busy}
            title="Permanently remove this event, its raw clips, and any produced clip"
          >
            Delete
          </button>
        </div>
      </div>
    </div>
  );
}

function fmtClock(ms) {
  // HH:MM:SS.mmm in US Central time (auto CST/CDT). Both the tee and
  // green overlays use the same zone, so at the same real instant they
  // read identically — that's what lets the operator eyeball sync.
  const d = new Date(ms);
  const parts = new Intl.DateTimeFormat("en-US", {
    timeZone: "America/Chicago",
    hourCycle: "h23",
    hour: "2-digit", minute: "2-digit", second: "2-digit",
  }).formatToParts(d);
  const get = (t) => parts.find((p) => p.type === t)?.value || "00";
  const millis = String(d.getMilliseconds()).padStart(3, "0");
  return `${get("hour")}:${get("minute")}:${get("second")}.${millis}`;
}

function VideoLightbox({ url, title, startedAt, startedApprox, fps, onClose }) {
  const videoRef = useRef(null);
  const [curTime, setCurTime] = useState(0);
  const startMs = startedAt ? parseApiDate(startedAt)?.getTime() ?? null : null;
  const hasFps = !!(fps && fps > 0);
  const showOverlay = startMs != null || hasFps;

  // Track the video's currentTime via rAF so the clock + frame
  // readouts stay smooth during playback and update on seek/scrub.
  useEffect(() => {
    if (!url || !showOverlay) return undefined;
    let raf = 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) setCurTime(v.currentTime);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [url, showOverlay]);

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
        <div style={{ position: "relative" }}>
          <video
            ref={videoRef}
            src={url}
            controls
            autoPlay
            style={{
              width: "100%", maxHeight: "80vh",
              background: "#000", borderRadius: 6, display: "block",
            }}
          />
          {showOverlay && (
            <div
              style={{
                position: "absolute", top: 8, left: 8,
                background: "rgba(0,0,0,0.7)", color: "#3ee37a",
                fontFamily: "monospace", fontSize: "1rem",
                padding: "4px 8px", borderRadius: 4,
                pointerEvents: "none", letterSpacing: "0.5px",
                display: "flex", flexDirection: "column", gap: 2,
              }}
            >
              {startMs != null && (
                <span
                  title={startedApprox
                    ? "Reckoned from the clip's start time (the trigger), "
                      + "not the camera's own first-frame stamp — right to "
                      + "a fraction of a second, but do not read a "
                      + "tee-to-green offset off it"
                    : "The camera's own reported first-frame time plus the "
                      + "position in the clip. The tee and green overlays "
                      + "are in the same zone, so at the same real instant "
                      + "they read identically"}
                >
                  {startedApprox ? "≈" : ""}
                  {fmtClock(startMs + curTime * 1000)} CT
                </span>
              )}
              {hasFps && <span>Frame {Math.floor(curTime * fps)}</span>}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

// Plots the classical detector's per-frame motion signal (mean pixel
// change over time). Each swing is a burst that rises above the red
// threshold line; green dashed lines mark where a swing was detected.
function MotionChart({ motion, ballPeaks, posePeaks, aiBallPeaks }) {
  if (!motion || !Array.isArray(motion.series) || motion.series.length < 2) {
    return (
      <div className="small muted" style={{ marginBottom: 12 }}>
        Motion signal unavailable (detector found no usable frames).
      </div>
    );
  }
  const series = motion.series;
  const n = series.length;
  const dur = motion.duration_sec || n - 1;
  const W = 1000;
  const H = 200;
  const thr = motion.threshold || 0;
  const maxV = Math.max(thr, ...series) * 1.05 || 1;
  const xOf = (i) => (i / (n - 1)) * W;
  const yOf = (v) => H - (v / maxV) * (H - 4);
  const pts = series.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const thrY = yOf(thr);
  const peaks = motion.swing_peaks || [];
  const balls = ballPeaks || [];
  const poses = posePeaks || [];
  const aiBalls = aiBallPeaks || [];
  return (
    <div style={{ marginBottom: 16 }}>
      <div className="small muted" style={{ marginBottom: 4 }}>
        Motion signal — mean pixel change per frame
        {motion.hz ? ` (~${Math.round(motion.hz)} Hz)` : ""}. Each swing is a
        burst above the <span style={{ color: "#e74c3c" }}>red threshold</span>.{" "}
        <span style={{ color: "#1a9d55" }}>Green</span> = motion,{" "}
        <span style={{ color: "#e67e22" }}>orange</span> = ball,{" "}
        <span style={{ color: "#9b59b6" }}>purple</span> = pose,{" "}
        <span style={{ color: "#00b8d4" }}>cyan</span> = AI ball.{" "}
        median={motion.median != null ? motion.median.toFixed(3) : "?"} ·
        threshold={thr.toFixed(3)} · motion={peaks.length} · ball={balls.length} ·
        pose={poses.length} · ai={aiBalls.length}
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{
          width: "100%", height: 200, borderRadius: 6,
          background: "rgba(127,127,127,0.10)",
        }}
      >
        {peaks.map((t, i) => (
          <line
            key={`p${i}`}
            x1={(t / dur) * W}
            x2={(t / dur) * W}
            y1={0}
            y2={H}
            stroke="#1a9d55"
            strokeWidth={2}
            strokeDasharray="5 4"
            opacity={0.75}
          />
        ))}
        {balls.map((t, i) => (
          <line
            key={`b${i}`}
            x1={(t / dur) * W}
            x2={(t / dur) * W}
            y1={0}
            y2={H}
            stroke="#e67e22"
            strokeWidth={2}
            opacity={0.85}
          />
        ))}
        {poses.map((t, i) => (
          <line
            key={`po${i}`}
            x1={(t / dur) * W}
            x2={(t / dur) * W}
            y1={0}
            y2={H}
            stroke="#9b59b6"
            strokeWidth={2}
            strokeDasharray="2 3"
            opacity={0.85}
          />
        ))}
        {aiBalls.map((t, i) => (
          <line
            key={`ai${i}`}
            x1={(t / dur) * W}
            x2={(t / dur) * W}
            y1={0}
            y2={H}
            stroke="#00b8d4"
            strokeWidth={2}
            strokeDasharray="6 2"
            opacity={0.9}
          />
        ))}
        <line
          x1={0}
          x2={W}
          y1={thrY}
          y2={thrY}
          stroke="#e74c3c"
          strokeWidth={1.5}
          strokeDasharray="7 5"
        />
        <polyline points={pts} fill="none" stroke="#3b82f6" strokeWidth={1.5} />
      </svg>
      <div
        className="small muted"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>0s</span>
        <span>{dur.toFixed(0)}s</span>
      </div>
    </div>
  );
}

// Draw the tee-box ROI on a reference frame. The ball detector only looks
// inside the box; drawn once per course (fixed camera). Fractions of the
// displayed image map directly to fractions of the frame.
function TeeBoxRoi({ refUrl, initialRoi, courseId, adminPassword, onSaved }) {
  const boxRef = useRef(null);
  const [rect, setRect] = useState(initialRoi || null);
  const [drag, setDrag] = useState(null);
  const [busy, setBusy] = useState(false);
  if (!refUrl) return null;

  const toFrac = (e) => {
    const r = boxRef.current.getBoundingClientRect();
    return {
      x: Math.min(1, Math.max(0, (e.clientX - r.left) / r.width)),
      y: Math.min(1, Math.max(0, (e.clientY - r.top) / r.height)),
    };
  };
  const onDown = (e) => {
    e.preventDefault();
    const p = toFrac(e);
    setDrag(p);
    setRect({ x: p.x, y: p.y, w: 0, h: 0 });
  };
  const onMove = (e) => {
    if (!drag) return;
    const p = toFrac(e);
    setRect({
      x: Math.min(drag.x, p.x),
      y: Math.min(drag.y, p.y),
      w: Math.abs(p.x - drag.x),
      h: Math.abs(p.y - drag.y),
    });
  };
  const onUp = () => setDrag(null);

  async function save() {
    if (!rect || rect.w < 0.01 || rect.h < 0.01 || !courseId) return;
    setBusy(true);
    try {
      await api.setBallRoi(adminPassword, courseId, rect);
      if (onSaved) await onSaved();
    } finally {
      setBusy(false);
    }
  }
  async function clearRoi() {
    if (!courseId) return;
    setBusy(true);
    try {
      await api.setBallRoi(adminPassword, courseId, null);
      setRect(null);
      if (onSaved) await onSaved();
    } finally {
      setBusy(false);
    }
  }

  return (
    <div style={{ marginBottom: 16 }}>
      <div className="small muted" style={{ marginBottom: 4 }}>
        <strong>Tee-box ROI</strong> — drag a box around where the ball rests.
        The ball detector only looks inside it (kills shoes/glints elsewhere).
        Saved for this course; drawn once.
      </div>
      <div
        ref={boxRef}
        onMouseDown={onDown}
        onMouseMove={onMove}
        onMouseUp={onUp}
        onMouseLeave={onUp}
        style={{
          position: "relative", display: "inline-block", cursor: "crosshair",
          userSelect: "none", maxWidth: "100%",
        }}
      >
        <img
          src={refUrl}
          alt="tee frame"
          draggable={false}
          style={{ display: "block", maxWidth: "100%", borderRadius: 6 }}
        />
        {rect && (
          <div
            style={{
              position: "absolute",
              left: `${rect.x * 100}%`,
              top: `${rect.y * 100}%`,
              width: `${rect.w * 100}%`,
              height: `${rect.h * 100}%`,
              border: "2px solid #e67e22",
              background: "rgba(230,126,34,0.15)",
              pointerEvents: "none",
            }}
          />
        )}
      </div>
      <div style={{ display: "flex", gap: 8, marginTop: 6, alignItems: "center" }}>
        <button
          className="small"
          onClick={save}
          disabled={busy || !rect || rect.w < 0.01}
          style={{ width: "auto" }}
        >
          {busy ? "Saving…" : "Save ROI & re-run"}
        </button>
        <button
          className="small ghost"
          onClick={clearRoi}
          disabled={busy}
          style={{ width: "auto" }}
        >
          Clear ROI
        </button>
        {rect && (
          <span className="small muted">
            box {(rect.w * 100).toFixed(0)}%×{(rect.h * 100).toFixed(0)}% @{" "}
            {(rect.x * 100).toFixed(0)},{(rect.y * 100).toFixed(0)}%
          </span>
        )}
      </div>
    </div>
  );
}

// Pose wrist-SPEED waveform. A swing is a burst of fast wrist motion
// (backswing + downswing); each burst above the threshold is a swing.
function PoseChart({ pose }) {
  if (!pose || !Array.isArray(pose.series) || pose.series.length < 2) return null;
  const series = pose.series;
  const n = series.length;
  const dur = pose.duration_sec || n - 1;
  const W = 1000;
  const H = 140;
  const thr = pose.threshold || 0;
  const maxV = Math.max(...series, thr, 0.01);
  const minV = Math.min(...series, 0);
  const range = maxV - minV || 1;
  const xOf = (i) => (i / (n - 1)) * W;
  const yOf = (v) => H - ((v - minV) / range) * (H - 4);
  const pts = series.map((v, i) => `${xOf(i).toFixed(1)},${yOf(v).toFixed(1)}`).join(" ");
  const thrY = yOf(thr);
  const peaks = pose.peaks || [];
  const bursts = Array.isArray(pose.bursts_detail) ? pose.bursts_detail : [];
  // Human-readable outcome per burst, for the markers + list below.
  const BURST_META = {
    swing: { color: "#9b59b6", label: "swing" },
    upright: { color: "#e67e22", label: "upright — no forward tilt" },
    bend_unknown_weak: { color: "#e67e22", label: "posture unclear, too weak" },
    ratio_low: { color: "#e67e22", label: "below 5× — waggle/walk speed" },
    ratio_high: { color: "#e67e22", label: "above 25× — tracking glitch" },
    too_short: { color: "#7f8c8d", label: "burst too short" },
    too_long: { color: "#7f8c8d", label: "burst too long" },
    nms_suppressed: { color: "#7f8c8d", label: "merged with a nearer swing" },
  };
  const metaFor = (s) => BURST_META[s] || { color: "#7f8c8d", label: s || "?" };
  const dropped = bursts.filter((b) => b.status !== "swing");
  return (
    <div style={{ marginTop: 6, marginBottom: 6 }}>
      <div className="small muted" style={{ marginBottom: 4 }}>
        Wrist speed — a swing is a burst above the{" "}
        <span style={{ color: "#e74c3c" }}>red threshold</span>;{" "}
        <span style={{ color: "#9b59b6" }}>purple</span> marks detected swings,{" "}
        <span style={{ color: "#e67e22" }}>orange</span>/{" "}
        <span style={{ color: "#7f8c8d" }}>grey</span> mark bursts that were
        dropped (hover for why).
      </div>
      <svg
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="none"
        style={{
          width: "100%", height: 140, borderRadius: 6,
          background: "rgba(155,89,182,0.10)",
        }}
      >
        {peaks.map((t, i) => (
          <line
            key={`pp${i}`}
            x1={(t / dur) * W}
            x2={(t / dur) * W}
            y1={0}
            y2={H}
            stroke="#9b59b6"
            strokeWidth={2}
            strokeDasharray="4 3"
            opacity={0.8}
          />
        ))}
        <line
          x1={0}
          x2={W}
          y1={thrY}
          y2={thrY}
          stroke="#e74c3c"
          strokeWidth={1.5}
          strokeDasharray="7 5"
        />
        <polyline points={pts} fill="none" stroke="#9b59b6" strokeWidth={1.5} />
        {bursts.map((b, i) => {
          const m = metaFor(b.status);
          const isSwing = b.status === "swing";
          return (
            <g key={`b${i}`}>
              <circle
                cx={(b.t / dur) * W}
                cy={8}
                r={isSwing ? 5 : 4}
                fill={m.color}
                opacity={isSwing ? 1 : 0.85}
              >
                <title>
                  {`${b.t}s · ratio ${b.ratio}× · ${b.dur}s · bend ${
                    b.bend == null ? "n/a" : `${b.bend}°`
                  } → ${m.label}`}
                </title>
              </circle>
            </g>
          );
        })}
      </svg>
      <div
        className="small muted"
        style={{ display: "flex", justifyContent: "space-between" }}
      >
        <span>0s</span>
        <span>{dur.toFixed(0)}s</span>
      </div>
      {dropped.length > 0 && (
        <div className="tiny muted" style={{ marginTop: 6, lineHeight: 1.7 }}>
          <b>Dropped bursts:</b>{" "}
          {dropped.map((b, i) => {
            const m = metaFor(b.status);
            return (
              <span
                key={`d${i}`}
                style={{
                  display: "inline-block",
                  marginRight: 8,
                  padding: "1px 7px",
                  borderRadius: 999,
                  border: `1px solid ${m.color}`,
                  color: m.color,
                }}
                title={`ratio ${b.ratio}× · ${b.dur}s · bend ${
                  b.bend == null ? "n/a" : `${b.bend}°`
                }`}
              >
                {b.t}s · {m.label}
              </span>
            );
          })}
        </div>
      )}
    </div>
  );
}

// Dev-only diagnostic overlay. Shows, per detected swing, what the
// classical-CV tracer found (motion heatmap + ball/candidate counts) next
// to the AI tracer's result on the same swing, so the two can be compared.
/**
 * Static (non-interactive) flight map for the debug report: the raw
 * motion heat as background, the timed dots with frame labels, and a
 * magenta line drawn from the resting ball through the AI-plotted
 * launch points — the direction the flight starts, at a glance.
 */
function FlightMapStatic({
  bgUrl, dots, restBall, aiPoints, trackPoints, frameW, frameH,
  impactFrame, region,
}) {
  // Native dims: prop when the backend knows them, else measured off
  // the heat image itself (it's rendered at native resolution).
  const [imgDims, setImgDims] = useState(null);
  const fw = frameW || imgDims?.w || null;
  const fh = frameH || imgDims?.h || null;
  if (!bgUrl) return null;
  const ds = (dots || []).filter(
    (p) => impactFrame == null || p.frame >= impactFrame,
  );
  const line = [];
  if (restBall?.x != null && restBall?.y != null) {
    line.push([restBall.x, restBall.y]);
  }
  for (const p of aiPoints || []) {
    if (p?.x != null && p?.y != null) line.push([p.x, p.y]);
  }
  // The FULL mapped track (AI + launch tracker + MOG2 chain + arc
  // completion), sorted by frame — the whole-arc line.
  const track = [...(trackPoints || [])].sort((a, b) => a.frame - b.frame);
  const trackLine = [];
  if (restBall?.x != null && restBall?.y != null) {
    trackLine.push([restBall.x, restBall.y]);
  }
  for (const p of track) trackLine.push([p.x, p.y]);
  return (
    <div
      style={{
        position: "relative", marginTop: 3,
        pointerEvents: "none", userSelect: "none",
      }}
    >
      <img
        src={bgUrl}
        alt="flight map"
        style={{ width: "100%", display: "block", borderRadius: 6 }}
        onLoad={(e) => {
          if (!imgDims && e.target.naturalWidth) {
            setImgDims({
              w: e.target.naturalWidth,
              h: e.target.naturalHeight,
            });
          }
        }}
      />
      {fw && fh && ds.map((p, i) => (
        <div
          key={`${p.frame}-${i}`}
          style={{
            position: "absolute",
            left: `${(p.x / fw) * 100}%`,
            top: `${(p.y / fh) * 100}%`,
            transform: "translate(-50%, -50%)",
          }}
        >
          <div
            style={{
              width: 7, height: 7, borderRadius: "50%",
              border: "1px solid #f59e0b",
              background: "rgba(245,158,11,0.35)",
            }}
          />
          <span
            style={{
              position: "absolute", left: 8,
              top: i % 2 === 0 ? -11 : 5,
              fontSize: 8.5, fontWeight: 600, color: "#fde047",
              textShadow: "0 0 3px #000, 0 0 3px #000",
              whiteSpace: "nowrap",
            }}
          >
            {p.frame}
          </span>
        </div>
      ))}
      {fw && fh && (line.length >= 2 || trackLine.length >= 2) && (
        <svg
          viewBox={`0 0 ${fw} ${fh}`}
          preserveAspectRatio="none"
          style={{
            position: "absolute", inset: 0,
            width: "100%", height: "100%",
          }}
        >
          {region && region.length === 4 && (
            <rect
              x={region[0]}
              y={region[1]}
              width={region[2] - region[0]}
              height={region[3] - region[1]}
              fill="none"
              stroke="#ef4444"
              strokeWidth={Math.max(2, fw / 600)}
              strokeDasharray={`${fw / 80} ${fw / 160}`}
              opacity={0.85}
            />
          )}
          {trackLine.length >= 2 && (
            <polyline
              points={trackLine.map(([x, y]) => `${x},${y}`).join(" ")}
              fill="none"
              stroke="#38bdf8"
              strokeWidth={Math.max(2, fw / 520)}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.85}
            />
          )}
          {track.map((p, i) => (
            <circle
              key={`t${i}`}
              cx={p.x}
              cy={p.y}
              r={
                p.source === "arc"
                  ? Math.max(4, fw / 220)
                  : Math.max(2.5, fw / 340)
              }
              fill={p.source === "arc" ? "none" : "#38bdf8"}
              stroke={p.source === "arc" ? "#fb923c" : "#fff"}
              strokeWidth={
                p.source === "arc"
                  ? Math.max(2, fw / 640)
                  : Math.max(0.8, fw / 1600)
              }
            />
          ))}
          {line.length >= 2 && (
            <polyline
              points={line.map(([x, y]) => `${x},${y}`).join(" ")}
              fill="none"
              stroke="#ff00ff"
              strokeWidth={Math.max(2, fw / 480)}
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity={0.9}
            />
          )}
          {line.map(([x, y], i) => (
            <circle
              key={i}
              cx={x}
              cy={y}
              r={Math.max(3, fw / 260)}
              fill={i === 0 ? "#22d3ee" : "#ff00ff"}
              stroke="#fff"
              strokeWidth={Math.max(1, fw / 1200)}
            />
          ))}
        </svg>
      )}
    </div>
  );
}

function ProduceDebugModal({ data, adminPassword, onRerun, onClose }) {
  // Per-swing 🔥 toggle for the launch-tracker film-strip (plain vs
  // motion-mask tint).
  const [launchHeat, setLaunchHeat] = useState({});
  const [earlyHeat, setEarlyHeat] = useState({});
  // Per-swing toggle for the anchor-check strip: photo tiles (what the
  // AI judged) vs the MOG2 frame-diff twin of the same tiles.
  const [anchorHeat, setAnchorHeat] = useState({});
  const swings = data.swings || [];
  const okBadge = (ok) => (
    <span
      style={{
        fontWeight: 700,
        color: ok ? "#1a9d55" : "#c0392b",
      }}
    >
      {ok ? "✅ found ball" : "❌ no ball"}
    </span>
  );
  const stat = (label, val) =>
    val == null || val === "" ? null : (
      <div className="small muted">
        {label}: <span style={{ color: "inherit", fontWeight: 600 }}>{String(val)}</span>
      </div>
    );
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, background: "rgba(0,0,0,0.75)",
        zIndex: 1000, display: "flex", alignItems: "flex-start",
        justifyContent: "center", overflow: "auto", padding: 20,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: "var(--card-bg, #1b1b1f)", color: "inherit",
          borderRadius: 12, maxWidth: 1100, width: "100%", padding: 20,
          boxShadow: "0 10px 40px rgba(0,0,0,0.5)",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 6 }}>
          <h2 style={{ margin: 0, flex: 1 }}>
            🐞 Produce debug — upload #{data.uploadId}
          </h2>
          <button className="small ghost" onClick={onClose} style={{ width: "auto" }}>
            Close
          </button>
        </div>
        <p className="small muted" style={{ marginTop: 0 }}>
          One run, two views: the clip is produced &amp; saved normally, and
          this report renders that SAME run&apos;s recorded work — the exact
          heat images the AI judge saw, the exact ball-departure calls, the
          exact decisions. Below, each surviving swing shows the production
          tracer next to a fresh classical-CV comparison on the same window.
        </p>
        <div className="small" style={{ marginBottom: 12 }}>
          {data.running
            ? `Analyzing… ${data.done}/${data.total || "?"} swings`
            : `Done — ${swings.length} swing(s) analyzed`}
          {" · "}
          {data.ai_available
            ? "AI tracer: on"
            : "AI tracer: OFF (set ANTHROPIC_API_KEY on this deployment)"}
          {data.single_run ? " · 🔗 single run — produce's own record" : ""}
          {data.error ? ` · error: ${data.error}` : ""}
        </div>

        {data.final_verdict?.available && (
          <div
            style={{
              border: "2px solid rgba(26,157,85,0.6)", borderRadius: 8,
              padding: "8px 12px", marginBottom: 14,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>
              ⚖️ Final verdict — {data.final_verdict.n_produced}/
              {data.final_verdict.n_candidates} produced
            </div>
            <div className="small" style={{ marginBottom: 6 }}>
              {data.final_verdict.summary}
            </div>
            {(data.final_verdict.swings || []).map((s) => (
              <div
                key={s.swing}
                className="small"
                style={{
                  marginTop: 4, paddingTop: 4,
                  borderTop: "1px solid rgba(120,120,120,0.25)",
                }}
              >
                <b style={{ color: s.produced ? "#1a9d55" : "#c0392b" }}>
                  swing {s.swing} @ {s.t}s: {s.produced ? "✅ PRODUCE" : "❌ eliminated"}
                </b>
                <span className="muted"> — {s.explanation}</span>
              </div>
            ))}
          </div>
        )}

        {data.ai_ball && (
          <div
            style={{
              border: "1px solid rgba(0,184,212,0.4)", borderRadius: 8,
              padding: "8px 12px", marginBottom: 14,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>
              🤖 Real vs practice (ball departure) —{" "}
              {data.ai_ball.available
                ? `${data.ai_ball.n_real ?? 0}/${data.ai_ball.n_swings ?? 0} real`
                : "unavailable"}
            </div>
            {data.ai_ball.available ? (
              <>
                <div className="small muted">
                  ball at rest before the swing → gone after = a real shot;
                  no ball, or still there after = practice / whiff. Only{" "}
                  <b>real</b> swings get produced.
                </div>
                {(data.ai_ball.swings || []).map((s, i) => {
                  const real = s.verdict === "real";
                  const unknown = s.verdict === "unknown";
                  const skipped = s.verdict === "skipped";
                  const col = real ? "#1a9d55" : (unknown || skipped) ? "#888" : "#c0392b";
                  return (
                    <div
                      key={i}
                      style={{
                        marginTop: 8, paddingTop: 6,
                        borderTop: "1px solid rgba(120,120,120,0.25)",
                      }}
                    >
                      <div className="small" style={{ fontWeight: 700, color: col }}>
                        swing {s.swing}: {real ? "✅ REAL" : skipped ? "⏭ skipped" : unknown ? "? unknown" : "❌ practice"}
                        <span className="small muted" style={{ fontWeight: 400 }}>
                          {" "}· {s.reason}
                        </span>
                      </div>
                      {!s.anchor && (real || unknown) && (
                        <div
                          className="tiny"
                          style={{ marginTop: 2, color: "#b7791f" }}
                        >
                          \u26a0 \ud83e\udd16 AI anchor walk + launch plot SKIPPED \u2014 the
                          before/after check never found the resting ball,
                          so there was no anchor to walk from. The tracer
                          below is MOG2-only for this swing.
                        </div>
                      )}
                      {s.anchor?.early_image_url && (
                        <div style={{ marginTop: 3 }}>
                          <div className="tiny">
                            {"\ud83d\udd0d"} MOG2 across the same launch
                            window —{" "}
                            <span
                              style={{
                                color: s.anchor.early_n
                                  ? "#1a9d55"
                                  : "#b7791f",
                              }}
                            >
                              {s.anchor.early_n ?? 0} point(s)
                            </span>{" "}
                            filling the frames the AI missed. AI picks win
                            per frame; these fill the gaps.
                          </div>
                          {s.anchor.early_image_heat_url && (
                            <button
                              type="button"
                              className={
                                earlyHeat[s.swing] ? "small" : "ghost small"
                              }
                              style={{
                                width: "auto", padding: "1px 8px",
                                marginBottom: 3,
                              }}
                              onClick={() =>
                                setEarlyHeat((m) => ({
                                  ...m,
                                  [s.swing]: !m[s.swing],
                                }))
                              }
                              title="Toggle the motion mask on these tiles — what MOG2 actually saw in the launch window."
                            >
                              🔥 heat {earlyHeat[s.swing] ? "on" : "off"}
                            </button>
                          )}
                          <a
                            href={
                              earlyHeat[s.swing] &&
                              s.anchor.early_image_heat_url
                                ? s.anchor.early_image_heat_url
                                : s.anchor.early_image_url
                            }
                            target="_blank"
                            rel="noreferrer"
                            style={{ display: "block" }}
                          >
                            <img
                              src={
                                earlyHeat[s.swing] &&
                                s.anchor.early_image_heat_url
                                  ? s.anchor.early_image_heat_url
                                  : s.anchor.early_image_url
                              }
                              alt="MOG2 launch-window strip"
                              style={{ maxWidth: "100%", borderRadius: 6 }}
                            />
                          </a>
                        </div>
                      )}
                      {s.anchor?.assumed_impact_image_url && (
                        <div style={{ marginTop: 3 }}>
                          <div className="tiny muted">
                            the frame impact was assumed on — cyan = the
                            pose hands the launch plot was seeded from,
                            magenta = each ball the AI then found
                          </div>
                          <a
                            href={s.anchor.assumed_impact_image_url}
                            target="_blank"
                            rel="noreferrer"
                            style={{ display: "block" }}
                          >
                            <img
                              src={s.anchor.assumed_impact_image_url}
                              alt="assumed impact frame"
                              style={{ maxWidth: "100%", borderRadius: 6 }}
                            />
                          </a>
                        </div>
                      )}
                      {s.anchor && (
                        <div className="tiny" style={{ marginTop: 2 }}>
                          {"\u2693"}{" "}
                          {s.anchor.verified && s.anchor.assumed_impact ? (
                            <span style={{ color: "#b7791f" }}>
                              impact ASSUMED at f{s.anchor.impact_frame} (the
                              pose peak) — no ball departure was found, so
                              this frame is an estimate, not something we
                              watched happen. Everything below ran from it
                              exactly as it would from a measured departure.
                            </span>
                          ) : s.anchor.verified ? (
                            <span style={{ color: "#1a9d55" }}>
                              impact PINNED at f{s.anchor.impact_frame} by
                              ball departure (MOG2, no AI){" "}
                              {s.anchor.impact_delta != null
                                ? `· ${s.anchor.impact_delta >= 0 ? "+" : ""}${s.anchor.impact_delta}f vs pose peak`
                                : ""}
                              {s.anchor.snapped
                                ? ` · rest snapped ${s.anchor.snap_px}px`
                                : ""}
                            </span>
                          ) : (
                            <span style={{ color: "#b7791f" }}>
                              departure pin failed — {s.anchor.reason}
                            </span>
                          )}
                          {s.anchor.ai_fallback_reason != null && (
                            <div className="tiny" style={{ color: "#b7791f" }}>
                              ⚠ AI anchor check bailed (
                              {s.anchor.ai_fallback_reason}) — the strip
                              below is the PIXEL fallback, not the AI's
                              read
                            </div>
                          )}
                          {s.anchor.image_url && (
                            <div style={{ marginTop: 3 }}>
                              {s.anchor.image_mog2_url && (
                                <button
                                  type="button"
                                  className={
                                    anchorHeat[s.swing]
                                      ? "small"
                                      : "ghost small"
                                  }
                                  style={{ width: "auto", padding: "1px 8px", marginBottom: 3 }}
                                  onClick={() =>
                                    setAnchorHeat((m) => ({
                                      ...m,
                                      [s.swing]: !m[s.swing],
                                    }))
                                  }
                                  title="Toggle the MOG2 view — the same tiles as frame-diff heat vs the pre-impact baseline. Tiles stay dark while the ball rests; the vacated spot lights up from the departure frame on."
                                >
                                  🔥 mog2 {anchorHeat[s.swing] ? "on" : "off"}
                                </button>
                              )}
                              <a
                                href={
                                  anchorHeat[s.swing] && s.anchor.image_mog2_url
                                    ? s.anchor.image_mog2_url
                                    : s.anchor.image_url
                                }
                                target="_blank"
                                rel="noreferrer"
                                style={{ display: "block" }}
                                title="Departure film-strip: numbered tiles of the rest patch, yellow box=the frame the ball left (impact), ring=watched spot"
                              >
                                <img
                                  src={
                                    anchorHeat[s.swing] && s.anchor.image_mog2_url
                                      ? s.anchor.image_mog2_url
                                      : s.anchor.image_url
                                  }
                                  alt="departure film-strip"
                                  style={{ maxWidth: "100%", borderRadius: 6 }}
                                />
                              </a>
                            </div>
                          )}
                          {s.anchor.ai_launch_n != null && (
                            <div style={{ marginTop: 3 }}>
                              \ud83e\udd16 AI launch plot:{" "}
                              <span
                                style={{
                                  color: s.anchor.ai_launch_n > 0
                                    ? "#1a9d55"
                                    : "#b7791f",
                                }}
                              >
                                {s.anchor.ai_launch_reason}
                              </span>
                            </div>
                          )}
                          {s.anchor.ai_launch_image_url && (
                            <a
                              href={s.anchor.ai_launch_image_url}
                              target="_blank"
                              rel="noreferrer"
                              style={{ display: "block", marginTop: 3 }}
                              title="AI launch plot \u2014 the first 5 post-impact frames sent to the vision model. Magenta ring = its ball pick, cyan ring = the rest spot, green border = found, red = not found. These picks override the pixel tracker on their frames; MOG2 owns the rest of the flight."
                            >
                              <img
                                src={s.anchor.ai_launch_image_url}
                                alt="AI launch plot film-strip"
                                style={{ maxWidth: "100%", borderRadius: 6 }}
                              />
                            </a>
                          )}
                          {s.anchor.launch_n != null && (
                            <div style={{ marginTop: 3 }}>
                              {"\ud83d\ude80"} launch tracker:{" "}
                              {s.anchor.launch_n > 0 ? (
                                <span style={{ color: "#1a9d55" }}>
                                  {s.anchor.launch_n} flight points —{" "}
                                  {s.anchor.launch_reason}
                                </span>
                              ) : (
                                <span style={{ color: "#b7791f" }}>
                                  {s.anchor.launch_reason}
                                </span>
                              )}
                            </div>
                          )}
                          {s.anchor.launch_image_url && (
                            <div style={{ marginTop: 3 }}>
                              {s.anchor.launch_image_heat_url && (
                                <button
                                  type="button"
                                  className={
                                    launchHeat[s.swing]
                                      ? "small"
                                      : "ghost small"
                                  }
                                  style={{ width: "auto", padding: "1px 8px", marginBottom: 3 }}
                                  onClick={() =>
                                    setLaunchHeat((m) => ({
                                      ...m,
                                      [s.swing]: !m[s.swing],
                                    }))
                                  }
                                  title="Toggle the MOG2/frame-diff motion mask (red tint) on the launch-tracker tiles — what the tracker actually looked at."
                                >
                                  🔥 heat {launchHeat[s.swing] ? "on" : "off"}
                                </button>
                              )}
                              <a
                                href={
                                  launchHeat[s.swing]
                                    ? s.anchor.launch_image_heat_url
                                    : s.anchor.launch_image_url
                                }
                                target="_blank"
                                rel="noreferrer"
                                style={{ display: "block" }}
                                title="Adaptive-square launch tracker: the square rides the ball (bottom-third bias while ascending), widens on a miss, shrinks on a find. Green=found (ball ringed), red=missed, box size labelled per tile."
                              >
                                <img
                                  src={
                                    launchHeat[s.swing]
                                      ? s.anchor.launch_image_heat_url
                                      : s.anchor.launch_image_url
                                  }
                                  alt="launch tracker film-strip"
                                  style={{ maxWidth: "100%", borderRadius: 6 }}
                                />
                              </a>
                            </div>
                          )}
                        </div>
                      )}
                      <div style={{ display: "flex", gap: 10, marginTop: 4 }}>
                        {[["before", s.before], ["after", s.after]].map(
                          ([tag, sh]) =>
                            sh && sh.image_url ? (
                              <div key={tag} style={{ width: 200, maxWidth: "100%" }}>
                                <a href={sh.image_url} target="_blank" rel="noreferrer">
                                  <img
                                    src={sh.image_url}
                                    alt={`${tag} swing ${s.swing}`}
                                    style={{
                                      width: "100%", borderRadius: 6, display: "block",
                                      outline: sh.present
                                        ? "2px solid #00b8d4"
                                        : "2px solid #c0392b",
                                    }}
                                  />
                                </a>
                                <div className="small muted">
                                  {tag}: {sh.present ? "ball" : "no ball"} @ {sh.t}s
                                </div>
                              </div>
                            ) : null,
                        )}
                      </div>
                    </div>
                  );
                })}
              </>
            ) : (
              <div className="small muted">
                {data.ai_ball.reason || "needs ANTHROPIC_API_KEY"}
              </div>
            )}
          </div>
        )}

        {data.pose && (
          <div
            style={{
              border: "1px solid rgba(155,89,182,0.4)", borderRadius: 8,
              padding: "8px 12px", marginBottom: 14,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>
              🧍 Pose detector — {data.pose.available ? `${data.pose.n_swings ?? 0} swing(s)` : "unavailable"}
            </div>
            {data.pose.available ? (
              <>
                <div className="small muted">
                  wrist speed gated by back bend (≥
                  {data.pose.back_bend_min_deg ?? 15}° spine tilt) ·
                  pose tracked {data.pose.n_pose_frames ?? 0}/
                  {data.pose.n_samples ?? "?"} frames
                  {data.pose.coverage != null
                    ? ` (${Math.round(data.pose.coverage * 100)}%)`
                    : ""}
                  {data.pose.n_bend_rejected
                    ? ` · rejected ${data.pose.n_bend_rejected} upright (fast hands, no swing posture)`
                    : ""}
                </div>
                <PoseChart pose={data.pose} />
                {(data.pose.screenshots || []).length > 0 && (
                  <div
                    style={{
                      display: "flex", flexWrap: "wrap", gap: 10, marginTop: 6,
                    }}
                  >
                    {data.pose.screenshots.map((s, i) => (
                      <div key={i} style={{ width: 220, maxWidth: "100%" }}>
                        <a href={s.image_url} target="_blank" rel="noreferrer">
                          <img
                            src={s.image_url}
                            alt={`pose swing ${i + 1}`}
                            style={{ width: "100%", borderRadius: 6, display: "block" }}
                          />
                        </a>
                        <div className="small muted" style={{ marginTop: 2 }}>
                          swing {i + 1}: {s.t}s
                          {s.back_bend_deg != null
                            ? ` · bend ${Math.round(s.back_bend_deg)}°`
                            : " · bend n/a"}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div className="small muted">
                {data.pose.reason || "mediapipe not installed"} — install it on
                the dev deployment (<code>pip install mediapipe</code>) to enable.
              </div>
            )}
          </div>
        )}


        {data.heat_check?.swings?.length > 0 && (
          <div
            style={{
              border: "1px solid rgba(230,126,34,0.45)", borderRadius: 8,
              padding: "8px 12px", marginBottom: 14,
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 2 }}>
              🔥 MOG2 swing check —{" "}
              {data.heat_check.swings.filter(
                (s) => s.verdict && s.verdict !== "no_swing",
              ).length}
              /{data.heat_check.swings.length} confirmed
              {!data.heat_check.enabled && " (filter disabled)"}
            </div>
            <div className="small muted" style={{ marginBottom: 6 }}>
              The AI judge looks at the motion-heat composite and decides
              for every swing (club-fan heuristic is the no-key fallback).
              ✅ swing = kept. ❌ no swing = eliminated: skipped by every
              later stage and not produced. The launch chain (red line) is
              drawn as evidence only — it no longer decides anything.
              (Fail-safe: heuristic-only rejections are resurrected if
              everything was rejected; an AI-judged "not a swing" stays out.)
            </div>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 10 }}>
              {data.heat_check.swings.map((s) => (
                <div key={s.swing} style={{ width: 260, maxWidth: "100%" }}>
                  {s.image_url && (
                    <a href={s.image_url} target="_blank" rel="noreferrer">
                      <img
                        src={s.image_url}
                        alt={`heat check swing ${s.swing}`}
                        style={{ width: "100%", borderRadius: 6, display: "block" }}
                      />
                    </a>
                  )}
                  <div className="small" style={{ marginTop: 2 }}>
                    swing {s.swing} · {s.t}s:{" "}
                    {s.verdict === "club_swing" || s.verdict === "ball_flight" ? (
                      <b style={{ color: "#1a9d55" }}>
                        ✅ swing
                        {s.ai_judge != null
                          ? " (AI judge)"
                          : ` (${s.n_rays} rays @ ${s.n_angles ?? "?"} angles)`}
                      </b>
                    ) : s.verdict === "no_swing" || s.verdict === "no_ball_flight" ? (
                      <b style={{ color: "#c0392b" }}>
                        ❌ no swing ({s.n_timed} dots · {s.n_rays ?? 0} rays)
                      </b>
                    ) : (
                      <span className="muted">
                        check unavailable{s.reason ? ` — ${s.reason}` : ""}
                      </span>
                    )}
                    {s.chain_len > 0 && (
                      <span className="muted">
                        {" "}· chain {s.chain_len} (f{s.chain_f0}–f{s.chain_f1},
                        evidence only)
                      </span>
                    )}
                    {s.ai_judge != null && (
                      <div className="tiny muted">
                        🤖 AI judge: {s.ai_judge ? "swing" : "not a swing"}
                        {s.ai_reason ? ` — ${s.ai_reason}` : ""}
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {swings.length === 0 && !data.running && (
          <div className="small muted">
            {data.final_verdict?.n_candidates > 0
              ? "Every pose candidate was eliminated by the filters — no tracer comparisons to run."
              : "No swings detected in this clip."}
          </div>
        )}

        {swings.map((s) => (
          <div
            key={s.idx}
            style={{
              border: "1px solid rgba(120,120,120,0.35)", borderRadius: 10,
              padding: 12, marginBottom: 12,
            }}
          >
            <div style={{ marginBottom: 8, fontWeight: 700 }}>
              Swing {s.idx + 1}
              {s.hole_number != null ? ` · hole ${s.hole_number}` : ""}
              {s.peak_time_sec != null ? ` · impact ~${s.peak_time_sec}s` : ""}
              {s.ball_verdict ? ` · ball: ${s.ball_verdict}` : ""}
            </div>
            <div
              style={{
                display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14,
              }}
            >
              {/* Classical */}
              <div>
                <div style={{ marginBottom: 6 }}>
                  <strong>Classical CV</strong> — {okBadge(s.classical?.ok)}
                </div>
                {stat("candidates", s.classical?.n_candidates)}
                {stat("tracer points", s.classical?.n_points)}
                {s.classical?.error && (
                  <div className="small" style={{ color: "#c0392b" }}>
                    {s.classical.error}
                  </div>
                )}
                {s.classical?.heatmap_url && (
                  <div style={{ marginTop: 6 }}>
                    <div className="small muted">motion heatmap + candidates:</div>
                    <a href={s.classical.heatmap_url} target="_blank" rel="noreferrer">
                      <img
                        src={s.classical.heatmap_url}
                        alt="classical heatmap"
                        style={{ maxWidth: "100%", borderRadius: 6, marginTop: 4 }}
                      />
                    </a>
                  </div>
                )}
                {s.classical?.traced_url && (
                  <video
                    src={s.classical.traced_url}
                    controls
                    style={{ width: "100%", borderRadius: 6, marginTop: 6 }}
                  />
                )}
              </div>
              {/* AI */}
              <div>
                <div style={{ marginBottom: 6 }}>
                  <strong>
                    {s.ai?.production ? "Production tracer" : "AI tracer"}
                  </strong>
                  {s.ai?.engine ? ` (${s.ai.engine})` : ""} — {okBadge(s.ai?.ok)}
                </div>
                {stat("address frame", s.ai?.address_frame)}
                {stat("impact frame", s.ai?.impact_frame)}
                {stat("handedness", s.ai?.handedness)}
                {stat("ball-track points", s.ai?.n_track)}
                {s.ai?.anchor_check && (
                  <div className="tiny" style={{ marginTop: 2 }}>
                    {"\u2693"} anchors:{" "}
                    {s.ai.anchor_check.verified ? (
                      <span style={{ color: "#1a9d55" }}>
                        verified — rest{" "}
                        {s.ai.anchor_check.snapped
                          ? `snapped ${s.ai.anchor_check.snap_px}px`
                          : "exact"}
                        , impact by ball departure (
                        {s.ai.anchor_check.impact_delta >= 0 ? "+" : ""}
                        {s.ai.anchor_check.impact_delta}f vs estimate)
                      </span>
                    ) : (
                      <span style={{ color: "#b7791f" }}>
                        unverified — {s.ai.anchor_check.reason}
                      </span>
                    )}
                    {s.ai.anchor_check.ai_fallback_reason != null && (
                      <div className="tiny" style={{ color: "#b7791f" }}>
                        ⚠ AI check bailed (
                        {s.ai.anchor_check.ai_fallback_reason}) — pixel
                        fallback shown
                      </div>
                    )}
                    {s.ai.anchor_check.image_url && (
                      <div style={{ marginTop: 4 }}>
                        {s.ai.anchor_check.image_mog2_url && (
                          <button
                            type="button"
                            className={
                              anchorHeat[`ai-${s.swing}`]
                                ? "small"
                                : "ghost small"
                            }
                            style={{ width: "auto", padding: "1px 8px", marginBottom: 3 }}
                            onClick={() =>
                              setAnchorHeat((m) => ({
                                ...m,
                                [`ai-${s.swing}`]: !m[`ai-${s.swing}`],
                              }))
                            }
                            title="Toggle the MOG2 view — the same tiles as frame-diff heat vs the pre-impact baseline."
                          >
                            🔥 mog2 {anchorHeat[`ai-${s.swing}`] ? "on" : "off"}
                          </button>
                        )}
                        <a
                          href={
                            anchorHeat[`ai-${s.swing}`] &&
                            s.ai.anchor_check.image_mog2_url
                              ? s.ai.anchor_check.image_mog2_url
                              : s.ai.anchor_check.image_url
                          }
                          target="_blank"
                          rel="noreferrer"
                          style={{ display: "block" }}
                          title="Anchor-check film-strip — numbered rest-patch crops across impact, yellow box = the departure frame (impact), ring = watched spot. Click to open full size."
                        >
                          <img
                            src={
                              anchorHeat[`ai-${s.swing}`] &&
                              s.ai.anchor_check.image_mog2_url
                                ? s.ai.anchor_check.image_mog2_url
                                : s.ai.anchor_check.image_url
                            }
                            alt="anchor check film-strip"
                            style={{ maxWidth: "100%", borderRadius: 6 }}
                          />
                        </a>
                      </div>
                    )}
                  </div>
                )}
                {s.ai?.anchor_check?.ai_launch_image_url && (
                  <div style={{ marginTop: 4 }}>
                    <div className="tiny">
                      🤖 AI launch plot — how the magenta points are
                      found:{" "}
                      <span
                        style={{
                          color: s.ai.anchor_check.ai_launch_n
                            ? "#1a9d55"
                            : "#b7791f",
                        }}
                      >
                        {s.ai.anchor_check.ai_launch_reason ||
                          `${s.ai.anchor_check.ai_launch_n ?? 0} frame(s)`}
                      </span>
                      . These points are 📍NED in the tracer fit DASH
                      nothing overrides them.
                    </div>
                    <a
                      href={s.ai.anchor_check.ai_launch_image_url}
                      target="_blank"
                      rel="noreferrer"
                      style={{ display: "block" }}
                      title="The first post-impact frames sent to the vision model, one crop per frame. Magenta ring = its ball pick, cyan ring = the previous position, green border = found, red = not found."
                    >
                      <img
                        src={s.ai.anchor_check.ai_launch_image_url}
                        alt="AI launch plot film-strip"
                        style={{ maxWidth: "100%", borderRadius: 6 }}
                      />
                    </a>
                  </div>
                )}
                {s.ai?.mog2_stats && (
                  <div className="tiny muted" style={{ marginTop: 3 }}>
                    AI launch points handed to the render:{" "}
                    <b
                      style={{
                        color: s.ai.mog2_stats.n_launch_in
                          ? "inherit"
                          : "#c0392b",
                      }}
                    >
                      {s.ai.mog2_stats.n_launch_in ?? "?"}
                    </b>
                    {" · "}added to the arc: {s.ai.mog2_stats.n_added_track ?? 0}
                  </div>
                )}
                {(() => {
                  const ri = s.ai?.render_info;
                  if (!ri) return null;
                  const moved =
                    ri.rest_anchor_relocated ||
                    ri.rest_anchor_dropped ||
                    ri.rest_anchor_synthesized;
                  if (!moved) {
                    return (
                      <div
                        className="tiny"
                        style={{ marginTop: 3, color: "#1a9d55" }}
                      >
                        📍 start point LOCKED on the verified rest/impact
                        — the render did not move it.
                      </div>
                    );
                  }
                  return (
                    <div
                      className="tiny"
                      style={{ marginTop: 3, color: "#c0392b" }}
                    >
                      {ri.rest_anchor_dropped && (
                        <>
                          📍 the render DROPPED the rest anchor (
                          {ri.rest_anchor_dropped.nearest_detection_px}px from
                          the nearest detection, limit{" "}
                          {ri.rest_anchor_dropped.threshold_px}px) — the
                          line starts at the first tracked point instead.
                        </>
                      )}
                      {ri.rest_anchor_relocated && (
                        <>
                          📍 the render MOVED the start: (
                          {ri.rest_anchor_relocated.from?.join(", ")}) → (
                          {ri.rest_anchor_relocated.to?.join(", ")}),{" "}
                          {ri.rest_anchor_relocated.dist_px}px.
                        </>
                      )}
                      {ri.rest_anchor_synthesized && (
                        <>
                          📍 no rest anchor reached the render — start
                          synthesized at the extrapolated launch origin (
                          {ri.rest_anchor_synthesized.xy?.join(", ")}).
                        </>
                      )}
                    </div>
                  );
                })()}
                {s.ai?.raw_motion_url &&
                  ((s.ai.timed_points || []).length > 0 ||
                    (s.ai.anchor_check?.ai_launch_points || []).length >
                      0) && (
                  <div style={{ marginTop: 4 }}>
                    <div className="tiny muted">
                      🗺 flight map — blue line = the FULL mapped arc
                      (AI + launch + MOG2), orange rings = pool dots
                      added by arc completion, dashed red = the
                      arc-completion search region, magenta = rest →
                      AI launch (cyan dot = rest), amber = unused pool
                      dots with frame labels
                    </div>
                    <FlightMapStatic
                      bgUrl={s.ai.raw_motion_url}
                      dots={s.ai.timed_points}
                      restBall={s.ai.ball}
                      aiPoints={s.ai.anchor_check?.ai_launch_points}
                      trackPoints={s.ai.track_points}
                      frameW={s.ai.frame_w}
                      frameH={s.ai.frame_h}
                      impactFrame={s.ai.impact_frame}
                      region={s.ai.arc_region}
                    />
                  </div>
                )}
                {s.ai?.error && (
                  <div className="small" style={{ color: "#c0392b" }}>
                    {s.ai.error}
                  </div>
                )}
                {s.ai?.traced_url && (
                  <video
                    src={s.ai.traced_url}
                    controls
                    style={{ width: "100%", borderRadius: 6, marginTop: 6 }}
                  />
                )}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

// Debug and Debug2 are retired: Debug3's pipeline now runs inside produce
// (settings.debug3_tracer). The code and endpoints are left in place so a
// swing can still be compared against the old methods by flipping this,
// but the buttons are off by default -- three diagnostic buttons on every
// row is noise once one of them is the thing that actually ships.
const SHOW_LEGACY_DEBUG = false;

export default function AdminProduction() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  // Search + sort for the production queue. `courseSearch` is the live
  // text input; `courseQuery` is the debounced value actually sent to the
  // backend so typing doesn't refetch on every keystroke. `sortKey`
  // combines the sort field and direction into one dropdown value.
  const [courseSearch, setCourseSearch] = useState("");
  const [courseQuery, setCourseQuery] = useState("");
  const [sortKey, setSortKey] = useState("created_desc");
  useEffect(() => {
    const t = setTimeout(() => setCourseQuery(courseSearch.trim()), 350);
    return () => clearTimeout(t);
  }, [courseSearch]);
  const [sortField, sortOrder] =
    sortKey === "created_asc"
      ? ["created", "asc"]
      : sortKey === "course_asc"
        ? ["course", "asc"]
        : sortKey === "course_desc"
          ? ["course", "desc"]
          : ["created", "desc"];

  const fetchUploads = useCallback(
    (limit, offset) =>
      api.listLongUploads(adminPassword, limit, offset, {
        course: courseQuery,
        sort: sortField,
        order: sortOrder,
      }),
    [adminPassword, courseQuery, sortField, sortOrder],
  );
  const fetchEvents = useCallback(
    (limit, offset) => api.listCameraEvents(adminPassword, limit, offset),
    [adminPassword],
  );
  const uploadsList = useInfiniteList(fetchUploads, {
    pageSize: 25,
    deps: [adminPassword, courseQuery, sortField, sortOrder],
  });
  const eventsList = useInfiniteList(fetchEvents, {
    pageSize: 25,
    deps: [adminPassword],
  });
  // Alias to keep the existing JSX (which reads `rows` and `events`)
  // untouched. Both come from the hook so they share the same null →
  // [...] lifecycle.
  const rows = uploadsList.items;
  const events = eventsList.items;
  // Hide uploads whose raw source is gone (e.g. a dev deployment that
  // lost its ephemeral files) — a "File missing" card can't be edited,
  // produced, or replayed, so it's pure clutter. The tee is the primary
  // source; when it's missing the whole row is dead. Raw list still
  // drives pagination (hasMore reads the unfiltered batch length).
  const visibleRows = rows ? rows.filter((r) => !r.tee_missing) : rows;
  const allHidden =
    rows !== null && rows.length > 0 && visibleRows.length === 0;

  const [actionError, setActionError] = useState(null);
  const error = actionError || uploadsList.error || eventsList.error;
  const setError = setActionError;
  const [busyId, setBusyId] = useState(null);
  // What the greyed card SAYS it is doing, when it isn't a produce.
  // Null means "a produce run", which the overlay narrates from the
  // server's own stage fields.
  const [busyLabel, setBusyLabel] = useState(null);
  // A centred confirmation: {title, body, confirmLabel, onConfirm}.
  const [confirmBox, setConfirmBox] = useState(null);
  const [busyEventId, setBusyEventId] = useState(null);
  const [viewer, setViewer] = useState(null); // {url, title, startedAt, fps}
  const [editingRow, setEditingRow] = useState(null);
  // When the current optimistic "busy" began, so a completion stamped
  // before it (the PREVIOUS run's) cannot be mistaken for this one's.
  const busySinceRef = useRef(0);
  // Standalone click-to-plot modal opened from a card's 🖱 button:
  // {row, swingPos} — swingPos is the index into edit_metrics.swings.
  const [plotModal, setPlotModal] = useState(null);
  // Bulk-delete selection: a Set of long-upload row ids the operator
  // has ticked. Cleared after a bulk delete completes.
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  // "Pull new from prod" — server-side mirror. Hidden unless the backend
  // is configured (MIRROR_COURSE_ID set on this deployment).
  const [mirror, setMirror] = useState({ configured: false, running: false });
  // "Produce (debug)" — dev tool. Produces normally AND opens a per-swing
  // diagnostic comparing the classical-CV and AI tracers. Hidden unless the
  // backend enables it (PRODUCE_DEBUG_ENABLED on this deployment).
  const [produceDebug, setProduceDebug] = useState({ enabled: false });
  // Debug2 report — kept in memory only; the run writes images but
  // changes no swing data, so there is nothing to persist.
  const [d2, setD2] = useState(null);
  // Debug3 report — same deal: images on disk, no swing data touched.
  const [d3, setD3] = useState(null);
  // Swing test — the ball-departure detector alone. Read-only as well.
  const [swingTest, setSwingTest] = useState(null);
  const [ballScan, setBallScan] = useState(null);
  const [hitArea, setHitArea] = useState(null);
  // Tee <-> green calibration, opened from under the green raw video.
  // { uploadId, tee, green, existing, scope } once the two frames and
  // any saved map are in hand; { uploadId, loading } while fetching.
  const [greenCal, setGreenCal] = useState(null);
  const [pinModal, setPinModal] = useState(null);
  const [debugModal, setDebugModal] = useState(null); // { uploadId, ...report }

  function pollDebug(uploadId) {
    const tick = async () => {
      try {
        const s = await api.produceDebugStatus(adminPassword, uploadId);
        setDebugModal((m) => (m && m.uploadId === uploadId ? { ...m, ...s } : m));
        if (s.running) setTimeout(tick, 2500);
      } catch {
        /* transient — stop polling */
      }
    };
    tick();
  }

  // Debug2/Debug3 are background runs now: POST kicks them off, then we
  // poll. Holding the connection open for the whole pipeline is what
  // produced the 502s -- the proxy dropped it and the browser retried,
  // restarting the run from scratch each time.
  function pollDebugX(kind, uploadId, setter) {
    const statusCall =
      kind === "debug3" ? api.debug3Status
        : kind === "swingtest" ? api.swingTestStatus
          : kind === "ballscan" ? api.ballScanStatus
            : kind === "ballscanproduce" ? api.ballScanProduceStatus
          : api.debug2Status;
    const tick = async () => {
      try {
        const st = await statusCall(adminPassword, uploadId);
        // MERGE, do not replace. The opener puts things on this state
        // that no poll knows about -- the admin password and a re-run
        // handle for the tee-box drawer -- and a plain object here
        // silently dropped them on the first tick.
        setter((prev) => ({
          ...(prev || {}),
          running: !!st.running,
          uploadId,
          stage: st.stage,
          done: st.done,
          total: st.total,
          report: st.report || null,
          error: st.error || null,
        }));
        if (st.running) setTimeout(tick, 2500);
        else setBusyId((cur) => (cur === uploadId ? null : cur));
      } catch (e) {
        setter((prev) => ({
          ...(prev || {}),
          running: false, uploadId, report: null, error: e.message,
        }));
        setBusyId((cur) => (cur === uploadId ? null : cur));
      }
    };
    setTimeout(tick, 1200);
  }

  async function handleDebug2(row) {
    // Open the window FIRST. Anything that throws after this still leaves
    // the operator with a visible panel carrying the error, instead of a
    // button that appears to do nothing.
    setD2({ running: true, uploadId: row.id, report: null, error: null });
    busySinceRef.current = Date.now();
    setBusyId(row.id);
    try {
      await api.debug2(adminPassword, row.id);
      pollDebugX("debug2", row.id, setD2);
    } catch (e) {
      setD2({
        running: false, uploadId: row.id, report: null, error: e.message,
      });
      setBusyId((cur) => (cur === row.id ? null : cur));
    }
  }

  // What sat still and looked like a ball, across the whole clip. No
  // pose, no swing, no produce — the one question that everything else
  // here assumes has already been answered.
  async function handleBallScan(row) {
    setBallScan({ running: true, uploadId: row.id, report: null, error: null });
    busySinceRef.current = Date.now();
    setBusyId(row.id);
    try {
      await api.ballScan(adminPassword, row.id);
      pollDebugX("ballscan", row.id, setBallScan);
    } catch (e) {
      setBallScan({
        running: false, uploadId: row.id, report: null, error: e.message,
      });
      setBusyId((cur) => (cur === row.id ? null : cur));
    }
  }

  // Straight from the two numbers the scan measured -- rest position and
  // impact frame -- to a traced clip, for every candidate that sat long
  // enough to have been a ball waiting to be hit.
  async function handleBallScanProduce(row) {
    setBallScan({
      running: true, uploadId: row.id, report: null, error: null,
      producing: true,
    });
    busySinceRef.current = Date.now();
    setBusyId(row.id);
    try {
      await api.ballScanProduce(adminPassword, row.id);
      pollDebugX("ballscanproduce", row.id, setBallScan);
    } catch (e) {
      setBallScan({
        running: false, uploadId: row.id, report: null, error: e.message,
      });
      setBusyId((cur) => (cur === row.id ? null : cur));
    }
  }

  async function handleDebug3(row) {
    // Window first, same reason as Debug2: a throw after this still leaves
    // a visible panel carrying the error.
    // adminPassword and a re-run handle ride along so the tee-box drawer
    // inside the modal can save a box and immediately search inside it,
    // which is the only reason anyone draws one.
    setD3({
      running: true, uploadId: row.id, report: null, error: null,
      adminPassword, onRerun: () => handleDebug3(row),
    });
    busySinceRef.current = Date.now();
    setBusyId(row.id);
    try {
      await api.debug3(adminPassword, row.id);
      pollDebugX("debug3", row.id, setD3);
    } catch (e) {
      setD3({
        running: false, uploadId: row.id, report: null, error: e.message,
      });
      setBusyId((cur) => (cur === row.id ? null : cur));
    }
  }

  // The ball question on its own: where did it look, did it find a ball,
  // did that ball leave, and on which frame. No pose, no produce.
  async function handleSwingTest(row) {
    setSwingTest({ running: true, uploadId: row.id, report: null, error: null });
    busySinceRef.current = Date.now();
    setBusyId(row.id);
    try {
      await api.swingTest(adminPassword, row.id);
      pollDebugX("swingtest", row.id, setSwingTest);
    } catch (e) {
      setSwingTest({
        running: false, uploadId: row.id, report: null, error: e.message,
      });
      setBusyId((cur) => (cur === row.id ? null : cur));
    }
  }

  async function handleProduceDebug(row) {
    setError(null);
    setDebugModal({
      uploadId: row.id, running: true, total: 0, done: 0, swings: [],
      ai_available: false,
    });
    try {
      const r = await api.produceDebug(adminPassword, row.id);
      if (r && r.ok === false) {
        setError(r.error || "Produce debug is not enabled.");
        setDebugModal(null);
        return;
      }
      pollDebug(row.id);
      refreshAll(); // the normal produce is running in parallel
    } catch (e) {
      setError(e.message);
      setDebugModal(null);
    }
  }

  // Re-run ONLY the ball detector with the new ROI (fast, synchronous) and
  // patch the modal in place — the tracer comparison is unaffected by ROI so
  // there's no need to re-run the whole (slow) analysis.
  async function rerunBallScan(uploadId) {
    setError(null);
    try {
      const r = await api.rescanBall(adminPassword, uploadId);
      if (r && r.ok === false) {
        setError(r.error || "Ball rescan failed.");
        return;
      }
      setDebugModal((m) =>
        m && m.uploadId === uploadId
          ? {
              ...m,
              ball: r.ball,
              ball_roi: r.ball_roi,
              ref_frame_url: r.ref_frame_url || m.ref_frame_url,
            }
          : m,
      );
    } catch (e) {
      setError(e.message);
    }
  }



  function pollMirror() {
    const tick = async () => {
      try {
        const s = await api.mirrorFromProdStatus(adminPassword);
        setMirror(s);
        if (s.running) {
          setTimeout(tick, 2000);
        } else {
          await refreshAll(); // surface newly-imported events
        }
      } catch {
        /* transient — stop polling */
      }
    };
    tick();
  }

  async function handlePullFromProd() {
    setError(null);
    try {
      const r = await api.mirrorFromProd(adminPassword);
      if (r && r.ok === false) {
        setError(r.error || "Pull from prod is not configured.");
        return;
      }
      setMirror((m) => ({ ...m, running: true, done: 0, total: 0, failed: 0 }));
      pollMirror();
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    api
      .mirrorFromProdStatus(adminPassword)
      .then((s) => {
        setMirror(s);
        if (s.running) pollMirror();
      })
      .catch(() => {});
    // upload_id 0 never exists — the status route still returns `enabled`
    // with an empty report, which is all we need to show/hide the button.
    api
      .produceDebugStatus(adminPassword, 0)
      .then((s) => setProduceDebug({ enabled: !!s.enabled }))
      .catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function toggleSelected(id) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }
  function clearSelection() {
    setSelectedIds(new Set());
  }
  function selectAllVisible() {
    setSelectedIds(new Set((visibleRows || []).map((r) => r.id)));
  }

  function handleBulkDelete() {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    setConfirmBox({
      title: `Delete ${ids.length} selected upload${ids.length === 1 ? "" : "s"}?`,
      body: "Their source videos and anything produced from them are "
        + "removed. It cannot be undone.",
      confirmLabel: `Delete ${ids.length}`,
      onConfirm: async () => {
        setConfirmBox(null);
        setBulkBusy(true);
        setError(null);
        clearSelection();
        let failed = 0;
        for (const id of ids) {
          try {
            await api.deleteLongUpload(adminPassword, id);
            // One at a time, as each lands — a batch of twenty should
            // visibly shrink rather than sit still and then jump.
            dropRow(id);
          } catch (e) {
            failed += 1;
          }
        }
        await refreshAll();
        setBulkBusy(false);
        if (failed) setError(`${failed} of ${ids.length} deletions failed.`);
      },
    });
  }

  function openViewer(url, title, startedAt = null, fps = null,
                      startedApprox = false) {
    if (!url) return;
    setViewer({ url, title, startedAt, fps, startedApprox });
  }

  // Re-fetch the currently-loaded range on both lists. Called by the
  // background poll loop and by every handler that mutates server state,
  // so a successful action shows its effect without a full reload (which
  // would yank the user back to page 1).
  const refreshAll = useCallback(async () => {
    setError(null);
    await Promise.all([uploadsList.refresh(), eventsList.refresh()]);
  }, [uploadsList.refresh, eventsList.refresh]);

  // THE CARD IS RENDERED FROM THE ROW, SO CHANGE THE ROW. An optimistic
  // flag held beside the list (busyId) only greys the card if every
  // other piece of the hand-off agrees; patching the row itself means
  // the card renders the new state for exactly the same reason it
  // renders the server's, with nothing in between. Both of these are
  // synchronous: no await stands between the operator's click and the
  // card changing.
  const patchRow = useCallback((id, patch) => {
    uploadsList.setItems((prev) =>
      (prev || []).map((r) => (r.id === id ? { ...r, ...patch } : r)),
    );
  }, [uploadsList.setItems]);

  const dropRow = useCallback((id) => {
    uploadsList.setItems((prev) => (prev || []).filter((r) => r.id !== id));
  }, [uploadsList.setItems]);

  // The same, for the camera-events list. Takes a function because an
  // event's interesting state is nested (produced_clip), so a shallow
  // merge is not enough.
  const patchEvent = useCallback((id, fn) => {
    eventsList.setItems((prev) =>
      (prev || []).map((e) => (e.id === id ? fn(e) : e)),
    );
  }, [eventsList.setItems]);

  const dropEvent = useCallback((id) => {
    eventsList.setItems((prev) => (prev || []).filter((e) => e.id !== id));
  }, [eventsList.setItems]);

  async function handleReproduceEvent(ev) {
    setBusyEventId(ev.id);
    try {
      await api.reprocessCameraEvent(adminPassword, ev.id);
      await refreshAll();
      // Do NOT clear busy here. The POST returns before the worker has
      // flipped the event to "processing", so clearing it in a finally
      // produced: grey for a second -> back to normal -> grey again a few
      // seconds later. The effect below hands over to the server's status
      // once that status actually says processing, so the row never
      // un-greys in between.
    } catch (e) {
      setError(e.message);
      setBusyEventId((cur) => (cur === ev.id ? null : cur));
    }
  }

  // Hand off from the optimistic busy flag to the server's own status.
  useEffect(() => {
    if (busyEventId == null) return;
    const ev = (events || []).find((e) => e.id === busyEventId);
    if (!ev) return;
    // Once the worker owns it, the status badge carries the state and the
    // local flag has done its job. Also release if the run already finished
    // (fast swings), so the row cannot stick.
    // Only "processing" -- see the note on the upload effect above: the
    // event is already "processed" when the operator clicks, so accepting
    // terminal states here cleared busy immediately.
    if (ev.status === "processing") {
      setBusyEventId(null);
    }
  }, [events, busyEventId]);

  useEffect(() => {
    if (busyEventId == null) return undefined;
    const t = setTimeout(() => setBusyEventId(null), 180_000);
    return () => clearTimeout(t);
  }, [busyEventId]);

  function handleDeleteEvent(ev) {
    setConfirmBox({
      title: `Delete camera event #${ev.id}?`,
      body: "The raw tee and green clips and the produced clip are all "
        + "removed. It cannot be undone.",
      confirmLabel: "Delete event",
      onConfirm: () => {
        // Dialog down and event gone, both on the click — same as the
        // production card's Delete.
        setConfirmBox(null);
        dropEvent(ev.id);
        (async () => {
          try {
            await api.deleteCameraEvent(adminPassword, ev.id);
          } catch (e) {
            setError(e.message);
          }
          await refreshAll();
        })();
      },
    });
  }

  function handleBroadcastEvent(ev) {
    // Toggle the produced clip's is_highlight flag — same semantic as
    // the long-upload Broadcast button, just sourced from the event's
    // produced_clip relation instead of produced_clips[0]. Optimistic
    // for the same reason: one boolean, nothing to wait for.
    const clip = ev.produced_clip;
    if (!clip) return;
    const next = !clip.is_highlight;
    const flip = (v) => patchEvent(ev.id, (e) => ({
      ...e,
      produced_clip: { ...e.produced_clip, is_highlight: v },
    }));
    flip(next);
    (async () => {
      try {
        await api.setClipBroadcast(adminPassword, clip.id, next);
      } catch (e) {
        flip(!next);
        setError(e.message);
        return;
      }
      refreshAll();
    })();
  }

  // Something mid-produce means the stage text on its card is changing;
  // poll faster so it reads as live rather than as a frozen label.
  const anyProducing = (rows || []).some(
    (r) => r.processing_status === "processing" || r.queue_state,
  );

  useEffect(() => {
    if (!adminPassword) return;
    // Poll while anything is actively producing so the badge clears
    // automatically when the background worker finishes. The hook
    // handles the initial page-1 fetch on mount itself; we just need
    // to keep it warm.
    const id = setInterval(refreshAll, anyProducing ? 3000 : 8000);
    return () => clearInterval(id);
  }, [adminPassword, refreshAll, anyProducing]);

  function handleDelete(row) {
    setConfirmBox({
      title: `Delete upload #${row.id}?`,
      body: "This removes the source video(s) and anything produced from "
        + "them. It cannot be undone.",
      confirmLabel: "Delete upload",
      onConfirm: () => {
        // Dialog down and row gone, both on the click. A delete that
        // waits for the server to answer before anything moves reads as
        // a dead button -- and the answer is never in doubt from the
        // operator's side. If it does fail, the refresh below puts the
        // row back and the banner says why.
        setConfirmBox(null);
        dropRow(row.id);
        (async () => {
          try {
            await api.deleteLongUpload(adminPassword, row.id);
          } catch (e) {
            setError(e.message);
          }
          await refreshAll();
        })();
      },
    });
  }

  function handleBroadcast(row) {
    // Toggle the produced clip's is_highlight flag so it shows up on
    // the Broadcast channel. The Production card surfaces the latest
    // produced_clip; that's what we operate on.
    const clip = row.produced_clips?.[0];
    if (!clip) return;
    const next = !clip.is_highlight;
    // FLIP THE LABEL ON THE CLICK. This is a boolean on one row: there
    // is nothing to compute and nothing to wait for, and the button used
    // to sit unchanged for a couple of seconds while a POST and a full
    // list refetch went by. Set it here, send it, and put it back only
    // if the server disagrees.
    const flip = (v) => patchRow(row.id, {
      produced_clips: (row.produced_clips || []).map(
        (c) => (c.id === clip.id ? { ...c, is_highlight: v } : c),
      ),
    });
    flip(next);
    (async () => {
      try {
        await api.setClipBroadcast(adminPassword, clip.id, next);
      } catch (e) {
        flip(!next);
        setError(e.message);
        return;
      }
      // Quietly reconcile: the Broadcast channel is shared state and
      // this row is not the only thing that can change it.
      refreshAll();
    })();
  }

  function handleEdit(row, opts = {}) {
    // Opens the EditWizard. `focusClipId` edits ONE produced clip and
    // hides the swing selector; `startNewSwing` opens on a fresh blank
    // swing. Neither is a second wizard -- both are the same component
    // told where to land, which is why an edit made from a clip and an
    // edit made from the row cannot drift apart.
    setEditingRow({
      ...row,
      __focusClipId: opts.focusClipId ?? null,
      __startNewSwing: !!opts.startNewSwing,
    });
  }

  async function openGreenCal(row) {
    // THE SAME CALIBRATOR THE WIZARD OPENS, reachable without one. It
    // is a property of the two CAMERAS -- clicking the same ground
    // features in a tee frame and a green frame fits one homography per
    // hole, and every swing those cameras ever record is aimed by it --
    // so needing to be part-way through editing a swing to get at it was
    // an accident of where the button happened to live.
    //
    // The frames are only backdrops to click on, so any pair will do:
    // the first produced swing's impact frame when there is one, else
    // frame 0 on both.
    const sw = (row.edit_metrics?.swings || [])[0] || row.edit_metrics || {};
    const teeF = sw.impact_frame ?? 0;
    const greenF = sw.landing_frame ?? 0;
    setGreenCal({ uploadId: row.id, loading: true });
    try {
      const [t, g, vm] = await Promise.all([
        api.getLongUploadFrame(adminPassword, row.id, teeF, "tee"),
        api.getLongUploadFrame(adminPassword, row.id, greenF, "green",
                               sw.impact_frame ?? null),
        api.getViewMap(adminPassword, row.id).catch(() => null),
      ]);
      setGreenCal({
        uploadId: row.id,
        tee: { ...t, frame: teeF },
        green: { ...g, frame: greenF },
        existing: vm?.view_map || null,
        mismatch: vm?.mismatch || null,
        // Worth saying out loud: this is not saved against this upload,
        // so it is about to change every swing those two cameras
        // record -- not only the one the operator opened it from.
        scope: vm?.course_name
          ? `${vm.course_name} · hole ${vm.hole}`
            + (vm.key_reason ? ` · filed by ${vm.key_reason}` : "")
          : null,
      });
    } catch (e) {
      setGreenCal({ uploadId: row.id, error: e?.message || String(e) });
    }
  }

  async function handleProduce(row) {
    // Stub: kicks off a default reprocess on the existing row. Matches
    // the auto-produce defaults used by /clips/quick-upload.
    busySinceRef.current = Date.now();
    setBusyId(row.id);
    // Same as the wizard's Produce: say it in the row, so the card is
    // greyed by the render that follows the click and not by a later
    // round trip.
    patchRow(row.id, {
      processing_status: "processing",
      produce_stage: null,
      produce_done: 0,
      produce_total: 0,
    });
    try {
      const fd = new FormData();
      fd.append("segments", "[]");
      fd.append("auto_detect_swings", "true");
      fd.append("starting_hole", "1");
      await api.reprocessLongUpload(adminPassword, row.id, fd);
      await refreshAll();
      // Busy is NOT cleared here. The POST returns before the worker flips
      // processing_status, so clearing it in a finally left a gap where the
      // row un-greyed and the button looked unpressed, then greyed again a
      // few seconds later. The effect below hands over once the server's
      // own status says processing.
    } catch (e) {
      setError(e.message);
      setBusyId((cur) => (cur === row.id ? null : cur));
    }
  }

  // A label outlives nothing: when the card stops being busy it stops
  // saying anything.
  useEffect(() => {
    if (busyId == null) setBusyLabel(null);
  }, [busyId]);

  // Hand the greyed-out state from the optimistic flag to the server's
  // status, without a gap between them.
  useEffect(() => {
    if (busyId == null) return;
    // A delete is not a produce: the row's processing_status is whatever
    // the last produce left behind, so both hand-offs below would fire
    // immediately and un-grey a card that is still mid-delete. The
    // handler clears it itself when the request comes back.
    if (busyLabel) return;
    const r = (rows || []).find((x) => x.id === busyId);
    if (!r) return;
    // NOT the terminal states on their own. The row is ALREADY
    // "completed" from its last run at the moment you click Re-Produce,
    // so clearing on that fired on the very next render and un-greyed
    // instantly -- the flicker.
    //
    // Nor on "processing" any more. Handing the grey over to the
    // server's status the moment it says processing left a window where
    // a refresh that landed BEFORE the worker claimed the row -- the
    // one fired as the wizard closes -- reported the old "completed"
    // and un-greyed the card until the next poll. The flag costs
    // nothing while the row agrees with it, so it stays up until the
    // run is actually done.
    // A run can also FINISH before any poll catches it mid-flight.
    // A wizard produce is a few seconds; if every refresh lands either
    // side of that window, the row goes straight completed -> completed
    // and the flag above never hands over, leaving the card greyed until
    // the 180s failsafe. So a completion stamped AFTER we went busy also
    // releases it -- that is this run finishing, not the previous one.
    const done = r.processing_completed_at
      ? Date.parse(r.processing_completed_at)
      : NaN;
    if (Number.isFinite(done) && done >= busySinceRef.current) {
      setBusyId(null);
    }
  }, [rows, busyId, busyLabel]);

  // Failsafe: never leave a card greyed for good if the worker dies before
  // it claims the row, or crashes without setting a status.
  useEffect(() => {
    if (busyId == null) return undefined;
    const t = setTimeout(() => setBusyId(null), 180_000);
    return () => clearTimeout(t);
  }, [busyId]);

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
        <Link to="/admin/cameras">Cameras</Link>
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

      <div
        className="card"
        style={{
          display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
          padding: "10px 14px", marginBottom: 12,
        }}
      >
        <input
          type="search"
          value={courseSearch}
          onChange={(e) => setCourseSearch(e.target.value)}
          placeholder="Search by course…"
          className="small"
          style={{ width: "auto", flex: "1 1 220px", maxWidth: 320 }}
        />
        <label style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
          <span className="tiny upper muted">Sort</span>
          <select
            value={sortKey}
            onChange={(e) => setSortKey(e.target.value)}
            className="small"
            style={{ width: "auto" }}
          >
            <option value="created_desc">Upload date — newest</option>
            <option value="created_asc">Upload date — oldest</option>
            <option value="course_asc">Course — A→Z</option>
            <option value="course_desc">Course — Z→A</option>
          </select>
        </label>
        {courseQuery ? (
          <span className="tiny muted">filtering “{courseQuery}”</span>
        ) : null}
      </div>

      {visibleRows && visibleRows.length > 0 && (
        <div
          className="card"
          style={{
            display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap",
            padding: "10px 14px", marginBottom: 12,
          }}
        >
          <label
            style={{ display: "inline-flex", alignItems: "center", gap: 6, cursor: "pointer" }}
            title="Select every upload loaded on this page"
          >
            <input
              type="checkbox"
              checked={
                selectedIds.size > 0 && selectedIds.size === visibleRows.length
              }
              ref={(el) => {
                if (el) {
                  el.indeterminate =
                    selectedIds.size > 0 &&
                    selectedIds.size < visibleRows.length;
                }
              }}
              onChange={(e) => (e.target.checked ? selectAllVisible() : clearSelection())}
              style={{ width: 18, height: 18, cursor: "pointer" }}
            />
            <span className="small">Select all loaded</span>
          </label>
          <span className="small muted">{selectedIds.size} selected</span>
          {mirror.configured && (
            <button
              className="small"
              onClick={handlePullFromProd}
              disabled={mirror.running}
              style={{ width: "auto" }}
              title="Import new camera clips from the production site"
            >
              {mirror.running
                ? `Pulling ${mirror.done}/${mirror.total || "?"}…`
                : "⬇ Pull new from prod"}
            </button>
          )}
          {!mirror.running && mirror.finished_at && (mirror.done || mirror.failed) ? (
            <span className="small muted">
              imported {mirror.done}
              {mirror.failed ? `, ${mirror.failed} failed` : ""}
            </span>
          ) : null}
          <div style={{ flex: 1 }} />
          {selectedIds.size > 0 && (
            <button
              className="small ghost"
              onClick={clearSelection}
              disabled={bulkBusy}
              style={{ width: "auto" }}
            >
              Clear
            </button>
          )}
          <button
            className="small danger"
            onClick={handleBulkDelete}
            disabled={bulkBusy || selectedIds.size === 0}
            style={{ width: "auto" }}
          >
            {bulkBusy ? "Deleting…" : `Delete selected (${selectedIds.size})`}
          </button>
        </div>
      )}

      {/* Camera-event-sourced uploads now flow through the long-upload
          pipeline (see backend/_process_camera_event_job), so they
          render in the same list below — no separate CameraEventCard
          section needed. The "From Camera #N" badge on each card
          identifies Pi-sourced rows. */}

      {(rows !== null) && (rows?.length || 0) === 0 && (
        <div className="card muted center" style={{ padding: 40 }}>
          {courseQuery ? (
            <>No uploads match “{courseQuery}”.</>
          ) : (
            <>
              Nothing in the production queue yet. Either{" "}
              <Link to="/admin/upload-videos">upload a video</Link> or wait for
              the on-course cameras to capture a swing.
            </>
          )}
        </div>
      )}

      {allHidden && (
        <div className="card muted center" style={{ padding: 40 }}>
          {rows.length} upload{rows.length === 1 ? "" : "s"} loaded, but their
          source files are missing — hidden. Scroll for more.
        </div>
      )}

      {rows === null && (
        <div className="card">
          <div className="shimmer" style={{ height: 200 }} />
        </div>
      )}

      {visibleRows?.map((row) => {
        const state = uploadState(row, busyId === row.id);
        const greyed = state === "processing";
        const busy = busyId === row.id;
        // THE PREVIEW HAS TO BE THE NEW ONE THE MOMENT THE GREY LIFTS.
        // A re-produce can hand back the same URL with different bytes,
        // and the browser will happily keep showing the picture it
        // already has -- so the card un-greyed onto the OLD video and
        // only caught up whenever the cache felt like it. Stamping the
        // run's completion into the URL makes it a different resource
        // for every run.
        const producedClips = bustClips(
          row.produced_clips, row.processing_completed_at,
        );
        return (
          // Wrapper exists so the status overlay can sit OUTSIDE the
          // opacity: a child of the greyed card would be dimmed to 0.6
          // too, and the whole point is that it stays readable.
          <div key={row.id} style={{ position: "relative", marginBottom: 12 }}>
          <div
            className="card"
            style={{
              marginBottom: 0,
              opacity: greyed ? 0.6 : 1,
              position: "relative",
              outline: selectedIds.has(row.id)
                ? "2px solid var(--primary, #22c55e)"
                : "none",
              outlineOffset: 2,
            }}
          >
            <div
              className="row"
              style={{
                gap: 10, flexWrap: "wrap", alignItems: "center",
                marginBottom: 10,
              }}
            >
              <input
                type="checkbox"
                checked={selectedIds.has(row.id)}
                onChange={() => toggleSelected(row.id)}
                onClick={(e) => e.stopPropagation()}
                title="Select for bulk delete"
                style={{ width: 18, height: 18, cursor: "pointer", flexShrink: 0 }}
              />
              <h4 style={{ margin: 0 }}>
                #{row.id} · {row.course_name || `course ${row.course_id}`}
                {row.source?.kind === "camera" && row.source?.hole_number != null
                  ? ` · hole ${row.source.hole_number}`
                  : ""}
              </h4>
              {row.source?.kind === "camera" ? (
                <span
                  className="small"
                  style={{
                    padding: "2px 8px",
                    borderRadius: 999,
                    background: "rgba(56, 132, 255, 0.12)",
                    border: "1px solid rgba(56, 132, 255, 0.4)",
                  }}
                  title={
                    row.source.triggered_at
                      ? `Triggered ${fmtDateTime(row.source.triggered_at)}`
                      : undefined
                  }
                >
                  From {row.source.camera_name
                    || `Camera #${row.source.camera_id}`}
                </span>
              ) : null}
              <span className="small muted">·</span>
              <span className="small muted">
                {row.source?.kind === "camera"
                  ? `Captured ${fmtDateTime(row.source.triggered_at || row.created_at)}`
                  : `Uploaded ${fmtDateTime(row.created_at)}`}
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
                  recordingStartedAt={row.tee_recording_started_at}
                  onOpenViewer={openViewer}
                  footer={(
                    <button
                      className="small ghost"
                      style={{ width: "100%" }}
                      onClick={() => setHitArea({ uploadId: row.id })}
                      title="View and draw the hitting area for this hole and day — the box every ball search on this camera is restricted to. Tilt it to match the tee deck."
                    >
                      ⬛ Hitting area
                    </button>
                  )}
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
                  recordingStartedAt={row.dual_camera ? row.green_recording_started_at : null}
                  onOpenViewer={openViewer}
                  footer={(
                    <button
                      className="small ghost"
                      style={{ width: "100%" }}
                      disabled={!row.dual_camera}
                      onClick={() => setPinModal({ row })}
                      title={row.dual_camera
                        ? "Mark the base of the flagstick on the green camera. It is dated with THIS swing's capture time and carries forward to every later swing until it is marked again — the calibration itself is done once per camera pair, on the Cameras page."
                        : "No green camera on this upload"}
                    >
                      ⚑ Set flagstick
                    </button>
                  )}
                />
                <ProducedTile
                  clips={producedClips}
                  swings={row.edit_metrics?.swings}
                  onOpenViewer={openViewer}
                  // EDIT OPENS THE PLOT MAP NOW. The wizard's field
                  // list lives there, and it is the screen with the
                  // pictures; keeping a second dialog that showed the
                  // same five numbers against one still frame was the
                  // thing being merged away.
                  onEditClip={(clip, swingRec, idx) => {
                    const arr = row.edit_metrics?.swings || [];
                    const byClip = arr.findIndex(
                      (sw) => sw?.clip_id != null && sw.clip_id === clip?.id);
                    setPlotModal({
                      row,
                      swingPos: byClip >= 0
                        ? byClip
                        : (swingRec
                          ? arr.findIndex((sw) => sw === swingRec)
                          : idx) || 0,
                    });
                  }}
                  // ADD OPENS THE SAME MAP, EMPTY. Pointing it one past
                  // the end of the swings array is what makes every
                  // field blank -- there is no swing at that position to
                  // read a tee spot or an impact frame from -- and the
                  // save appends instead of patching. Same screen, same
                  // gestures, nothing held over from the last clip.
                  onAddClip={() => setPlotModal({ row, addNew: true })}
                  onDeleteClip={(clip, clipIdx) => {
                    const label = clip.hole_number != null
                      ? `clip ${clipIdx + 1} (hole ${clip.hole_number})`
                      : `clip ${clipIdx + 1}`;
                    setConfirmBox({
                      title: `Delete produced ${label}?`,
                      body:
                        "The video, its files and the swing it was cut " +
                        "from are all removed — so a re-produce will not " +
                        "bring it back. The raw upload and the other " +
                        "clips stay.",
                      confirmLabel: "Delete clip",
                      onConfirm: async () => {
                        // Dialog down, card greyed, THEN the request —
                        // same order as the wizard's Produce, and for the
                        // same reason: the operator has to be able to see
                        // the state that the click just set.
                        setConfirmBox(null);
                        busySinceRef.current = Date.now();
                        setBusyLabel("Deleting the clip…");
                        setBusyId(row.id);
                        // ...and the video goes with it. Leaving the
                        // clip on the row until the refresh answered
                        // meant the card un-greyed with the deleted
                        // video still on screen, which then vanished a
                        // few seconds later on its own.
                        {
                          // The swing goes with the clip, server-side.
                          // Mirror it here by the same rule (clip_id
                          // only) so the wizard cannot be opened on a
                          // swing that is already gone.
                          const _sw = row.edit_metrics?.swings;
                          const _dropSwing =
                            Array.isArray(_sw)
                            && _sw.some((s) => s?.clip_id === clip.id);
                          patchRow(row.id, {
                            produced_clips: (row.produced_clips || [])
                              .filter((c) => c.id !== clip.id),
                            ...(_dropSwing
                              ? {
                                  edit_metrics: {
                                    ...row.edit_metrics,
                                    swings: _sw.filter(
                                      (s) => s?.clip_id !== clip.id,
                                    ),
                                  },
                                }
                              : {}),
                          });
                        }
                        try {
                          await api.deleteClip(adminPassword, clip.id);
                        } catch (e) {
                          // Already gone is the outcome we wanted.
                          if (!/^404/.test(e?.message || "")) {
                            setError(e.message);
                          }
                        }
                        try {
                          await refreshAll();
                        } finally {
                          setBusyLabel(null);
                          setBusyId((cur) => (cur === row.id ? null : cur));
                        }
                      },
                    });
                  }}
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
                {/* Edit moved under Produced Video, where the clip it
                    edits is on screen and can be handed straight to it.
                    Add clip is beside it, for the swing the detector
                    missed. */}
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
                {SHOW_LEGACY_DEBUG && produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleProduceDebug(row)}
                        title="Dev: produce AND run a per-swing diagnostic — classical-CV heatmap vs AI tracer"
                  >
                    🐞 Debug
                  </button>
                )}
                {SHOW_LEGACY_DEBUG && produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleDebug2(row)}
                        title="Dev: pose candidates → impact + ball from the club arc → AI judge → windowed MOG2 heat → chain walked up from the ball. Shows every stage."
                  >
                    🔬 Debug2
                  </button>
                )}
                {produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleDebug3(row)}
                        title="Dev: MOG2 per frame → drop the golfer → keep ball-sized blobs → link across frames → RANSAC parabola. A different method from Debug2; shows every stage."
                  >
                    🧿 Debug3
                  </button>
                )}
                {produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleSwingTest(row)}
                        title="Dev: the ball-departure detector alone — where it looked, whether it found a ball, and the exact frame that ball left."
                  >
                    ⛳ Swing test
                  </button>
                )}
                {/* DEV ONLY, and `mirror.configured` is what says so:
                    only the dev backend has a prod to pull from. A bare
                    scan produces nothing -- it is a thing to argue with
                    a produce about, not a thing to hand an operator at
                    a course. */}
                {produceDebug.enabled && mirror.configured && (
                  <button
                    className="small ghost"
                    onClick={() => handleBallScan(row)}
                    title="Dev: every resting-ball candidate in the tee box — where, first frame seen, last frame before it went, with pictures. People are masked out; motion is not used, because a resting ball does not move."
                  >
                    ⚪ Scan for ball
                  </button>
                )}
                {/* BOTH PLACES. This runs exactly what Produce runs --
                    same scan, same renderer, same clips committed -- and
                    the only difference is that it shows its working:
                    which candidates it found, what it decided about
                    each, and where the time went. On the deployment
                    whose runs are the ones behaving oddly, that is not
                    a debug toy, it is the only view of what happened. */}
                {/* NOT BEHIND THE DEBUG FLAG. It was, and that flag is
                    off in production -- so asking for this button there
                    got nothing. It is no longer a debug tool: it runs
                    exactly what Produce runs and commits the same clips,
                    and the only thing it adds is a record of what it
                    did. Gating the explanation behind a setting that is
                    off wherever the explaining is needed is backwards. */}
                <button
                  className="small ghost"
                  onClick={() => handleBallScanProduce(row)}
                  disabled={greyed || busy}
                  title="What Produce runs, with the report attached — scan for resting balls, then trace every candidate that sat 7s or longer, straight from the measured rest position and the frame it went. Same clips, plus a per-stage timing breakdown."
                >
                  ⚪▶ Scan &amp; produce
                </button>
                {(() => {
                  // Broadcast button is enabled when the wizard has
                  // produced a clip on this upload. Toggles
                  // is_highlight on the most-recent produced clip.
                  const lastClip = row.produced_clips?.[0];
                  const onBroadcast = !!lastClip?.is_highlight;
                  return (
                    <button
                      className={onBroadcast ? "small" : "small ghost"}
                      onClick={() => handleBroadcast(row)}
                      disabled={greyed || busy || !lastClip}
                      title={lastClip
                        ? (onBroadcast
                          ? "Remove from the Broadcast channel"
                          : "Send the produced clip to the Broadcast channel")
                        : "No produced clip yet"}
                    >
                      {onBroadcast ? "On Broadcast" : "Broadcast"}
                    </button>
                  );
                })()}
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
          <ProduceStatusOverlay
            row={row}
            greyed={greyed}
            override={busy ? busyLabel : null}
          />
          </div>
        );
      })}

      {rows && rows.length > 0 && (
        /* A BUTTON, not only a scroll trigger. Auto-load is a
           convenience and it is at the mercy of the browser deciding
           the sentinel moved; this is the control that always works,
           and it says how many uploads are actually on screen so
           "nothing happened" and "nothing more to load" cannot be
           confused for each other. */
        <div
          ref={uploadsList.sentinelRef}
          className="muted center small"
          style={{
            padding: 16, display: "flex", flexDirection: "column",
            alignItems: "center", gap: 8,
          }}
        >
          {uploadsList.hasMore ? (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto", minWidth: 200 }}
              disabled={uploadsList.loadingMore}
              onClick={() => uploadsList.loadMore()}
            >
              {uploadsList.loadingMore
                ? "Loading more uploads…"
                : "Load 25 more uploads"}
            </button>
          ) : (
            <span>End of long uploads</span>
          )}
          <span className="tiny">
            {(uploadsList.items?.length ?? 0)} loaded
            {rows.length !== (uploadsList.items?.length ?? 0)
              ? ` · ${rows.length} shown` : ""}
          </span>
        </div>
      )}

      <VideoLightbox
        url={viewer?.url}
        title={viewer?.title}
        startedAt={viewer?.startedAt}
        startedApprox={viewer?.startedApprox}
        fps={viewer?.fps}
        onClose={() => setViewer(null)}
      />

      {debugModal && (
        <ProduceDebugModal
          data={debugModal}
          adminPassword={adminPassword}
          onRerun={() => rerunBallScan(debugModal.uploadId)}
          onClose={() => setDebugModal(null)}
        />
      )}

      {editingRow && (
        <Boundary name="Edit wizard"
                  onClose={() => { setEditingRow(null); refreshAll(); }}>
          <EditWizard
            row={editingRow}
            focusClipId={editingRow.__focusClipId ?? null}
            startNewSwing={!!editingRow.__startNewSwing}
            adminPassword={adminPassword}
            onClose={() => { setEditingRow(null); refreshAll(); }}
            onSaved={refreshAll}
            onProducing={(on) => {
              if (on === false) {
                // Nothing was queued: put the row back the way it was.
                patchRow(editingRow.id, {
                  processing_status: editingRow.processing_status,
                  produce_stage: editingRow.produce_stage ?? null,
                });
                setBusyId((cur) => (cur === editingRow.id ? null : cur));
                return;
              }
              busySinceRef.current = Date.now();
              setBusyId(editingRow.id);
              // ...and say it in the row, which is what the card actually
              // renders from. The server claims the row before its POST
              // returns, so this is what the next refresh reports anyway
              // -- we are only refusing to wait for it.
              patchRow(editingRow.id, {
                processing_status: "processing",
                produce_stage: null,
                produce_done: 0,
                produce_total: 0,
              });
            }}
            onProduceError={(msg) => setError(msg)}
          />
        </Boundary>
      )}

      {plotModal && (
        <Boundary name="Click-to-plot"
                  onClose={() => setPlotModal(null)}>
          <ClickToPlotModal
            row={plotModal.row}
            swingPos={plotModal.swingPos ?? null}
            addNew={!!plotModal.addNew}
            adminPassword={adminPassword}
            onClose={() => setPlotModal(null)}
            // The save runs after this modal has closed, so its progress
            // belongs on the card: greyed, named stage, held until the new
            // video is actually in.
            onBackground={(msg) => {
              const id = plotModal.row.id;
              busySinceRef.current = Date.now();
              setBusyLabel(msg);
              setBusyId(id);
              patchRow(id, {
                processing_status: "processing",
                produce_stage: null,
                produce_done: 0,
                produce_total: 0,
              });
            }}
            onDone={async (ok, err) => {
              if (!ok) setError(err);
              // Only now: the card stays greyed until the re-rendered
              // video is on the row, not until the first call returns.
              await refreshAll();
              setBusyLabel(null);
              setBusyId((cur) => (cur === plotModal.row.id ? null : cur));
            }}
          />
        </Boundary>
      )}

      {d2 && <Debug2Modal state={d2} onClose={() => setD2(null)} />}
      {d3 && <Debug3Modal state={d3} onClose={() => setD3(null)} />}
      {ballScan && (
        <BallScanModal state={ballScan} adminPassword={adminPassword}
                       onClose={() => setBallScan(null)} />
      )}
      {hitArea && (
        <HittingAreaModal state={hitArea} adminPassword={adminPassword}
                          onClose={() => setHitArea(null)} />
      )}
      {greenCal && (greenCal.loading || greenCal.error ? (
        <div
          role="dialog"
          onClick={() => setGreenCal(null)}
          style={{
            position: "fixed", inset: 0, zIndex: 1200,
            background: "rgba(0,0,0,0.75)", display: "flex",
            alignItems: "center", justifyContent: "center", padding: 16,
          }}
        >
          <div className="card" onClick={(e) => e.stopPropagation()}
               style={{ margin: 0, padding: 18, maxWidth: 460 }}>
            <div className="row" style={{ justifyContent: "space-between",
                                          gap: 12 }}>
              <b>⊹ Calibrate green</b>
              <button className="btn ghost" style={{ width: "auto" }}
                      onClick={() => setGreenCal(null)}>Close ✕</button>
            </div>
            <div style={{ marginTop: 10 }}>
              {greenCal.error
                ? <div className="err-text small">{greenCal.error}</div>
                : <div className="small">Fetching a frame from each camera…</div>}
            </div>
          </div>
        </div>
      ) : (
        <ViewMapModal
          uploadId={greenCal.uploadId}
          adminPassword={adminPassword}
          teeFrame={greenCal.tee}
          greenFrame={greenCal.green}
          existing={greenCal.existing}
          mismatch={greenCal.mismatch}
          scope={greenCal.scope}
          onClose={() => setGreenCal(null)}
          onSaved={() => { setGreenCal(null); refreshAll(); }}
        />
      ))}
      {pinModal && (
        <FlagstickModal
          row={pinModal.row}
          adminPassword={adminPassword}
          onClose={() => setPinModal(null)}
          onSaved={() => refreshAll()}
        />
      )}
      {swingTest && (
        <SwingTestModal
          state={swingTest}
          adminPassword={adminPassword}
          onClose={() => setSwingTest(null)}
          onRerun={(uploadId) => handleSwingTest({ id: uploadId })}
        />
      )}

      <ConfirmDialog
        open={!!confirmBox}
        title={confirmBox?.title}
        body={confirmBox?.body}
        confirmLabel={confirmBox?.confirmLabel}
        onConfirm={confirmBox?.onConfirm}
        onCancel={() => setConfirmBox(null)}
      />
    </div>
  );
}
