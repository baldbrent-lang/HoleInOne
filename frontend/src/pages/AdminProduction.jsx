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
import { useCallback, useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";
import { useInfiniteList } from "../hooks/useInfiniteList.js";
import { parseApiDate } from "../time.js";

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
function ProduceStatusOverlay({ row, greyed }) {
  const queued = row.queue_state === "queued";
  if (!greyed && !queued) return null;

  const stage = row.produce_stage;
  const total = row.produce_total || 0;
  const done = row.produce_done || 0;
  // Only a per-candidate stage carries a meaningful total; the one-off
  // stages report 0 and get an indeterminate bar rather than a fake 0%.
  const pct = total > 0 ? Math.min(100, Math.round((done / total) * 100)) : null;

  const label = queued
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
                     videoUrl, recordingStartedAt, onOpenViewer }) {
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
        onClick={videoUrl ? () => onOpenViewer(videoUrl, label, recordingStartedAt, fps) : undefined}
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

function ProducedTile({ clips, swings, onOpenViewer, onClickToPlot, onDeleteClip }) {
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
      {has && (clips.length > 1 || onDeleteClip) && (
        <div
          style={{
            display: "flex", alignItems: "center", justifyContent: "center",
            gap: 8, marginTop: 4,
          }}
        >
          {clips.length > 1 && (
            <button
              type="button"
              className="ghost"
              style={{ width: "auto", padding: "1px 8px", fontSize: "0.9rem" }}
              onClick={() => nav(-1)}
              title="Previous clip"
            >
              ◀
            </button>
          )}
          <span className="tiny">
            clip {idx + 1}/{clips.length}
            {cur?.hole_number != null ? ` · hole ${cur.hole_number}` : ""}
          </span>
          {clips.length > 1 && (
            <button
              type="button"
              className="ghost"
              style={{ width: "auto", padding: "1px 8px", fontSize: "0.9rem" }}
              onClick={() => nav(1)}
              title="Next clip"
            >
              ▶
            </button>
          )}
          {onDeleteClip && cur && (
            <button
              type="button"
              className="ghost"
              style={{
                width: "auto", padding: "1px 6px", fontSize: "0.9rem",
                color: "var(--danger)",
              }}
              onClick={() => onDeleteClip(cur, idx)}
              title={`Delete this produced clip (clip ${idx + 1}${
                cur?.hole_number != null ? ` · hole ${cur.hole_number}` : ""
              }) and its files. The raw upload and other clips stay; Re-Produce can recreate it.`}
            >
              🗑
            </button>
          )}
        </div>
      )}
      {curSwing && (
        <button
          type="button"
          className="small"
          onClick={() => {
            if (onClickToPlot) {
              onClickToPlot(curSwing.idx ?? idx);
            } else if (curSwing.mog2_overlay_url) {
              window.open(curSwing.mog2_overlay_url, "_blank");
            }
          }}
          style={{
            display: "block", width: "100%", textAlign: "center",
            marginTop: 4, padding: "3px 8px", borderRadius: 6,
            border: "1px solid rgba(230,126,34,0.5)",
            background: "transparent", cursor: "pointer",
            // Base button CSS is white-on-green; on a transparent
            // background the white label vanishes.
            color: "var(--ink)",
          }}
          title={
            "Open the click-to-plot editor — the motion heat zoomable " +
            "with every timed dot clickable, one click marks the ball " +
            "for that dot's frame." +
            (curSwing.mog2_stats
              ? ` AI picks: ${curSwing.mog2_stats.n_ai ?? "?"} · MOG2 dots: ` +
                `${curSwing.mog2_stats.n_cv ?? "?"} · matched: ` +
                `${curSwing.mog2_stats.n_matched ?? "?"} · added to arc: ` +
                `${curSwing.mog2_stats.n_added ?? 0}` +
                (curSwing.mog2_stats.n_added_descent === 0 &&
                 curSwing.mog2_stats.descent_debug
                  ? ` · descent 0 because: ` +
                    `${curSwing.mog2_stats.descent_debug.stopped || "?"} ` +
                    `(seen ${curSwing.mog2_stats.descent_debug.seen ?? 0}, ` +
                    `step-rejected ${curSwing.mog2_stats.descent_debug.step_rej ?? 0}, ` +
                    `corridor-rejected ${curSwing.mog2_stats.descent_debug.corr_rej ?? 0})`
                  : "")
              : "")
          }
        >
          🖱 Click-to-plot
          {curSwing.mog2_stats?.n_added > 0
            ? ` (+${curSwing.mog2_stats.n_added} added)`
            : ""}
        </button>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 2 }}>
        <MetaRow k="Clips" v={has ? clips.length : ""} />
        <MetaRow k="Aces" v={has ? aces : ""} />
        <MetaRow k="Holes" v={holes.length ? holes.join(", ") : ""} />
      </div>
    </div>
  );
}

// Frames of clickable detections either side of impact. A few frames of
// lead-in covers an impact frame estimated slightly late (the assumed-
// impact path pins it to the pose peak, which can sit a frame or two off
// the strike); 100 frames after is ~2s of flight at 50fps, by which point
// the ball is long gone and every remaining dot is the golfer walking off,
// a cart, or wind in the trees.
const PLOT_WINDOW_PRE = 5;
const PLOT_WINDOW_POST = 100;

/**
 * Standalone click-to-plot modal, opened from a production card's
 * 🖱 Click-to-plot button. Big zoomable heat view with every timed dot
 * clickable; Save & close bakes the queued picks into the swing's
 * tracer (cv2 fast render, no AI), re-finalizes the video with the
 * saved graphics, and commits it to Produced Clips — the same
 * pipeline as the wizard's Produce, minus the wizard.
 */
function ClickToPlotModal({ row, swingPos, adminPassword, onClose, onSaved }) {
  const swings = row.edit_metrics?.swings || [];
  const swing = swings[swingPos] || {};
  // FLIGHT WINDOW. Pre-swing motion (waggle, address, shadow) is noise on
  // this map, and so is everything long after the ball has gone — the
  // golfer walking off, a cart, wind in the trees. Both crowd the map with
  // dots that can only ever be wrong picks. Show impact-5 (a few frames of
  // lead-in, in case impact is estimated a touch late) through impact+100,
  // which at 50fps is two seconds of flight.
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
  const [placingBall, setPlacingBall] = useState(false);
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
  const [busy, setBusy] = useState(false);
  const [busyMsg, setBusyMsg] = useState(null);
  const [error, setError] = useState(null);
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
  // Resolve this swing's produced clip by IDENTITY (clip_id) first —
  // positional lookup goes stale as soon as a clip is deleted.
  const clipForSwing =
    (row.produced_clips || []).find(
      (c) => swing.clip_id != null && c.id === swing.clip_id,
    ) ?? row.produced_clips?.[swingPos];
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
    setImpactFrame(swing.impact_frame ?? null);
    setBallAtRest(swing.ball ?? null);
    setPlacingBall(false);
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
      overrides.length === 0 && cleared.length === 0
      && !movedImpact && !movedBall
    ) {
      onClose();
      return;
    }
    setBusy(true);
    setError(null);
    try {
      // 1. Bake the picks into the swing's track (cv2 only, no AI).
      setBusyMsg("Re-rendering tracer…");
      const hasWindow =
        swing.start_frame != null && swing.end_frame != null;
      const fast = await api.renderWizardTracerFast(adminPassword, row.id, {
        manual_positions: overrides,
        cleared_frames: cleared,
        base_track_frames: swing.ball_track_frames || [],
        impact_frame: impactFrame ?? null,
        ball_at_rest: ballAtRest || null,
        target: swing.target || null,
        render_window: hasWindow
          ? { start_frame: swing.start_frame, end_frame: swing.end_frame }
          : null,
      });
      let nextSwings = swings.map((s, i) =>
        i === swingPos
          ? {
              ...s,
              impact_frame: impactFrame ?? s.impact_frame,
              ...(ballAtRest
                // ball_manual marks it operator-placed; the produce
                // worker checks that flag before writing a detected rest
                // position, so a re-produce cannot move it back.
                ? { ball: ballAtRest, ball_manual: true }
                : {}),
              tracer_url: fast.tracer_url,
              ball_track_frames: fast.ball_track_frames || [],
            }
          : s
      );
      await api.saveEditMetrics(adminPassword, row.id, { swings: nextSwings });
      // 2. Re-finalize with the swing's saved graphics + frame window.
      // `swing` gives this swing its OWN final file so it can't clobber
      // the video behind another swing's committed clip.
      setBusyMsg("Applying graphics…");
      const fin = await api.finalizeWizardVideo(adminPassword, row.id, {
        player_name: swing.finalized_player_name || "Brent Baldwin",
        hole_number: holeNumber,
        yardage: swing.finalized_yardage ?? null,
        start_frame: swing.start_frame ?? null,
        end_frame: swing.end_frame ?? null,
        cut_frame: swing.cut_frame ?? null,
        // WITHOUT THIS THE CLIP STOPS CUTTING TO THE GREEN. finalize
        // decides the cutover from cut_frame, falling back to
        // impact_frame + 2.5s when there is no explicit cut. Produce
        // never persists a per-swing cut_frame, so that fallback is the
        // only thing that puts the green in — and it needs an impact
        // frame. Omitting it meant _pick_frame fell through to the
        // TOP-LEVEL edit_metrics, which on a multi-swing upload has no
        // impact_frame at all, so cut_src_sec came out None and the
        // finalize silently produced a tee-only clip.
        impact_frame: impactFrame ?? swing.impact_frame ?? null,
        swing: swing.idx ?? swingPos,
      });
      nextSwings = nextSwings.map((s, i) =>
        i === swingPos
          ? {
              ...s,
              finalized_video_url: fin.final_video_url,
              finalized_hole_number: holeNumber,
              finalized_player_name:
                swing.finalized_player_name || "Brent Baldwin",
            }
          : s
      );
      await api.saveEditMetrics(adminPassword, row.id, { swings: nextSwings });
      // 3. Commit to Produced Clips — target THIS swing's clip. Prefer
      // the clip id recorded on the swing (survives clip deletions that
      // shift positions); fall back to position (clip order matches
      // swing order on an untouched row). Without clip_id the backend
      // updates the upload's most recent clip, i.e. some other swing.
      setBusyMsg("Updating Produced Clips…");
      const clipId = swing.clip_id ?? clipForSwing?.id ?? null;
      const committed = await api.commitWizardClip(
        adminPassword, row.id,
        clipId != null ? { clip_id: clipId } : {},
      );
      // Remember which clip this swing committed into so later saves
      // target it directly even after other clips are deleted.
      if (committed?.clip_id != null) {
        const withClip = nextSwings.map((s, i) =>
          i === swingPos ? { ...s, clip_id: committed.clip_id } : s
        );
        await api.saveEditMetrics(adminPassword, row.id, {
          swings: withClip,
        });
      }
      onSaved?.();
      onClose();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
      setBusyMsg(null);
    }
  }

  const { overrides: pendAdd, cleared: pendClear } = pendingChanges();
  const impactMoved =
    impactFrame != null && impactFrame !== (swing.impact_frame ?? null);
  const ballMoved =
    !!ballAtRest &&
    (ballAtRest.x !== (swing.ball?.x ?? null) ||
      ballAtRest.y !== (swing.ball?.y ?? null));
  const nChanged =
    pendAdd.length + pendClear.length + (impactMoved ? 1 : 0) + (ballMoved ? 1 : 0);
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
      onClick={busy ? undefined : onClose}
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
          maxWidth: "min(1500px, 98vw)", width: "100%",
          maxHeight: "96vh", height: "96vh", overflow: "hidden",
          cursor: "default", margin: 0,
          display: "flex", flexDirection: "column",
        }}
      >
        <div
          className="row"
          style={{ alignItems: "center", justifyContent: "space-between", gap: 8, marginBottom: 6 }}
        >
          <div className="small">
            <b>🖱 Click-to-plot</b>
            <span className="muted">
              {" "}· #{row.id} · swing {(swing.idx ?? swingPos) + 1} · hole{" "}
              {holeNumber} · {dots.length} dots
              {denseDots.length > 0 && ` · ${denseDots.length} candidates`}
              {winLo != null && ` · showing f${winLo}–f${winHi}`}
            </span>
          </div>
          <div className="row" style={{ gap: 8, alignItems: "center" }}>
            {busy && busyMsg && (
              <span className="small muted">
                <span className="shimmer" style={{ display: "inline-block", width: 12, height: 12, borderRadius: "50%", marginRight: 6, verticalAlign: "middle" }} />
                {busyMsg}
              </span>
            )}
            {!busy && nChanged > 0 && (
              <span className="small" style={{ color: "var(--emerald-700)" }}>
                {pendAdd.length > 0 && `${pendAdd.length} new`}
                {pendAdd.length > 0 && pendClear.length > 0 && " · "}
                {pendClear.length > 0 && `${pendClear.length} removed`}
              </span>
            )}
            <span
              className="small"
              style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              title="The frame the ball is struck. Drives the flight window shown here and where the rendered tracer line starts."
            >
              <span className="muted">impact</span>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto", padding: "0 6px" }}
                disabled={busy || impactFrame == null}
                onClick={() => setImpactFrame((f) => Math.max(0, (f ?? 0) - 10))}
              >
                −10
              </button>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto", padding: "0 6px" }}
                disabled={busy || impactFrame == null}
                onClick={() => setImpactFrame((f) => Math.max(0, (f ?? 0) - 1))}
              >
                −1
              </button>
              <input
                type="number"
                value={impactFrame ?? ""}
                disabled={busy}
                onChange={(e) => {
                  const n = parseInt(e.target.value, 10);
                  setImpactFrame(Number.isFinite(n) ? Math.max(0, n) : null);
                }}
                style={{ width: 80, textAlign: "center", padding: "1px 4px" }}
              />
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto", padding: "0 6px" }}
                disabled={busy || impactFrame == null}
                onClick={() => setImpactFrame((f) => (f ?? 0) + 1)}
              >
                +1
              </button>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto", padding: "0 6px" }}
                disabled={busy || impactFrame == null}
                onClick={() => setImpactFrame((f) => (f ?? 0) + 10)}
              >
                +10
              </button>
              {firstTrackF != null && firstTrackF !== impactFrame && (
                <button
                  type="button"
                  className="ghost small"
                  style={{ width: "auto", padding: "0 6px" }}
                  disabled={busy}
                  onClick={() => setImpactFrame(firstTrackF)}
                  title={`The earliest point in the saved track is f${firstTrackF}. If the ball is already moving there, that is closer to the real strike than f${impactFrame}.`}
                >
                  ← f{firstTrackF}
                </button>
              )}
              {impactMoved && (
                <span style={{ color: "var(--emerald-700)" }}>
                  (was f{swing.impact_frame})
                </span>
              )}
            </span>
            <button
              type="button"
              className={placingBall ? "small" : "ghost small"}
              style={{ width: "auto" }}
              disabled={busy}
              onClick={() => setPlacingBall((v) => !v)}
              title="Click the map to set where the tracer line STARTS - the ball at impact. This anchors the whole line, so it matters more than any single flight point."
            >
              {placingBall
                ? "click the map…"
                : ballAtRest
                  ? `⦿ start ${ballAtRest.x},${ballAtRest.y}`
                  : "⦿ set tracer start"}
            </button>
            {ballMoved && (
              <span className="small" style={{ color: "var(--emerald-700)" }}>
                start moved
              </span>
            )}
            <button
              type="button"
              className="ghost small"
              onClick={resetMarks}
              style={{ width: "auto" }}
              disabled={busy || nChanged === 0}
              title="Put every point back to what was saved before this modal was opened"
            >
              Reset
            </button>
            <button
              type="button"
              className="ghost small"
              onClick={clearAllMarks}
              style={{ width: "auto" }}
              disabled={busy || Object.keys(marks).length === 0}
              title="Remove ALL plotted points for this swing. Saving then re-renders the tracer with none of them."
            >
              Clear all ({Object.keys(marks).length})
            </button>
            <button
              type="button"
              className="ghost small"
              onClick={onClose}
              style={{ width: "auto" }}
              disabled={busy}
            >
              Cancel
            </button>
            <button
              type="button"
              className="small"
              onClick={saveAndClose}
              style={{ width: "auto" }}
              disabled={busy || nChanged === 0}
              title={
                nChanged === 0
                  ? "Click dots on the heat to add/remove ball points first — green dots are already in the saved track"
                  : "Re-render the tracer with the changes, re-apply graphics, and update Produced Clips"
              }
            >
              {busy ? "Saving…" : "Save & close"}
            </button>
          </div>
        </div>

        <div style={{ flex: 1, minHeight: 0, display: "flex", alignItems: "center", justifyContent: "center" }}>
          {(dots.length > 0 || denseDots.length > 0) && bgUrl ? (
            <PlotHeatCanvas
              bgUrl={bgUrl}
              dots={dots}
              denseDots={denseDots}
              frameW={frameW}
              frameH={frameH}
              marks={marks}
              track={(swing.ball_track_frames || []).filter(
                (r) => r.found && r.x != null && r.y != null,
              )}
              onToggleDot={toggleDot}
              ballXY={ballAtRest}
              placingBall={placingBall}
              onPlaceBall={(pt) => {
                setBallAtRest(pt);
                setPlacingBall(false);
              }}
              scanRegion={async (region) => {
                const out = await api.scanPlotRegion(adminPassword, row.id, {
                  ...region,
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
        {/* The legend is a dozen lines of prose that was pushing the map
            up the screen on every open, long after the operator had read
            it once. Collapsed by default; the map gets the height. */}
        <details style={{ marginTop: 6 }}>
        <summary className="tiny muted" style={{ cursor: "pointer" }}>
          How this map works (legend &amp; shortcuts)
        </summary>
        <div className="tiny muted" style={{ marginTop: 4 }}>
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

        {Object.keys(marks).length > 0 && (
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
                  disabled={busy}
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

        {error && <div className="err-text small" style={{ marginTop: 4 }}>{error}</div>}
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
function EditWizard({ row, adminPassword, onClose, onSaved }) {
  // Hydrate from whatever was already persisted: only auto-detect on
  // the very first Edit. Subsequent re-opens skip the AI call and
  // pre-fill the wizard from row.edit_metrics.
  const saved = row?.edit_metrics || null;

  const [draft, setDraft] = useState(null);
  const [frameDims, setFrameDims] = useState({
    width: null, height: null, totalFrames: null,
  });
  const [running, setRunning] = useState(false);
  const [error, setError] = useState(null);
  const [editing, setEditing] = useState(null);
  // 'metrics' = step 1 (handedness/frames/ball/ROI/target);
  // 'tracer'  = step 2 (rendered tracer + per-frame ball editor).
  const [step, setStep] = useState("metrics");
  const [tracer, setTracer] = useState(null); // { url, frames }
  const [renderingTracer, setRenderingTracer] = useState(false);
  const [tracerError, setTracerError] = useState(null);
  // Tracer engine A/B: "ai" (Claude vision, default) vs "classical"
  // (CV motion + parabola). Switching engines does NOT re-render on
  // Next — the ↻ Render button forces a fresh render with the selection.
  const [tracerEngine, setTracerEngine] = useState("ai");
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
  const [selectedSwing, setSelectedSwing] = useState(0);
  const [detectingSwings, setDetectingSwings] = useState(false);
  // Mirror of selectedSwing readable inside async callbacks without
  // re-creating them, so a render that finishes after the operator
  // switched tabs only updates the display if they're still on that swing.
  const selectedSwingRef = useRef(selectedSwing);
  useEffect(() => { selectedSwingRef.current = selectedSwing; }, [selectedSwing]);
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
    // Start the engine toggle on whatever produced the saved tracer.
    setTracerEngine(s.tracer_engine || "ai");
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
    let cancelled = false;

    // Multi-swing: detect swings (if not cached), pick swing 0,
    // hydrate from edit_metrics.swings[0].
    if (isMulti) {
      const cached = saved?.swings;
      if (Array.isArray(cached) && cached.length > 0) {
        setSwings(cached);
        applySaved(cached[selectedSwing] || cached[0] || {});
        setFrameDims({
          width: saved.frame_width ?? row.tee_width ?? null,
          height: saved.frame_height ?? row.tee_height ?? null,
          totalFrames: row.tee_nb_frames || null,
        });
        return;
      }
      setDetectingSwings(true);
      setError(null);
      api
        .detectSwingsForUpload(adminPassword, row.id)
        .then(async (data) => {
          if (cancelled) return;
          let list = data.swings || [];
          // Auto-detect found nothing — common for clips with no
          // clean audio impact (mic too far, quiet scene, indoor
          // testing). Seed a placeholder swing so the wizard still
          // opens and the operator can pick the address/impact
          // frames manually from the timeline.
          if (list.length === 0) {
            list = [
              {
                idx: 0,
                start_frame: 0,
                end_frame: row.tee_nb_frames || null,
                address_frame: 0,
                impact_frame: 0,
                fps: row.tee_fps || 30,
              },
            ];
          }
          setSwings(list);
          setFrameDims({
            width: row.tee_width || null,
            height: row.tee_height || null,
            totalFrames: row.tee_nb_frames || null,
          });
          applySaved(list[0]);
          try { onSaved?.(); } catch {}
        })
        .catch((e) => { if (!cancelled) setError(e.message); })
        .finally(() => { if (!cancelled) setDetectingSwings(false); });
      return () => { cancelled = true; };
    }

    // Single-swing: hydrate from edit_metrics, else auto-detect.
    if (saved && (saved.address_frame != null || saved.ball)) {
      applySaved(saved);
      setFrameDims({
        width: saved.frame_width ?? row.tee_width ?? null,
        height: saved.frame_height ?? row.tee_height ?? null,
        totalFrames: row.tee_nb_frames || null,
      });
      return;
    }

    // Auto-detect runs at upload time and writes straight into
    // edit_metrics, so the wizard never calls /auto-detect itself.
    // If the saved blob is still empty when the wizard opens, the
    // background detect hasn't finished yet (or this upload predates
    // the upload-time spawn). Poll the row every few seconds until
    // metrics show up; the operator can also click Re-detect to
    // force a fresh run from the source.
    setRunning(true);
    const tick = async () => {
      try {
        const rows = await api.listLongUploads(adminPassword);
        const fresh = (rows || []).find((r) => r.id === row.id);
        const em = fresh?.edit_metrics;
        if (em && (em.address_frame != null || em.ball)) {
          if (cancelled) return;
          applySaved(em);
          setFrameDims({
            width: em.frame_width ?? fresh.tee_width ?? null,
            height: em.frame_height ?? fresh.tee_height ?? null,
            totalFrames: fresh.tee_nb_frames || null,
          });
          setRunning(false);
          return true;
        }
      } catch (e) {
        if (!cancelled) setError(e.message);
      }
      return false;
    };
    (async () => {
      // First check is immediate; then poll every 3s up to ~2 min.
      if (await tick()) return;
      for (let i = 0; i < 40 && !cancelled; i++) {
        await new Promise((r) => setTimeout(r, 3000));
        if (cancelled) return;
        if (await tick()) return;
      }
      if (!cancelled) {
        setRunning(false);
        // Auto-detect didn't complete or found nothing. Seed a
        // placeholder draft so the wizard still opens and the
        // operator can pick the address/impact frames manually
        // from the timeline.
        applySaved({});
        setFrameDims({
          width: row.tee_width || null,
          height: row.tee_height || null,
          totalFrames: row.tee_nb_frames || null,
        });
      }
    })();
    return () => { cancelled = true; };
  }, [row, adminPassword]);  // eslint-disable-line react-hooks/exhaustive-deps

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
        engine: tracerEngine,
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
          engine: out.engine || tracerEngine,
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
                tracer_engine: out.engine || tracerEngine,
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
      roi: draft.roi,
      target: draft.target,
    });
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
        engine: tracerEngine,
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
        engine: out.engine || tracerEngine,
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
        tracer_engine: out.engine || tracerEngine,
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
            <h3 style={{ margin: 0 }}>
              Edit wizard {step === "finalize"
                ? "· Step 3: Final video"
                : step === "tracer"
                  ? "· Step 2: Tracer"
                  : "· Step 1: Metrics"}
            </h3>
            <div className="small muted">
              Upload #{row.id} · {row.course_name || `course ${row.course_id}`} ·{" "}
              {isMulti
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
          {running && (
            <div className="row" style={{ alignItems: "center", gap: 12 }}>
              <div className="shimmer" style={{ width: 18, height: 18, borderRadius: "50%" }} />
              <span className="small">
                Waiting for upload-time auto-detect to finish — usually 10–20s…
              </span>
            </div>
          )}
          {detectingSwings && (
            <div className="row" style={{ alignItems: "center", gap: 12 }}>
              <div className="shimmer" style={{ width: 18, height: 18, borderRadius: "50%" }} />
              <span className="small">
                Detecting swings (audio + motion)…
              </span>
            </div>
          )}
          {error && (
            <div className="err-text small">{error}</div>
          )}
          {tracerError && step === "metrics" && (
            <div className="err-text small" style={{ marginBottom: 8 }}>
              Tracer render failed: {tracerError} — hit Next to retry, or
              switch the Tracer engine below.
            </div>
          )}
          {isMulti && !detectingSwings && swings.length > 0 && (
            <SwingSelectorBar
              swings={swings}
              selectedSwing={selectedSwing}
              setSelectedSwing={setSelectedSwing}
              onDeleteSwing={deleteSwing}
              onAddSwing={addSwing}
            />
          )}
          {!running && !detectingSwings && !error && draft && step === "metrics" && (
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
          {!running && !error && draft && step === "tracer" && tracerStats && (
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
          {!running && !error && draft && step === "tracer" && (
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
          {!running && !error && draft && step === "finalize" && (
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
          {step === "metrics" && (
            <div className="row" style={{ gap: 4, alignItems: "center", marginRight: "auto" }}>
              <span className="tiny upper muted">Tracer</span>
              <button
                type="button"
                className={tracerEngine === "ai" ? "small" : "ghost small"}
                style={{ width: "auto" }}
                onClick={() => setTracerEngine("ai")}
                disabled={renderingTracer}
                title="Claude vision tracer"
              >
                AI
              </button>
              <button
                type="button"
                className={tracerEngine === "classical" ? "small" : "ghost small"}
                style={{ width: "auto" }}
                onClick={() => setTracerEngine("classical")}
                disabled={renderingTracer}
                title="Classical CV tracer — MOG2 background subtraction (motion + parabola, no API)"
              >
                Classical
              </button>
              <button
                type="button"
                className={tracerEngine === "knn" ? "small" : "ghost small"}
                style={{ width: "auto" }}
                onClick={() => setTracerEngine("knn")}
                disabled={renderingTracer}
                title="Classical CV tracer with the KNN background subtractor — same pipeline as Classical, different motion detector; often cleaner against drifting clouds / rippling water"
              >
                KNN
              </button>
              <button
                type="button"
                className={tracerEngine === "ai_mog2" ? "small" : "ghost small"}
                style={{ width: "auto" }}
                onClick={() => setTracerEngine("ai_mog2")}
                disabled={renderingTracer}
                title="Produce's engine: AI tracer first, then the MOG2 per-frame candidate trail fills the launch (impact → first AI pick) and extends past the last pick — frames increasing gradually, 4s post-impact cap. Needs ANTHROPIC_API_KEY."
              >
                MOG2+AI
              </button>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto" }}
                onClick={handleForceRender}
                disabled={running || !!error || !draft || renderingTracer}
                title="Force a fresh tracer render with the selected engine. Next → reuses the existing ball track unless start/impact/end frames changed."
              >
                {renderingTracer ? "Rendering…" : "↻ Render"}
              </button>
            </div>
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
          {step === "metrics" && (
            <button
              type="button"
              disabled={running || !!error || !draft || renderingTracer}
              onClick={handleNext}
              style={{ width: "auto" }}
              title="Save metrics and continue. Re-renders the tracer only if the start/impact/end frames changed (or there's no track yet) — use ↻ Render to force a fresh render."
            >
              {renderingTracer ? "Rendering tracer…" : "Next →"}
            </button>
          )}
          {step === "tracer" && (
            <button
              type="button"
              disabled={renderingTracer || finalizing}
              onClick={handleAdvanceToFinalize}
              style={{ width: "auto" }}
              title="Apply graphics and review the final video"
            >
              {finalizing ? "Finalizing…" : "Next →"}
            </button>
          )}
          {step === "finalize" && (
            <button
              type="button"
              disabled={committing || finalizing}
              onClick={onClose}
              style={{ width: "auto" }}
              title="Close the wizard. The most recent Produce run is already on Produced Clips."
            >
              Finish
            </button>
          )}
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
  const [navLoading, setNavLoading] = useState(false);

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
  const FRAME_PICK_MODES = new Set(["address", "impact", "start", "end", "cut"]);
  const frameForMode = {
    address: draft.addressFrame,
    impact: draft.impactFrame,
    start: draft.startFrame ?? 0,
    end: draft.endFrame ?? (totalFrames ? totalFrames - 1 : 0),
    cut: effectiveCutFrame ?? 0,
  };

  useEffect(() => {
    if (!FRAME_PICK_MODES.has(editing)) return;
    loadFrame(frameForMode[editing] ?? 0);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [editing]);

  async function loadFrame(frameIdx) {
    setNavLoading(true);
    try {
      const data = await api.getLongUploadFrame(adminPassword, row.id, frameIdx);
      setNavFrame(data.frame);
      setNavUrl(data.image_url);
      if (data.total_frames) setNavTotal(data.total_frames);
    } catch (e) {
      console.warn("frame fetch failed", e);
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

  let leftImageUrl = draft.addressImageUrl;
  let leftFrameLabel = `Address frame · ${draft.addressFrame}`;
  const showFrameNav = FRAME_PICK_MODES.has(editing);
  if (showFrameNav) {
    leftImageUrl = navUrl || draft.addressImageUrl;
    const total = navTotal != null ? ` / ${navTotal - 1}` : "";
    const labels = {
      address: "Address", impact: "Impact",
      start: "Start", end: "End", cut: "Cut",
    };
    leftFrameLabel =
      `${labels[editing] || "Frame"} frame · ${navFrame ?? "—"}${total}`;
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
            frameW={frameW}
            frameH={frameH}
            editing={editing}
            draft={draft}
            setDraft={setDraft}
            loading={navLoading}
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
      </div>

      <div
        style={{
          display: "flex", flexDirection: "column", gap: 10,
          overflowY: "auto", minHeight: 0, paddingRight: 4,
        }}
      >
        <EditableRow
          label="Start frame"
          value={draft.startFrame != null
            ? `Frame ${draft.startFrame}`
            : "Frame 0 (clip start)"}
          active={editing === "start"}
          onActivate={() => setEditing(editing === "start" ? null : "start")}
        >
          <div className="tiny muted" style={{ marginBottom: 6 }}>
            Step backward / forward to trim the clip in. Defaults to
            the start of the source (frame 0). Address / impact frames
            below the new start are bumped forward to match.
          </div>
          <FrameStepper
            current={navFrame}
            total={navTotal}
            loading={navLoading}
            onStep={(delta) => loadFrame(clampedStep(delta))}
            onJump={(n) => loadFrame(clampedJump(n))}
            onApply={() => {
              if (navFrame == null) return;
              const newStart = navFrame;
              // Auto-bump address / impact when the operator picks a
              // start frame past them — keeps the wizard's frame order
              // sane (start ≤ address ≤ impact) without forcing an
              // extra round-trip to re-pick them. End frame is left
              // alone; an operator who deliberately set a short clip
              // can re-trim.
              setDraft((d) => {
                const next = { ...d, startFrame: newStart };
                if (d.addressFrame != null && d.addressFrame < newStart) {
                  next.addressFrame = newStart;
                }
                if (d.impactFrame != null && d.impactFrame < newStart) {
                  next.impactFrame = newStart;
                }
                return next;
              });
              const patch = { start_frame: newStart };
              if (draft.addressFrame != null && draft.addressFrame < newStart) {
                patch.address_frame = newStart;
              }
              if (draft.impactFrame != null && draft.impactFrame < newStart) {
                patch.impact_frame = newStart;
              }
              persistPatch(patch);
              setEditing(null);
            }}
          />
        </EditableRow>

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
          label="End frame"
          value={draft.endFrame != null
            ? `Frame ${draft.endFrame}`
            : (totalFrames
              ? `Frame ${totalFrames - 1} (clip end)`
              : "Clip end")}
          active={editing === "end"}
          onActivate={() => setEditing(editing === "end" ? null : "end")}
        >
          <div className="tiny muted" style={{ marginBottom: 6 }}>
            Step backward / forward to trim the clip out. Defaults to
            the last frame of the source.
          </div>
          <FrameStepper
            current={navFrame}
            total={navTotal}
            loading={navLoading}
            onStep={(delta) => loadFrame(clampedStep(delta))}
            onJump={(n) => loadFrame(clampedJump(n))}
            onApply={() => {
              if (navFrame == null) return;
              setDraft((d) => ({ ...d, endFrame: navFrame }));
              persistPatch({ end_frame: navFrame });
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
          label="Target"
          value={draft.target ? `${draft.target.x}, ${draft.target.y} px` : "Not set"}
          active={editing === "target"}
          onActivate={() => setEditing(editing === "target" ? null : "target")}
        >
          <div className="tiny muted">
            Drag the red flag on the left to mark where the flag is
            on the green.
          </div>
          <button
            type="button"
            style={{ width: "auto", marginTop: 6 }}
            onClick={() => {
              if (draft.target) persistPatch({ target: draft.target });
              setEditing(null);
            }}
          >
            Done
          </button>
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
          onClick={async () => {
            if (!confirm(
              "Re-run auto-detect from the source video? This replaces "
              + "the current handedness / address / impact / ball / "
              + "ROI / target with a fresh detection."
            )) return;
            try {
              // Persist into edit_metrics directly; the wizard reads
              // from there on next reload.
              await api.autoDetectLongUpload(adminPassword, row.id);
              window.location.reload();
            } catch (e) {
              alert(`Re-detect failed: ${e.message}`);
            }
          }}
          title="Re-run auto-detect from the source video and replace the current metrics"
        >
          Re-detect from source
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

function FramePreview({ imageUrl, frameW, frameH, editing, draft, setDraft, loading, frameNav }) {
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
    }
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
  const showBall = !!draft.ball;
  const showTarget = !!draft.target;
  const ballEditable = editing === "ball";
  const targetEditable = editing === "target";
  const roiEditable = editing === "roi";

  const pct = (v, span) => `${(v / span) * 100}%`;

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
        cursor: (ballEditable || targetEditable) ? "crosshair" : "default",
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
 * Debug3 report — the blob-and-track method. Where Debug2 reads the shape
 * the swing draws in a motion composite, this one never looks at a
 * composite: per-frame MOG2, drop the golfer, keep ball-sized blobs, link
 * them over time, fit a parabola. Read-only, so nothing to save.
 */
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
            {rep.rest_ball && (
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
              <li>MOG2 heat over impact−5 … impact+100</li>
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
  scanRegion, track, ballXY, placingBall, onPlaceBall,
}) {
  const [zoom, setZoom] = useState(1);
  const [focus, setFocus] = useState({ x: 50, y: 50 });
  // Extra detections pulled by 🔍 Scan (frame-diff over the zoomed
  // region) — merged into the dense layer for this session.
  const [scanDots, setScanDots] = useState([]);
  const [scanning, setScanning] = useState(false);
  const [scanNote, setScanNote] = useState(null);
  const hasDims = !!(frameW && frameH);
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
      const found = await scanRegion(region);
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
        setScanNote(
          fresh.length
            ? `+${fresh.length} new detections in view`
            : "no new motion found in this area",
        );
        return [...prev, ...fresh];
      });
    } catch (e) {
      console.warn("region scan failed", e);
      setScanNote("scan failed — try again");
    } finally {
      setScanning(false);
    }
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
      style={{
        display: "flex", flexDirection: "column", gap: 6,
        height: "100%", minHeight: 0, maxWidth: "100%", width: "100%",
        alignItems: "stretch",
      }}
    >
      {/* Image area — takes the whole box. The zoom / scan / pan controls
          are absolutely positioned INSIDE it (see the overlay below)
          rather than stacked above, so nothing but the map competes for
          the modal's height. */}
      <div
        style={{
          position: "relative",
          flex: 1, minHeight: 0, alignSelf: "center",
          maxHeight: "100%", maxWidth: "100%",
          aspectRatio: hasDims ? `${frameW} / ${frameH}` : "16 / 9",
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
        style={{
          position: "absolute", inset: 0,
          transform: `scale(${zoom})`,
          transformOrigin: `${focus.x}% ${focus.y}%`,
          transition: "transform 120ms ease",
          cursor: placingBall ? "crosshair" : undefined,
        }}
      >
        <img
          src={bgUrl}
          alt="Raw motion heat"
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
        {hasDims && showDense &&
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
        {hasDims &&
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
      {(scanNote || (extraDots.length > 0 && !showDense)) && (
        <div
          style={{
            position: "absolute", left: 8, bottom: 8,
            background: "rgba(0,0,0,0.55)", color: "#fde047",
            padding: "3px 10px", borderRadius: 6, fontSize: 12,
            pointerEvents: "none", backdropFilter: "blur(4px)",
            // Stay clear of the control overlay in the opposite corner.
            maxWidth: "min(55%, 520px)",
          }}
        >
          {scanNote
            ? `🔍 ${scanNote}`
            : `🔍 zoom to ${DENSE_DOT_ZOOM}×+ to reveal ${extraDots.length} more clickable detections`}
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
                : "Deep-scan the visible area: frame-diff over the swing window with much looser gates than the tracer — every transient blob in view becomes a clickable dot. Takes a few seconds."
            }
          >
            {scanning ? "Scanning…" : "🔍 Scan"}
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
              dots={(tracer?.timedPoints || []).filter(
                (p) => draft?.impactFrame == null
                  || p.frame >= draft.impactFrame,
              )}
              denseDots={(tracer?.candidates || []).filter(
                (p) => draft?.impactFrame == null
                  || p.frame >= draft.impactFrame,
              )}
              frameW={frameW}
              frameH={frameH}
              marks={manualPositions}
              onToggleDot={toggleTimedDot}
              onClose={() => setPlotAll(false)}
              scanRegion={async (region) => {
                const out = await api.scanPlotRegion(adminPassword, row.id, {
                  ...region,
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
  const teeStartsAt = triggeredAt;
  // Green starts ~5s before trigger because of pre-roll; we don't
  // have the exact wall-clock the green Pi committed, so use the
  // shared trigger time for both tiles to keep the math honest.
  const greenStartsAt = triggeredAt;
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

function VideoLightbox({ url, title, startedAt, fps, onClose }) {
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
              {startMs != null && <span>{fmtClock(startMs + curTime * 1000)} CT</span>}
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
  const [busyEventId, setBusyEventId] = useState(null);
  const [viewer, setViewer] = useState(null); // {url, title, startedAt, fps}
  const [editingRow, setEditingRow] = useState(null);
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
      kind === "debug3" ? api.debug3Status : api.debug2Status;
    const tick = async () => {
      try {
        const st = await statusCall(adminPassword, uploadId);
        setter({
          running: !!st.running,
          uploadId,
          stage: st.stage,
          done: st.done,
          total: st.total,
          report: st.report || null,
          error: st.error || null,
        });
        if (st.running) setTimeout(tick, 2500);
        else setBusyId((cur) => (cur === uploadId ? null : cur));
      } catch (e) {
        setter({
          running: false, uploadId, report: null, error: e.message,
        });
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

  async function handleDebug3(row) {
    // Window first, same reason as Debug2: a throw after this still leaves
    // a visible panel carrying the error.
    setD3({ running: true, uploadId: row.id, report: null, error: null });
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

  async function handleBulkDelete() {
    const ids = Array.from(selectedIds);
    if (ids.length === 0) return;
    if (!confirm(
      `Delete ${ids.length} selected upload(s)? This removes their source `
      + `video(s) and can't be undone.`,
    )) return;
    setBulkBusy(true);
    setError(null);
    let failed = 0;
    for (const id of ids) {
      try {
        await api.deleteLongUpload(adminPassword, id);
      } catch (e) {
        failed += 1;
      }
    }
    clearSelection();
    await refreshAll();
    setBulkBusy(false);
    if (failed) setError(`${failed} of ${ids.length} deletions failed.`);
  }

  function openViewer(url, title, startedAt = null, fps = null) {
    if (!url) return;
    setViewer({ url, title, startedAt, fps });
  }

  // Re-fetch the currently-loaded range on both lists. Called by the
  // background poll loop and by every handler that mutates server state,
  // so a successful action shows its effect without a full reload (which
  // would yank the user back to page 1).
  const refreshAll = useCallback(async () => {
    setError(null);
    await Promise.all([uploadsList.refresh(), eventsList.refresh()]);
  }, [uploadsList.refresh, eventsList.refresh]);

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

  async function handleDeleteEvent(ev) {
    if (!confirm(
      `Delete camera event #${ev.id}? This removes the raw tee/green clips and the produced clip.`,
    )) return;
    setBusyEventId(ev.id);
    try {
      await api.deleteCameraEvent(adminPassword, ev.id);
      await refreshAll();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyEventId(null);
    }
  }

  async function handleBroadcastEvent(ev) {
    // Toggle the produced clip's is_highlight flag — same semantic as
    // the long-upload Broadcast button, just sourced from the event's
    // produced_clip relation instead of produced_clips[0].
    const clip = ev.produced_clip;
    if (!clip) return;
    setBusyEventId(ev.id);
    try {
      await api.setClipBroadcast(adminPassword, clip.id, !clip.is_highlight);
      await refreshAll();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyEventId(null);
    }
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

  async function handleDelete(row) {
    if (!confirm(`Delete upload #${row.id}? This removes the source video(s).`)) return;
    setBusyId(row.id);
    try {
      await api.deleteLongUpload(adminPassword, row.id);
      await refreshAll();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  }

  async function handleBroadcast(row) {
    // Toggle the produced clip's is_highlight flag so it shows up on
    // the Broadcast channel. The Production card surfaces the latest
    // produced_clip; that's what we operate on.
    const clip = row.produced_clips?.[0];
    if (!clip) return;
    setBusyId(row.id);
    try {
      await api.setClipBroadcast(adminPassword, clip.id, !clip.is_highlight);
      await refreshAll();
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyId(null);
    }
  }

  function handleEdit(row) {
    // Both single-swing and multi-swing uploads open the EditWizard.
    // The wizard switches its internal flow based on row.swing_count
    // — single uses the flat edit_metrics; multi works on a per-swing
    // edit_metrics.swings[] array with a swing-selector bar at the top.
    setEditingRow(row);
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

  // Hand the greyed-out state from the optimistic flag to the server's
  // status, without a gap between them.
  useEffect(() => {
    if (busyId == null) return;
    const r = (rows || []).find((x) => x.id === busyId);
    if (!r) return;
    // ONLY "processing". Including the terminal states was the bug: the
    // row is ALREADY "completed" from its last run at the moment you click
    // Re-Produce, so this effect fired on the very next render and cleared
    // busy instantly -- which is the flicker. Wait for the worker to
    // actually claim it.
    if (r.processing_status === "processing") {
      setBusyId(null);
    }
  }, [rows, busyId]);

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
                />
                <ProducedTile
                  clips={row.produced_clips}
                  swings={row.edit_metrics?.swings}
                  onOpenViewer={openViewer}
                  onClickToPlot={(swingIdx) => {
                    // Map the produce swing idx to its position in the
                    // swings array (they diverge if a swing was deleted).
                    const arr = row.edit_metrics?.swings || [];
                    const pos = arr.findIndex((s) => s?.idx === swingIdx);
                    setPlotModal({
                      row,
                      swingPos: pos >= 0 ? pos : swingIdx,
                    });
                  }}
                  onDeleteClip={async (clip, clipIdx) => {
                    const label = clip.hole_number != null
                      ? `clip ${clipIdx + 1} (hole ${clip.hole_number})`
                      : `clip ${clipIdx + 1}`;
                    if (!window.confirm(
                      `Delete produced ${label}? The video and its files ` +
                      `are removed; the raw upload and the other clips ` +
                      `stay, and Re-Produce can recreate it.`
                    )) {
                      return;
                    }
                    setBusyId(row.id);
                    try {
                      await api.deleteClip(adminPassword, clip.id);
                      await refreshAll();
                    } catch (e) {
                      setError(e.message);
                    } finally {
                      setBusyId(null);
                    }
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
                {SHOW_LEGACY_DEBUG && produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleProduceDebug(row)}
                    disabled={busy}
                    title="Dev: produce AND run a per-swing diagnostic — classical-CV heatmap vs AI tracer"
                  >
                    🐞 Debug
                  </button>
                )}
                {SHOW_LEGACY_DEBUG && produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleDebug2(row)}
                    disabled={busy}
                    title="Dev: pose candidates → impact + ball from the club arc → AI judge → windowed MOG2 heat → chain walked up from the ball. Shows every stage."
                  >
                    🔬 Debug2
                  </button>
                )}
                {produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleDebug3(row)}
                    disabled={busy}
                    title="Dev: MOG2 per frame → drop the golfer → keep ball-sized blobs → link across frames → RANSAC parabola. A different method from Debug2; shows every stage."
                  >
                    🧿 Debug3
                  </button>
                )}
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
          <ProduceStatusOverlay row={row} greyed={greyed} />
          </div>
        );
      })}

      {rows && rows.length > 0 && (
        <div
          ref={uploadsList.sentinelRef}
          className="muted center small"
          style={{ padding: 16 }}
        >
          {uploadsList.loadingMore
            ? "Loading more uploads…"
            : uploadsList.hasMore
              ? "Scroll for more uploads"
              : "End of long uploads"}
        </div>
      )}

      <VideoLightbox
        url={viewer?.url}
        title={viewer?.title}
        startedAt={viewer?.startedAt}
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
        <EditWizard
          row={editingRow}
          adminPassword={adminPassword}
          onClose={() => { setEditingRow(null); refreshAll(); }}
          onSaved={refreshAll}
        />
      )}

      {plotModal && (
        <ClickToPlotModal
          row={plotModal.row}
          swingPos={plotModal.swingPos}
          adminPassword={adminPassword}
          onClose={() => { setPlotModal(null); refreshAll(); }}
          onSaved={refreshAll}
        />
      )}

      {d2 && <Debug2Modal state={d2} onClose={() => setD2(null)} />}
      {d3 && <Debug3Modal state={d3} onClose={() => setD3(null)} />}
    </div>
  );
}
