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

function ProducedTile({ clips, swings, onOpenViewer }) {
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
  // points) into edit_metrics.swings — clip order matches swing order.
  const withOverlay = (swings || []).filter((s) => s?.mog2_overlay_url);
  const curSwing =
    withOverlay.length === 1 && (clips || []).length <= 1
      ? withOverlay[0]
      : (swings || []).find((s) => s?.idx === idx && s?.mog2_overlay_url);
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
      {curSwing && (
        <a
          href={curSwing.mog2_overlay_url}
          target="_blank"
          rel="noreferrer"
          className="small"
          style={{
            display: "block", textAlign: "center", marginTop: 4,
            padding: "3px 8px", borderRadius: 6,
            border: "1px solid rgba(230,126,34,0.5)",
            textDecoration: "none",
          }}
          title={
            curSwing.mog2_stats
              ? `AI picks: ${curSwing.mog2_stats.n_ai ?? "?"} · MOG2 dots: ` +
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
              : "MOG2 raw motion heat with AI + MOG2 plot points"
          }
        >
          🔥 MOG2 vs AI points
          {curSwing.mog2_stats?.n_added > 0
            ? ` (+${curSwing.mog2_stats.n_added} added)`
            : ""}
        </a>
      )}
      <div style={{ display: "flex", flexDirection: "column", gap: 2, marginTop: 2 }}>
        <MetaRow k="Clips" v={has ? clips.length : ""} />
        <MetaRow k="Aces" v={has ? aces : ""} />
        <MetaRow k="Holes" v={holes.length ? holes.join(", ") : ""} />
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
  // (CV motion + parabola). tracerEngineUsed tracks what produced the
  // current tracer so switching engines forces a fresh render.
  const [tracerEngine, setTracerEngine] = useState("ai");
  const [tracerEngineUsed, setTracerEngineUsed] = useState(null);
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
    });
    // Remember which engine produced the saved tracer so clicking "Next"
    // on Step 1 reuses it instead of re-rendering (which would wipe the
    // operator's manually-plotted points). Mark "used" as null when this
    // swing has no render yet, so Next / auto-detect renders a fresh one.
    const eng = s.tracer_engine || "ai";
    setTracerEngine(eng);
    setTracerEngineUsed(hasSavedTracer ? eng : null);
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
        });
        setTracerEngineUsed(out.engine || tracerEngine);
        setRenderedFrameSig(frameSig(draft));
        setTracerStats({
          engine: out.engine || tracerEngine,
          n_points: out.n_points,
          n_candidates: out.n_candidates,
          n_backfilled: out.n_backfilled,
          n_ai_anchors: out.n_ai_anchors ?? null,
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

  async function handleNext() {
    // Persist current draft, then either reuse the cached tracer or
    // render a fresh one.
    await persistPatch({
      handedness: draft.handedness,
      address_frame: draft.addressFrame,
      address_image_url: draft.addressImageUrl,
      impact_frame: draft.impactFrame,
      ball: draft.ball,
      roi: draft.roi,
      target: draft.target,
    });
    // NEVER re-render an unchanged swing. If it already has a rendered
    // tracer OR plotted ball points, and nothing that affects the trace
    // was edited on Step 1, just advance — re-rendering would wipe the
    // existing points (the operator's plots AND the ones carried over
    // from the original production). Only render when:
    //   - there's no existing trace at all (first time through), or
    //   - the start / impact / end frames changed (points are anchored to
    //     the old window, so they're stale), or
    //   - the operator switched tracer engines (A/B) — UNLESS they have
    //     manual points we'd overwrite, in which case keep them.
    // Crucially the reuse no longer requires a tracer_url: a swing can
    // carry ball_track_frames without a (still-valid) rendered video, and
    // those points must survive Next.
    const framesChanged =
      renderedFrameSig !== null && frameSig(draft) !== renderedFrameSig;
    const engineChanged =
      tracerEngineUsed !== null && tracerEngineUsed !== tracerEngine;
    const hasManualPoints =
      (tracer?.frames || []).some((f) => f && f.manual) ||
      Object.keys(manualPositions).length > 0;
    const hasExistingTrace =
      !!(tracer?.url) || (tracer?.frames?.length || 0) > 0;
    if (
      hasExistingTrace &&
      !framesChanged &&
      (!engineChanged || hasManualPoints)
    ) {
      setStep("tracer");
      return;
    }
    // A frame edit invalidates any queued manual marks — clear them so
    // they aren't baked into the fresh render at the wrong positions.
    if (framesChanged && Object.keys(manualPositions).length > 0) {
      setManualPositions({});
    }
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
      });
      // Adopt the flight-derived rest position (never over an operator-set
      // one) so the Step-2 rest card starts where the render anchored.
      if (out.ball_at_rest && !out.ball_manual) {
        setDraft((d) => ({ ...d, ball: out.ball_at_rest, ballManual: false }));
      }
      setTracerEngineUsed(out.engine || tracerEngine);
      setRenderedFrameSig(frameSig(draft));
      setTracerStats({
        engine: out.engine || tracerEngine,
        n_points: out.n_points,
        n_candidates: out.n_candidates,
        n_backfilled: out.n_backfilled,
        n_ai_anchors: out.n_ai_anchors ?? null,
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
      await api.commitWizardClip(adminPassword, row.id);
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
                      : "AI"}
              </b>
              {tracerStats.n_ai_anchors != null && (
                <> · {tracerStats.n_ai_anchors} AI anchors</>
              )}
              {" · "}{tracerStats.n_points ?? "—"} points plotted
              {tracerStats.n_candidates != null && (
                <> · {tracerStats.n_candidates} candidates</>
              )}
              {tracerStats.n_backfilled ? (
                <> · {tracerStats.n_backfilled} backfilled</>
              ) : null}
              {" — switch the Tracer toggle on Step 1 and re-run to compare."}
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
                className={tracerEngine === "hybrid" ? "small" : "ghost small"}
                style={{ width: "auto" }}
                onClick={() => setTracerEngine("hybrid")}
                disabled={renderingTracer}
                title="MOG2 detections cross-validated with ~10 AI ball fixes: only CV points that agree with the AI-anchored curve survive. CV = density, AI = truth. Needs ANTHROPIC_API_KEY."
              >
                MOG2+AI
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
              title="Save metrics and render the tracer"
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
          area · Red flag = target. Click a field on the right to edit.
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
  const editorRef = useRef(null);

  // Default zoom level when dropping straight onto a ball position.
  // 3x is tight enough to nudge pixel-perfect, loose enough to keep
  // surrounding context visible if the AI was a couple frames off.
  const BALL_ZOOM = 3;

  const frames = tracer?.frames || [];
  const hasDims = !!(frameW && frameH);
  const maxFrame = totalFrames ? totalFrames - 1 : null;

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

  function onEditorPointerDown(e) {
    // Click auto-queues the position so navigating to another frame
    // doesn't drop the work. Add Frame button is just for explicit
    // confirmation now — clicking already commits.
    const pt = editorEventToFrame(e);
    if (!pt) return;
    setEditorBall(pt);
    // Clicking on the REST card's frame moves the resting-ball anchor
    // itself (the start of the tracer line), not a flight point.
    if (selectedFrame != null && selectedFrame === restFrame) {
      setDraft?.((d) => ({ ...d, ball: { x: pt.x, y: pt.y }, ballManual: true }));
      persistPatch?.({ ball: { x: pt.x, y: pt.y }, ball_manual: true });
      return;
    }
    if (selectedFrame != null) {
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
          {selectedFrame != null
            ? `Editing frame ${selectedFrame}`
            : "Rendered tracer"}
        </div>
        <div
          style={{
            flex: 1, minHeight: 0,
            display: "flex", alignItems: "center", justifyContent: "center",
          }}
        >
          {selectedFrame != null ? (
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
          {selectedFrame != null
            ? "Click on the ball to queue this frame as a tracer point. Navigate to other frames and add more — including past the AI's 12-frame stop, all the way to the green. Re-generate tracer re-renders here (no AI calls)."
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
  const badge = eventStatusBadge(ev.status);
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
function ProduceDebugModal({ data, adminPassword, onRerun, onClose }) {
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
          The clip is also being produced &amp; saved normally. Same pipeline
          as Produce — this just shows the work. Only swings that survive
          every filter get the classical-CV vs AI tracer comparison below;
          eliminated ones are skipped (produce skips them too).
        </p>
        <div className="small" style={{ marginBottom: 12 }}>
          {data.running
            ? `Analyzing… ${data.done}/${data.total || "?"} swings`
            : `Done — ${swings.length} swing(s) analyzed`}
          {" · "}
          {data.ai_available
            ? "AI tracer: on"
            : "AI tracer: OFF (set ANTHROPIC_API_KEY on this deployment)"}
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
                  <strong>AI tracer</strong> — {okBadge(s.ai?.ok)}
                </div>
                {stat("address frame", s.ai?.address_frame)}
                {stat("impact frame", s.ai?.impact_frame)}
                {stat("handedness", s.ai?.handedness)}
                {stat("ball-track points", s.ai?.n_track)}
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
  // Bulk-delete selection: a Set of long-upload row ids the operator
  // has ticked. Cleared after a bulk delete completes.
  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [bulkBusy, setBulkBusy] = useState(false);
  // "Pull new from prod" — server-side mirror. Hidden unless the backend
  // is configured (MIRROR_COURSE_ID set on this deployment).
  const [mirror, setMirror] = useState({ configured: false, running: false });
  // "Scan for non-golf videos" — dev tool. Pre-checks the delete boxes on
  // clips that don't look like a real golf shot. Hidden unless the backend
  // enables it (SCAN_NON_GOLF_ENABLED on this deployment).
  const [scan, setScan] = useState({ enabled: false, running: false });
  // "Produce (debug)" — dev tool. Produces normally AND opens a per-swing
  // diagnostic comparing the classical-CV and AI tracers. Hidden unless the
  // backend enables it (PRODUCE_DEBUG_ENABLED on this deployment).
  const [produceDebug, setProduceDebug] = useState({ enabled: false });
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

  function pollScan() {
    const tick = async () => {
      try {
        const s = await api.scanNonGolfStatus(adminPassword);
        setScan(s);
        if (s.running) {
          setTimeout(tick, 2000);
        } else {
          // Pre-check EVERY flagged upload, not just the ones currently
          // rendered. The queue is an infinite-scroll list, so intersecting
          // with the loaded rows dropped everything below the fold; the
          // per-row checkbox reads selectedIds, so ids for not-yet-loaded
          // rows stay checked and simply light up as you scroll to them.
          setSelectedIds(new Set(s.flagged || []));
        }
      } catch {
        /* transient — stop polling */
      }
    };
    tick();
  }

  async function handleScanNonGolf() {
    setError(null);
    try {
      const r = await api.scanNonGolf(adminPassword);
      if (r && r.ok === false) {
        setError(r.error || "Scan is not enabled.");
        return;
      }
      setScan((s) => ({ ...s, running: true, done: 0, total: r.total || 0 }));
      pollScan();
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
    api
      .scanNonGolfStatus(adminPassword)
      .then((s) => {
        setScan(s);
        if (s.running) pollScan();
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
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyEventId(null);
    }
  }

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

  useEffect(() => {
    if (!adminPassword) return;
    // Poll while anything is actively producing so the badge clears
    // automatically when the background worker finishes. The hook
    // handles the initial page-1 fetch on mount itself; we just need
    // to keep it warm.
    const id = setInterval(refreshAll, 8000);
    return () => clearInterval(id);
  }, [adminPassword, refreshAll]);

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
          {scan.enabled && (
            <button
              className="small"
              onClick={handleScanNonGolf}
              disabled={scan.running}
              style={{ width: "auto" }}
              title="Scan the loaded clips and pre-check the ones that aren't golf shots (no person / indoor). Nothing is deleted — review, then Delete."
            >
              {scan.running
                ? `Scanning ${scan.done}/${scan.total || "?"}…`
                : "🔍 Scan for non-golf videos"}
            </button>
          )}
          {scan.enabled && !scan.running && scan.finished_at ? (
            <span className="small muted">
              flagged {(scan.flagged || []).length}
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
              ) : (
                <span className="small muted">
                  {row.swing_count === "single" ? "One swing" : "Multiple swings"}
                </span>
              )}
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
                {produceDebug.enabled && (
                  <button
                    className="small ghost"
                    onClick={() => handleProduceDebug(row)}
                    disabled={busy}
                    title="Dev: produce AND run a per-swing diagnostic — classical-CV heatmap vs AI tracer"
                  >
                    🐞 Debug
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
    </div>
  );
}
