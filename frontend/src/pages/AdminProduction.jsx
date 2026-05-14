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
import { useEffect, useRef, useState } from "react";
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
  const [finalUrl, setFinalUrl] = useState(null);
  const [finalizing, setFinalizing] = useState(false);
  const [finalError, setFinalError] = useState(null);
  const [committing, setCommitting] = useState(false);

  useEffect(() => {
    if (!row) return;
    let cancelled = false;

    function applySaved(s) {
      setDraft({
        handedness: s.handedness || "right",
        addressFrame: s.address_frame ?? 0,
        addressImageUrl: s.address_image_url || null,
        impactFrame: s.impact_frame ?? 0,
        ball: s.ball || null,
        roi: s.roi || null,
        target: s.target || null,
      });
      if (s.tracer_url || s.ball_track_frames) {
        setTracer({
          url: s.tracer_url || null,
          frames: s.ball_track_frames || [],
        });
      }
      if (s.finalized_video_url) setFinalUrl(s.finalized_video_url);
    }

    // Already persisted → skip auto-detect entirely. Frame dims come
    // straight off the upload's probe info.
    if (saved && (saved.address_frame != null || saved.ball)) {
      applySaved(saved);
      setFrameDims({
        width: row.tee_width || null,
        height: row.tee_height || null,
        totalFrames: row.tee_nb_frames || null,
      });
      return;
    }

    setRunning(true);
    setError(null);
    api
      .autoDetectLongUpload(adminPassword, row.id)
      .then(async (data) => {
        if (cancelled) return;
        setFrameDims({
          width: data.frame_width,
          height: data.frame_height,
          totalFrames: data.total_frames || row.tee_nb_frames || null,
        });
        const seeded = {
          handedness: data.handedness?.value || "right",
          address_frame: data.address?.frame ?? 0,
          address_image_url: data.address?.image_url || null,
          impact_frame: data.impact?.frame ?? 0,
          ball: data.ball_at_rest || null,
          roi: data.ball_detection_area || null,
          target: data.target ? { x: data.target.x, y: data.target.y } : null,
        };
        applySaved(seeded);
        // Persist the auto-detected seed so we never re-run on re-open.
        try {
          await api.saveEditMetrics(adminPassword, row.id, seeded);
          onSaved?.();
        } catch (e) {
          console.warn("seed-save failed", e);
        }
      })
      .catch((e) => { if (!cancelled) setError(e.message); })
      .finally(() => { if (!cancelled) setRunning(false); });
    return () => { cancelled = true; };
  }, [row, adminPassword]);  // eslint-disable-line react-hooks/exhaustive-deps

  if (!row) return null;

  async function persistPatch(patch) {
    try {
      const r = await api.saveEditMetrics(adminPassword, row.id, patch);
      onSaved?.(r);
    } catch (e) {
      console.warn("save metrics failed", e);
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
    if (tracer?.url) {
      setStep("tracer");
      return;
    }
    setRenderingTracer(true);
    setTracerError(null);
    try {
      const out = await api.renderWizardTracer(adminPassword, row.id, {
        handedness: draft.handedness,
        impact_frame: draft.impactFrame,
        ball_at_rest: draft.ball,
      });
      setTracer({
        url: out.tracer_url,
        frames: out.ball_track_frames || [],
      });
      onSaved?.();
      setStep("tracer");
    } catch (e) {
      setTracerError(e.message);
    } finally {
      setRenderingTracer(false);
    }
  }

  async function handleAdvanceToFinalize() {
    // Step 2 → Step 3. Reuse the cached final video when present;
    // otherwise apply the intro overlay on top of the rendered tracer.
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
      const out = await api.finalizeWizardVideo(adminPassword, row.id, {});
      setFinalUrl(out.final_video_url);
      onSaved?.();
    } catch (e) {
      setFinalError(e.message);
    } finally {
      setFinalizing(false);
      setStep("finalize");
    }
  }

  async function handleSaveToProduced() {
    if (!finalUrl) return;
    setCommitting(true);
    try {
      await api.commitWizardClip(adminPassword, row.id);
      onSaved?.();
      onClose();
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
                Auto-detecting handedness, address frame, and ball position…
              </span>
            </div>
          )}
          {error && (
            <div className="err-text small">Auto-detect failed: {error}</div>
          )}
          {!running && !error && draft && step === "metrics" && (
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
          {!running && !error && draft && step === "tracer" && (
            <TracerStep
              row={row}
              adminPassword={adminPassword}
              draft={draft}
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
            />
          )}
          {!running && !error && draft && step === "finalize" && (
            <FinalizeStep
              row={row}
              finalUrl={finalUrl}
              finalizing={finalizing}
              error={finalError}
              frameW={fw}
              frameH={fh}
              onReRender={async () => {
                setFinalUrl(null);
                await handleAdvanceToFinalize();
              }}
            />
          )}
        </div>

        <div className="row" style={{ gap: 8, justifyContent: "flex-end" }}>
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
              disabled={!finalUrl || committing || finalizing}
              onClick={handleSaveToProduced}
              style={{ width: "auto" }}
              title="Commit this clip to Produced Clips"
            >
              {committing ? "Saving…" : "Save"}
            </button>
          )}
        </div>
      </div>
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

  useEffect(() => {
    if (editing !== "address" && editing !== "impact") return;
    const start = editing === "address" ? draft.addressFrame : draft.impactFrame;
    loadFrame(start);
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
    const cur = navFrame ?? (editing === "impact" ? draft.impactFrame : draft.addressFrame);
    const max = (navTotal ?? totalFrames ?? 1) - 1;
    return Math.max(0, Math.min(max, (cur || 0) + delta));
  }

  let leftImageUrl = draft.addressImageUrl;
  let leftFrameLabel = `Address frame · ${draft.addressFrame}`;
  const showFrameNav = editing === "address" || editing === "impact";
  if (showFrameNav) {
    leftImageUrl = navUrl || draft.addressImageUrl;
    const total = navTotal != null ? ` / ${navTotal - 1}` : "";
    leftFrameLabel =
      `${editing === "address" ? "Address" : "Impact"} frame · ${navFrame ?? "—"}${total}`;
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
          label="Handedness"
          value={draft.handedness === "left" ? "Left" : "Right"}
          active={editing === "handedness"}
          onActivate={() => setEditing(editing === "handedness" ? null : "handedness")}
        >
          <div className="row" style={{ gap: 6 }}>
            <button
              type="button"
              className={draft.handedness === "right" ? "" : "ghost"}
              style={{ width: "auto", flex: 1 }}
              onClick={() => {
                setDraft((d) => ({ ...d, handedness: "right" }));
                persistPatch({ handedness: "right" });
              }}
            >
              Right
            </button>
            <button
              type="button"
              className={draft.handedness === "left" ? "" : "ghost"}
              style={{ width: "auto", flex: 1 }}
              onClick={() => {
                setDraft((d) => ({ ...d, handedness: "left" }));
                persistPatch({ handedness: "left" });
              }}
            >
              Left
            </button>
          </div>
        </EditableRow>

        <EditableRow
          label="Address frame"
          value={`Frame ${draft.addressFrame}`}
          active={editing === "address"}
          onActivate={() => setEditing(editing === "address" ? null : "address")}
        >
          <FrameStepper
            current={navFrame}
            total={navTotal}
            loading={navLoading}
            onStep={(delta) => loadFrame(clampedStep(delta))}
            onApply={() => {
              if (navFrame == null) return;
              const url = navUrl || draft.addressImageUrl;
              setDraft((d) => ({
                ...d, addressFrame: navFrame, addressImageUrl: url,
              }));
              persistPatch({ address_frame: navFrame, address_image_url: url });
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
            onApply={() => {
              if (navFrame == null) return;
              setDraft((d) => ({ ...d, impactFrame: navFrame }));
              persistPatch({ impact_frame: navFrame });
              setEditing(null);
            }}
          />
        </EditableRow>

        <EditableRow
          label="Resting ball"
          value={draft.ball ? `${draft.ball.x}, ${draft.ball.y} px` : "Not set"}
          active={editing === "ball"}
          onActivate={() => setEditing(editing === "ball" ? null : "ball")}
        >
          <div className="tiny muted">
            Drag the green dot on the left to set the ball-at-rest
            position. Address frame is shown.
          </div>
          <button
            type="button"
            style={{ width: "auto", marginTop: 6 }}
            onClick={() => {
              if (draft.ball) persistPatch({ ball: draft.ball });
              setEditing(null);
            }}
          >
            Done
          </button>
        </EditableRow>

        <EditableRow
          label="Detection area"
          value={draft.roi
            ? `${draft.roi.w} × ${draft.roi.h} px @ (${draft.roi.x}, ${draft.roi.y})`
            : "Not set"}
          active={editing === "roi"}
          onActivate={() => setEditing(editing === "roi" ? null : "roi")}
        >
          <div className="tiny muted">
            Drag the green rectangle to move it. Drag any corner to resize.
          </div>
          {hasDims && draft.roi && (
            <div className="row" style={{ gap: 6, marginTop: 6 }}>
              <button
                type="button"
                className="ghost"
                style={{ width: "auto" }}
                onClick={() => setDraft((d) => ({
                  ...d, roi: scaleRoi(d.roi, 0.85, frameW, frameH),
                }))}
              >
                Shrink
              </button>
              <button
                type="button"
                className="ghost"
                style={{ width: "auto" }}
                onClick={() => setDraft((d) => ({
                  ...d, roi: scaleRoi(d.roi, 1.18, frameW, frameH),
                }))}
              >
                Grow
              </button>
              <button
                type="button"
                style={{ width: "auto", marginLeft: "auto" }}
                onClick={() => {
                  if (draft.roi) persistPatch({ roi: draft.roi });
                  setEditing(null);
                }}
              >
                Done
              </button>
            </div>
          )}
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
        padding: 8,
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
        <div style={{ fontSize: "0.95rem" }}>{value}</div>
      </button>
      {active && (
        <div style={{ marginTop: 8 }}>{children}</div>
      )}
    </div>
  );
}

function FrameStepper({ current, total, loading, onStep, onApply }) {
  const disabled = loading || current == null;
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

function TracerStep({
  row, adminPassword, draft, tracer, setTracer,
  rendering, setRendering, error, setError,
  frameW, frameH, totalFrames, onSaved,
}) {
  // Manual ball corrections accumulated since the last render. Each is
  // a {frame, x, y} entry; clicking Re-generate ships these via the
  // /render-tracer endpoint which merges them into ball_track_frames.
  const [manualPositions, setManualPositions] = useState({});
  const [selectedFrame, setSelectedFrame] = useState(null);
  const [editorBg, setEditorBg] = useState(null); // {url, frame}
  const [editorBall, setEditorBall] = useState(null); // {x, y}
  const [zoom, setZoom] = useState(1);
  const editorRef = useRef(null);

  const frames = tracer?.frames || [];
  const hasDims = !!(frameW && frameH);
  const maxFrame = totalFrames ? totalFrames - 1 : null;

  // Pivot for the zoom transform — defaults to the ball detection
  // ROI from Step 1 so the operator drops straight into the ball
  // area. Falls back to the resting-ball point, then frame centre.
  const focusPct = (() => {
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
    const m = manualPositions[f.frame];
    if (m) return { x: m.x, y: m.y, manual: true };
    if (f.found && f.x != null && f.y != null) return { x: f.x, y: f.y, manual: !!f.manual };
    return null;
  }

  async function loadEditorFrame(frameIdx) {
    setSelectedFrame(frameIdx);
    setEditorBg(null);
    setEditorBall(null);
    setZoom(autoZoom);
    try {
      const data = await api.getLongUploadFrame(adminPassword, row.id, frameIdx);
      setEditorBg({ url: data.image_url, frame: data.frame });
      const existing = (tracer?.frames || []).find((f) => f.frame === frameIdx);
      const m = manualPositions[frameIdx];
      if (m) setEditorBall({ x: m.x, y: m.y });
      else if (existing?.found && existing.x != null) {
        setEditorBall({ x: existing.x, y: existing.y });
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
    setManualPositions((m) => {
      const next = { ...m };
      delete next[selectedFrame];
      return next;
    });
  }

  function addFrame(delta) {
    const lastTracked = frames.length
      ? Math.max(...frames.map((f) => f.frame))
      : (selectedFrame ?? 0);
    const target = Math.max(0, Math.min(maxFrame ?? lastTracked + delta, lastTracked + delta));
    loadEditorFrame(target);
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
    const pt = editorEventToFrame(e);
    if (pt) setEditorBall(pt);
  }

  async function regenerate() {
    setRendering(true);
    setError(null);
    try {
      const overrides = [];
      Object.entries(manualPositions).forEach(([f, p]) => {
        overrides.push({ frame: parseInt(f, 10), x: p.x, y: p.y });
      });
      const out = await api.renderWizardTracer(adminPassword, row.id, {
        handedness: draft.handedness,
        impact_frame: draft.impactFrame,
        ball_at_rest: draft.ball,
        manual_ball_positions: overrides,
      });
      setTracer({
        url: out.tracer_url,
        frames: out.ball_track_frames || [],
      });
      setManualPositions({});
      setSelectedFrame(null);
      setEditorBg(null);
      setEditorBall(null);
      onSaved?.();
    } catch (e) {
      setError(e.message);
    } finally {
      setRendering(false);
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
                    src={editorBg.url}
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
                  onClick={() => setZoom(autoZoom)}
                  title="Auto zoom to ball detection area"
                >Auto</button>
                <button
                  type="button"
                  style={{ ...zoomBtn, width: 36 }}
                  onClick={() => setZoom(1)}
                  title="Fit full frame"
                >Fit</button>
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
            ? "Click anywhere on the frame to place the ball. Apply to queue, Re-generate to render."
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
            </div>
            <div className="row" style={{ gap: 6, flexWrap: "wrap" }}>
              <button
                type="button"
                style={{ width: "auto" }}
                disabled={!editorBall}
                onClick={applyEditorBall}
              >
                Apply
              </button>
              <button
                type="button"
                className="ghost"
                style={{ width: "auto" }}
                onClick={clearEditorBall}
                disabled={!manualPositions[selectedFrame]}
              >
                Clear
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
            disabled={rendering || Object.keys(manualPositions).length === 0}
            onClick={regenerate}
          >
            {rendering
              ? "Re-rendering…"
              : `Re-generate${Object.keys(manualPositions).length
                ? ` (${Object.keys(manualPositions).length})`
                : ""}`}
          </button>
        </div>

        <div className="tiny upper muted" style={{ marginTop: 4 }}>
          Per-frame ball-track ({frames.length})
        </div>
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fill, minmax(120px, 1fr))",
            gap: 6,
          }}
        >
          {frames.map((f) => {
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
                  {hasDims && ball && (
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
                    {" "}{f.found ? "found" : "no ball"}
                    {isQueued ? " · queued" : ""}
                  </span>
                </div>
              </button>
            );
          })}
          {!frames.length && (
            <div className="muted small">
              No ball-track yet. Re-generate to populate.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function FinalizeStep({ row, finalUrl, finalizing, error, frameW, frameH, onReRender }) {
  const hasDims = !!(frameW && frameH);
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
          }}
        >
          {finalizing ? (
            <div className="row" style={{ alignItems: "center", gap: 12 }}>
              <div className="shimmer" style={{ width: 18, height: 18, borderRadius: "50%" }} />
              <span className="small">Applying graphics… 10–30s.</span>
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
        </div>
        <div className="tiny muted" style={{ marginTop: 6 }}>
          The player banner, course / hole / par / yardage are baked
          in. Click <b>Save</b> to commit this clip to Produced Clips.
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

        <button
          type="button"
          className="ghost"
          style={{ width: "100%" }}
          onClick={onReRender}
          disabled={finalizing}
        >
          {finalizing ? "Re-rendering…" : "Re-apply graphics"}
        </button>

        <div className="tiny muted">
          Re-apply if you change anything on Step 1 or Step 2 — the
          finalized video is cached until you click here.
        </div>
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
          onClose={() => { setEditingRow(null); load(); }}
          onSaved={load}
        />
      )}
    </div>
  );
}
