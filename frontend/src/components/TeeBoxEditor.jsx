/**
 * The daily tee box: the patch of turf the ball search is confined to.
 *
 * LIVES HERE BECAUSE TWO PAGES DRAW IT. It began on the Debug3 report,
 * which is where the instruction "draw one on the frame below" was
 * printed with nothing underneath it -- the one action the report asked
 * for was the one action it did not offer. The Cameras page needs the
 * same control for a different reason: the markers move every morning,
 * and the person who sets them is standing at the camera, not reading a
 * scan report.
 *
 * `tb` is whatever GET /long-uploads/{id}/tee-box returned:
 *   {roi, source, note, hole, day, course_id, frame_url}
 * The box is saved against the COURSE, hole and day -- not the upload --
 * so drawing it once serves every clip of that hole that day.
 */
import { useRef, useState } from "react";

import { api } from "../api.js";

/** The tee box, drawn on a frame you can redraw it on.
 *
 * The ROI resolver has always written "draw one on the frame below" when
 * no box exists for the hole and day. That was true of the swing test and
 * has never been true here -- Debug3 printed the instruction with nothing
 * underneath it, so the one action it asked for was the one action it did
 * not offer.
 */
export function D3TeeBox({ tb, uploadId, adminPassword, onRerun }) {
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
