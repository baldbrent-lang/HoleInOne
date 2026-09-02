/**
 * The green -> tee calibrator, and the zoomable still it clicks on.
 *
 * LIVES HERE BECAUSE IT BELONGS TO THE CAMERAS. It is a homography
 * between two bolted-down viewpoints -- it does not change when a new
 * group tees off -- so it is reached from the Cameras page, where a
 * pair is a thing you set up once, as well as from Production, where it
 * was originally bolted onto a card and read as something you do per
 * clip.
 */
import { useRef, useState } from "react";

import { api } from "../api.js";

/* One clickable still, with the marks made on it. Used twice by the
   view-map modal — once per camera — so the two panes cannot drift
   apart in how a click becomes a coordinate. */
export function ClickableStill({ title, frame, marks, pending, colour,
                                onClick, onMoveMark, selected = null,
                                onSelect }) {
  const ref = useRef(null);
  const hasDims = !!(frame?.width && frame?.height);
  // ZOOM IS NOT A CONVENIENCE ON THE TEE PANE. From 180 m back the whole
  // green is about 200px wide and a handful of pixels tall, so a
  // bunker corner is a two-pixel target at fit-to-width -- and the fit
  // is only as good as the clicks. Magnified, the same corner is
  // something you can actually put a cursor on.
  const [zoom, setZoom] = useState(1);
  // Centre of the visible region, as a fraction of the frame.
  const [focus, setFocus] = useState({ x: 0.5, y: 0.5 });

  // The visible sub-rectangle, clamped so the view never leaves the
  // picture. Everything below -- the image offset and the click maths --
  // is derived from these two, so they cannot disagree.
  const span = 1 / zoom;
  const left = Math.max(0, Math.min(1 - span, focus.x - span / 2));
  const top = Math.max(0, Math.min(1 - span, focus.y - span / 2));

  function toFrame(e) {
    if (!hasDims || !ref.current) return null;
    const r = ref.current.getBoundingClientRect();
    const u = left + ((e.clientX - r.left) / r.width) * span;
    const v = top + ((e.clientY - r.top) / r.height) * span;
    return {
      x: Math.max(0, Math.min(frame.width - 1, Math.round(u * frame.width))),
      y: Math.max(0, Math.min(frame.height - 1, Math.round(v * frame.height))),
    };
  }

  // WHICH MARK IS UNDER THE CURSOR, if any. Measured in frame units at
  // the radius the ring is DRAWN at -- which is itself divided by the
  // zoom -- so the grab target is the same size on screen at 1x and at
  // 16x. What looks grabbable is grabbable.
  function markUnder(pt) {
    if (!pt || !onMoveMark) return null;
    const reach = frame.width / 110 / zoom;
    let best = null;
    let bestD = reach;
    marks.forEach((m, i) => {
      const d = Math.hypot(m.x - pt.x, m.y - pt.y);
      if (d <= bestD) { bestD = d; best = i; }
    });
    return best;
  }

  // A drag pans, a click places, and a drag that STARTS on an existing
  // mark moves that mark. The third case is why this component exists
  // in its current form: a calibration used to be append-only, so a
  // pair clicked two pixels off could be corrected only by undoing
  // every pair placed after it. Nudging the one that is wrong is the
  // whole job.
  //
  // A mark is only grabbed when no pair is half-placed. With a pending
  // point waiting for its partner, every click on this side means
  // "here is the partner" -- including one that lands on top of an
  // older mark, which happens whenever two holes share a bunker corner.
  function onDown(e) {
    if (!hasDims || !ref.current) return;
    const startX = e.clientX;
    const startY = e.clientY;
    const grab = pending ? null : markUnder(toFrame(e));
    if (grab != null) {
      const t = e.currentTarget;
      t.setPointerCapture?.(e.pointerId);
      let dragged = false;
      const mv = (ev) => {
        if (!dragged && Math.hypot(ev.clientX - startX,
                                   ev.clientY - startY) < 3) return;
        dragged = true;
        const p = toFrame(ev);
        if (p) onMoveMark(grab, p);
      };
      const done = () => {
        t.releasePointerCapture?.(e.pointerId);
        t.removeEventListener("pointermove", mv);
        t.removeEventListener("pointerup", done);
        // A grab that never moved is a click ON a mark: select it, so
        // the footer can offer to delete that one pair.
        if (!dragged) onSelect?.(grab === selected ? null : grab);
      };
      t.addEventListener("pointermove", mv);
      t.addEventListener("pointerup", done);
      return;
    }
    const r = ref.current.getBoundingClientRect();
    const from = { ...focus };
    let moved = false;
    const target = e.currentTarget;
    target.setPointerCapture?.(e.pointerId);
    const move = (ev) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      if (!moved && Math.hypot(dx, dy) < 4) return;
      moved = true;
      if (zoom <= 1) return;
      setFocus({
        x: Math.min(1, Math.max(0, from.x - (dx / r.width) * span)),
        y: Math.min(1, Math.max(0, from.y - (dy / r.height) * span)),
      });
    };
    const up = (ev) => {
      target.releasePointerCapture?.(e.pointerId);
      target.removeEventListener("pointermove", move);
      target.removeEventListener("pointerup", up);
      if (!moved) {
        const pt = toFrame(ev);
        if (pt) onClick(pt);
      }
    };
    target.addEventListener("pointermove", move);
    target.addEventListener("pointerup", up);
  }

  // Wheel zooms about the cursor, so the feature you are aiming at
  // stays put instead of sliding off as you magnify.
  function onWheel(e) {
    if (!hasDims || !ref.current) return;
    e.preventDefault();
    const r = ref.current.getBoundingClientRect();
    const u = left + ((e.clientX - r.left) / r.width) * span;
    const v = top + ((e.clientY - r.top) / r.height) * span;
    const next = Math.max(1, Math.min(16, zoom * (e.deltaY < 0 ? 1.25 : 0.8)));
    setZoom(next);
    if (next > 1) setFocus({ x: u, y: v });
    else setFocus({ x: 0.5, y: 0.5 });
  }

  const btn = {
    width: "auto", padding: "0 8px", minWidth: 28, background: "rgba(0,0,0,0.6)",
    color: "#fff", border: "1px solid rgba(255,255,255,0.25)",
  };
  return (
    <div style={{ flex: 1, minWidth: 0 }}>
      <div className="row" style={{ alignItems: "center", gap: 6,
                                    marginBottom: 4 }}>
        <span className="tiny upper muted">{title}</span>
        <span className="row" style={{ gap: 4, marginLeft: "auto" }}>
          <button type="button" className="small" style={btn}
                  onClick={() => setZoom((z) => Math.max(1, z / 1.6))}>−</button>
          <span className="tiny muted" style={{ minWidth: 34,
                                                textAlign: "center" }}>
            {zoom.toFixed(1)}×
          </span>
          <button type="button" className="small" style={btn}
                  onClick={() => setZoom((z) => Math.min(16, z * 1.6))}>+</button>
          <button type="button" className="small" style={btn}
                  onClick={() => { setZoom(1); setFocus({ x: 0.5, y: 0.5 }); }}
                  title="Back to the whole frame">Fit</button>
        </span>
      </div>
      <div
        ref={ref}
        onPointerDown={onDown}
        onWheel={onWheel}
        style={{
          position: "relative", width: "100%",
          aspectRatio: hasDims ? `${frame.width} / ${frame.height}` : "16 / 9",
          background: "var(--border, #222)", borderRadius: 6,
          overflow: "hidden", userSelect: "none",
          cursor: zoom > 1 ? "grab" : "crosshair",
          touchAction: "none",
        }}
      >
        {/* The image and its marks share one scaled, offset wrapper, so
            a mark can never drift from the pixel it was placed on. */}
        <div
          style={{
            position: "absolute",
            width: `${zoom * 100}%`, height: `${zoom * 100}%`,
            left: `${-left * zoom * 100}%`, top: `${-top * zoom * 100}%`,
            pointerEvents: "none",
          }}
        >
          {frame?.image_url && (
            <img
              src={frame.image_url}
              alt={title}
              draggable={false}
              style={{ width: "100%", height: "100%", objectFit: "fill",
                       imageRendering: zoom >= 4 ? "pixelated" : "auto" }}
            />
          )}
          {hasDims && (
            <svg
              viewBox={`0 0 ${frame.width} ${frame.height}`}
              preserveAspectRatio="none"
              style={{ position: "absolute", inset: 0, width: "100%",
                       height: "100%" }}
            >
              {marks.map((m, i) => (
                <g key={i}>
                  {/* THE SELECTED PAIR WEARS A HALO on both pictures at
                      once, which is the point of selecting one: pair 4
                      is only meaningful if you can see which dot it is
                      in each view without counting labels. */}
                  {i === selected && (
                    <circle cx={m.x} cy={m.y} r={frame.width / 90 / zoom}
                            fill="none" stroke="#fff"
                            strokeWidth={frame.width / 300 / zoom}
                            opacity={0.9} />
                  )}
                  {/* Sized in SCREEN terms: divided by the zoom so a
                      marker stays the same size on screen instead of
                      swelling into a blob that hides its own target. */}
                  <circle cx={m.x} cy={m.y} r={frame.width / 160 / zoom}
                          fill="none" stroke={colour}
                          strokeWidth={frame.width / 400 / zoom} />
                  <circle cx={m.x} cy={m.y} r={frame.width / 700 / zoom}
                          fill={colour} />
                  <text x={m.x + frame.width / 110 / zoom}
                        y={m.y - frame.width / 220 / zoom}
                        fontSize={frame.width / 40 / zoom} fontWeight={700}
                        fill={i === selected ? "#fff" : colour} stroke="#000"
                        strokeWidth={frame.width / 500 / zoom}
                        paintOrder="stroke">
                    {i + 1}
                  </text>
                </g>
              ))}
              {pending && (
                /* The half-finished pair, dashed amber: it is waiting
                   for its partner in the other picture, unsaved. */
                <circle cx={pending.x} cy={pending.y}
                        r={frame.width / 120 / zoom}
                        fill="none" stroke="#f59e0b"
                        strokeWidth={frame.width / 300 / zoom}
                        strokeDasharray={`${frame.width / 90 / zoom} `
                          + `${frame.width / 140 / zoom}`} />
              )}
            </svg>
          )}
        </div>
        {zoom > 1 && (
          <div className="tiny" style={{
            position: "absolute", left: 6, bottom: 6, color: "#fff",
            background: "rgba(0,0,0,0.6)", padding: "1px 6px",
            borderRadius: 4, pointerEvents: "none",
          }}>
            drag to pan · scroll to zoom · drag a numbered dot to nudge it
          </div>
        )}
      </div>
    </div>
  );
}

/* CALIBRATE THE TWO VIEWS AGAINST EACH OTHER.
 *
 * The tracer has to finish where the ball landed, and the landing is
 * marked on the green camera. Both cameras look at the same flat
 * ground, so one homography carries green pixels to tee pixels — and
 * fitting it needs nothing but the same four ground features clicked in
 * both pictures. No tape measure, no yardage book, no walking out.
 *
 * ON THE GROUND is the one rule. A homography is exact only for points
 * on the surface it was fitted to. Bunker corners, the flagstick's
 * BASE, a bend in the cart path: yes. The top of the flagstick, a
 * treetop, anything in the air: no — it will skew the whole fit.
 */
export function ViewMapModal({
  uploadId, adminPassword, teeFrame, greenFrame, existing, scope,
  onClose, onSaved, mismatch = null,
}) {
  // A MISMATCHED MAP DOES NOT GET LOADED. Its points were clicked on
  // two different cameras' frames, so on these they land in the sky and
  // the tree line -- and the moment they are on screen the operator is
  // one Save away from overwriting the pair they really belong to.
  // Start empty and say why, instead.
  const [pairs, setPairs] = useState(
    () => (mismatch ? [] : existing?.points || []));
  // A saved calibration comes back with its pairs already on the
  // pictures. Say so: an operator who cannot tell a loaded calibration
  // from a blank one re-does work that was already correct.
  const preloaded = mismatch ? 0 : (existing?.points || []).length;
  const [pending, setPending] = useState(null);   // {side, x, y}
  const [saving, setSaving] = useState(false);
  const [note, setNote] = useState(null);
  // WHICH PAIR IS BEING WORKED ON. Selecting one is what makes deleting
  // a single pair possible; before this the only remover was "undo the
  // last one", so correcting pair 2 of 9 meant destroying seven good
  // pairs to reach it.
  const [selected, setSelected] = useState(null);

  // NUDGE ONE END OF ONE PAIR. Dragging a mark rewrites that side's
  // coordinate and leaves the other alone -- the two views are clicked
  // independently and are wrong independently, so a pair whose tee dot
  // is two pixels off does not want its green dot touched.
  function moveMark(side, i, pt) {
    setPairs((p) => p.map((q, j) => (
      j === i ? { ...q, [side]: [pt.x, pt.y] } : q
    )));
    setSelected(i);
  }

  function dropPair(i) {
    setPairs((p) => p.filter((_, j) => j !== i));
    setSelected(null);
  }

  function place(side, pt) {
    if (!pending) { setPending({ side, ...pt }); return; }
    if (pending.side === side) { setPending({ side, ...pt }); return; }
    const pair = pending.side === "tee"
      ? { tee: [pending.x, pending.y], green: [pt.x, pt.y] }
      : { green: [pending.x, pending.y], tee: [pt.x, pt.y] };
    setPairs((p) => [...p, pair]);
    setPending(null);
    setNote(null);
  }

  async function save() {
    setSaving(true);
    setNote(null);
    try {
      const out = await api.saveViewMap(adminPassword, uploadId, {
        points: pairs,
        green_size: [greenFrame?.width, greenFrame?.height],
        tee_size: [teeFrame?.width, teeFrame?.height],
      });
      onSaved?.(out);
    } catch (e) {
      setNote(e?.message || String(e));
    } finally {
      setSaving(false);
    }
  }

  // SIX, not four. Four pairs determine a homography exactly, so they
  // also reproduce every click error exactly with nothing left over to
  // average away — and the tee camera sees the whole green as a sliver
  // about 200px wide and 5px tall, which is the worst possible shape to
  // fit a projective transform into. Measured with 1.5px of click error:
  // four pairs land ~20px out (worst case: off the frame), six land
  // ~1px out. Eight is better still and costs four more clicks.
  const MIN_PAIRS = 6;
  const ready = pairs.length >= MIN_PAIRS;
  return (
    <div
      style={{
        position: "fixed", inset: 0, zIndex: 1200,
        background: "rgba(0,0,0,0.75)", display: "flex",
        alignItems: "center", justifyContent: "center", padding: 16,
      }}
      onClick={onClose}
    >
      <div
        className="card"
        style={{
          margin: 0, maxWidth: 1400, width: "100%", maxHeight: "94vh",
          overflowY: "auto", padding: 14,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row" style={{ alignItems: "baseline", gap: 10 }}>
          <b>Calibrate tee ↔ green</b>
          {scope && <span className="small">{scope}</span>}
          {preloaded > 0 && (
            <span className="tiny" style={{ color: "#3ee37a" }}>
              already mapped — {preloaded} saved pairs loaded. Drag any
              numbered dot to nudge it, click one to delete just that
              pair, or add more. Or just Close; this does not need
              redoing.
            </span>
          )}
          {mismatch && (
            <span className="tiny" style={{ color: "#f87171", maxWidth: 460 }}>
              ⚠ A mapping is stored here, but not for these two cameras —
              so it has NOT been loaded. {mismatch}
            </span>
          )}
          <span className="tiny muted">
            Click the same GROUND feature in both pictures — a bunker
            corner, the flagstick&apos;s BASE, a bend in the cart path.
            Six pairs minimum, eight is better, spread as widely as the
            green allows. Anything off the ground (a flag top, a
            treetop) skews the whole fit. Saved against the HOLE, so it
            aims every swing recorded there — not just this one.
          </span>
          <button
            type="button"
            className="ghost small"
            style={{ width: "auto", marginLeft: "auto" }}
            onClick={onClose}
          >
            Close
          </button>
        </div>

        <div className="row" style={{ gap: 10, marginTop: 8,
                                      alignItems: "flex-start" }}>
          <ClickableStill
            title={`Tee camera · frame ${teeFrame?.frame ?? "—"}`}
            frame={teeFrame}
            colour="#38bdf8"
            marks={pairs.map((p) => ({ x: p.tee[0], y: p.tee[1] }))}
            pending={pending?.side === "tee" ? pending : null}
            onClick={(pt) => place("tee", pt)}
            onMoveMark={(i, pt) => moveMark("tee", i, pt)}
            selected={selected}
            onSelect={setSelected}
          />
          <ClickableStill
            title={`Green camera · frame ${greenFrame?.frame ?? "—"}`}
            frame={greenFrame}
            colour="#22c55e"
            marks={pairs.map((p) => ({ x: p.green[0], y: p.green[1] }))}
            pending={pending?.side === "green" ? pending : null}
            onClick={(pt) => place("green", pt)}
            onMoveMark={(i, pt) => moveMark("green", i, pt)}
            selected={selected}
            onSelect={setSelected}
          />
        </div>

        <div className="row" style={{ gap: 10, marginTop: 10,
                                      alignItems: "center", flexWrap: "wrap" }}>
          <span className="small">
            {pending
              ? `Now click the SAME feature on the ${
                pending.side === "tee" ? "green" : "tee"} picture`
              : selected != null
                ? `Pair ${selected + 1} selected — drag either dot to nudge it`
                : `${pairs.length} pair${pairs.length === 1 ? "" : "s"}`
                  + (ready
                    ? (pairs.length >= 8 ? " — good" : " — enough; 8 is better")
                    : ` — need ${MIN_PAIRS - pairs.length} more`)}
          </span>
          {/* ADJUSTING BEATS REDOING. A calibration is eight pairs of
              hand-clicked pixels and the whole fit is only as good as
              the worst of them, so the useful operation is almost never
              "start again" -- it is "pair 3 is a bit high". Drag it on
              either picture; delete just that one if it is hopeless. */}
          {selected != null && (
            <>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto" }}
                onClick={() => dropPair(selected)}
                title={"Remove this pair only — the ones after it move "
                  + "up a number, and nothing else is touched"}
              >
                Delete pair {selected + 1}
              </button>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto" }}
                onClick={() => setSelected(null)}
              >
                Deselect
              </button>
            </>
          )}
          {pairs.length > 0 && selected == null && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto" }}
              onClick={() => { setPairs((p) => p.slice(0, -1)); setPending(null); }}
            >
              Undo last pair
            </button>
          )}
          {pending && (
            <button
              type="button"
              className="ghost small"
              style={{ width: "auto" }}
              onClick={() => setPending(null)}
            >
              Cancel this point
            </button>
          )}
          <button
            type="button"
            className="small"
            style={{ width: "auto", marginLeft: "auto", minWidth: 160 }}
            disabled={!ready || saving}
            onClick={save}
            title={ready
              ? "Fit the mapping and store it against this hole"
              : "Six pairs minimum — four fit exactly and bake in every "
                + "click error, which on this geometry is ~20px of aim"}
          >
            {saving ? "Fitting…" : "Save mapping"}
          </button>
        </div>
        {/* HELD-OUT ERROR IS THE READOUT — not the residual against the
            fit's own points, which comes back near zero even on a bad
            fit here: with the tee-side points strung along a line the
            homography has freedom left to absorb the click noise
            exactly. In tee pixels, where the whole green is ~200px
            wide, so single digits is good. */}
        {existing?.cv_px != null && (
          <div className="tiny" style={{ marginTop: 6 }}>
            <span style={{
              color: existing.cv_px <= 8 ? "#3ee37a"
                : existing.cv_px <= 20 ? "#f59e0b" : "#ef4444",
            }}>
              Saved fit misses a held-out pair by {existing.cv_px}px
            </span>
            {existing.tee_spread_px && (
              <span className="muted">
                {" "}· the green spans {existing.tee_spread_px[0]}×
                {existing.tee_spread_px[1]}px in the tee view, which is
                why this needs pairs rather than precision
              </span>
            )}
          </div>
        )}
        {existing?.calibrated_at && (
          <div className="tiny muted" style={{ marginTop: 6 }}>
            Currently saved: {existing.n_points} pairs
            {existing.rms_px != null ? `, off by ${existing.rms_px}px` : ", exact fit"}
            {" · "}{existing.calibrated_at.slice(0, 16).replace("T", " ")}
          </div>
        )}
        {note && (
          <div className="err-text small" style={{ marginTop: 8 }}>{note}</div>
        )}
      </div>
    </div>
  );
}
