/**
 * THE TWO THINGS THAT CHANGE EVERY MORNING, set from the Cameras page.
 *
 * The green->tee calibration is a fact about two bolted-down viewpoints
 * and is done once. These are the opposite: the pin is cut to a new
 * spot each morning and the tee markers are walked forward or back, so
 * they are set daily, by whoever is standing at the camera — which is
 * why they belong here rather than buried in a produce report.
 *
 * Both are stored where the calibration is NOT: the pin against the
 * hole in GREEN pixels (so re-calibrating the views corrects the target
 * instead of leaving a stale tee coordinate behind), the tee box
 * against the course, hole and day. Neither is attached to the upload
 * whose frames are being clicked on — that capture is only a backdrop.
 */
import { useEffect, useState } from "react";

import { api } from "../api.js";
import { ClickableStill } from "./ViewMapModal.jsx";
import { D3TeeBox } from "./TeeBoxEditor.jsx";

export function DailyMarksModal({ adminPassword, cam, onClose }) {
  const [src, setSrc] = useState(null);
  const [err, setErr] = useState(null);
  const [tab, setTab] = useState("green");

  // The flagstick, in green pixels. `pin` is what will be saved; it
  // starts at whatever the hole already has so an operator can see
  // yesterday's pin and judge whether it has actually moved.
  const [pin, setPin] = useState(null);
  const [pinSaving, setPinSaving] = useState(false);
  const [pinNote, setPinNote] = useState(null);

  const [tb, setTb] = useState(null);

  useEffect(() => {
    let dead = false;
    (async () => {
      try {
        const s = await api.cameraCalibrationSource(adminPassword, cam.id);
        const [t, g] = await Promise.all([
          api.getLongUploadFrame(adminPassword, s.upload_id, 0, "tee"),
          api.getLongUploadFrame(adminPassword, s.upload_id, 0, "green"),
        ]);
        if (dead) return;
        setSrc({ ...s, tee: { ...t, frame: 0 }, green: { ...g, frame: 0 } });
        const _p = (s.view_map || {}).pin_green;
        if (_p) setPin({ x: _p[0], y: _p[1] });
        // The tee box comes from its own endpoint because it carries the
        // frame it should be drawn on, the hole and the DAY — none of
        // which the calibration source knows.
        try {
          const b = await api.getTeeBox(adminPassword, s.upload_id);
          if (!dead) setTb(b);
        } catch (e) {
          if (!dead) setTb({ error: e?.message || String(e) });
        }
      } catch (e) {
        if (!dead) setErr(e?.message || String(e));
      }
    })();
    return () => { dead = true; };
  }, [adminPassword, cam.id]);

  async function savePin(clear) {
    setPinSaving(true);
    setPinNote(null);
    try {
      await api.saveHolePin(adminPassword, src.upload_id, {
        green: clear ? null : [pin.x, pin.y],
      });
      if (clear) setPin(null);
      setPinNote(clear
        ? "Cleared — no pin is set for this hole."
        : `Saved. Every swing on hole ${src.hole} will aim at this until `
          + "it is moved again.");
    } catch (e) {
      setPinNote(e?.message || String(e));
    } finally {
      setPinSaving(false);
    }
  }

  const scope = src
    ? (src.course_name ? `${src.course_name} · hole ${src.hole}`
      : `hole ${src.hole}`)
    : "";
  const tabBtn = (k, label) => (
    <button
      type="button"
      className={tab === k ? "small" : "ghost small"}
      style={{ width: "auto" }}
      onClick={() => setTab(k)}
    >
      {label}
    </button>
  );

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
          margin: 0, maxWidth: 1100, width: "100%", maxHeight: "94vh",
          overflowY: "auto", padding: 14,
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="row" style={{ alignItems: "baseline", gap: 10 }}>
          <b>Today&apos;s flag &amp; tee box</b>
          {scope && <span className="small">{scope}</span>}
          <span className="row" style={{ gap: 6, marginLeft: 12 }}>
            {tabBtn("green", "⛳ Green view — flag stick")}
            {tabBtn("tee", "▭ Tee view — tee box")}
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

        {err && <div className="err-text small" style={{ marginTop: 10 }}>{err}</div>}
        {!src && !err && (
          <div className="small" style={{ marginTop: 10 }}>
            Finding a capture from this pair to click on…
          </div>
        )}

        {src && tab === "green" && (
          <div style={{ marginTop: 8 }}>
            <div className="tiny muted" style={{ marginBottom: 6 }}>
              Click the <b>BASE</b> of the flagstick — where it enters the
              ground. The base is the hole; the top of the stick is
              several feet of air above it, and a distance measured to
              the air is wrong by exactly that much. Drag the dot to
              nudge it. Stored in green-camera pixels against hole{" "}
              {src.hole}, so re-calibrating the two views moves the tee
              target with it instead of leaving a stale one behind.
            </div>
            <div style={{ maxWidth: 760 }}>
              <ClickableStill
                title={`Green camera · frame ${src.green?.frame ?? "—"}`}
                frame={src.green}
                colour="#22c55e"
                marks={pin ? [pin] : []}
                pending={null}
                onClick={(pt) => { setPin(pt); setPinNote(null); }}
                onMoveMark={(_i, pt) => { setPin(pt); setPinNote(null); }}
              />
            </div>
            <div className="row" style={{ gap: 10, marginTop: 8,
                                          alignItems: "center",
                                          flexWrap: "wrap" }}>
              <span className="small">
                {pin ? `Flag base at ${pin.x}, ${pin.y}`
                  : "No flag placed — click the picture"}
              </span>
              <button
                type="button"
                className="small"
                style={{ width: "auto", minWidth: 150 }}
                disabled={!pin || pinSaving}
                onClick={() => savePin(false)}
              >
                {pinSaving ? "Saving…" : "Save today's flag"}
              </button>
              <button
                type="button"
                className="ghost small"
                style={{ width: "auto" }}
                disabled={pinSaving}
                onClick={() => savePin(true)}
                title={"Forget the pin for this hole. Better than a "
                  + "stale one: a pin from Tuesday, used on Thursday's "
                  + "swing, measures a distance to somewhere the hole "
                  + "is not."}
              >
                Clear the flag
              </button>
            </div>
            {pinNote && (
              <div className="tiny" style={{ marginTop: 6 }}>{pinNote}</div>
            )}
          </div>
        )}

        {src && tab === "tee" && (
          <div style={{ marginTop: 8 }}>
            {tb?.error ? (
              <div className="err-text small">{tb.error}</div>
            ) : tb ? (
              <D3TeeBox tb={tb} uploadId={src.upload_id}
                        adminPassword={adminPassword} />
            ) : (
              <div className="small">Loading the tee frame…</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}
