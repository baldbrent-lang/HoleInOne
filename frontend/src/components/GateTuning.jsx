/**
 * PER-HOLE GATE TUNING, edited where the gates are seen to fail.
 *
 * The report already prints, for every refused chain, the measured
 * value beside the limit it broke: "rises at 0.9, outside 1.0-12.0".
 * That sentence is the whole diagnosis, and until now the only way to
 * act on it was to change a constant and redeploy — which changed it
 * for every hole, including the ones it was already right for.
 *
 * So the limits are editable here, beside the evidence. Everything is
 * keyed per camera pair and hole: tuning hole 8 cannot move hole 3.
 *
 * ONLY DIFFERENCES ARE STORED. A hole tuned to exactly the defaults is
 * an untuned hole, and saving it as a copy of today's numbers would
 * pin it there — a later change to a default would then lift every
 * hole except the ones somebody had "confirmed".
 */
import { useEffect, useState } from "react";

import { api } from "../api.js";

export function GateTuning({ adminPassword, uploadId, onSaved }) {
  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [draft, setDraft] = useState({});     // {key: string} being typed
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState(null);

  async function load() {
    setErr(null);
    try {
      const d = await api.getGateTuning(adminPassword, uploadId);
      setData(d);
      setDraft({});
    } catch (e) {
      setErr(e?.message || String(e));
    }
  }
  useEffect(() => { load(); /* eslint-disable-next-line */ }, [uploadId]);

  if (err) return <div className="err-text tiny">{err}</div>;
  if (!data) return <div className="tiny muted">Loading this hole&apos;s gates…</div>;

  const spec = data.spec || {};
  const values = data.values || {};
  const defaults = data.defaults || {};
  const tuned = data.tuned || {};
  // What would be sent: the saved overrides, with anything typed in
  // this session laid over them. A field cleared to empty means "back
  // to the default", which is the same as not sending it.
  const pending = () => {
    const out = { ...tuned };
    for (const [k, v] of Object.entries(draft)) {
      if (v === "") delete out[k];
      else if (Number.isFinite(Number(v))) out[k] = Number(v);
    }
    return out;
  };
  const shown = (k) => (draft[k] !== undefined ? draft[k]
    : (tuned[k] !== undefined ? String(tuned[k]) : ""));
  const dirty = Object.keys(draft).length > 0;
  const nTuned = Object.keys(pending()).length;

  async function save(clearAll) {
    setBusy(true);
    setNote(null);
    try {
      const out = await api.setGateTuning(
        adminPassword, uploadId, clearAll ? {} : pending());
      setNote(Object.keys(out.tuned || {}).length
        ? `Saved against ${out.key} — ${Object.keys(out.tuned).length} `
          + "gate(s) tuned. Re-produce to see it applied."
        : "Cleared — this hole is back on the defaults.");
      await load();
      onSaved?.(out);
    } catch (e) {
      setNote(e?.message || String(e));
    } finally {
      setBusy(false);
    }
  }

  const rows = Object.keys(spec);
  const asc = rows.filter((k) => k.startsWith("ascent_"));
  const desc = rows.filter((k) => k.startsWith("descent_"));
  const label = (k) => k.replace(/^(ascent|descent)_/, "").replace(/_/g, " ");

  const group = (title, keys) => (
    <div style={{ flex: 1, minWidth: 260 }}>
      <div className="tiny upper muted" style={{ marginBottom: 4 }}>{title}</div>
      <table className="tiny" style={{ width: "100%" }}>
        <tbody>
          {keys.map((k) => {
            const sp = spec[k] || {};
            const isTuned = pending()[k] !== undefined;
            return (
              <tr key={k}>
                <td style={{ paddingRight: 6, whiteSpace: "nowrap" }}>
                  <span style={{ color: isTuned ? "#f59e0b" : undefined,
                                 fontWeight: isTuned ? 700 : 400 }}>
                    {label(k)}
                  </span>
                  {sp.verdict && (
                    <span className="muted"> · {sp.verdict}</span>
                  )}
                </td>
                <td style={{ paddingRight: 6, textAlign: "right" }}
                    className="muted">
                  {defaults[k]}
                </td>
                <td>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={shown(k)}
                    placeholder={String(defaults[k])}
                    onChange={(e) => setDraft((d) => (
                      { ...d, [k]: e.target.value.replace(/[^\d.\-]/g, "") }
                    ))}
                    title={`${sp.int ? "Whole number" : "Number"} between `
                      + `${sp.lo} and ${sp.hi}. Blank = use the default `
                      + `(${defaults[k]}). In force now: ${values[k]}.`}
                    style={{ width: 72, padding: "0 4px", fontSize: 12,
                             textAlign: "right" }}
                  />
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );

  return (
    <div style={{ marginTop: 8 }}>
      <div className="tiny muted" style={{ marginBottom: 6 }}>
        Gates for <b>{data.course_name || "this course"} · hole {data.hole}</b>,
        filed under <code>{data.key}</code> ({data.key_reason}). The middle
        column is the default; type a number to override it for this hole
        only, or leave it blank to use the default. Nothing here touches
        any other hole.
      </div>
      <div className="row" style={{ gap: 16, flexWrap: "wrap",
                                    alignItems: "flex-start" }}>
        {group("Ascent — leaving the tee", asc)}
        {group("Descent — arriving on the green", desc)}
      </div>
      <div className="row" style={{ gap: 8, marginTop: 8,
                                    alignItems: "center", flexWrap: "wrap" }}>
        <span className="tiny muted">
          {nTuned
            ? `${nTuned} gate${nTuned === 1 ? "" : "s"} tuned for this hole`
            : "Running the defaults"}
        </span>
        <button type="button" className="small" style={{ width: "auto" }}
                disabled={busy || !dirty} onClick={() => save(false)}>
          {busy ? "Saving…" : "Save gates for this hole"}
        </button>
        {dirty && (
          <button type="button" className="ghost small" style={{ width: "auto" }}
                  disabled={busy} onClick={() => setDraft({})}>
            Discard edits
          </button>
        )}
        {Object.keys(tuned).length > 0 && (
          <button type="button" className="ghost small" style={{ width: "auto" }}
                  disabled={busy} onClick={() => save(true)}
                  title={"Forget every override on this hole and run the "
                    + "module defaults again"}>
            Reset this hole to defaults
          </button>
        )}
      </div>
      {note && <div className="tiny" style={{ marginTop: 6 }}>{note}</div>}
    </div>
  );
}
