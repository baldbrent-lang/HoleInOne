import { useEffect, useMemo, useState, useRef } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import ClipPlayer from "../components/ClipPlayer.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

export default function AdminParticipants() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [params, setParams] = useSearchParams();
  const [courses, setCourses] = useState([]);
  const [rows, setRows] = useState([]);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);
  const [selected, setSelected] = useState(null); // participant for clip drawer
  const [clipData, setClipData] = useState(null); // {course, participant, clips}
  const [toast, setToast] = useState(null);

  const filters = useMemo(
    () => ({
      course_id: params.get("course_id") || "",
      date: params.get("date") || "",
      q: params.get("q") || "",
      paid: params.get("paid") || "",
    }),
    [params],
  );

  function setFilter(k, v) {
    const next = new URLSearchParams(params);
    if (v === "" || v === null || v === undefined) next.delete(k);
    else next.set(k, v);
    setParams(next, { replace: true });
  }

  async function load() {
    if (!adminPassword) return;
    setLoading(true);
    setError(null);
    try {
      const [c, r] = await Promise.all([
        courses.length ? courses : api.listCourses(adminPassword),
        api.listParticipants(adminPassword, {
          course_id: filters.course_id || undefined,
          date: filters.date || undefined,
          q: filters.q || undefined,
          paid: filters.paid || undefined,
        }),
      ]);
      if (!courses.length) setCourses(c);
      setRows(r);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.course_id, filters.date, filters.paid]);

  // Debounced search
  useEffect(() => {
    const t = setTimeout(() => load(), 300);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters.q]);

  async function openClips(p) {
    setSelected(p);
    setClipData(null);
    try {
      const data = await api.participantClips(adminPassword, p.id);
      setClipData(data);
    } catch (e) {
      setClipData({ clips: [], course: {}, participant: {} });
      showToast(`Error: ${e.message}`);
    }
  }

  async function resend(p) {
    // SAY IT ON THE CLICK. The toast used to appear only once the send
    // came back, so on a slow link the button read as dead and the
    // operator clicked it again -- twice-sent gallery links.
    showToast(`Re-sending the gallery link to ${p.name}…`);
    try {
      await api.resendGallery(adminPassword, p.id);
      showToast(`Gallery link re-sent to ${p.name}`);
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  }

  async function thanks(p) {
    // This email sends itself a few hours after the gallery. The button
    // is the "send it now" override, so it says so -- an operator who
    // thinks nothing is scheduled will click it on every row.
    showToast(`Sending the thank-you to ${p.email || p.name}…`);
    try {
      let r = await api.sendThanks(adminPassword, p.id);
      if (r.already_sent) {
        const when = new Date(r.thanks_sent_at + "Z").toLocaleString();
        if (!window.confirm(`${p.name} was already sent the thank-you on ${when}. Send it again?`)) {
          showToast("Left it alone.");
          return;
        }
        r = await api.sendThanks(adminPassword, p.id, true);
      }
      showToast(
        r.review_url
          ? `Thank-you sent to ${p.email || p.name}`
          : `Thank-you sent to ${p.email || p.name} — no REVIEW_URL set, so it went without the review link`,
      );
      load();
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  }

  async function noClips(p) {
    // Confirmed, because this one tells a customer we failed them —
    // sending it to the wrong row is worse than not sending it.
    const already = !!p.refunded_at;
    if (!window.confirm(
      `Tell ${p.name} we didn't record their shots?` +
      (already
        ? " The email will say their refund is on its way."
        : " They have NOT been refunded — the email will offer to put it right."),
    )) return;
    showToast(`Sending to ${p.email || p.name}…`);
    try {
      await api.sendNoClips(adminPassword, p.id, already);
      showToast(`Sent to ${p.email || p.name}`);
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  }

  async function refund(p) {
    if (!window.confirm(`Refund ${p.name}? This issues a Stripe refund (or no-op in mock mode), marks the round as refunded, and EMAILS ${p.email || "the golfer"} to say the money is on its way. Their clips stay accessible.`)) return;
    showToast(`Refunding ${p.name}…`);
    try {
      const r = await api.refundParticipant(adminPassword, p.id);
      if (r.already_refunded) {
        showToast(`${p.name} was already refunded — no second email sent.`);
      } else {
        showToast(
          r.notified
            ? `Refund issued (${r.mode || "ok"}) and ${p.email} emailed`
            : `Refund issued (${r.mode || "ok"}) — no email on file`,
        );
      }
      load();
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  }

  // One timer, not one per toast. A "sending…" toast followed by its
  // result used to leave the first timer running, which cleared the
  // RESULT a moment after it appeared.
  const toastTimer = useRef(null);
  function showToast(msg) {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2200);
  }

  function clearFilters() {
    setParams({}, { replace: true });
  }

  if (!adminPassword) {
    return (
      <div className="wrap">
        <Brand subtitle="Operator Console" />
        <div className="card center">
          <h2>Admin password required</h2>
          <Link to="/admin"><button style={{ marginTop: 10 }}>Sign in</button></Link>
        </div>
      </div>
    );
  }

  const counts = {
    paid: rows.filter((r) => r.paid).length,
    withClips: rows.filter((r) => r.clips.assigned > 0).length,
  };

  return (
    <div className="wrap wide">
      <Brand subtitle="Operator Console" />
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants" className="active">Players</Link>
        <Link to="/admin/courses">Courses</Link>
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production">Production</Link>
        <Link to="/admin/produced-clips">Produced Clips</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
      </div>

      <div className="card">
        <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 12 }}>
          <h3>Players</h3>
          <span className="small muted">
            {rows.length} result{rows.length === 1 ? "" : "s"}
            {" · "}
            {counts.paid} paid
            {" · "}
            {counts.withClips} with clips
          </span>
        </div>

        <div className="row" style={{ flexWrap: "wrap" }}>
          <div className="field" style={{ flex: "1 1 200px", marginBottom: 0 }}>
            <label>Search</label>
            <input
              value={filters.q}
              onChange={(e) => setFilter("q", e.target.value)}
              placeholder="Name, email, or mobile"
            />
          </div>
          <div className="field" style={{ flex: "1 1 180px", marginBottom: 0 }}>
            <label>Course</label>
            <select value={filters.course_id} onChange={(e) => setFilter("course_id", e.target.value)}>
              <option value="">All courses</option>
              {courses.map((c) => (
                <option key={c.id} value={c.id}>{c.name}</option>
              ))}
            </select>
          </div>
          <div className="field" style={{ flex: "1 1 160px", marginBottom: 0 }}>
            <label>Date</label>
            <input
              type="date"
              value={filters.date}
              onChange={(e) => setFilter("date", e.target.value)}
            />
          </div>
          <div className="field" style={{ flex: "1 1 140px", marginBottom: 0 }}>
            <label>Payment</label>
            <select value={filters.paid} onChange={(e) => setFilter("paid", e.target.value)}>
              <option value="">Any</option>
              <option value="true">Paid only</option>
              <option value="false">Unpaid only</option>
            </select>
          </div>
          <div style={{ alignSelf: "end" }}>
            <button className="ghost" onClick={clearFilters}>Clear</button>
          </div>
        </div>
      </div>

      {error && <div className="card" style={{ color: "var(--danger)" }}>{error}</div>}

      <div className="card" style={{ padding: 0, overflow: "hidden" }}>
        {loading && rows.length === 0 ? (
          <div style={{ padding: 20 }}>
            <div className="shimmer" style={{ height: 24, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 24, marginBottom: 8 }} />
            <div className="shimmer" style={{ height: 24 }} />
          </div>
        ) : rows.length === 0 ? (
          <div className="muted center" style={{ padding: 30 }}>
            No players match your filters.
          </div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Golfer</th>
                <th>Course</th>
                <th>Tee time</th>
                <th>Clips</th>
                <th>Status</th>
                <th style={{ width: 1 }}></th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => (
                <tr key={p.id}>
                  <td>
                    <div className="inline" style={{ gap: 10 }}>
                      {p.selfie_url ? (
                        <img
                          src={p.selfie_url}
                          alt=""
                          style={{ width: 36, height: 48, objectFit: "cover", borderRadius: 6, border: "1px solid var(--border)" }}
                        />
                      ) : (
                        <div
                          style={{
                            width: 36, height: 48, borderRadius: 6, background: "var(--surface-alt)",
                            display: "grid", placeItems: "center", color: "var(--ink-mute)",
                            fontSize: "0.8rem",
                          }}
                        >?</div>
                      )}
                      <div>
                        <b>{p.name}</b>
                        <div className="small muted">
                          {p.mobile || "—"} {p.email ? `· ${p.email}` : ""}
                        </div>
                      </div>
                    </div>
                  </td>
                  <td className="small">{p.course.name}<div className="muted tiny upper">{p.course.location}</div></td>
                  <td className="small">
                    {new Date(p.tee_time.starts_at).toLocaleString([], {
                      month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
                    })}
                  </td>
                  <td className="small">
                    <b>{p.clips.assigned}</b>
                    {p.clips.total !== p.clips.assigned && <span className="muted"> / {p.clips.total}</span>}
                  </td>
                  <td>
                    {p.paid
                      ? <span className="pill ok">Paid</span>
                      : (p.refunded_at
                        ? <span className="pill warn">Refunded</span>
                        : <span className="pill err">Unpaid</span>)}
                  </td>
                  <td>
                    <div style={{ display: "flex", gap: 6, whiteSpace: "nowrap" }}>
                      <button className="secondary small" onClick={() => openClips(p)}>
                        View clips
                      </button>
                      <a
                        className="btn secondary small"
                        href={`/g/${p.gallery_token}`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        Gallery ↗
                      </a>
                      <button className="ghost small" onClick={() => resend(p)} title="Re-send gallery link">
                        Resend
                      </button>
                      <button
                        className="ghost small"
                        onClick={() => thanks(p)}
                        title="Send the thank-you / review request now, ahead of its 4-hour timer"
                      >
                        Thank you
                      </button>
                      <button
                        className="ghost small"
                        onClick={() => noClips(p)}
                        title="Tell this golfer we didn't record their shots"
                      >
                        No clips
                      </button>
                      {p.paid && !p.refunded_at && (
                        <button className="ghost small" onClick={() => refund(p)} title="Refund this round" style={{ color: "var(--danger)" }}>
                          Refund
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {selected && (
        <div className="card" style={{ marginTop: 18 }}>
          <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 12 }}>
            <h3>Clips for {selected.name}</h3>
            <button className="ghost small" onClick={() => { setSelected(null); setClipData(null); }}>Close</button>
          </div>
          {!clipData ? (
            <div className="shimmer" style={{ height: 80 }} />
          ) : clipData.clips.length === 0 ? (
            <div className="muted small">No clips yet for this golfer.</div>
          ) : (
            <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
              {clipData.clips.map((c) => (
                <div key={c.id} className="card tight" style={{ margin: 0 }}>
                  <ClipPlayer
                    clip={c}
                    courseName={clipData.course?.name}
                    golferName={clipData.participant?.name}
                    yardage={clipData.course?.hole_yardages?.[String(c.hole_number)]}
                  />
                  <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginTop: 8 }}>
                    <b>Hole {c.hole_number}</b>
                    <span className={`pill ${c.processing_status === "assigned" ? "ok" : c.processing_status === "flagged" ? "err" : "warn"}`}>
                      {c.processing_status}
                    </span>
                  </div>
                  <div className="small muted" style={{ marginTop: 4 }}>
                    {c.camera_type.replace("_", " ")}
                    {c.carry_yards ? ` · ${c.carry_yards} yd carry` : ""}
                    {c.ball_speed_mph ? ` · ${c.ball_speed_mph} mph` : ""}
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
