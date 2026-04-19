import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";

export default function AdminReview() {
  const adminPassword = localStorage.getItem(ADMIN_PW_STORAGE) || "";
  const [events, setEvents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState(null);

  async function load() {
    try {
      const list = await api.listHIO(adminPassword);
      setEvents(list);
      if (list.length && !selected) setSelected(list[0].id);
    } catch (e) { setError(e.message); }
  }

  async function loadDetail(id) {
    try { setDetail(await api.hioDetail(adminPassword, id)); }
    catch (e) { setError(e.message); }
  }

  useEffect(() => { load(); /* eslint-disable-next-line */ }, []);
  useEffect(() => { if (selected) loadDetail(selected); /* eslint-disable-next-line */ }, [selected]);

  async function decide(action) {
    if (!reviewer) { setError("Reviewer name required."); return; }
    try {
      await api.hioDecide(adminPassword, selected, action, reviewer, note);
      setNote("");
      await load();
      await loadDetail(selected);
    } catch (e) { setError(e.message); }
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

  return (
    <div className="wrap wide">
      <Brand subtitle="Operator Console" />
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/participants">Participants</Link>
        <Link to="/admin/review" className="active">Hole-in-one review</Link>
      </div>

      {error && <div className="card" style={{ color: "var(--danger)" }}>{error}</div>}

      <div className="grid" style={{ alignItems: "start", gridTemplateColumns: "minmax(240px, 320px) 1fr" }}>
        <div className="card">
          <h3 style={{ marginBottom: 10 }}>Queue</h3>
          {events.length === 0 && <div className="muted small">Nothing to review.</div>}
          <div className="stack">
            {events.map((e) => (
              <button
                key={e.id}
                onClick={() => setSelected(e.id)}
                className="ghost"
                style={{
                  textAlign: "left",
                  padding: 12,
                  borderRadius: 10,
                  background: selected === e.id ? "var(--primary-soft)" : "transparent",
                  border: selected === e.id ? "1px solid var(--emerald-200)" : "1px solid transparent",
                  width: "100%",
                }}
              >
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <b>Hole {e.hole_number}</b>
                  <span className={`pill ${pillClass(e.status)}`}>{e.status}</span>
                </div>
                <div className="small muted" style={{ marginTop: 2 }}>
                  participant #{e.participant_id} · {new Date(e.created_at).toLocaleString()}
                </div>
              </button>
            ))}
          </div>
        </div>

        {detail ? (
          <div className="card">
            <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 6 }}>
              <h3>Hole {detail.event.hole_number} — {detail.participant.name}</h3>
              <span className={`pill ${pillClass(detail.event.status)}`}>{detail.event.status}</span>
            </div>
            <p className="small muted" style={{ marginBottom: 16 }}>
              {detail.participant.mobile || "—"} · {detail.participant.email || "—"}
            </p>

            <div className="grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {["tee", "wide_green", "hole"].map((angle) => {
                const c = detail.clips.find((x) => x.camera_type === angle);
                return (
                  <div key={angle} className="card tight" style={{ margin: 0 }}>
                    <div className="tiny upper muted" style={{ marginBottom: 6 }}>{angle.replace("_", " ")}</div>
                    {c ? (
                      <>
                        <video
                          src={c.source_url}
                          poster={c.thumbnail_url}
                          controls
                          style={{ width: "100%", borderRadius: 8, background: "#000", aspectRatio: "16/9" }}
                        />
                        {c.ball_in_cup && <span className="pill ok" style={{ marginTop: 6 }}>ball in cup</span>}
                      </>
                    ) : (
                      <div className="muted small center" style={{ padding: "30px 0" }}>no clip</div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="divider" />

            <div className="row">
              <div className="field">
                <label>Reviewer</label>
                <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} placeholder="your name" />
              </div>
              <div className="field" style={{ flex: 2 }}>
                <label>Note (optional)</label>
                <input value={note} onChange={(e) => setNote(e.target.value)} placeholder="e.g. clean ace, camera confirms entry" />
              </div>
            </div>

            <div className="row">
              <button onClick={() => decide("approve")}>
                <Icon name="check" size={16} /> Approve
              </button>
              <button className="secondary" onClick={() => decide("needs_more")}>Request more</button>
              <button className="danger" onClick={() => decide("reject")}>Reject</button>
            </div>
            {detail.event.status !== "pending" && detail.event.reviewer && (
              <p className="small muted" style={{ marginTop: 10 }}>
                Last decided by {detail.event.reviewer}
                {detail.event.decision_note ? ` — "${detail.event.decision_note}"` : ""}
              </p>
            )}
          </div>
        ) : (
          <div className="card muted center" style={{ padding: 40 }}>
            Select an event on the left.
          </div>
        )}
      </div>
    </div>
  );
}

function pillClass(status) {
  if (status === "approved") return "ok";
  if (status === "rejected") return "err";
  return "warn";
}
