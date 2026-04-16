import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";

const ADMIN_KEY_STORAGE = "parone.adminKey";

export default function AdminReview() {
  const adminKey = localStorage.getItem(ADMIN_KEY_STORAGE) || "";
  const [events, setEvents] = useState([]);
  const [selected, setSelected] = useState(null);
  const [detail, setDetail] = useState(null);
  const [reviewer, setReviewer] = useState("");
  const [note, setNote] = useState("");
  const [error, setError] = useState(null);

  async function load() {
    try {
      const list = await api.listHIO(adminKey);
      setEvents(list);
      if (list.length && !selected) setSelected(list[0].id);
    } catch (e) {
      setError(e.message);
    }
  }

  async function loadDetail(id) {
    try {
      setDetail(await api.hioDetail(adminKey, id));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => { load(); }, []);
  useEffect(() => { if (selected) loadDetail(selected); }, [selected]);

  async function decide(action) {
    if (!reviewer) {
      setError("Reviewer name required.");
      return;
    }
    try {
      await api.hioDecide(adminKey, selected, action, reviewer, note);
      setNote("");
      await load();
      await loadDetail(selected);
    } catch (e) {
      setError(e.message);
    }
  }

  if (!adminKey) {
    return (
      <div className="wrap">
        <div className="card">
          <h1>Admin key required</h1>
          <Link to="/admin">Sign in first</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap wide">
      <div className="brand"><div className="dot" /><h1>Par One — HIO Review</h1></div>
      <div className="nav">
        <Link to="/admin">Dashboard</Link>
        <Link to="/admin/review" className="active">Hole-in-one review</Link>
      </div>
      {error && <div className="card" style={{ color: "var(--danger)" }}>{error}</div>}
      <div className="grid" style={{ alignItems: "start" }}>
        <div className="card">
          <h3>Queue</h3>
          {events.length === 0 && <div className="muted small">Nothing to review.</div>}
          {events.map((e) => (
            <div
              key={e.id}
              onClick={() => setSelected(e.id)}
              style={{
                padding: 10, borderRadius: 10, cursor: "pointer",
                background: selected === e.id ? "var(--panel2)" : "transparent",
                marginBottom: 6,
              }}
            >
              <b>Hole {e.hole_number}</b>{" "}
              <span className={`pill ${pillClass(e.status)}`}>{e.status}</span>
              <div className="small muted">participant #{e.participant_id} · {new Date(e.created_at).toLocaleString()}</div>
            </div>
          ))}
        </div>

        {detail ? (
          <div className="card">
            <h3>Hole {detail.event.hole_number} — {detail.participant.name}</h3>
            <p className="small muted">{detail.participant.mobile} · {detail.participant.email}</p>

            <div className="grid" style={{ gridTemplateColumns: "repeat(3, 1fr)" }}>
              {["tee", "wide_green", "hole"].map((angle) => {
                const c = detail.clips.find((c) => c.camera_type === angle);
                return (
                  <div key={angle} className="card" style={{ margin: 0 }}>
                    <div className="muted small">{angle.replace("_", " ")}</div>
                    {c ? (
                      <>
                        <video
                          src={c.source_url}
                          poster={c.thumbnail_url}
                          controls
                          style={{ width: "100%", borderRadius: 8, background: "#000" }}
                        />
                        {c.ball_in_cup && <span className="pill warn">ball in cup</span>}
                      </>
                    ) : (
                      <div className="muted small">no clip</div>
                    )}
                  </div>
                );
              })}
            </div>

            <div className="field" style={{ marginTop: 16 }}>
              <label>Reviewer</label>
              <input value={reviewer} onChange={(e) => setReviewer(e.target.value)} />
            </div>
            <div className="field">
              <label>Note</label>
              <textarea rows={2} value={note} onChange={(e) => setNote(e.target.value)} />
            </div>
            <div className="row">
              <button onClick={() => decide("approve")}>Approve</button>
              <button className="secondary" onClick={() => decide("needs_more")}>Request more</button>
              <button className="danger" onClick={() => decide("reject")}>Reject</button>
            </div>
            {detail.event.status !== "pending" && (
              <p className="small muted" style={{ marginTop: 10 }}>
                Previously decided by {detail.event.reviewer || "?"}:{" "}
                <span className={`pill ${pillClass(detail.event.status)}`}>{detail.event.status}</span>
              </p>
            )}
          </div>
        ) : (
          <div className="card muted">Select an event on the left.</div>
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
