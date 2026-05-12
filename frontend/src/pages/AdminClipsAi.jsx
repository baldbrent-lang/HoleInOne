import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

/**
 * AI tracer test bench — step 1: detect handedness.
 *
 * The plan is to build the AI tracer up one capability at a time. This
 * page currently exercises just one thing: send a few frames to Claude
 * and ask whether the golfer is right- or left-handed. Each subsequent
 * step (ball at rest, impact, tracking, render) will be layered in.
 */
export default function AdminClipsAi() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [clips, setClips] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState({});
  const [results, setResults] = useState({});

  async function load() {
    try {
      const list = await api.listAllClips(adminPassword);
      setClips(list);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (adminPassword) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  async function runOne(clipId) {
    setRunning((r) => ({ ...r, [clipId]: true }));
    try {
      const data = await api.aiTrace(adminPassword, clipId);
      setResults((r) => ({ ...r, [clipId]: data }));
    } catch (e) {
      setResults((r) => ({ ...r, [clipId]: { error: e.message } }));
    } finally {
      setRunning((r) => ({ ...r, [clipId]: false }));
    }
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
        <Link to="/admin/upload">Upload clip</Link>
        <Link to="/admin/long-upload">Long upload</Link>
        <Link to="/admin/clips">All clips</Link>
        <Link to="/admin/clips/ai" className="active">AI tracer</Link>
        <Link to="/admin/showcase">Home videos</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>AI tracer — step 1: handedness</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Rebuilding the AI tracer one step at a time. Right now, hitting
          <b> Run</b> sends a few sampled frames to Claude and asks whether
          the golfer is right- or left-handed. That's it. We'll layer in
          ball-at-rest, impact detection, tracking, and rendering one
          step at a time — each verified before moving on.
        </p>
      </div>

      {error && (
        <div className="card err-text small">{error}</div>
      )}

      {clips === null && (
        <div className="card"><div className="shimmer" style={{ height: 200 }} /></div>
      )}

      {clips?.length === 0 && (
        <div className="card muted center" style={{ padding: 40 }}>
          No clips uploaded yet. <Link to="/admin/upload">Upload one →</Link>
        </div>
      )}

      {clips?.map((c) => {
        const result = results[c.id];
        const hand = result?.handedness;
        const isComposite = (c.source_url || "").includes("_composite");
        return (
          <div key={c.id} className="card" style={{ marginBottom: 12 }}>
            <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 8 }}>
              <div>
                <b>Clip #{c.id}</b>{" "}
                <span className="muted small">
                  · {c.course_name || `course #${c.course_id}`} · hole {c.hole_number}{" "}
                  · {c.captured_at ? new Date(c.captured_at).toLocaleString() : "—"}
                </span>
              </div>
              <div className="inline" style={{ gap: 8 }}>
                {c.participant_name ? (
                  <span className="pill ok small">{c.participant_name}</span>
                ) : (
                  <span className="pill warn small">unassigned</span>
                )}
                {c.ball_in_cup && <span className="pill ok small">ace</span>}
              </div>
            </div>

            <div className="grid" style={{ gridTemplateColumns: "1fr 1fr", gap: 12 }}>
              <div>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>Source</div>
                <video
                  src={c.source_url}
                  poster={c.thumbnail_url || undefined}
                  controls
                  style={{ width: "100%", borderRadius: 8, background: "#000" }}
                />
              </div>
              <div>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Handedness
                </div>
                <div
                  style={{
                    aspectRatio: "16/9",
                    display: "grid",
                    placeItems: "center",
                    border: "2px dashed var(--border)",
                    borderRadius: 8,
                    padding: 16,
                    textAlign: "center",
                  }}
                >
                  {running[c.id] ? (
                    <div className="muted">Asking Claude…</div>
                  ) : hand?.ok ? (
                    <div style={{ width: "100%" }}>
                      <div style={{ fontSize: 32, fontWeight: 700, textTransform: "uppercase" }}>
                        {hand.handedness}
                      </div>
                      <div className="small muted" style={{ marginTop: 6 }}>
                        confidence: <b>{hand.confidence || "—"}</b>
                        {hand.camera_position && (
                          <> · camera: <b>{hand.camera_position}</b></>
                        )}
                      </div>
                      {hand.notes && (
                        <div className="small muted" style={{ marginTop: 4, fontStyle: "italic" }}>
                          {hand.notes}
                        </div>
                      )}
                      {Array.isArray(hand.per_frame) && hand.per_frame.length > 0 && (
                        <details style={{ marginTop: 8, textAlign: "left" }}>
                          <summary className="small muted" style={{ cursor: "pointer" }}>
                            Per-frame reasoning ({hand.per_frame.length})
                          </summary>
                          <table style={{ width: "100%", marginTop: 6, fontSize: 12, borderCollapse: "collapse" }}>
                            <thead>
                              <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                                <th style={{ padding: "3px 6px" }}>Frame</th>
                                <th style={{ padding: "3px 6px" }}>Phase</th>
                                <th style={{ padding: "3px 6px" }}>Evidence</th>
                              </tr>
                            </thead>
                            <tbody>
                              {hand.per_frame.map((pf, i) => (
                                <tr key={`${pf.frame}-${i}`} style={{ borderBottom: "1px solid var(--border)" }}>
                                  <td style={{ padding: "3px 6px" }}><code>{pf.frame}</code></td>
                                  <td style={{ padding: "3px 6px" }}>{pf.phase}</td>
                                  <td style={{ padding: "3px 6px" }} className="muted">{pf.evidence}</td>
                                </tr>
                              ))}
                            </tbody>
                          </table>
                        </details>
                      )}
                      {hand.frames_sent && (
                        <div className="tiny muted" style={{ marginTop: 6 }}>
                          Frames sent: [{hand.frames_sent.join(", ")}] · model {hand.model}
                        </div>
                      )}
                    </div>
                  ) : hand?.error ? (
                    <div className="err-text small">
                      Error: <code>{hand.error}</code>
                    </div>
                  ) : (
                    <div className="muted">No result yet</div>
                  )}
                </div>
              </div>
            </div>

            {result?.error && (
              <p className="small err-text" style={{ marginTop: 8 }}>
                Request failed: <code>{result.error}</code>
              </p>
            )}

            <div className="row" style={{ marginTop: 12 }}>
              <button
                onClick={() => runOne(c.id)}
                disabled={!!running[c.id] || isComposite}
                title={isComposite ? "Composite clips not supported" : "Ask Claude whether this golfer is right- or left-handed"}
              >
                <Icon name="play" size={14} />{" "}
                {running[c.id] ? "Running…" : "Run handedness check"}
              </button>
              {isComposite && (
                <span className="small muted" style={{ alignSelf: "center" }}>
                  Composite — not supported
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}
