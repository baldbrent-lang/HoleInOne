import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

/**
 * AI-only tracer iteration page. Runs the Claude Vision tracer end-to-end
 * with NO classical CV fallback. The intent is to evaluate the pure AI
 * pipeline in isolation so we can tune prompt + sampling strategy
 * without classical results muddying the comparison.
 */
export default function AdminClipsAi() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [clips, setClips] = useState(null);
  const [error, setError] = useState(null);
  const [running, setRunning] = useState({});  // {clip_id: true}
  const [results, setResults] = useState({});  // {clip_id: payload}

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

  async function aiTraceOne(clipId) {
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
        <h3 style={{ marginBottom: 4 }}>AI tracer test bench</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Runs the Claude Vision tracer with <b>no classical CV fallback</b>.
          Pipeline: (1) send a labeled 16-frame montage spanning the clip and
          ask Claude which frame is impact; (2) sample ~10 frames over the
          next 2.5s and ask Claude for ball coordinates in each; (3) fit a
          parabola through the verified anchors and render the dashed overlay.
          Cyan rings mark frames Claude verified directly. ~11 API calls
          per clip, typically 10-15s wall time.
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
        const aiUrl = result?.ai_tracer_url;
        const info = result?.ai_info;
        const aiDebugUrl = result?.ai_debug_url;
        const isComposite = (c.source_url || "").includes("_composite");
        const anchors = info?.ai_info?.anchors || [];
        const impactFrame = info?.ai_info?.impact_frame;
        const impactDiag = info?.ai_info?.impact_diag || {};
        const stopReason = info?.ai_info?.stop_reason;
        const model = info?.ai_info?.model;
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
                  AI tracer
                  {info?.n_points != null && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · {info.n_points} pts
                      {info.frame_range && <> · frames {info.frame_range[0]}–{info.frame_range[1]}</>}
                    </span>
                  )}
                </div>
                {aiUrl ? (
                  <video
                    src={aiUrl}
                    controls
                    style={{ width: "100%", borderRadius: 8, background: "#000" }}
                  />
                ) : (
                  <div className="muted center" style={{
                    aspectRatio: "16/9", display: "grid", placeItems: "center",
                    border: "2px dashed var(--border)", borderRadius: 8,
                  }}>
                    {running[c.id] ? "Running Claude…" : "No AI trace yet"}
                  </div>
                )}
              </div>
            </div>

            {(model || stopReason || impactFrame != null) && (
              <div className="small muted" style={{ marginTop: 8 }}>
                {model && <>Model: <code>{model}</code> · </>}
                {impactFrame != null && (
                  <>Impact frame: <code>{impactFrame}</code>
                    {impactDiag.confidence && <> ({impactDiag.confidence})</>}
                    {impactDiag.notes && <> — <i>{impactDiag.notes}</i></>}
                    {" · "}
                  </>
                )}
                {stopReason && <>Stop: <code>{stopReason}</code></>}
              </div>
            )}

            {anchors.length > 0 && (
              <details style={{ marginTop: 8 }}>
                <summary className="small muted" style={{ cursor: "pointer" }}>
                  Per-frame tracking results ({anchors.filter(a => a.found).length}/{anchors.length} found)
                </summary>
                <div style={{ marginTop: 8, fontSize: 12 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ textAlign: "left", borderBottom: "1px solid var(--border)" }}>
                        <th style={{ padding: "4px 8px" }}>Frame</th>
                        <th style={{ padding: "4px 8px" }}>Found</th>
                        <th style={{ padding: "4px 8px" }}>Confidence</th>
                        <th style={{ padding: "4px 8px" }}>Coords</th>
                        <th style={{ padding: "4px 8px" }}>Notes</th>
                      </tr>
                    </thead>
                    <tbody>
                      {anchors.map((a, i) => (
                        <tr key={`${a.frame}-${i}`} style={{ borderBottom: "1px solid var(--border)" }}>
                          <td style={{ padding: "4px 8px" }}><code>{a.frame}</code></td>
                          <td style={{ padding: "4px 8px" }}>
                            <span className={`pill small ${a.found ? "ok" : "warn"}`}>
                              {a.found ? "ball" : "no ball"}
                            </span>
                          </td>
                          <td style={{ padding: "4px 8px" }}>{a.confidence}</td>
                          <td style={{ padding: "4px 8px" }}>
                            {a.x != null && a.y != null ? `(${a.x}, ${a.y})` : "—"}
                          </td>
                          <td style={{ padding: "4px 8px" }} className="muted">{a.notes}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </details>
            )}

            {aiDebugUrl && (
              <div style={{ marginTop: 10 }}>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  AI anchor montage
                </div>
                <img
                  src={aiDebugUrl}
                  alt="Claude anchor frames"
                  style={{ width: "100%", maxWidth: 1200, borderRadius: 8, border: "1px solid var(--border)" }}
                />
              </div>
            )}

            {result?.error && (
              <p className="small err-text" style={{ marginTop: 8 }}>
                AI trace failed: <code>{result.error}</code>
              </p>
            )}

            {info && info.ok === false && !result?.error && (
              <p className="small muted" style={{ marginTop: 8 }}>
                AI tracer couldn't find a flight
                {info.error ? <> — <code>{info.error}</code></> : null}.
              </p>
            )}

            <div className="row" style={{ marginTop: 12 }}>
              <button
                onClick={() => aiTraceOne(c.id)}
                disabled={!!running[c.id] || isComposite}
                title={isComposite ? "Composite clips need raw halves" : "Run AI-only tracer (no classical CV fallback)"}
              >
                <Icon name="play" size={14} />{" "}
                {running[c.id] ? "Running Claude…" : "Run AI tracer"}
              </button>
              {isComposite && (
                <span className="small muted" style={{ alignSelf: "center" }}>
                  Composite — AI tracer doesn't support stitched halves
                </span>
              )}
            </div>
          </div>
        );
      })}
    </div>
  );
}

