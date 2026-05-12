import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

/**
 * AI tracer test bench — step 2: identify the address frame.
 *
 * Camera is always behind the golfer. Pressing the button sends ~12
 * frames spanning the clip to Claude and asks it to pick the single
 * frame closest to address (golfer set up over the ball, just before
 * takeaway begins). The picked frame is rendered next to the source
 * video so we can verify visually.
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
        <h3 style={{ marginBottom: 4 }}>AI tracer — step 2: address frame</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Camera is assumed to be <b>behind the golfer</b>, hitting toward a
          target away from the camera. Hitting <b>Find address frame</b> sends
          12 evenly-spaced frames to Claude in one request and asks it to
          pick the single frame closest to address (set up over the ball,
          just before takeaway). The picked frame appears next to the
          source video for visual verification.
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
        const addr = result?.address;
        const addrImage = result?.address_image_url;
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
                  Address frame
                  {addr?.address_frame != null && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · frame <code>{addr.address_frame}</code>
                    </span>
                  )}
                </div>
                {running[c.id] ? (
                  <div
                    style={{
                      aspectRatio: "16/9", display: "grid", placeItems: "center",
                      border: "2px dashed var(--border)", borderRadius: 8,
                    }}
                    className="muted"
                  >
                    Asking Claude…
                  </div>
                ) : addrImage ? (
                  <img
                    src={addrImage}
                    alt="Claude's pick for address frame"
                    style={{ width: "100%", borderRadius: 8, background: "#000", display: "block" }}
                  />
                ) : addr?.error ? (
                  <div
                    style={{
                      aspectRatio: "16/9", display: "grid", placeItems: "center",
                      border: "2px dashed var(--border)", borderRadius: 8,
                      padding: 16, textAlign: "center",
                    }}
                    className="err-text small"
                  >
                    Error: <code>{addr.error}</code>
                  </div>
                ) : (
                  <div
                    style={{
                      aspectRatio: "16/9", display: "grid", placeItems: "center",
                      border: "2px dashed var(--border)", borderRadius: 8,
                    }}
                    className="muted"
                  >
                    No result yet
                  </div>
                )}
              </div>
            </div>

            {addr?.ok && (
              <div className="small muted" style={{ marginTop: 8 }}>
                Address confidence: <b>{addr.confidence || "—"}</b>
                {addr.notes && (
                  <>{" "}— <i>{addr.notes}</i></>
                )}
              </div>
            )}
            {addr?.frames_sent && (
              <div className="tiny muted" style={{ marginTop: 4 }}>
                Candidate frames sent: [{addr.frames_sent.join(", ")}] · model {addr.model}
              </div>
            )}

            {hand && (
              <div
                style={{
                  marginTop: 10,
                  padding: 10,
                  borderRadius: 8,
                  background: "var(--surface-2, rgba(0,0,0,0.03))",
                }}
              >
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Handedness (from address frame)
                </div>
                {hand.ok ? (
                  <div>
                    <span
                      style={{
                        fontSize: 22,
                        fontWeight: 700,
                        textTransform: "uppercase",
                        marginRight: 10,
                      }}
                    >
                      {hand.handedness}
                    </span>
                    <span className="small muted">
                      confidence: <b>{hand.confidence || "—"}</b>
                    </span>
                    {(hand.hands_x != null || hand.clubhead_x != null) && (
                      <div className="tiny muted" style={{ marginTop: 4 }}>
                        hands=<code>({hand.hands_x ?? "—"}, {hand.hands_y ?? "—"})</code>{" · "}
                        clubhead=<code>({hand.clubhead_x ?? "—"}, {hand.clubhead_y ?? "—"})</code>{" · "}
                        shaft=<code>{hand.shaft_direction ?? "—"}</code>
                        {hand.image_width != null && (
                          <> · image <code>{hand.image_width}×{hand.image_height ?? "—"}</code>px</>
                        )}
                      </div>
                    )}
                    {hand.notes && (
                      <div className="small muted" style={{ marginTop: 4, fontStyle: "italic" }}>
                        {hand.notes}
                      </div>
                    )}
                  </div>
                ) : (
                  <div className="small err-text">
                    Handedness failed: <code>{hand.error}</code>
                  </div>
                )}
              </div>
            )}

            {result?.error && (
              <p className="small err-text" style={{ marginTop: 8 }}>
                Request failed: <code>{result.error}</code>
              </p>
            )}

            <div className="row" style={{ marginTop: 12 }}>
              <button
                onClick={() => runOne(c.id)}
                disabled={!!running[c.id] || isComposite}
                title={isComposite ? "Composite clips not supported" : "Ask Claude to find the address frame"}
              >
                <Icon name="play" size={14} />{" "}
                {running[c.id] ? "Running…" : "Find address frame"}
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
