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
        <h3 style={{ marginBottom: 4 }}>AI tracer — address → handedness → impact → ball flight</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Camera is assumed to be <b>behind the target line</b>, golfer hitting
          away from camera. Each Run executes five steps via Claude:
          (1) find the <b>address frame</b>; (2) on that frame locate hands +
          ball and infer <b>handedness</b>; (3) rough <b>impact</b> across 12
          candidates over ~2 s after address; (4) <b>refine</b> the impact
          frame ±5 and locate the shaft on it; (5) <b>track the ball</b>
          forward frame-by-frame from impact until it leaves view, with
          a yellow highlight ring drawn on each native-resolution output.
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
        const impact = result?.impact;
        const impactRefined = result?.impact_refined;
        const impactImage = result?.impact_image_url;
        const ballTrack = result?.ball_track;
        const ballTrackFrames = result?.ball_track_frames || [];
        const isComposite = (c.source_url || "").includes("_composite");
        return (
          <div key={c.id} className="card" style={{ marginBottom: 12 }}>
            <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 8 }}>
              <div>
                <b>Clip #{c.id}</b>{" "}
                <span className="muted small">
                  · {c.course_name || `course #${c.course_id}`} · hole {c.hole_number}{" "}
                  · {c.captured_at ? new Date(c.captured_at).toLocaleString() : "—"}
                  {c.fps != null && (
                    <> · <code>{c.fps}</code> fps</>
                  )}
                  {c.source_device && (
                    <> · {c.source_device}</>
                  )}
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
                    {(hand.hands_x != null || hand.ball_x != null) && (
                      <div className="tiny muted" style={{ marginTop: 4 }}>
                        hands=<code>({hand.hands_x ?? "—"}, {hand.hands_y ?? "—"})</code>{" · "}
                        ball start=<code>({hand.ball_x ?? "—"}, {hand.ball_y ?? "—"})</code>{" · "}
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

            {(impact || impactImage) && (
              <div style={{ marginTop: 12 }}>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Impact frame
                  {(impactRefined?.impact_frame ?? impact?.impact_frame) != null && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · frame <code>{impactRefined?.impact_frame ?? impact?.impact_frame}</code>
                      {impact?.impact_frame != null
                        && impactRefined?.impact_frame != null
                        && impactRefined.impact_frame !== impact.impact_frame && (
                        <> (rough <code>{impact.impact_frame}</code>)</>
                      )}
                    </span>
                  )}
                </div>
                {impactImage ? (
                  <img
                    src={impactImage}
                    alt="Claude's pick for the impact frame, with ball circle and shaft overlay"
                    style={{ width: "100%", maxWidth: 800, borderRadius: 8, background: "#000", display: "block" }}
                  />
                ) : impact?.error ? (
                  <div className="err-text small">
                    Impact failed: <code>{impact.error}</code>
                  </div>
                ) : null}
                {impactRefined?.ok ? (
                  <div className="small muted" style={{ marginTop: 6 }}>
                    Refined confidence: <b>{impactRefined.confidence || "—"}</b>
                    {impactRefined.notes && (
                      <>{" "}— <i>{impactRefined.notes}</i></>
                    )}
                  </div>
                ) : impactRefined?.error ? (
                  <div className="small err-text" style={{ marginTop: 6 }}>
                    Refinement failed: <code>{impactRefined.error}</code>
                  </div>
                ) : impact?.ok ? (
                  <div className="small muted" style={{ marginTop: 6 }}>
                    Confidence: <b>{impact.confidence || "—"}</b>
                    {impact.notes && (
                      <>{" "}— <i>{impact.notes}</i></>
                    )}
                  </div>
                ) : null}
                {(impactRefined?.hands_x != null || impactRefined?.clubhead_x != null) && (
                  <div className="tiny muted" style={{ marginTop: 2 }}>
                    hands=<code>({impactRefined.hands_x ?? "—"}, {impactRefined.hands_y ?? "—"})</code>{" · "}
                    clubhead=<code>({impactRefined.clubhead_x ?? "—"}, {impactRefined.clubhead_y ?? "—"})</code>
                    {impactRefined.image_width != null && (
                      <> · image <code>{impactRefined.image_width}×{impactRefined.image_height ?? "—"}</code>px</>
                    )}
                  </div>
                )}
                {impactRefined?.frames_sent && (
                  <div className="tiny muted" style={{ marginTop: 2 }}>
                    Refinement candidates: [{impactRefined.frames_sent.join(", ")}]
                  </div>
                )}
                {impact?.frames_sent && (
                  <div className="tiny muted" style={{ marginTop: 2 }}>
                    Rough candidates: [{impact.frames_sent.join(", ")}] · model {impact.model}
                  </div>
                )}
              </div>
            )}

            {(ballTrack || ballTrackFrames.length > 0) && (
              <div style={{ marginTop: 14 }}>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Ball flight
                  {ballTrack?.n_frames_found != null && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · ball found in <code>{ballTrack.n_frames_found}</code>
                      {" of "}<code>{ballTrack.n_frames_processed}</code> frames
                      {ballTrack.n_frames_found_via_retry > 0 && (
                        <> (incl. <code>{ballTrack.n_frames_found_via_retry}</code> via hint retry)</>
                      )}
                      {ballTrack.first_lost_run_start != null && (
                        <> · lost from frame <code>{ballTrack.first_lost_run_start}</code></>
                      )}
                    </span>
                  )}
                </div>
                {ballTrack?.error && (
                  <div className="small err-text" style={{ marginBottom: 6 }}>
                    Tracking error: <code>{ballTrack.error}</code>
                  </div>
                )}
                {ballTrackFrames.length > 0 && (
                  <div
                    style={{
                      display: "grid",
                      gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                      gap: 6,
                    }}
                  >
                    {ballTrackFrames.map((rec) => {
                      const tooltip = rec.found
                        ? `Frame ${rec.frame} · ball at (${rec.x}, ${rec.y}) · ${rec.confidence || "—"}${rec.retry ? " · via retry" : ""}${rec.notes ? `\n${rec.notes}` : ""}`
                        : `Frame ${rec.frame} · ball NOT FOUND${rec.notes ? `\n${rec.notes}` : ""}`;
                      const badgeBg = rec.found
                        ? (rec.retry ? "rgba(255,170,0,0.85)" : "rgba(40,150,80,0.85)")
                        : "rgba(180,40,40,0.85)";
                      const badgeText = rec.found
                        ? (rec.retry ? `f${rec.frame} · retry` : `f${rec.frame}${rec.confidence ? ` · ${rec.confidence}` : ""}`)
                        : `f${rec.frame} · no ball`;
                      return (
                        <div key={rec.frame} style={{ position: "relative" }}>
                          {rec.image_url ? (
                            <a
                              href={rec.image_url}
                              target="_blank"
                              rel="noreferrer"
                              title={tooltip}
                            >
                              <img
                                src={rec.image_url}
                                alt={`Frame ${rec.frame}`}
                                style={{
                                  width: "100%",
                                  display: "block",
                                  borderRadius: 4,
                                  background: "#000",
                                  opacity: rec.found ? 1 : 0.65,
                                }}
                              />
                            </a>
                          ) : (
                            <div
                              style={{
                                aspectRatio: "16/9",
                                display: "grid",
                                placeItems: "center",
                                border: "1px dashed var(--border)",
                                borderRadius: 4,
                                fontSize: 11,
                                padding: 4,
                                textAlign: "center",
                              }}
                              className="muted"
                              title={tooltip}
                            >
                              f{rec.frame}<br />no image
                            </div>
                          )}
                          <div
                            style={{
                              position: "absolute",
                              left: 4,
                              top: 4,
                              background: badgeBg,
                              color: "#fff",
                              padding: "1px 5px",
                              borderRadius: 3,
                              fontSize: 10,
                              fontWeight: 600,
                              letterSpacing: 0.2,
                            }}
                          >
                            {badgeText}
                          </div>
                        </div>
                      );
                    })}
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
                title={isComposite ? "Composite clips not supported" : "Run address + handedness + impact pipeline"}
              >
                <Icon name="play" size={14} />{" "}
                {running[c.id] ? "Running…" : "Run AI analysis"}
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
