import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import DraggableMarker from "../components/DraggableMarker.jsx";

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
  const [selectedModels, setSelectedModels] = useState({}); // {clipId: modelId}
  const [audioImpact, setAudioImpact] = useState({});        // {clipId: {image_url, frame, ratio, error?}}
  const [audioImpactRunning, setAudioImpactRunning] = useState({});
  const [audioImpactMinRatio, setAudioImpactMinRatio] = useState({}); // {clipId: number}
  const [overrideDrafts, setOverrideDrafts] = useState({});   // {clipId: {impactFrame, maxFrames, ballAtRestX, ballAtRestY, manualJson, handedness}}
  const [overridesOpen, setOverridesOpen] = useState({});      // {clipId: bool}
  // Per-frame manual ball positions populated by dragging the
  // ball-track thumbnail markers. Shape: {clipId: {frameNumber: {x, y}}}
  // — collapsed into the manualBallPositions array on submit.
  const [manualPerFrame, setManualPerFrame] = useState({});

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

  async function runOne(clipId, overrides = {}) {
    setRunning((r) => ({ ...r, [clipId]: true }));
    const model = selectedModels[clipId] || "claude-opus-4-7";
    try {
      const data = await api.aiTrace(adminPassword, clipId, {
        model,
        ...overrides,
      });
      setResults((r) => ({ ...r, [clipId]: data }));
    } catch (e) {
      setResults((r) => ({ ...r, [clipId]: { error: e.message } }));
    } finally {
      setRunning((r) => ({ ...r, [clipId]: false }));
    }
  }

  async function runWithOverrides(clipId) {
    const draft = overrideDrafts[clipId] || {};
    const overrides = {};
    if (draft.impactFrame !== undefined && draft.impactFrame !== "") {
      const n = parseInt(draft.impactFrame, 10);
      if (Number.isFinite(n) && n >= 0) overrides.impactFrameOverride = n;
    }
    if (draft.maxFrames !== undefined && draft.maxFrames !== "") {
      const n = parseInt(draft.maxFrames, 10);
      if (Number.isFinite(n) && n > 0) overrides.ballTrackMaxFrames = n;
    }
    if (draft.ballAtRestX !== undefined && draft.ballAtRestX !== "") {
      const n = parseInt(draft.ballAtRestX, 10);
      if (Number.isFinite(n)) overrides.ballAtRestX = n;
    }
    if (draft.ballAtRestY !== undefined && draft.ballAtRestY !== "") {
      const n = parseInt(draft.ballAtRestY, 10);
      if (Number.isFinite(n)) overrides.ballAtRestY = n;
    }
    if (draft.handedness && draft.handedness !== "auto") {
      overrides.handednessOverride = draft.handedness;
    }
    // Combine per-frame drag overrides with any pasted JSON. Drag
    // values win on duplicate frame numbers.
    const merged = {};
    if (draft.manualJson !== undefined && draft.manualJson.trim() !== "") {
      try {
        const parsed = JSON.parse(draft.manualJson);
        if (Array.isArray(parsed)) {
          for (const entry of parsed) {
            if (entry && entry.frame != null) {
              merged[entry.frame] = { x: entry.x, y: entry.y };
            }
          }
        }
      } catch (e) {
        setResults((r) => ({
          ...r,
          [clipId]: { error: `manual_ball_positions invalid JSON: ${e.message}` },
        }));
        return;
      }
    }
    const perFrame = manualPerFrame[clipId] || {};
    for (const [frame, pos] of Object.entries(perFrame)) {
      merged[frame] = { x: pos.x, y: pos.y };
    }
    const positions = Object.entries(merged).map(([frame, pos]) => ({
      frame: parseInt(frame, 10),
      x: pos.x,
      y: pos.y,
    }));
    if (positions.length > 0) overrides.manualBallPositions = positions;
    await runOne(clipId, overrides);
  }

  function setOverrideField(clipId, field, value) {
    setOverrideDrafts((d) => ({
      ...d,
      [clipId]: { ...(d[clipId] || {}), [field]: value },
    }));
  }

  async function runAudioImpactFrame(clipId) {
    const ratio = parseFloat(audioImpactMinRatio[clipId] ?? 25);
    const minRatio = Number.isFinite(ratio) && ratio > 0 ? ratio : 25;
    setAudioImpactRunning((r) => ({ ...r, [clipId]: true }));
    try {
      const data = await api.audioImpactFrame(adminPassword, clipId, { minRatio });
      setAudioImpact((r) => ({ ...r, [clipId]: data }));
    } catch (e) {
      setAudioImpact((r) => ({ ...r, [clipId]: { ok: false, error: e.message } }));
    } finally {
      setAudioImpactRunning((r) => ({ ...r, [clipId]: false }));
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
        <Link to="/admin/participants">Players</Link>
        <Link to="/admin/courses">Courses</Link>
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production">Production</Link>
        <Link to="/admin/produced-clips">Produced Clips</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>AI tracer — address → handedness → impact → ball flight → tracer video</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Camera is assumed to be <b>behind the target line</b>, golfer hitting
          away from camera. Each Run executes six steps:
          (1) find the <b>address frame</b>; (2) on that frame locate hands +
          ball and infer <b>handedness</b>; (3) rough <b>impact</b> — first
          tries detecting the loudest audio transient (the "thwack"), falls
          back to AI vision picking from 12 candidates if audio is unavailable
          or unclear; (4) <b>refine</b> the impact frame ±5 and locate the
          shaft on it; (5) <b>track the ball</b> forward frame-by-frame from
          impact until it leaves view, with a phase-2 crop+upscale retry on
          missed frames; (6) render the <b>tracer video</b> — source clip
          with a smoothed dashed orange line from the ball-at-rest position
          through each tracked ball position, fading off along its natural
          arc past the last detection.
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
          No clips uploaded yet. <Link to="/admin/upload-videos">Upload one →</Link>
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
        const tracerVideo = result?.tracer_video;
        const tracerVideoUrl = result?.tracer_video_url;
        const isComposite = (c.source_url || "").includes("_composite");
        // A composite clip is still analyzable if the backend stored the raw
        // tee-only cut alongside it. Only disable the button when no tee_clip_url
        // is available to fall back to.
        const canAnalyze = !isComposite || !!c.tee_clip_url;
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
                  (() => {
                    // Drag the green ball-at-rest circle to correct
                    // it. Initial position: AI's native-coord pick if
                    // available, else the operator's prior override
                    // draft, else null (image rendered without
                    // marker; click anywhere to drop one).
                    const draft = overrideDrafts[c.id] || {};
                    let bx = null, by = null;
                    if (draft.ballAtRestX !== undefined && draft.ballAtRestX !== "") {
                      bx = parseInt(draft.ballAtRestX, 10);
                    } else if (result?.ball_rest_xy_native) {
                      bx = Math.round(result.ball_rest_xy_native[0]);
                    }
                    if (draft.ballAtRestY !== undefined && draft.ballAtRestY !== "") {
                      by = parseInt(draft.ballAtRestY, 10);
                    } else if (result?.ball_rest_xy_native) {
                      by = Math.round(result.ball_rest_xy_native[1]);
                    }
                    return (
                      <DraggableMarker
                        imageUrl={addrImage}
                        alt="Claude's pick for address frame — drag the green circle to correct ball-at-rest"
                        x={bx}
                        y={by}
                        color="#22dd66"
                        label="ball"
                        size={26}
                        imageStyle={{ borderRadius: 8 }}
                        onChange={(nx, ny) => {
                          setOverrideField(c.id, "ballAtRestX", String(nx));
                          setOverrideField(c.id, "ballAtRestY", String(ny));
                        }}
                        onClick={(nx, ny) => {
                          setOverrideField(c.id, "ballAtRestX", String(nx));
                          setOverrideField(c.id, "ballAtRestY", String(ny));
                        }}
                      />
                    );
                  })()
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

            {(tracerVideoUrl || tracerVideo?.error) && (
              <div style={{ marginTop: 12 }}>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  AI tracer video
                  {tracerVideo?.n_points != null && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · <code>{tracerVideo.n_points}</code> smoothed pts
                      {tracerVideo.frame_range && (
                        <> · frames {tracerVideo.frame_range[0]}–{tracerVideo.frame_range[1]}</>
                      )}
                      {tracerVideo.n_outliers_rejected > 0 && (
                        <> · {tracerVideo.n_outliers_rejected} outlier{tracerVideo.n_outliers_rejected === 1 ? "" : "s"} rejected</>
                      )}
                    </span>
                  )}
                </div>
                {tracerVideoUrl ? (
                  <video
                    src={tracerVideoUrl}
                    controls
                    style={{ width: "100%", borderRadius: 8, background: "#000" }}
                  />
                ) : tracerVideo?.error ? (
                  <div className="err-text small">
                    Tracer render failed: <code>{tracerVideo.error}</code>
                  </div>
                ) : null}
              </div>
            )}

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
                {impact?.method === "audio" && (
                  <div className="tiny muted" style={{ marginTop: 2 }}>
                    Rough impact via <b>audio peak</b>
                    {impact?.audio?.peak_time_sec != null && (
                      <> @ <code>{impact.audio.peak_time_sec.toFixed(3)}s</code></>
                    )}
                    {impact?.audio?.ratio != null && (
                      <> · peak/median = <code>{impact.audio.ratio.toFixed(1)}</code></>
                    )}
                    {impact?.confidence && (
                      <> · confidence <b>{impact.confidence}</b></>
                    )}
                  </div>
                )}
                {impact?.method === "ai_vision" && impact?.frames_sent && (
                  <div className="tiny muted" style={{ marginTop: 2 }}>
                    Rough impact via <b>AI vision</b> · candidates [{impact.frames_sent.join(", ")}] · model {impact.model}
                    {impact?.audio?.error && (
                      <> · audio fallback reason: <code>{impact.audio.error}</code></>
                    )}
                  </div>
                )}
                {impact?.method !== "audio" && impact?.method !== "ai_vision" && impact?.frames_sent && (
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
                      {ballTrack.n_frames_clahe_enhanced > 0 && (
                        <> · <code>{ballTrack.n_frames_clahe_enhanced}</code> frame{ballTrack.n_frames_clahe_enhanced === 1 ? "" : "s"} CLAHE-boosted</>
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
                  <>
                    <div className="tiny muted" style={{ marginBottom: 6 }}>
                      Drag the green circle on any thumbnail to correct the AI's
                      ball position for that frame. Click on a red ("no ball") frame
                      to drop a marker where the ball actually is. Changes apply on
                      the next "Re-run with overrides" click.
                    </div>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "repeat(auto-fill, minmax(180px, 1fr))",
                        gap: 6,
                      }}
                    >
                      {ballTrackFrames.map((rec) => {
                        const perFrameDraft = (manualPerFrame[c.id] || {})[rec.frame];
                        const hasOverride = !!perFrameDraft;
                        const mx = hasOverride
                          ? perFrameDraft.x
                          : (rec.found ? rec.x : null);
                        const my = hasOverride
                          ? perFrameDraft.y
                          : (rec.found ? rec.y : null);
                        const showMarker = mx != null && my != null;
                        const tooltip = hasOverride
                          ? `Frame ${rec.frame} · MANUAL override at (${mx}, ${my})`
                          : (rec.found
                            ? `Frame ${rec.frame} · AI: (${rec.x}, ${rec.y}) · ${rec.confidence || "—"}${rec.retry ? " · via retry" : ""}`
                            : `Frame ${rec.frame} · ball NOT FOUND — click to drop marker`);
                        const badgeBg = hasOverride
                          ? "rgba(40,180,220,0.85)"
                          : (rec.found
                            ? (rec.retry ? "rgba(255,170,0,0.85)" : "rgba(40,150,80,0.85)")
                            : "rgba(180,40,40,0.85)");
                        const badgeText = hasOverride
                          ? `f${rec.frame} · manual`
                          : (rec.found
                            ? (rec.retry ? `f${rec.frame} · retry` : `f${rec.frame}${rec.confidence ? ` · ${rec.confidence}` : ""}`)
                            : `f${rec.frame} · no ball`);
                        const markerColor = hasOverride
                          ? "#28b4dc"
                          : (rec.found ? "#22dd66" : "#ff6b6b");
                        return (
                          <div key={rec.frame} style={{ position: "relative" }} title={tooltip}>
                            {rec.image_url ? (
                              <DraggableMarker
                                imageUrl={rec.image_url}
                                alt={`Frame ${rec.frame}`}
                                x={showMarker ? mx : null}
                                y={showMarker ? my : null}
                                color={markerColor}
                                size={18}
                                imageStyle={{
                                  borderRadius: 4,
                                  opacity: rec.found || hasOverride ? 1 : 0.65,
                                }}
                                onChange={(nx, ny) => {
                                  setManualPerFrame((prev) => ({
                                    ...prev,
                                    [c.id]: {
                                      ...(prev[c.id] || {}),
                                      [rec.frame]: { x: nx, y: ny },
                                    },
                                  }));
                                }}
                                onClick={(nx, ny) => {
                                  // Only treat clicks as drop-marker
                                  // for frames the AI missed AND don't
                                  // already have a manual override.
                                  if (showMarker) return;
                                  setManualPerFrame((prev) => ({
                                    ...prev,
                                    [c.id]: {
                                      ...(prev[c.id] || {}),
                                      [rec.frame]: { x: nx, y: ny },
                                    },
                                  }));
                                }}
                              />
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
                                pointerEvents: "none",
                              }}
                            >
                              {badgeText}
                            </div>
                            {hasOverride && (
                              <button
                                type="button"
                                onClick={() => {
                                  setManualPerFrame((prev) => {
                                    const next = { ...(prev[c.id] || {}) };
                                    delete next[rec.frame];
                                    return { ...prev, [c.id]: next };
                                  });
                                }}
                                style={{
                                  position: "absolute",
                                  right: 4,
                                  top: 4,
                                  background: "rgba(0,0,0,0.6)",
                                  color: "#fff",
                                  border: "none",
                                  borderRadius: 3,
                                  padding: "1px 5px",
                                  fontSize: 10,
                                  cursor: "pointer",
                                }}
                                title="Discard manual override for this frame"
                              >
                                ↶
                              </button>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  </>
                )}
              </div>
            )}

            {result?.error && (
              <p className="small err-text" style={{ marginTop: 8 }}>
                Request failed: <code>{result.error}</code>
              </p>
            )}

            <div className="row" style={{ marginTop: 12, gap: 8 }}>
              <button
                onClick={() => runOne(c.id)}
                disabled={!!running[c.id] || !canAnalyze}
                title={
                  !canAnalyze
                    ? "Composite clip with no raw tee cut on file — re-process the long upload to populate it"
                    : isComposite
                    ? "Composite — will analyze the stored raw tee cut"
                    : "Run address + handedness + impact pipeline"
                }
              >
                <Icon name="play" size={14} />{" "}
                {running[c.id] ? "Running…" : "Run AI analysis"}
              </button>
              <label
                className="small muted"
                style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
              >
                model:
                <select
                  value={selectedModels[c.id] || "claude-opus-4-7"}
                  onChange={(e) =>
                    setSelectedModels((m) => ({ ...m, [c.id]: e.target.value }))
                  }
                  disabled={!!running[c.id]}
                  style={{ fontSize: 13 }}
                >
                  <option value="claude-opus-4-7">Opus 4.7 (default)</option>
                  <option value="claude-haiku-4-5">Haiku 4.5 (faster, cheaper)</option>
                </select>
              </label>
              {isComposite && (
                <span className="small muted" style={{ alignSelf: "center" }}>
                  Composite — {canAnalyze ? "analyzing raw tee cut" : "no raw tee cut available"}
                </span>
              )}
            </div>

            {/* Audio-only impact-frame test. Skips the AI impact-pick +
                refine-impact calls; just runs find_impact_via_audio at
                a configurable min_ratio and returns the matching frame
                as a JPG. Useful for verifying we can drop those two AI
                calls from the production pipeline. */}
            <div
              className="row"
              style={{ marginTop: 8, gap: 8, alignItems: "center", flexWrap: "wrap" }}
            >
              <button
                onClick={() => runAudioImpactFrame(c.id)}
                disabled={!!audioImpactRunning[c.id] || !canAnalyze}
                title="Detect impact frame via audio, then derive the address frame (impact - 1.5s) and grab both. No AI calls."
                className="secondary"
              >
                {audioImpactRunning[c.id] ? "Detecting…" : "Pick impact + address (audio)"}
              </button>
              <label
                className="small muted"
                style={{ display: "inline-flex", alignItems: "center", gap: 4 }}
                title="Min peak/median envelope ratio required. Higher = stricter; below this the detector returns no impact."
              >
                min ×ratio:
                <input
                  type="number"
                  min="5"
                  step="1"
                  value={audioImpactMinRatio[c.id] ?? 25}
                  onChange={(e) =>
                    setAudioImpactMinRatio((m) => ({ ...m, [c.id]: e.target.value }))
                  }
                  disabled={!!audioImpactRunning[c.id]}
                  style={{ width: 56, fontSize: 13, padding: "2px 6px" }}
                />
              </label>
              {audioImpact[c.id] && audioImpact[c.id].ok === false && (
                <span className="small err-text">
                  ✗ {audioImpact[c.id].error}
                  {audioImpact[c.id].ratio != null && <> (ratio ×{audioImpact[c.id].ratio.toFixed?.(1) ?? audioImpact[c.id].ratio})</>}
                </span>
              )}
            </div>
            {audioImpact[c.id]?.ok && (
              <div
                style={{
                  marginTop: 8,
                  display: "grid",
                  gridTemplateColumns: audioImpact[c.id].address_image_url
                    ? "1fr 1fr" : "1fr",
                  gap: 12,
                }}
              >
                {audioImpact[c.id].address_image_url && (
                  <div>
                    <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                      Address frame{" "}
                      <span style={{ textTransform: "none" }}>
                        · frame {audioImpact[c.id].address_frame}
                        {audioImpact[c.id].address_offset_sec != null && (
                          <> (impact − {audioImpact[c.id].address_offset_sec}s)</>
                        )}
                      </span>
                    </div>
                    <img
                      src={audioImpact[c.id].address_image_url}
                      alt="audio-derived address frame"
                      style={{
                        width: "100%", borderRadius: 8,
                        border: "1px solid var(--border)",
                      }}
                    />
                  </div>
                )}
                <div>
                  <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                    Impact frame{" "}
                    <span style={{ textTransform: "none" }}>
                      · frame {audioImpact[c.id].impact_frame}
                      {audioImpact[c.id].audio_peak_frame != null && audioImpact[c.id].pre_peak_offset_frames > 0 && (
                        <> (audio peak {audioImpact[c.id].audio_peak_frame} − {audioImpact[c.id].pre_peak_offset_frames}f)</>
                      )}
                      {audioImpact[c.id].peak_time_sec != null && (
                        <> · {audioImpact[c.id].peak_time_sec.toFixed(3)}s</>
                      )}
                      {audioImpact[c.id].ratio != null && (
                        <> · ×{audioImpact[c.id].ratio.toFixed(1)}</>
                      )}
                    </span>
                  </div>
                  <img
                    src={audioImpact[c.id].image_url}
                    alt="audio-derived impact frame"
                    style={{
                      width: "100%", borderRadius: 8,
                      border: "1px solid var(--border)",
                    }}
                  />
                </div>
              </div>
            )}

            {/* Manual overrides — visible after AI has run at least
                once. Lets the operator hand-correct anything the AI
                got wrong: impact frame, how many frames to keep
                tracking after impact, ball-at-rest position, and
                missed-in-flight ball positions. Each override
                bypasses the matching AI step entirely. */}
            {result && !result.error && (
              <div style={{ marginTop: 10 }}>
                <button
                  type="button"
                  className="ghost small"
                  onClick={() =>
                    setOverridesOpen((o) => ({ ...o, [c.id]: !o[c.id] }))
                  }
                  style={{ width: "auto" }}
                >
                  {overridesOpen[c.id] ? "▼" : "▶"} Manual overrides
                </button>
                {overridesOpen[c.id] && (() => {
                  const draft = overrideDrafts[c.id] || {};
                  // Pre-fill defaults from AI's output so the operator
                  // only has to edit what they want to change.
                  const defaultImpact =
                    impactRefined?.impact_frame ??
                    impact?.impact_frame ?? "";
                  // ball_x/y from handedness are in SENT image coords.
                  // Convert to native coords for the override input,
                  // since the override is interpreted as native pixels.
                  let defaultRestX = "";
                  let defaultRestY = "";
                  const hb = result?.handedness;
                  if (
                    hb?.ball_x != null && hb?.ball_y != null
                    && hb?.image_width && hb?.image_height
                    && result?.ball_sent_dims
                  ) {
                    // hb is already in sent coords; just expose them
                    // as-is — backend treats override as native, so we
                    // need to scale to native. We don't have native
                    // dims directly on the result, but ball_xy_sent /
                    // ball_sent_dims is the same info. Simpler: use
                    // ball_rest_xy_native when set.
                    if (result.ball_rest_xy_native) {
                      defaultRestX = Math.round(result.ball_rest_xy_native[0]);
                      defaultRestY = Math.round(result.ball_rest_xy_native[1]);
                    }
                  }
                  return (
                    <div
                      className="card tight"
                      style={{ margin: "8px 0 0", padding: 10, background: "var(--surface-alt)" }}
                    >
                      <div className="tiny muted" style={{ marginBottom: 8 }}>
                        Leave any field blank to keep the AI's value.
                        Coords are in native source-video pixels (origin top-left).
                      </div>
                      <div className="row" style={{ gap: 10, flexWrap: "wrap", alignItems: "flex-end" }}>
                        <label className="small" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                          Impact frame
                          <input
                            type="number" min="0"
                            placeholder={String(defaultImpact)}
                            value={draft.impactFrame ?? ""}
                            onChange={(e) => setOverrideField(c.id, "impactFrame", e.target.value)}
                            disabled={!!running[c.id]}
                            style={{ width: 90, fontSize: 13, padding: "2px 6px" }}
                          />
                        </label>
                        <label className="small" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                          Handedness
                          <select
                            value={draft.handedness ?? "auto"}
                            onChange={(e) => setOverrideField(c.id, "handedness", e.target.value)}
                            disabled={!!running[c.id]}
                            style={{ fontSize: 13, padding: "2px 6px" }}
                          >
                            <option value="auto">auto (use AI)</option>
                            <option value="right">right-handed</option>
                            <option value="left">left-handed</option>
                            <option value="unknown">unknown</option>
                          </select>
                        </label>
                        <label className="small" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                          Frames to track (post-impact)
                          <input
                            type="number" min="1" max="120"
                            placeholder="12"
                            value={draft.maxFrames ?? ""}
                            onChange={(e) => setOverrideField(c.id, "maxFrames", e.target.value)}
                            disabled={!!running[c.id]}
                            style={{ width: 90, fontSize: 13, padding: "2px 6px" }}
                          />
                        </label>
                        <label className="small" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                          Ball-at-rest X (px)
                          <input
                            type="number"
                            placeholder={String(defaultRestX)}
                            value={draft.ballAtRestX ?? ""}
                            onChange={(e) => setOverrideField(c.id, "ballAtRestX", e.target.value)}
                            disabled={!!running[c.id]}
                            style={{ width: 100, fontSize: 13, padding: "2px 6px" }}
                          />
                        </label>
                        <label className="small" style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                          Ball-at-rest Y (px)
                          <input
                            type="number"
                            placeholder={String(defaultRestY)}
                            value={draft.ballAtRestY ?? ""}
                            onChange={(e) => setOverrideField(c.id, "ballAtRestY", e.target.value)}
                            disabled={!!running[c.id]}
                            style={{ width: 100, fontSize: 13, padding: "2px 6px" }}
                          />
                        </label>
                      </div>
                      <div style={{ marginTop: 10 }}>
                        <label className="small" style={{ display: "block", marginBottom: 4 }}>
                          Manual ball positions for missed frames (JSON array):
                        </label>
                        <textarea
                          rows={4}
                          placeholder='[{"frame": 142, "x": 800, "y": 200}, {"frame": 145, "x": 850, "y": 180}]'
                          value={draft.manualJson ?? ""}
                          onChange={(e) => setOverrideField(c.id, "manualJson", e.target.value)}
                          disabled={!!running[c.id]}
                          style={{
                            width: "100%", fontFamily: "monospace",
                            fontSize: 12, padding: 8,
                            border: "1px solid var(--border)", borderRadius: 6,
                          }}
                        />
                        <div className="tiny muted" style={{ marginTop: 4 }}>
                          One row per frame the AI missed (or got wrong).
                          x/y are pixel coords in the source-video frame.
                          Overrides existing AI ball positions on matching
                          frame numbers; inserts new ones otherwise.
                        </div>
                      </div>
                      <div className="row" style={{ marginTop: 10, gap: 8, alignItems: "center", flexWrap: "wrap" }}>
                        <button
                          type="button"
                          onClick={() => runWithOverrides(c.id)}
                          disabled={!!running[c.id] || !canAnalyze}
                        >
                          {running[c.id] ? "Re-running…" : "Re-run with overrides"}
                        </button>
                        <button
                          type="button"
                          className="ghost small"
                          onClick={() => {
                            setOverrideDrafts((d) => ({ ...d, [c.id]: {} }));
                            setManualPerFrame((p) => ({ ...p, [c.id]: {} }));
                          }}
                          disabled={!!running[c.id]}
                        >
                          Clear all
                        </button>
                        {(() => {
                          const n = Object.keys(manualPerFrame[c.id] || {}).length;
                          if (n === 0) return null;
                          return (
                            <span className="small muted">
                              {n} per-frame drag override{n === 1 ? "" : "s"} queued
                            </span>
                          );
                        })()}
                      </div>
                    </div>
                  );
                })()}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
