import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

/**
 * Iteration page: every uploaded clip, with a Retry-tracer button on
 * each row so we can re-run the detector after threshold changes
 * without re-uploading footage every time.
 */
export default function AdminClips() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [clips, setClips] = useState(null);
  const [error, setError] = useState(null);
  const [retrying, setRetrying] = useState({});      // {clip_id: true}
  const [sensitivity, setSensitivity] = useState({}); // {clip_id: float}
  const [lightboxUrl, setLightboxUrl] = useState(null);
  const [results, setResults] = useState({});        // {clip_id: {tracer_url, tracer_info, tracer_debug_url}}

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

  async function retryOne(clipId) {
    setRetrying((r) => ({ ...r, [clipId]: true }));
    const s = parseFloat(sensitivity[clipId]);
    const opts = (Number.isFinite(s) && s > 0 && s !== 1.0) ? { sensitivity: s } : {};
    try {
      const data = await api.retryTracer(adminPassword, clipId, opts);
      setResults((r) => ({ ...r, [clipId]: data }));
      // Patch the clip in-place so the row reflects new tracer_url
      setClips((cs) =>
        (cs || []).map((c) => (c.id === clipId ? { ...c, tracer_url: data.tracer_url } : c))
      );
    } catch (e) {
      setResults((r) => ({ ...r, [clipId]: { error: e.message } }));
    } finally {
      setRetrying((r) => ({ ...r, [clipId]: false }));
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
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production">Production</Link>
        <Link to="/admin/clips" className="active">All clips</Link>
        <Link to="/admin/clips/ai">AI tracer</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>All uploaded reelz</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Re-run the tracer on any clip without re-uploading. Useful while we're
          iterating on the detector — tune thresholds in the code, push, then
          hit Retry on a clip you've already uploaded to see the new result.
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
        const tracerUrl = result?.tracer_url ?? c.tracer_url;
        const info = result?.tracer_info;
        const debugUrl = result?.tracer_debug_url;
        const isComposite = (c.source_url || "").includes("_composite");
        // For dual-cam composites the page shows the raw tee cut + runs
        // the tracer on it; the composite source_url is unusable for the
        // classical-CV tracer (overlay baked in, green half concatenated).
        const sourceUrlForPlayer =
          isComposite && c.tee_clip_url ? c.tee_clip_url : c.source_url;
        const canRetry = !isComposite || !!c.tee_clip_url;
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
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Source
                  {isComposite && c.tee_clip_url && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · raw tee cut (composite hidden for tracer iteration)
                    </span>
                  )}
                </div>
                <video
                  src={sourceUrlForPlayer}
                  poster={c.thumbnail_url || undefined}
                  controls
                  style={{ width: "100%", borderRadius: 8, background: "#000" }}
                />
              </div>
              <div>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Tracer
                  {info?.residual_px != null && (
                    <span className="muted" style={{ marginLeft: 6 }}>
                      · {info.residual_px.toFixed(1)}px · {info.n_points} pts
                    </span>
                  )}
                </div>
                {tracerUrl ? (
                  <video
                    src={tracerUrl}
                    controls
                    style={{ width: "100%", borderRadius: 8, background: "#000" }}
                  />
                ) : (
                  <div className="muted center" style={{
                    aspectRatio: "16/9", display: "grid", placeItems: "center",
                    border: "2px dashed var(--border)", borderRadius: 8,
                  }}>
                    No tracer yet
                  </div>
                )}
              </div>
            </div>

            {debugUrl && (
              <div style={{ marginTop: 10 }}>
                <div className="tiny upper muted" style={{ marginBottom: 4 }}>
                  Detector debug image{" "}
                  <span style={{ textTransform: "none" }}>· click to enlarge</span>
                </div>
                <img
                  src={debugUrl}
                  alt="ball candidates debug"
                  onClick={() => setLightboxUrl(debugUrl)}
                  style={{
                    width: "100%", maxWidth: 800, borderRadius: 8,
                    border: "1px solid var(--border)", cursor: "zoom-in",
                  }}
                />
              </div>
            )}

            {result?.error && (
              <p className="small err-text" style={{ marginTop: 8 }}>
                Retry failed: <code>{result.error}</code>
              </p>
            )}

            {info && !info.ok && !result?.error && (
              <p className="small muted" style={{ marginTop: 8 }}>
                Tracer didn't find a clean trajectory
                {info.error ? <> — <code>{info.error}</code></> : null}
                {info.n_candidates != null && <> · {info.n_candidates} raw candidates</>}
                .
              </p>
            )}

            <div
              className="row"
              style={{ marginTop: 12, gap: 8, alignItems: "center", flexWrap: "wrap" }}
            >
              <label
                className="small muted"
                style={{ display: "inline-flex", alignItems: "center", gap: 6 }}
                title="Sensitivity multiplier. 1.0 = current defaults. Higher = more dots / looser filters (BG var, circularity, upward-streak length, upward dy). Lower = stricter."
              >
                sensitivity
                <input
                  type="number"
                  min="0.2"
                  max="5"
                  step="0.1"
                  value={sensitivity[c.id] ?? 1.0}
                  onChange={(e) =>
                    setSensitivity((s) => ({ ...s, [c.id]: e.target.value }))
                  }
                  disabled={!!retrying[c.id]}
                  style={{ width: 64, fontSize: 13, padding: "2px 6px" }}
                />
              </label>
              <button
                onClick={() => retryOne(c.id)}
                disabled={!!retrying[c.id] || !canRetry}
                title={
                  !canRetry
                    ? "Composite clip with no raw tee cut on file — re-process the long upload to populate it"
                    : isComposite
                    ? "Re-run the classical-CV tracer on the stored raw tee cut"
                    : "Re-run the tracer with current thresholds"
                }
              >
                <Icon name="play" size={14} />{" "}
                {retrying[c.id] ? "Running tracer…" : "Retry tracer"}
              </button>
              {isComposite && (
                <span className="small muted" style={{ alignSelf: "center" }}>
                  Composite — {canRetry ? "tracing raw tee cut" : "no raw tee cut available"}
                </span>
              )}
            </div>
          </div>
        );
      })}

      {lightboxUrl && (
        <div
          role="dialog"
          aria-modal="true"
          onClick={() => setLightboxUrl(null)}
          style={{
            position: "fixed",
            inset: 0,
            background: "rgba(0,0,0,0.85)",
            display: "grid",
            placeItems: "center",
            zIndex: 1000,
            cursor: "zoom-out",
            padding: 24,
          }}
        >
          <img
            src={lightboxUrl}
            alt="debug image enlarged"
            onClick={(e) => e.stopPropagation()}
            style={{
              maxWidth: "100%",
              maxHeight: "100%",
              borderRadius: 8,
              cursor: "default",
            }}
          />
        </div>
      )}
    </div>
  );
}
