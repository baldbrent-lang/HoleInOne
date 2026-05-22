/**
 * Produced clips — finished output of the production pipeline.
 *
 * Filters /admin/clips to clips that have actually been produced
 * (tracer URL is rendered OR the source is a dual-cam composite).
 * Raw, mid-pipeline, and un-processed clips stay on the legacy
 * /admin/clips iteration page (still reachable by URL).
 */
import { useCallback, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";
import { useInfiniteList } from "../hooks/useInfiniteList.js";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

function isProduced(c) {
  if (!c) return false;
  if (c.tracer_url) return true;
  if ((c.source_url || "").includes("_composite")) return true;
  return false;
}

export default function AdminProducedClips() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";

  // The "produced" filter runs client-side after fetch — the backend
  // /clips endpoint includes mid-pipeline rows too. That means a given
  // page of 25 raw clips might yield anywhere from 0–25 "produced"
  // ones, so the list density is a bit variable. Good enough; switching
  // to a server-side filter would mean a new endpoint.
  const fetchClips = useCallback(
    async (limit, offset) => {
      const list = await api.listAllClips(adminPassword, limit, offset);
      return (list || []).filter(isProduced);
    },
    [adminPassword],
  );
  const {
    items: clips,
    setItems: setClips,
    sentinelRef,
    hasMore,
    loadingMore,
    error: listError,
  } = useInfiniteList(fetchClips, { pageSize: 25, deps: [adminPassword] });
  const [actionError, setActionError] = useState(null);
  const error = actionError || listError;
  const setError = setActionError;
  const [busyId, setBusyId] = useState(null);

  async function toggleBroadcast(clip) {
    setBusyId(clip.id);
    try {
      const next = !clip.is_highlight;
      await api.setClipBroadcast(adminPassword, clip.id, next);
      setClips((cs) => (cs || []).map(
        (c) => (c.id === clip.id ? { ...c, is_highlight: next } : c)
      ));
    } catch (e) {
      setError(e.message);
    } finally {
      setBusyId(null);
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
        <Link to="/admin/produced-clips" className="active">Produced Clips</Link>
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>Produced clips</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Every clip that's finished the production pipeline — tracer
          rendered or dual-cam composite cut. Newest first. The full
          raw catalog (including in-progress clips) lives on{" "}
          <Link to="/admin/clips">/admin/clips</Link>.
        </p>
      </div>

      {error && <div className="card err-text small">{error}</div>}

      {clips === null && (
        <div className="card"><div className="shimmer" style={{ height: 200 }} /></div>
      )}

      {clips?.length === 0 && (
        <div className="card muted center" style={{ padding: 40 }}>
          No produced clips yet. Once a round finishes processing on{" "}
          <Link to="/admin/production">Production</Link>, finished clips show up here.
        </div>
      )}

      <div
        className="grid"
        style={{
          gridTemplateColumns: "repeat(auto-fill, minmax(280px, 1fr))",
          gap: 12,
        }}
      >
        {clips?.map((c) => {
          const playUrl = c.tracer_url || c.source_url;
          const isAce = !!c.ball_in_cup;
          return (
            <div key={c.id} className="card" style={{ margin: 0 }}>
              <div
                className="row"
                style={{
                  gap: 6, alignItems: "baseline", flexWrap: "wrap",
                  marginBottom: 6,
                }}
              >
                <b>#{c.id}</b>
                <span className="small muted">
                  {c.course_name || `course ${c.course_id}`} · hole {c.hole_number}
                </span>
                <div style={{ flex: 1 }} />
                {isAce && <span className="pill ok small">ace</span>}
                {c.participant_name ? (
                  <span className="pill ok small">{c.participant_name}</span>
                ) : (
                  <span className="pill warn small">unassigned</span>
                )}
              </div>
              <video
                src={playUrl}
                poster={c.thumbnail_url || undefined}
                controls
                style={{ width: "100%", borderRadius: 6, background: "#000" }}
              />
              <div className="tiny muted" style={{ marginTop: 6 }}>
                {c.captured_at ? new Date(c.captured_at).toLocaleString() : "—"}
                {c.fps != null && <> · <code>{c.fps}</code> fps</>}
              </div>
              <button
                type="button"
                className={c.is_highlight ? "" : "ghost"}
                style={{ width: "100%", marginTop: 8 }}
                onClick={() => toggleBroadcast(c)}
                disabled={busyId === c.id}
                title={c.is_highlight
                  ? "Remove from the Broadcast channel"
                  : "Send this clip to the Broadcast channel"}
              >
                {busyId === c.id
                  ? "…"
                  : c.is_highlight ? "On Broadcast" : "Broadcast"}
              </button>
            </div>
          );
        })}
      </div>

      {clips && clips.length > 0 && (
        <div
          ref={sentinelRef}
          className="muted center small"
          style={{ padding: 20 }}
        >
          {loadingMore
            ? "Loading more…"
            : hasMore
              ? "Scroll for more"
              : "End of produced clips"}
        </div>
      )}
    </div>
  );
}
