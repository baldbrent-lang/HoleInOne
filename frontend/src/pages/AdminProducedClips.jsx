/**
 * Produced clips — finished output of the production pipeline.
 *
 * Filters /admin/clips to clips that have actually been produced
 * (tracer URL is rendered OR the source is a dual-cam composite).
 * Raw, mid-pipeline, and un-processed clips stay on the legacy
 * /admin/clips iteration page (still reachable by URL).
 */
import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";

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

  const [clips, setClips] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!adminPassword) return;
    api
      .listAllClips(adminPassword)
      .then((list) => setClips(list.filter(isProduced)))
      .catch((e) => setError(e.message));
  }, [adminPassword]);

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
            </div>
          );
        })}
      </div>
    </div>
  );
}
