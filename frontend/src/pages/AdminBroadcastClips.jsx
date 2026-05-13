import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

/**
 * Broadcast-style review page: every dual-camera composite clip produced
 * by /admin/long-upload (tee with AI tracer overlay spliced into the
 * green-side landing) plays back here in a single scroll. Newest first.
 */
export default function AdminBroadcastClips() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [clips, setClips] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    try {
      const list = await api.listBroadcastClips(adminPassword);
      setClips(list);
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (adminPassword) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!adminPassword) {
    return (
      <div className="wrap">
        <Brand subtitle="Operator Console" />
        <div className="card center">
          <h2>Admin password required</h2>
          <Link to="/admin">
            <button style={{ marginTop: 10 }}>Sign in</button>
          </Link>
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
        <Link to="/admin/clips/ai">AI tracer</Link>
        <Link to="/admin/broadcast-clips" className="active">Broadcast</Link>
        <Link to="/admin/showcase">Home videos</Link>
        <Link to="/admin/review">Hole-in-one review</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>Broadcast clips</h3>
        <p className="small muted" style={{ marginBottom: 0 }}>
          Every dual-camera composite produced by Long upload — tee-side with
          the AI tracer overlay, then a hard cut to the green-side feed for
          the landing. Newest first. Single-camera clips and raw tee cuts
          appear in <Link to="/admin/clips">All clips</Link> instead.
        </p>
      </div>

      {error && <div className="card err-text small">{error}</div>}

      {clips === null && (
        <div className="card">
          <div className="shimmer" style={{ height: 200 }} />
        </div>
      )}

      {clips?.length === 0 && (
        <div className="card muted center" style={{ padding: 40 }}>
          No broadcast clips yet. Head to{" "}
          <Link to="/admin/long-upload">Long upload</Link> and upload a
          tee + green pair to generate some.
        </div>
      )}

      {clips?.map((c) => {
        const isAce = !!c.ball_in_cup;
        return (
          <div key={c.id} className="card" style={{ marginBottom: 12 }}>
            <div
              className="inline"
              style={{
                justifyContent: "space-between",
                width: "100%",
                marginBottom: 8,
              }}
            >
              <div>
                <b>Clip #{c.id}</b>{" "}
                <span className="muted small">
                  · {c.course_name || `course #${c.course_id}`} · hole{" "}
                  {c.hole_number}{" "}
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
                {isAce && <span className="pill ok small">ace</span>}
                <span className="pill small">dual-cam</span>
              </div>
            </div>

            {c.source_url ? (
              <video
                src={c.source_url}
                poster={c.thumbnail_url || undefined}
                controls
                playsInline
                preload="metadata"
                style={{
                  width: "100%",
                  borderRadius: 8,
                  background: "#000",
                  display: "block",
                }}
              />
            ) : (
              <div
                className="muted center"
                style={{
                  aspectRatio: "16/9",
                  display: "grid",
                  placeItems: "center",
                  border: "2px dashed var(--border)",
                  borderRadius: 8,
                }}
              >
                Source missing
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
