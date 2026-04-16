import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api.js";

export default function Gallery() {
  const { galleryToken } = useParams();
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);

  async function load() {
    try {
      setData(await api.gallery(galleryToken));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    load();
  }, [galleryToken]);

  async function flag(clipId) {
    const note = window.prompt("What's wrong with this clip?");
    if (!note) return;
    try {
      await api.flagClip(galleryToken, clipId, note);
      setToast("Flagged for review. Thanks.");
      load();
    } catch (e) {
      setToast(`Error: ${e.message}`);
    }
    setTimeout(() => setToast(null), 2500);
  }

  function share(clip) {
    const url = clip.source_url;
    if (navigator.share) {
      navigator.share({ title: "My Par One shot", url }).catch(() => {});
    } else {
      navigator.clipboard?.writeText(url);
      setToast("Link copied to clipboard");
      setTimeout(() => setToast(null), 2000);
    }
  }

  if (error) {
    return (
      <div className="wrap">
        <div className="card">
          <h1>Gallery not found</h1>
          <p className="muted small">{error}</p>
        </div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="wrap">
        <div className="card muted">Loading your gallery…</div>
      </div>
    );
  }

  const byHole = data.clips.reduce((acc, c) => {
    acc[c.hole_number] ??= [];
    acc[c.hole_number].push(c);
    return acc;
  }, {});
  const holes = Object.keys(byHole).sort((a, b) => Number(a) - Number(b));

  return (
    <div className="wrap">
      <div className="brand">
        <div className="dot" />
        <h1>Par One</h1>
      </div>
      <div className="card">
        <h2>{data.participant.name}'s shots</h2>
        <p className="muted small">{data.course_name}</p>
      </div>

      {holes.length === 0 && (
        <div className="card muted">
          No clips yet — we'll text you the moment they're ready.
        </div>
      )}

      {holes.map((h) => (
        <div key={h} className="card">
          <h3>Hole {h}</h3>
          {byHole[h].map((c) => (
            <div key={c.id} className="clip">
              <div
                className={`thumb ${c.ball_in_cup ? "hio" : ""}`}
                style={c.thumbnail_url ? { backgroundImage: `url(${c.thumbnail_url})` } : {}}
              />
              <div className="meta">
                <b>{c.camera_type.replace("_", " ")}</b>
                {c.ball_in_cup && <span className="pill warn" style={{ marginLeft: 6 }}>HIO review</span>}
                <div className="stats">
                  {c.carry_yards ? `${c.carry_yards} yds · ` : ""}
                  {c.apex_feet ? `${c.apex_feet} ft apex · ` : ""}
                  {c.ball_speed_mph ? `${c.ball_speed_mph} mph` : ""}
                </div>
                <div className="row" style={{ marginTop: 8 }}>
                  <a className="btn" href={c.source_url} download style={{ textAlign: "center" }}>
                    Download
                  </a>
                  <button className="secondary" onClick={() => share(c)}>Share</button>
                  <button className="secondary" onClick={() => flag(c.id)}>Flag</button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ))}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
