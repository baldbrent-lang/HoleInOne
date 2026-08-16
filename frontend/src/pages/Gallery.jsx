import { useEffect, useState, useRef } from "react";
import { useParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import ClipPlayer from "../components/ClipPlayer.jsx";

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
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [galleryToken]);

  async function flag(clipId) {
    const note = window.prompt("What's wrong with this clip?");
    if (!note) return;
    showToast("Sending…");
    try {
      await api.flagClip(galleryToken, clipId, note);
      showToast("Flagged for review. Thanks.");
      load();
    } catch (e) {
      showToast(`Error: ${e.message}`);
    }
  }

  function share(clip) {
    const url = clip.source_url;
    if (navigator.share) {
      navigator.share({ title: "My GolfReelz shot", url }).catch(() => {});
    } else {
      navigator.clipboard?.writeText(url);
      showToast("Link copied to clipboard");
    }
  }

  const toastTimer = useRef(null);
  function showToast(msg) {
    setToast(msg);
    if (toastTimer.current) clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(null), 2200);
  }

  if (error) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card"><h2>Gallery not found</h2><p className="small muted">{error}</p></div>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card"><div className="shimmer" style={{ height: 120 }} /></div>
        <div className="card"><div className="shimmer" style={{ height: 120 }} /></div>
      </div>
    );
  }

  const byHole = data.clips.reduce((acc, c) => {
    acc[c.hole_number] ??= [];
    acc[c.hole_number].push(c);
    return acc;
  }, {});
  const holes = Object.keys(byHole).sort((a, b) => Number(a) - Number(b));
  const hasHIO = data.clips.some((c) => c.ball_in_cup);

  return (
    <div className="wrap">
      <Brand />

      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="flag" size={14} /> {data.course_name}</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3.2vw, 2.2rem)" }}>
          {data.participant.name}'s par-3s
        </h1>
        <p>{data.clips.length} clip{data.clips.length === 1 ? "" : "s"} · {holes.length} hole{holes.length === 1 ? "" : "s"}{hasHIO && " · possible ace under review"}</p>
      </div>

      {data.livestream_url && (
        <a href={data.livestream_url} target="_blank" rel="noreferrer" className="live-banner" style={{ textDecoration: "none" }}>
          <span className="dot" />
          <div>
            <div className="label">Live now</div>
            <div className="small">Watch the course livestream.</div>
          </div>
          <span>Watch →</span>
        </a>
      )}

      {holes.length === 0 && (
        <div className="card center" style={{ padding: 28 }}>
          <div className="muted inline" style={{ margin: "0 auto 8px" }}>
            <Icon name="clock" size={18} /> Still processing
          </div>
          <p className="muted small">We'll text the moment your clips are ready.</p>
        </div>
      )}

      {holes.map((h) => (
        <div key={h} className="gallery-hole">
          <h3>
            <span className="hole-badge">{h}</span>
            Hole {h}
          </h3>
          <div className="stack">
            {byHole[h].map((c) => (
              <div key={c.id}>
                <ClipPlayer
                  clip={c}
                  courseName={data.course_name}
                  golferName={data.participant.name}
                  yardage={data.hole_yardages?.[String(c.hole_number)]}
                />
                <div className="inline" style={{ justifyContent: "space-between", marginTop: 8, flexWrap: "wrap", gap: 8 }}>
                  <div className="small muted" style={{ textTransform: "capitalize" }}>
                    {c.camera_type.replace("_", " ")}
                  </div>
                  <div style={{ display: "flex", gap: 6 }}>
                    <a className="btn secondary small" href={c.source_url} download>
                      <Icon name="download" size={14} /> Save
                    </a>
                    <button className="secondary small" onClick={() => share(c)}>
                      <Icon name="share" size={14} /> Share
                    </button>
                    <button className="ghost small" onClick={() => flag(c.id)}>
                      <Icon name="flag" size={14} /> Flag
                    </button>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}

      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}
