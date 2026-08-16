import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import ClipPlayer from "../components/ClipPlayer.jsx";

/**
 * Public preview of what a real golfer's gallery looks like — using the
 * featured showcase clip as filler video for all 4 holes. Linked from the
 * Home page so prospective customers can see the deliverable before signing
 * up. No real participant data; nothing here writes to the DB.
 */
export default function SampleGallery() {
  const [showcase, setShowcase] = useState(null);

  useEffect(() => {
    api.listShowcase().then(setShowcase).catch(() => setShowcase([]));
  }, []);

  const featured = (showcase || []).find((s) => s.position === 1 && s.source_url);
  const sampleClipBase = featured
    ? { source_url: featured.source_url, thumbnail_url: featured.thumbnail_url }
    : null;

  const courseName = "Maridoe Golf Club";
  const golferName = "Sample Golfer";
  const holes = [
    { hole_number: 3, camera_type: "tee", ball_in_cup: false },
    { hole_number: 8, camera_type: "tee", ball_in_cup: false },
    { hole_number: 11, camera_type: "tee", ball_in_cup: false },
    { hole_number: 14, camera_type: "tee", ball_in_cup: true },
  ];
  const yardages = { "3": 173, "8": 165, "11": 205, "14": 192 };

  return (
    <div className="wrap">
      <Brand subtitle="Sample gallery preview" />

      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="sparkle" size={14} /> Demo</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3.2vw, 2.2rem)" }}>
          What your gallery looks like
        </h1>
        <p>
          A sample round at {courseName}. Yours arrives in your inbox the same
          way — one email, every par-3 clip attached, every overlay branded.
        </p>
        <Link to="/" className="btn small" style={{ width: "auto", marginTop: 14 }}>
          ← Back home
        </Link>
      </div>

      <div className="card center">
        <div className="muted small">Sample gallery — not a real round</div>
        <h2 style={{ marginTop: 4 }}>{golferName}'s par-3s</h2>
        <p className="muted small">{courseName} · 4 clips · 1 ace 🏌</p>
      </div>

      {!sampleClipBase ? (
        <div className="card muted small">
          The featured clip used in this preview hasn't been uploaded yet.
          Add one in the admin under Home videos.
        </div>
      ) : (
        holes.map((h) => (
          <div key={h.hole_number} className="gallery-hole">
            <h3>
              <span className="hole-badge">{h.hole_number}</span>
              Hole {h.hole_number}
            </h3>
            <ClipPlayer
              clip={{ ...sampleClipBase, ...h }}
              courseName={courseName}
              golferName={golferName}
              yardage={yardages[String(h.hole_number)]}
            />
            <div className="small muted" style={{ marginTop: 8 }}>
              {h.camera_type.replace("_", " ")}
              {h.ball_in_cup && "  ·  ACE!"}
            </div>
          </div>
        ))
      )}

      <div className="card center" style={{ background: "var(--primary-soft)", border: "1px solid var(--emerald-200)" }}>
        <h3 style={{ color: "var(--emerald-800)" }}>Want this for your next round?</h3>
        <p className="small" style={{ color: "var(--emerald-800)", marginBottom: 12 }}>
          Pick a course and register. $20 — chance at $10,000 if you ace one.
        </p>
        <Link to="/courses" className="btn" style={{ maxWidth: 260, margin: "0 auto" }}>
          Browse courses
        </Link>
      </div>
    </div>
  );
}
