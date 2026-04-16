import { Link } from "react-router-dom";

export default function Home() {
  return (
    <div className="wrap">
      <div className="brand">
        <div className="dot" />
        <h1>Par One</h1>
      </div>
      <div className="card">
        <h2>Golf shot videos, automatic.</h2>
        <p className="muted">
          Scan the QR code at your course, register in under a minute, and get a
          personal gallery of your par-3 shots — tracer overlays and all.
        </p>
      </div>
      <div className="card">
        <h3>For demos</h3>
        <p className="muted small">
          This is a V0 scaffold. Real flows live at these paths:
        </p>
        <ul className="small muted">
          <li><code>/r/&lt;course_token&gt;</code> — mobile registration</li>
          <li><code>/g/&lt;gallery_token&gt;</code> — golfer gallery</li>
          <li><Link to="/admin">/admin</Link> — admin dashboard</li>
          <li><Link to="/admin/review">/admin/review</Link> — hole-in-one review</li>
        </ul>
      </div>
    </div>
  );
}
