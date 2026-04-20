import { Link } from "react-router-dom";
import { Brand, Icon } from "../components/Brand.jsx";

export default function Home() {
  return (
    <div className="wrap">
      <Brand subtitle="Par 3 Videos and Hole In One Sweepstakes!" />

      <div className="hero">
        <span className="eyebrow">
          <Icon name="sparkle" size={14} /> Now on course
        </span>
        <h1>Every par-3 shot, tracked and delivered.</h1>
        <p>
          Scan the QR code at your course, register in under a minute, and we
          text your personal gallery when your round is done — tracer overlays,
          stats, and shareable clips included.
        </p>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 10 }}>How it works</h3>
        <div className="feature-row">
          <div className="icon"><Icon name="qr" /></div>
          <div>
            <h4>1. Scan at the pro shop</h4>
            <p>A QR code on the counter opens this page, pre-loaded for your course and today's tee sheet.</p>
          </div>
        </div>
        <div className="feature-row">
          <div className="icon"><Icon name="camera" /></div>
          <div>
            <h4>2. Register + snap an outfit photo</h4>
            <p>We match your shots to you using your outfit. Head-to-toe is all we need — 60 seconds flat.</p>
          </div>
        </div>
        <div className="feature-row">
          <div className="icon"><Icon name="sparkle" /></div>
          <div>
            <h4>3. Play golf. We'll do the rest.</h4>
            <p>Every par-3 tee shot is filmed by on-course cameras, auto-processed with tracer overlays, and assembled into your gallery.</p>
          </div>
        </div>
        <div className="feature-row">
          <div className="icon"><Icon name="share" /></div>
          <div>
            <h4>4. Text or email, one tap to share</h4>
            <p>Download a clip, share to Instagram, or send it to your group chat. Hole-in-one? We verify it with the cup camera.</p>
          </div>
        </div>
      </div>

      <div className="card" style={{ background: "var(--primary-soft)", border: "1px solid var(--emerald-200)" }}>
        <h3 style={{ color: "var(--emerald-800)" }}>For operators + testers</h3>
        <p className="small" style={{ color: "var(--emerald-800)" }}>
          Real flows live at these paths:
        </p>
        <div className="stack" style={{ gap: 4, marginTop: 8 }}>
          <div className="small"><code>/r/&lt;course_token&gt;</code> — mobile registration</div>
          <div className="small"><code>/g/&lt;gallery_token&gt;</code> — golfer gallery</div>
          <div className="small"><Link to="/admin">/admin</Link> — operator dashboard</div>
          <div className="small"><Link to="/admin/review">/admin/review</Link> — hole-in-one verification queue</div>
        </div>
      </div>
    </div>
  );
}
