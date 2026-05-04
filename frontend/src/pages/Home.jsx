import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Brand, Icon } from "../components/Brand.jsx";
import { api } from "../api.js";

export default function Home() {
  const [courses, setCourses] = useState(null);
  const [showcase, setShowcase] = useState(null);

  useEffect(() => {
    api.listPublicCourses().then(setCourses).catch(() => setCourses([]));
    api.listShowcase().then(setShowcase).catch(() => setShowcase([]));
  }, []);

  // Single featured video for now — only slot 1 appears on Home.
  const featured = (showcase || []).find((s) => s.position === 1 && s.source_url);
  const showcaseLoaded = showcase !== null;

  return (
    <div className="wrap wide">
      <Brand subtitle="Par 3 Videos and Hole In One Sweepstakes!" />

      <div className="hero">
        <span className="eyebrow">
          <Icon name="sparkle" size={14} /> Now on course
        </span>
        <h1>Every par-3 shot, tracked and delivered.</h1>
        <p>
          Two ways to register: right here on the website, or by scanning the
          QR code at your course. Every par-3 tee shot is filmed with tracer
          overlays and emailed to you after your round. Make a hole-in-one and
          win <b>$10,000</b>.
        </p>
      </div>

      <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(380px, 1fr))", gap: 16 }}>
        {(featured || !showcaseLoaded) && (
          <div className="card" style={{ marginBottom: 0 }}>
            <h3 style={{ marginBottom: 4 }}>Our videos in action</h3>
            <p className="small muted" style={{ marginBottom: 14 }}>
              A clip from a real GolfReelz round.
            </p>
            {!showcaseLoaded ? (
              <div className="shimmer" style={{ aspectRatio: "16/9", borderRadius: 8 }} />
            ) : (
              <>
                <video
                  src={featured.source_url}
                  poster={featured.thumbnail_url || undefined}
                  controls
                  playsInline
                  preload="metadata"
                  style={{ width: "100%", aspectRatio: "16/9", borderRadius: 8, background: "#000", display: "block" }}
                />
                {(featured.title || featured.caption) && (
                  <div style={{ marginTop: 10 }}>
                    {featured.title && <b style={{ display: "block" }}>{featured.title}</b>}
                    {featured.caption && <div className="small muted">{featured.caption}</div>}
                  </div>
                )}
              </>
            )}
          </div>
        )}

        <div className="card" style={{ marginBottom: 0 }}>
          <h3 style={{ marginBottom: 4 }}>Pick your course</h3>
          <p className="small muted" style={{ marginBottom: 14 }}>
            Tap a course to start your registration.
          </p>
          {courses === null ? (
            <>
              <div className="shimmer" style={{ height: 64, marginBottom: 8 }} />
              <div className="shimmer" style={{ height: 64, marginBottom: 8 }} />
              <div className="shimmer" style={{ height: 64 }} />
            </>
          ) : courses.length === 0 ? (
            <div className="muted small">No courses available yet.</div>
          ) : (
            <div className="stack" style={{ gap: 10 }}>
              {courses.map((c) => (
                <Link
                  key={c.id}
                  to={`/r/${c.qr_token}`}
                  className="card"
                  style={{
                    margin: 0,
                    display: "flex",
                    alignItems: "center",
                    gap: 14,
                    textDecoration: "none",
                    color: "inherit",
                    padding: 14,
                  }}
                >
                  <div
                    className="logo"
                    style={{ background: "var(--primary-soft)", color: "var(--emerald-700)", flexShrink: 0 }}
                    aria-hidden="true"
                  >
                    <Icon name="flag" />
                  </div>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <b style={{ display: "block" }}>{c.name}</b>
                    <div className="small muted">{c.location || "—"}</div>
                  </div>
                  <span style={{ color: "var(--ink-soft)" }}>→</span>
                </Link>
              ))}
            </div>
          )}
        </div>
      </div>

      <div className="card" style={{ marginTop: 16 }}>
        <h3 style={{ marginBottom: 14 }}>How it works</h3>
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))", gap: 12 }}>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="qr" /></div>
            <div>
              <h4>1. Register before you play</h4>
              <p>Sign up right here on the website, or scan the QR code at your course's pro shop. Either way takes under a minute.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="camera" /></div>
            <div>
              <h4>2. Register + snap an outfit photo</h4>
              <p>We match your shots to you using your outfit. Head-to-toe is all we need — 60 seconds flat.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="sparkle" /></div>
            <div>
              <h4>3. Play golf. We'll do the rest.</h4>
              <p>Every par-3 tee shot is filmed by on-course cameras, auto-processed with tracer overlays, and assembled into your gallery.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="share" /></div>
            <div>
              <h4>4. Email delivery, one tap to share</h4>
              <p>One email lands in your inbox with all your par-3 clips attached. Hole-in-one? We verify it with the cup camera.</p>
            </div>
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
