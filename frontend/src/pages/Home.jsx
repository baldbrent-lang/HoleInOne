import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { Brand, Icon } from "../components/Brand.jsx";
import LeaderboardCard from "../components/LeaderboardCard.jsx";
import { api } from "../api.js";
import useAuth from "../hooks/useAuth.js";

export default function Home() {
  const { user } = useAuth();
  const [showcase, setShowcase] = useState(null);
  const [contestData, setContestData] = useState(null);
  const [courses, setCourses] = useState(null);
  const [stats, setStats] = useState(null);

  useEffect(() => {
    api.listShowcase().then(setShowcase).catch(() => setShowcase([]));
    api.contests().then(setContestData).catch(() => setContestData({}));
    api.listPublicCourses().then(setCourses).catch(() => setCourses([]));
    api.publicStats().then(setStats).catch(() => setStats(null));
  }, []);

  const monthlyContest = contestData?.monthly?.contests?.[0];
  const yearlyContest = contestData?.yearly?.contests?.[0];

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
        <div style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
          <Link to="/courses" className="btn" style={{ width: "auto" }}>Pick a course — $20</Link>
          <Link to="/sample" className="btn secondary" style={{ width: "auto" }}>See sample gallery</Link>
        </div>
      </div>

      {courses && courses.length > 0 && (
        <div className="trust-band">
          <div className="label">Now live at</div>
          <div className="logos">
            {courses.map((c) => (
              <div key={c.id} className="logo-plate">
                <div className="name">{c.name}</div>
                {c.location && <div className="loc">{c.location}</div>}
              </div>
            ))}
          </div>
          {stats && (stats.clips_this_week > 0 || stats.golfers_today > 0 || stats.aces_pending > 0 || stats.total_clips_delivered > 0) && (
            <div className="stats">
              <span className="pulse" aria-hidden="true" />
              {stats.clips_this_week > 0 && (
                <span><b>{stats.clips_this_week}</b>clip{stats.clips_this_week === 1 ? "" : "s"} delivered this week</span>
              )}
              {stats.golfers_today > 0 && (
                <span><b>{stats.golfers_today}</b>golfer{stats.golfers_today === 1 ? "" : "s"} playing today</span>
              )}
              {stats.aces_pending > 0 && (
                <span><b>{stats.aces_pending}</b>ace claim{stats.aces_pending === 1 ? "" : "s"} under review</span>
              )}
              {stats.clips_this_week === 0 && stats.golfers_today === 0 && stats.total_clips_delivered > 0 && (
                <span><b>{stats.total_clips_delivered}</b>clips delivered to date</span>
              )}
            </div>
          )}
        </div>
      )}

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
          <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 14, flexWrap: "wrap", gap: 8 }}>
            <h3>Current leaderboard</h3>
            <div className="small" style={{ display: "flex", gap: 12 }}>
              <Link to="/contests">Daily / monthly / yearly →</Link>
              <Link to="/leaderboards">All-time →</Link>
            </div>
          </div>
          {!contestData ? (
            <div className="shimmer" style={{ height: 220 }} />
          ) : (
            <div className="stack" style={{ gap: 14 }}>
              <LeaderboardCard
                title={monthlyContest?.title || "Most Rounds Played"}
                icon={monthlyContest?.icon || "users"}
                rows={(monthlyContest?.rows || []).slice(0, 3)}
                emptyText="No rounds logged this month — be the first."
                footer={monthlyContest && (
                  <div className="small center" style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--border)", color: "var(--ink-soft)" }}>
                    <span className="tiny upper muted" style={{ marginRight: 6 }}>This month</span>
                    <b style={{ color: "var(--emerald-700)" }}>{monthlyContest.prize}</b>
                  </div>
                )}
              />
              <LeaderboardCard
                title={yearlyContest?.title || "Most Hole-in-Ones"}
                icon={yearlyContest?.icon || "dollar"}
                rows={(yearlyContest?.rows || []).slice(0, 3)}
                emptyText="No aces yet — be the first to grab the pool."
                footer={yearlyContest && (
                  <div className="small center" style={{ marginTop: 12, paddingTop: 10, borderTop: "1px solid var(--border)", color: "var(--ink-soft)" }}>
                    <span className="tiny upper muted" style={{ marginRight: 6 }}>This year</span>
                    <b style={{ color: "var(--emerald-700)" }}>{yearlyContest.prize}</b>
                  </div>
                )}
              />
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
              <p>Sign up here, or scan the QR at the pro shop. Takes under a minute.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="camera" /></div>
            <div>
              <h4>2. Snap an outfit photo</h4>
              <p>Head-to-toe is all we need — that's how we match shots to you.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="sparkle" /></div>
            <div>
              <h4>3. Play golf. We do the rest.</h4>
              <p>Every par-3 tee shot is filmed and tracer-overlaid automatically.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="share" /></div>
            <div>
              <h4>4. Email delivery</h4>
              <p>One email with all your par-3 clips attached, ready to share.</p>
            </div>
          </div>
          <div className="feature-row" style={{ borderBottom: "none", padding: "8px 0" }}>
            <div className="icon"><Icon name="dollar" /></div>
            <div>
              <h4>5. Hole-in-one? Win <b>$10,000</b>.</h4>
              <p>Ace any par 3, our cup camera verifies it, you get a $10,000 check.</p>
            </div>
          </div>
        </div>
      </div>

      {!user && (
        <div
          className="card"
          style={{
            background: "linear-gradient(135deg, var(--emerald-50), var(--emerald-100))",
            border: "1px solid var(--emerald-200)",
            display: "flex",
            gap: 16,
            alignItems: "flex-start",
          }}
        >
          <div
            style={{
              flexShrink: 0,
              width: 44, height: 44, borderRadius: 12,
              background: "var(--emerald-600)", color: "white",
              display: "grid", placeItems: "center",
            }}
          >
            <Icon name="users" />
          </div>
          <div style={{ flex: 1, minWidth: 0 }}>
            <h3 style={{ color: "var(--emerald-800)", marginBottom: 4 }}>
              Make an account, keep every shot
            </h3>
            <p className="small" style={{ color: "var(--emerald-800)", marginBottom: 12 }}>
              Sign up with your email and every round you ever play with
              GolfReelz lands in your personal dashboard. Pull up past shots,
              re-share clips, and re-register for new rounds in one tap — no
              more digging through old emails.
            </p>
            <div style={{ display: "flex", gap: 8 }}>
              <Link to="/signup" className="btn small" style={{ width: "auto" }}>
                Create free account
              </Link>
              <Link to="/login" className="btn secondary small" style={{ width: "auto" }}>
                Log in
              </Link>
            </div>
          </div>
        </div>
      )}

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
