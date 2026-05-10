import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

/**
 * Near-live broadcast channel. Polls /api/broadcast/next, autoplays the
 * returned clip, then asks for the next one. During dead time the server
 * returns `kind: "slate"` and we render a leaderboard card for a few
 * seconds before re-polling.
 *
 * Query params:
 *   ?course_id=N  — restrict the channel to one course (clubhouse-TV mode).
 *   ?fullscreen=1 — hide all chrome for kiosk displays.
 */

const VIEWER_KEY = "golfreelz.viewerId";

function viewerId() {
  let id = localStorage.getItem(VIEWER_KEY);
  if (!id) {
    id = `v_${crypto.randomUUID ? crypto.randomUUID() : Math.random().toString(36).slice(2)}`;
    localStorage.setItem(VIEWER_KEY, id);
  }
  return id;
}

export default function Watch() {
  const [params] = useSearchParams();
  const courseId = params.get("course_id") || null;
  const kiosk = params.get("fullscreen") === "1";

  const [item, setItem] = useState(null);
  const [error, setError] = useState(null);
  const [contests, setContests] = useState(null);
  const videoRef = useRef(null);
  const advanceTimer = useRef(null);

  async function next() {
    try {
      const data = await api.broadcastNext(viewerId(), courseId);
      setItem(data);
      setError(null);
    } catch (e) {
      setError(e.message);
      // Try again in a bit; don't tightloop on errors.
      advanceTimer.current = setTimeout(next, 6000);
    }
  }

  // Initial pull + grab leaderboard data for slate frames.
  useEffect(() => {
    next();
    api.contests().then(setContests).catch(() => {});
    return () => clearTimeout(advanceTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [courseId]);

  // Slate frames advance themselves on a timer.
  useEffect(() => {
    if (item?.kind === "slate") {
      clearTimeout(advanceTimer.current);
      advanceTimer.current = setTimeout(next, item.duration_ms || 8000);
    }
    return () => clearTimeout(advanceTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [item]);

  // Video advances on ended / errored.
  function onVideoEnded() { next(); }
  function onVideoError() {
    // Give it a beat in case it's a transient buffering thing.
    clearTimeout(advanceTimer.current);
    advanceTimer.current = setTimeout(next, 2500);
  }

  return (
    <div className={kiosk ? "watch-stage kiosk" : "watch-stage"}>
      {!kiosk && (
        <div className="wrap wide" style={{ paddingBottom: 0 }}>
          <Brand subtitle="Broadcast" />
          <div className="inline" style={{ justifyContent: "space-between", marginBottom: 8 }}>
            <p className="small muted" style={{ margin: 0 }}>
              <Icon name="play" size={14} /> Par-3 swings, near-live.
              {courseId ? " Single-course feed." : " All courses."}
            </p>
            <div className="inline" style={{ gap: 8 }}>
              <Link to="/leaderboards" className="btn ghost small" style={{ width: "auto" }}>Leaderboards</Link>
              <Link to="/contests" className="btn ghost small" style={{ width: "auto" }}>Contests</Link>
            </div>
          </div>
        </div>
      )}

      <div className="watch-frame">
        {error && (
          <div className="watch-error">
            <p>Couldn't load the channel: <code>{error}</code></p>
          </div>
        )}

        {item?.kind === "swing" && (
          <>
            <video
              ref={videoRef}
              src={item.video_url}
              poster={item.thumbnail_url || undefined}
              autoPlay
              playsInline
              muted /* required for autoplay on most browsers */
              onEnded={onVideoEnded}
              onError={onVideoError}
              className="watch-video"
            />
            <ClipOverlay item={item} />
          </>
        )}

        {item?.kind === "slate" && (
          <Slate contests={contests} courseId={courseId} />
        )}

        {!item && !error && <div className="watch-loading">Loading channel…</div>}
      </div>
    </div>
  );
}

function ClipOverlay({ item }) {
  return (
    <>
      <div className="watch-overlay tr">
        <div className="tiny upper muted-on-dark">
          {item.course_name || "GolfReelz"}
        </div>
        <div className="watch-overlay-title">
          Hole {item.hole_number}
          {item.yardage ? ` · ${item.yardage}y` : ""}
        </div>
        {item.is_highlight && (
          <span className="pill ok" style={{ marginTop: 6 }}>
            {item.highlight_tag === "01-ace" ? "ACE" :
             item.highlight_tag === "02-stiff" ? "STIFF" :
             "HIGHLIGHT"}
          </span>
        )}
      </div>
      {item.golfer_name && (
        <div className="watch-overlay br">
          <div className="watch-overlay-title">{item.golfer_name}</div>
          {item.ball_in_cup && (
            <div className="tiny upper" style={{ color: "var(--emerald-300)" }}>
              ball in cup
            </div>
          )}
          {!item.ball_in_cup && item.distance_from_pin_feet != null && (
            <div className="tiny upper muted-on-dark">
              {item.distance_from_pin_feet}ft from pin
            </div>
          )}
        </div>
      )}
    </>
  );
}

function Slate({ contests, courseId }) {
  // Lift a few rows to show during dead time. Cycle through them based on
  // current second so consecutive slates show different content.
  if (!contests) {
    return <div className="watch-loading">…</div>;
  }
  const slateIndex = Math.floor(Date.now() / 8000) % 3;
  if (slateIndex === 0) return <SlateMonthly contests={contests} />;
  if (slateIndex === 1) return <SlateYearly contests={contests} />;
  return <SlateDaily contests={contests} courseId={courseId} />;
}

function SlateMonthly({ contests }) {
  const m = contests?.monthly?.contests?.[0];
  if (!m) return <SlateFallback />;
  return (
    <div className="watch-slate">
      <div className="tiny upper muted-on-dark">This month</div>
      <h1 className="watch-slate-title">{m.title}</h1>
      <ol className="watch-slate-list">
        {(m.entries || []).slice(0, 5).map((e, i) => (
          <li key={i}>
            <span className="rank">#{i + 1}</span>
            <span className="who">{e.name}</span>
            <span className="metric">{e.value}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function SlateYearly({ contests }) {
  const y = contests?.yearly?.contests?.[0];
  if (!y) return <SlateFallback />;
  return (
    <div className="watch-slate">
      <div className="tiny upper muted-on-dark">This year · $10,000 still up for grabs</div>
      <h1 className="watch-slate-title">{y.title}</h1>
      <ol className="watch-slate-list">
        {(y.entries || []).slice(0, 5).map((e, i) => (
          <li key={i}>
            <span className="rank">#{i + 1}</span>
            <span className="who">{e.name}</span>
            <span className="metric">{e.value}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function SlateDaily({ contests, courseId }) {
  const list = contests?.daily?.contests || [];
  const c = (courseId
    ? list.find((x) => String(x.course_id) === String(courseId))
    : list[0]) || list[0];
  if (!c) return <SlateFallback />;
  return (
    <div className="watch-slate">
      <div className="tiny upper muted-on-dark">Today · Closest to the pin</div>
      <h1 className="watch-slate-title">{c.title}</h1>
      <ol className="watch-slate-list">
        {(c.entries || []).slice(0, 5).map((e, i) => (
          <li key={i}>
            <span className="rank">#{i + 1}</span>
            <span className="who">{e.name}</span>
            <span className="metric">{e.value}</span>
          </li>
        ))}
      </ol>
    </div>
  );
}

function SlateFallback() {
  return (
    <div className="watch-slate">
      <div className="tiny upper muted-on-dark">GolfReelz</div>
      <h1 className="watch-slate-title">Every par-3 shot, tracked and delivered.</h1>
      <p className="muted-on-dark">Hole in one on any par 3 — win $10,000.</p>
    </div>
  );
}
