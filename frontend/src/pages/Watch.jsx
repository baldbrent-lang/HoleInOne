import { useEffect, useRef, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import { fmtDateTime } from "../time.js";

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
  // Share-link mode (/c/<token>): the token is signed server-side and
  // resolves to EXACTLY one channel — the only thing this URL can play.
  // Handed to club pros / GMs; no channel picker, no admin chrome.
  const { shareToken } = useParams();
  const courseId = shareToken ? null : params.get("course_id") || null;
  // Channel mode (?channel=course-2 | best): a continuous looping
  // playlist of Broadcast clips instead of the per-viewer near-live
  // picker. The playlist refetches on every wrap so freshly promoted
  // clips join the loop without a reload.
  const channel = shareToken || params.get("channel") || null;
  const kiosk = params.get("fullscreen") === "1";

  const [item, setItem] = useState(null);
  const [error, setError] = useState(null);
  const [contests, setContests] = useState(null);
  const [channelInfo, setChannelInfo] = useState(null);
  const [airplayAvailable, setAirplayAvailable] = useState(false);
  const videoRef = useRef(null);
  const frameRef = useRef(null);
  const advanceTimer = useRef(null);
  const playlistRef = useRef({ clips: [], idx: -1 });

  // AirPlay (Safari/iOS/macOS): the video element announces when a
  // playback target (Apple TV / AirPlay-capable TV) is reachable.
  useEffect(() => {
    const v = videoRef.current;
    if (!v || !window.WebKitPlaybackTargetAvailabilityEvent) return;
    const onAvail = (e) =>
      setAirplayAvailable(e.availability === "available");
    v.addEventListener("webkitplaybacktargetavailabilitychanged", onAvail);
    return () =>
      v.removeEventListener(
        "webkitplaybacktargetavailabilitychanged", onAvail,
      );
  }, [item?.kind]);

  function showAirplayPicker() {
    try {
      videoRef.current?.webkitShowPlaybackTargetPicker?.();
    } catch {
      /* not supported — button only renders when it is */
    }
  }

  function toggleFullscreen() {
    const el = frameRef.current;
    if (!el) return;
    if (document.fullscreenElement || document.webkitFullscreenElement) {
      (document.exitFullscreen || document.webkitExitFullscreen)?.call(
        document,
      );
      return;
    }
    (el.requestFullscreen || el.webkitRequestFullscreen)?.call(el);
  }

  async function next() {
    if (channel) {
      try {
        let pl = playlistRef.current;
        if (pl.idx + 1 >= pl.clips.length) {
          const data = shareToken
            ? await api.sharedChannelPlaylist(shareToken)
            : await api.broadcastChannelPlaylist(channel);
          setChannelInfo({ label: data.label, count: (data.clips || []).length });
          pl = { clips: data.clips || [], idx: -1 };
          playlistRef.current = pl;
        }
        if (!pl.clips.length) {
          // Empty channel — show a slate and re-check in a bit.
          setItem({ kind: "slate", duration_ms: 15000 });
          setError(null);
          return;
        }
        pl.idx += 1;
        setItem(pl.clips[pl.idx]);
        setError(null);
      } catch (e) {
        setError(e.message);
        advanceTimer.current = setTimeout(next, 6000);
      }
      return;
    }
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
    playlistRef.current = { clips: [], idx: -1 };
    next();
    api.contests().then(setContests).catch(() => {});
    return () => clearTimeout(advanceTimer.current);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [courseId, channel]);

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
              <Icon name="play" size={14} />{" "}
              {channel
                ? `${channelInfo?.label || "Channel"} — continuous broadcast${
                    channelInfo ? ` · ${channelInfo.count} clips` : ""
                  }`
                : `Par-3 swings, near-live.${
                    courseId ? " Single-course feed." : " All courses."
                  }`}
            </p>
            {!shareToken && (
              <div className="inline" style={{ gap: 8 }}>
                <Link to="/contests" className="btn ghost small" style={{ width: "auto" }}>Contests</Link>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="watch-frame" ref={frameRef}>
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
              x-webkit-airplay="allow"
              onEnded={onVideoEnded}
              onError={onVideoError}
              className="watch-video"
            />
            <ClipOverlay item={item} />
          </>
        )}

        {/* Player controls — bottom-left, out of the way of the clip
            info. AirPlay only appears when a target is reachable
            (Safari-family browsers). */}
        <div
          style={{
            position: "absolute", left: 10, bottom: 10, zIndex: 5,
            display: "flex", gap: 8,
          }}
        >
          <button
            type="button"
            onClick={toggleFullscreen}
            title="Fill the screen"
            style={{
              background: "rgba(0,0,0,0.45)", color: "#fff",
              border: "1px solid rgba(255,255,255,0.35)",
              borderRadius: 6, width: 34, height: 30,
              cursor: "pointer", fontSize: 15, lineHeight: 1, padding: 0,
            }}
          >
            ⛶
          </button>
          {airplayAvailable && (
            <button
              type="button"
              onClick={showAirplayPicker}
              title="Play on a TV via AirPlay"
              style={{
                background: "rgba(0,0,0,0.45)", color: "#fff",
                border: "1px solid rgba(255,255,255,0.35)",
                borderRadius: 6, width: 34, height: 30,
                cursor: "pointer", fontSize: 14, lineHeight: 1, padding: 0,
              }}
            >
              📺
            </button>
          )}
        </div>

        {item?.kind === "slate" && (
          <Slate contests={contests} courseId={courseId} />
        )}

        {!item && !error && <div className="watch-loading">Loading channel…</div>}
      </div>
    </div>
  );
}

function ClipOverlay({ item }) {
  // Deliberately minimal chrome: no course/hole/HIGHLIGHT box — just
  // the golfer (when known) and the clip's capture date + time in the
  // bottom-right corner.
  const when = item.captured_at
    ? fmtDateTime(item.captured_at, [], {
        month: "short",
        day: "numeric",
        hour: "numeric",
        minute: "2-digit",
      })
    : null;
  if (!when && !item.golfer_name) return null;
  return (
    <div className="watch-overlay br">
      {item.golfer_name && (
        <div className="watch-overlay-title">{item.golfer_name}</div>
      )}
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
      {when && (
        <div className="tiny upper muted-on-dark">{when}</div>
      )}
    </div>
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
