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
  const [deleting, setDeleting] = useState({}); // {clip_id: true}
  const [copied, setCopied] = useState({});     // {clip_id: true} for ~1.5s

  /**
   * Compose the title / text / URL we hand to every share target.
   * Native share API gets all three; the social / email / SMS deep
   * links each pick the fields that make sense for them.
   */
  function buildShareInfo(c) {
    const player = c.participant_name || "GolfReelz player";
    const course = c.course_name || "the course";
    const holeStr = c.hole_number ? `Hole ${c.hole_number}` : "this hole";
    const acePrefix = c.ball_in_cup ? "🎯 HOLE-IN-ONE! " : "";
    const title = `${player} — ${holeStr} at ${course}`;
    const text = `${acePrefix}${player} on ${holeStr} at ${course}`;
    return { title, text, url: c.source_url || "" };
  }

  function flashCopied(clipId) {
    setCopied((c) => ({ ...c, [clipId]: true }));
    setTimeout(
      () => setCopied((c) => ({ ...c, [clipId]: false })),
      1600,
    );
  }

  async function nativeShare(c) {
    const info = buildShareInfo(c);
    if (!info.url) return;
    if (navigator.share) {
      try {
        await navigator.share(info);
        return;
      } catch (e) {
        // User cancelled, or the platform doesn't accept these
        // fields — fall through to clipboard.
      }
    }
    try {
      await navigator.clipboard?.writeText(info.url);
      flashCopied(c.id);
    } catch (e) {
      window.prompt("Copy this link:", info.url);
    }
  }

  async function copyLink(c) {
    if (!c.source_url) return;
    try {
      await navigator.clipboard?.writeText(c.source_url);
      flashCopied(c.id);
    } catch (e) {
      window.prompt("Copy this link:", c.source_url);
    }
  }

  function smsHref(c) {
    const { text, url } = buildShareInfo(c);
    // iOS uses `sms:&body=`, Android `sms:?body=`. Most modern OSes
    // tolerate both; `?body=` is the widely-supported form.
    return `sms:?body=${encodeURIComponent(`${text}  ${url}`)}`;
  }

  function emailHref(c) {
    const { title, text, url } = buildShareInfo(c);
    return (
      `mailto:?subject=${encodeURIComponent(title)}` +
      `&body=${encodeURIComponent(`${text}\n\n${url}`)}`
    );
  }

  function twitterHref(c) {
    const { text, url } = buildShareInfo(c);
    return (
      `https://twitter.com/intent/tweet?` +
      `text=${encodeURIComponent(text)}&url=${encodeURIComponent(url)}`
    );
  }

  function facebookHref(c) {
    return `https://www.facebook.com/sharer/sharer.php?u=${encodeURIComponent(c.source_url || "")}`;
  }

  async function load() {
    try {
      const list = await api.listBroadcastClips(adminPassword);
      setClips(list);
    } catch (e) {
      setError(e.message);
    }
  }

  async function deleteClip(clipId) {
    if (!window.confirm("Delete this composite clip and its source files? This can't be undone (the underlying long upload, if any, is kept).")) {
      return;
    }
    setDeleting((d) => ({ ...d, [clipId]: true }));
    try {
      await api.deleteClip(adminPassword, clipId);
      setClips((cs) => (cs || []).filter((c) => c.id !== clipId));
    } catch (e) {
      setError(e.message);
    } finally {
      setDeleting((d) => ({ ...d, [clipId]: false }));
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

            <div
              className="row"
              style={{
                marginTop: 8,
                gap: 6,
                flexWrap: "wrap",
                justifyContent: "flex-end",
                alignItems: "center",
              }}
            >
              <button
                type="button"
                className="small"
                onClick={() => nativeShare(c)}
                disabled={!c.source_url}
                title="Open the OS share sheet (mobile) or copy the link (desktop)"
              >
                <Icon name="share" size={14} /> Share
              </button>
              <button
                type="button"
                className="secondary small"
                onClick={() => copyLink(c)}
                disabled={!c.source_url}
                title="Copy the clip URL to your clipboard"
              >
                {copied[c.id] ? "Copied!" : "Copy link"}
              </button>
              <a
                className="btn secondary small"
                href={c.source_url || "#"}
                download
                style={{ width: "auto" }}
                title="Download the .mp4 file"
              >
                <Icon name="download" size={14} /> Download
              </a>
              <a
                className="btn secondary small"
                href={c.source_url ? smsHref(c) : "#"}
                style={{ width: "auto" }}
                title="Open your default Messages app with this clip prefilled"
              >
                Text
              </a>
              <a
                className="btn secondary small"
                href={c.source_url ? emailHref(c) : "#"}
                style={{ width: "auto" }}
                title="Open your default email app with this clip prefilled"
              >
                Email
              </a>
              <a
                className="btn secondary small"
                href={c.source_url ? twitterHref(c) : "#"}
                target="_blank"
                rel="noopener noreferrer"
                style={{ width: "auto" }}
                title="Post to X / Twitter"
              >
                X
              </a>
              <a
                className="btn secondary small"
                href={c.source_url ? facebookHref(c) : "#"}
                target="_blank"
                rel="noopener noreferrer"
                style={{ width: "auto" }}
                title="Post to Facebook"
              >
                Facebook
              </a>
              <button
                type="button"
                className="ghost small err-text"
                onClick={() => deleteClip(c.id)}
                disabled={!!deleting[c.id]}
                title="Delete this composite + source files"
              >
                {deleting[c.id] ? "Deleting…" : "Delete"}
              </button>
            </div>
          </div>
        );
      })}
    </div>
  );
}
