import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import { useInfiniteList } from "../hooks/useInfiniteList.js";
import { fmtDateTime } from "../time.js";

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
  const fetchClips = useCallback(
    (limit, offset) => api.listBroadcastClips(adminPassword, limit, offset),
    [adminPassword],
  );
  const {
    items: clips,
    setItems: setClips,
    sentinelRef,
    hasMore,
    loadingMore,
    error: listError,
    reload,
  } = useInfiniteList(fetchClips, { pageSize: 25, deps: [adminPassword] });
  const [error, setError] = useState(null);
  // Combine the hook's load error with action errors so the UI keeps one
  // error banner.
  const displayError = error || listError;
  const [deleting, setDeleting] = useState({}); // {clip_id: true}
  const [copied, setCopied] = useState({});     // {clip_id: true} for ~1.5s
  const [fileSharing, setFileSharing] = useState({}); // {clip_id: true} while we fetch the .mp4 for a file share

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

  /**
   * Build a descriptive filename for the downloaded / shared file.
   * Keeps the destination apps (TikTok, Instagram) from showing a
   * cryptic UUID as the clip's name.
   */
  function suggestedFilename(c) {
    const slug = (c.participant_name || "golfreelz")
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, "_")
      .replace(/^_+|_+$/g, "") || "golfreelz";
    const hole = c.hole_number ? `_hole${c.hole_number}` : "";
    return `${slug}${hole}.mp4`;
  }

  /**
   * Fetch the clip's .mp4 as a File so we can hand it to
   * navigator.share — the only way to natively post to Instagram /
   * TikTok / other apps that don't accept URL-share intents.
   */
  async function fetchClipAsFile(c) {
    if (!c.source_url) return null;
    try {
      const res = await fetch(c.source_url);
      if (!res.ok) return null;
      const blob = await res.blob();
      return new File([blob], suggestedFilename(c), {
        type: blob.type || "video/mp4",
      });
    } catch (e) {
      return null;
    }
  }

  /**
   * Try a file-share via the OS share sheet (mobile native flow
   * surfaces Instagram, TikTok, Stories, etc. as targets). On any
   * failure — unsupported browser, fetch error, user cancel — fall
   * back to downloading the .mp4 with a hint to open the target app.
   *
   * `appHint` is purely cosmetic, used in the fallback message.
   */
  async function fileShareForApp(c, appHint) {
    if (!c.source_url) return;
    setFileSharing((s) => ({ ...s, [c.id]: true }));
    try {
      const file = await fetchClipAsFile(c);
      if (!file) {
        window.alert(
          `Couldn't fetch the clip to share. Try Download, then open ${appHint} and upload manually.`,
        );
        return;
      }
      if (
        navigator.canShare &&
        navigator.canShare({ files: [file] }) &&
        navigator.share
      ) {
        try {
          await navigator.share({
            files: [file],
            title: buildShareInfo(c).title,
            text: buildShareInfo(c).text,
          });
          return;
        } catch (e) {
          if (e && e.name === "AbortError") return; // user closed sheet
          // fall through to download fallback
        }
      }
      // Desktop / unsupported browsers: trigger a download and
      // explain the manual next step.
      const url = URL.createObjectURL(file);
      const a = document.createElement("a");
      a.href = url;
      a.download = file.name;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      setTimeout(() => URL.revokeObjectURL(url), 5000);
      window.alert(
        `Downloaded the clip as ${file.name}. Open ${appHint} on your phone and upload it from your camera roll / files.`,
      );
    } finally {
      setFileSharing((s) => ({ ...s, [c.id]: false }));
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
        <Link to="/admin/participants">Players</Link>
        <Link to="/admin/courses">Courses</Link>
        <Link to="/admin/upload-videos">Upload</Link>
        <Link to="/admin/production">Production</Link>
        <Link to="/admin/produced-clips">Produced Clips</Link>
        <Link to="/admin/broadcast-clips" className="active">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
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

      {displayError && (
        <div className="card err-text small">{displayError}</div>
      )}

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
                  · {fmtDateTime(c.captured_at)}
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
                className="secondary small"
                onClick={() => fileShareForApp(c, "Instagram")}
                disabled={!c.source_url || !!fileSharing[c.id]}
                title="On mobile: opens the OS share sheet with the video as a file (Instagram appears as a target). On desktop: downloads the .mp4 and reminds you to upload manually."
              >
                {fileSharing[c.id] ? "Preparing…" : "Instagram"}
              </button>
              <button
                type="button"
                className="secondary small"
                onClick={() => fileShareForApp(c, "TikTok")}
                disabled={!c.source_url || !!fileSharing[c.id]}
                title="On mobile: opens the OS share sheet with the video as a file (TikTok appears as a target). On desktop: downloads the .mp4 and reminds you to upload manually."
              >
                {fileSharing[c.id] ? "Preparing…" : "TikTok"}
              </button>
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

      {clips && clips.length > 0 && (
        <div
          ref={sentinelRef}
          className="muted center small"
          style={{ padding: 20 }}
        >
          {loadingMore
            ? "Loading more…"
            : hasMore
              ? "Scroll for more"
              : "End of broadcast clips"}
        </div>
      )}
    </div>
  );
}
