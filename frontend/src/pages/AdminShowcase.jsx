import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

const ADMIN_PW_STORAGE = "golfreelz.adminPassword";
const LEGACY_ADMIN_PW_STORAGE = "parone.adminPassword";

export default function AdminShowcase() {
  const adminPassword =
    localStorage.getItem(ADMIN_PW_STORAGE) ||
    localStorage.getItem(LEGACY_ADMIN_PW_STORAGE) ||
    "";
  const [slots, setSlots] = useState([]);
  const [error, setError] = useState(null);
  const [toast, setToast] = useState(null);

  async function load() {
    try {
      setSlots(await api.adminListShowcase(adminPassword));
    } catch (e) {
      setError(e.message);
    }
  }

  useEffect(() => {
    if (adminPassword) load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  function showToast(msg) { setToast(msg); setTimeout(() => setToast(null), 2200); }

  if (!adminPassword) {
    return (
      <div className="wrap">
        <Brand subtitle="Operator Console" />
        <div className="card center"><h2>Admin password required</h2><Link to="/admin"><button style={{ marginTop: 10 }}>Sign in</button></Link></div>
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
        <Link to="/admin/broadcast-clips">Broadcast</Link>
        <Link to="/admin/cameras">Cameras</Link>
      </div>

      <div className="card">
        <h3 style={{ marginBottom: 4 }}>Home page video</h3>
        <p className="small muted" style={{ marginBottom: 14 }}>
          Shown on the public Home page under "Our videos in action". Upload
          an MP4 or paste an external URL. Leave blank to hide the section.
        </p>
      </div>

      {slots
        .filter((s) => s.position === 1)
        .map((s) => (
          <ShowcaseSlot
            key={s.position}
            slot={s}
            adminPassword={adminPassword}
            onChange={() => { showToast("Saved"); load(); }}
            onError={(m) => showToast(m)}
          />
        ))}

      {error && <div className="card err-text small" style={{ marginTop: 12 }}>{error}</div>}
      {toast && <div className="toast">{toast}</div>}
    </div>
  );
}

function ShowcaseSlot({ slot, adminPassword, onChange, onError }) {
  const [title, setTitle] = useState(slot.title || "");
  const [caption, setCaption] = useState(slot.caption || "");
  const [externalUrl, setExternalUrl] = useState("");
  const [file, setFile] = useState(null);
  const [progress, setProgress] = useState(0);
  const [busy, setBusy] = useState(false);

  async function uploadFile() {
    if (!file) return;
    setBusy(true);
    setProgress(0);
    try {
      const fd = new FormData();
      fd.append("video", file, file.name);
      if (title.trim()) fd.append("title", title.trim());
      if (caption.trim()) fd.append("caption", caption.trim());
      await api.uploadShowcase(adminPassword, slot.position, fd, setProgress);
      setFile(null);
      onChange?.();
    } catch (e) {
      onError?.(`Error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function saveExternal() {
    setBusy(true);
    try {
      await api.updateShowcase(adminPassword, slot.position, {
        source_url: externalUrl.trim() || null,
        thumbnail_url: null,
        title: title.trim() || null,
        caption: caption.trim() || null,
      });
      setExternalUrl("");
      onChange?.();
    } catch (e) {
      onError?.(`Error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function saveMeta() {
    setBusy(true);
    try {
      await api.updateShowcase(adminPassword, slot.position, {
        title: title.trim() || null,
        caption: caption.trim() || null,
      });
      onChange?.();
    } catch (e) {
      onError?.(`Error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  async function clear() {
    if (!window.confirm(`Clear slot ${slot.position}?`)) return;
    setBusy(true);
    try {
      await api.clearShowcase(adminPassword, slot.position);
      onChange?.();
    } catch (e) {
      onError?.(`Error: ${e.message}`);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="card">
      <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 8 }}>
        <h3>Featured video</h3>
        {slot.source_url && (
          <button className="ghost small" onClick={clear} disabled={busy}>Clear</button>
        )}
      </div>

      {slot.source_url ? (
        <video
          src={slot.source_url}
          poster={slot.thumbnail_url || undefined}
          controls
          playsInline
          preload="metadata"
          style={{ width: "100%", aspectRatio: "16/9", borderRadius: 8, background: "#000", marginBottom: 10 }}
        />
      ) : (
        <div
          style={{
            aspectRatio: "16/9",
            background: "var(--surface-alt)",
            border: "2px dashed var(--border-strong)",
            borderRadius: 8,
            display: "grid",
            placeItems: "center",
            color: "var(--ink-soft)",
            marginBottom: 10,
            fontSize: "0.85rem",
          }}
        >
          (empty)
        </div>
      )}

      <div className="field">
        <label>Title <span className="muted small">(optional)</span></label>
        <input value={title} onChange={(e) => setTitle(e.target.value)} placeholder="e.g. Ben's hole-in-one" />
      </div>
      <div className="field">
        <label>Caption <span className="muted small">(optional)</span></label>
        <input value={caption} onChange={(e) => setCaption(e.target.value)} placeholder="e.g. Maridoe Golf Club · Hole 14" />
      </div>

      <div className="row">
        <button className="secondary small" onClick={saveMeta} disabled={busy} style={{ flex: 1 }}>
          Save title / caption
        </button>
      </div>

      <div className="divider" />

      <div className="field">
        <label>Upload a video</label>
        <input
          type="file"
          accept="video/*"
          onChange={(e) => setFile(e.target.files?.[0] || null)}
        />
        {file && (
          <div className="small muted" style={{ marginTop: 4 }}>
            {file.name} · {(file.size / 1024 / 1024).toFixed(1)} MB
          </div>
        )}
      </div>
      {busy && progress > 0 && (
        <div style={{ height: 6, background: "var(--surface-alt)", borderRadius: 3, overflow: "hidden", marginBottom: 8 }}>
          <div style={{ width: `${progress}%`, height: "100%", background: "var(--primary)" }} />
        </div>
      )}
      <button onClick={uploadFile} disabled={busy || !file}>
        {busy && progress > 0 ? `Uploading… ${progress}%` : `Upload to slot ${slot.position}`}
      </button>

      <div className="divider" />

      <div className="field">
        <label>…or paste a video URL <span className="muted small">(YouTube /embed, MP4 link, etc.)</span></label>
        <input value={externalUrl} onChange={(e) => setExternalUrl(e.target.value)} placeholder="https://…" />
      </div>
      <button className="secondary" onClick={saveExternal} disabled={busy || !externalUrl.trim()}>
        Use this URL
      </button>
    </div>
  );
}
