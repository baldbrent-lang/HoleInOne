import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import SelfieCamera from "../components/SelfieCamera.jsx";

/**
 * Invited group-member flow. Lead golfer added them during registration;
 * they hit this page to add their outfit photo so the matcher can find their
 * clips. No payment, no contact info — that's already on file.
 */
export default function Invite() {
  const { galleryToken } = useParams();
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(null);

  useEffect(() => {
    api.inviteInfo(galleryToken).then(setInfo).catch((e) => setError(e.message));
  }, [galleryToken]);

  function onPicked(f) {
    if (!f) return;
    setFile(f);
    const reader = new FileReader();
    reader.onload = () => setPreview(reader.result);
    reader.readAsDataURL(f);
    setCameraOpen(false);
  }

  async function submit() {
    if (!file) return;
    setSubmitting(true);
    try {
      const res = await api.inviteSelfie(galleryToken, file);
      setDone(res);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (error && !info) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card"><h2>Invite link not found</h2><p className="small muted">{error}</p></div>
      </div>
    );
  }
  if (!info) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card"><div className="shimmer" style={{ height: 100 }} /></div>
      </div>
    );
  }

  if (done) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card center" style={{ padding: 36 }}>
          <div
            style={{
              width: 72, height: 72, borderRadius: "50%",
              background: "var(--primary-soft)", color: "var(--emerald-700)",
              display: "grid", placeItems: "center", margin: "0 auto 16px",
            }}
          >
            <Icon name="check" size={32} />
          </div>
          <h2>You're all set, {info.name}.</h2>
          <p className="muted">We'll email you the moment your clips are ready.</p>
          {done.gallery_url && (
            <p className="small" style={{ marginTop: 16 }}>
              Bookmark your gallery: <a href={done.gallery_url}>{done.gallery_url}</a>
            </p>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <Brand />

      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="flag" size={14} /> {info.course.name}</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3.2vw, 2.1rem)" }}>Welcome, {info.name}.</h1>
        <p style={{ fontSize: "0.95rem" }}>
          You've been added to a round at {info.course.name}
          {info.tee_time && ` — tee time ${new Date(info.tee_time).toLocaleString([], {
            month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
          })}`}.
          One quick step: snap an outfit photo so we can match your shots.
        </p>
      </div>

      <form className="card" onSubmit={(e) => { e.preventDefault(); submit(); }}>
        <div className="field">
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Icon name="camera" size={16} /> Outfit photo
          </label>
          <p className="hint">Head-to-toe, full outfit visible. Hand the phone to a friend or use a mirror.</p>
          {cameraOpen ? (
            <SelfieCamera onCapture={onPicked} onCancel={() => setCameraOpen(false)} />
          ) : preview ? (
            <div className="selfie-preview">
              <img src={preview} alt="outfit preview" />
              <div style={{ flex: 1 }}>
                <div className="ok-text inline"><Icon name="check" size={16} /> Looks great</div>
                <div className="small muted">We'll match clips to this outfit.</div>
              </div>
              <button type="button" className="secondary small" onClick={() => setCameraOpen(true)}>
                <Icon name="retake" size={14} /> Retake
              </button>
            </div>
          ) : (
            <button type="button" onClick={() => setCameraOpen(true)}>
              <Icon name="camera" size={16} /> Open camera
            </button>
          )}
        </div>

        {error && <p className="err-text small">{error}</p>}
        <button disabled={submitting || !file}>
          {submitting ? "Submitting…" : "Submit photo"}
        </button>
      </form>
    </div>
  );
}
