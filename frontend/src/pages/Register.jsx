import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import SelfieCamera from "../components/SelfieCamera.jsx";

export default function Register() {
  const { courseToken } = useParams();
  const nav = useNavigate();
  const [course, setCourse] = useState(null);
  const [teeTimes, setTeeTimes] = useState([]);
  const [teeTimeId, setTeeTimeId] = useState("");
  const [name, setName] = useState("");
  const [mobile, setMobile] = useState("");
  const [email, setEmail] = useState("");
  const [groupSize, setGroupSize] = useState(4);
  const [selfieFile, setSelfieFile] = useState(null);
  const [selfiePreview, setSelfiePreview] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const [c, tt] = await Promise.all([
          api.courseByToken(courseToken),
          api.teeTimes(courseToken),
        ]);
        setCourse(c);
        setTeeTimes(tt);
      } catch (e) {
        setError(e.message);
      }
    })();
  }, [courseToken]);

  function onSelfiePicked(file) {
    if (!file) return;
    if (!file.type.startsWith("image/")) {
      setError("That doesn't look like a photo. Try again?");
      return;
    }
    setError(null);
    setSelfieFile(file);
    const reader = new FileReader();
    reader.onload = () => setSelfiePreview(reader.result);
    reader.readAsDataURL(file);
    setCameraOpen(false);
  }

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (!mobile && !email) throw new Error("Enter a mobile number or an email so we can text you your clips.");
      if (!selfieFile) throw new Error("Please take an outfit photo so we can match your shots.");

      const fd = new FormData();
      fd.append("course_token", courseToken);
      fd.append("tee_time_id", String(teeTimeId));
      fd.append("name", name);
      fd.append("mobile", mobile);
      fd.append("email", email);
      fd.append("group_size", String(groupSize));
      fd.append("selfie", selfieFile, selfieFile.name || "selfie.jpg");

      const res = await api.register(fd);
      nav(`/confirm/${res.participant_id}`, { state: res });
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (error && !course) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card">
          <h2>Couldn't load this course</h2>
          <p className="muted small">{error}</p>
        </div>
      </div>
    );
  }

  if (!course) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card"><div className="shimmer" style={{ height: 140 }} /></div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <Brand />

      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="flag" size={14} /> {course.name}</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3.2vw, 2.1rem)" }}>Let's get your shots on tape.</h1>
        <p style={{ fontSize: "0.95rem" }}>{course.location}</p>
      </div>

      {course.livestream_url && (
        <a href={course.livestream_url} target="_blank" rel="noreferrer" className="live-banner" style={{ textDecoration: "none" }}>
          <span className="dot" />
          <div>
            <div className="label">Live now</div>
            <div className="small">Watch the course livestream while you wait.</div>
          </div>
          <span>Watch →</span>
        </a>
      )}

      <form className="card" onSubmit={submit}>
        <div className="field">
          <label>Your name</label>
          <input required value={name} onChange={(e) => setName(e.target.value)} placeholder="Alex Morgan" />
        </div>
        <div className="row">
          <div className="field">
            <label>Mobile</label>
            <input type="tel" value={mobile} onChange={(e) => setMobile(e.target.value)} placeholder="+1 555 123 4567" />
          </div>
          <div className="field">
            <label>Email</label>
            <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="you@email.com" />
          </div>
        </div>
        <div className="hint small muted" style={{ margin: "-8px 0 14px" }}>
          We'll text or email when your gallery's ready. At least one required.
        </div>

        <div className="field">
          <label>Tee time</label>
          <select required value={teeTimeId} onChange={(e) => setTeeTimeId(e.target.value)}>
            <option value="">Pick your tee time…</option>
            {teeTimes.map((tt) => {
              const full = tt.spots_taken >= tt.max_players;
              const label = new Date(tt.starts_at).toLocaleString([], {
                month: "short", day: "numeric", hour: "numeric", minute: "2-digit",
              });
              return (
                <option key={tt.id} value={tt.id} disabled={full}>
                  {label} {full ? "(full)" : `· ${tt.spots_taken}/${tt.max_players}`}
                </option>
              );
            })}
          </select>
        </div>

        <div className="field">
          <label>Group size</label>
          <select value={groupSize} onChange={(e) => setGroupSize(e.target.value)}>
            {[1, 2, 3, 4].map((n) => (<option key={n} value={n}>{n}</option>))}
          </select>
        </div>

        <div className="field">
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Icon name="camera" size={16} /> Outfit photo
          </label>
          <p className="hint">
            Hand your phone to a friend or use a mirror — head to toe, full outfit visible.
            We use this to pick your shots out of the footage.
          </p>
          {cameraOpen ? (
            <SelfieCamera onCapture={onSelfiePicked} onCancel={() => setCameraOpen(false)} />
          ) : selfiePreview ? (
            <div className="selfie-preview">
              <img src={selfiePreview} alt="outfit preview" />
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

        <div className="divider" />
        <div className="inline" style={{ justifyContent: "space-between", marginBottom: 14, width: "100%" }}>
          <span className="small muted"><Icon name="lock" size={14} /> Secure Stripe checkout</span>
          <span className="inline" style={{ fontWeight: 600 }}>$20</span>
        </div>

        {error && <p className="err-text small">{error}</p>}
        <button disabled={submitting || !teeTimeId || !name || !selfieFile}>
          {submitting ? "Processing…" : "Pay $20 and register"}
        </button>
      </form>
    </div>
  );
}
