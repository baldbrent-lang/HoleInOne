import { useEffect, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../api.js";

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
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);
  const fileInputRef = useRef();

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
      setError("Selfie must be an image.");
      return;
    }
    setError(null);
    setSelfieFile(file);
    const reader = new FileReader();
    reader.onload = () => setSelfiePreview(reader.result);
    reader.readAsDataURL(file);
  }

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (!mobile && !email) throw new Error("Enter a mobile number or an email.");
      if (!selfieFile) throw new Error("Please take a selfie so we can match your shots.");

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
        <div className="card">
          <h1>Couldn't load course</h1>
          <p className="muted">{error}</p>
        </div>
      </div>
    );
  }

  if (!course) {
    return (
      <div className="wrap">
        <div className="card muted">Loading…</div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <div className="brand">
        <div className="dot" />
        <h1>Par One</h1>
      </div>
      <div className="card">
        <h2>{course.name}</h2>
        <p className="muted small">{course.location}</p>
      </div>

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

        <div className="field">
          <label>Tee time</label>
          <select required value={teeTimeId} onChange={(e) => setTeeTimeId(e.target.value)}>
            <option value="">Select a tee time…</option>
            {teeTimes.map((tt) => {
              const full = tt.spots_taken >= tt.max_players;
              const label = new Date(tt.starts_at).toLocaleString([], {
                month: "short",
                day: "numeric",
                hour: "numeric",
                minute: "2-digit",
              });
              return (
                <option key={tt.id} value={tt.id} disabled={full}>
                  {label} {full ? "(full)" : `(${tt.spots_taken}/${tt.max_players})`}
                </option>
              );
            })}
          </select>
        </div>

        <div className="field">
          <label>Group size</label>
          <select value={groupSize} onChange={(e) => setGroupSize(e.target.value)}>
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>{n}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>Selfie (so we can match your shots)</label>
          <p className="muted small" style={{ margin: "0 0 8px" }}>
            Stand back, full outfit visible — hat, shirt, pants, shoes. We use this
            to identify you in each clip.
          </p>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*"
            capture="user"
            style={{ display: "none" }}
            onChange={(e) => onSelfiePicked(e.target.files?.[0])}
          />
          {selfiePreview ? (
            <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
              <img
                src={selfiePreview}
                alt="selfie preview"
                style={{ width: 96, height: 96, objectFit: "cover", borderRadius: 12, border: "1px solid #26543a" }}
              />
              <button type="button" className="secondary" onClick={() => fileInputRef.current?.click()}>
                Retake
              </button>
            </div>
          ) : (
            <button type="button" onClick={() => fileInputRef.current?.click()}>
              Take selfie
            </button>
          )}
        </div>

        <p className="muted small" style={{ marginBottom: 12 }}>
          $20 registration — charged now. You'll get a text/email when your videos are ready.
        </p>
        {error && <p style={{ color: "var(--danger)" }} className="small">{error}</p>}
        <button disabled={submitting || !teeTimeId || !name || !selfieFile}>
          {submitting ? "Processing…" : "Pay $20 and register"}
        </button>
      </form>
    </div>
  );
}
