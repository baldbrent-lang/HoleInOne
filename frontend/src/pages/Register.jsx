import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import SelfieField from "../components/SelfieField.jsx";
import useAuth from "../hooks/useAuth.js";

const EMPTY_MEMBER = { name: "", email: "", mobile: "", selfieFile: null, selfiePreview: null };

export default function Register() {
  const { courseToken } = useParams();
  const nav = useNavigate();
  const { user } = useAuth();
  const [course, setCourse] = useState(null);
  const [teeTimes, setTeeTimes] = useState([]);
  const [teeTimeId, setTeeTimeId] = useState("");
  const [name, setName] = useState(user?.name || "");
  const [mobile, setMobile] = useState("");
  const [email, setEmail] = useState(user?.email || "");
  const [groupSize, setGroupSize] = useState(1);
  const [groupMembers, setGroupMembers] = useState([
    { ...EMPTY_MEMBER }, { ...EMPTY_MEMBER }, { ...EMPTY_MEMBER },
  ]);
  const [selfieFile, setSelfieFile] = useState(null);
  const [selfiePreview, setSelfiePreview] = useState(null);
  // Single openCameraId across the whole form so only one device camera is mounted at a time.
  const [openCameraId, setOpenCameraId] = useState(null);
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

  function readPreview(file, set) {
    const reader = new FileReader();
    reader.onload = () => set(reader.result);
    reader.readAsDataURL(file);
  }

  function onLeadSelfie(file) {
    if (!file?.type?.startsWith("image/")) { setError("That doesn't look like a photo."); return; }
    setError(null);
    setSelfieFile(file);
    readPreview(file, setSelfiePreview);
  }

  function onMemberSelfie(idx, file) {
    if (!file?.type?.startsWith("image/")) { setError("That doesn't look like a photo."); return; }
    setError(null);
    const next = [...groupMembers];
    next[idx] = { ...next[idx], selfieFile: file };
    setGroupMembers(next);
    readPreview(file, (preview) => {
      const after = [...next];
      after[idx] = { ...after[idx], selfiePreview: preview };
      setGroupMembers(after);
    });
  }

  const memberCount = Math.max(0, Number(groupSize) - 1);
  const activeMembers = groupMembers.slice(0, memberCount);
  const allMembersComplete = activeMembers.every(
    (m) => m.name.trim() && (m.email.trim() || m.mobile.trim()) && m.selfieFile,
  );

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (!mobile && !email) throw new Error("Enter a mobile number or an email so we can text you your clips.");
      if (!selfieFile) throw new Error("Take your outfit photo so we can match your shots.");
      if (memberCount > 0 && !allMembersComplete) {
        throw new Error("Each player needs a name, contact, and an outfit photo.");
      }

      const cleanMembers = activeMembers.map((m) => ({
        name: m.name.trim(),
        email: m.email.trim(),
        mobile: m.mobile.trim(),
      }));

      const fd = new FormData();
      fd.append("course_token", courseToken);
      fd.append("tee_time_id", String(teeTimeId));
      fd.append("name", name);
      fd.append("mobile", mobile);
      fd.append("email", email);
      fd.append("group_size", String(groupSize));
      fd.append("group_members", JSON.stringify(cleanMembers));
      fd.append("selfie", selfieFile, selfieFile.name || "selfie.jpg");
      activeMembers.forEach((m, i) => {
        if (m.selfieFile) {
          // member_2_selfie, member_3_selfie, member_4_selfie
          fd.append(`member_${i + 2}_selfie`, m.selfieFile, m.selfieFile.name || `member_${i + 2}.jpg`);
        }
      });

      const res = await api.register(fd);
      if (res.client_secret && !res.paid) {
        nav(`/pay/${res.participant_id}`, {
          state: {
            client_secret: res.client_secret,
            gallery_url: res.gallery_url,
            total: 20 * Number(groupSize),
          },
        });
      } else {
        nav(`/confirm/${res.participant_id}`, { state: res });
      }
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
        {!user && (
          <p className="small" style={{ marginTop: 12, color: "rgba(255,255,255,0.85)" }}>
            Have an account? <Link to={`/login?next=/r/${courseToken}`} style={{ color: "white", textDecoration: "underline" }}>Log in</Link> to save this round to your history.
          </p>
        )}
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
          {Number(groupSize) > 1 && (
            <div className="hint small muted">
              You'll pay <b>${20 * Number(groupSize)}</b> total ($20 × {groupSize}). Each player needs their own outfit photo for personal clips.
            </div>
          )}
        </div>

        <div className="field">
          <label style={{ display: "flex", alignItems: "center", gap: 6 }}>
            <Icon name="camera" size={16} /> Your outfit photo
          </label>
          <p className="hint">
            Hand your phone to a friend or use a mirror — head to toe, full
            outfit visible. We use this to pick your shots out of the footage.
          </p>
          <SelfieField
            id="lead"
            openCameraId={openCameraId}
            setOpenCameraId={setOpenCameraId}
            preview={selfiePreview}
            onPicked={onLeadSelfie}
          />
        </div>

        {memberCount > 0 && (
          <div className="card" style={{ background: "var(--primary-soft)", border: "1px solid var(--emerald-200)", margin: "0 0 16px" }}>
            <h4 style={{ marginBottom: 6 }}>Add your group</h4>
            <p className="hint">
              Each player needs an outfit photo so we can build them their own
              personal gallery. Hand the phone around — capture each one before
              you tee off.
            </p>
            {activeMembers.map((m, i) => (
              <div
                key={i}
                style={{
                  marginTop: 12, paddingTop: 12,
                  borderTop: i === 0 ? "none" : "1px solid var(--emerald-200)",
                }}
              >
                <div className="tiny upper muted" style={{ marginBottom: 6 }}>
                  Player {i + 2}
                  {m.selfieFile && <span className="ok-text" style={{ marginLeft: 8 }}>· photo captured</span>}
                </div>
                <div className="field" style={{ marginBottom: 8 }}>
                  <input
                    placeholder="Name"
                    value={m.name}
                    onChange={(e) => {
                      const next = [...groupMembers];
                      next[i] = { ...next[i], name: e.target.value };
                      setGroupMembers(next);
                    }}
                  />
                </div>
                <div className="row" style={{ marginBottom: 10 }}>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <input
                      type="tel"
                      placeholder="Mobile"
                      value={m.mobile}
                      onChange={(e) => {
                        const next = [...groupMembers];
                        next[i] = { ...next[i], mobile: e.target.value };
                        setGroupMembers(next);
                      }}
                    />
                  </div>
                  <div className="field" style={{ marginBottom: 0 }}>
                    <input
                      type="email"
                      placeholder="Email"
                      value={m.email}
                      onChange={(e) => {
                        const next = [...groupMembers];
                        next[i] = { ...next[i], email: e.target.value };
                        setGroupMembers(next);
                      }}
                    />
                  </div>
                </div>
                <SelfieField
                  id={`m${i}`}
                  openCameraId={openCameraId}
                  setOpenCameraId={setOpenCameraId}
                  preview={m.selfiePreview}
                  onPicked={(file) => onMemberSelfie(i, file)}
                />
              </div>
            ))}
          </div>
        )}

        <div className="divider" />
        <div className="inline" style={{ justifyContent: "space-between", marginBottom: 14, width: "100%" }}>
          <span className="small muted"><Icon name="lock" size={14} /> Secure Stripe checkout</span>
          <span className="inline" style={{ fontWeight: 600 }}>${20 * Number(groupSize)}</span>
        </div>

        {error && <p className="err-text small">{error}</p>}
        <button
          disabled={
            submitting || !teeTimeId || !name || !selfieFile ||
            (memberCount > 0 && !allMembersComplete)
          }
        >
          {submitting ? "Processing…" : `Pay $${20 * Number(groupSize)} and register`}
        </button>
      </form>
    </div>
  );
}
