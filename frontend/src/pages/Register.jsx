import { useEffect, useState } from "react";
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
  const [playingOrder, setPlayingOrder] = useState(1);
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

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (!mobile && !email) {
        throw new Error("Enter a mobile number or an email.");
      }
      const res = await api.register({
        course_token: courseToken,
        tee_time_id: Number(teeTimeId),
        name,
        mobile: mobile || null,
        email: email || null,
        playing_order: Number(playingOrder),
        group_size: Number(groupSize),
      });
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
            <input
              type="tel"
              value={mobile}
              onChange={(e) => setMobile(e.target.value)}
              placeholder="+1 555 123 4567"
            />
          </div>
          <div className="field">
            <label>Email</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              placeholder="you@email.com"
            />
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

        <div className="row">
          <div className="field">
            <label>Group size</label>
            <select value={groupSize} onChange={(e) => setGroupSize(e.target.value)}>
              {[1, 2, 3, 4].map((n) => (
                <option key={n} value={n}>{n}</option>
              ))}
            </select>
          </div>
          <div className="field">
            <label>You're hitting</label>
            <select value={playingOrder} onChange={(e) => setPlayingOrder(e.target.value)}>
              {[1, 2, 3, 4].slice(0, Number(groupSize)).map((n) => (
                <option key={n} value={n}>{ordinal(n)} off the tee</option>
              ))}
            </select>
          </div>
        </div>

        <p className="muted small" style={{ marginBottom: 12 }}>
          $20 registration — charged now. You'll get a text/email when your
          videos are ready.
        </p>
        {error && <p style={{ color: "var(--danger)" }} className="small">{error}</p>}
        <button disabled={submitting || !teeTimeId || !name}>
          {submitting ? "Processing…" : "Pay $20 and register"}
        </button>
      </form>
    </div>
  );
}

function ordinal(n) {
  return ["1st", "2nd", "3rd", "4th"][n - 1] ?? `${n}th`;
}
