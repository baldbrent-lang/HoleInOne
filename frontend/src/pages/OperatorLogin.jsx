import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, setOperatorToken } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

/**
 * Per-course operator login. Each course's pro shop gets a password
 * (set by the system admin) and a scoped dashboard that only shows
 * their course's stats.
 */
export default function OperatorLogin() {
  const nav = useNavigate();
  const [courses, setCourses] = useState([]);
  const [courseToken, setCourseToken] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  useEffect(() => {
    api.listPublicCourses().then((cs) => setCourses(cs)).catch(() => setCourses([]));
  }, []);

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await api.operatorLogin({ course_token: courseToken, password });
      setOperatorToken(res.token);
      nav("/operator", { replace: true });
    } catch (e) {
      setError(e.message.replace(/^\d+:\s*/, ""));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="wrap">
      <Brand subtitle="Course operator portal" />
      <form className="card" onSubmit={submit} style={{ maxWidth: 460, margin: "20px auto 0" }}>
        <h2 style={{ marginBottom: 4 }}>
          <Icon name="flag" size={18} /> Operator sign-in
        </h2>
        <p className="small muted" style={{ marginBottom: 16 }}>
          Pro shops + course managers only. Your read-only dashboard for daily
          activity, revenue, and recent clips.
        </p>

        <div className="field">
          <label>Course</label>
          <select required value={courseToken} onChange={(e) => setCourseToken(e.target.value)} autoFocus>
            <option value="">Select your course…</option>
            {courses.map((c) => (
              <option key={c.id} value={c.qr_token}>{c.name}</option>
            ))}
          </select>
        </div>

        <div className="field">
          <label>Password</label>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            autoComplete="current-password"
          />
          <div className="hint small muted">
            Don't have a password? Email <a href="mailto:hello@golfreelz.com">hello@golfreelz.com</a>.
          </div>
        </div>

        {error && <p className="err-text small">{error}</p>}
        <button disabled={submitting || !courseToken || !password}>
          {submitting ? "Signing in…" : "Sign in"}
        </button>
        <p className="small muted center" style={{ marginTop: 12 }}>
          Looking for the system console? <Link to="/admin">Admin login →</Link>
        </p>
      </form>
    </div>
  );
}
