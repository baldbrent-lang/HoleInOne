import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";
import useAuth from "../hooks/useAuth.js";

export default function Login() {
  const { adopt } = useAuth();
  const nav = useNavigate();
  const loc = useLocation();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      const res = await api.login({ email, password });
      adopt(res.token, res.user);
      const next = new URLSearchParams(loc.search).get("next") || "/me";
      nav(next, { replace: true });
    } catch (e) {
      setError(e.message.replace(/^\d+:\s*/, ""));
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="wrap">
      <Brand hideAccount />
      <form className="card" onSubmit={submit} style={{ maxWidth: 420, margin: "20px auto 0" }}>
        <h2 style={{ marginBottom: 4 }}>Log in</h2>
        <p className="small muted" style={{ marginBottom: 16 }}>
          Welcome back. New here?{" "}
          <Link to="/signup">Create an account</Link>.
        </p>

        <div className="field">
          <label>Email</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
            autoFocus
          />
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
        </div>
        {error && <p className="err-text small">{error}</p>}
        <button disabled={submitting || !email || !password}>
          {submitting ? "Signing in…" : "Log in"}
        </button>
      </form>
    </div>
  );
}
