import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";
import useAuth from "../hooks/useAuth.js";

export default function Signup() {
  const { adopt } = useAuth();
  const nav = useNavigate();
  const [email, setEmail] = useState("");
  const [name, setName] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState(null);
  const [submitting, setSubmitting] = useState(false);

  async function submit(e) {
    e.preventDefault();
    setSubmitting(true);
    setError(null);
    try {
      if (password.length < 8) throw new Error("Password must be at least 8 characters.");
      const res = await api.signup({ email, password, name: name || null });
      adopt(res.token, res.user);
      nav("/me", { replace: true });
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
        <h2 style={{ marginBottom: 4 }}>Create your account</h2>
        <p className="small muted" style={{ marginBottom: 16 }}>
          Save your rounds and access past clips. Already have an account?{" "}
          <Link to="/login">Log in</Link>.
        </p>
        <div className="field">
          <label>Name <span className="muted small">(optional)</span></label>
          <input value={name} onChange={(e) => setName(e.target.value)} placeholder="Ben Baldwin" autoFocus />
        </div>
        <div className="field">
          <label>Email</label>
          <input
            type="email"
            required
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            autoComplete="email"
          />
          <div className="hint small muted">
            If you've already registered for a round with this email, we'll link your past rounds.
          </div>
        </div>
        <div className="field">
          <label>Password</label>
          <input
            type="password"
            required
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            placeholder="At least 8 characters"
            autoComplete="new-password"
            minLength={8}
          />
        </div>
        {error && <p className="err-text small">{error}</p>}
        <button disabled={submitting || !email || password.length < 8}>
          {submitting ? "Creating account…" : "Create account"}
        </button>
      </form>
    </div>
  );
}
