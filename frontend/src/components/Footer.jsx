import { Link } from "react-router-dom";

export default function Footer() {
  return (
    <footer
      style={{
        marginTop: 40,
        paddingTop: 20,
        borderTop: "1px solid var(--border)",
        color: "var(--ink-soft)",
        fontSize: "0.82rem",
      }}
    >
      <div style={{ display: "flex", flexWrap: "wrap", gap: 18, justifyContent: "center" }}>
        <Link to="/legal/faq">FAQ</Link>
        <Link to="/legal/terms">Terms of Service</Link>
        <Link to="/legal/privacy">Privacy Policy</Link>
        <Link to="/legal/rules">Contest Rules</Link>
        <Link to="/operator/login">Course operator portal</Link>
        <a href="mailto:hello@golfreelz.com">Contact</a>
      </div>
      <div style={{ textAlign: "center", marginTop: 12 }}>
        © {new Date().getFullYear()} GolfReelz. $20 per round on the par-3 video
        system · $10,000 <Link to="/legal/rules">hole-in-one contest</Link>.
      </div>
    </footer>
  );
}
