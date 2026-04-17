import { useLocation, useParams } from "react-router-dom";
import { Brand, Icon } from "../components/Brand.jsx";

export default function Confirmation() {
  const { state } = useLocation();
  const { participantId } = useParams();

  return (
    <div className="wrap">
      <Brand />

      <div className="card center" style={{ padding: "36px 20px" }}>
        <div
          style={{
            width: 72, height: 72, borderRadius: "50%",
            background: "var(--primary-soft)", color: "var(--emerald-700)",
            display: "grid", placeItems: "center", margin: "0 auto 16px",
            boxShadow: "inset 0 0 0 6px var(--emerald-100)",
          }}
        >
          <Icon name="check" size={32} />
        </div>
        <h2>You're registered!</h2>
        <p className="muted">We'll text and email you the moment your videos are ready.</p>

        {state?.gallery_url && (
          <div
            className="card"
            style={{ background: "var(--primary-soft)", border: "1px solid var(--emerald-200)", marginTop: 18, textAlign: "left" }}
          >
            <div className="small upper muted" style={{ marginBottom: 6 }}>Your gallery link</div>
            <a href={state.gallery_url} className="small" style={{ wordBreak: "break-all" }}>
              {state.gallery_url}
            </a>
          </div>
        )}

        <div className="small muted" style={{ marginTop: 16 }}>
          {state?.paid ? (
            <span className="inline"><Icon name="check" size={14} /> Payment confirmed</span>
          ) : (
            `Confirmation ref #${participantId}`
          )}
        </div>
      </div>

      <div className="card">
        <h3>While you wait</h3>
        <div className="feature-row">
          <div className="icon"><Icon name="flag" /></div>
          <div>
            <h4>Play your round</h4>
            <p>Don't change outfits mid-round — we match by what you're wearing now.</p>
          </div>
        </div>
        <div className="feature-row">
          <div className="icon"><Icon name="sparkle" /></div>
          <div>
            <h4>Tee off with confidence</h4>
            <p>Every par-3 tee shot gets a tracer overlay and auto-stats.</p>
          </div>
        </div>
        <div className="feature-row">
          <div className="icon"><Icon name="share" /></div>
          <div>
            <h4>Post-round delivery</h4>
            <p>A link lands in your texts — no app, no login.</p>
          </div>
        </div>
      </div>
    </div>
  );
}
