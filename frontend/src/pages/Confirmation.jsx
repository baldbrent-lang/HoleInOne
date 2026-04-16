import { useLocation, useParams } from "react-router-dom";

export default function Confirmation() {
  const { state } = useLocation();
  const { participantId } = useParams();

  return (
    <div className="wrap">
      <div className="brand">
        <div className="dot" />
        <h1>Par One</h1>
      </div>
      <div className="card">
        <h2>You're registered! 🏌</h2>
        <p>
          We'll text/email you when your videos are ready. In the meantime,
          bookmark your personal gallery link:
        </p>
        {state?.gallery_url ? (
          <p>
            <a href={state.gallery_url}>{state.gallery_url}</a>
          </p>
        ) : (
          <p className="muted small">Confirmation for participant #{participantId}</p>
        )}
        <p className="muted small">
          {state?.paid
            ? "Payment confirmed."
            : "Payment pending — finish on the Stripe page if prompted."}
        </p>
      </div>
    </div>
  );
}
