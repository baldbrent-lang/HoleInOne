import { useEffect, useMemo, useState } from "react";
import { useLocation, useNavigate, useParams } from "react-router-dom";
import { Elements, PaymentElement, useElements, useStripe } from "@stripe/react-stripe-js";
import { loadStripe } from "@stripe/stripe-js";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";

/**
 * Stripe checkout for a freshly-created participant. PaymentElement
 * automatically surfaces Apple Pay (Safari/iOS), Google Pay (Chrome/Android),
 * Link, and card based on the visitor's device + payment methods.
 *
 * Reached after Register.jsx posts the form when STRIPE is configured.
 * Routed there with route state { client_secret, gallery_url, total }.
 */
export default function Pay() {
  const { state } = useLocation();
  const { participantId } = useParams();
  const nav = useNavigate();
  const [config, setConfig] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!state?.client_secret) {
      // No state — likely deep-link or refresh. Bounce home.
      nav("/", { replace: true });
      return;
    }
    api.stripeConfig().then(setConfig).catch((e) => setError(e.message));
  }, [state, nav]);

  const stripePromise = useMemo(() => {
    if (!config?.publishable_key) return null;
    return loadStripe(config.publishable_key);
  }, [config]);

  if (error) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card err-text">{error}</div>
      </div>
    );
  }
  if (!state?.client_secret || !config) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card"><div className="shimmer" style={{ height: 200 }} /></div>
      </div>
    );
  }
  if (!stripePromise) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card">
          <h2>Payments not yet configured</h2>
          <p className="small muted">
            The site operator hasn't added a Stripe publishable key. Set
            <code> STRIPE_PUBLISHABLE_KEY </code> + <code>STRIPE_SECRET_KEY</code> in Secrets and restart.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <Brand />
      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="lock" size={14} /> Secure checkout</span>
        <h1 style={{ fontSize: "clamp(1.5rem, 3vw, 2rem)" }}>
          Confirm your ${(state.total ?? config.price_cents / 100).toFixed(2)} payment
        </h1>
        <p>Apple Pay, Google Pay, Link, or card — pick whatever you have set up.</p>
      </div>

      <Elements
        stripe={stripePromise}
        options={{
          clientSecret: state.client_secret,
          appearance: { theme: "stripe", variables: { colorPrimary: "#059669" } },
        }}
      >
        <CheckoutForm participantId={participantId} galleryUrl={state.gallery_url} />
      </Elements>
    </div>
  );
}

function CheckoutForm({ participantId, galleryUrl }) {
  const stripe = useStripe();
  const elements = useElements();
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (!stripe || !elements) return;
    setSubmitting(true);
    setError(null);

    const returnUrl = `${window.location.origin}/confirm/${participantId}`;
    const { error: stripeError } = await stripe.confirmPayment({
      elements,
      confirmParams: { return_url: returnUrl },
      // If 3DS / wallet redirect not required, stay on page; we route ourselves on success.
      redirect: "if_required",
    });

    if (stripeError) {
      setError(stripeError.message || "Payment failed.");
      setSubmitting(false);
      return;
    }

    // No redirect happened → wallet confirmed inline. Send the user to /confirm.
    window.location.href = `/confirm/${participantId}` + (galleryUrl ? `?g=${encodeURIComponent(galleryUrl)}` : "");
  }

  return (
    <form className="card" onSubmit={submit}>
      <PaymentElement />
      {error && <p className="err-text small" style={{ marginTop: 12 }}>{error}</p>}
      <button disabled={!stripe || submitting} style={{ marginTop: 16 }}>
        {submitting ? "Processing…" : "Pay now"}
      </button>
      <p className="small muted center" style={{ marginTop: 12 }}>
        <Icon name="lock" size={12} /> Secured by Stripe.
      </p>
    </form>
  );
}
