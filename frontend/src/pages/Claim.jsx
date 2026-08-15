import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";

/**
 * Prize claim for a confirmed hole-in-one, linked from the confirmation
 * email.
 *
 * Contact fields arrive pre-filled from what we already hold — a winner
 * should not have to retype their own email to collect a prize. No bank
 * or card details are asked for here, by design; the payout is arranged
 * directly once the claim is in.
 */
export default function Claim() {
  const { galleryToken } = useParams();
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [email, setEmail] = useState("");
  const [mobile, setMobile] = useState("");
  const [address, setAddress] = useState("");
  const [note, setNote] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    api
      .claimContext(galleryToken)
      .then((r) => {
        setInfo(r);
        setEmail(r.email || "");
        setMobile(r.mobile || "");
      })
      .catch((e) => setError(e.message));
  }, [galleryToken]);

  async function submit() {
    setSubmitting(true);
    try {
      const r = await api.submitClaim(galleryToken, {
        email,
        mobile,
        mailing_address: address,
        note,
      });
      setInfo(r);
      setDone(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setSubmitting(false);
    }
  }

  if (error && !info) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card">
          <h2>Claim link not found</h2>
          <p className="small muted">{error}</p>
        </div>
      </div>
    );
  }

  if (!info) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card">
          <div className="shimmer" style={{ height: 120 }} />
        </div>
      </div>
    );
  }

  // No confirmed ace on this account. Says so plainly rather than
  // showing a form that would refuse the submission anyway.
  if (!info.eligible) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card">
          <h2>Nothing to claim yet</h2>
          <p className="small muted">
            We don't have a confirmed hole-in-one on your account. If you
            believe that's wrong, reply to the email we sent you and we'll
            take another look.
          </p>
          <Link className="btn secondary" to={`/g/${galleryToken}`}>
            Back to your gallery
          </Link>
        </div>
      </div>
    );
  }

  if (done) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card" style={{ textAlign: "center" }}>
          <h2>Claim received, {info.name}.</h2>
          <p className="small muted">
            We'll be in touch at {info.email || info.mobile || "your contact details"}{" "}
            to arrange your prize. Congratulations again.
          </p>
          <Link className="btn secondary" to={`/g/${galleryToken}`}>
            Back to your gallery
          </Link>
        </div>
      </div>
    );
  }

  const where = [
    info.hole_number ? `hole ${info.hole_number}` : null,
    info.course_name,
  ]
    .filter(Boolean)
    .join(" at ");

  return (
    <div className="wrap">
      <Brand />
      <div className="card">
        <h2>Claim your {info.prize_label || ""} prize</h2>
        <p className="small muted">
          Congratulations, {info.name} — your hole-in-one
          {where ? ` on ${where}` : ""} is confirmed
          {info.prize_label ? `, and ${info.prize_label} is waiting for you` : ""}.
          Confirm how to reach you and we'll arrange everything.
        </p>

        {info.already_claimed && (
          <p className="small muted">
            You've already filed this claim
            {info.status ? ` (${info.status})` : ""}. You can update your
            details below.
          </p>
        )}

        <div className="field">
          <label>Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            placeholder="you@example.com"
          />
        </div>

        <div className="field">
          <label>Mobile</label>
          <input
            type="tel"
            value={mobile}
            onChange={(e) => setMobile(e.target.value)}
            placeholder="(555) 555-5555"
          />
        </div>

        <div className="field">
          <label>Mailing address (only if your prize needs posting)</label>
          <textarea
            rows={3}
            value={address}
            maxLength={2000}
            onChange={(e) => setAddress(e.target.value)}
            placeholder="Street, city, state, ZIP"
          />
        </div>

        <div className="field">
          <label>Anything we should know? (optional)</label>
          <textarea
            rows={3}
            value={note}
            maxLength={2000}
            onChange={(e) => setNote(e.target.value)}
          />
        </div>

        <p className="small muted">
          We'll never ask for bank or card details on this page. If you get
          a message that does, it isn't from us.
        </p>

        {error && <p className="small err-text">{error}</p>}

        <button
          disabled={submitting || (!email && !mobile)}
          onClick={submit}
          style={{ marginTop: 8 }}
        >
          {submitting
            ? "Sending…"
            : info.already_claimed
              ? "Update my claim"
              : "Submit claim"}
        </button>
        {!email && !mobile && (
          <p className="small muted" style={{ marginTop: 8 }}>
            Give us an email or a mobile number so we can reach you.
          </p>
        )}
      </div>
    </div>
  );
}
