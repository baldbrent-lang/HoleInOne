import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api.js";
import { Brand } from "../components/Brand.jsx";

/**
 * Leave-a-review page, linked from the thank-you email.
 *
 * The gallery token identifies the golfer, so there is no login and no
 * "which round was this" — they land on a form that already knows their
 * name and course. Rating is the only required field: a golfer who taps
 * five stars and leaves is a review we would rather have than a blank
 * one they abandoned at a required textarea.
 */
const STARS = [1, 2, 3, 4, 5];

export default function Review() {
  const { galleryToken } = useParams();
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);
  const [rating, setRating] = useState(0);
  const [hover, setHover] = useState(0);
  const [comment, setComment] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [done, setDone] = useState(false);

  useEffect(() => {
    api
      .reviewContext(galleryToken)
      .then((r) => {
        setInfo(r);
        // Pre-fill from an earlier review so coming back to add detail
        // means editing what they wrote, not retyping it.
        if (r.already_reviewed) {
          setRating(r.rating || 0);
          setComment(r.comment || "");
        }
      })
      .catch((e) => setError(e.message));
  }, [galleryToken]);

  async function submit() {
    if (!rating) return;
    setSubmitting(true);
    try {
      await api.submitReview(galleryToken, { rating, comment });
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
          <h2>Review link not found</h2>
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
          <div className="shimmer" style={{ height: 100 }} />
        </div>
      </div>
    );
  }

  if (done) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card" style={{ textAlign: "center" }}>
          <div style={{ fontSize: 40, lineHeight: 1 }}>★</div>
          <h2 style={{ marginTop: 12 }}>Thank you, {info.name}.</h2>
          <p className="small muted">
            We read every one of these. It genuinely helps.
          </p>
          <Link className="btn secondary" to={`/g/${galleryToken}`}>
            Back to your gallery
          </Link>
        </div>
      </div>
    );
  }

  const at = info.course_name ? ` at ${info.course_name}` : "";
  const shown = hover || rating;

  return (
    <div className="wrap">
      <Brand />
      <div className="card">
        <h2>How was your round{at}?</h2>
        <p className="small muted">
          {info.already_reviewed
            ? "You have already left a review — change it below if you like."
            : `Thanks for playing with GolfReelz, ${info.name}. A quick rating helps other golfers find us.`}
        </p>

        <div
          style={{ display: "flex", gap: 8, margin: "18px 0 6px" }}
          onMouseLeave={() => setHover(0)}
        >
          {STARS.map((n) => (
            <button
              key={n}
              type="button"
              aria-label={`${n} star${n === 1 ? "" : "s"}`}
              onClick={() => setRating(n)}
              onMouseEnter={() => setHover(n)}
              style={{
                background: "none",
                border: "none",
                cursor: "pointer",
                padding: 0,
                fontSize: 38,
                lineHeight: 1,
                // Gold rather than the navy --accent: a star reads as a
                // rating at a glance only in the colour people expect.
                color: n <= shown ? "#f59e0b" : "var(--border-strong)",
                transition: "color .12s",
              }}
            >
              ★
            </button>
          ))}
        </div>

        <div className="field" style={{ marginTop: 14 }}>
          <label>Anything you would like to add? (optional)</label>
          <textarea
            rows={5}
            value={comment}
            maxLength={4000}
            onChange={(e) => setComment(e.target.value)}
            placeholder="What did you think of the videos?"
          />
        </div>

        {error && <p className="small err-text">{error}</p>}

        <button
          disabled={!rating || submitting}
          onClick={submit}
          style={{ marginTop: 8 }}
        >
          {submitting
            ? "Sending…"
            : info.already_reviewed
              ? "Update my review"
              : "Submit review"}
        </button>
        {!rating && (
          <p className="small muted" style={{ marginTop: 8 }}>
            Pick a rating to continue.
          </p>
        )}
      </div>
    </div>
  );
}
