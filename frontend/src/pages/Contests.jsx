import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, viewerId } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import LeaderboardCard from "../components/LeaderboardCard.jsx";

function Countdown({ endsAt }) {
  const compute = () => Math.max(0, new Date(endsAt) - new Date());
  const [ms, setMs] = useState(compute);
  useEffect(() => {
    const id = setInterval(() => setMs(compute()), 1000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endsAt]);

  if (ms <= 0) return <span>locked</span>;
  const days = Math.floor(ms / 86400000);
  const hours = Math.floor((ms % 86400000) / 3600000);
  const minutes = Math.floor((ms % 3600000) / 60000);
  const seconds = Math.floor((ms % 60000) / 1000);
  if (days > 0) return <span><b>{days}d</b> {hours}h {minutes}m</span>;
  if (hours > 0) return <span><b>{hours}h</b> {minutes}m {seconds}s</span>;
  return <span><b>{minutes}m</b> {seconds}s</span>;
}

const CADENCES = [
  { key: "daily",   label: "Daily contests", blurb: "Reset at midnight UTC. Small but constant — show up daily." },
  { key: "monthly", label: "Monthly draw",   blurb: "Every round you play this month is one entry. Drawn on the 1st." },
];

/**
 * The draw has no standings — every entry is equal, so a table would be
 * a list of names in no meaningful order. The number of entries is the
 * only true thing to show, and it doubles as the reason to play again:
 * a visible count is a visible chance.
 */
function DrawCounter({ contest }) {
  return (
    <div className="card" style={{ textAlign: "center" }}>
      <h3 style={{ marginBottom: 4 }}>{contest.title}</h3>
      <div
        style={{
          fontSize: "clamp(2.6rem, 8vw, 4rem)",
          fontWeight: 700,
          lineHeight: 1.05,
          color: "var(--emerald-700)",
          margin: "10px 0 2px",
        }}
      >
        {contest.count}
      </div>
      <p className="small muted" style={{ marginBottom: 0 }}>
        {contest.count_label}
      </p>
      <div
        className="small center"
        style={{
          marginTop: 14,
          paddingTop: 12,
          borderTop: "1px solid var(--border)",
          color: "var(--ink-soft)",
        }}
      >
        <span className="tiny upper muted" style={{ marginRight: 6 }}>Prize</span>
        <b style={{ color: "var(--emerald-700)" }}>{contest.prize}</b>
      </div>
    </div>
  );
}

/**
 * Shot of the Week. Five clips we picked, one vote each.
 *
 * The empty state matters as much as the full one: a shortlist we have
 * not put up yet is not a contest with no entries, and rendering it as
 * one would read like nobody plays here. So it says plainly that the
 * vote is coming, and when.
 */
function ShotOfWeek() {
  const [data, setData] = useState(null);
  const [busy, setBusy] = useState(null);
  const [error, setError] = useState(null);

  async function load() {
    try {
      setData(await api.shotOfWeek(viewerId()));
    } catch (e) {
      setError(e.message);
    }
  }
  useEffect(() => { load(); }, []);

  async function vote(nomineeId) {
    setBusy(nomineeId);
    try {
      const r = await api.voteShotOfWeek(nomineeId, viewerId());
      // Take the server's counts rather than guessing locally — a vote
      // that MOVED has to decrement the old one too.
      setData((d) => d && {
        ...d,
        my_vote: r.my_vote,
        nominees: d.nominees.map((n) => ({
          ...n, votes: Number(r.votes?.[String(n.id)] ?? n.votes),
        })),
      });
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(null);
    }
  }

  if (error && !data) return null;
  if (!data) {
    return (
      <section style={{ marginBottom: 28 }}>
        <div className="card"><div className="shimmer" style={{ height: 160 }} /></div>
      </section>
    );
  }

  return (
    <section id="sotw" style={{ marginBottom: 28 }}>
      <div className="cadence-header">
        <div>
          <h2 style={{ marginBottom: 2 }}>Shot of the Week</h2>
          <p className="small muted" style={{ margin: 0 }}>
            We shortlist the week&rsquo;s best. You pick the winner.
          </p>
        </div>
        <div className="cadence-countdown">
          <span className="tiny upper muted" style={{ marginRight: 6 }}>voting ends</span>
          <Countdown endsAt={data.ends_at} />
        </div>
      </div>

      {!data.open ? (
        <div className="card" style={{ textAlign: "center", padding: "30px 24px" }}>
          <h3 style={{ marginBottom: 6 }}>Voting opens soon</h3>
          <p className="small muted" style={{ maxWidth: 460, margin: "0 auto" }}>
            We are still going through this week&rsquo;s footage. Once the
            shortlist is up, five shots appear here and the vote is yours.
          </p>
          <div
            className="small center"
            style={{
              marginTop: 16, paddingTop: 12,
              borderTop: "1px solid var(--border)", color: "var(--ink-soft)",
            }}
          >
            <span className="tiny upper muted" style={{ marginRight: 6 }}>Prize</span>
            <b style={{ color: "var(--emerald-700)" }}>{data.prize}</b>
          </div>
        </div>
      ) : (
        <>
          <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
            {data.nominees.map((n) => {
              const mine = data.my_vote === n.id;
              return (
                <div key={n.id} className="card" style={{ padding: 12 }}>
                  <video
                    src={n.source_url}
                    poster={n.thumbnail_url || undefined}
                    controls
                    playsInline
                    preload="metadata"
                    style={{ width: "100%", borderRadius: 8, display: "block", background: "#000" }}
                  />
                  <div style={{ marginTop: 10 }}>
                    <b>{n.golfer}</b>
                    <span className="small muted">
                      {n.course ? ` · ${n.course}` : ""}
                      {n.hole ? ` · Hole ${n.hole}` : ""}
                    </span>
                    {n.ball_in_cup && (
                      <span className="pill ok" style={{ marginLeft: 6 }}>ACE</span>
                    )}
                  </div>
                  {n.caption && (
                    <p className="small muted" style={{ margin: "6px 0 0" }}>{n.caption}</p>
                  )}
                  <div
                    style={{
                      display: "flex", alignItems: "center", gap: 10,
                      marginTop: 10, justifyContent: "space-between",
                    }}
                  >
                    <button
                      className={mine ? "small" : "secondary small"}
                      style={{ width: "auto" }}
                      disabled={busy === n.id}
                      onClick={() => vote(n.id)}
                    >
                      {busy === n.id ? "Voting…" : mine ? "✓ Your vote" : "Vote"}
                    </button>
                    <span className="small muted">
                      {n.votes} vote{n.votes === 1 ? "" : "s"}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
          <div
            className="small center"
            style={{ marginTop: 12, color: "var(--ink-soft)" }}
          >
            <span className="tiny upper muted" style={{ marginRight: 6 }}>Prize</span>
            <b style={{ color: "var(--emerald-700)" }}>{data.prize}</b>
            {data.my_vote && (
              <span className="muted"> · you can change your vote until voting closes</span>
            )}
          </div>
        </>
      )}
    </section>
  );
}

export default function Contests() {
  const [data, setData] = useState(null);
  useEffect(() => { api.contests().then(setData).catch(() => setData({})); }, []);

  return (
    <div className="wrap wide">
      <Brand subtitle="Live contests" />
      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="sparkle" size={14} /> Contests</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3.2vw, 2.2rem)" }}>
          Play a round. Win a prize.
        </h1>
        <p>
          Every registered round counts. Closest to the pin, shot of the week,
          and the monthly draw. The $10,000 hole-in-one sweepstakes runs
          alongside.
        </p>
        <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
          <Link to="/courses" className="btn small" style={{ width: "auto" }}>Pick a course — $20</Link>
        </div>
      </div>

      {!data ? (
        <div className="card"><div className="shimmer" style={{ height: 240 }} /></div>
      ) : (
        CADENCES.map(({ key, label, blurb }) => {
          const section = data[key];
          if (!section) return null;
          return (
            <section key={key} style={{ marginBottom: 28 }}>
              <div className="cadence-header">
                <div>
                  <h2 style={{ marginBottom: 2 }}>{label}</h2>
                  <p className="small muted" style={{ margin: 0 }}>{blurb}</p>
                </div>
                <div className="cadence-countdown">
                  <span className="tiny upper muted" style={{ marginRight: 6 }}>ends in</span>
                  <Countdown endsAt={section.ends_at} />
                </div>
              </div>
              <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
                {section.contests.map((c) => (
                  c.kind === "counter" ? (
                    <DrawCounter key={c.id} contest={c} />
                  ) : (
                  <LeaderboardCard
                    key={c.id}
                    title={c.title}
                    icon={c.icon}
                    rows={c.rows}
                    emptyText="No entries yet — be the first."
                    footer={
                      <div
                        className="small center"
                        style={{
                          marginTop: 14,
                          paddingTop: 12,
                          borderTop: "1px solid var(--border)",
                          color: "var(--ink-soft)",
                        }}
                      >
                        <span className="tiny upper muted" style={{ marginRight: 6 }}>Prize</span>
                        <b style={{ color: "var(--emerald-700)" }}>{c.prize}</b>
                      </div>
                    }
                  />
                  )
                ))}
              </div>
            </section>
          );
        })
      )}

      <ShotOfWeek />

      <div className="card" style={{ background: "var(--primary-soft)", border: "1px solid var(--emerald-200)" }}>
        <h3 style={{ color: "var(--emerald-800)" }}>About the prizes</h3>
        <p className="small" style={{ color: "var(--emerald-800)" }}>
          Closest to the pin is measured from real clips. Shot of the week is
          shortlisted by us and decided by your votes. The monthly draw is one
          entry per round, drawn at random. Winners are locked when the timer
          hits zero, and we email you if it is you.
        </p>
      </div>
    </div>
  );
}
