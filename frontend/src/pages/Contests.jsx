import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
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

      <div className="card" style={{ background: "var(--primary-soft)", border: "1px solid var(--emerald-200)" }}>
        <h3 style={{ color: "var(--emerald-800)" }}>About the prizes</h3>
        <p className="small" style={{ color: "var(--emerald-800)" }}>
          Closest to the pin is measured from real clips; shot of the week and
          the monthly draw are picked by us. Winners are locked when the timer
          hits zero, and we email you if it is you.
        </p>
      </div>
    </div>
  );
}
