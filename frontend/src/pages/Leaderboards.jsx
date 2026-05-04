import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import LeaderboardCard from "../components/LeaderboardCard.jsx";

export default function Leaderboards() {
  const [data, setData] = useState(null);

  useEffect(() => {
    api.leaderboards(10).then(setData).catch(() => setData({}));
  }, []);

  return (
    <div className="wrap wide">
      <Brand subtitle="Current leaderboards" />
      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow"><Icon name="chart" size={14} /> Top of the board</span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3.2vw, 2.2rem)" }}>
          The longest, the fastest, the cleanest aces.
        </h1>
        <p>Rankings update as new clips get matched. Want on the board? Pick a course and play.</p>
        <Link to="/courses" className="btn small" style={{ width: "auto", marginTop: 14 }}>Pick a course</Link>
      </div>

      {!data ? (
        <div className="card"><div className="shimmer" style={{ height: 240 }} /></div>
      ) : (
        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))" }}>
          <LeaderboardCard
            title="Longest carry"
            icon="sparkle"
            rows={data.longest_carry || []}
            emptyText="No carries logged yet."
          />
          <LeaderboardCard
            title="Fastest ball"
            icon="capture"
            rows={data.fastest_ball || []}
            emptyText="No ball-speed data yet."
          />
          <LeaderboardCard
            title="Most aces"
            icon="dollar"
            rows={data.most_aces || []}
            emptyText="No aces… yet."
          />
          <LeaderboardCard
            title="Most rounds played"
            icon="users"
            rows={data.most_rounds || []}
            emptyText="No registered users yet."
          />
        </div>
      )}
    </div>
  );
}
