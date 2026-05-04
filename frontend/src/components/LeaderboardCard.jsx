import { Icon } from "./Brand.jsx";

/**
 * Renders one leaderboard's top N rows. Used both in the Home preview and
 * on the full /leaderboards page.
 */
export default function LeaderboardCard({ title, icon = "chart", rows, emptyText, footer }) {
  return (
    <div className="card">
      <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 10 }}>
        <h3 style={{ display: "flex", alignItems: "center", gap: 8 }}>
          <span
            style={{
              width: 28, height: 28, borderRadius: 8,
              background: "var(--primary-soft)", color: "var(--emerald-700)",
              display: "grid", placeItems: "center",
            }}
            aria-hidden="true"
          >
            <Icon name={icon} size={16} />
          </span>
          {title}
        </h3>
      </div>
      {rows.length === 0 ? (
        <div className="muted small">{emptyText || "No entries yet — be the first."}</div>
      ) : (
        <ol style={{ listStyle: "none", padding: 0, margin: 0 }}>
          {rows.map((r, i) => (
            <li
              key={i}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 12,
                padding: "8px 0",
                borderTop: i === 0 ? "none" : "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  width: 26, textAlign: "center",
                  color: i === 0 ? "var(--emerald-700)" : "var(--ink-soft)",
                  fontWeight: 700,
                  fontFamily: "Fraunces, serif",
                }}
              >
                {i + 1}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <b>{r.golfer}</b>
                {(r.course || r.hole) && (
                  <div className="small muted">
                    {r.course}
                    {r.hole && ` · Hole ${r.hole}`}
                  </div>
                )}
              </div>
              <div style={{ fontWeight: 700, fontFamily: "Fraunces, serif" }}>
                {r.value}
                <span className="small muted" style={{ marginLeft: 4, fontFamily: "inherit", fontWeight: 500 }}>{r.unit}</span>
              </div>
            </li>
          ))}
        </ol>
      )}
      {footer}
    </div>
  );
}
