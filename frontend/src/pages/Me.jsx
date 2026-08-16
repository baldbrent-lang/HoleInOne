import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api } from "../api.js";
import { Brand, Icon } from "../components/Brand.jsx";
import ClipPlayer from "../components/ClipPlayer.jsx";
import useAuth from "../hooks/useAuth.js";

function ordinalSuffix(n) {
  const s = ["th", "st", "nd", "rd"];
  const v = n % 100;
  return s[(v - 20) % 10] || s[v] || s[0];
}

function formatRoundDate(value) {
  const d = new Date(value);
  const weekday = d.toLocaleDateString("en-US", { weekday: "long" });
  const month = d.toLocaleDateString("en-US", { month: "long" });
  const day = d.getDate();
  return `${weekday} ${month} ${day}${ordinalSuffix(day)}`;
}

export default function Me() {
  const { user, loading } = useAuth();
  const nav = useNavigate();
  const [groups, setGroups] = useState(null);
  const [error, setError] = useState(null);
  const [openRoundId, setOpenRoundId] = useState(null);
  const [openRoundData, setOpenRoundData] = useState(null);

  useEffect(() => {
    if (loading) return;
    if (!user) {
      nav(`/login?next=${encodeURIComponent("/me")}`, { replace: true });
      return;
    }
    api.myRounds().then(setGroups).catch((e) => setError(e.message));
  }, [user, loading, nav]);

  async function openRound(participantId) {
    if (openRoundId === participantId) {
      setOpenRoundId(null); setOpenRoundData(null); return;
    }
    setOpenRoundId(participantId);
    setOpenRoundData(null);
    try {
      setOpenRoundData(await api.myRoundClips(participantId));
    } catch (e) {
      setError(e.message);
    }
  }

  if (loading || groups === null) {
    return (
      <div className="wrap">
        <Brand subtitle="Your rounds" />
        <div className="card"><div className="shimmer" style={{ height: 100 }} /></div>
      </div>
    );
  }

  return (
    <div className="wrap">
      <Brand subtitle="Your rounds" />

      <div className="hero" style={{ padding: "28px 24px" }}>
        <span className="eyebrow">
          <Icon name="users" size={14} /> Welcome back
        </span>
        <h1 style={{ fontSize: "clamp(1.6rem, 3vw, 2.1rem)" }}>
          Hi {user?.name || user?.email?.split("@")[0]}.
        </h1>
        <p>{groups.length === 0 ? "No rounds yet — pick a course to get started." : "Pick a round below to watch your clips."}</p>
        <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
          <Link to="/courses" className="btn small" style={{ width: "auto" }}>
            Register for a new round
          </Link>
          {user?.id && (
            <Link to={`/p/${user.id}`} className="btn secondary small" style={{ width: "auto" }}>
              View my public profile →
            </Link>
          )}
        </div>
      </div>

      {error && <div className="card err-text small">{error}</div>}

      {groups.length === 0 && (
        <div className="card center" style={{ padding: 28 }}>
          <p className="muted">No rounds linked to your account yet.</p>
          <Link to="/courses" className="btn" style={{ marginTop: 12, maxWidth: 220, marginLeft: "auto", marginRight: "auto" }}>
            Browse courses
          </Link>
        </div>
      )}

      {groups.map((g) => (
        <div key={g.course.id} className="card">
          <div className="inline" style={{ justifyContent: "space-between", width: "100%", marginBottom: 8 }}>
            <h3>{g.course.name}</h3>
            <span className="small muted">{g.course.location}</span>
          </div>

          <div className="stack" style={{ gap: 8 }}>
            {g.rounds.map((r) => {
              const open = openRoundId === r.participant_id;
              return (
                <div key={r.participant_id} className="card tight" style={{ margin: 0 }}>
                  <div
                    className="inline pointer"
                    style={{ justifyContent: "space-between", width: "100%" }}
                    onClick={() => openRound(r.participant_id)}
                  >
                    <div>
                      <b>{formatRoundDate(r.tee_time)}</b>
                      <div className="small muted">
                        {r.clips.assigned} of {r.clips.total || r.clips.assigned} clip
                        {r.clips.assigned === 1 ? "" : "s"} matched
                        {r.summary_sent_at ? "  ·  email sent" : "  ·  in progress"}
                      </div>
                    </div>
                    <div style={{ display: "flex", gap: 6 }}>
                      <a
                        className="btn secondary small"
                        href={r.gallery_url}
                        target="_blank"
                        rel="noreferrer"
                        onClick={(e) => e.stopPropagation()}
                      >
                        Public gallery ↗
                      </a>
                      <button className="ghost small" onClick={(e) => { e.stopPropagation(); openRound(r.participant_id); }}>
                        {open ? "Hide" : "View clips"}
                      </button>
                    </div>
                  </div>

                  {open && (
                    <div style={{ marginTop: 12 }}>
                      {!openRoundData ? (
                        <div className="shimmer" style={{ height: 120 }} />
                      ) : openRoundData.clips.length === 0 ? (
                        <div className="muted small">No clips matched yet.</div>
                      ) : (
                        <div className="grid" style={{ gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))" }}>
                          {openRoundData.clips.map((c) => (
                            <div key={c.id}>
                              <ClipPlayer
                                clip={c}
                                courseName={openRoundData.course?.name}
                                golferName={openRoundData.participant?.name}
                                yardage={openRoundData.course?.hole_yardages?.[String(c.hole_number)]}
                              />
                              <div className="small muted" style={{ marginTop: 6 }}>
                                Hole {c.hole_number}{" · "}
                                {c.camera_type.replace("_", " ")}
                              </div>
                            </div>
                          ))}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}
