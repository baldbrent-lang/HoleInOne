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

/** Most recent first. The last round played is the one being looked for. */
function byDateDesc(a, b) {
  return new Date(b.tee_time) - new Date(a.tee_time);
}

/**
 * What to say about a round under its date.
 *
 * This used to read "email sent" or "in progress" from summary_sent_at,
 * a column nothing writes any more since the round-summary email was
 * removed — so every round, however complete, said "in progress"
 * forever. It describes the clips now, which is what the golfer came to
 * find out and what we can actually answer.
 */
function roundStatus(r) {
  const { assigned, total } = r.clips;
  const known = total || assigned;
  if (known === 0) return "No clips yet";
  if (assigned === 0) return `${known} clip${known === 1 ? "" : "s"} being matched`;
  if (assigned < known) return `${assigned} of ${known} clips ready`;
  return `${assigned} clip${assigned === 1 ? "" : "s"} ready to watch`;
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
  // Which course we have drilled into. Null = showing the course picker.
  const [courseId, setCourseId] = useState(null);

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

  // A picker with one option is just an extra click, so a golfer who has
  // only ever played one course lands straight on their rounds.
  const selected =
    (courseId && groups.find((g) => g.course.id === courseId)) ||
    (groups.length === 1 ? groups[0] : null);

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
        <p>
          {groups.length === 0
            ? "No rounds yet — pick a course to get started."
            : selected
              ? "Pick a date to watch your clips."
              : "Pick a course to see the rounds you played there."}
        </p>
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

      {/* COURSE PICKER. The list used to be every course with every
          round nested under it, which on a third visit is a long scroll
          past rounds you are not looking for. Pick the course, then the
          date. */}
      {groups.length > 0 && selected === null && (
        <div
          className="grid"
          style={{ gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}
        >
          {groups.map((g) => {
            const rounds = [...g.rounds].sort(byDateDesc);
            const last = rounds[0];
            return (
              <div
                key={g.course.id}
                className="card pointer"
                onClick={() => { setCourseId(g.course.id); setOpenRoundId(null); }}
              >
                <h3 style={{ marginBottom: 2 }}>{g.course.name}</h3>
                <div className="small muted">{g.course.location}</div>
                <div className="small" style={{ marginTop: 10 }}>
                  <b>{rounds.length}</b> round{rounds.length === 1 ? "" : "s"}
                  {last && (
                    <span className="muted"> · last {formatRoundDate(last.tee_time)}</span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* THE ROUNDS AT ONE COURSE, newest first. */}
      {selected && (
        <>
          <div
            className="inline"
            style={{ justifyContent: "space-between", width: "100%", marginBottom: 10, flexWrap: "wrap", gap: 8 }}
          >
            <div>
              {groups.length > 1 && (
                <button
                  className="ghost small"
                  style={{ width: "auto", marginBottom: 4 }}
                  onClick={() => { setCourseId(null); setOpenRoundId(null); }}
                >
                  ← All courses
                </button>
              )}
              <h3 style={{ margin: 0 }}>{selected.course.name}</h3>
            </div>
            <span className="small muted">{selected.course.location}</span>
          </div>

          <div className="stack" style={{ gap: 8 }}>
            {[...selected.rounds].sort(byDateDesc).map((r) => {
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
                      <div className="small muted">{roundStatus(r)}</div>
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
        </>
      )}

    </div>
  );
}
