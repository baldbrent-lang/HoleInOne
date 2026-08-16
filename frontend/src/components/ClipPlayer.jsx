/**
 * Video player with branded corner overlays:
 *   top-right  — course name + "Hole N, YYY Yards"
 *   bottom-right — golfer name
 *   top-left (when ball_in_cup) — ACE badge
 *
 * Overlays are CSS-composited, not burned into the media. For shareable
 * downloads we'd re-render via server-side ffmpeg (out of scope for V0).
 */
export default function ClipPlayer({ clip, courseName, golferName, yardage }) {
  // The hole's yardage only. Measured carry is gone: it was a guess
  // dressed as a number, and the hole's length is a fact.
  const yards = yardage;
  return (
    <div className="clip-player">
      <video
        src={clip.source_url}
        poster={clip.thumbnail_url || undefined}
        controls
        playsInline
        preload="metadata"
      />
      {clip.ball_in_cup && (
        <div className="clip-overlay clip-overlay-ace">ACE</div>
      )}
      <div className="clip-overlay clip-overlay-top-right">
        {courseName && <div className="course-name">{courseName}</div>}
        <div className="hole-info">
          Hole {clip.hole_number}
          {yards ? `, ${yards} Yards` : ""}
        </div>
      </div>
      {golferName && (
        <div className="clip-overlay clip-overlay-bottom-right">
          {golferName}
        </div>
      )}
    </div>
  );
}
