# GolfReelz — Shot Detection & Production Architecture

**This is the agreed direction for how GolfReelz captures and produces
golf-shot clips (locked 2026-06-26). All future work follows this model.**

## The core design

> Activate the video upon human motion in the tee box. Keep videoing until
> there is no human motion for more than 30 seconds. So we'll have one raw
> video of an entire group's shots. The full video is uploaded to production.
> The full video could contain 1 up to 5 golfers' shots. The production
> process cuts up the clips of each individual shot — with tracer, green
> view, and everything — as produced clips.

## Why this approach (division of labor)

The Pi and the backend each do the job they're good at:

- **Pi = the EASY job:** "is a human in the tee box, yes/no?" It does **not**
  try to detect individual swings. It records while someone is present and
  stops after 30s of an empty tee box.
- **Backend = the HARD job:** take the full clip and find/cut each individual
  shot. The server has the compute, can take its time, and can be re-run and
  tuned without ever touching the cameras.

This is already most of how the system is built — the long-upload pipeline
detects multiple swings in one video and cuts each into its own clip
("Multi-swing rounds auto-produce on upload").

## Capture flow (Pi)

Tee camera:
1. Detect a **human present** in the tee box — reliable person detection (a
   lightweight ML model), **not** the crude motion/pixel-count detector.
2. Start recording. Reset a timer on each human detection.
3. Stop when **no human for 30 seconds**. One clip = one group (1–5 shots).
4. Hand the finished clip to a **background uploader** and immediately go back
   to watching — never block detection on an upload (so we never miss a swing).

Green camera: records the same window (woken by the tee, stops when the tee
stops — existing event-trigger / event-stop / event-status mechanism), and
uploads in the background.

Record as **hardware H.264 at a controlled bitrate (~2.5–3 Mbps)** so a 4–5
minute group clip lands around 75–90 MB (plenty of quality for the tracer).

## Production flow (backend)

For each uploaded (tee + green) pair:
1. Run multi-swing detection on the tee clip → find each swing.
2. Confirm each is a real shot (the ball departs the tee).
3. For **each** shot: cut the segment, draw the tracer, composite the green
   view → one produced clip per golfer's shot.

Output: **N produced clips from one raw group video.**

## Implementation plan (priority order)

1. **Pi: reliable person detection** (ML model — "human present", the
   foundation) **+ the 30-second group-recording window.**
2. **Pi: background uploads + H.264 bitrate control** (handles the bigger
   group clips; uploads never block detection).
3. **Backend: move heavy processing to a background worker** so a long clip
   can't choke the web server or cause the timeouts we kept hitting.
4. **Backend: validate + tune the per-shot cutting** on real multi-shot
   footage — the make-or-break, and it needs real group clips to tune.

## File size

A full group at ~2.5–3 Mbps H.264 ≈ 75–90 MB. Background uploads make that a
non-issue. Extra levers if ever needed: trim the no-motion dead time before
upload; lower fps (30→24); chunked upload.

## Already built vs. new work

- **Already built:** long-upload multi-swing detect → per-shot cut → tracer →
  dual-camera composite. The camera path already routes through it.
- **New work:** reliable person detection on the Pi; the 30s group window;
  background uploads + bitrate control; a backend processing worker; and
  tuning the cutter against real footage.

## Known systemic issues this replaces / fixes

- Crude motion detector false-triggering (blank wall) vs. missing distant
  golfers — replaced by a real person detector.
- Pi uploading inside the detection loop (blocks detection, can miss swings,
  timeouts pile up) — replaced by background uploads.
- Server timing out because it processes clips on the same machine that
  serves requests — fixed by the background worker.

---

# Camera Placement (locked 2026-06-26, research-backed)

GolfReelz is a **cosmetic shot-video + tracer** product, NOT a launch monitor.
So we use the tracer-app / Rapsodo down-the-line view, not the perpendicular
launch-monitor view (that's for measuring 3D launch angles, which we don't do).

**Recommended placement (down-the-line, behind + slightly to the side):**
- Position: behind the golfer, ~2–4 ft to the NON-target side of the target
  line (camera-right for a RH golfer). Keeps you off the swing arc and out of
  the ball's flight path.
- Distance: ~10–14 ft behind the ball (~3–4 m).
- Height: ~4–5 ft (chest/shoulder) — see over the lead shoulder to the ball.
- Aim: down the target line, tilted UP ~10–20° so the top ~60–70% of frame is
  sky. Orient so the ball flies away from the sun.
- Zoom: wide end, ~8–10mm on the 8–50mm lens (ball still ~25–30 px at address;
  widest swath for the tracer). Don't exceed ~12mm.
- **Ball-against-sky is the #1 factor** for reliable detection AND a clean
  tracer (uniform sky vs. cluttered grass/trees).
- **Fast shutter (~1/1000 s or faster in daylight)** to freeze the ball.
  Motion blur is from EXPOSURE TIME, not frame rate — set a short shutter.
- The IMX477 is rolling shutter: fine for a cosmetic launch-arc tracer with a
  fast shutter; just don't expect 3D-measurement-grade ball flight.

Reality check: one fixed 30fps camera gives a great shot video + a tracer over
the launch/early arc against sky. Full 3D distance/trajectory needs multi-cam
triangulation (Toptracer) or radar (Trackman) — out of scope.

# Detection approach (locked)

**Person trigger ("is a human in the tee box?") → YOLOv8n / YOLO11n via NCNN.**
- Export once: `YOLO("yolov8n.pt").export(format="ncnn")`.
- Run ~5×/sec on a 320px frame, person class only (conf > ~0.4) + min box area
  / tee-box ROI gate. ~30 FPS headroom at 320px on the Pi 5 CPU.
- Debounce state machine: start after person present N frames; stop after the
  tee box is empty for the 30s window. This kills the false-trigger problem.
- Replaces the crude motion-density detector. (MediaPipe Pose DOES install on
  Pi OS Bookworm 64-bit ~6 FPS — our old "no wheel" error was a too-new OS
  (Trixie). But YOLO-NCNN is faster, more accurate, and OS-independent → use it.)

**Shot confirmation ("ball on tee → departed") → classical CV in a fixed ROI.**
- Find at address: white/bright round blob in a small fixed tee ROI (threshold
  + blob/contour by area+circularity, optional Hough confirm). Camera+tee are
  fixed, so this is reliable and cheap — no ML for the ball.
- Confirm departure: in that same ROI, "ball present → sudden motion spike →
  ball absent for K frames." Two-condition test rejects waggles / re-tees.

# Deployment order: one camera per tee, prioritized by usage (research-backed)

Don't put one high/wide camera over all tees — that makes the ball tiny and the
tracer weak (a bad video doesn't sell). Use one camera per tee that earns it,
each placed for a great ball view. Priority by where golfers actually play:

1. **Middle / regular ("white") tee — DEPLOY FIRST.** ~85% of men cluster on
   the middle tees; architects give them 50–70% of tee area. Avg male: 216–224
   yd drive, 14.0 handicap → ~6,100 yd tees = middle.
2. **Forward ("red") tee — second.** Avg female: 148–170 yd, 28.8 handicap →
   forward (~47% of women's rounds come from forward tees).
3. **Back / blue tee — only if it has traffic.** Used ~3% (7,000 yd courses) to
   <10% (>6,600 yd) of the time.
4. **Championship / tips — skip.** Too few players to pay for a camera.

Par-3 note: par-3 tees are much closer together than long-hole tees (avg-male
par-3 ~145–155 yds, avg-female ~105–115 yds), so two adjacent tees MAY be
coverable by one camera — measure the actual spacing to decide.

Key sources: USGA "Tee Options on Golf Courses" and "Helping Golfers Choose
Their Best Tees" (Green Section, ~55M scores); USGA/PGA Tee It Forward
guidelines (drive × 28 = ideal yardage); USGA 2019 Distance Report + Arccos
data; ASGCA "Tees" (tee-area allocation).
