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
