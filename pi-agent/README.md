# GolfReelz Pi capture agent (Phase 3)

The unattended Python service that runs on each Raspberry Pi to feed
the GolfReelz backend. One agent codebase, two roles — the role is
chosen at runtime from the camera's backend registration:

- **Tee Pi**: reads the camera feed continuously, runs MediaPipe Pose
  at 5 fps, triggers when a person enters the configured tee-box ROI
  for ≥ 2 s. Persists a 5 s pre-roll + everything until the person
  leaves, then uploads.
- **Green Pi**: continuously buffers 5 s of frames in RAM, long-polls
  the backend for triggers, commits the buffer + records 12 s on
  trigger, then uploads.

Both share the same config schema, the same HTTP client, the same
systemd unit. The only operational difference is which `auth_token`
(tee or green from `/admin/cameras`) you put in `config.yaml`.

## Architecture recap

```
[GoPro / Pi Camera] ──> [Pi 5 running this agent] ──HTTPS──> Backend
                            │
                            ├─ POST /api/cameras/{token}/heartbeat   (60s)
                            ├─ POST /api/cameras/{token}/event-trigger  (tee, on detect)
                            ├─ GET  /api/cameras/{token}/poll-trigger   (green, long-poll)
                            └─ POST /api/cameras/{token}/upload-event   (both, after record)
```

The backend handles cutting, AI tracer, dual-cam composite, intro
overlay, share buttons, matching to players — see
`docs/field-deployment.md` and `backend/app/routers/cameras.py`.

## Files

| File | Purpose |
|---|---|
| `golfreelz_agent.py` | Entry point; first heartbeat decides tee or green |
| `agent/common.py` | Config loader, HTTP client, ring buffer, heartbeat thread |
| `agent/tee.py` | Tee-side detect-and-record loop |
| `agent/green.py` | Green-side long-poll-and-record loop |
| `focus.py` | Live focus meter — turn the lens ring until the score peaks |
| `config.example.yaml` | Template config; copy to `config.yaml` and fill in |
| `requirements.txt` | Python deps (OpenCV, MediaPipe, requests, PyYAML) |
| `install.sh` | One-shot installer for a fresh Raspberry Pi OS install |
| `systemd/golfreelz-agent.service` | systemd unit that runs the agent on boot |

## Setup (fresh Pi)

Assuming you've flashed Raspberry Pi OS Lite (Bookworm or newer)
onto an SD card, set Wi-Fi / SSH in the imager, and booted the Pi.

```bash
# 1. SSH into the Pi
ssh pi@raspberrypi.local

# 2. Copy this directory to /opt/golfreelz-agent (e.g. via scp from
#    your laptop, or git clone the repo + symlink). For first-time
#    test you can just clone the whole HoleInOne repo:
sudo mkdir -p /opt/golfreelz-agent
sudo chown -R "$USER:$USER" /opt/golfreelz-agent
cd /opt/golfreelz-agent
git clone --depth 1 https://github.com/baldbrent-lang/HoleInOne tmp
mv tmp/pi-agent/* tmp/pi-agent/.* . 2>/dev/null || true
rm -rf tmp

# 3. Run the installer (apt deps + venv + service user + systemd)
sudo bash /opt/golfreelz-agent/install.sh

# 4. Create config.yaml with this device's auth_token + backend URL
sudo cp config.example.yaml config.yaml
sudo nano config.yaml   # fill in backend_url, auth_token, tee_box_roi

# 5. Start the agent
sudo systemctl start golfreelz-agent
sudo journalctl -u golfreelz-agent -f
```

The agent will log its first heartbeat, then either start the tee or
green main loop based on the role the backend returned for this
token.

## Finding the tee-box ROI

Only required for tee cameras. The ROI is a rectangle in the
camera's captured frame (pixel coords, origin top-left) — anything
inside that rectangle counts as "on the tee box."

After mounting the camera in its final position:

```bash
# Snap a sample frame
cd /opt/golfreelz-agent
sudo -u golfreelz ./venv/bin/python3 -c "import cv2; \
  cap=cv2.VideoCapture(0); cap.set(3,1920); cap.set(4,1080); \
  ok,f=cap.read(); cv2.imwrite('/tmp/frame.jpg', f); print('saved')"

# Copy /tmp/frame.jpg back to your laptop, open in an image editor,
# draw a rectangle around the tee markers, note the pixel coords.
# Plug them into config.yaml as tee_box_roi.x/y/w/h.
```

## Checking focus

The HQ camera (IMX477) has **no autofocus** — focus and aperture are
physical rings on the C/CS lens. `focus.py` gives you a number to tune
against instead of squinting at a preview on a laptop in sunlight.

```bash
sudo systemctl stop golfreelz-agent        # the service holds the camera
cd /opt/golfreelz-agent
sudo -u golfreelz ./venv/bin/python3 focus.py --roi tee --save /tmp/focus.jpg
# ... turn the focus ring slowly, stop where `score` peaks, Ctrl-C ...
sudo systemctl start golfreelz-agent
```

Output looks like:

```
score    1284  (best    1301)  [###################  ]  bright 118  zones L 940 C 1284 R 1102
```

- **score** — sharpness right now. The absolute value means nothing across
  scenes; only its **peak as you turn the ring** matters.
- **best** — highest seen this run, so you can tell you have gone past it.
- **zones** — left/centre/right thirds. A large spread means the camera
  isn't square to the scene, or the lens is soft off-axis (stopping the
  aperture down usually fixes the latter).
- **bright** — mean luma. Below ~40 or above ~220 the score is unreliable;
  fix exposure first.

Three things look identical in a finished clip, and this separates them:

| What you see | Cause | Fix |
|---|---|---|
| A *static* object at the golfer's distance scores low here | focus | the lens ring |
| Scores fine here, but *moving* things smear in the video | motion blur | shutter speed / more light |
| Scores fine and the saved 100% crop is sharp, but the uploaded clip is soft | compression | bitrate (green uploads 720p @ 1500 kbps) |

**If the saved `_crop100.jpg` is sharp, focus is not the problem.**

Two things worth doing while you are out there:

- **Stop the aperture down.** Wide open (f/1.2 on the 6mm CS lens) the
  depth of field is very shallow and the corners are soft. Around f/4–5.6
  sharpens the frame and buys enough depth that the ball at address and
  the ball a few yards downrange are both acceptable. In daylight you have
  light to spare.
- **Focus through the enclosure**, at the real mounting distance — acrylic
  shifts the focal plane slightly and shows up any haze or condensation.

Lock both rings with their grub screws when done; transport walks them.

## Updating the agent code

```bash
# Default: pull main + restart the service
sudo /opt/golfreelz-agent/update.sh

# Or pull a specific branch / tag
sudo /opt/golfreelz-agent/update.sh some-feature-branch
```

`update.sh` clones the repo into a tmp dir, stops the service, copies
the agent files into place, refreshes pip requirements, and restarts.
It only touches code under `agent/` plus the top-level entry script —
`config.yaml`, the systemd unit, and any other local edits are left
alone.

If `update.sh` itself is missing (e.g. on a Pi installed before it
existed), one-time bootstrap:

```bash
sudo curl -fsSL -o /opt/golfreelz-agent/update.sh \
  https://raw.githubusercontent.com/baldbrent-lang/HoleInOne/main/pi-agent/update.sh
sudo chmod +x /opt/golfreelz-agent/update.sh
sudo chown golfreelz:golfreelz /opt/golfreelz-agent/update.sh
```

After that, every future update is just `sudo /opt/golfreelz-agent/update.sh`.

## Troubleshooting

- **`initial heartbeat failed: 404`** — the `auth_token` in
  `config.yaml` doesn't match any camera on the backend. Re-check
  the token in `/admin/cameras` (or rotate it there and update
  `config.yaml`).
- **`backend returned unknown assigned_role`** — the camera exists
  but its `assigned_role` is something other than `tee` or `green`.
  Set it correctly via `/admin/cameras`.
- **`mediapipe not importable — falling back to motion-density`** —
  MediaPipe didn't install (most often: 32-bit Pi OS on a Pi 3 / armv7).
  The motion-fallback detector keeps the network plumbing testable
  but is much noisier in production. Either install a 64-bit Pi OS
  or limit deployment to Pi 4 / Pi 5.
- **GoPro USB-C not detected as `/dev/video*`** — make sure the
  GoPro is set to *USB-webcam* mode (Hero 11+ supports this out of
  the box; older Heros need GoPro Labs firmware). Plug the USB-C
  cable in *while the GoPro is off*, then power it on; the Pi sees
  it as a UVC device.
- **Recordings are upside-down / mirrored** — most cameras don't
  rotate automatically. If you need to rotate, edit
  `agent/common.py:open_camera` to wrap the cap in a rotating
  decorator (one-liner with `cv2.rotate`).

## What this doesn't do (yet)

- **PIR sleep cycle** — the production tier in
  `docs/field-deployment.md` describes wake-on-PIR for indefinite
  solar operation. Not built; the agent runs flat-out. Add it later
  by gating the main loop on a GPIO input.
- **OTA updates** — the agent doesn't pull new code on its own.
  Re-deploy manually via SSH for now.
- **Per-event diagnostics** — failures log to the journal but don't
  surface anywhere the operator can see. Pair this with the Phase
  2 admin pages (camera card shows `last_seen_at` + last event
  status) for now; a richer per-event view is on the roadmap.
