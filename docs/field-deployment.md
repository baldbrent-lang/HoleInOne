# Always-on Camera Field Deployment

How to deploy GolfReelz capture hardware on a course so swings are
auto-detected, recorded, and produced into broadcast clips without
operator involvement.

The current backend already handles cutting, AI tracer rendering,
dual-cam composites, participant matching, the intro overlay, and
social sharing. This doc covers the **camera and infrastructure side**
— what hardware to buy, how to mount it, and what software still
needs to be built to feed the backend.

---

## Architecture overview

Each par-3 hole gets two camera positions:

```
TEE                                                       GREEN
 ┌──────────────────────┐                  ┌──────────────────────┐
 │ Pi HQ Camera (wide)  │                  │ Pi HQ Camera + lens  │
 │   ↓ CSI ribbon       │                  │   ↓ CSI ribbon       │
 │ Raspberry Pi 5       │                  │ Raspberry Pi 5       │
 │   ↓ MediaPipe Pose   │                  │   ↓ continuous       │
 │   detects person     │                  │   5s circular buffer │
 │   on tee box         │                  │                      │
 └──────────┬───────────┘                  └──────────┬───────────┘
            │                                         │
            │ LTE / Wi-Fi                             │ LTE / Wi-Fi
            ↓                                         ↓
        ┌────────────────────────────────────────────────────┐
        │ GolfReelz backend                                  │
        │   POST /api/cameras/{token}/event-trigger          │
        │   POST /api/cameras/{token}/upload-event           │
        │   Pairs by camera_id + session_id, runs the        │
        │   existing dual-cam composite pipeline             │
        └────────────────────────────────────────────────────┘
```

Both ends run the **same camera** — a Raspberry Pi HQ Camera on a CSI
ribbon — differing only in lens (tee: wide, to cover the tee box + the
start of ball flight; green: telephoto, sized to the green distance).
This replaces the earlier GoPro-over-USB tee plan: the GoPro's
USB-webcam mode needed re-activation on every bus re-enumeration,
contended for USB power, and delivered only ~29 unique fps (dropping
~1 frame/s), which showed up as recorded-clip stutter. A CSI sensor
feeds OpenCV at a steady true rate and removes that whole class of
failures, and running identical rigs at both ends halves the parts /
knowledge you have to maintain.

Tee Pi runs vision detection. When a person enters the tee-box ROI
for ≥2 s, it (a) starts persisting its camera recording with 5 s of
pre-roll, and (b) POSTs an event-trigger to the backend with a
shared `session_id`. Backend forwards "start recording" to the
paired green Pi, which commits its 5 s pre-roll buffer and keeps
recording for 10–12 s. Both clips upload, the backend pairs them by
session_id, and the existing per-segment pipeline produces the final
composite — same path a Reprocess uses today.

---

## Single-hole field test (MVP)

**Goal:** validate the architecture on one par 3 for a single
tournament day. Find out:

- Does MediaPipe Pose on the tee feed catch every real swing?
- What's the false-positive rate (groundskeeper triggers, cart traffic)?
- Does the Pi HQ Camera + wide lens give acceptable tee-camera image
  quality — enough resolution on the ball as it shrinks downrange?
- Is the green Pi's pre-roll + remote trigger reliable for catching the ball landing?
- Does cellular signal hold at both ends?

**Scope:** one tee + one green at the same par 3. Daily battery swap
(no solar). Operator-attended deployment in case of failures.

### Bill of materials — single-hole MVP

| Component | Part | Price | Notes |
|---|---|---|---|
| **Tee end** | | | |
| Camera | Pi HQ Camera + 6 mm wide lens | ~$100 | wide FOV: tee box + start of ball flight |
| Compute | Raspberry Pi 5 8 GB + case + power supply + 64 GB microSD | ~$110 | |
| Camera ribbon | 30 cm CSI ribbon cable | $5 | |
| Enclosure | Pelican 1050 + cable glands | $50 | weatherproof |
| Mount | Manfrotto Super Clamp + 2" ratchet strap | $30 | tree-mountable |
| **Green end** | | | |
| Camera | Pi HQ Camera + 16 mm telephoto lens | $80 | swap lens by mount distance |
| Compute | Raspberry Pi 5 8 GB + case + power supply + 64 GB microSD | ~$110 | |
| Camera ribbon | 30 cm CSI ribbon cable | $5 | |
| Enclosure | Pelican 1050 + cable glands | $50 | |
| Mount | Manfrotto Super Clamp + 2" ratchet strap | $30 | |
| **Shared** | | | |
| Network (×2) | NETGEAR LM1200 LTE USB modem + month of data | $80 + $20 | one per Pi |
| Power (×2) | Anker 537 PowerHouse 200 Wh power station | $300 | swap daily |
| **Total MVP** | | **~$980** | + $20/mo data |

Both ends are now the same Pi HQ Camera body — only the lens differs
(tee 6 mm wide, green 16 mm telephoto). Buy two camera bodies. The
MVP is ~$90 more than the GoPro plan only because the GoPro was
"already owned"; in exchange you drop the USB-webcam tooling and the
stutter, and the two rigs become interchangeable spares.

### Pre-deploy checklist

- [ ] Course superintendent has approved camera mounting on the
      selected trees (strap mounts, no fasteners into bark)
- [ ] Cellular signal verified at both tee and green (phone, no Wi-Fi,
      get a download speed test ≥ 2 Mbps at each location)
- [ ] Tree trunks selected: tee within 25 yd of the tee markers,
      green within 30 yd of the apron, both with unobstructed sight
      line to the relevant action area
- [ ] Lenses picked: **tee** gets a wide 6 mm (it sits close behind the
      tee and must see the box + the start of ball flight); **green**
      gets 16 mm if its mount is 15–30 m from the landing zone, 25 mm
      if 30–55 m (see lens distance table below)

### Lens distance reference

| Mount distance from action | Pi HQ lens | Cost | Horizontal FOV |
|---|---|---|---|
| 8–15 m | 6 mm wide | $25 | ~54° |
| 15–30 m | 16 mm | $30 | ~22° |
| 30–55 m | 25 mm | $40 | ~14° |
| 55–100 m | 50 mm | $60 | ~7° |

Aim to cover a ~10 m wide area of interest. Trees too close to the
play area can be hit by errant shots; trees too far need expensive
glass.

### Setup steps

1. **Backend prep (one-time)** — Add the `Camera` model + token-auth
   endpoints (see Software section below). Register two cameras for
   this hole, note their tokens.
2. **Bench-test indoors** — Flash both Pis with the GolfReelz
   capture agent (separate project, see Software section). Verify
   they can record a clip and upload it to the backend over LTE.
3. **Field install** — Mount tee enclosure ~25 yd behind the tee
   markers, strap to a tree on the cart-path side. Mount green
   enclosure ~20 yd off the green's apron, also tree-strapped. Both
   enclosures live ~3 ft off the ground (low enough that wind sway
   is minimal, high enough to see over walking golfers).
4. **Cabling** — Pi HQ Camera to its Pi via CSI ribbon at **both**
   ends (camera and Pi share an enclosure). Both Pis powered from
   their respective Anker 537 power stations sitting in the enclosure.
5. **Tee-box ROI calibration** — Tee Pi capture agent has a config
   file for the tee-box rectangle in pixel coordinates. With the
   camera mounted, send the live feed to the operator's phone, draw
   the rectangle around the tee markers, save.
6. **Pairing** — In the admin console, link the two cameras to the
   same `paired_with_camera_id`. The backend uses this to relay the
   event-trigger.
7. **Smoke test** — Operator walks into the tee box, swings a club.
   Verify within 30 s: tee Pi triggered, green Pi recorded its
   buffer, both clips landed in the backend, composite produced and
   visible on `/admin/broadcast-clips`.
8. **Tournament day** — Swap power stations every morning at 6 am.
   Monitor the broadcast page through the day; flag any missed swings.
9. **Post-mortem** — Compare clips produced vs. swings actually
   played (from the tee sheet). Compute detection rate, false-positive
   rate, average upload latency, image quality issues. Decide whether
   to scale to all holes.

---

## Full production setup

**Goal:** always-on, season-long, all par-3 tees on the course.
Operator's job is checking the broadcast page, not maintaining
cameras.

**Typical scope:** 4 par-3 holes per course → 8 cameras (4 tee + 4 green).

### Bill of materials — per hole, production

| Component | Part | Price | Notes |
|---|---|---|---|
| **Tee end** | | | |
| Camera | Pi HQ Camera + 6 mm wide lens | $75–100 | wide FOV for tee box + ball flight |
| Compute | Raspberry Pi 5 8 GB kit | $110 | |
| Mount + enclosure | Strap mount + Pelican 1050 + glands | $80 | |
| **Green end** | | | |
| Camera | Pi HQ Camera + lens (per-mount, $25–60) | $75–110 | |
| Compute | Raspberry Pi 5 8 GB kit | $110 | |
| Mount + enclosure | Strap mount + Pelican 1050 + glands | $80 | |
| **Per side (×2 — tee + green)** | | | |
| Network | LTE USB modem + SIM | $30 + $10/mo | one per Pi |
| Power | 100 W solar panel + 100 Wh LiFePO4 battery + charge controller + PIR sensor | $200 | self-sustaining |
| **Per hole total** | | **~$1,195–1,230** | + $20/mo data |
| **4-hole course** | | **~$4,800** | + $80/mo data |

Switching the tee from a $400 GoPro to a ~$100 Pi HQ Camera drops the
per-hole cost by ~$300 (a ~$1,200 saving across a 4-hole course) and
makes both ends identical rigs. The production tier still runs more
than the MVP mainly because of solar/battery instead of swappable
power stations.

### What changes vs. MVP

| Area | MVP | Production |
|---|---|---|
| Power | Anker power station, swap daily | 100 W solar + 100 Wh LiFePO4 battery + PIR sleep |
| Operator involvement | Daily battery swap, check broadcast page | Seasonal checkup, automated alerts on camera offline |
| Failure handling | Phone the operator | Backend monitors `Camera.last_seen_at`, auto-emails when a camera goes silent for >30 min |
| Scale | 1 hole | 4 holes (all par 3s) |
| Detection model | MediaPipe Pose always running | Wake-on-PIR → MediaPipe only while PIR active |

### Power budget with PIR sleep cycle

| State | Tee draw | Green draw | Combined |
|---|---|---|---|
| Idle (PIR not firing) | ~0.3 W | ~0.3 W | ~0.6 W |
| Active (person detected, recording) | ~14 W | ~12 W | ~26 W |

Realistic day on a busy par 3 (80 groups × 30 s active per group):
- Active: 80 × 30 s = 40 min/day @ 26 W = ~17 Wh
- Idle: 23.3 hr/day @ 0.6 W = ~14 Wh
- **~31 Wh/day total per hole**

A 100 W solar panel produces 300–500 Wh of usable energy per day in
the typical 4 hours of effective solar in mid-latitude conditions.
That's a 10× safety margin — covers cloudy weeks and dawn / dusk
operation.

### Production setup steps

Same as the MVP per hole, plus:

1. **Site survey before purchase** — walk every par 3 with a tape
   measure. Identify the best tee tree and best green tree. Measure
   the distance from each tree to the action area. Order the
   appropriate lens for each green-side camera based on the table
   above.
2. **Provision per-hole device IDs** — register all 8 cameras in
   the admin console up front, generate auth tokens, pre-configure
   each Pi's SD card with its token before going on-site.
3. **Stagger installs** — install one hole per day. Day-1 lessons
   (mount height, cable routing, ROI calibration) flow into
   subsequent installs.
4. **Document each mount** — photo of the mounted enclosure, GPS
   coordinates, tree species, lens used. Lives in a simple Google
   Sheet that the maintenance crew refers to.
5. **Schedule maintenance** — 6-week visit per camera: check strap
   tightness, wipe solar panel + lens, clear branches that have
   grown into FOV, confirm enclosure is dry, log to the sheet.

### Operational ongoing costs

| Item | Per hole / month | 4-hole course |
|---|---|---|
| LTE data (~1 GB/cam/mo) | $20 (2 cameras) | $80 |
| Backend hosting | shared across all courses | included |
| Spare parts buffer | ~$10 | $40 |
| **Total recurring** | **$30** | **$120** |

Capex amortized over 3 years on a 4-hole course: $6,000 / 36 mo =
$167/mo. **Total ~$290/mo per course operational cost**, or roughly
$70/hole/month all-in including hardware amortization.

---

## Software to build

The backend already handles per-segment processing (cut, AI tracer,
composite, match, intro overlay, sharing). Two new pieces are needed
to feed it from cameras:

### Backend additions (~3 days)

1. **`Camera` model** — `id`, `course_id`, `assigned_hole`,
   `assigned_role` (tee | green), `paired_with_camera_id`,
   `auth_token`, `tee_box_roi` (JSON rect, tee cameras only),
   `last_seen_at`, `firmware_version`.
2. **`POST /api/cameras/{token}/event-trigger`** — tee Pi calls this
   when it detects a person. Body: `session_id`, optional metadata.
   Backend looks up the paired green camera and pushes a
   "start recording" message to it (long-poll or WebSocket).
3. **`POST /api/cameras/{token}/upload-event`** — both Pis upload
   their recorded clip here with the shared `session_id`. Backend
   matches the pair, queues a single per-segment processing job
   (existing `_process_long_upload_segments` helper with
   `auto_detect_swings=False`).
4. **Admin pages**:
   - `/admin/cameras` — list all cameras, last seen, last upload,
     errors, links to register / pair / unpair.
   - Heartbeat / offline alert — email the operator when any camera
     hasn't checked in for > 30 min during course hours.

### Pi capture agent (~2 weeks, separate project)

A standalone Python service that runs on each Pi.

**Tee agent responsibilities:**
- Read frames from the Pi HQ Camera (CSI, via libcamera → V4L2) with
  OpenCV in a dedicated capture thread (already a Pi 5 dependency).
- MediaPipe Pose at 5 fps, ROI intersection check.
- Maintain a 5 s circular buffer of full-res frames.
- On trigger (person in ROI ≥2 s): POST event-trigger to backend
  with `session_id = uuid4()`, start recording the camera feed at the
  sensor's true fps, persist the pre-roll.
- After 5 s of no person: stop recording, upload clip with same
  `session_id`, delete local file.
- Retry on network errors with exponential backoff.

**Green agent responsibilities:**
- Continuously buffer 5 s of camera feed in RAM (no disk writes).
- Listen for "start recording" from backend (long-poll endpoint).
- On signal: commit buffer to disk, continue recording for 10–12 s,
  upload with the supplied `session_id`.
- Same retry / error handling.

**Shared:**
- Heartbeat every 60 s to keep `last_seen_at` fresh.
- Wake-on-PIR loop (production tier only) to drop idle power.
- OTA-update mechanism (pull from a known URL daily, restart if
  newer commit available).

### Software not yet built but worth knowing about

- **Vertical re-encode** for Reels/TikTok-ready outputs — converts
  the 16:9 composite to 9:16 with smart subject framing.
- **Landing detection + zoom on green** — auto-zoom into the ball-at-rest
  position in the green segment of the composite.
- **Distance-to-pin estimate** — flagstick-based pixel-to-feet scale.

All three are deferred features sketched in chat but not committed.

---

## Risks and mitigations

| Risk | Likelihood | Mitigation |
|---|---|---|
| Cellular signal dead at a tee | Medium | Site survey before purchase; some courses need a Wi-Fi mesh extender from the clubhouse |
| Tree sway causes camera shake | Low–Medium | Mount low on trunk, use rigid strap-saddle, prefer smaller trees |
| Theft | Low (course is access-controlled) but high-cost | Lock enclosures with small padlocks; signal-jammer detection is overkill but possible |
| Person detection misses swings | Medium | Field test specifically measures this; fallback is operator-triggered recording from a separate app |
| Person detection false positives (groundskeepers) | Medium | Add ROI tightening + minimum-dwell-time of 2 s; ignore detections during known maintenance windows from tee sheet |
| Battery dies before solar can recover | Low | 10× safety margin in the production power budget; backend alerts on `last_seen_at` stale |
| Lens fogs in dew / humidity | Medium | Anti-fog spray on inner enclosure window weekly during humid season |
| Insurance / liability if a ball hits a camera | Real but low-frequency | Course super and ops sign off on mount locations; cameras well clear of typical errant-ball zones |

---

## Decision points

1. **Field test commitment** — buy the ~$980 MVP kit and pick one
   par 3 to deploy for a single tournament day. Goal: real data on
   detection rate and image quality before scaling.
2. **Software build** — schedule the Pi capture agent + backend
   `Camera` plumbing (~3 weeks total) once the field test confirms
   the architecture works.
3. **Scale to full course** — only after the single-hole test
   produces clean clips. ~$4.8 k per 4-hole course, ~3-week install
   + maintenance training.

If revenue per clip clears ~$5 net of compute + share-API fees, a
4-hole production course pays back its hardware in 3–4 tournaments.

---

*Last updated when this doc was drafted. The backend reference
implementation and per-hole costs are correct against the codebase
state at that time; revisit when hardware prices move significantly.*
