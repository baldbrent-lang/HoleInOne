# Battery power — wiring a camera to a 12V LiFePO4 pack

Build guide for the **battery-only** deployment: a 12V 100Ah LiFePO4
pack at the base of the tree, a DC-DC converter at the camera, and a
Raspberry Pi 5 in a weatherproof enclosure above it.

This is the no-solar variant. `field-deployment.md` plans for a 100W
panel and Anker power stations; this document is what actually gets
built when the panel isn't there and a pack is swapped or recharged
between sessions.

Written to record the *reasoning* as much as the steps, so when a number
here looks wrong later, the argument for it is next to it.

---

## The chain

The topology the capture agent already assumes, from
`pi-agent/agent/battery.py`:

```
battery -> fuse -> INA226 (optional) -> buck converter -> Pi
```

Everything below is that sentence, expanded.

---

## Decide this first: convert at the camera, not at the battery

**Run 12V up the tree and step down to 5V at the Pi.** Not the reverse.

Power lost in a cable is `I²R` — current *squared* times resistance — and
the voltage that drop eats comes out of whatever you were carrying. At
5V there is no headroom to lose.

Over a 50 ft run of 18 AWG:

| Carrying | Current | Cable loss | Voltage drop | Result |
|---|---|---|---|---|
| **12V up, convert at camera** | ~1.05 A | 0.76 W | 0.70 V | 12.8 → 12.1 V. Converter accepts ≥8V. Fine. |
| 5V up from a converter at the battery | ~2.5 A | 4.0 W | 1.0 V | 5.1 → 4.1 V. Brownouts. |

Same wire, same distance. One works; the other fails intermittently in a
way that reads like a software bug — random reboots, half-written clips,
and eventually a corrupt SD card.

This decision matters far more than wire gauge. Get it right and gauge
becomes almost irrelevant (see *Does thin wire cost me runtime?* below).

---

## Wire

**Requirement: sunlight-resistant, outdoor-rated, 16 AWG or heavier.**
16 AWG is a floor, not a target — anything thicker is fine.

The property that actually matters is UV resistance. Ordinary hookup
wire goes chalky and cracks within a season on a tree. Look for
`SUNLIGHT RESISTANT` or a PV/direct-burial rating printed on the jacket.

Options, best first:

| Wire | Verdict |
|---|---|
| **4mm² solar PV cable** (`H1Z2Z2-K`) | Best. ≈12 AWG, already red/black, double-insulated, rated for a decade outdoors. Cut the MC4 connectors off. Stiff — see below. |
| **14/2 landscape wire**, bulk | Clean second choice. Printed gauge, red/black conductors, no connectors to remove. ~$40 per 100 ft. |
| **Low-voltage landscape lighting cable** (e.g. Lightkiwi `ULEC 30V DIR-BUR`) | Works. Sunlight-resistant and 30V-rated. Two catches: often no printed gauge, and **both conductors are black** — see polarity below. |

**PV cable is stiff.** Use it for the battery-to-enclosure run; use
something thinner for the short hop inside the housing, where 4mm² will
fight for space and strain the converter's solder joints.

### Polarity on unmarked wire

Landscape cable frequently has two identical black conductors, marked
only by printing on one and a ridge on the other. **Do not trust the
markings.** Ring out the pair with a multimeter's continuity test — probe
one conductor at each end until it beeps — then put **red tape on both
ends of that same conductor**.

Do this before installation. Once the cable is zip-tied up a trunk there
is no way to work it out, and reversed polarity destroys the converter
instantly.

### Terminal sizes

The colour coding on crimp kits is by gauge, and the wrong one will not
close:

| Colour | Gauge |
|---|---|
| Red | 22–16 AWG |
| Blue | 16–14 AWG |
| **Yellow** | **12–10 AWG** (use for 4mm² PV cable) |

---

## Phase A — Bench build

**Do this indoors, at an outlet.** Solder-seal heat-shrink connectors
need a heat gun, and a heat gun needs AC power. You will not be running
one at a tee box. Arrive at the course with finished harnesses.

Repeat per camera. **Label each harness TEE or GREEN** with tape and a
Sharpie as you finish it. A swapped pair does not fail loudly — it
produces a clip that cuts to the "green" camera showing the tee box,
and you will spend an hour looking for a bug that is in the hardware.

### A1 — Cut the battery lead

Cut a length long enough to reach from the battery box on the ground to
the enclosure on the tree, **plus 3 ft of slack**.

If using PV cable, cut the MC4 connectors off both ends and keep them.

### A2 — Splice in the fuse holder

On the **red conductor only**, about 4 inches from the battery end, cut
it and splice the inline fuse holder into the gap — one heat-shrink butt
splice each side.

**Leave the fuse itself out.** It is the master switch for the entire
build: with no fuse in the holder, nothing downstream can be live no
matter what you have mis-wired.

### A3 — Terminate the battery end

Crimp ring terminals onto both conductors. **Measure the battery posts
first** — most 100Ah packs are M8, and a terminal that doesn't fit is
the one thing that can stop the whole install cold.

### A4 — Terminate the camera end

Fit the **female XT60**. Female on the supply side, so the live pins are
never exposed when it is unplugged.

12 AWG is a tight fit in an XT60 — strip short and tin the strands. If
it fights you, terminate at a screw block on the enclosure and run a
short 16 AWG tail to the converter.

### A5 — Build the converter pigtail

Fit the **male XT60** to the converter's input leads. **Red to IN+,
black to IN−.**

Check twice. Reverse polarity kills these modules instantly.

### A6 — Seal

Heat gun over every splice until the solder ring flows and the ends
seal.

### A7 — Bench test, before a Pi is anywhere near it

**This is the step that separates a dead converter from a dead Pi.**

1. Connect the harness to the battery (ordering per B4 — same rules on
   the bench).
2. Insert a **5A** blade fuse.
3. Mate the XT60.
4. Multimeter on **DC volts**: black probe on converter output ground,
   red on output positive.
5. **Expect 5.0–5.2 V.**

Anything else — 12V, zero, above 5.3V — unmate the XT60 immediately and
recheck. If the module has an adjustment pot and reads low, trim to
5.1V.

Only once this reads correctly, connect the Pi's USB-C and confirm it
boots. Then pull the fuse, disconnect, and pack the harness as a
labelled unit.

### Why a 5A fuse

The converter tops out at 5V × 5A = 25W out, which at 12V is roughly
**2.3A in**. A 5A fuse leaves headroom for inrush while staying far
below what 16 AWG can carry.

**The fuse protects the wire, not the Pi.** Do not fit a bigger one
because something blew — a blown fuse means find the fault.

---

## Phase B — Field install

### B1 — Mount the enclosure

Strap to the tree, no fasteners into bark. Aim the camera before
dressing any cable.

### B2 — Run the cable

Battery box on the ground at the base. Secure to the trunk with
**black UV-resistant zip ties** every 18–24 inches. White nylon goes
brittle and snaps after a few months of sun, generally while nobody is
on site.

Leave a **service loop** of 2–3 ft near the enclosure so the housing can
be brought to eye level for service instead of worked on overhead.

### B3 — Drip loop

Route the cable so it hangs **below** the gland and comes back up to
enter. Water runs off the low point instead of tracking along the jacket
into the housing. Six inches of wire; prevents the most common failure.

Tighten the gland onto the jacket, not the conductors. A flat landscape
cable will not seal in a round gland — use a short round pigtail through
the gland and put the splice inside the enclosure.

### B4 — Connect the battery

Fuse still out.

1. **Black** ring terminal → battery **negative (−)** first.
2. **Red** ring terminal → battery **positive (+)** second.

Roughly 8–10 Nm. Snug, not gorilla-tight.

**On disconnect, reverse it — positive off first.** If a tool slips
while you are on the last terminal, this ordering is what stops it
shorting to the battery case. A 100Ah LiFePO4 will deliver hundreds of
amps into a short: it will not shock you at 12V, but it will weld the
tool and can vent the cell. Rings and watches off.

### B5 — Power up

1. Insert the **5A** fuse.
2. **Re-check 5V at the converter output.** The run up the tree is new
   since the bench test; a pinched conductor or an over-crushed gland
   shows up here.
3. Connect the Pi's USB-C.
4. Confirm the Pi boots and appears on the Cameras page.

### B6 — Close up

Dress cables clear of the lid gasket. Seat the lid, tighten screws
evenly. **Silicone only where you drilled** — not on the gasket or the
lid seam, or the enclosure will never open again. Let it cure before
the housing goes out in weather.

---

## Optional — INA226 battery telemetry

Readings ride along on every heartbeat so the admin dashboard can show
per-camera battery state and warn before a pack dies. Best-effort by
design: no sensor just means heartbeats go out without battery fields.

Wire the breakout **in series on the positive lead**, between fuse and
converter (`IN+` toward the battery, `IN−` toward the converter), plus
four wires to the Pi header:

| INA226 | Pi 5 |
|---|---|
| VCC | 3.3V — pin 1 |
| GND | GND — pin 6 |
| SDA | pin 3 (GPIO2) |
| SCL | pin 5 (GPIO3) |

Enable I2C once per Pi:

```bash
sudo raspi-config nonint do_i2c 0
```

Set `battery.shunt_ohms` in `config.yaml` to match the board — **0.002**
for the 20A screw-terminal breakout, **0.1** for the blue CJMCU style.
The wrong value produces readings that look plausible and are wrong.

---

## Runtime

100Ah × 12.8V = **1280 Wh** nominal, against roughly 12W at the Pi
(≈13.3W after converter losses).

**≈ 90–95 hours per camera. Days, not hours.**

Treat as an estimate until the INA226 is reporting real numbers.

### Does thin wire cost me runtime?

Barely. Cable loss is `I²R`, and at ~1.05A the squared term stays small:

| Wire, 50 ft run | Cable loss | Battery draw | Est. runtime |
|---|---|---|---|
| Ideal (no loss) | — | 13.3 W | ~96 h |
| 14 AWG | 0.28 W | 13.6 W | ~94 h |
| 18 AWG | 0.76 W | 14.1 W | ~91 h |

Roughly 5 hours out of 90. Buy wire for durability and ease of
handling, not for gauge arithmetic.

### What actually determines runtime

- **Pi draw** is ~95% of the budget. The cable is ~5%. If runtime
  becomes a problem, look here.
- **Curfew** is the big lever. `pi-agent/agent/curfew.py` halts the Pi
  outside playing hours and wakes it by RTC — roughly doubles usable
  days per charge, and protects the SD card at the same time.
- **Heat.** LiFePO4 in a black box in direct summer sun loses little
  capacity but shortens pack life, and many BMS units cut out around
  140°F. That presents as the camera dying in the afternoon and
  working again at dusk — hard to diagnose from the app. **Shade the
  battery box.** North side of the trunk, off bare dirt.

### Charging

A 20A LiFePO4 charger (14.6V) takes ~5–6 hours from empty. Plan an
overnight charge rather than a top-up between rounds.

---

## Known gap: no low-voltage shutdown

`battery.py` **reports** voltage; nothing halts the Pi when the pack
runs down. `curfew.py` does clean *scheduled* shutdowns, not
voltage-triggered ones.

At 90+ hours of runtime the risk is low, but a hard power cut is exactly
what corrupts SD cards. Two mitigations, both cheap:

- **Enable curfew**, so the Pi is never running at the pack's lowest
  point.
- **Clone the SD cards before deployment.** Raspberry Pi Imager reads a
  working card to a `.img`; write that to a spare. A blank card is
  nearly useless in the field — it needs the OS, `pi-agent`, Wi-Fi
  credentials and that Pi's camera token re-entered. A cloned card is a
  30-second swap.

---

## Troubleshooting

For a camera that is **online and triggering but whose clip never
arrives**, follow [`camera-triage.md`](./camera-triage.md) — it starts
with `vcgencmd get_throttled`, which rules power in or out in two
minutes.

| Symptom | Check |
|---|---|
| No 5V, no lights | Fuse seated? Blown? XT60 fully mated? |
| 12V at the converter output | Converter dead or wired through — disconnect now |
| Pi boots then reboots | Voltage sagging — pinched conductor or bad splice |
| Nothing at all, wiring looks right | Polarity — converter may already be dead |
| Camera dies afternoons, works at dusk | Battery over-temperature — shade the box |
| Pi won't boot after a flat battery | SD card corruption — swap in the clone |

---

## Shopping list

Buy consumables locally rather than flying with them.

- Low-voltage landscape wire, **14/2**, 100 ft — ~$40
- **Black** UV-resistant zip ties, 8" and 11"
- Ring terminals sized to the battery posts (measure first — usually M8)
- Ratchet or cam straps, 6–8 ft (longer than what ships with trail-cam
  mounts)
- Butt splices in the matching colour (blue for 14 AWG, yellow for 12)
- Cable glands sized to the cable's actual OD
- Silicone sealant

### Tools

Screwdriver + precision set · **step drill bit** (a twist bit cracks
thin enclosure plastic; a step bit gives clean sized holes) · cordless
drill · wire strippers · flush cutters · 13mm socket or small adjustable
wrench for M8 posts · **multimeter** · heat gun · headlamp · Sharpie and
tape for labelling.

### Flying with this

- **Spare lithium batteries cannot go in checked baggage** — FAA rule.
  That covers CR2032 coin cells (the `UN3090` label) and any cordless
  tool battery pack. Cabin only; drill body checks fine.
- **Butane torches and canisters are banned from both** carry-on and
  checked. Do the solder-seal work at an outlet instead.
- **Silicone sealant** is a hazmat grey zone and cargo-hold pressure
  makes tubes ooze. Buy it there.
- **Carry on anything irreplaceable locally**: Pis, SD cards, camera
  modules, DC-DC converters. A delayed bag with tools in it costs a
  morning; one with the Pis in it costs the trip.
