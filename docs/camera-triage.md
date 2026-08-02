# Camera triage — a trigger fires but no clip arrives

Field runbook for the case where a camera is **online and triggering but
its clip never reaches the backend**. Written to be followed by someone
who wasn't in the debugging session — read it top to bottom, stop when
you get an answer.

**Current open case (Baldwin Links, hole 3):** the TEE camera (#1)
triggers, stays online, and its live view works intermittently. The GREEN
camera (#2) works perfectly. Every event parks with **TEE ✗ never
arrived · GREEN ✓ arrived** and nothing reaches Production.

---

## First: read the panel, don't guess

**`/admin/cameras` → Camera events** (bottom of the page). Tick
**"Only stuck / failed"**.

Each event shows two chips. They are the whole diagnosis:

| Chips | Meaning |
|---|---|
| TEE ✗ · GREEN ✓ | Tee Pi is failing to upload. **This is the open case.** |
| TEE ✓ · GREEN ✗ | Green never uploaded; the tee-only fallback should have produced anyway after 3 min |
| Both ✗ | Neither Pi uploaded — suspect the backend or the network, not one camera |
| Both ✓, stuck at "Both clips in" | Upload is fine; the produce queue is stalled |

Two traps worth knowing:

- **A green-only event can never produce.** The tracer needs the tee
  view, and the tee-only fallback filters on `tee_clip_filename`, so
  nothing sweeps it. It sits forever. Delete those; Re-process cannot
  help, because the tee clip does not exist anywhere.
- **"Processed" does not mean a clip was made.** A capture with no
  detectable swing is deliberately marked processed. Check Production
  for the actual row.

---

## Step 1 — Is it power? (2 minutes)

SSH into the affected Pi and run:

```bash
vcgencmd get_throttled
```

| Result | Meaning | Next |
|---|---|---|
| `throttled=0x0` | **Power is clean.** | Skip to Step 2. The battery is not the problem — don't swap it. |
| bit 0 set (`0x1`) | Under-voltage **right now** | Step 3 |
| bit 16 set (`0x50000`, `0x10000`…) | Under-voltage **has happened** since boot | Step 3 |

Confirm with:

```bash
dmesg | grep -i -E "under-?voltage|throttl"
uptime          # a recent boot means it power-cycled
```

**Run this before doing anything else.** If it returns `0x0`, power is
ruled out and a bigger battery would waste a day.

---

## Step 2 — What does the agent actually say?

```bash
journalctl -u golfreelz-agent -n 200 --no-pager
journalctl -u golfreelz-agent | grep -iE "upload|timeout|error|traceback" | tail -30
```

Look for: upload timeouts, HTTP errors, tracebacks, ffmpeg failures,
`No space left on device`.

Also check the scratch space — **`work_dir` defaults to
`/tmp/golfreelz-agent`, which is tmpfs (RAM)**:

```bash
df -h /tmp
free -h
```

If tmpfs is full, recordings fail to write and there is nothing to
upload. Fix by pointing `work_dir` at an SD-card path in `config.yaml`.

---

## Step 3 — Voltage under load, not at idle

This is the measurement that matters, and the one usually skipped.

Multimeter on **DC volts**. Probe the **converter's input** (the 12V
side) **while the Pi is uploading** — trigger a swing, or at minimum
catch it during boot.

| Input under load | Meaning |
|---|---|
| Holds 12 V+ | Battery and wiring are fine. The converter is the weak link — swap in the spare. |
| Sags below ~10 V | Power delivery problem: battery, or a resistive joint |

Then measure the **converter output** the same way. It must stay
**5.0–5.2 V** under load. A supply that reads 5.1 V at idle and sags to
4.6 V during upload looks perfect on a bench test and fails in the field.

### Why idle readings lie

`V_drop = I × R`. A 0.5 Ω bad joint drops 0.2 V at idle — invisible — and
1 V at 2 A, which browns out a Pi. **Same joint, same battery, opposite
outcome.** Upload is the peak-current moment: H.264 encode plus radio
transmit together.

---

## Step 4 — Swap the harnesses

The single most informative test, and it needs no instruments.

**Move the tee camera onto the green camera's harness** (battery, fuse,
cable, converter — the whole chain).

| Result | Conclusion |
|---|---|
| Tee camera now uploads fine | **It's the harness**, not the Pi and not the battery. Rebuild the tee harness. |
| Tee camera still fails | It's the Pi, its SD card, or its config — not power |

The green camera working perfectly on identical hardware is already
strong evidence that the *design* is sound and something specific to the
tee build is wrong.

---

## Decision tree

```
get_throttled == 0x0 ?
├── YES → not power. Go to journalctl (Step 2).
│         Likely: upload timeout, tmpfs full, agent crash.
└── NO  → it has browned out.
          Measure converter INPUT under load (Step 3).
          ├── holds 12V  → converter is weak. Swap the spare.
          └── sags <10V  → swap harnesses (Step 4).
                           ├── fixed  → tee harness is bad. Rebuild.
                           └── still  → battery genuinely undersized/flat.
```

---

## Known-good reference

- Converter output: **5.0–5.2 V**, at idle *and* under load
- Fuse: **5 A**, standard ATC (tan), within ~6" of the battery
- Pi 5 LED: red = powered but not booting (usually SD card); green
  activity = booting normally
- `vcgencmd get_throttled` on a healthy rig: `throttled=0x0`

## Report back

- `vcgencmd get_throttled` output
- Last 30 lines of `journalctl -u golfreelz-agent`
- Converter input and output voltage **under load**
- Whether the harness swap changed anything

That set is enough to name the cause without another site visit.

---

## Related

- [`battery-power-wiring.md`](./battery-power-wiring.md) — the full power
  build, terminal sizes, and the wiring order
- [`field-deployment.md`](./field-deployment.md) — hardware BOM and
  placement
