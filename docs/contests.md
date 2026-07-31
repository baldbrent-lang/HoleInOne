# GolfReelz contests — design and rules

Planning document for the prize games. Written before any of it is built,
so it records the *reasoning* as much as the rules — when a number here
looks wrong later, the argument for it is next to it and can be argued
with.

Nothing in this document is implemented yet. The "What exists today"
sections say honestly what the codebase already gives us and what has to
be built.

---

## The four games

| Game | Cadence | Scope | Prize currency | Needs |
|---|---|---|---|---|
| Hole-in-One | anytime | per course + network | cash (insured) | nothing new |
| Closest to the Pin | daily | **per hole** | free round | green-camera calibration |
| Shot of the Week | weekly | network-wide | cash | nothing new |
| Monthly Draw | monthly | network-wide | cash | round-counting per player |

Two of the four can be built today. The other two are gated on one piece
of work each.

---

## 1. Closest to the Pin

### Why per hole, not combined

This is the decision that matters most and it is easy to get wrong.

A 101-yard par 3 into a bowl green and a 185-yard par 3 over water
produce completely different distributions. On a combined leaderboard the
winner is decided by *which course they happened to play*, not by how
well they struck it — and the golfers at the harder course work that out
within a week and stop entering.

There is also a measurement argument. Closest-to-the-pin needs a distance
from the hole, and unless every camera is calibrated to the same scale on
the same green, those numbers are not on a common axis. Per hole, each
camera only has to be internally consistent, which is a far weaker
requirement and one that degrades gracefully.

And commercially: a pro shop will happily fund a prize for *their*
members on *their* hole. A network-wide closest-to-the-pin is entirely
your prize to pay for and means less to the player who wins it.

**A course with two camera'd par 3s runs two separate contests.**

### How the measurement works

Pixels are not yards, and the conversion changes across the frame — a
ball 30 feet from the pin but further from the camera covers fewer pixels
than one 30 feet away and near. So a flat "pixels per foot" is wrong
everywhere except one distance. What is needed is a mapping from the
image to the plane of the green.

**The flagstick does most of the work.** It is the one object in frame
with a known height (regulation is 7 ft), standing vertically at the
exact point being measured from. It gives us three things at once:

- the **origin** — the base of the pin *is* the hole
- a **scale reference** at a known location
- a **vertical**, which fixes the camera's orientation relative to the
  ground plane

**Step 1 — calibration, once per camera.** The cameras are fixed, so this
is set-and-forget, in the same spirit as the existing per-course
`tee_box_roi`. An operator marks four points on the green in a still
frame — the pin base plus three whose relative positions are known (the
corners of a paced-out rectangle will do) — and that yields a homography:
image pixel → position on the green in feet. Store it on the camera row.

A cheaper fallback if pacing is a nuisance: mark the pin base and the pin
top and assume the green is level. Seven feet of known vertical gives
scale. Less accurate than four ground points, but it takes ten seconds.

**Step 2 — find where the ball comes to rest.** The same machinery the
tee camera already uses, in reverse. MOG2 finds the ball arriving; the
discriminator is that it then *stops moving* — a small bright blob in
motion that becomes stationary for about a second. That is the
resting-ball signature `detect_swings_from_ball` looks for at the tee,
applied to the green.

**Step 3 — distance** is the ground-plane gap between the ball's base and
the pin's base, after the homography.

### Accuracy, and why it decides the rules

Realistically **±1–3 ft** at 1080p from a typical green-camera distance.

That number drives two rules rather than being a footnote:

- **Ties are real.** Two players inside a foot of each other cannot be
  honestly separated. Declare it a tie and split, or break it by earliest
  shot — decided in advance, in the published rules.
- **Do not over-report.** Displaying `7' 3"` implies inch accuracy we do
  not have. Show feet and inches but state the tolerance in the rules.

### Edge cases that must have an answer before launch

- **Ball finishes out of frame** (through the green, or short) — no
  measurement, no entry. The player should be *told* why, not silently
  dropped.
- **Ball behind the flag** — occluded, no measurement. Same treatment.
- **Two balls on the green at once** (a group playing together) — the
  measurement has to attach to the right player. Match by the clip's
  timestamp against the registration window, the same way clips are
  matched to participants today.
- **The pin moves.** Greenkeepers change pin positions. The calibration
  homography survives that (it maps the plane, not the pin), but the
  **origin** does not. Either re-mark the pin base daily, or detect the
  flagstick per shot.

### Rules to publish

- One entry per player per day per hole; best shot counts.
- Resets at **midnight in the course's local timezone**, not UTC. Getting
  this wrong puts a 7pm shot on tomorrow's board.
- The ball must come to rest on the green, in view of the camera.
- Distances accurate to ±1 ft; ties split.

### What exists today

The green camera and its clips exist. Ball-at-rest detection exists in
spirit (`detect_swings_from_ball`) but is aimed at the tee. **The
calibration screen does not exist and is the blocking piece** — nothing
else here can be tested without it. It is a small admin page: pick a
camera, grab a still, click four points, save the homography.

---

## 2. Hole-in-One

The simplest game to run and the best marketing asset the product will
have — an ace wall is the thing people screenshot and send to their
friends.

### How it works

Binary. No measurement, no calibration, and it is comparable across every
course without normalisation, which makes it the one game that works
network-wide as-is.

### Prize funding — do not self-fund

**Buy hole-in-one insurance.** It is a standard product: a modest premium
in exchange for the insurer covering the payout. It is how a $10,000 ace
can be advertised without carrying a $10,000 liability.

Two things to settle before printing a number:

- **Verification.** Insurers require proof. The cameras are ideal here —
  there is footage of every shot, which is a stronger claim record than
  most operators can offer. Worth leading with in the conversation.
- **Minimum distance.** Insurers set one. Check the par 3s qualify before
  advertising.

### Rules to publish

- Must be a registered, paid round with camera coverage.
- Verified from the footage; GolfReelz's determination is final.
- Insurer's terms apply (state them, including any minimum distance).

### What exists today

`VideoClip.ball_in_cup` is already a field, and the notification flow for
`hio_review` → `hio_confirmed` already exists and sends email. The wall
itself — a page listing confirmed aces per course and network-wide — is
new but small.

---

## 3. Shot of the Week

### Why this one earns its place

It needs **no measurement at all**, and it rewards the thing the product
is uniquely good at: a beautiful tracer. It also gives a reason to email
every player every week, which is worth more than the prize.

### How it works

Clips go into the broadcast channel; viewers vote; the most-voted clip of
the week wins. Announced Monday, promoted on social.

### The decisions that make or break it

- **Who can vote.** Open voting gets brigaded — whoever has the biggest
  group chat wins, every week. Options: one vote per registered player,
  or a shortlist curated by the operator with public voting on the final
  few. The shortlist is more work but produces better clips *and* removes
  the brigading problem.
- **Which clips are eligible.** All clips from that week, or only ones
  the operator tags? Tagging (`is_highlight` already exists) keeps
  quality up.
- **Consent.** The winning clip gets promoted on social media with the
  player's name on it. That needs to be covered at registration — a line
  in the terms, not an afterthought.

### Rules to publish

- Eligible: clips from rounds played Mon–Sun.
- Voting opens Monday, closes Wednesday; winner announced Thursday.
- One vote per player.
- Winner's clip may be used in GolfReelz marketing.

### What exists today

The broadcast channel, `is_highlight` tagging and the playlist all exist.
**Voting does not** — that is the new piece, and it is modest: a votes
table keyed by clip and voter, plus a results view.

---

## 4. Monthly Draw

### Why a draw rather than "most rounds wins"

The first instinct is to reward the player with the most sign-ups in a
month. That is the wrong shape, for three reasons:

1. **It is a spending contest.** At $20 a round, "most sign-ups" means
   "whoever paid the most wins money back". That reads badly and makes
   the prize feel like a rebate.
2. **Only one person can win, and it is the same person every month.** By
   month two everyone else has done the arithmetic and disengaged. A
   prize nobody believes they can win generates no behaviour.
3. **It rewards the player who least needs the nudge.** The heavy user is
   already coming back. The budget does more work aimed at the player
   who has been twice and might come a third time.

A draw keeps the identical frequency incentive — more rounds, more
entries — while leaving a real chance for the casual player, so they stay
engaged instead of tuning out.

### How entry works

**Automatic. One entry per round played.** No opt-in, no form, no box to
tick. Friction is what kills these; every extra step loses people who
would otherwise have been entered.

The rule, in one line for the site:

> **Every round you play in a calendar month = one entry into that
> month's draw.**

- Entries **reset monthly** — no carry-over. The point is a fresh reason
  to come back on the 1st.
- Drawn on the 1st, winner announced by email and on the leaderboard.

### The free entry route

A paid draw needs a **free alternate method of entry** — a page where
someone submits name and email once per month for a single entry, no
payment. Very few people use it; it costs almost nothing; it is what
keeps a paid draw on the right side of the line. Link it in the footer
and reference it in the rules.

**This is not legal advice.** The three skill games (closest to the pin,
hole-in-one, shot of the week) are on much firmer ground than a random
draw tied to a paid entry, and Texas has specific rules. Worth twenty
minutes with someone local before it goes live.

### Identity — the dependency everything else rests on

Entries attach to a **person**, not a round. Email is the natural key —
it is already collected and it is where clips are sent. Two consequences:

- **Normalise before counting** — lowercase and trim. `Ben@X.com` and
  `ben@x.com ` are one player.
- **A mistyped email is a second identity**, and it silently splits that
  player's entries. Worth a confirm-email step on the registration form.

This is the piece to decide early, because *every* loyalty mechanic
depends on it — the draw, the threshold reward, and any future "your
stats" page.

### The line that makes it work

Tell people their entry count, in the emails already being sent:

> *"That's your 3rd round this month — you have 3 entries in the June
> draw."*

That turns a passive mechanic into a reason to come back, and it costs
nothing: the gallery-ready email already goes out after every round. Same
line on the gallery page.

### What exists today

Participants, their emails and their rounds are all stored. Counting
rounds per normalised email per calendar month is a straightforward
query. **The free-entry form, the draw itself and the announcement are
new.**

---

## Prize budgeting

### The rule: a percentage of revenue, never a fixed number

A fixed prize means the first slow week eats the margin and a good month
underpays the thing driving it. Budget **~15% of revenue** and let the
amounts move.

### The frequency problem

The four games have wildly different costs because of how often they pay
out, per course:

| Game | Payouts/month |
|---|---|
| Closest to the pin | **30** |
| Shot of the week | 4 |
| Monthly draw | 1 |
| Hole-in-one | rare |

A $25 daily closest-to-the-pin prize is **$750/month** — the entire
budget at launch volumes, before anything else is paid, and it costs the
same whether five people played that day or fifty.

### The fix: change the currency, not just the amount

**Free rounds for the frequent prizes, cash for the rare ones.**

A free round has $20 of face value to the winner and costs close to
nothing — the camera was running anyway. It is also self-reinforcing: the
prize *is* another play, which is another draw entry, which is another
clip they share. Cash leaves the building; a free round comes back.

Pro-shop credit is the other good option for the daily prize, and the
course may co-fund it since it is spend in their shop.

### Numbers at 15% of revenue, one course, $20 a round

| Rounds/day | Monthly revenue | Prize budget | CTP daily | Shot of week | Monthly draw |
|---|---|---|---|---|---|
| 5 | $3,000 | ~$450 | free round | $25 ×4 | $250 |
| 15 | $9,000 | ~$1,350 | free round | $50 ×4 | $750 |
| 40 | $24,000 | ~$3,600 | free round + $25 | $100 ×4 | $1,500 |

At current volumes — a handful of players, $60 lifetime gross — **start at
or below the first row.**

### The copy line that saves you later

> *"Prize amounts shown are current and may increase as more players
> join."*

That allows raising them as a good-news announcement instead of being
locked into launch numbers.

---

## Cross-cutting rules

These apply to every game and should be stated once, prominently:

- **One entry per player per day** on the skill games — otherwise it
  becomes a contest of who hit the most balls.
- **Stated tolerances.** "Distances accurate to ±1 ft; ties split." You
  *will* have ties and you do not want to negotiate that with someone
  standing at the counter.
- **Verification is from the footage**, and GolfReelz's determination is
  final. The footage is the product's advantage — lean on it.
- **Marketing consent** for winning clips, captured at registration.
- **Local midnight**, per course, for every daily reset.

---

## Build order

Roughly cheapest-and-most-valuable first:

1. **Ace wall.** Nearly free — `ball_in_cup` and the notification flow
   already exist. Best marketing asset per hour of work.
2. **Monthly draw.** Needs identity normalisation and a round count; the
   free-entry form is a static page plus a table. The entry-count line in
   the existing emails is where most of the value is.
3. **Shot of the Week.** Needs a votes table and a results view. The
   channel and highlight tagging already exist.
4. **Closest to the Pin.** Blocked on the green-camera calibration
   screen, which should be built first and tested on a real shot before
   any of the contest logic is written.

---

## Open questions

- **Player accounts, or email-as-identity?** Everything loyalty-shaped
  depends on this and it should be settled before the draw ships.
- **Who funds which prize?** Course-funded pro-shop credit versus
  GolfReelz cash changes the economics materially and is worth agreeing
  with the first course before launch.
- **Hole-in-one insurance** — premium, verification requirements and
  minimum distance, before any number is advertised.
- **Shot of the Week voting model** — open vote or curated shortlist.
  Open voting will be brigaded.
