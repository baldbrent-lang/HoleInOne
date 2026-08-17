import { Link, useParams } from "react-router-dom";
import { Brand } from "../components/Brand.jsx";

/**
 * Static legal + FAQ pages. DRAFT TEXT — a real lawyer must review this
 * before any public launch, and every [BRACKETED] value below has to be
 * filled in first.
 *
 * The hole-in-one prize is written as a CONTEST OF SKILL, not a
 * sweepstakes. The $20 buys the par-3 video system; making an ace is a
 * feat of skill, not a draw, so there are no "entries", no odds, and no
 * alternate free entry method. That framing has to stay consistent
 * everywhere or it stops being true: the moment the prize turns on
 * chance rather than the shot, it needs the whole sweepstakes apparatus
 * back (and, for a prize over $5,000, registration and bonding in FL
 * and NY).
 */

const DOCS = {
  faq: {
    title: "FAQ",
    sections: [
      ["What is GolfReelz?",
        "GolfReelz films your par-3 tee shots, adds tracer overlays, and emails the clips to you after your round. Golfers registered on the par-3 video system are also eligible for our $10,000 hole-in-one contest."],
      ["How much does it cost?",
        "$20 per round per golfer. That covers the par-3 video system in full — capture, processing, and delivery. Eligibility for the hole-in-one contest comes with being registered; there is no separate entry fee for it."],
      ["Is the $10,000 prize for real?",
        "Yes. Any ace you make on a camera-equipped par 3 while registered is eligible, subject to the Official Rules. We verify with the on-cup camera, so disputes are minimal. If approved, we pay within thirty (30) days of verification."],
      ["What if my clip doesn't show up?",
        "Tap the 'Flag' button on any clip in your gallery, or reply to the delivery email. Our ops team manually reviews flagged clips and either re-assigns or refunds."],
      ["Do I need to download an app?",
        "No. Everything is web-based. Register on the site, snap a selfie, get clips by email."],
      ["What's the selfie for?",
        "Your outfit is how we match shots to you in the footage. We don't run face recognition; we identify by clothing color and pattern. The photo is deleted 90 days after your last round."],
      ["Can I get a refund?",
        "Yes — if we don't deliver any matched clips after your round, we refund automatically within 5 business days. If matching was incomplete, we refund pro-rata."],
      ["Which courses are supported?",
        "Currently Maridoe Golf Club (Carrollton, TX) and Pebble Beach (CA). Want your course added? Email hello@golfreelz.com."],
    ],
  },

  terms: {
    title: "Terms of Service",
    sections: [
      ["Acceptance",
        "By registering for a round, you agree to these Terms and to our Privacy Policy. If you don't agree, don't use the service."],
      ["The service",
        "GolfReelz operates a par-3 video system at participating golf courses: we capture, process, and deliver video clips of your tee shots. Registering also makes you eligible for our hole-in-one contest (Official Rules separately published)."],
      ["Pricing & payment",
        "Each round registration is $20.00 USD, charged at registration via Stripe. Group registrations charge the lead golfer for all players. All sales final except as described under Refunds."],
      ["Refunds",
        "If we fail to deliver any matched clips for your round, we refund the registration in full automatically. If matching is partial (some par-3s missing), we refund pro-rata. Subjective dissatisfaction with clip quality is not a refund condition."],
      ["Video rights",
        "You grant GolfReelz a non-exclusive license to film, process, store, and deliver clips of your tee shots, and to use anonymized clips (no personally identifying info) for product marketing. You retain all rights to publicly share your own clips. We will remove clips on request within 30 days."],
      ["Acceptable use",
        "Don't impersonate another golfer, misrepresent a shot to claim a prize, or attempt to bypass clip-matching to claim someone else's shots. Violations void any prize eligibility and may result in account termination."],
      ["Disclaimers",
        "Service provided as-is. We don't guarantee 100% capture rate (cameras malfunction, weather happens). We aren't responsible for course conditions, your scores, or your slice."],
      ["Liability cap",
        "Our total liability for any claim is capped at the registration fee you paid for the round in question."],
      ["Changes",
        "We may update these Terms from time to time. We'll post the updated version here with a new effective date. Continued use means you accept the changes."],
      ["Contact",
        "hello@golfreelz.com"],
    ],
  },

  privacy: {
    title: "Privacy Policy",
    sections: [
      ["What we collect",
        "Name, email, mobile number, payment method (handled by Stripe — we never see your card), an outfit selfie, and the video clips of your tee shots. We also log basic technical info (IP, browser) for security."],
      ["How we use it",
        "To deliver your clips, charge your registration, contact you about your round (and only your round), verify hole-in-one claims, and improve clip-matching accuracy."],
      ["Selfies & biometric data",
        "We store an embedding (a numeric fingerprint) of your selfie to identify your outfit in footage. We do NOT run facial recognition. We delete both the selfie image and its embedding 90 days after your last round, or sooner on request."],
      ["What we don't do",
        "We don't sell your data. We don't share it with advertisers. We don't run AI training on your clips. We don't keep payment-card info on our servers."],
      ["Sharing",
        "With our service providers only: Stripe (payments), SendGrid/Twilio (email/SMS), our cloud hosting (Replit, AWS S3 for video), and the on-course video processing partner. All under contractual confidentiality."],
      ["Your rights",
        "Email privacy@golfreelz.com to: download all your data, delete your account + all clips/selfies, or opt out of marketing communications. We respond within 30 days."],
      ["Illinois (BIPA)",
        "Illinois residents: by registering you provide informed consent for our limited biometric processing as described above. Schedule of retention: selfie embeddings deleted 90 days post-round; raw selfie images deleted on the same schedule. We do not sell or trade biometric data."],
      ["California (CCPA / CPRA)",
        "California residents have the right to know, delete, correct, and opt out of any sale (we don't sell). Email privacy@golfreelz.com or call your designated request line."],
      ["Children",
        "Service is not directed to anyone under 18. We don't knowingly collect data from minors. Contact us if you believe a minor has registered."],
      ["Contact",
        "privacy@golfreelz.com"],
    ],
  },

  rules: {
    title: "$10,000 Hole-in-One Contest — Official Rules",
    sections: [
      ["What this is",
        "This is a CONTEST OF SKILL, not a sweepstakes or a lottery. The prize is won by making a hole-in-one — a feat of skill — and not by any drawing, random selection, or element of chance. Your $20 registration pays for the GolfReelz par-3 video system (capture, processing, and delivery of your clips). No part of it is an entry fee for this contest, and paying more does not improve your chances, because there are no chances to improve."],
      ["Eligibility",
        "Open to legal residents of the 50 United States and the District of Columbia, age 18 or older. Void where prohibited or restricted by law. GolfReelz employees, contractors, and their immediate families are not eligible."],
      ["Sponsor",
        "GolfReelz, Inc. (the 'Sponsor'), [SPONSOR ADDRESS]."],
      ["Contest period",
        "Runs from [LAUNCH DATE] until [END DATE] (the 'Contest Period'). Shots made outside this period are not eligible."],
      ["How to take part",
        "Register for a round on the GolfReelz par-3 video system at a participating course and play the camera-equipped par-3 holes. Registration must be completed BEFORE you tee off on the hole in question — a shot made before registering cannot be verified against a registered golfer and is not eligible."],
      ["Qualifying shot",
        "A qualifying shot is a hole-in-one made from the designated tee markers on a camera-equipped par-3 hole during the Contest Period, in the course of an ordinary round of golf, in accordance with the USGA Rules of Golf. The hole must play at least [MINIMUM YARDAGE] yards from the tee used. Practice swings, replayed shots, mulligans, and shots taken outside an ordinary round do not qualify."],
      ["Witnesses",
        "The shot must be witnessed by at least [NUMBER] other golfers aged 18 or older who are not immediate family of the claimant, and who are willing to confirm the shot in writing if asked."],
      ["Verification",
        "All claimed aces are reviewed by the Sponsor using cup-camera and tee-camera footage. Where a verification camera was not functioning, the claim is reviewed by an independent panel; where the evidence is insufficient, the claim is denied. By taking part you agree to abide by the Sponsor's verification decision, which is final."],
      ["Prize",
        "Ten thousand US dollars ($10,000.00), paid to the verified winner within thirty (30) days of verification. Limit one prize per person per Contest Period. The Sponsor may substitute a prize of equal or greater value if the advertised prize becomes unavailable. [NOTE FOR COUNSEL: confirm this matches the terms of the prize indemnity policy — insurers commonly set their own minimum yardage, witness, and advance-notice conditions, and any condition in the policy must also appear in these rules.]"],
      ["Claiming",
        "Verified winners are notified by email and asked to complete a short claim form confirming how to reach them. The Sponsor will never ask for bank or card details by email or on that form."],
      ["Taxes",
        "The winner is solely responsible for all federal, state, and local taxes on the prize. A 1099-MISC will be issued."],
      ["Publicity release",
        "By accepting the prize, the winner grants the Sponsor the right to use their name, likeness, and the video of the winning shot for promotional purposes without further compensation, except where prohibited by law."],
      ["Limitation of liability",
        "Participants release the Sponsor and its agents from any liability for claims arising out of participation. The Sponsor is not responsible for camera or technical failures, mis-captures, or any condition beyond its control. Where a shot cannot be verified because of such a failure, the sole remedy is the refund described in the Terms of Service."],
      ["Governing law",
        "These rules are governed by the laws of [STATE], without regard to conflict-of-law principles. Disputes shall be resolved in [COUNTY], [STATE]."],
      ["Winner's list",
        "For a list of winners, write to the Sponsor address above within 90 days of the end of the Contest Period."],
    ],
  },
};


export default function Legal() {
  const { doc } = useParams();
  const data = DOCS[doc];
  if (!data) {
    return (
      <div className="wrap">
        <Brand />
        <div className="card">
          <h2>Page not found</h2>
          <Link to="/">Go home</Link>
        </div>
      </div>
    );
  }
  return (
    <div className="wrap">
      <Brand />
      <div className="card">
        <h1 style={{ marginBottom: 16 }}>{data.title}</h1>
        <p className="small muted" style={{ marginBottom: 24 }}>
          Last updated 2026 — placeholder text. Operator should have counsel
          review before any public launch.
        </p>
        {data.sections.map(([heading, body]) => (
          <section key={heading} style={{ marginBottom: 18 }}>
            <h3 style={{ marginBottom: 6 }}>{heading}</h3>
            <p style={{ color: "var(--ink)", fontSize: "0.95rem" }}>{body}</p>
          </section>
        ))}
        <div className="divider" />
        <p className="small muted">
          Other docs:{" "}
          {Object.keys(DOCS).filter((k) => k !== doc).map((k, i, all) => (
            <span key={k}>
              <Link to={`/legal/${k}`}>{DOCS[k].title}</Link>
              {i < all.length - 1 && " · "}
            </span>
          ))}
        </p>
      </div>
    </div>
  );
}
