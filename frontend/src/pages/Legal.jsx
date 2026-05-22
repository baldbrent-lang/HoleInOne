import { Link, useParams } from "react-router-dom";
import { Brand } from "../components/Brand.jsx";

/**
 * Static legal + FAQ pages. Boilerplate text — operator should have a real
 * lawyer review before any public launch. Especially the sweepstakes rules
 * (state-by-state requirements) and the privacy policy (BIPA in IL,
 * CCPA in CA).
 */

const DOCS = {
  faq: {
    title: "FAQ",
    sections: [
      ["What is GolfReelz?",
        "GolfReelz films your par-3 tee shots, adds tracer overlays + stats, and emails the clips to you after your round. Every registered round is also entered into a $10,000 hole-in-one sweepstakes."],
      ["How much does it cost?",
        "$20 per round per golfer. That covers all video capture, processing, and delivery, plus your sweepstakes entry. Optional alternate (free) entry available — see Official Rules."],
      ["Is the $10,000 prize for real?",
        "Yes. Each round you register includes one entry. Aces on any par-3 you played that round are eligible. We verify with the on-cup camera, so disputes are minimal. If approved, we cut a check within 7 days."],
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
        "GolfReelz captures, processes, and delivers video clips of your par-3 tee shots at participating golf courses. Each registration also entitles you to one entry in our hole-in-one sweepstakes (Official Rules separately published)."],
      ["Pricing & payment",
        "Each round registration is $20.00 USD, charged at registration via Stripe. Group registrations charge the lead golfer for all players. All sales final except as described under Refunds."],
      ["Refunds",
        "If we fail to deliver any matched clips for your round, we refund the registration in full automatically. If matching is partial (some par-3s missing), we refund pro-rata. Subjective dissatisfaction with clip quality is not a refund condition."],
      ["Video rights",
        "You grant GolfReelz a non-exclusive license to film, process, store, and deliver clips of your tee shots, and to use anonymized clips (no personally identifying info) for product marketing. You retain all rights to publicly share your own clips. We will remove clips on request within 30 days."],
      ["Acceptable use",
        "Don't impersonate another golfer, share gallery links to defraud the sweepstakes, or attempt to bypass clip-matching to claim someone else's shots. Violations void your sweepstakes entry and may result in account termination."],
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
    title: "$10,000 Hole-in-One Sweepstakes — Official Rules",
    sections: [
      ["No purchase necessary",
        "NO PURCHASE OR PAYMENT NECESSARY TO ENTER OR WIN. A purchase or payment will not increase your chances of winning."],
      ["Eligibility",
        "Open to legal residents of the 50 United States and the District of Columbia, age 18 or older at time of entry. Void where prohibited or restricted by law. GolfReelz employees, contractors, and immediate family are not eligible."],
      ["Sponsor",
        "GolfReelz, Inc. (the 'Sponsor'), [Sponsor address — fill in before launch]."],
      ["Sweepstakes period",
        "Entries accepted from [LAUNCH DATE] until [END DATE] ('Sweepstakes Period'). Entries received before or after this period are void."],
      ["How to enter",
        "Method 1 (paid): Register for a GolfReelz round at a participating course for $20. Each round gives you one (1) entry per par-3 you tee off. Method 2 (free / AMOE): Mail a 3x5 card with your name, mailing address, email, phone, date of birth, and the participating course you'll be playing to: GolfReelz Sweepstakes Entry, [P.O. Box], [City, State, ZIP]. Limit one (1) free entry per person per day. Free entries have equal odds of winning as paid entries."],
      ["Winning condition",
        "A valid entry wins the prize when, during the Sweepstakes Period, the entered golfer makes a hole-in-one on a par-3 hole at the participating course, captured on the GolfReelz cup-cam, and verified by the Sponsor as a hole-in-one in accordance with USGA Rules of Golf. The Sponsor's determination is final."],
      ["Prize",
        "Ten thousand US dollars ($10,000.00) paid by check to the verified winner within thirty (30) days of verification. Limit one prize per person per Sweepstakes Period. Sponsor reserves the right to substitute a prize of equal or greater value if the advertised prize becomes unavailable."],
      ["Verification",
        "All claimed aces are reviewed by Sponsor's verification queue using cup-camera footage and tee-camera footage. If any verification camera was malfunctioning, the claim is reviewed by an independent panel; in case of insufficient evidence, the claim is denied. By participating, you agree to abide by the Sponsor's verification decision."],
      ["Taxes",
        "Winner is solely responsible for all federal, state, and local taxes on the prize. A 1099-MISC will be issued."],
      ["Publicity release",
        "By accepting the prize, the winner grants Sponsor the right to use their name, likeness, voice, and biographical information for promotional purposes without further compensation, except where prohibited."],
      ["Limitation of liability",
        "Participants release Sponsor and its agents from any liability for claims arising out of participation. Sponsor is not responsible for technical failures, mis-captures, or any condition beyond its control."],
      ["Governing law",
        "These rules are governed by the laws of [State], without regard to conflict-of-law principles. Disputes shall be resolved in [County], [State]."],
      ["Winner's list",
        "For a list of winners, send a self-addressed stamped envelope to the Sponsor address above within 90 days of the end of the Sweepstakes Period."],
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
