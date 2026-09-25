---
target: frontend of the app
total_score: 24
max_score: 40
na_heuristics: 
p0_count: 2
p1_count: 2
target_identity: "file:C:\\Users\\andre\\OneDrive\\Personal Projects\\typeid\\frontend\\src\\App.jsx"
target_fingerprint: "sha256:73ede5cfdade0a4e18d2d84fd7fa0aad2bbcfb45c5328550e763d4d177cbb9ae"
target_path: "C:\\Users\\andre\\OneDrive\\Personal Projects\\typeid\\frontend\\src\\App.jsx"
timestamp: 2026-09-24T23-37-49Z
slug: frontend-src-app-jsx
---
Method: dual-agent (A: design review · B: detector + browser evidence). Both drove the live app at 1440 and 390px. Better-evidenced than the 19/40 baseline, where A was source-only.

## Design Health Score

| # | Heuristic | Score | Key Issue |
|---|-----------|-------|-----------|
| 1 | Visibility of System Status | 3 | Strong in-flight feedback; the result never scrolls into view or announces itself |
| 2 | Match System / Real World | 3 | Good strapline, then "threshold 0.70 / below threshold / aalto_147862" at the payoff |
| 3 | User Control and Freedom | 1 | switchMode() wipes enrollment progress unconditionally, no confirm, no undo |
| 4 | Consistency and Standards | 3 | One coherent token system; "Map" is an <a> styled identically to the mode buttons |
| 5 | Error Prevention | 1 | Typed text never checked against the prompt; name collisions silently overwrite |
| 6 | Recognition Rather Than Recall | 3 | Prompt visible, name echoed, dots shown; Enter-to-submit undiscoverable |
| 7 | Flexibility and Efficiency | 2 | Enter-to-submit is the only accelerator; mobile tab order hits destructive controls first |
| 8 | Aesthetic and Minimalist Design | 4 | Brand-sampled palette, themed browser chrome, nothing unearned |
| 9 | Error Recovery | 2 | Declined path has the best copy in the app but renders zero candidates |
| 10 | Help and Documentation | 2 | No route from the demo to the CMC curve or Rank-1 evidence |
| **Total** | | **24/40** | **Acceptable** — up from 19/40 |

## Design Specificity Verdict

Fixed. Authored for keystroke biometrics in more than one place: ScanRings reusing the mark's arcs as the busy state, palette sampled from image.png, --color-caution existing because declining to guess is a correct outcome, confidenceBand() anchored to the backend threshold. The 19/40 run's biggest failure is closed.

Lapse: the result view. "Ranked candidates" + five floats is a leaderboard any search product could ship — the most important screen is the most generic.

Detector: clean, zero findings. Validated as non-no-op via a fixture with known anti-patterns, which it correctly flagged.

Measured contrast: every pair passes. Tightest 5.35:1 (the "match" band label); the flattened disabled CTA clears at 6.74:1. No horizontal overflow at either width. Zero console errors across both full flows.

## Rejected finding

B reported the focus ring as invisible (currentColor, 1.00:1) on the primary CTA. FALSE POSITIVE: Tailwind's `transition` utility includes outline-color in transition-property, so the ring fades in over ~150ms and B measured frame 0. After settling it resolves to #5dcaa5 at 9.03:1, verified with real Tab navigation.

The second half was real: border-radius: 4px in the :focus-visible rule overrode rounded-full and snapped every pill to a rectangle on focus. Fixed this run.

## Priority Issues

**[P0] The result never states a verdict.** Renders a heading and five decimals, never a sentence naming the person. Top row 17px vs 15px. No scrollIntoView, so on mobile the verdict lands ~y=640 of 844. Two simultaneous "strong match" rows with no tiebreak observed. Fix: lead with a verdict block, scrollIntoView on arrival, demote the rest to a disclosure. Command: bolder

**[P0] Silent destruction of work and of other people's data.** switchMode() calls resetEnrollment() unconditionally — tapping Identify after 4 of 5 rounds discards all four with no prompt. person_id: name.toLowerCase() + ON CONFLICT DO UPDATE means a second "Alex" silently overwrites the first Alex's template while reporting success. Fix: guard the reset, 409 on collision, add an enrolled-list with delete. Command: harden

**[P1] The transcription prompt is never enforced or measured.** Typing unrelated text against five prompts enrolled successfully and matched at 0.873. The README's positioning rests on transcription forcing rhythm over content; a reviewer can falsify it in 30 seconds. Fix: enforce with a live diff, or change the copy. Command: harden

**[P1] Mobile: header collision and sub-44px targets.** Tab pills 32px tall, sentence-refresh 36x36, "Change name" 99x19.5, "Show raw event log" 116x19.5 (both fail even 24x24). The "1 of 5" counter wraps at 390px and crowds "Change name". Enroll rounds don't blur the input, so the keyboard covers "Next sentence" four times. Command: layout

**[P2] No route from the demo to the evidence.** Nothing links to the CMC curve, 56.9% Rank-1, or the subject-disjoint protocol. No Open Graph tags, so a shared link renders as bare text. Command: clarify

## Persona Red Flags

Casey (mobile): header collision, three sub-44px targets, keyboard covers the CTA four rounds running, verdict below the fold.
Jordan (first-timer): handed decimals and aalto_###### identifiers with no explanation; follows the app's own "Switch to Identify" instruction and watches the confirmation vanish.
Sam (accessibility): the SUCCESS path is silent — setStatus(null) on successful identify and no live region on the ranked list, so a screen-reader user hears nothing. Failure paths announce; success doesn't. Reduced motion genuinely handled.

## Minor Observations

- Visibility implemented by string-prefixing "hidden " into className (App.jsx:549, 591, 597) — the one place styling reads as a hack.
- Disabled "Start enrollment" is visually near-identical to the ghostButton used for "Enroll another person".
- "Map" is an <a> with pillBase, reading as a third tab but opening a new tab.
- The identify result leaves the CTA enabled with the text still in the box, re-submittable for an identical score.
- textarea rows=3 leaves ~70px of dead space below the one-line placeholder.
- Copy nit: no time estimate on "5 short sentences"; "about 60 seconds" would cut round-2 abandonment.
