# Best-model plan: accuracy first, training cost no longer a constraint

Status (2026-10-27): proposed, nothing started. Decided with the user after the KeyRecs cross-session check
(see `kvc-plan.md`, "KeyRecs zero-shot"). Earlier goal was the most data-efficient recipe; the goal is now the
best model, whatever the training cost.

## Why more Aalto training alone won't do it

- More people: 115k vs 68k train people, KVC EER +0.02 (tie). Aalto saturates around 68k at this model size.
- More steps: long runs didn't beat 600k; C1 checkpoints 50k–final are flat cross-session (8.2–9.4).
- Aalto has one sitting per person, so it can't teach "same person, another visit". On KeyRecs, cross-session
  KVC EER is ~8.5 vs ~4.5 within one session, for every model.
- The efficiency study is itself a result: past ~68k people the recipe mattered (C beat the warm-up), quantity
  didn't.

## Levers, roughly by expected payoff

| # | Lever | What | Cross-session? | Cost |
|---|---|---|---|---|
| 1 | More typing per sample | Average several 50-key windows per query instead of one; in the app, ask for 2–3 sentences. TypeNet's results improve steeply with length. | Probably (averages out momentary state) | Minutes, no training |
| 2 | Ensembles | Average scores of trained models (C1, C1_s1, C2, v1). | Somewhat | Minutes, models exist |
| 3 | Score normalisation | Cohort / s-norm: rescale each comparison by the people's scores against a cohort. Helps one global threshold (the KVC metric). | Yes (corrects shifts) | Hours of code, no training |
| 4 | Bigger / better architecture | Wider or deeper LSTM, transformer. KVC winner Type2Branch is a two-branch model. Data is no longer the bottleneck; the model may be. | Not by itself | Days, many GPU hours |
| 5 | Key identity in the input | Model knows which key was pressed, compares like with like. Breaks the "no key identity" rule on purpose; risk of learning content, needs tests for that. | Unclear | Days |
| 6 | Multi-session data | Buffalo (3 sessions, 28 days, keyboard swaps; email Prof. Upadhyaya), Clarkson II, KeyRecs: train or fine-tune. Only lever aimed straight at cross-session. | Yes | Depends on access |
| 7 | Multi-visit enrolment | Enrol across 2+ visits, or adaptive gallery (add confident matches over time). | Yes | Product change, no training |

## Extra training data: free text, mobile (lever 6 in detail)

| Dataset | Text | People | Sessions | Access |
|---|---|---|---|---|
| Buffalo (SUNY) | Free text + transcription | ~148–157 | 3 over 28 days; keyboard swaps in the "rotation" subset | Request from the authors (cubs.buffalo.edu) |
| Clarkson II | Real free text, logged on own PCs in normal use | 103 | ~2.5 years, continuous | Request from the authors |
| BB-MAS (Syracuse) | Free + fixed text, desktop, phone and tablet | 117 | One 2–2.5 h session | Open, IEEE DataPort |
| Aalto mobile / KVC mobile | Transcription on phones | ~60k+ | 1 | Through KVC |
| KeyRecs (already used) | Transcription, 2×5 passages | 99 | 2, minutes apart | Open, Zenodo |

Access details for Buffalo, Clarkson II and BB-MAS not yet verified.

**Free text (desktop): yes, but for the sessions more than the free text.**
- What Aalto lacks most is positives from different sessions. Training anchor/positive pairs drawn from
  different sessions (Buffalo, Clarkson II) is the direct way to teach "same person, another visit".
- Free text adds variety (thinking pauses, composing vs copying), which should help robustness, and the live app
  could move towards free typing later.
- These sets are tiny (~100–150 people) next to Aalto's 68k: training on them alone would overfit. Mix them into
  Aalto training (oversampled, cross-session positives only) or fine-tune an Aalto model on them, never replace it.
- Same rules as Aalto: split by person, and keep the evaluation people out of training. Buffalo can't be both the
  training set and the final test: split its people, or keep KeyRecs `kvc_cross` as the test.
- BB-MAS is one session, so it can't teach cross-session; its value is free vs fixed text and cross-device for
  the same people. Open, so it's the one usable now.

**Mobile: low priority for this app.**
- Touchscreen typing is a different signal: tap hold times, almost no key rollover, autocorrect and swipe change
  the timing. TypeNet and KVC both treat desktop and mobile as separate tasks with separate models.
- The webapp is desktop. Joint training could spend model capacity on phone patterns and blur desktop ones; it
  doesn't teach cross-session either (one session per person).
- Possible use: Aalto mobile's ~60k people as extra pre-training scale, then fine-tune on desktop. Uncertain
  gain; try only after levers 1–4, and only if a phone mode in the app becomes a goal.
- Cross-device (enrol on desktop, identify on phone) is a separate, much harder problem; BB-MAS is the dataset
  for it if it ever matters.

## Results: levers 1–3 (2026-10-27, `comparison/levers.py`, results `comparison/results/levers_*.json`)

KVC-style EER, 5 enrollment sequences; k = sequences averaged per query (score level). Ensemble = C1_s1 + C1 + C2
+ v1, each model's distances scaled by its median impostor distance. s-norm cohort: 300 people (Aalto) / ~20
(KeyRecs), never scored. Paired 95% CIs vs the same model at k=1 in the JSONs.

| | Aalto `unseen_nf` (9,701) | KeyRecs within (79) | KeyRecs cross-session (78) |
|---|---|---|---|
| C1_s1 k=1 / 2 / 3 | 2.57 / 1.61 / 1.27 | 4.19 / 3.00 / 2.09 | 7.84 / 6.56 / 6.75 |
| v1 k=1 / 2 / 3 | 2.99 / 1.83 / 1.41 | 4.87 / 2.85 / 2.20 | 8.00 / 5.92 / 5.82 |
| ensemble k=1 / 2 / 3 | 2.35 / 1.45 / 1.15 | 4.00 / 2.65 / 2.08 | 7.61 / 6.25 / 6.34 |
| C1_s1 s-norm / ensemble s-norm | 2.53 / 2.36 | 4.63 / 4.10 | 7.22 / 6.78 |

- **More typing per query is the big lever.** k=2 cuts Aalto EER ~40%, k=3 ~50% (FNMR@1% 8.9 → 2.0 for C2).
  Cross-session it helps less (−1.3 to −2.2, ~20–27% relative): averaging doesn't remove the session step.
- v1 gains most from pooling: at k≥2 the v1-vs-C gap closes, and cross-session v1 k=3 (5.82) is best.
- **Ensemble:** −0.22 on Aalto at k=1 (CI −0.26, −0.19), stacks with pooling (1.15 at k=3). Noise-level on KeyRecs.
- **s-norm:** ~0 on Aalto (−0.04), noise within KeyRecs, −0.6 to −1.1 cross-session (CIs near zero). Only worth it
  for cross-session; check again on Buffalo.
- Product implication: the demo identifies from 1 prompt "for speed"; 2–3 prompts roughly halve errors.

## Order

1. Levers 1–3 first: existing models, minutes to hours, testable now on Aalto (`unseen_nf`) and KeyRecs
   (`kvc_cross` for cross-session, `within`).
2. Then 4–5, the research part, ideally once KVC data arrives (bigger, cleaner training pool).
3. 6 when Buffalo answers; keep Buffalo as the untouched final test if possible.

## Rules for this plan

- Score every candidate on both yardsticks: same-session (Aalto / KVC) and cross-session (KeyRecs `kvc_cross`).
  KVC rewards same-sitting performance; the live app needs cross-session. A change can favour one over the other.
- KeyRecs used to choose a fix stops being an honest final test: final claims need Buffalo or a part of KeyRecs
  set aside and not looked at until the end.
- The critical invariant still holds: features only through `features/extract.py`.
