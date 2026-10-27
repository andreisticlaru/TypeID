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
