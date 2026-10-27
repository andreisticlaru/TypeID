# KVC plan: pick the best training recipe, then enter the Keystroke Verification Challenge

Status (2026-10-23): Phase 1 run C1 training (`runs/p6_c_f`). Done: `data/split_kvc_scale.json`
(`data_v2/make_scale_split.py`; 115,120 train, 0 overlap with unseen_both, contains TN's 68k) and the KVC-style
metric in `comparison/paired.py` (`kvc_eer`, `kvc_fnmr1`; own rng so old metrics reproduce; 15 tests pass).
**Phase 1 done (2026-10-23). Winner: recipe C (semi-hard from step 0, hardest for the last 100k).** C3 skipped
(C2 did not beat C1 on the deciding metric). GPU idle. Phase 2 waits for the user's KVC registration.

### Phase 1 results (`comparison/results/paired_p6_{unseen_nf,unseen_f,paper_view}.json`, log `runs/p6_paired/`)

v1 / v2f_hard / TN_M0 reproduce the 2026-10-11 JSONs exactly. KVC-style = 5 enroll, 10 queries, 20 random
impostors, one global threshold.

| `unseen_nf` (10,001) | Rank-1 | EER G=1 | EER G=5 | KVC EER | KVC FNMR@1% | DIR@1% FAR |
|---|---|---|---|---|---|---|
| v1 | 88.48 | 3.69 | 1.01 | 2.98 | 10.39 | 14.42 |
| v2f_hard | 87.92 | 3.69 | 0.97 | 2.93 | 10.51 | 11.79 |
| C1 `p6_c_f` (68k) | 87.94 | 3.56 | 0.87 | **2.64** | **8.74** | 11.53 |
| C2 `p6_c_f_115k` (115k) | 87.97 | **3.42** | **0.84** | 2.66 | 8.82 | 13.69 |

Paired (95% CI): KVC EER C1 − v2f_hard −0.29 [−0.34, −0.24]; C1 − v1 −0.34 [−0.39, −0.29]; C2 − C1 +0.02
[−0.02, +0.07] (tie). Rank-1 C1 − v2f_hard +0.02 [−0.28, +0.30]; C1 − v1 −0.55 [−0.86, −0.26]; C2 − C1 +0.03
(tie). EER G=1 C2 − C1 −0.14 [−0.24, −0.04].
`unseen_f` (2,238): KVC EER v1 2.78, v2f_hard 2.73, C1 2.39, C2 2.42; C1 − v2f_hard −0.34 [−0.43, −0.23]; Rank-1
all within noise. Paper view (100,593): C1 88.22 Rank-1 (= v2f_hard 88.21), EER G=1 3.51 (−0.11), KVC EER 2.63
vs 2.91 (−0.28 [−0.30, −0.26]); TN_M0 KVC EER 5.93 (KVC's own TypeNet result: 6.76 with harder impostors).

Reading: dropping the random warm-up does not change identification but improves verification (KVC EER −10%
relative), at the same triplet budget. 115k vs 68k people: no gain on the deciding metric, a small per-person EER
gain. One training seed each; the hardest-phase seed std was 0.17 Rank-1, so the KVC gains (~0.3, tight CIs over
people) are the claim most at risk from seed noise only if seed spread on KVC EER is > 0.1 (not measured).

**Seed check (2026-10-26, `paired_p6_seeds_nf.json`, log `runs/p6_paired/seeds_log.txt`, `unseen_nf`):** the three
v2-long-hard seeds score KVC EER 3.02 / 3.29 / 3.05 (std ≈ 0.15), KVC FNMR@1% 10.96 / 12.48 / 11.03, Rank-1
87.23 / 87.07 / 87.41. A difference of two single-seed runs has noise ≈ 0.21, so C1 − v2f_hard −0.29 is ~1.4× seed
noise: likely real but not established. For C: C1 (2.64) is 0.38 below the best seed, and C2 (2.66, an
independent C run) lands at the same level; the four non-C runs span 2.93–3.29. Claim: recipe C improves
verification by roughly 0.2–0.3 KVC EER; identification unchanged. A second C1 seed would settle it.

**Second C1 seed (2026-10-27, `runs/p6_c_f_s1`, from scratch with `--seed 1`, otherwise identical to C1;
`paired_p6_c1_seeds_nf.json`, log `runs/p6_paired/c1_seeds_log.txt`):** KVC EER 2.56, KVC FNMR@1% 8.44, Rank-1
88.43, EER G=1 3.23. Paired: C1_s1 − v2f_hard KVC EER −0.37 [−0.41, −0.32]; C1_s1 − C1 −0.08 [−0.12, −0.04].
v2f_hard and C1 reproduce exactly. **Verdict: the verification win holds.** The two C seeds differ by 0.08 and
both sit 0.29–0.37 below v2f_hard (mean gain ≈ −0.33). Rank-1 differs by 0.49 between the C seeds, so full-run
seed noise on Rank-1 is ~0.5: C1's "0.5 behind v1" was within seed noise (C1_s1 88.43 vs v1 88.48).
Use the C recipe for Phase 2.
Paired against v1 (`paired_p6_v1_vs_c1s1_nf.json`): Rank-1 C1_s1 − v1 −0.05 [−0.36, +0.24] (tie), C1 − v1 −0.55
[−0.86, −0.26]; KVC EER C1_s1 − v1 −0.42 [−0.47, −0.37], C1 − v1 −0.34 [−0.39, −0.29]. So recipe C ties v1 on
identification within seed noise and beats it on verification in both seeds.

### KeyRecs zero-shot (2026-10-27): cross-session check

KeyRecs (Zenodo 7886743, CC BY 4.0): 99 people, own keyboards, online; 2 transcription sessions of 5 passages
each (~2,800 keys), session 1 the same passages for everyone, session 2 different ones. Whole study 14–56 min,
so session 2 is minutes later in the same visit, not another day. Loader `data_v2/keyrecs.py` rebuilds exact
(press, release) from the digraph columns (tests `data_v2/test_keyrecs.py`); 15 evenly spaced 51-key chunks per
person; `--background 99`. Results `comparison/results/paired_keyrecs_{within,cross,kvc_cross,kvc_cross_sweep}.json`.

| KVC EER | v1 | v2f_hard | C1 | C1_s1 |
|---|---|---|---|---|
| within (all from session 1) | 5.05 | 4.86 | 4.56 | 4.44 |
| kvc_cross (enroll s1, query s2) | 8.41 | 8.40 | 8.78 | 8.47 |

- Same session: same order as Aalto (C best), CIs ±0.6 so not significant alone.
- Cross session: EER roughly doubles for every model; all tie (C − v1 +0.06 / +0.37, CIs ±1). The mixed `cross`
  layout had suggested C worse (+0.9–1.1); that was an artefact of mixing same- and cross-session queries.
- C1 checkpoint sweep 50k–final on kvc_cross: 8.2–9.4, no trend. Nothing in the recipe moves cross-session.
- Drift is a step, not gradual: genuine/impostor distance ratio is flat ~0.66 through session 1 (across its 5
  passages), jumps to ~0.70 at the session boundary, flat after. Same for v1 and C1_s1. Coarse timing stats
  (tempo, pauses, overlap, Backspace rate) change no more across sessions than within one, so tempo
  augmentation would not target it.
- Implication: Aalto (one sitting per person) and KVC can't measure this. The live app enrolls once and
  identifies later, so cross-session is the number that matters there. Fix needs multi-session data (Buffalo:
  3 sessions, 28 days, keyboard swaps; access by emailing the authors) or system-side multi-session enrollment.

## Goal

1. Find the best training recipe with 1–2 more runs, scored so they are directly comparable with v1 and v2f_hard.
2. Train that recipe from scratch on KVC's development data and submit to KVC-onGoing (desktop task).

## What KVC is (desktop task)

- Organisers: BiDA Lab (UAM Madrid, TypeNet authors) + Fraunhofer IGD. Site: https://sites.google.com/view/bida-kvc,
  CodaLab: https://codalab.lisn.upsaclay.fr/competitions/14063,
  starter code: https://github.com/giuseppe-stragapede/Keystroke-Verification-Challenge.
- Data: Aalto transcribed sentences. Development set 115,120 people (with demographics), evaluation set 15,000
  hidden people.
- Protocol per evaluation person: 5 enrollment sessions, 10 query sessions. A query is scored against the 5
  enrollment sessions (averaged): 10 genuine + 20 impostor scores per person (10 from same age/gender group).
- Metrics: **one global threshold**: EER and FNMR at 10% / 1% / 0.1% FMR. Best published desktop: EER 3.33%,
  FNMR@1%FMR 11.96%. TypeNet on KVC: 6.76% global, 2.71% mean per-person.
- Submission: zip with `desktop_predictions.txt` (one score per comparison from their lists), built by their
  `generate_submission_files.py`.

## Phase 1: pick the recipe (our data, comparable with v1 and v2f_hard)

All runs: v1 network (`--arch v1`), batch 64, lr 1e-3, floor cache (`data/preprocessed`, read only), seed 0,
checkpoint every 50k. Only the listed difference changes.

| Run | Id | People | Recipe | Steps / triplets | GPU | Isolates |
|---|---|---|---|---|---|---|
| v2f_hard (done) | `p4_v2f_hard` | 68,000 (TN split) | random → semi @200k → hardest @500k | 600k / 38.4M | — | baseline |
| **C1** (running) | `p6_c_f` | 68,000 (TN split) | semi from step 0 → hardest @500k | 600k / 38.4M | ~3.2 h | the warm-up (one flag vs v2f_hard) |
| **C2** | `p6_c_f_115k` | 115,120 (KVC scale) | same as C1 | 600k / 38.4M | ~3.2 h | number of training people (vs C1) |
| C3 (optional) | `p6_c_f_115k_long` | 115,120 | semi from 0 → hardest at last 100k | 1M / 64M | ~5.5 h | longer budget (vs C2) |

C3 runs only if C2 beats C1, i.e. if scale still pays off and more training could too.

**C2's split** (new file `data/split_kvc_scale.json`; `split.json` and `split_typenet.json` untouched):
- train = TypeNet's 68,000 + 47,120 drawn at random (fixed seed) from TypeNet's eval people **minus the 10,001 in
  `split_unseen_both.json`** (pool: 90,592). So C2's training set contains C1's: the only change is more people.
- eval = the same 10,001 `unseen_both` people. Neither v1, C1, C2 nor v2f_hard trained on them.
- Cost: C2 and C3 can't be scored on TypeNet's paper view (they train on part of its test set). Paper-view
  numbers stay for C1 only.

**Scoring (one GPU job after training):** `comparison/paired.py` on `unseen_nf` and `unseen_f`, same settings
as the 2026-10-11 runs (10 repetitions, background 1,000, 2,000 bootstraps), models v1 (reference), v2f_hard,
C1, C2[, C3]. Sanity check: v1's and v2f_hard's numbers must reproduce the old JSONs exactly. C1 also on paper
view (TN_M0 reference).

**New metric needed: KVC-style global EER.** Unseen people have exactly 15 sessions, which fits KVC's 5 + 10:
first 5 = enrollment, last 10 = queries, 20 impostor queries per person from other people in the same
background, mean pairwise distance, **one threshold for all people**. Report EER and FNMR@1%FMR, bootstrapped over
people like the other metrics. Add to `paired.py` with a unit test. (KVC picks half its impostors from the same
age/gender group; Aalto has no demographics in our cache, so ours are random: our numbers will be optimistic vs
KVC's.)

**Decision rule:** winner = lowest KVC-style global EER on `unseen_nf`, paired against v2f_hard. If the 95%
interval includes 0, it is a tie and the simpler/cheaper recipe wins. Rank-1 and per-person EER are reported but
don't decide (KVC scores verification only).

## Phase 2: KVC submission

1. **User:** fill the KVC enrollment form and register on CodaLab with the same email. Download the Public Data.
   First check the CodaLab page that submissions are still open and how many are allowed.
2. **Loader** (`data_kvc/`): KVC raw files → (press, release) pairs → `features.extract.windows_from_keystrokes()`.
   No new feature code. Build a KVC cache in a new folder (never `data/preprocessed/`).
3. **Hold out 5,000** development people as our private validation set (fixed seed); train on 110,120.
4. **Train the winning recipe from a random start on KVC data only.** No v1/v2 weights, no Aalto caches: KVC's
   15,000 hidden people are Aalto participants, so anything trained on our Aalto data may have seen them.
5. **Validate** on the 5,000 with the KVC-style metric. Go/no-go: no collapse, global EER in the range Phase 1
   predicts.
6. **Score** KVC's comparison lists → `desktop_predictions.txt` → zip with their script → upload.
7. Optional: if the result is near the top, retrain on all 115,120 with the same recipe and resubmit.

Open choice for step 2: sessions longer than 50 keystrokes (TypeNet used 150 characters in KVC). Default: our
M = 50 windows averaged per session, as in the live system. Test 1 window vs all windows on the 5k holdout.

## Rules

- One GPU job at a time; runs in the background with watchers on checkpoints, HARD-START, COLLAPSED, crashes.
- If a hardest phase collapses (spread falls, loss pinned at the margin), the last semi-hard checkpoint is the result.
- Never touch `model/encoder_hard.pt`, `data/preprocessed/`, `data/split.json`, the galleries.
- Results JSONs to `comparison/results/`, checkpoints to `runs/` (gitignored).

## Time

Phase 1: C1 3.2 h (running) + C2 3.2 h + scoring ~1 h (+ C3 5.5 h). Phase 2: loader ~half a day of work, training
~3–5 h, validation and submission ~1 h.
