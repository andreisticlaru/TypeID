# Shift plan: transcription → free text, across sessions and keyboards

Status (2026-10-27): proposed, nothing started. Background, literature and sources: `paper-plan.md`.

## The question

A model trained on copied text (Aalto): does it still recognise people when they **compose freely**, **days
later**, on **another keyboard**? Measure each gap, then close it. That is the paper. The KVC submission
(`kvc-plan.md` Phase 2) is the credibility number beside it, not the topic.

## Why this, not more training

- More Aalto people and longer runs saturate (`best-model-plan.md`); nothing in the recipe moved cross-session.
- Architecture is the race Type2Branch / EERLoss already lead.
- Nobody has measured the copied → composed transfer for a deep model at this scale; the 8-week ageing study and
  our KeyRecs step show session drift is unsolved.

## Steps

| # | Step | Needs | Output | Decision |
|---|---|---|---|---|
| 0 | Email Buffalo (cubs.buffalo.edu) and Stony Brook authors; register for KVC | 15 min | Access in weeks | — |
| 1 | BB-MAS loader `data_v2/bbmas.py` (desktop only), like `data_v2/keyrecs.py`: raw events → (press, release) → `windows_from_keystrokes`. Layouts: `fixed` (enrol fixed, query fixed) and `fixed_free` (enrol fixed, query free). Tests like `test_keyrecs.py`. | Open download | Cache + split JSONs | — |
| 2 | **Measure the task gap**: `comparison.paired` / `comparison.levers` on BB-MAS `fixed` vs `fixed_free`, models v1, C1, C1_s1 (+ ensemble, k-pooling) | CPU/GPU minutes | `comparison/results/*_bbmas_*.json` | Gap > seed noise (~0.2 EER) and CI excludes 0? Yes → steps 3–5. No → task isn't the problem; skip to 6. |
| 3 | **Where the gap lives**: score with long intervals capped (IL/PL/RL at Aalto's 99th pct) and log-compressed, inside the model's forward pass (eval flag in `comparison/loaders.py`, extractor untouched). Control: same amount of random timing noise. | Hours | Gap with / without pauses | Cap closes > half the gap → pauses carry it, go to 4. |
| 4 | KLiCKe loader `data_v2/klicke.py`: check it has key-up times; fit pause distributions per boundary (within-word / between-word / sentence, ex-Gaussian). Also a within-essay identification set (first half enrol, second half query). | Open download | Pause stats JSON + composition test set | No key-up times → use pause stats only. |
| 5 | **Cognitive-pause augmentation** training run: `typenet/train.py --cpa`, inserting sampled pauses into raw pairs of the positive before extraction (pause placed after Space/punctuation using Aalto key identity at training time only; the model input stays timing-only). New checkpoint in `runs/`. One run + one seed. | GPU ~3 h ×2 | `runs/p7_cpa*` | Beats C1 on BB-MAS `fixed_free` and KLiCKe without losing > 0.2 on Aalto `unseen_nf`? |
| 6 | **Session back-end**: PLDA / WCCN on frozen embeddings, fitted on KeyRecs (dev) then the Buffalo dev half; calibrated log-LR. New variants in `comparison/levers.py`. | CPU hours | Cross-session EER vs cosine / s-norm | Beats s-norm on `kvc_cross` beyond CI? |
| 7 | **Three decisions from one score**: global-threshold EER (one-time), SPRT over 50-key windows → ANGA/ANIA (continuous), Cllr + Tippett (forensic), DIR@FPIR (open-set ID). | Days, CPU | Tables for the paper | — |
| 8 | **Final test, once**: Buffalo test half (task × session × keyboard factorial), SBU (compose vs retype same text), BB-MAS, KLiCKe. Models: C1, best of 5, + 6 back-end; TypeNet rebuild; Type2Branch from its code if it runs. | When data arrives | Paper results | — |
| 9 | Write: arXiv → IJCB or BIOSIG. | — | Paper | — |

GPU order (one job at a time): KVC run (`kvc-plan.md` Phase 2, optionally with EERLoss/ArcFace) → CPA run(s).
Steps 1–4 and 6–7 are CPU work and run alongside.

## Rules

- Features only through `features/extract.py`; augmentation edits raw (press, release) pairs *before* it,
  pause capping lives in the model, never a second extractor.
- Split by person. KeyRecs is now dev (used to choose fixes); final claims only on held-out Buffalo / SBU /
  BB-MAS / KLiCKe, scored once.
- Never touch `model/encoder_hard.pt`, `data/preprocessed/`, `data/split.json`, the galleries. New caches in new
  folders, checkpoints in `runs/`.
- Every comparison paired over people with bootstrap CIs (`comparison/paired.py`); ≥ 2 seeds for trained models.
- Content-leak test for CPA: the embedding must not predict sentence identity better than C1.

## Risks

- Buffalo / SBU access is the critical path. Fallback: BB-MAS (task) + KeyRecs (session) + KLiCKe (composition),
  gaps measured separately, not crossed.
- The task gap may be small (Killourhy & Maxion 2012). Still a finding: session is the real gap, and step 6 is
  the method.
- BB-MAS is one session, 117 people: small CIs need pooling (k) and all its free-text answers.
