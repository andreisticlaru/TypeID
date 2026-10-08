# Plan: controlled model study on TypeNet's split → v2 encoder

Branch: **`feat/models-v2`** (all work, files, training and commits happen here; never merge to `main`,
never push unless the user asks). Executor: Claude, **working alone** for hours at a time (see §7).

Tick boxes (`- [x]`) in this file as you go, and add one line under **Progress log** per finished step.
This file is the single source of truth for where the work stands.

**Two non-negotiables:**
1. **You work unassisted.** The user is not watching and will not answer questions mid-run. Use background
   jobs and watchers, recover from failures yourself, and keep moving through the steps (§7). No planned stops
   (the M0 calibration passed).
2. **The webapp must keep working exactly as it does now.** The user runs it from this same working tree. Do not
   change its behaviour, its model, its data or its processes (§1), and verify it at the end (Step D.4).

---

## 0. Goal in one paragraph

Find a model **better and/or cheaper than v1** (the backend's `model/encoder_hard.pt`), learning along the way
which ingredients matter. Models differ **one ingredient at a time**, on **TypeNet's own subject split and data
rules**, so every model is also comparable with TypeNet's *published* numbers (the only remaining difference
being TypeNet's key-code input, which can only help TypeNet). v1 is compared on people neither it nor the new
models trained on (Step A). The winners combine into **v2**, confirmed over 3 training seeds, trained to v1's
budget and measured under the live app's 25-keystroke floor. The current app (v1 model + webapp) stays untouched.
Revised 2026-10-04: see §4 for why.

---

## 1. Hard rules (read before every session)

**The working system must not change.** The user runs the webapp from this same working tree, so any edit here
is live in their app immediately.

| Never modify | Why |
|---|---|
| `model/` (code and every `*.pt`, especially `model/encoder_hard.pt`) | the backend serves `encoder_hard.pt` |
| `backend/`, `frontend/` | the live webapp |
| `eval/`, `data/split.json`, `data/preprocessed/` | v1's evaluation and its training cache |
| `data/build_cache.py` — **never run it** | it writes straight into `data/preprocessed/` and would overwrite v1's cache |
| `backend/gallery.db`, `backend/gallery-populated.db` | the user's enrolled people |

- **One sanctioned edit to an existing file:** `features/extract.py` gets a keyword-only argument
  `min_keystrokes: int = MIN_KEYSTROKES` on `windows_from_keystrokes` (Step 1.2). With the default, behaviour is
  byte-for-byte identical, so the backend is unaffected. It is guarded by tests and a live backend check. No other
  edit to `features/`.
- **Never reimplement the HL/IL/PL/RL math.** All features come from `features.extract.windows_from_keystrokes`
  (CLAUDE.md's critical invariant).
- **No key code** anywhere. Timing features only.
- **Import existing code, don't copy it** (`model.network.KeystrokeEncoder`, `model.triplet_sampler.TripletSampler`,
  `model.train.save_checkpoint` / `load_for_resume` / `to_tensors`, `eval.typenet_protocol.*`, `eval.rank_n.*`).
- **Processes:** the user's dev servers run on :8000 (uvicorn) and :5173 (vite). **Never kill by port or by
  image name.** Only stop PIDs you started, identified by their exact command line (they contain `typenet.train`
  or the run id). On Windows, `TaskStop` on a background shell can leave the `python.exe` child alive: verify with
  `Get-CimInstance Win32_Process -Filter "name='python.exe'"` and stop only matching command lines.
- **One GPU job at a time** (RTX A2000 laptop, 4 GB, shared with the user's backend). Two concurrent trainings
  thrashed memory (5 s/step) and one crashed with `CUDA error: an illegal memory access`. **User rule (2026-09-30):
  one GPU task at a time, evaluations included** — never run an evaluation next to a training.
- Commit locally at the end of each phase (message ends with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Never commit `*.pt`, caches, `gallery.db`,
  `identify_log.jsonl`, `TODO.md` or `.claude/settings.local.json`.

---

## 2. Context you need to get this right

### 2.1 TypeNet (Acien et al. 2021, arXiv 2101.05570v3) — verified quotes

- **Architecture (IV-B):** "two LSTM layers of 128 units (tanh). Between the LSTM layers, we perform batch
  normalization and dropout at a rate of 0.5 … each LSTM layer has a recurrent dropout rate of 0.2." A Masking
  layer handles padding. Output = 128-d embedding. **200,458 trainable parameters** (with key code; ours without
  it = 199,944, asserted in `typenet/test_network.py`).
- **Inputs (IV-A):** HL, IL, PL, RL **in seconds** + key code/255 (we drop the key code).
- **Sequence rule (IV-B):** "truncate the end of the input sequence when N>M and zero pad at the end when N<M",
  M = 50. One sequence per session, **no minimum length**.
- **Loss (Eq. 4):** `max(0, d²(A,P) − d²(A,N) + α)`, Euclidean d, **α = 1.5**. Random triplets, no mining.
- **Training (IV-D), verbatim:** "the best results for both models were achieved with a learning rate of 0.05,
  Adam optimizer with β1=0.9, β2=0.999 and ϵ=10−8, and the margin set to α=1.5. The models were trained for 200
  epochs with 150 batches per epoch and 512 sequences in each batch." → 30,000 steps.
- **Split (IV-D / V):** "we train the models using only the first 68,000 subjects from the Dhakal dataset" and
  "The remaining 100,000 subjects were employed only for model evaluation".
- **Protocol (V):** per test subject, 15 sequences: gallery = first 10 (G ≤ 10 for authentication), query = last 5;
  score = mean pairwise Euclidean distance. Authentication: **k = 1,000** subjects, EER per subject, averaged.
- **Published results, desktop, triplet loss, M = 50:** EER **5.4 / 3.6 / 2.2 / 1.8 / 1.6** at
  G = 1/2/5/7/10 (Table II); identification **Rank-1 67.4%** (Table V).
- **Resolved (Step 1.1, TeX source):** Table V caption is `background size $\mathfrak{B}=1$,$000$` (the
  "1,000,000" was a rendering artefact of `1$,$000`). Sec. VI-C: "background of 𝔅=1,000 subjects, k = 10,000 test
  subjects, G = 10 gallery sequences per subject, M = 50". Contributions list: "Rank-n identification rates using
  a background set of 1,000 subjects". Our evaluator draws 1,000 people and queries each against all 1,000
  galleries; 10 draws = 10,000 queries → matches 𝔅 = 1,000, k = 10,000. `--background` stays 1,000.

### 2.2 What the pilot on our old split already taught us (`typenet/`, `comparison/`, commit 762ffca)

| Finding | Evidence | Consequence for this plan |
|---|---|---|
| **lr 0.05 kills the network** in this implementation | loss spikes at step ~400; at 10k, mean \|weight\| 2.2 vs 0.16 at lr 1e-3, 89% of 2nd-layer units output a constant; Rank-1 1.0% | all runs use **lr 1e-3**; the paper recipe is not rerun |
| TypeNet at lr 1e-3 matches the paper | 68.3% Rank-1 at 30k (old split, with floor) | M0 is the calibration check |
| Semi-hard mining helps TypeNet | 74.2% (+6 pts) | M4 |
| **Hardest mining collapses TypeNet** | loss pinned at 1.4999 = margin, all embeddings within 0.004 of each other | M5/M6 test the cure |
| Why it collapses | hardest-mined loss at mining start: TypeNet 1.9× margin (batch 512), 4.7× from the semi model; v1 0.96× (batch 64). Collapse makes loss = 1.0× margin, so it *lowers* the loss whenever the ratio > 1 | log this ratio at every hardest-phase start (Step 1.6) |
| Speed | 0.95 → 0.16 s/step via `unbind` in the LSTM loop and one grouped pass (`groups=3`) for A/P/N | keep both; they are tested |
| Pilot comparison was on **our** split with **our** floor | 3,771 eligible test people; galleries overlap across draws | this plan fixes both |

Pilot checkpoints and logs are in `runs/pilot_old_split/`, their scores in
`comparison/results/pilot_old_split/` (both gitignored); pilot numbers stay in the README labelled as the
old-split pilot until Phase 4 replaces them.

### 2.3 Data facts

- Raw Aalto: `data/config.py:AALTO_RAW_PATH`, 168,595 `<id>_keystrokes.txt` files, ≤ 15 sessions each.
  `data.aalto_loader.iter_participant_files` / `load_participant_file` → `Session(participant_id,
  test_section_id, pairs)`.
- v1's cache (`data/preprocessed/`, read-only): **25-keystroke floor**, a session is split into several
  50-step windows, a leftover window under 25 is dropped; arrays `windows, mask, subject_ids, session_ids`
  (strings). Of train-subject windows, 8.2% are 2nd+ windows of a session; median window length 47.
- `windows_from_keystrokes(pairs)` gives N−1 feature vectors for N keystrokes. TypeNet rule for us =
  `windows_from_keystrokes(pairs[:51], min_keystrokes=1)` → exactly one window of 1–50 vectors per session
  (sessions with < 2 keystrokes have no features and are skipped).
- Existing helpers that hard-code v1's split/cache and therefore **must not be used for new runs**:
  `model.train.build_sampler`, `eval.rank_n.load_eval_data`. Build equivalents in `data_v2/` (Step 1.4).

### 2.4 Budget and timing

- Standard budget = **15.36M triplets** = 30,000 steps × 512 (TypeNet's). Batch-64 runs get 240,000 steps.
- Measured: TypeNet arch at batch 512 ≈ 0.16–0.20 s/step → ~1.5 h per 30k run. Batch-64 timings are unknown:
  measure in the smoke test (Step 2) and write them in the Progress log before launching.

---

## 3. Folder layout (new unless marked)

```
data_v2/                    split + TypeNet-rule cache + data loaders for any (split, cache)
  make_split.py                  data/split_typenet.json: first 68,000 subject ids (numeric order) → train
  make_unseen_split.py           data/split_unseen_both.json: v1 eval ∩ TypeNet eval = 10,001 people (Step A)
  build_cache.py                 data/preprocessed_typenet/: one window per session, first 50, no floor
  datasets.py                    make_sampler(split, cache), load_eval_data(split, cache)
data/split_typenet.json, data/split_unseen_both.json   (committed; small)
data/preprocessed_typenet/  (gitignored)
typenet/                    network ablation flags, cosine losses, train.py (--arch/--split/--cache/mining schedule)
comparison/                 evaluate.py: --split/--cache/--tag, Rank-1/5/20/50
runs/<run_id>/              (gitignored) log.txt, step<N>.pt, final.pt, eval.txt
runs/queue.sh, queue.txt    (gitignored) sequential job queue; runs/watch.sh watcher
comparison/results/<run_id>_<stem>[_<tag>].json   committed, one per evaluated checkpoint × test set
```

Result tags: none = TypeNet split, no-floor cache (paper view); `unseen_nf` / `unseen_f` = the 10,001-person
unseen-by-both set on the no-floor / floor cache (the v1 comparison).

---

## 4. Goal, yardsticks and decision rule (revised 2026-10-04)

**Goal:** a model **better than v1** (`model/encoder_hard.pt`) and/or **cheaper** (fewer triplets, GPU hours or
training people for the same accuracy), with every number also **comparable to TypeNet's published table**.

**Two yardsticks, both reported for every candidate:**

| View | Test set | Answers |
|---|---|---|
| Paper view | TypeNet split, no-floor cache, 100,593 people (no tag) | how we compare with TypeNet's 67.4% / EER 5.4 |
| v1 view | `split_unseen_both.json`, 10,001 people neither v1 nor any new model trained on, **no-floor and floor** caches (`unseen_nf`, `unseen_f`) | does it beat v1? The floor view is what the live app sees |

"Better than v1" = higher Rank-1 **and** EER G=1 not worse, on `unseen_nf` and `unseen_f`.
"Cheaper" = reaches v1's `unseen_*` numbers with fewer triplets (learning curve) or fewer GPU hours.

**Why the plan changed** (see Progress log 2026-09-30 → 10-02):
- Semi-hard mining is the dominant lever (v1-style model 54 → 79% Rank-1; TypeNet +6 in the pilot).
  Comparing ingredients under random negatives answers the wrong question (M1: L2 lost 20 pts without mining,
  yet v1, which is L2 + cosine, wins with mining). → **All remaining ablations use semi-hard mining.**
- Hardest mining collapsed a model with 9M triplets on 68k people but helped v1 after 19M on 151k. It is a
  late-training tool, not an ingredient. → **No hardest phase in ablations**; one hardest continuation is tried
  on the long v2 run only (Step C).
- M3–M6 as specified (random negatives / early hardest) are dropped. M3's question (BatchNorm) is folded into
  R1 vs R2 (v1's net has no BN); M5/M6's question (L2, batch 64 under mining) likewise.
- Old Phase 2 (M0-f, M7-f) is replaced by scoring every candidate on `unseen_f` and training v2-f (Step C).

**Schedule rule for every ablation:** same budget as TypeNet (15.36M triplets), random negatives for the first
40%, semi-hard for the rest (= M7's schedule up to its hardest switch). Seed 0 unless stated.

**Decision rule:** adopt a change only if Rank-1 (paper view) improves by **≥ max(2, 2 × |R1 − R1b|) points** and
EER G=1 is not worse by > 0.3. R1b (seed 1) measures training-seed noise; one-seed differences smaller than that
are noise. Otherwise keep the simpler option.

---

## 5. Model roster

All: Adam lr **1e-3**, betas (0.9, 0.999), eps 1e-8, checkpoint every 10% of steps.
"TN" = `TypeNetEncoder` defaults (input ÷1000, masked BatchNorm, LSTM 128 + recurrent dropout 0.2, BN,
dropout 0.5, LSTM 128, last-real-step readout, raw output, squared Euclidean, margin 1.5).
"v1 net" = `model.network.KeystrokeEncoder` (ms input, no BN, mean readout, Linear, L2, cosine margin 0.5).

### Done (random negatives unless stated; paper view)

| Id | Model | Rank-1 | Note |
|---|---|---|---|
| M0 | TN, random | 66.7% | calibration ✓ (paper 67.4) |
| M1 | TN + L2 + cosine, random | 46.0% | L2 hurts without mining |
| M2 | TN + mean readout, random | 69.0% | +2.3, one seed |
| M7 | v1 net, b64, random → semi @96k → hard @144k | 79.3% @144k, 12.5% final | collapsed under hardest |

### Step B ablations (all semi-hard after 40%, no hardest)

| Id | Run id | Model | Batch × steps | Flags | Question |
|---|---|---|---|---|---|
| **R1** | `p1_r1_nf` | v1 net | 64 × 240k | resume `runs/p1_m7_nf/step144000.pt`, `--semi-from 96000` | does semi keep improving past 79.3%? (identical to M7 up to 144k, so resuming saves ~1 h) |
| **R1b** | `p1_r1b_nf` | v1 net, **seed 1** | 64 × 240k | `--seed 1 --semi-from 96000` | training-seed noise for the decision rule |
| **R2** | `p1_r2_nf` | TN | 512 × 30k | `--semi-from 12000` | TN vs v1 net, both mined (TN's version of old M4) |
| **R3** | `p1_r3_nf` | TN + mean readout | 512 × 30k | `--readout mean --semi-from 12000` | does M2's gain survive mining? |
| **R4** | `p1_r4_nf` | best of R1–R3 | same | `--semi-from 0` | cheaper: is the random warm-up needed? |
| R5 (optional) | `p1_r5_nf` | TN + L2 + cosine | 512 × 30k | `--normalize --distance cosine --margin 0.5 --semi-from 12000` | only if Q2 (Step A) shows M0's norm carries little identity, i.e. L2 isn't the problem by itself |

### Step C v2

| Id | Spec | Seeds |
|---|---|---|
| **v2** | winner of Step B (decision rule in §4), 15.36M triplets | 3 (`p3_v2_s0/s1/s2`) |
| **v2-long** | v2 at 32M triplets (v1's budget): 62.5k × 512 or 500k × 64 | 1 (`p3_v2long`) |
| **v2-long-hard** | resume v2-long's final, hardest mining for +20% steps | 1 (`p3_v2long_hard`) — the only hardest test |
| **v2-f** | v2-long's spec on the floor cache (`--cache data/preprocessed`) | 1 (`p3_v2_f`) |

---

## Phase 1 build (done)

Steps 1.1–1.6 and Step 2 (smoke tests, regression, timing) are done; see Progress log. Code:
`features/extract.py` (`min_keystrokes`), `data_v2/`, `typenet/{network,losses,train}.py` + tests,
`comparison/evaluate.py`. Committed in 84a4966.

---

## Step A — yardstick against v1 and quick diagnostics (~1.5 h GPU, no training)

- [x] `data_v2/make_unseen_split.py` → `data/split_unseen_both.json`: **10,001** people (read-only use of `data/split.json`).
- [x] `comparison/evaluate.py`: `--tag` (result file `<name>_<tag>.json`, never overwrites) and Rank-50 (paper
      Table V reports Rank-1/50/100).
- [x] Score on `unseen_nf` and `unseen_f`: `model/encoder_hard.pt` (v1, 32M triplets, after hardest),
      `model/encoder_mine.pt` (v1 before hardest, 19.2M), M0 final, M2 final, M7 step144000, M7 final.
      Record eligible people for the floor set (~2.2k: fewer than 10 × 1,000, so draws overlap; say so).
- [x] Re-score M0 final, M2 final, M7 step144000 on the paper view (no tag): same numbers as before + Rank-50
      (paper: 99.8). Any change from the old JSON = nondeterminism bug; investigate.
- [x] Q2: `runs/diag_scripts/fast_eval.py --l2` on M0 final and M2 final. Big drop vs raw ⇒ TN stores identity
      in the embedding norm, explaining M1; R5 is then not worth running.
- [x] Progress log: v1-view table (model | triplets | train people | Rank-1 | EER G=1/5/10 | strict), plus the gap
      v1 − M7@144k. Does v1's hardest phase beat its pre-hardest checkpoint on unseen people?

## Step B — ablations under semi-hard mining

- [ ] Queue (after Step A): R1, R2, R3, R1b, each followed by its evaluations. Paper view on the 40%, 70% and final
      checkpoints (learning curve); `unseen_nf` + `unseen_f` on the final only.
- [x] R1 done  - [x] R1b done  - [x] R2 done  - [x] R3 done
- [x] Apply the decision rule (§4); pick R4's base; queue R4 (and R5 if Q2 says so). → base = v1 net; R4 not yet queued (user go-ahead).
- [ ] R4 done. Write the v2 spec + the reason for each choice in the Progress log.
- [ ] Commit (`feat: ablations under semi-hard mining`): code, results JSONs, plan.

## Step C — v2

- [ ] Implement any option v2 needs that isn't built yet (with a test).
- [ ] v2 × 3 seeds. Report mean ± std across **training** seeds (separate from the 10 evaluation draws).
- [ ] v2 beats every Step B model by more than its seed std? If not, say so plainly; no winner claim.
- [ ] v2-long (32M triplets): evaluate every checkpoint on `unseen_nf` → the triplet count at which it passes v1
      (the "cheaper" claim, if any). Still improving at the end?
- [ ] v2-long-hard: does hardest mining help a long-trained model (as it did for v1) or collapse it?
- [ ] v2-f: floor-cache training; does it beat v2-long on `unseen_f` (the live app's view)?
- [ ] Commit (`feat: v2 encoder`).

v2 is **not** wired into the backend. Swapping the served model is a separate decision for the user.

## Step D — test, document, hand over

### D.1 Final table
- [ ] One table with every final checkpoint: model | what changed | triplets | GPU h | train people | paper view
      (Rank-1/50, EER G=1/2/5/7/10, strict) | `unseen_nf` | `unseen_f`; rows include TypeNet published and v1.
- [ ] Caveat for the write-up: v1 trained on 151,734 people and 32M triplets, new models on 68,000; a new model
      beating v1 wins despite fewer people, not on equal terms.

### D.2 Tests
- [ ] All unit tests pass (`features.test_extract typenet.test_network typenet.test_losses data_v2.test_data_v2`).
- [ ] v1 regression numbers reproduce exactly (89.8 / 56.9 and 60.5 / 28.4 on v1's split and cache).
- [ ] `git diff main --stat -- model eval backend frontend data/preprocessed data/split.json` is **empty**.
      `features/extract.py` shows only the `min_keystrokes` change.

### D.3 Documentation
- [ ] README: replace the old-split pilot table with the final table. State plainly: same split, data rule and
      protocol as the paper except the key code; seeds; what each comparison shows.
- [ ] Keep the pilot findings (lr 0.05 saturation, hardest-mining collapse) as a short "what we learned first" paragraph.
- [ ] TODO.md item 4: mark done, summarize, list anything left open.

### D.4 Webapp check (the app must work with no problems)
- [ ] `python -m unittest features.test_extract` passes.
- [ ] Start your own backend on **port 8001** (`cd backend && ../.venv-model/Scripts/python -m uvicorn app.main:app --port 8001`,
      background, log to a scratch file). Check `GET /identify/rates` and one `POST /identify` succeed and that
      it loads `model/encoder_hard.pt`. Stop **that PID only**. Never touch :8000 / :5173.
- [ ] Commit (`docs: model study results`). Do not push unless asked.

---

## 6. Briefing the user (when they ask "update"/"status"/"brief")

Short, plain, no jargon walls. Always this shape:

```
Phase X, step Y — <one line: what's happening now>
Done since last update: <1–3 bullets>
Results so far: <small table, only finished runs: model | Rank-1 | EER G=1 | note>
Running: <run id>, step N/M, loss L, ETA hh:mm   (or "nothing running")
Problems: <none | one line each, with what you did about it>
Next: <one line>
```

Lead with the answer to "is it working?". Say plainly when something broke and why. Numbers only from
finished evaluations; never predict a running job's result.

---

## 7. Working alone

The user is not available during the run. Don't ask questions mid-phase, don't wait for approval between steps,
and don't end a turn with a job running unless a watcher or background waiter will wake you.

- **Launch** long jobs with `run_in_background: true` (Bash). Never foreground a run that takes more than a few minutes.
- **Watch** with one `Monitor` over all active logs, filtered to what you'd act on:
  `tail -q -n0 -f runs/*/log.txt | grep -E --line-buffered "Step [0-9]*0000/|HARD-START|COLLAPSED|saved|Traceback|Error|non-finite"`.
  Monitors expire after 30 min: re-arm. For "tell me once when X exists", use a background
  `until [ -f … ] || grep -q Traceback …; do sleep 30; done` instead.
- **Failure signatures** to catch: `Traceback`, `CUDA error`, `non-finite loss`, s/step suddenly ≥ 3× normal
  (another GPU job, or memory thrash: check `nvidia-smi`), loss stuck at the margin (collapse).
- **On a crash:** read the traceback, fix if it's a code bug (test first), resume from the last checkpoint with
  `--resume`, note it in the Progress log. A CUDA illegal-memory-access with a second GPU process running is
  environmental: make sure only one job runs, then resume.
- **When a result surprises you** (much better or worse than expected), check for a bug before believing it:
  wrong split, wrong cache, eval subjects leaking into training, wrong checkpoint.
- **One GPU task at a time** (user rule): no evaluation next to a training; queue evaluations between runs.
- **Tool quirks seen:** complex edits via bash heredoc/sed mangle quotes — write Python edit scripts to the
  scratchpad with the Write tool; the auto-mode classifier sometimes fails transiently — retry once, then do
  read-only work and come back.
- **Decisions you'd normally ask about:** make the conservative choice, write it and the reason in the Progress
  log, and carry on. Exceptions: anything that would touch the webapp or v1 (§1) — don't do it at all.

---

## 8. Progress log

(one line per finished step: date, step, key numbers)

- 2026-09-29 — Pilot on old split done (commit 762ffca): TN lr 0.05 collapses (1.0%), TN lr 1e-3 68.3%,
  + semi 74.2%, + hardest collapses; v1 hardest 89.8%. See §2.2.
- 2026-09-30 — 1.1: TeX confirms 𝔅 = 1,000, k = 10,000 test subjects, G = 10, 5 queries, mean pairwise
  Euclidean. Our 10 draws × 1,000 = same design; evaluator unchanged.
- 2026-09-30 — 1.2: `min_keystrokes` added; 17 extractor tests pass; own backend on :8001 served /identify/rates and
  POST /identify, still 422s a 2-keystroke query with the same message; 8001 PID stopped, identify_log.jsonl restored.
- 2026-09-30 — 1.3: split 68,000 / **100,593** (168,593 raw files on disk, not 168,595). No-floor cache: 2,528,895
  windows = exactly 15 sessions × 168,593 subjects, 0 sessions skipped, 43.2% full 50-step, median length 46, built
  in 6 min (12 workers). TypeNet-eligible test people (≥15 sessions): **100,593** no-floor vs **22,431** floor cache.
- 2026-09-30 — 1.4–1.6: `data_v2/` loaders + tests (4 pass); TypeNet flags (default outputs, BN buffers and
  state_dict keys byte-identical to before the change); cosine losses; new `typenet/train.py`; evaluator
  `--split/--cache`, results named `<run_id>_<stem>.json`. `comparison/loaders.py` needed no change (already
  builds from `config["arch"]` + `config["model"]`). 17 typenet tests pass.
- 2026-09-30 — Step 2 regression: modified evaluator (v1 split/cache defaults) reproduces encoder_hard 89.8 / 56.9
  and encoder 60.5 / 28.4 — every metric identical to the pilot JSONs.
- 2026-09-30 — Step 2 smoke: M0–M7 all OK (mining schedules compressed to 100/200 so every mode ran; HARD-START
  logged for M5/M6/M7). s/step: M0 0.69, M1 0.70, M2 0.69, M3 0.59, M4 0.70, M5 0.70, M6 (b64) 0.185, M7 (v1 b64) 0.048.
  **3.6× slower than the pilot (0.17)**: bench of HEAD vs new `network.py` = 0.616 s/step both, so not the code; GPU
  power/thermal-capped (~15–20 W, 1365/2100 MHz, throttle 0x24). Projected: M0–M5 ~5.8 h each, M7 ~3.2 h, M6 at
  240k ~12.3 h → **M6 halved to 120k × 64 (7.7M triplets), schedule 40k/80k** (same thirds); budget caveat for M6 vs M5.
  One evaluation (1 checkpoint, no-floor, 100,593 eligible) = 8.4 min → evaluate **final + 20/40/60/80 %** only
  (~40 min/run instead of ~1.3 h). Smoke M0 eval sanity: Rank-1 7.8% after 300 steps (chance 0.1%).
- 2026-09-30 — Step 3: queue launched (`runs/queue.txt`, one GPU job at a time, eval after each training).
  Order: M0 (calibration), M7, M1, M2, M3, M4, M5, M6. Watcher: `runs/watch.sh`.
- 2026-09-30 — **Reboot 07:26** (Windows update, build 26200→26300) killed the queue at M0 step 17,100. Resumed at
  11:13 from `step15000.pt` (Adam state restored; the sampler RNG restarts from seed 0, negligible). Loss continuous
  (0.154). After the reboot: **0.357 s/step** (was 0.70), so part of the throttling is gone.
- 2026-09-30 — M0 at step 20k runs **0.176 s/step** (pilot speed, throttling gone) → M6 restored to the full plan
  spec 240k × 64, schedule 80k/160k (≈3 h projected); the halving note above no longer applies.
- 2026-09-30 — **M0 calibration PASSED:** Rank-1 66.7% (paper 67.4%), 100,593 eligible. EER G=1/5/10 7.31/3.43/2.60
  vs paper 5.4/2.2/1.6 — authentication ~1–2 pts behind while identification matches; likely the missing key code
  (open question for the write-up). Strict Rank-1 28.4%.
  M0 learning curve (Rank-1): 6k 45.6 → 12k 57.7 → 18k 62.0 → 24k 65.7 → 30k 66.7 (flattening). M7 started 12:18.
- 2026-09-30 — M7 (v1 recipe, b64) 0.020 s/step. **HARD-START ratio 1.10** (v1 on old split: 0.96). Hardest phase:
  loss pinned at margin 0.499, spread 1.40 → 0.46 (148k) → 0.12 (158k) → plateau ~0.1 (168k) = **partial collapse**
  (guard needs <1% of start, so the run continues). step144000 (pre-hard) is in the eval set for comparison.
- 2026-09-30 — **M7 eval:** final (after hardest) **12.5%** Rank-1, EER 25.6/21.7/20.8 → collapsed. **step144000 (random
  + semi, 9.2M triplets) 79.3%**, EER 5.12/1.72/1.23, strict 40.4% — beats TypeNet published (67.4%, 5.4/2.2/1.6) on
  every metric with 60% of the triplets and no key code. Interpretation for Step 3.2: the collapse rejects **hardest
  mining on no-floor data**, not v1's recipe; final call after M5/M6 (which test whether L2+cosine / batch 64 help).
  M7 curve (Rank-1): 48k 50.5 → 96k 54.3 (random) → 144k 79.3 (semi: +25 pts) → 192k 16.1 → 240k 12.5 (hard).
  v1's edge comes from semi-hard mining; its random-only model (54%) is below M0 (67%). M1 started 14:01.
- 2026-09-30 — **M1 (M0 + L2 + cosine, margin 0.5): 46.0%** Rank-1, EER 8.19/4.59/3.67, strict 20.2% → −20.7 pts vs
  M0, behind at every checkpoint (12k 39.3, 18k 41.6). Likely: random negatives rarely violate a 0.5 cosine margin on
  unit vectors (loss ~0.07 vs M0 ~0.14), so few triplets carry gradient. Under random negatives normalization loses;
  M5 tests it under mining.
- 2026-09-30 — **M2 (M0 + mean readout): 69.0%** Rank-1 (+2.3 vs M0), EER 7.43/3.56/2.69 (G=1 +0.12), strict 29.1%.
  Passes the Step 3.2 rule (≥ 2 pts, EER ≤ +0.3) — narrowly, one training seed. Curve: 12k 63.5 → 18k 65.7 → 24k 67.3
  → 30k 69.0 (learns faster early: +5.8 over M0 at 12k). step6000 not scored (stopped during its eval).
- 2026-09-30 ~18:10 — **STOPPED by the user** after M2 (queue, eval and watcher killed; GPU idle). Nothing committed.
- 2026-10-01/02 — **Investigation of the open questions** (no Phase 1 runs resumed). Scripts in `runs/diag_scripts/`
  (`diag_hard.py`, `fast_eval.py`, `diag_active.py`; run from the repo root, gitignored).
  - **Q1 (hardest collapse) — the floor is NOT the cause; the starting model is.** Length check: no-floor cache has
    9.8% windows < 25 steps, otherwise same distribution as the floor cache (median 46 vs 47). Hard-start ratio over
    300 batches: M7@144k 1.027 no-floor / 1.030 floor; v1 `encoder_mine.pt` (pre-hard) 1.005 / 0.997. 15k-step
    hardest resumes (`runs/diag_*`): **E1** M7@144k on the floor cache → collapsed like the original (spread 1.40 →
    0.12 by 156k); **E2** v1 `encoder_mine.pt` on no-floor → survives (spread 1.40 → 1.30 at 315k, loss 0.487 <
    margin); **E3** same on floor → survives (1.33 at 310k; stopped there by the user). v1 entered hardest after
    200k random + 100k semi on 151,734 people; M7 after 96k + 48k on 68,000. ⇒ M5/M6/M7's equal-budget 40/20/40
    schedule starts hardest too early; the mean ratio is too close to 1 to be a reliable predictor.
  - **Q2 (M1)** — the logged explanation ("0.5 margin too easy") is doubtful: margin ÷ typical impostor distance is
    ~0.52 for M1 vs ~0.13 for M0, so M1's margin is relatively *harder*. New hypothesis: M0 carries identity in the
    embedding norm, which L2 discards. Not yet tested (`fast_eval.py --l2` on M0; `diag_active.py` on M0/M1).
  - **Q3 (EER gap)** — protocol ruled out: TypeNet's official README EER code (distance grid, 100−x flip) is
    equivalent to ours (only difference: fixed impostor sequence #11 vs our random one); identification text (Sec.
    IV-F) matches ours. Remaining difference is the key code (not testable under our rules). Paper Table V for
    triplet: Rank-1/50/100 = 67.4/99.8/99.9; our Rank-50 not yet computed (`fast_eval.py` needs a Rank-50 line).
  - **Q4 (M2 seed)** — not started: needs M0 + M2 seed 1 (~1.5 h each unthrottled, ~5 h throttled).
  - GPU throttled again during this session (630 MHz, 16 W; v1 b64 0.06 s/step instead of 0.02).
- 2026-10-02 — Found a fair test set for v1 vs new models: 10,001 people held out by both splits → Step 4.1b
  (user: run it later).
- 2026-10-02 00:35 — **STOPPED by the user** after `runs/diag_v1_floor/step310000.pt`. GPU idle, nothing running.
- 2026-10-04 — **Plan revised with the user** (§4): goal = better and/or cheaper than v1 + comparable to TypeNet. All
  remaining ablations use semi-hard mining (R1–R5), no hardest phase except one test on the long v2 run; M3–M6 and
  the old Phase 2 dropped; fair v1 test set (Step A) moved to the front. Built `data/split_unseen_both.json` (10,001
  people); evaluator gets `--tag` and Rank-50.
- 2026-10-04 — **Step A done.** Re-scored paper-view JSONs identical (only Rank-50 added: M0 99.6, M2 99.6, M7@144k
  99.9; paper 99.8). v1 view (10 draws; `unseen_f` has only 2,238 eligible people, so draws overlap):

  | Model | Triplets | Train people | unseen_nf Rank-1 / EER G=1/5/10 / strict | unseen_f Rank-1 / EER G=1/5/10 / strict |
  |---|---|---|---|---|
  | v1 `encoder_hard` (after hardest) | 32M | 151,734 | **89.1** / 3.73/0.99/0.68 / 52.8 | **89.0** / 3.15/0.86/0.56 / 56.3 |
  | v1 `encoder_mine` (before hardest) | 19.2M | 151,734 | 81.6 / 4.83/1.56/1.08 / 43.4 | 81.6 / 4.08/1.40/0.94 / 47.0 |
  | M7 @144k (v1 net, semi) | 9.2M | 68,000 | 79.7 / 5.28/1.75/1.27 / 40.9 | 80.0 / 4.70/1.61/1.08 / 43.7 |
  | M2 (TN + mean, random) | 15.4M | 68,000 | 69.2 / 7.27/3.39/2.55 / 30.2 | 71.8 / 6.67/3.09/2.31 / 31.1 |
  | M0 (TN, random) | 15.4M | 68,000 | 66.7 / 7.11/3.22/2.42 / 28.1 | 68.6 / 6.65/2.90/2.23 / 30.1 |
  | M7 final (collapsed) | 15.4M | 68,000 | 12.7 | 14.6 |

  Readings: unseen_nf agrees with the paper view (M0 66.7 both, M7@144k 79.7 vs 79.3), so the 10k set is a sound
  yardstick. v1's hardest phase is worth **+7.5 pts** on unseen people → hardest mining is v1's edge, but only
  after a long semi-hard phase (Step C tests it). M7@144k is 1.9 pts behind v1-before-hardest with half the
  triplets and 45% of the people. Floor vs no-floor test data barely moves v1-net models (±0.3), TN gains ~2–3.
  **Q2:** L2 on M0's raw embeddings: 66.7 → 63.5 (−3.2); M2: 69.0 → 60.6 (−8.4). Far less than M1's −20.7, so
  the norm is not the main carrier of identity; M1 lost by how it trained, not by L2 itself → **R5 queued** (after R1b).

- 2026-10-07 — **R1 (v1 net, semi to 240k, 15.4M triplets): 82.1%** paper view (R50 99.9), EER 4.61/1.42/0.98,
  strict 44.2; curve 144k 79.3 → 168k 79.3 → 240k 82.1, no collapse. `unseen_nf` **82.7** (EER 4.66/1.42/1.00),
  `unseen_f` **82.9** (EER 4.01/1.25/0.84). Beats v1-before-hardest (81.6, 19.2M, 152k people) with 80% of the
  triplets and 45% of the people; v1 final (89.1) still +6.4. Queue resumed 19:05 (R2, R3, R1b, R5).

- 2026-10-07 — **Audit of R1's claims (subagent, read-only, CPU re-run reproduced the JSONs exactly).** No leakage,
  no metric bug: split disjoint (68,000 / 100,593, all 168,593 files); sampler filters anchors, positives and
  negatives; checkpoint configs record the TypeNet split/cache; exactly 15 one-window sessions per eval person; metrics
  correct (per-person EER = paper's method, optimistic vs a global threshold). Caveats for the write-up: no validation
  split (all choices made on test numbers, < 1 pt effect); 10 draws cover 9,559 distinct people (unseen_nf 6,471,
  unseen_f 2,234 → std understates uncertainty); M0 calibration matches Rank-1 but its EER is ~2 pts worse; "first
  68,000" = numeric-id reading. Verdicts: **R1 > TypeNet published: supported** (+14.7 Rank-1, better EER at every
  G). **R1 > v1 pre-hardest: provisional** (+1.08 paired, 9/10 draws, one seed, below the 2-pt rule) — wait for R1b.
  Side finding: `--resume` reseeds numpy with `--seed`, so R1's steps 144k–240k replayed the triplet draws of steps
  1–96k (fewer unique triplets; not leakage, if anything a handicap). **Fixed:** a resume reseeds numpy with
  seed + start_step (`typenet/train.py`); fresh runs unchanged.

- 2026-10-07 — **R2 (TN + semi from 12k): 71.8%** paper view (R50 99.7), EER 6.88/2.99/2.18, strict 32.4; curve
  12k 57.7 (= M0's 12k exactly: deterministic) → 21k 69.4 → 30k 71.8. `unseen_nf` 72.2. Semi adds +5.1 to TN
  (pilot +6); v1 net under the same budget and mining (R1) is **+10.3** → v1 net is v2's base. GPU throttled again
  late in R2 (0.44 s/step, 1035 MHz, ~20 W; nothing else on the GPU).

- 2026-10-08 — **R3 (TN + mean readout + semi): 76.4%** paper view (R50 99.7), EER 7.07/3.17/2.34, strict 34.2; curve
  12k 63.5 (= M2's 12k) → 21k 73.4 → 30k 76.4. `unseen_nf` 76.1, `unseen_f` 77.6 (R50  99.9). Mean readout under mining: **+4.6 vs R2**
  (+2.3 without mining), EER G=1 +0.19 (within 0.3) → passes the rule pending R1b's noise; still −5.7 vs R1.

- 2026-10-08 — **R1b (R1's recipe, seed 1, fresh run): 82.3%** paper view (R50 99.9), EER 4.44/1.37/0.96, strict 45.0;
  curve 96k 52.8 (M7 seed 0: 54.3) → 168k 80.7 (R1: 79.3) → 240k 82.3. `unseen_nf` 82.3 (EER 4.50), `unseen_f`
  83.2. **Seed noise |R1 − R1b| = 0.2 pts** → decision threshold stays 2 pts. The resume-replay bug did not
  hurt R1 (fresh R1b lands at the same place). Both seeds beat v1-before-hardest on unseen_nf (82.7 / 82.3 vs 81.6).

- 2026-10-08 — **R5 (TN + L2 + cosine 0.5 + semi): 72.0%** paper view (R50 99.7), EER 7.01/3.02/2.21, strict 33.5;
  curve 12k 39.3 (= M1's 12k) → 21k 67.7 → 30k 72.0. `unseen_nf` 72.7, `unseen_f` 74.8. vs R2 (raw output): +0.2 = seed
  noise → L2 is neutral under mining; M1's −20.7 was a random-negatives effect (mining recovers it: 46.0 → 72.0).
- 2026-10-08 — **Step B decision (rule: ≥ max(2, 2 × 0.2) = 2 pts Rank-1, EER G=1 ≤ +0.3):**
  semi-hard mining (R2 vs M0 +5.1) ✓ adopt · mean readout on TN (R3 vs R2 +4.6, EER +0.19) ✓ · L2 + cosine on TN
  (R5 vs R2 +0.2) ✗ keep raw · **v1 net vs best TN (R1 82.1/R1b 82.3 vs R3 76.4: +5.7, EER −2.5) ✓ → v2 base = v1 net
  + semi-hard** (v1 net already has mean readout and L2). Unexplained part of v1 net's edge over TN+mean (~6 pts): no
  BN, ms input, Linear head, dropout 0.2 — not separated (not needed for v2). Next: R4 (v1 net, semi from step 0)
  for the cheaper question, then Step C.

### Where to resume (handover)

**Status (2026-10-08):** Step A and Step B runs R1, R1b, R2, R3, R5 are done and evaluated (results above, JSONs in
`comparison/results/`). Queue empty, GPU idle. Committed and pushed on `feat/models-v2`. **v2 base = v1 net +
semi-hard** (Step B decision above). README rewritten (identification + authentication; the TypeNet study is
deliberately left out of it until Step C is done).

**Next steps, in order** (one GPU job at a time; append to `runs/queue.txt`, launch `bash runs/queue.sh` with
run_in_background, watch with `bash runs/watch.sh`; each training followed by its evals as in the R-runs' queue lines):
1. **R4 — cheaper?** v1 net, semi-hard from step 0, 15.4M triplets (~2 h):
   `typenet.train --run-id p1_r4_nf --arch v1 --batch-size 64 --steps 240000 --semi-from 0`.
   Eval paper view (step96000, step168000, final) + `unseen_nf` / `unseen_f` (final). Compare with R1/R1b (82.1/82.3):
   within 2 pts ⇒ the random warm-up is unnecessary and Step C uses `--semi-from 0`; worse ⇒ keep 40% random.
2. **Commit** Step B (`feat: ablations under semi-hard mining`) once R4 is in.
3. **Step C** (schedule from step 1 = "S"):
   - `p3_v2_s2`: v2 at 15.4M, `--seed 2` (~2 h). R1 = seed 0 and R1b = seed 1 already are v2 at this budget
     (if R4 changes the schedule, rerun seeds 0/1 with it instead). Report mean ± std over the 3 training seeds.
   - `p3_v2long`: 500k × 64 = 32M triplets (v1's budget), schedule S scaled (40% random = `--semi-from 200000`, or 0),
     `--save-every 50000` (~4–5 h). Evaluate every checkpoint on `unseen_nf`: when (if ever) does it pass v1's 89.1?
   - `p3_v2long_hard`: `--resume runs/p3_v2long/final.pt --steps 600000 --hard-from 500000 --save-every 20000`
     (+ same `--semi-from`) (~1–2 h). Watch HARD-START ratio and the COLLAPSED guard. v1 gained +7.5 from hardest.
     First resume since the seed fix: check its first log lines differ from a fresh run's.
   - `p3_v2_f`: the better of v2long / v2long_hard's recipe on the floor cache (`--cache data/preprocessed`).
   - Evaluate all on paper view + `unseen_nf` + `unseen_f`; commit (`feat: v2 encoder`).
4. **Step D:** final table, tests, README section on the TypeNet comparison (left out of README on the user's request until
   Step C is done), webapp check.

**Open questions for the write-up:**
1. Our EER is ~1–2 pts behind the paper at equal Rank-1 (M0) — key code? (protocol ruled out, Q3)
2. v1 net beats TN + mean readout by ~6 pts; the remaining differences (no BN, ms input, Linear head, dropout 0.2)
   are not separated.
3. v1's last 6.4 pts: hardest phase (+7.5 measured on v1), budget (32M) or 2.2× more training people? Step C's
   v2-long / v2-long-hard answer the first two; the people count stays untested.

**Environment:** pause Windows updates for long runs (an update reboot killed the queue once). GPU throttling comes
and goes (0.18–0.69 s/step for TN b512; 0.02–0.06 for v1 net b64). `runs/queue.pending.old` (old M3–M6) is
superseded.

### Results table (fill as runs finish)

| Run | Data | Triplets | Rank-1 | EER G=1/5/10 | Strict Rank-1 | Note |
|---|---|---|---|---|---|---|
| TypeNet published | their split, key code | 15.4M | 67.4% | 5.4 / 2.2 / 1.6 | — | reference |
| M0 p1_m0_nf | TN split, no floor | 15.4M | 66.7% | 7.31 / 3.43 / 2.60 | 28.4% | calibration ✓ |
| M7 p1_m7_nf final | TN split, no floor | 15.4M | 12.5% | 25.6 / 21.7 / 20.8 | 8.5% | collapsed in hardest phase |
| M7 @144k (pre-hard) | TN split, no floor | 9.2M | 79.3% | 5.12 / 1.72 / 1.23 | 40.4% | random + semi only |
| M1 p1_m1_nf | TN split, no floor | 15.4M | 46.0% | 8.19 / 4.59 / 3.67 | 20.2% | L2 + cosine hurts (random negatives) |
| M2 p1_m2_nf | TN split, no floor | 15.4M | 69.0% | 7.43 / 3.56 / 2.69 | 29.1% | mean readout, +2.3 (passes rule) |
| R1 p1_r1_nf | TN split, no floor | 15.4M | 82.1% | 4.61 / 1.42 / 0.98 | 44.2% | v1 net, semi to the end |
| R2 p1_r2_nf | TN split, no floor | 15.4M | 71.8% | 6.88 / 2.99 / 2.18 | 32.4% | TN + semi |
| R3 p1_r3_nf | TN split, no floor | 15.4M | 76.4% | 7.07 / 3.17 / 2.34 | 34.2% | TN + mean readout + semi |
| R1b p1_r1b_nf | TN split, no floor | 15.4M | 82.3% | 4.44 / 1.37 / 0.96 | 45.0% | R1, seed 1 |
| R5 p1_r5_nf | TN split, no floor | 15.4M | 72.0% | 7.01 / 3.02 / 2.21 | 33.5% | TN + L2 + cosine + semi |
| v1 encoder_hard (unseen_nf) | 10,001 unseen-by-both | 32M | 89.1% | 3.73 / 0.99 / 0.68 | 52.8% | the target (not paper view) |
