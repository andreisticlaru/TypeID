# Plan: controlled model study on TypeNet's split → v2 encoder

Branch: **`feat/models-v2`** (all work, files, training and commits happen here; never merge to `main`,
never push unless the user asks). Executor: Claude, **working alone** for hours at a time (see §6).

Tick boxes (`- [x]`) in this file as you go, and add one line under **Progress log** per finished step.
This file is the single source of truth for where the work stands.

**Two non-negotiables:**
1. **You work unassisted.** The user is not watching and will not answer questions mid-run. Use background
   jobs and watchers, recover from failures yourself, and keep moving through the phases (§6). The only planned
   stop is a failed M0 calibration check (Step 3).
2. **The webapp must keep working exactly as it does now.** The user runs it from this same working tree. Do not
   change its behaviour, its model, its data or its processes (§1), and verify it at the end (Step 4.4).

---

## 0. Goal in one paragraph

Train a small family of models that differ **one ingredient at a time**, on **TypeNet's own subject split and
data rules**, so that (a) every model can be compared fairly with TypeNet's *published* numbers (the only
remaining difference being TypeNet's key-code input, which can only help TypeNet), and (b) we learn *which*
ingredient makes a model good. Then combine the winners into **v2**, confirm it over 3 training seeds, and
measure it under the live app's 25-keystroke floor. The current app (v1 model + webapp) stays untouched.

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
data_v2/                    NEW  split + TypeNet-rule cache + data loaders for any (split, cache)
  __init__.py
  make_split.py                  data/split_typenet.json: first 68,000 subject ids (numeric order) → train
  build_cache.py                 data/preprocessed_typenet/: one window per session, first 50, no floor
  datasets.py                    make_sampler(split, cache), load_eval_data(split, cache)
  test_data_v2.py
data/split_typenet.json     NEW  (committed; small)
data/preprocessed_typenet/  NEW  (gitignored)
typenet/                    EXISTS  network gets ablation flags (defaults = paper); losses get cosine;
                                    train.py gets --arch/--split/--cache/mining schedule
comparison/                 EXISTS  evaluate.py gets --split/--cache; loaders.py knows arch "v1" and "typenet"
runs/<run_id>/              NEW  (gitignored) log.txt, step<N>.pt, final.pt
runs/queue.sh               NEW  (gitignored) sequential job queue
comparison/results/<run_id>.json   committed, one per evaluated checkpoint
```

Run ids: `p1_m0_nf`, `p1_m1_nf`, … ; `p2_m0_f`, `p2_m7_f`; `p3_v2_s0`, `p3_v2_s1`, `p3_v2_s2`, `p3_v2long`,
`p3_v2_f`. (`nf` = no floor / TypeNet rule, `f` = 25-keystroke floor = v1's cache.)

---

## 4. Model roster

All: Adam lr **1e-3**, betas (0.9, 0.999), eps 1e-8, seed 0 unless stated, checkpoint every 10% of steps.
"TN" = `TypeNetEncoder` defaults (input ÷1000, masked BatchNorm, LSTM 128 + recurrent dropout 0.2, BN,
dropout 0.5, LSTM 128, last-real-step readout, raw output, squared Euclidean, margin 1.5).

### Phase 1 — which ingredient matters (TypeNet data rule: no floor, first 50)

| Id | Builds on | One change | Batch × steps | Negatives | Question |
|---|---|---|---|---|---|
| **M0** | — | TN as published, lr 1e-3 | 512 × 30k | random | reference; must land near 67.4% (calibration) |
| **M1** | M0 | L2-normalized output + cosine distance, margin 0.5 | 512 × 30k | random | does v1's output/distance help? |
| **M2** | M0 | masked mean readout instead of last step | 512 × 30k | random | does v1's readout help? |
| **M3** | M0 | no BatchNorm (both layers removed) | 512 × 30k | random | is BN useful? |
| **M4** | M0 | semi-hard mining | 512 × 30k | random ≤10k, semi after | how much does mining add to TN? |
| **M5** | M1 | semi then hardest | 512 × 30k | random ≤10k, semi ≤20k, hard after | does normalization stop the collapse? |
| **M6** | M5 | batch 64 | 64 × 240k | random ≤80k, semi ≤160k, hard after | smaller pool → safer/better mining? |
| **M7** | — | v1 recipe: `KeystrokeEncoder` (ms input, no BN, mean readout, Linear, L2, cosine margin 0.5, dropout 0.2) | 64 × 240k | random ≤96k, semi ≤144k, hard after (v1's 40/20/40 %) | v1's full recipe at equal budget |

### Phase 2 — does the floor change the picture? (v1's cache, 25-keystroke floor, TypeNet split)

| Id | Same as | Data |
|---|---|---|
| **M0-f** | M0 | `data/preprocessed/` (read-only) + `split_typenet.json` |
| **M7-f** | M7 | same |

### Phase 3 — v2

| Id | Spec | Seeds |
|---|---|---|
| **v2** | winning option of each Phase 1 comparison (decision rule in Step 3.2) | 3 (s0, s1, s2) |
| **v2-long** | v2 at 32M triplets (v1's budget): 62.5k × 512 or 500k × 64 | 1 |
| **v2-f** | v2 on the floor cache (the live app's number) | 1 |

---

## Phase 1 — build and train the ablations

### Step 1.1 Resolve the protocol question
- [x] Download the TeX source, extract, grep Table V's caption and the identification paragraph.
- [x] Write the answer (background size, how gallery/query are ranked) into §2.1 and the Progress log.
- [x] (not needed) If identification needs a background other than 1,000, set the evaluator's `--background` default
      accordingly and note it. Authentication stays k = 1,000.

### Step 1.2 `min_keystrokes` in the canonical extractor (the one sanctioned edit)
- [x] `windows_from_keystrokes(pairs, *, min_keystrokes: int = MIN_KEYSTROKES)`: use it in both floor checks
      (session total and trailing window). Keep the `ValueError` message. `extract_features` is unchanged.
- [x] Tests in `features/test_extract.py`: (a) default call is identical to before on the existing fixtures;
      (b) `min_keystrokes=1` accepts a 3-keystroke session → one window with 2 real steps;
      (c) a 60-keystroke session with `min_keystrokes=1` keeps its 9-step trailing window.
- [x] `python -m unittest features.test_extract -v` passes.
- [x] **Webapp check** (procedure in Step 4.4): your own backend on port 8001 starts, serves `/identify/rates`
      and one `POST /identify`; stop your 8001 PID afterwards.

### Step 1.3 TypeNet split + no-floor cache (`data_v2/`)
- [x] `make_split.py`: list subject ids from the raw file names, sort **numerically**, first 68,000 → train, rest
      → test. Write `data/split_typenet.json` with `{"rule": "first 68000 by numeric id (TypeNet IV-D)",
      "train_subjects": [...], "eval_subjects": [...]}`. Print both counts (expect 68,000 / 100,595).
- [x] `build_cache.py`: for every session, `windows_from_keystrokes(session.pairs[:51], min_keystrokes=1)`;
      skip sessions that raise (fewer than 2 keystrokes) and count them. Save the same four arrays to
      `data/preprocessed_typenet/`. Log progress every 5,000 participants. Run it in the background with a
      watcher (full dataset ≈ 40 min). It must refuse to write anywhere under `data/preprocessed/`.
- [x] `.gitignore`: add `data/preprocessed_typenet/` and `runs/` (done early, with `comparison/results/pilot_old_split/`).
- [x] Sanity numbers into the Progress log: windows, sessions skipped, % windows with exactly 50 steps, test
      subjects with ≥ 15 sessions (TypeNet-eligible) for both caches.

### Step 1.4 Loaders for any (split, cache) (`data_v2/datasets.py`)
- [x] `make_sampler(split_path, cache_dir) -> TripletSampler` restricted to the split's train subjects.
- [x] `load_eval_data(split_path, cache_dir)` returning the same 4-tuple as `eval.rank_n.load_eval_data`, so
      every function in `eval/typenet_protocol.py` and `eval/rank_n.py` works unchanged.
- [x] `test_data_v2.py`: train and eval subjects are disjoint; the sampler never yields an eval subject
      (sample 50 batches); the TypeNet-rule cache has exactly one row per (subject, session).

### Step 1.5 Model and loss options (`typenet/`)
- [x] `TypeNetEncoder(..., normalize=False, readout="last", batchnorm=True)`: defaults reproduce the current
      model exactly (existing tests must still pass unmodified). `readout="mean"` = masked mean over real steps.
      `normalize=True` = L2-normalize the output. `batchnorm=False` removes both `MaskedBatchNorm`s.
- [x] `typenet/losses.py`: add `distance="sqeuclidean" | "cosine"` to `triplet_loss` and `mined_loss`
      (cosine = 1 − a·b on unit vectors, the rule in `model/losses.py`). Margin passed explicitly.
- [x] Tests: each flag changes the output (flag on vs off differs); defaults give identical output to a model
      built before the change (same seed); padding invariance holds for every flag combination;
      `mined_loss` selects correctly with cosine (extend `typenet/test_losses.py`).

### Step 1.6 Training script (`typenet/train.py`)
- [x] Flags: `--arch {typenet,v1}`, `--normalize`, `--readout {last,mean}`, `--no-batchnorm`,
      `--distance {sqeuclidean,cosine}`, `--margin`, `--batch-size`, `--steps`, `--semi-from N`, `--hard-from N`
      (negatives: random until semi-from, semi until hard-from, hardest after; omitted = never),
      `--split`, `--cache`, `--run-id` (writes `runs/<run_id>/`), `--resume`, `--seed`, `--save-every`.
- [x] `--arch v1` builds `model.network.KeystrokeEncoder()` (imported), runs A/P/N as three calls (it has no
      BatchNorm, so grouping is irrelevant) and forces cosine distance (its output is already normalized).
- [x] Config saved in every checkpoint: all flags + `"arch"`, `"model"` kwargs, split/cache paths, triplets seen.
- [x] Log line every 100 steps: `step, loss, negatives mode, mean embedding norm, mean pairwise distance in batch,
      s/step`. The last two are the collapse detectors.
- [x] **Collapse guard:** if, during a hardest phase, loss is within 0.2% of the margin *and* mean pairwise
      distance < 1% of its value at the start of that phase, for 2,000 consecutive steps: save a checkpoint, log
      `COLLAPSED at step N`, exit 0. A collapse is a result, not a crash.
- [x] **Mining-start check:** when the hardest phase begins, log the hardest-mined loss ÷ margin on the current
      batch (`HARD-START ratio=…`). > 1 predicts collapse (§2.2).
- [x] `comparison/loaders.py`: build `TypeNetEncoder(**model)` or `KeystrokeEncoder(**model)` from `config["arch"]`.
- [x] `comparison/evaluate.py`: `--split` and `--cache` (default = v1's, so old results stay reproducible);
      record them in the JSON.

---

## Step 2 — smoke tests and timing (before any long run)

- [x] Unit tests: `python -m unittest features.test_extract typenet.test_network typenet.test_losses data_v2.test_data_v2 -v`.
- [x] Regression: `comparison.evaluate` on `model/encoder_hard.pt` and `model/encoder.pt` with the default
      (v1) split and cache must still print 89.8 / 60.5 and 56.9 / 28.4.
- [x] 300-step smoke run of each Phase 1 config (M0–M7) with its exact flags, `--run-id smoke_<id>`, on the
      no-floor cache: loss finite and falling, no error, timing recorded.
- [x] Put measured s/step and projected hours per run in the Progress log; order the queue so short runs come
      first. If a batch-64 run projects over 8 h, halve its steps, note it, and keep the budget comparison honest
      in the write-up.

---

## Step 3 — Phase 1 runs

- [x] Write `runs/queue.sh` (sequential, one job at a time, each job appends to its own `runs/<id>/log.txt`,
      a failure is logged and the queue moves on). Launch it with `run_in_background`.
- [ ] Watchers (§6). Evaluations run in the queue between trainings (one GPU task at a time), not in parallel.
- [x] M0-nf done — **calibration:** Rank-1 in [60, 75]%? If not, **stop the queue and investigate** (split
      direction, cache rule, protocol) before continuing, and record what was found. This is the only point
      where stopping is expected.
- [x] M1-nf done  - [x] M2-nf done  - [ ] M3-nf done  - [ ] M4-nf done
- [ ] M5-nf done (note HARD-START ratio; collapsed?)  - [ ] M6-nf done (same)  - [x] M7-nf done (same: ratio 1.10, partial collapse → 12.5%)

### Step 3.1 Evaluate every finished run
- [ ] `comparison.evaluate --split data/split_typenet.json --cache data/preprocessed_typenet --seeds 0 1 2 3 4 5 6 7 8 9`
      on the final checkpoint and every intermediate one (learning curve).
- [ ] Eligible test people for this cache/split are printed and recorded (expect ~100k with no floor).
- [ ] Add a row to the results table in the Progress log.

### Step 3.2 Decision rule for v2 (apply after all Phase 1 runs)
- For each comparison (M1/M2/M3/M4 vs M0, M5 vs M4, M6 vs M5, M7 vs the best TypeNet-based model), adopt the
  change only if Rank-1 improves by **≥ 2 points** *and* EER at G=1 does not get worse by > 0.3. Otherwise keep
  the simpler/paper option. A collapsed run means that option is rejected.
- Options are assumed to combine additively; Phase 3 checks that assumption (v2 must beat every Phase 1 model).
- Write the chosen v2 spec, with the reason for each choice, into the Progress log.

**Phase 1 exit:** commit (`feat: phase 1 ablations on TypeNet split`) with code, results JSONs and plan updates.

---

## Phase 2 — the floor

- [ ] M0-f and M7-f: same flags as M0/M7, `--cache data/preprocessed --split data/split_typenet.json`.
- [ ] Evaluate each twice: on the floor cache (deployment view) and on the no-floor cache (paper view).
- [ ] Record: Δ Rank-1 from the floor for each model, and whether it is the same for both (±2 pts). If the floor
      helps one model much more than the other, flag it: the Phase 1 ranking may be floor-dependent, and the top
      two Phase 1 candidates get a floor rerun.
- [ ] Commit (`feat: phase 2 floor check`).

---

## Phase 3 — v2

- [ ] Implement v2 with the flags chosen in Step 3.2 (if it needs an option not yet built, add it with a test).
- [ ] v2 × 3 seeds on the no-floor cache. Report mean ± std across **training** seeds (separate from the
      10 evaluation draws).
- [ ] v2 beats every Phase 1 model by more than its seed std? If not, say so plainly; no winner claim.
- [ ] v2-long (32M triplets, seed 0): still improving at the end? (learning curve from its checkpoints)
- [ ] v2-f (floor cache, seed 0): the number the live app would get.
- [ ] Commit (`feat: v2 encoder`).

v2 is **not** wired into the backend. Swapping the served model is a separate decision for the user.

---

## Phase 4 — test, document, hand over

### Step 4.1 Final evaluation
- [ ] Re-run `comparison.evaluate` on every final checkpoint in one go, write the full table.
- [ ] Strict protocol (3 enroll + 1 query, cosine on averaged profile = the backend rule) for every model,
      labelled as not TypeNet's metric.

### Step 4.1b Fair head-to-head with the live v1 model (unseen-by-both test set)
Why: v1 (`model/encoder_hard.pt`) and the new models can't be scored on each other's test sets. v1 trained on a
random 90% of everyone (`data/split.json`, so it has seen most of TypeNet's test people), and the new models trained on
the first 68,000 ids (which include ~40% of v1's test people). Until now the only comparison was 89.8% (v1, its
split, floor) vs 79.3% (M7 @144k, TypeNet split, no floor): different people, different data rule.
- The fair set is people in **both** eval lists: `data/split.json` eval ∩ `data/split_typenet.json` eval =
  **10,001 people** (counted 2026-10-02). Neither v1 nor any new model trained on them.
- [ ] Write `data/split_unseen_both.json` (`"rule": "v1 eval ∩ TypeNet eval"`, `train_subjects`: [], `eval_subjects`:
      the 10,001), so `comparison.evaluate --split` works unchanged. Read-only use of `data/split.json`.
- [ ] Score `model/encoder_hard.pt` (read-only) and every final / best Phase 1–3 checkpoint on it, on **both** caches:
      no-floor (all 10,001 eligible) and floor (only people with ≥ 15 floor sessions, ~2.2k; fewer than 10 × 1,000,
      so draws overlap; say so). Same 10 seeds for every model.
- [ ] Table: model | triplets | train people | Rank-1 | EER G=1/5/10 | strict Rank-1, per cache. This is the one
      table that answers "does a new model beat the app's model?".
- Caveat for the write-up: v1 trained on 151,734 people and 32M triplets, the new models on 68,000 people; a
  model that beats v1 here wins despite fewer people, not on equal terms.

### Step 4.2 Tests
- [ ] All unit tests pass (Step 2 command).
- [ ] v1 regression numbers reproduce exactly (Step 2).
- [ ] `git diff main --stat -- model eval backend frontend data/preprocessed data/split.json` is **empty**.
      `features/extract.py` shows only the `min_keystrokes` change.

### Step 4.3 Documentation
- [ ] README: replace the old-split pilot table with the Phase 1–3 table. Columns: model, what changed, triplets,
      Rank-1, EER G=1/2/5/7/10, strict Rank-1; rows include TypeNet published. State plainly: same split, data
      rule and protocol as the paper except the key code; one training seed except v2; what each comparison shows.
- [ ] Keep the pilot findings (lr 0.05 saturation, hardest-mining collapse ratio) as a short "what we learned
      first" paragraph.
- [ ] TODO.md item 4: mark done, summarize, list anything left open.

### Step 4.4 Webapp check (the app must work with no problems)
- [ ] `python -m unittest features.test_extract` passes.
- [ ] Start your own backend on **port 8001** (`cd backend && ../.venv-model/Scripts/python -m uvicorn app.main:app --port 8001`,
      background, log to a scratch file). Check `GET /identify/rates` and one `POST /identify` succeed and that
      it loads `model/encoder_hard.pt`. Stop **that PID only**. Never touch :8000 / :5173.
- [ ] Commit (`docs: model study results`). Do not push unless asked.

---

## 5. Briefing the user (when they ask "update"/"status"/"brief")

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

## 6. Working alone

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

## 7. Progress log

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

### Where to resume (handover)

- **Done:** Steps 1.1–1.6, Step 2, Phase 1 runs M0, M7, M1, M2 (trained + evaluated). Results in `runs/p1_*/eval.txt`
  and `comparison/results/p1_*.json` (JSONs are to be committed at Phase 1 exit).
- **Not done:** M3, M4, M5, M6. Their 8 queue lines (train + eval each, M6 at full 240k × 64) are in
  `runs/queue.pending`. Optional: add `$PY -u -m comparison.evaluate --split data/split_typenet.json --cache
  data/preprocessed_typenet --checkpoint runs/p1_m2_nf/step6000.pt >> runs/p1_m2_nf/eval.txt 2>&1` for M2's missing point.
- **Resume:** `cp runs/queue.pending runs/queue.txt && bash runs/queue.sh` (run_in_background), then Monitor
  `bash runs/watch.sh` (re-arm every 30 min; it keeps its read position in `runs/.watch_state`).
- **Uncommitted code** (all tested): `features/extract.py` (+`min_keystrokes`), `data_v2/`, `data/split_typenet.json`,
  `typenet/{network,losses,train}.py` + tests, `comparison/evaluate.py`. Phase 1 commit happens after M3–M6.
- **Open questions for Step 3.2 / write-up:**
  1. Hardest mining collapsed v1 on no-floor data (79.3% → 12.5%) but was v1's best step on the floor data — short
     sequences suspected; M5/M6 test whether it holds for TypeNet+L2 and batch 64; Phase 2 (M7-f) tests the floor.
  2. L2 + cosine loses 20.7 pts under random negatives (M1); does it win under mining (M5 vs M4)?
  3. Our EER is ~1–2 pts behind the paper at equal Rank-1 (M0) — key code?
  4. M2's +2.3 is one seed; v2's 3-seed run will tell whether it's real.
- **Investigation resume (2026-10-02):** E3 can finish with `$PY -u -m typenet.train --run-id diag_v1_floor --arch v1
  --batch-size 64 --resume runs/diag_v1_floor/step310000.pt --steps 315000 --hard-from 300000 --cache data/preprocessed
  --save-every 5000 >> runs/diag_v1_floor/log.txt 2>&1` (optional, its result is already clear). Then Q2 tests, Rank-50,
  and a decision on M5/M6 (likely: longer semi phase before hardest, or drop hardest) **before** running
  `runs/queue.pending` as written.
- **Environment:** a Windows-update reboot killed the queue once (07:26); pause updates for long runs. Before the
  reboot the GPU was power/thermal-throttled (0.69 s/step); after it 0.18–0.23 s/step.

### Results table (fill as runs finish)

| Run | Data | Triplets | Rank-1 | EER G=1/5/10 | Strict Rank-1 | Note |
|---|---|---|---|---|---|---|
| TypeNet published | their split, key code | 15.4M | 67.4% | 5.4 / 2.2 / 1.6 | — | reference |
| M0 p1_m0_nf | TN split, no floor | 15.4M | 66.7% | 7.31 / 3.43 / 2.60 | 28.4% | calibration ✓ |
| M7 p1_m7_nf final | TN split, no floor | 15.4M | 12.5% | 25.6 / 21.7 / 20.8 | 8.5% | collapsed in hardest phase |
| M7 @144k (pre-hard) | TN split, no floor | 9.2M | 79.3% | 5.12 / 1.72 / 1.23 | 40.4% | random + semi only |
| M1 p1_m1_nf | TN split, no floor | 15.4M | 46.0% | 8.19 / 4.59 / 3.67 | 20.2% | L2 + cosine hurts (random negatives) |
| M2 p1_m2_nf | TN split, no floor | 15.4M | 69.0% | 7.43 / 3.56 / 2.69 | 29.1% | mean readout, +2.3 (passes rule) |
