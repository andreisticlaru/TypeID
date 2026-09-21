# TODO — TypeID Project Progress

Track ongoing work to complete the keystroke biometrics identification system. Reference [README.md](README.md) for current state and [ARCHITECTURE.md](ARCHITECTURE.md) for specs.

## Phase 1: Data & Features

Essential foundation—everything downstream depends on this.

- [x] **Download/prepare Aalto dataset**
  - ✅ Aalto 136M Keystrokes dataset (Dhakal et al., 2018) downloaded
  - Location: `C:\Users\andre\OneDrive\Personal Projects\Keystroke Biometrics\Keystrokes\Keystrokes\files`
  - 168,595 keystroke files (user_id_keystrokes.txt format)
  - Accessible via `data/config.py` → `AALTO_RAW_PATH`
  - Run `python data/inspect.py` to verify structure and get stats
  - Goal: Enable training pipeline to read sessions ✓

- [x] **Implement feature extraction (`features/extract.py`)**
  - ✅ `windows_from_keystrokes()` computes HL/IL/PL/RL from (press, release) pairs, windows/pads/masks to M=50
  - ✅ `pair_browser_events()` pairs raw keydown/keyup events for the live path
  - ✅ `extract_features()` is the public entry point for `/enroll`/`/identify`
  - ✅ `data/aalto_loader.py` feeds pre-paired Aalto rows into the same `windows_from_keystrokes()`
  - ✅ Backspace kept as a real keystroke (no special-casing needed—dataset already logs it as `BKSP`)
  - ✅ `MIN_KEYSTROKES = 25` floor enforced, raises `ValueError` below it
  - ✅ Verified end-to-end on real Aalto files, including a CSV-quoting bug fix (`csv.QUOTE_NONE`—Aalto sentences contain literal `"` chars that broke default quoting and silently merged rows)
  - ✅ Per-window floor: a trailing window with fewer than `MIN_KEYSTROKES` real vectors is dropped (the session-level check alone let 1-24-vector tail windows into the cache)
  - ✅ `features/test_extract.py`: 12 unit tests covering windowing/masking/padding boundaries, the min-length guard, the dropped-trailing-window rule (and its boundary at exactly `MIN_KEYSTROKES`), event pairing (sequential, overlapping, orphaned keyup), and negative-IL preservation. All passing.

- [x] **Cache preprocessed Aalto sequences**
  - ✅ `data/build_cache.py` runs `windows_from_keystrokes()` over all Aalto sessions, saves to `/data/preprocessed/` as four flat `.npy` arrays (no per-subject files — grouping by subject happens later via an in-memory index over `subject_ids.npy`)
  - ✅ Full run (168,593 participants): `sessions_ok=2,280,166`, `sessions_skipped=248,729` (below `MIN_KEYSTROKES` floor), elapsed=2161s (~36 min)
  - ✅ Verified saved arrays:
    ```
    windows.npy      (2483167, 50, 4) float32   (rebuilt after the per-window floor; was 3,343,148)
    mask.npy         (2483167, 50) bool
    subject_ids.npy  (2483167,) <U6
    session_ids.npy  (2483167,) <U7

    unique subjects: 168593 (matches participants_processed)
    any NaN in windows: False
    mask true-count per row: min=25, max=50 (never below MIN_KEYSTROKES, never >50)
    padded region is all-zero: confirmed on sample rows
    first row: subject=100001, session=1090979 (1 window for this subject/session)
    ```
  - ✅ Example row at the minimum-length floor — row 61 (subject=100008, session=1091062, real length=25): `mask[61]` is `True` for indices 0-24, `False` for 25-49; `windows[61][20:29]` shows real HL/IL/PL/RL values through index 24, then exact `(0,0,0,0)` padding onward — confirms padding/masking works correctly at the boundary
  - Note: `sessions_ok` (2.28M) < total windows (2.48M) because sessions longer than M=50 split into multiple windows — expected, not a bug
  - Note: `data/preprocessed/` is a snapshot of the extractor. Any change to `features/extract.py` (including `MIN_KEYSTROKES`) needs `python -m data.build_cache --limit 168593` (~36 min) to take effect on training data.
  - Goal: Fast data loading during model training (avoid recomputing on every epoch) ✓

## Phase 2: Model Training

- [x] **Define LSTM embedding network**
  - 2-layer LSTM(128) with dropout
  - Masked mean-pool or final hidden state over variable-length sequences
  - Dense layer to 128-dim embedding
  - L2 normalization of output
  - File: `/model/network.py`
  - Verify: Input is (batch, M=50, 4), output is (batch, 128)

- [x] **Implement triplet loss**
  - Formula: `L = max(0, d(f(A), f(P)) - d(f(A), f(N)) + margin)`
  - Use cosine distance
  - Margin: start ~0.5, tune if needed
  - File: `/model/losses.py`

- [x] **Triplet construction from Aalto** (`model/triplet_sampler.py`)
  - ✅ Sample subject X (2+ sessions), draw two different sessions (A, P); sample different subject Y, draw one session (N)
  - ✅ Fresh triplets per batch (not precomputed); `sample_batch` stacks them; ~0.1 ms/triplet
  - ✅ Optional `subjects` allowlist for the train/eval split
  - ✅ Hard negative mining added later, in-batch (see `model/train.py` below); candidates larger than one batch are still untried

- [x] **Subject-disjoint train/eval split** (`data/make_split.py` → `data/split.json`)
  - ✅ 90/10 by subject, seed 42, saved so train and eval share the same held-out set (151,734 / 16,859)

- [x] **Training script (`model/train.py`)**
  - ✅ Adam (lr 1e-3), batches of 64 triplets sampled fresh each step; runs from `.venv-model` (system Python has no torch)
  - ✅ `--save-every` numbered checkpoints, `--resume` (weights + Adam state), `--out`, checkpoints bundle weights + config
  - ✅ In-batch negative mining: `--mine semi` (closest negative still farther than the positive) and `--mine hard` (closest valid negative); candidates from the anchor's own subject are masked out
  - Why mining: after 200k steps 78% of random triplets gave zero loss, which explains the plateau

- [x] **Evaluate on held-out subjects** (`eval/rank_n.py`)
  - ✅ Subject-disjoint split (151,734 train / 16,859 held-out), gallery of 1,000 held-out people, 10 random draws, mean ± std
  - ✅ Rank-1/5/10/20 (CMC data in the dashboard); untrained baseline included
  - ✅ Also runs the TypeNet-style protocol (`--enroll-sessions 10 --probe-sessions 5 --score pairwise`)
  - Standard protocol (3 enroll + 1 query), Rank-1: untrained 5.4% → 200k random negatives 28.4% → 500k semi-hard 51.7% → **500k hardest 56.9%**
  - TypeNet-style protocol, Rank-1: 200k baseline 65.2% (paper: 67.4%) → **hardest 93.3%** (98.0% with averaged profiles)
  - Not yet done: verification EER for a like-for-like comparison with TypeNet's headline metric

- [x] **Save trained weights**
  - ✅ `model/encoder_hard.pt` (500k steps, hardest mining) is the current best; `model/*.pt` is gitignored
  - ✅ Checkpoint stores `state_dict` + config (model kwargs, margin, lr, split seed, mining mode) + Adam state
  - Still to do: the backend must load this file (see Phase 3)

## Phase 3: Backend Inference & API

- [ ] **Implement embedding function (`backend/app/embedding.py`)**
  - Load frozen weights from `model/encoder_hard.pt` (the current best; build `KeystrokeEncoder(**ckpt["config"]["model"])`, then `load_state_dict(ckpt["state_dict"])`, then `.eval()`)
  - `embed(features: np.array) -> np.array` function
  - Input: (M=50, 4) or (batch, M=50, 4)
  - Output: 128-dim embedding, L2 normalized
  - Raise error if weights not found
  - Replace `NotImplementedError` stub

- [ ] **Implement gallery store (`backend/app/gallery.py`)**
  - ✅ Already done—SQLite table ready
  - Verify schema: `{person_id, name, embedding (128 floats), enrolled_at}`
  - Test insert/query operations

- [ ] **Wire `/enroll` endpoint**
  - POST `/enroll` with `{name, events: [{key, event_type, timestamp}, ...]}`
  - Call `features.extract.extract_features(events)` → array
  - Call `embed(features)` → 128-dim vector
  - Pool embeddings if multiple event sequences provided (average or first)
  - Insert `{person_id (UUID), name, embedding, enrolled_at (now)}` into gallery
  - Return `{person_id, name}` to client
  - Test: Enroll 3 sample users, verify gallery contains them

- [ ] **Wire `/identify` endpoint**
  - POST `/identify` with `{events: [...]}`
  - Extract features, embed
  - Query gallery: `cosine_similarity(query_emb, all gallery embeddings)`
  - Sort descending, return top-5: `[{person_id, name, similarity_score}, ...]`
  - Optional: threshold (return empty if best score < min_confidence)
  - Test: Query with enrolled user's new session, verify correct ranking

- [ ] **Test feature parity**
  - ✅ Verify `/enroll` and `/identify` use the same `features.extract.extract_features()`
  - No reimplementation in JavaScript or elsewhere
  - Add integration test: enroll → immediately identify → should rank top-1

## Phase 4: Frontend Wiring

- [ ] **Connect enroll form to `/enroll` endpoint**
  - Capture user name input
  - Capture 2-3 keystroke event sequences (browser already logs events)
  - POST to backend `/enroll`
  - Display confirmation with person_id or error message

- [ ] **Connect identify form to `/identify` endpoint**
  - Capture unknown keystroke event sequence
  - POST to backend `/identify`
  - Render top-5 results with similarity scores
  - Optional: confidence threshold UI (accept match only if top score > threshold)

- [ ] **Test end-to-end flow**
  - Enroll a test user in frontend
  - Query with new session of same user
  - Verify user appears in top-5 with high score
  - Enroll multiple users, verify ranking

## Phase 5: Evaluation & Reporting

- [ ] **Implement CMC curve script (`eval/cmc.py`)**
  - Load test gallery (subset of Aalto eval subjects)
  - For each query session, embed and rank against gallery
  - Compute Rank-N accuracy for N ∈ [1, 5, 10, 20]
  - Plot CMC curve: Rank-N accuracy vs. N
  - Save plot to `/eval/cmc_curve.png`

- [ ] **Implement Rank-N accuracy report (`eval/rank_n.py`)**
  - Report Rank-1, Rank-5, Rank-10 accuracy as percentages
  - Breakdown by subject (optional: per-subject accuracy)
  - Compare against TypeNet baseline if available

- [ ] **Document evaluation results**
  - Create `/eval/RESULTS.md`
  - Include CMC plot, Rank-N table, dataset/subject split used
  - Note domain limitations (transcription proxy, known forensic gap)
  - Link to [FUTURE_PROSPECTS.md](FUTURE_PROSPECTS.md)

## Phase 6: Documentation & Polish

- [ ] **Add model card / system card**
  - Dataset: Aalto, size, protocol
  - Model: architecture, training hyperparams, performance
  - Limitations: transcription-only, no cross-device eval, no free-composition
  - Intended use: identification ranking, not verification/authentication

- [ ] **Add running instructions to README**
  - Step-by-step: train model, start backend, start frontend, enroll, identify
  - Expected runtime for each phase

- [ ] **Verify all repo files exist**
  - ✅ `/features/extract.py` → complete
  - ✅ `/backend/app/embedding.py` → complete
  - ✅ `/model/network.py`, `losses.py`, `train.py` → complete
  - ✅ `/eval/cmc.py`, `rank_n.py` → complete
  - ✅ `/data/raw/`, `/data/preprocessed/` → populated

---

## Key Invariants & Pitfalls (Reference)

**Do not skip:**
1. **Feature extraction parity** — `features/extract.py` is the single canonical implementation. Used everywhere.
2. **Correct train/eval split** — Split by subject, not sample. Prevents information leak.
3. **Evaluation metrics** — CMC curve + Rank-N accuracy only. No classification metrics.
4. **Minimum sequence length** — Enforce ~25-30 keystrokes. Reject noise.

See [CLAUDE.md](CLAUDE.md#things-to-get-right) for full pitfalls list.

---

## Dependency Graph

```
Phase 1: Data & Features (required for everything)
  ├─ Phase 2: Model Training
  │   └─ Phase 3: Backend Inference
  │       ├─ Phase 4: Frontend Wiring
  │       └─ Phase 5: Evaluation
  └─ (Phase 3/4/5 can proceed in parallel once Phase 1 is done)

Phase 6: Documentation & Polish (final pass, can run in parallel with testing)
```

**Critical path:**
1. Feature extraction → 2. Model training → 3. Backend wiring → 4. Frontend → 5. Eval
