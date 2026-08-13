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
  - Still open: unit tests (`test_extract.py`) covering edge cases formally, rather than the one-off verification script used so far

- [ ] **Cache preprocessed Aalto sequences**
  - Apply `features/extract.py` to all Aalto sessions
  - Store as `.npy` or parquet in `/data/preprocessed/`
  - Index by subject_id + session_id for training triplet construction
  - Goal: Fast data loading during model training (avoid recomputing on every epoch)

## Phase 2: Model Training

- [ ] **Define LSTM embedding network**
  - 2-layer LSTM(128) with dropout
  - Masked mean-pool or final hidden state over variable-length sequences
  - Dense layer to 128-dim embedding
  - L2 normalization of output
  - File: `/model/network.py`
  - Verify: Input is (batch, M=50, 4), output is (batch, 128)

- [ ] **Implement triplet loss**
  - Formula: `L = max(0, d(f(A), f(P)) - d(f(A), f(N)) + margin)`
  - Use cosine distance
  - Margin: start ~0.5, tune if needed
  - File: `/model/losses.py`

- [ ] **Triplet construction from Aalto**
  - Sample subject X, draw two sessions (A, P) with different sentences
  - Sample subject Y, draw one session (N)
  - Resample fresh triplets per batch (not precomputed)
  - Future: add hard negative mining after warmup

- [ ] **Training script (`model/train.py`)**
  - Load preprocessed Aalto sequences
  - Batch triplets, forward pass, backprop
  - Adam optimizer, lr ≈ 1e-3
  - Log loss per batch
  - Save checkpoint every N epochs
  - Validate: Run training on subset (e.g., 100 subjects, 5-10 epochs) to verify convergence

- [ ] **Evaluate on held-out subjects**
  - Split Aalto: train on subjects 1-800, eval on 801-1000 (example split)
  - Build gallery from eval subjects (2-3 sessions each)
  - Query with held-out sessions from same subjects
  - Compute Rank-N accuracy (N=1,5,10) and CMC curve
  - Compare against TypeNet published baseline (~80% Rank-1 on similar protocol)

- [ ] **Save trained weights**
  - Serialize model to `/model/weights.pth` (or `.pt` / `.h5`)
  - Save training config (feature dims, embedding dim, loss margin)
  - Document model version and Aalto subset used

## Phase 3: Backend Inference & API

- [ ] **Implement embedding function (`backend/app/embedding.py`)**
  - Load frozen weights from `/model/weights.pth`
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
