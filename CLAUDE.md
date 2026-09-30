# CLAUDE.md — TypeID (keystroke biometrics)

Open-set 1:N identification by typing rhythm: an LSTM maps keystroke timing to an embedding, people enroll into a
gallery without retraining, and queries are ranked against it (with a "no match" threshold).

**Where things are:** [README.md](README.md) — current state and results. [ARCHITECTURE.md](ARCHITECTURE.md) —
technical design. `TODO.md` — private task tracker. `.claude/plans/` — active implementation plans.

## How to work with this user

This is a personal CV/learning project (AI/ML, security, forensics), not a client deliverable. The point is to
understand the material, not just to have working code appear.

- Explain *why* alongside *what*: the reasoning behind an architecture choice, a metric or a caveat matters as much
  as the code. Surface real tradeoffs instead of picking silently.
- Work in small steps. At meaningful boundaries (a new file, a new concept, a design decision), stop and explain in
  plain language what changed and why before moving on.
- "Just get it done" or "you're on your own" overrides this for that task only.
- Keep answers short and clear; lead with the answer. Tables and bullets when they help.

## The critical invariant

**Feature extraction must be identical in training and live use.** `features/extract.py` is the only place the
HL/IL/PL/RL math and the M=50 windowing live:
- dataset pipelines call `windows_from_keystrokes(pairs)` on (press, release) pairs;
- the backend calls `extract_features(events)` on browser events (it pairs them, then calls the same function).

Never reimplement it (not in JS, not in a script). A mismatch degrades the model silently.

## Pitfalls that stay true

- **Split by subject, never by sample.** Eval subjects must never be seen in training.
- **Rank-N / CMC for identification; FAR/FRR/EER only for verification.** Not classification accuracy.
- **Always threshold the top score** ("no match"); never force a top-1 pick.
- **No key identity in the input,** and anchor/positive from different sentences, or the model learns content.
- **Masking:** padding sits at the end of a window; padded steps must not affect the embedding.
- **25-keystroke floor** at enrollment and query (`MIN_KEYSTROKES`); reject shorter input.
- **Not fixed-N classification:** people are gallery entries, never output classes.

## The live system — don't break it

The user runs the webapp from this working tree, so edits are live immediately.
- The backend serves `model/encoder_hard.pt`. Don't overwrite or retrain into it; new models get new files.
- `data/preprocessed/` + `data/split.json` are v1's training cache and split. Never rebuild or edit them
  (`data/build_cache.py` writes straight into that folder).
- `backend/gallery.db` holds the user's enrollments (private); `backend/gallery-populated.db` is the Aalto-only copy
  that cloners get. Back up before touching either.
- The user's dev servers run on :8000 (backend) and :5173 (frontend). **Never kill processes by port or image name;**
  stop only PIDs you started. To test the backend yourself, run your own on another port.

## Commands

Python work uses `.venv-model` (system Python has no torch).

| Task | Command |
|---|---|
| Backend | `cd backend && ../.venv-model/Scripts/python -m uvicorn app.main:app --port 8000` |
| Frontend | `cd frontend && npm run dev` (build: `npx vite build`) |
| Tests | `.venv-model/Scripts/python -m unittest features.test_extract typenet.test_network typenet.test_losses` |
| Evaluate a checkpoint | `.venv-model/Scripts/python -m comparison.evaluate --checkpoint <path>` |

## Training runs

- The GPU is a 4 GB laptop RTX A2000: **one training job at a time** (two at once thrash or crash).
- Run anything longer than a few minutes in the background and watch its log for progress *and* failures.
- Checkpoints and logs go in `runs/` (gitignored); never commit `*.pt` files.

## Git

- Feature branches, merged to `main` by PR. Commit or push only when asked.
- Never commit: `backend/gallery.db`, `backend/identify_log.jsonl`, `TODO.md`, `PRODUCT.md`, `FUTURE_PROSPECTS.md`,
  `KNOWN_ISSUES.md`, `.claude/settings.local.json` (all gitignored — keep it that way).

## Code style

- Simplest thing that works; no speculative features, abstractions or error handling for impossible cases.
- Comments explain *why* (invariants, subtle bugs, workarounds), not *what*.
- One source of truth: one feature extractor, one inference path.
- Match the surrounding code's style and naming.
