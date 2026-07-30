# TypeID — Keystroke Biometrics Identification

A portfolio project exploring **open-set biometric identification from free-text keystroke
dynamics** — the typing-rhythm equivalent of a fingerprint or face-recognition search system.
Given an unknown typing sample, the goal is to return the top-K most likely matches from a
gallery of enrolled identities.

This is **not a classifier**. It's a 1:N gallery search built on a learned embedding space: a
network is trained once, offline, to map a keystroke sequence to a vector such that the same
person's typing lands close together and different people's typing lands far apart. New people
can be enrolled later without retraining anything — enrollment/query are just a forward pass
plus nearest-neighbor search.

Subjects prove identity by **transcribing a random on-screen sentence** they've never seen
before (not free composition, not a fixed repeated password). See [CLAUDE.md](CLAUDE.md) for
the full design rationale, including why this is a proxy task rather than a forensically
validated system, and [FUTURE_PROSPECTS.md](FUTURE_PROSPECTS.md) for open research questions
(free-composition mode, cross-device generalization, legal framing).

## Try it out

The demo currently only shows the **capture UI** (type a prompt, see your dwell/flight-time
rhythm) — enrollment and identification against a live gallery aren't wired up yet (see
[Current state](#current-state-vs-target) below).

**Backend** (FastAPI, from `backend/`):
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```
Serves on `http://127.0.0.1:8000`; `/health` should return `{"status": "ok"}`.

**Frontend** (Vite, from `frontend/`):
```
npm install
npm run dev
```
Serves on `http://localhost:5173`, CORS-allowed against the backend above.

## Current state vs. target

| Piece | Target (per [CLAUDE.md](CLAUDE.md)) | Current state |
|---|---|---|
| Feature extraction (`features/extract.py`) | Canonical HL/IL/PL/RL timing-vector extractor, shared byte-for-byte by training and live capture | Stub — raises `NotImplementedError`; not yet built against real Aalto data |
| Model (`/model`) | 2-layer LSTM triplet-loss embedding network, trained on Aalto | Doesn't exist yet — no directory, no training script, no weights |
| Embedding function (`backend/app/embedding.py`) | Frozen `f(features) -> 128-dim embedding` | Stub — raises `NotImplementedError` until a trained model lands |
| Gallery store (`backend/app/gallery.py`) | SQLite table of `{person_id, name, embedding, enrolled_at}` | **Done** — implemented and working |
| `/enroll`, `/identify` endpoints | Full extract → embed → pool/rank flow | Routing, schemas, pooling, and ranking logic are fully written and correct, but unreachable — both endpoints return `501` until extraction/embedding are implemented |
| Frontend capture UI | Prompt display, capture, submit to backend, render top-5 results | Prompt display + capture + client-side rhythm visualization work; **not yet wired to the backend** (enroll/identify calls are a TODO) |
| Evaluation (`/eval`) | CMC curve, Rank-N accuracy | Doesn't exist yet — depends on a trained model |
| Data (`/data`) | Cached preprocessed Aalto sequences | Doesn't exist yet — dataset not yet downloaded/prepared |

In short: the **application skeleton is real and correct** (API contracts, gallery persistence,
ranking/thresholding logic, frontend capture), but the **ML core is not built** — no feature
extractor, no trained embedding model. That's the critical path; everything downstream of it is
already waiting and wired up.

## Architecture

```
[Training — offline, once]        Aalto dataset -> triplet-loss LSTM -> frozen f()
[Enrollment]                      2-3 transcribed prompts -> extract_features() -> f() -> stored embedding
[Identification]                  1 transcribed prompt -> extract_features() -> f() -> cosine-rank gallery -> top-5
```

`features/extract.py` is the single canonical feature extractor — it must be used unchanged by
both the offline training pipeline and the live `/enroll` and `/identify` endpoints. This is the
most important invariant in the project; see CLAUDE.md's "Things to get right" section.

## Repo structure

```
/data/        Aalto dataset (raw + preprocessed) — not yet populated
/features/    canonical feature-extraction module — stubbed
/model/       network definition, training script, weights — not yet created
/eval/        CMC curve, Rank-N accuracy scripts — not yet created
/backend/     FastAPI app: gallery store (done), /enroll + /identify (wired, blocked on model)
/frontend/    Vite + Tailwind capture UI (working, not yet wired to backend)
```

For the full architecture, feature spec, model/loss details, and evaluation methodology, see
[CLAUDE.md](CLAUDE.md).
