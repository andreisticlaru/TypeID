# CLAUDE.md — Keystroke Biometrics Identification System

This file gives context for working on this project. It's a portfolio project implementing
**open-set biometric identification from free-text keystroke dynamics** — conceptually the
typing-rhythm equivalent of fingerprint (AFIS) or face-recognition search systems: given an
unknown typing sample, return the top-K most likely matches from a gallery of enrolled
identities.

**Capture protocol: random-prompt transcription.** The subject is shown a sentence they've
never seen before, drawn at random from a prompt pool, and types it verbatim. This is *not*
free composition (the user isn't writing their own thoughts) and *not* a fixed repeated
password — the text content varies every session, but the task (transcribe what's on screen)
is controlled. This matters because it rules out fixed-position features (see Data/Model
below) while still giving reasonably consistent typing behavior to compare across sessions.

**Forensic-realism caveat.** Transcription is a proxy task, not a validated stand-in for real
forensic use. Actual forensic targets — ransom notes, threatening emails, insider-threat chat
logs — are almost always **free composition**: the author is generating their own words, not
copying a shown prompt. Transcription (read-then-copy) and composition (think-then-type, with
pauses for word retrieval and self-correction) produce measurably different rhythm, so a model
trained only on transcription data has a known domain gap against composed text. Cross-device
generalization (enrollment and query typed on different physical keyboards) is a second gap
this project doesn't currently address. Random-prompt transcription is still the right v1
choice — it's a real improvement over a fixed repeated password (forces generalization to
unseen text, avoids rewarding memorized motor patterns for one password) and it's what the
standard public dataset (Aalto) provides — but write-ups and demos should describe this project
as validating the *architecture* (open-set embedding search) on a transcription proxy task, not
as a forensically validated system. If genuine forensic realism becomes a goal later, the
natural next step is adding a free-composition capture mode (e.g. "write a few sentences about
your day" instead of a shown prompt) and evaluating cross-device performance explicitly, rather
than assuming transcription results transfer.

## Project framing (read this first)

This is **not** a classifier. It is a **1:N gallery search system** built on a learned
embedding space. That distinction drives every architectural decision below, so it's worth
being explicit about it anywhere in the code or docs:

- A multiclass classifier bakes a fixed set of N identities into the output layer. Adding a
  person means retraining. That's the wrong model for "grow a database of suspects/enrollees
  over time."
- Instead: train an embedding network **once**, offline, on a public dataset. It never sees
  the people who will later be enrolled. At enrollment/query time, no training happens —
  only a forward pass + nearest-neighbor search.
- The network has no notion of "identity" at inference time. It only encodes a general
  notion of "typing-rhythm similarity." That's what lets it generalize to unseen people —
  a fundamentally different guarantee than a softmax classifier provides.

## Two related but distinct application modes

| | Identification (this project's focus) | Verification (secondary/optional) |
|---|---|---|
| Question | "Which of N enrolled people typed this?" | "Is this the person they claim to be?" |
| Operation | Rank ALL gallery entries by similarity, return top-K | Threshold ONE distance (query vs. claimed template) |
| Metric | Rank-N accuracy, CMC curve | FAR, FRR, EER |
| Real-world analogy | AFIS fingerprint search, face-recognition search | Phone unlock, continuous auth |

Both modes are powered by the **same frozen embedding network** — only the downstream
operation (rank-and-return-top-K vs. threshold-one-score) differs. If verification/continuous-auth
gets built later, do not create a second network; reuse `f()` below.

## Architecture overview

```
[Training phase — offline, once, on public dataset]
  Aalto "136M Keystrokes" dataset (random-sentence transcription, thousands of subjects)
        │
        ▼
  Triplet-loss training of a sequence-encoder embedding network f()
        │
        ▼
  Frozen model: f(keystroke event sequence) -> fixed-length embedding (e.g. 128-dim)


[Enrollment — building the gallery/database]
  New person transcribes 2-3 random prompts in the browser demo
        │
        ▼
  Raw keydown/keyup events -> SAME feature extractor used in training
        │
        ▼
  embedding = f(event sequence)  -> stored as {person_id, name, embedding} in gallery


[Identification / query — "find the culprit"]
  Unknown sample -> same feature extractor -> f() -> query embedding
        │
        ▼
  cosine_distance(query, every gallery embedding)
        │
        ▼
  sort ascending -> return top-5 {person_id, name, similarity_score}
  optionally flag "no confident match" if best score < threshold
```

**Critical constraint:** the feature extraction function must be byte-for-byte identical
(or at least dimensionally and semantically identical) whether it's processing:
1. Rows from the Aalto dataset during training, and
2. Raw browser-captured keydown/keyup events during enrollment/query.

Do not maintain two separate implementations of feature extraction (e.g., one in Python for
training, one reimplemented in JS for the demo). Route live browser events to a backend
endpoint that runs the one canonical Python feature-extraction function. Divergence here is
the single most likely source of silent bugs in this project — a model trained on
slightly-different features than what's fed at inference time will degrade quietly rather
than error loudly.

## Data

**Primary dataset:** Aalto University "136M Keystrokes" dataset (Dhakal et al., 2018).
Large-scale, public, collected under exactly this project's protocol: subjects are shown a
random English sentence and transcribe it, repeated over many sentences per subject. Thousands
of subjects — enough to hold out a fully disjoint set of identities for eval (see Evaluation).

Because the transcribed text differs every session, **fixed per-position features (as CMU-style
password datasets allow) don't apply** — subject A's sentence and subject B's sentence share no
common character positions to compare. The feature representation has to be text-independent:

**Per-keystroke timing features (no key identity, by default):**
For each keystroke `n` in the sequence, compute:
- **HL** (hold latency / dwell): `release_time(n) - press_time(n)`
- **IL** (inter-key latency / flight): `press_time(n+1) - release_time(n)`
- **PL** (press latency): `press_time(n+1) - press_time(n)`
- **RL** (release latency): `release_time(n+1) - release_time(n)`

This gives a variable-length sequence of 4-dim timing vectors — one per keystroke — rather than
a single fixed-length vector. **Deliberately omit key identity from the input by default.** This
is the TypeNet design choice (Acien et al., "TypeNet: Deep Learning Keystroke Biometrics",
2021) and it's the right default here: a model that only sees timing, not *what* was typed, is
forced to learn general typing rhythm rather than memorizing content, which is exactly what
"generalizes to unseen text" requires. Key-identity embeddings can be added later as an
ablation/enhancement, but timing-only should be the v1 baseline.

**Fixed window length:** pad/truncate each sample to a fixed number of keystrokes, `M = 50`
(TypeNet's value, and a reasonable default — long enough to carry rhythm signal, short enough
that most transcribed sentences supply it in one window). Samples longer than `M` are split into
non-overlapping windows; shorter ones are zero-padded with a mask. Document whichever value is
used and keep it identical between training and live capture.

**Backspace/typo handling:** decide once, apply identically in training preprocessing and live
capture. Recommended default: keep backspace as a real keystroke in the sequence (it's part of
the person's typing behavior) rather than silently deleting the corrected character — self-
correction patterns are themselves a weak biometric signal, and dropping them creates a subtle
train/inference mismatch if the live capture path ever handles it differently.

**Live capture format:** browser JS logs `{key, event_type: keydown|keyup, timestamp}` for
each event during prompt transcription. The backend feature extractor consumes this raw event
stream and produces the same HL/IL/PL/RL sequence used in training — never compute timing
features in JS.

## Model

**Type:** sequence encoder (2-layer LSTM, TypeNet-style), trained as a triplet network. A plain
MLP no longer applies once the input is a variable-length sequence of per-keystroke timing
vectors rather than a fixed-length feature vector.

```
Input: sequence of M=50 timing vectors, each (HL, IL, PL, RL)  [zero-padded + masked]
  -> LSTM(128) + dropout
  -> LSTM(128) + dropout
  -> take final hidden state (or masked mean-pool over timesteps)
  -> Dense(embedding_dim=128)  + L2 normalization
```

There is one set of weights. "Siamese"/"triplet network" refers to running this identical
encoder on multiple samples per training triplet, not to having multiple networks. A small
Transformer encoder (self-attention over the keystroke sequence) is a reasonable alternative
to the LSTM if there's appetite to explore it, but LSTM is the proven, simpler default —
don't reach for attention unless the LSTM baseline is already working.

**Loss: triplet loss.**

```
L = max(0, d(f(A), f(P)) - d(f(A), f(N)) + margin)
```
- Anchor (A) and Positive (P): two different sessions (different transcribed sentences) from
  the same subject. Using *different* sentences for A and P, not the same one, is what forces
  the network to learn subject-specific rhythm instead of sentence-specific timing — this is
  more important here than it was for the fixed-password version, since sentence content now
  varies freely.
- Negative (N): a sample from a different subject.
- `margin`: hyperparameter, start around 0.2–1.0.
- Triplets already satisfying the margin contribute zero loss/gradient — this is expected
  and fine, it means the space is already well-organized for that triplet.

**Triplet construction from Aalto:**
1. Sample subject X, draw two of their sessions (different sentences) as (A, P).
2. Sample a different subject Y, draw one session as N.
3. Resample fresh triplets per batch/epoch rather than precomputing all combinations.
4. Optional refinement (mention if implemented): semi-hard/hard negative mining — bias
   sampling toward negatives that are *currently* close in embedding space, once the space
   is minimally organized (e.g., after a few warmup epochs of random sampling). Improves
   separation quality but isn't required for a working v1.

**Training loop:** Adam, lr ≈ 1e-3, batch of ~32–128 triplets. Larger and slower than the
CMU/MLP version (sequence encoder over more data) — plan for GPU or a scaled-down subject/
session subset of Aalto if iterating on CPU.

**Output of training:** a single frozen function `f(keystroke event sequence) -> embedding`.
Everything downstream (enrollment, identification, verification) is a forward pass through this
function plus vector comparison — no further training at runtime.

## Gallery / database

For this project's scale, a plain in-memory store or SQLite table is sufficient:
`{person_id, name, embedding (128 floats), enrolled_at}`. FAISS/Annoy would only matter at a
gallery size where brute-force cosine comparison becomes slow (thousands+ entries) — worth
name-dropping as "considered, not needed at this scale" rather than actually integrating it,
unless demonstrating ANN search is itself a goal.

## Identification query logic

```python
def identify(query_features, gallery, top_k=5, min_confidence=None):
    query_emb = f(query_features)
    scored = [
        (entry.person_id, entry.name, cosine_similarity(query_emb, entry.embedding))
        for entry in gallery
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    results = scored[:top_k]
    if min_confidence is not None and results[0][2] < min_confidence:
        results = []  # "no confident match" — mirrors real forensic workflow,
                       # where a human examiner reviews candidates rather than
                       # trusting a forced top-1 pick
    return results
```

`query_features` is now a windowed sequence (see Data — fixed `M=50` window), not a single
vector. If a query's transcribed text produces more than one window, embed each window and
average (or otherwise pool) the resulting embeddings before the cosine comparison — decide and
document one pooling strategy rather than letting it vary ad hoc.

## Evaluation

Use the metrics standard to *identification/search* tasks, not generic classification metrics:

- **Rank-N accuracy**: fraction of queries where the true identity appears within the top-N
  returned candidates (e.g., Rank-1, Rank-5 accuracy).
- **CMC curve (Cumulative Match Characteristic)**: Rank-N accuracy plotted as a function of N.
  This is the standard evaluation artifact in fingerprint/face-recognition identification
  literature — include this plot, it's a strong signal of understanding the right framing
  for the problem.
- Keep FAR/FRR/EER reserved for if/when verification mode is evaluated — don't conflate the
  two metric families in write-ups.
- Reference point: TypeNet's published free-text results on Aalto are a reasonable sanity
  target when checking whether the model is in the right ballpark — don't expect to beat a
  published, larger-scale result with a portfolio-scope training run, but should be in the
  same neighborhood, not wildly off.

Held-out protocol: split Aalto subjects (not just sessions) so the network is evaluated on
identity-search performance in a way that reflects "gallery grows with new people" — e.g.
train the embedding on a subset of subjects, then build the eval gallery from a disjoint set
to test true generalization to unseen identities, not just unseen sessions of trained identities.

## Demo layer

Live web demo, random-prompt transcription:
1. **Enroll**: user transcribes 2-3 random prompts (different sentences each time); backend
   extracts per-keystroke timing features, embeds each, stores the pooled embedding as
   `{person_id, name, embedding}` in the gallery.
2. **Identify**: unknown user transcribes one random prompt; backend extracts features, embeds,
   ranks against gallery, returns top-5 with similarity scores in the UI.

Backend does all feature extraction and model inference (Python/Flask or FastAPI); frontend
JS only captures raw keydown/keyup timestamps, renders the current prompt, and renders results.
Keep the network frozen at demo time — no online learning/retraining triggered by demo usage.
Draw prompts from a pool of sentences disjoint from anything used in training/eval, and never
show the same prompt twice in a row to the same session (defeats the "unseen text" premise).

## Suggested repo structure

```
/data/                  Aalto 136M Keystrokes dataset (raw + cached preprocessed sequences)
/features/              canonical feature-extraction module (imported by both training
                         and the backend inference/demo code — single source of truth)
/model/                 network definition, triplet loss, training script, saved weights
/eval/                  CMC curve, rank-N accuracy scripts, plots
/backend/               API: /enroll, /identify endpoints, gallery storage
/frontend/              keystroke capture UI (prompt display + capture) + results display
CLAUDE.md               this file
README.md               user-facing project description
```

## Things to get right / common pitfalls

- **Feature extraction drift** between training and live capture (see above) — the #1 risk.
- **Evaluating with the wrong metric family.** Accuracy/confusion-matrix framing looks naive
  for this task; CMC/Rank-N is the framing that shows the right mental model.
- **Treating this as a fixed-N classifier** anywhere in the pipeline — defeats the entire
  point of the embedding approach and won't scale to new enrollees.
- **Forgetting the "no match" case.** A real identification system must be able to say "not
  in the gallery," not force a top-1 guess every time — this only requires a threshold on
  the best similarity score, but should be explicit and articulated in the design.
- **Splitting by sample instead of by subject** during evaluation, which silently inflates
  results by leaking identity information the model shouldn't have.
- **Letting the network learn content instead of rhythm.** If key identity is added to the
  input, or if anchor/positive pairs are drawn from the *same* sentence too often, the model
  can shortcut to recognizing sentence-specific timing rather than subject-specific rhythm —
  check this by evaluating on prompts the subject never typed during training/enrollment.
  This is the free-text-specific version of overfitting and is easy to miss because training
  loss still looks fine.
- **Padding/masking bugs in the sequence encoder.** Variable-length input via fixed-length
  padded windows is new complexity versus the old fixed-vector pipeline — make sure padded
  timesteps are actually masked out of the LSTM (not just zero-valued input, which still
  affects gradients if unmasked).
- **Samples too short to carry signal.** A handful of keystrokes isn't enough for reliable
  free-text identification — enforce a minimum prompt/session length (comfortably under the
  `M=50` window, e.g. 25-30 keystrokes) at both enrollment and query time, and reject or warn
  on shorter input rather than silently embedding noise.