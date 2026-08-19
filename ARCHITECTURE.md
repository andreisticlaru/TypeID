# Architecture — TypeID Keystroke Biometrics

## 1:N Gallery Search, Not Classification

This project is fundamentally a **gallery search system** built on learned embeddings, not a multiclass classifier. That distinction drives every architectural choice.

A classifier bakes N identities into the output layer—adding a person means retraining. Instead, train an embedding network once, offline, on a public dataset. The network never sees the people who will be enrolled later. At enrollment and query time, there's no training—only a forward pass and nearest-neighbor search. The network learns a general notion of typing-rhythm similarity that generalizes to unseen people.

**Identification vs. Verification:**

| | Identification | Verification |
|---|---|---|
| Question | Which of N enrolled people typed this? | Is this the claimed person? |
| Operation | Rank gallery by similarity, return top-K | Threshold one distance |
| Metric | Rank-N accuracy, CMC curve | FAR, FRR, EER |

Both modes use the same frozen embedding network `f()`. Only the downstream operation differs.

## Data: Text-Independent Timing Features

**Dataset:** Aalto University "136M Keystrokes" (Dhakal et al., 2018). Public, large-scale, collected under this project's protocol: subjects transcribe random English sentences, repeated over many sessions. Thousands of subjects enable a fully disjoint train/eval split.

Because transcribed text differs every session, fixed-position features don't work. The feature representation must be text-independent:

**Per-keystroke timing vectors (4 dimensions each):**
- **HL** (hold latency / dwell): `release(n) - press(n)`
- **IL** (inter-key latency / flight): `press(n+1) - release(n)`
- **PL** (press latency): `press(n+1) - press(n)`
- **RL** (release latency): `release(n+1) - release(n)`

One vector per keystroke yields a variable-length sequence. Omit key identity from input by default (TypeNet design choice)—a model seeing only timing is forced to learn typing rhythm rather than memorizing content, which is what generalizes to unseen text.

**Fixed window:** Pad/truncate sequences to `M = 50` keystrokes. Longer samples are split into non-overlapping windows; shorter ones are zero-padded with a mask. Keep this value identical between training and live capture.

**Backspace handling:** Keep backspace as a real keystroke in the sequence. Self-correction patterns are weak biometric signal; dropping them creates a subtle train/inference mismatch.

**Live capture format:** Browser logs `{key, event_type: keydown|keyup, timestamp}`. Backend feature extractor produces the same HL/IL/PL/RL sequence used in training—timing features are computed server-side, never in JavaScript.

### Pipeline walkthrough: two sources, one shape

Two things produce raw keystroke data: the Aalto dataset (training) and the browser (live enroll/identify). They arrive in different shapes but both get converted to the same thing before any math happens: an ordered list of `(press_time, release_time)` pairs, one per keystroke.

- `data/aalto_loader.py` reads Aalto's `PRESS_TIME`/`RELEASE_TIME` columns straight into pairs—already matched.
- `features/extract.py::pair_browser_events()` matches a flat stream of separate `keydown`/`keyup` events into pairs, tracking open presses per key (needed because keys can overlap, e.g. Shift held while W is pressed and released).

Once both sources produce that pair list, they funnel into the same function—`features/extract.py::windows_from_keystrokes()`—which is the one place the HL/IL/PL/RL math and windowing/masking live. This convergence is what makes the "one canonical extractor" invariant enforceable: there is only one function that could drift.

**Worked example.** Two real rows from Aalto participant 100001, session 1:

```
SHIFT   press=1473275372512   release=1473275372663
W       press=1473275372583   release=1473275372703
```

Treating SHIFT as keystroke `n` and W as `n+1`:

```
HL = 1473275372663 - 1473275372512 = 151   (SHIFT held for 151ms)
IL = 1473275372583 - 1473275372663 = -80   (W pressed 80ms BEFORE SHIFT released)
PL = 1473275372583 - 1473275372512 = 71
RL = 1473275372703 - 1473275372663 = 40
```

The negative IL is real, not a bug: it means W was pressed while SHIFT was still down—normal rollover when typing a capital letter fast. The extractor doesn't clip this to zero; it's genuine rhythm signal.

**Why N keystrokes yield N−1 vectors.** HL(n) only needs keystroke `n`, but IL/PL/RL(n) all need keystroke `n+1` too. The last keystroke in a session has no "next" to pair against, so it doesn't get a complete vector. A session of N keystrokes produces N−1 feature vectors—a deliberate boundary decision (`features/extract.py:92-96`), not an accident. The alternative would mean inventing a value for something that isn't actually measurable.

**Windowing and the mask, concretely.** A short session (say 25 vectors) becomes one window with the remaining 25 slots zero-padded. A long session (say 51 vectors) becomes two windows: the first holds 50 real vectors, the second holds 1 real vector plus 49 padding. Alongside every window, a same-shaped boolean mask records which slots are real (`True`) vs. padding (`False`). The padding is literal zeros—"0ms hold, 0ms gap"—which is impossible for genuine typing. Without the mask, the LSTM has no way to distinguish real zero-latency data (never happens) from "no more data here, ignore this." The mask is what lets training and inference process only the real timesteps.

**The floor.** Before any of this runs, `windows_from_keystrokes()` checks `len(pairs) >= MIN_KEYSTROKES + 1` (26 pairs → 25 vectors). Below that, it raises `ValueError` instead of producing a window—too little rhythm to trust, rejected outright rather than fed to the model as noise.

## Model: Sequence Encoder with Triplet Loss

**Architecture:**
```
Input: sequence of M=50 timing vectors, each (HL, IL, PL, RL) [zero-padded + masked]
  -> LSTM(128) + dropout
  -> LSTM(128) + dropout
  -> take final hidden state (or masked mean-pool over timesteps)
  -> Dense(embedding_dim=128) + L2 normalization
```

One set of weights. "Siamese" / "triplet network" refers to running this identical encoder on multiple training samples, not multiple networks.

**Loss: Triplet loss with margin**

```
L = max(0, d(f(A), f(P)) - d(f(A), f(N)) + margin)
```

- **Anchor (A) and Positive (P):** Two different sessions from the same subject, transcribing different sentences. Using *different* sentences forces the network to learn subject-specific rhythm, not sentence-specific timing.
- **Negative (N):** Sample from a different subject.
- **Margin:** Start ~0.2–1.0. Triplets already satisfying the margin contribute zero loss—this is expected and fine.

**Triplet construction:**
1. Sample subject X, draw two of their sessions as (A, P).
2. Sample different subject Y, draw one session as N.
3. Resample fresh triplets per batch/epoch, not precomputed.
4. Optional refinement: semi-hard/hard negative mining after warmup epochs.

**Training:** Adam, lr ≈ 1e-3, batches of 32–128 triplets. Sequence encoding is slower than fixed-vector approaches—plan for GPU or scaled-down subject subset on CPU.

**Output:** Single frozen function `f(keystroke event sequence) -> 128-dim embedding`. Everything downstream (enrollment, identification) is forward pass + vector comparison.

## Enrollment and Identification

**Enrollment:**
```
New person transcribes 2-3 random prompts
  -> feature_extract() -> f() -> embedding
  -> store as {person_id, name, embedding, enrolled_at}
```

**Identification:**
```python
def identify(query_features, gallery, top_k=5, min_confidence=None):
    query_emb = f(query_features)
    scored = [
        (entry.person_id, entry.name, cosine_similarity(query_emb, entry.embedding))
        for entry in gallery
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    results = scored[:top_k]
    if min_confidence and results[0][2] < min_confidence:
        results = []  # "no confident match"
    return results
```

If a query produces multiple windows, embed each and average (or pool) the resulting embeddings before ranking.

**Gallery storage:** For this project's scale, a plain SQLite table suffices: `{person_id, name, embedding (128 floats), enrolled_at}`. FAISS/Annoy only matter at thousands of entries.

## Evaluation: Rank-N Metrics

Use identification task metrics, not classification metrics:

- **Rank-N accuracy:** Fraction of queries where the true identity appears in the top-N results.
- **CMC curve (Cumulative Match Characteristic):** Rank-N accuracy plotted as N grows. Standard for fingerprint/face-recognition identification; include this plot.

Keep FAR/FRR/EER reserved for verification mode (if built).

**Held-out protocol:** Split Aalto by subject, not session. Train on one subject set, build eval gallery from a disjoint set. This tests generalization to unseen identities, not just unseen sessions of trained people.

## Critical Invariant: Feature Extraction Parity

The feature extraction function must be byte-for-byte identical (or at least dimensionally and semantically identical) whether processing:
1. Aalto dataset rows during training
2. Raw browser keydown/keyup events at enrollment/query

Do not maintain two separate implementations (Python for training, JavaScript for demo). Route live browser events to a backend endpoint running the canonical Python feature extractor. Divergence here is the single most likely source of silent bugs—a model trained on slightly different features will degrade quietly rather than error loudly.

## Things to Get Right

- **Feature extraction drift** between training and live capture (see above).
- **Evaluating with the wrong metric family.** CMC/Rank-N framing shows understanding; accuracy/confusion-matrix framing looks naive for a search task.
- **Treating this as a fixed-N classifier.** Defeats the embedding approach and won't scale to new enrollees.
- **Forgetting the "no match" case.** A real identification system must threshold the best similarity score rather than force a top-1 pick every time.
- **Splitting by sample instead of subject during evaluation.** Silently inflates results by leaking identity information the model shouldn't have.
- **Letting the network learn content instead of rhythm.** If key identity is added to input, or if anchor/positive pairs are drawn from the same sentence too often, the model shortcuts to recognizing sentence-specific timing. Check by evaluating on prompts the subject never typed during training.
- **Padding/masking bugs in the sequence encoder.** Variable-length input via fixed-length padded windows is new complexity—make sure padded timesteps are actually masked in the LSTM, not just zero-valued input.
- **Samples too short to carry signal.** Enforce a minimum prompt/session length (under `M=50`, e.g., 25–30 keystrokes) at enrollment and query. Reject or warn on shorter input rather than silently embedding noise.
