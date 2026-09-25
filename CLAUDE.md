# CLAUDE.md — Keystroke Biometrics Identification

**Start here:** [README.md](README.md) for project overview and current state. [ARCHITECTURE.md](ARCHITECTURE.md) for technical design details.

This file gives working conventions and pitfalls for contributing.

## Why This Project Exists

This is a personal CV/portfolio project, built out of genuine interest in AI/ML, cybersecurity, and digital forensics/intelligence—not a client deliverable with a deadline. The point is to actually understand the material over the course of the project, not just have working code appear.

**What this means in practice:**

- Explain *why* before or alongside implementing—the reasoning behind an architecture choice, a metric, a legal caveat, matters as much as the code itself.
- Favor walking through non-obvious decisions over silently making them. If there's a tradeoff worth understanding (e.g., why triplet loss over softmax, why Rank-N over accuracy), surface it rather than picking silently.
- It's fine to slow down for understanding, even where a faster path exists. Don't optimize for "task complete" over "concept understood."
- When asked to build something, it's fair to also explain what was built and why, unless told otherwise for a specific exchange.

**Pacing — work in steps, not one big dump:**

- Break implementation into small, single-purpose steps rather than building an entire phase in one uninterrupted pass.
- End each turn by explaining, in plain language, what changed and why—then stop. Treat that as a checkpoint, not a formality: leave room for questions before moving to the next step, don't plow straight into it unprompted.
- This applies most at meaningful boundaries (a new file, a new concept, a real design decision)—small mechanical follow-ups within an already-explained step don't each need their own pause.
- If asked to "just get it done" or similar for a specific exchange, that overrides this for that exchange only—it doesn't change the default.

## What This Project Is

An open-set 1:N gallery search system for keystroke biometrics. A network maps keystroke sequences to embeddings; the same person's typing clusters close, different people spread apart. New identities enroll later without retraining—enrollment and query are forward pass + ranking only.

Capture protocol: random-prompt transcription (read-then-copy sentences), not free composition. This is a proxy task, with a known domain gap against freely composed text.

## The Critical Invariant

**Feature extraction must be identical between training and live query.**

`features/extract.py` is the single canonical implementation of timing-vector extraction (HL/IL/PL/RL per keystroke). Both offline training and live `/enroll`/`/identify` endpoints must use this exact function—never reimplement it elsewhere (e.g., in JavaScript). Any divergence creates silent bugs: a model trained on slightly-different features degrades without error.

When extracting features for any dataset or demo, import and call `features.extract.extract_features()` directly. Route live browser events to a backend endpoint, not client-side processing.

## Things to Get Right

- **Feature extraction drift.** The #1 risk. Use the canonical extractor everywhere.
- **Wrong evaluation metric.** CMC curve and Rank-N accuracy are standard for identification search. Don't use classification metrics. Keep FAR/FRR/EER for verification mode only.
- **Evaluating on the wrong split.** Hold out subjects during eval, not samples. Splitting by sample leaks identity information and silently inflates results.
- **Forgetting "no match" case.** A real system thresholds the top similarity score rather than forcing a top-1 pick. Implement the threshold explicitly.
- **Letting the network memorize content.** If key identity is in the input, or if anchor/positive pairs are from the same sentence often, the model learns sentence-specific timing rather than subject-specific rhythm. Evaluate on sentences the subject never typed during training.
- **Padding/masking bugs.** Variable-length sequences padded to M=50 are new complexity. Make sure padded timesteps are actually masked in the LSTM, not just zero-valued input.
- **Minimum sequence length.** Enforce a floor (~25–30 keystrokes) at enrollment and query. Reject or warn on shorter input; don't silently embed noise.
- **Treating this as fixed-N classification.** Using enrolled people as output classes defeats the gallery-search approach and won't scale. The network has no notion of "identity"—only typing-rhythm similarity.

## Design Principles

- **Simplicity over speculation.** No features, abstractions, or error handling for impossible scenarios. Trust framework guarantees; validate only at edges (user input, external APIs).
- **No feature flags or backwards-compatibility shims.** Change the code directly.
- **Comments for WHY, not WHAT.** Only write when non-obvious: invariants, subtle bugs, workarounds.
- **Single source of truth.** One feature extractor, one model inference path. Duplication creates drift.

## Suggested Starting Points

**Model training:**
1. Download/reference Aalto 136M Keystrokes dataset.
2. Implement `features/extract.py` if stubbed—timing-vector extraction with M=50 window.
3. Implement training script: LSTM encoder, triplet loss, train on Aalto.
4. Save weights to `/model`.

**Wire the frontend:**
1. Implement `/enroll` endpoint: extract features from browser events, embed, store gallery.
2. Implement `/identify` endpoint: extract, embed, rank, return top-5.
3. Connect frontend forms to these endpoints.

**Evaluation:**
1. Implement CMC curve and Rank-N accuracy scripts in `/eval`.
2. Split Aalto by subject (disjoint train/eval identities).
3. Report results as CMC plot and Rank-N table.

See [ARCHITECTURE.md](ARCHITECTURE.md) for feature and model specifications.
