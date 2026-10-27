# Explain plan: make the embedding say what it encodes, in words

Status (2026-10-27): not started. GPU is idle while KVC Phase 2 waits for registration; this plan's GPU work
must queue behind any KVC run (one GPU job at a time).

## Goal

Teach a small LLM to read TypeID's embedding and describe the typist ("fast, very regular rhythm, heavy key
rollover") and explain a match ("Q matched A over B because both hold keys long and pause rarely"). The
explanations must be **faithful**: they describe what the embedding kept, not what is in the raw input.

The model is built in three stages, and each one has to pass a gate before the next starts. Most of the scientific
value is in Phases 1–2, which need no LLM. The LLM (Phase 3) only adds fluency on top of what the probes show is
decodable, so if the probes find nothing, we stop there and that is the result.

## Prior work (from the 2026-10-27 search)

- **Queen** (Princeton, arXiv 2610.03695): frozen Leela chess encoder + frozen SmolLM3-3B, joined by a gated
  cross-attention bridge (tanh gate at 0, decoder layer 2i reads encoder layer i). QA curriculum with
  programmatically verified answers, then self-distillation. Weakness: misnames or invents motifs; "cannot teach
  it to verbalize motifs it hasn't seen" (conceptual coherence 2.76 vs 3.61 for a frontier LLM).
- **ELM** (Google, ICLR 2024): an adapter maps a domain embedding vector into the LLM's token space. Closest to
  our case: one vector in, text out.
- **SensorLLM** (EMNLP 2025): a sensor encoder feeds an LLM, with training text generated from statistics and
  templates. Closest in data type: timing signals, no human labels.
- **LatentQA** (ICLR 2026): a decoder LLM answers questions about another model's activations.

What changes for us: chess comes with hundreds of named, checkable concepts, and typing timing has none. We define
the vocabulary ourselves (Phase 1). That caps what the system can name, which is Queen's limitation in our
setting. Phase 2b measures how much of the embedding the vocabulary covers.

## Decisions to confirm before Phase 1

1. **Which encoder.** Default: `model/encoder_hard.pt` (v1 `KeystrokeEncoder`, the live model), so explanations
   match the webapp's decisions. Also run Phase 1–2 on C1_s1 (`runs/p6_c_f_s1`, TypeNet arch), which is cheap.
   Do the two models encode different concepts?
2. **Which LLM** (Phase 3 only): Qwen2.5-0.5B-Instruct (first choice) or SmolLM2-360M-Instruct. Both fit the
   4 GB GPU in bf16 with LoRA. A 1.5B model needs 4-bit and is a later stretch.
3. **Environment.** `transformers`/`peft` are not in `.venv-model`, and the live backend runs from it. Use a new
   `.venv-explain` (same torch 2.6.0+cu124) so the backend's environment never changes.

## Data

- Read-only: `data/preprocessed/{windows,mask,subject_ids,session_ids}.npy` (2,483,167 windows) and
  `data/split.json` (151,734 train / 16,859 eval subjects). Never write to either.
- Use **eval subjects only**. The encoder never saw them, so probes measure generalisation and not memorisation.
  Split them again by subject: 70% probe-train / 30% probe-test, seed fixed, saved to `runs/explain/split.json`.
- Features are the HL/IL/PL/RL windows that `features/extract.py` already produced. Concepts are statistics *of
  those arrays*, so the timing math is never recomputed from press/release times. In the live path, concepts
  come from `extract_features(events)` output, through the same function.
- No key identity in the input means no key concepts ("slow on Shift"). Timing concepts only.

## Phase 1: concept vocabulary (CPU, ~1 day)

`explain/concepts.py`: `concepts(windows, mask) -> (n, C)` float array, masked, one column per concept. First set:

| Concept | From |
|---|---|
| speed | median PL (also as an approximate WPM) |
| hold time | mean, std of HL |
| gap | mean IL |
| rollover rate | fraction of IL < 0 (next key pressed before this one released) |
| rhythm regularity | coefficient of variation of PL |
| pause rate | fraction of PL > 2× the window's median PL |
| burstiness | ratio of PL interquartile range to median PL |
| tempo drift | slope of PL across the window (speeding up / slowing down) |
| hold/flight ratio | mean HL / mean PL |
| release order | fraction of RL < PL (releases bunch up vs follow presses) |

Each concept gets two levels:
- **window level**: computed on one 50-keystroke window;
- **subject level**: mean over all of that subject's windows.

The embedding is trained to be the same across a person's sessions, so it should encode stable traits and discard
per-window noise. Comparing the two levels tests that directly.

Tests (`explain/test_concepts.py`): hand-built windows with known answers (constant PL gives regularity 0, one
negative IL in 10 gives rollover 0.1), and padding must not change any value.
Verify: tests pass. Concept distributions on probe-train look sane (median PL ≈ 150–250 ms for Aalto).

## Phase 2: probes, the gate for everything after (GPU for embedding only, minutes; rest CPU)

`explain/probe.py`:
1. Embed all eval-subject windows once and cache them to `runs/explain/emb_<model>.npy`. This is the only GPU
   step, and it is short.
2. **2a, decodability:** for each concept and level, fit ridge regression from embedding to concept on
   probe-train and report R² on probe-test. Also fit a 3-bucket classifier (low/mid/high by probe-train
   tertiles) and report its accuracy, because bucket answers are what the LLM will say.
3. Controls, on every row of the table:
   - **floor:** the same probe on embeddings from a random-init encoder of the same architecture (what
     "decodable" means by chance through a random LSTM);
   - **ceiling:** a probe on simple pooled raw features (the information available in the input).
   A concept counts as "encoded" if probe R² is well above floor (target ≥ 0.3 and ≥ 2× floor). Tune the
   threshold after seeing the numbers, but write down the rule before looking.
4. **2b, coverage (the naming problem):** reverse regression from all concepts to the embedding gives the
   fraction of embedding variance the vocabulary explains. Then take the top-k PCA directions of the embedding
   and label each by its most correlated concept, or "unnamed" if |r| < 0.3. Unnamed directions with high
   variance are the concepts we are missing. Inspect them (sort windows along the direction and look at the
   extremes), name what we can, and add those names to the vocabulary.

Output: `runs/explain/probes_<model>.json` and a results table in this plan.
**Gate:** at least ~5 concepts encoded, at subject level, on the live model. If fewer, stop. The finding is
"the embedding is mostly not human-nameable with timing statistics", which is still worth writing up.

## Phase 3: LLM decoder (GPU, hours per run; only if Phase 2 passes)

Architecture (ELM / LLaVA style, simplest that matches the goal):
- Frozen encoder → 128-d embedding → trainable MLP adapter → k soft tokens (start k = 4), prepended to the
  prompt. The adapter is zero-initialised at the output, which is the same idea as Queen's zero gate.
- Frozen LLM in stage 3a (only the adapter trains), LoRA on the LLM in stage 3b only if 3a plateaus.
- Later option, closer to Queen: per-timestep hidden states from `lstm2` (50 × 128, via a forward hook, so the
  encoder code is never edited) as 50 soft tokens. Try it only if the single vector loses concepts that the
  probes say are present in the hidden states.

QA curriculum, generated by `explain/qa.py` from Phase 1 concepts (encoded ones only), with answers checked by
code like Queen's:
1. **Single bucket:** "Is this typist's rollover low, medium or high?"
2. **Description:** "Describe this typist's rhythm." The answer is a template sentence from the bucket values.
   Templates are varied so the LLM learns content and not one fixed phrasing.
3. **Comparison**, with two embeddings: "Who types faster, A or B?"
4. **Match explanation**, with a query and its top-2 gallery entries: which encoded concepts make the query closer
   to A than B. The answer lists concepts where |query − A| < |query − B|.

Evaluation (probe-test subjects, exact-match on bucket answers):
- **Faithfulness:** the LLM's bucket accuracy should approach the probe's and cannot meaningfully exceed it. If it
  beats the probe, it is reading something other than the embedding, so check for leakage.
- **Shuffled control:** pair each question with a random other subject's embedding. Accuracy must drop to
  chance, otherwise it answers from priors or from the prompt.
- **Held-out concept:** train without one concept and ask about it. We expect failure. This reproduces Queen's
  "can't name motifs it hasn't seen" on our data and is stated as a limitation, not hidden.
- **Spot-read 50 free-text descriptions** for invented concepts (Queen's failure mode).

Outputs: `runs/explain/llm_<run>/`. The adapter is saved as a new file and never touches `model/`.

## Out of scope until asked

- A webapp "why this match" panel. It would need a new backend endpoint and model loading in the live
  backend, so it gets its own plan.
- Self-distillation (Queen's Bellman-style loop). It improved Queen's *playing strength*, and our encoder is
  frozen, so there is no equivalent to improve yet.

## Rules

- New code lives in a new `explain/` package. No edits to `features/`, `model/`, `typenet/`, `data/`, `backend/`.
- Never write to `data/preprocessed/`, `data/split.json`, `model/encoder_hard.pt`, or the gallery DBs.
- Everything generated goes in `runs/explain/` (gitignored). No `*.pt` committed.
- One GPU job at a time, queued behind KVC if it is running. Long runs go in the background with a watched log.
- Probes split by subject, never by window.

## Time

Phase 1: ~1 day. Phase 2: ~1 day (CPU-bound, plus discussion of results). Phase 3: ~3–5 days including the
curriculum, with each training run a few hours on the A2000.
