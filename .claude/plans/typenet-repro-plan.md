# Plan: TypeNet reproduction + v2 encoder + controlled comparison (TODO Phase 7, item 4a)

## Context

The README compares our hard-mined model (89.8% Rank-1 under `eval/typenet_protocol.py`) with TypeNet's
*published* 67.4%. That mixes architecture, training recipe, mining, compute, subjects and TypeNet's key-code
input. The goal is twofold:
1. **A fair comparison:** retrain TypeNet's architecture and recipe on *our* data, split and 4 timing
   features, then measure what mining and our design add at **matched compute**.
2. **The best possible model:** a from-scratch v2 of our encoder that borrows the paper's useful ingredients
   (input scaling to seconds, large batch) and keeps our mining.

Fixed constraints:
- `features/extract.py` and `data/preprocessed/` stay untouched, and the key code is out of scope.
- Train on the train subjects only (`data/split.json`).
- The backend keeps serving `model/encoder_hard.pt`.
- **All new code lives in new folders; `model/` and `eval/` are not edited.**

## Runs and what each comparison shows

| Run | Architecture | Negatives | Batch | Steps |
|---|---|---|---|---|
| **A** TypeNet reproduction | TypeNet (§4.2) | random | 512 triplets | 0 → T, checkpoint every 10k |
| **B** TypeNet + mining | TypeNet | random until 10k, then hardest | 512 | resumes A's 10k checkpoint → T |
| **v2** | our encoder + ms→s scaling | random until 10k, then hardest | 512 | 0 → T, checkpoint every 10k |
| Ablation pair (short) | our encoder, scale 1 vs 1000 | random | 64 | 20k each, same seed |

- **A @ 30k** is exactly TypeNet as published (200 epochs × 150 batches × 512). A uses a constant lr and no
  schedule, so a longer run is identical up to 30k.
- **B vs A** shows the effect of mining on their architecture. **v2 vs B** shows the effect of architecture
  and loss under the same mining. **v2 vs A** compares best against best at 30k and again at T.
- **T** (the total budget) is set after the timing smoke tests. A tentative value is 100k steps (≈51M
  triplets; `encoder_hard` saw 32M).
- **Why hardest from 10k:** v1's random-negative phase plateaued at ~90k steps × 64 ≈ 5.8M triplets, which
  is ≈11k steps at batch 512. Starting mining before 30k also means the 30k checkpoints of B and v2 already
  include mining.
- The schedule is in absolute steps, so every intermediate checkpoint is a faithful truncation of its run.

## Folder layout (all new)

```
typenet/                 TypeNet architecture + recipe (runs A, B)
  __init__.py
  network.py             TypeNetEncoder, MaskedBatchNorm, LSTM layer with recurrent dropout
  losses.py              squared-Euclidean triplet loss + hardest-negative miner (Euclidean)
  train.py               training loop: --steps --save-every --hard-from --resume --seed --out
  test_network.py        unittest: masking, readout, param count
  checkpoints/           encoder_typenet_a*.pt, encoder_typenet_b*.pt (gitignored)
model_v2/                our encoder, v2 (v2 run + ablation pair)
  __init__.py
  network.py             KeystrokeEncoderV2(KeystrokeEncoder) with input_scale
  train.py               training loop: --batch-size --hard-from --input-scale --seed ...
  test_network.py        unittest: scaling-folding identity, padding invariance
  checkpoints/           (gitignored)
comparison/              evaluation across all models
  __init__.py
  loaders.py             load_any_encoder(path): builds the class named by ckpt["config"]["arch"]
  evaluate.py            runs both protocols on a list of checkpoints, writes JSON
  results/               one JSON per evaluated checkpoint
```

Only existing files touched: `.gitignore` (the two `checkpoints/*.pt` globs), plus README.md and TODO.md at the end.

## Existing code to reuse (import, don't copy)

- `model/triplet_sampler.py` `TripletSampler`: A/P from two sessions of one subject, N from another, train
  split only.
- `model/train.py`: `build_sampler`, `to_tensors`, `save_checkpoint`, `load_for_resume`,
  `mine_hardest` (cosine; used by v2), `train_step` (v2 can call it directly: cosine loss, `mine="hard"|None`).
- `model/network.py` `KeystrokeEncoder`: base class for v2.
- `eval/rank_n.py`: `load_eval_data`, `embed_rows`, `evaluate_model`, `summarize`, `RANKS_REPORTED`.
- `eval/typenet_protocol.py`: `session_embeddings`, `draw_background`, `identification_ranks`,
  `authentication_eer`, `ENROLLMENT_SIZES`.
- `features/test_extract.py`: unittest style to follow.

## Step 0: branch
You commit the `feat/authenticate` work, then I create `feat/typenet-repro` from it. Nothing gets committed
or pushed unless you ask.

## Step 1: TypeNet network (`typenet/network.py`) + tests → **checkpoint**

`TypeNetEncoder.forward(x_ms, mask)`:
1. `x = x_ms / 1000` (the paper feeds seconds, §4.1). The scale is fixed inside the model so any caller gets it.
2. `MaskedBatchNorm(4)`: batch statistics from real timesteps only. Uses Keras defaults (momentum 0.99 ≡ torch
   0.01, eps 1e-3) and keeps running stats for eval.
3. LSTM(128, tanh) with **recurrent dropout 0.2**. The input projection `x @ W_ih` is precomputed for all
   timesteps; a Python loop over t applies one dropout mask per sequence to h_{t-1} (Keras/Gal style).
   Initialization follows Keras: glorot-uniform input weights, orthogonal recurrent weights, zero bias with
   forget-gate bias 1.
4. `MaskedBatchNorm(128)` → Dropout(0.5) → LSTM(128), recurrent dropout 0.2.
5. Readout: **h at the last real timestep** (`mask.sum(1) - 1`, via gather). This is exact because padding
   is always at the end and the LSTM is unidirectional. No Linear layer and no normalization.

Config saved with checkpoints: `{"arch": "typenet", "model": {...kwargs}}`.

Tests (`python -m unittest typenet.test_network -v`):
- **Padding can't leak (eval):** random values in padded slots give an identical embedding.
- **Padding can't leak (train-mode BatchNorm statistics):** with dropouts set to 0, two batches that differ
  only in padded values give identical outputs.
- **Readout is the true last step:** a length-L window padded to 50 gives the same embedding as the same L
  steps with no padding.
- **Parameter count = 200,968.** That is the paper's 200,458 minus 10 key-code weights plus PyTorch's
  second LSTM bias (+1,024); the test asserts the figure and its docstring explains the arithmetic.

## Step 2: TypeNet loss + training (`typenet/losses.py`, `typenet/train.py`) → **checkpoint**

- Loss: `max(0, ||a−p||² − ||a−n||² + 1.5)`, averaged (Eq. 4).
- Hardest miner: same candidate pool (2B) and same-subject mask as `model/train.py:mine_hardest`, but with
  squared Euclidean distance. Only used when `step > --hard-from`.
- Adam: lr 0.05, betas (0.9, 0.999), eps 1e-8. Batch 512 triplets. Seeds set for `np.random` and torch.
  `save_checkpoint` stores the optimizer state and step.
- Smoke test: 300 steps. Loss should fall and must not diverge at lr 0.05; if it does, report it rather than
  quietly fixing it. Time steps 100–300 → seconds/step → wall-time estimate for T. If the Python loop is
  too slow, `torch.jit.script` the layer and retime.

## Step 3: v2 network + training (`model_v2/`) → **checkpoint**

- `KeystrokeEncoderV2(KeystrokeEncoder)`: adds `input_scale` (default 1000), and `forward` divides x before
  calling `super().forward`. Otherwise identical: mean-pool, Linear, L2 norm, dropout 0.2.
  Config: `{"arch": "v2", "model": {..., "input_scale": 1000}}`.
- `train.py`: reuses `model.train.train_step` / `build_sampler` / `save_checkpoint`; `mine = "hard" if
  step > hard_from else None`; lr 1e-3, margin 0.5 (cosine), batch 512.
- Tests: **folding identity.** V2 with scale 1000 and `lstm1.weight_ih × 1000` gives the same output as v1.
  This shows scaling changes optimization, not what the model can represent. Also a padding-invariance test.
- Smoke test plus timing, as in Step 2.

## Step 4: comparison tooling (`comparison/`) → **checkpoint**

- `loaders.load_any_encoder(path, device)`: reads `config["arch"]` (missing → v1 `KeystrokeEncoder`), builds
  the class, loads weights, and puts the model in eval mode.
- `evaluate.py --checkpoint ... --seeds 0..9`: for each checkpoint:
  - The TypeNet protocol (Rank-1/5/20, EER at G = 1/2/5/7/10), via the reused functions.
    **Metric: Euclidean distance on raw embeddings** (TypeNet's own rule).
  - The strict 3-enroll + 1-query Rank-N, via `evaluate_model` with `score="mean"`.
    **Metric: cosine on the averaged, renormalized profile**, which is not TypeNet's metric; labelled as such.
  - Writes `comparison/results/<checkpoint>.json`: mean ± std per metric, step, triplets seen, arch.
- Sanity check: running it on `model/encoder_hard.pt` and `model/encoder.pt` must reproduce the README's
  89.8% / 60.5% and 56.9% / 28.4%. That proves the new driver matches the old scripts.
- (`rank_n.py`'s `--score pairwise` assumes unit vectors, so it is never used for TypeNet embeddings.)

## Step 5: runs (long, background, one at a time on the RTX A2000)

1. Ablation pair (20k × 64, scale 1 vs 1000, seed 0) → evaluate strict Rank-N → **checkpoint**
   (does scaling help on its own?).
2. A to T.
3. B, resumed from A @ 10k, to T.
4. v2 to T.

Each run gets a report at completion: loss curve and the 30k/T numbers.

## Step 6: evaluate + document → **checkpoint**

- Evaluate A, B and v2 at every 10k checkpoint, plus v1 references (`encoder.pt`, `encoder_hard.pt`).
- README: a new table next to the existing TypeNet one, with rows for TypeNet published (Table 2 row M=50 /
  Table 5), A@30k, A@T, B@30k, B@T, v1 random baseline, v1 hardest, v2@30k, v2@T. Columns: Rank-1 and EER
  at G = 1/5/10, triplets seen, and the metric label.
- Plain-language interpretation: what each controlled pair shows, what it doesn't (one training seed per
  run; subjects and windowing are ours, not theirs; no key code).
- TODO.md item 4: record what was done. If v2 and A/B land within a few points, flag retraining with 2–3
  seeds before claiming a winner.

## Verification

- `python -m unittest typenet.test_network model_v2.test_network features.test_extract -v` all pass.
- The smoke runs show loss decreasing, with no NaN.
- The comparison driver reproduces the README's v1 numbers exactly (same seeds).
- `git diff --stat` touches no file under `model/`, `eval/`, `features/`, `data/` or `backend/`.
- The backend still starts and loads `model/encoder_hard.pt` (it's unchanged, but check once).
