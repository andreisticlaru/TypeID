# Type2Branch plan: the RNN + CNN baseline, on our protocols

Status (2026-11-04): proposed, nothing started. Offline baseline for `shift-plan.md` step 8 and `paper-plan.md` RQ3.
It never goes in the live app: it reads key identity, which the live model must not.

## Why

- It leads KVC-onGoing, so it is the bar `paper-plan.md` says TypeNet no longer is.
- On KLiCKe our recipe's lead disappears across tasks: C1 beats TypeNet m0 by 8.2 EER on copy → copy and loses by 2.0 on
  copy → essay. Architecture may matter under shift, and this is the strongest architecture to test.
- RQ3: do key-code models degrade more when the text changes? We assumed yes. The paper's own ablation hints otherwise
  (no key-code embedding: only +0.9% EER), so this is a real question.

## What it is

González, Stragapede, Vera-Rodriguez, Tolosana, TIFS 2025, [arXiv 2405.01088](https://arxiv.org/abs/2405.01088). Code:
[lsia/tifs-type2branch](https://github.com/lsia/tifs-type2branch), GPL-3.0, Keras 2.11 / TF 2.12, no weights listed.

| Part | Detail |
|---|---|
| Input (5 per key) | Key code, HT (hold), FT (press to press), SHT and SFT (each minus a population model's synthetic timing). Seconds, clipped at 10. |
| Recurrent branch | Temporal self-attention, then 2 bidirectional GRU layers, 512 units |
| Convolutional branch | Temporal attention, then 3 conv blocks (256, 512, 1024 filters, kernel 6) with channel attention, then global average pooling |
| Embedding | Concatenate, 3 dense layers of 512, 64-d output, Euclidean distance. BatchNorm and dropout 0.5 after each layer. |
| Set2set loss | SetMargin triplets over all sets in a batch + 0.05 × a radius penalty |
| Batch | K = 40 people × N = 15 samples = 600 sequences |
| Curriculum | From epoch 1, each batch: one person, their m nearest people by embedding centre (recomputed each epoch), the rest random |
| Training | Adam 1e-4, 115,120 Aalto people, 18 epochs of 20,000 steps in the paper (config: 25,000), on a 40 GB A100 |
| Results | Desktop: mean per-subject EER 0.77, global-threshold EER 3.25, 15,000 people |

Ablation worth knowing (reduced model, baseline 6.51): unidirectional RNN +25.1%, triplet K=10 +30.1%, RNN branch only
+5.7%, CNN branch only +20.4%, no synthetic features +5.6%, no curriculum +7.7%.
So bidirectionality and the set loss matter most. The RNN alone gets most of the way, and it is the cheap fallback.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Run upstream or port | **Port to PyTorch**, `type2branch/` beside `typenet/` | TF above 2.10 has no native Windows GPU support, and our eval stack is PyTorch. Upstream is the reference for reading, not a dependency. |
| Licence | Keep attribution in the port's header | A port of GPL-3.0 code is a derivative. This repo is AGPL-3.0, and GPLv3 §13 allows combining the two; check this before publishing. |
| Features | HT and FT come from `windows_from_keystrokes` columns (HL, PL, shifted one step); only the key codes are new | Keeps one timing implementation. Models get a 5-column cache, old caches stay as they are. |
| Same people | Add an optional per-model cache to `comparison.paired` and `comparison.task_gap` | Scores on one tensor per model, so paired bootstrap keeps working. |
| KLiCKe key codes | Map `DownEvent` names to Aalto's JS key codes in `data_v2/klicke.py` | KLiCKe has names, not codes. Shift and Control stay dropped, so T2B sees them on Aalto but not on KLiCKe: report it, and run the T2B-no-keys control. |

## Steps

| # | Step | Output | Gate |
|---|---|---|---|
| 0 | Read upstream `model.py`, `loss.py`, `training_generator.py`, `generate_dataset.py`, `conf.small.1Kusers.py`. Answer: M, margin α, m, key embedding size, and whether the synthetic-feature tool is obtainable (`merge_synth_features.py` takes externally computed features). | Notes at the end of this file | Tool unavailable → no SHT/SFT in v1, and we report that as a known handicap (about +5.6%). |
| 1 | Key-code caches: Aalto `data/preprocessed_keys`, KLiCKe and KeyRecs `*_keys`. New folders only; never `data/preprocessed/`. | Caches, tests | HL and PL columns byte-identical to the existing caches. |
| 2 | Port: `type2branch/network.py`, `losses.py`, `sampler.py`, `train.py`, tests like `typenet/test_*`. Register the arch in `comparison/loaders.py`. | Tests pass | **Masking test:** padded steps must not change the embedding. The backward GRU reads padding first, so it needs packed sequences, and attention, BatchNorm and pooling must be masked too. |
| 3 | Fit test on the 4 GB GPU: one step at K=40, N=15, full width; measure memory and s/step. | Budget table | Doesn't fit → in order: mixed precision, width 256, RNN branch only. Cutting K is last, because the set loss drives the result. A batch can't simply be split, since the loss spans all sets. |
| 4 | Fidelity: train the small config on 1,000 people, score mean per-subject EER on 1,000 unseen. | `runs/t2b_small` | Within about 1 point of the paper's 6.51. Otherwise fix the port before any long run. |
| 5 | Full run on `data/split_kvc_scale.json` (115k people, as C1_115k), then a second seed. | `runs/t2b_full*` | Step 3 sets the budget. If one run needs over about 3 days, stop and decide with the user. Aalto global EER should beat C1_115k; if not, suspect the port. |
| 6 | Control: same model and budget with key codes zeroed (`--no-keys`). | `runs/t2b_nokeys` | Separates the architecture's effect from the key codes'. |
| 7 | Score, dev data only: Aalto `unseen_nf`; KeyRecs `within`, `kvc_cross`; KLiCKe dev `copy`, `copy_essay`, `copy_essay_id`, `essay`; plus `comparison.backend`. | `paired_*`, `task_gap_*`, `backend_*` JSONs | See below. |
| 8 | Join `shift-plan.md` step 8. The KLiCKe test half is still scored only once, with everything else. | Paper table | — |

One GPU job at a time. Long runs in the background, logs in `runs/`.

## How to read step 7

| Question | Comparison | Answer if |
|---|---|---|
| Does architecture help under task shift? | T2B-no-keys vs C1 and TypeNet r2, copy → essay | Paired CI excludes 0 |
| Do key codes hurt under shift? (RQ3) | (T2B gap) − (T2B-no-keys gap), gap = copy → essay − copy → copy | Paired CI excludes 0 and above 0 |
| Do they help when the text matches? | T2B vs T2B-no-keys, Aalto and copy → copy | Paired CI excludes 0 |
| Is the back-end still needed? | T2B + `center+wccn` vs plain T2B, copy → essay | Paired CI excludes 0 |

Read `copy → copy` with care for key-code models: everyone types the same phrases, so it is the condition most
favourable to them. Report identification too (Rank-N, DIR), not only EER: the rankings differ between the two.

## Comparing with the paper

Their 3.25 is not comparable to our EERs: KVC draws half its impostors from the same age and gender group, and ours are
random. Only a KVC-onGoing submission settles that (`kvc-plan.md`). KLiCKe has age and gender, so same-group impostors
could be added there if we want a harder number.

## Risks

- **Size is the main risk.** 600 sequences through 512-unit bidirectional GRUs and 1024 filters may not fit in 4 GB, and
  a run that took an A100 18 epochs may take far too long here. Step 3 measures this before anything else is spent. A
  reduced model is a fair baseline only if it is named as reduced.
- **A weak port looks like a weak model.** Steps 2 and 4 exist so a bug isn't reported as a finding.
- **Scope.** The paper's question is shift, not the architecture race (`shift-plan.md`). This is a baseline: no tuning
  beyond getting the port faithful.
