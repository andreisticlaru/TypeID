# Paper plan: one transcription-trained embedding, three decisions, three shifts

Short actionable version: `shift-plan.md`. This file keeps the literature and the reasoning.

Status (2026-10-27): proposal, nothing started. Built on a literature sweep done the same day (sources at the end).
arXiv was blocked from the research session, so most claims below come from abstracts and summaries, not full
texts. Read the full papers marked **(verify)** before citing them.

## 1. The real picture (what the field actually does)

### Methods in use

| Use | Academic state of the art | Industry practice |
|---|---|---|
| One-time verification (login, KVC) | **Type2Branch** (González, Stragapede et al., TIFS 2025): key code + hold/flight + 2 "deviation from population" features, RNN+CNN dual branch with attention, Set2set loss, curriculum. KVC-onGoing desktop global EER 3.33%, FNMR@1% 11.96%. **EERLoss** (Jun 2026, same group): differentiable EER loss, up to ~30% lower EER on the same architecture **(verify)**. TypeFormer for mobile. | TypingDNA: "sametext" vs "anytext" patterns, statistics over the 44 most-used keys (320-value vector), no key sequences stored so text can't be rebuilt; 60–70 characters minimum; sold as a second factor, never on its own. |
| Continuous authentication | Trust models (Mondal & Bours): a confidence score updated per action, lock-out under a threshold. Metrics ANGA / ANIA (actions before a genuine user is locked out / an impostor is caught). Mostly small free-text sets (Clarkson II, Buffalo), EER 2.8% at 190 keys to 8.3% at 30 keys (TKCA on Clarkson II). | BehavioSec (LexisNexis): per-key-pair timing, a 0–100 trust score per session, step-up auth on risky actions. BioCatch (Visa, $2.4B): keystrokes are one signal among mouse, touch, sensors and navigation. |
| Identification / forensics | Tiny studies (9 participants, 2024). TypeNet's 100k-gallery identification (2020) is still the only large-scale result. No keystroke paper reports **likelihood ratios or Cllr**, the standard in forensic speaker and authorship work. Admissibility (Daubert) untested. | Patents on browser-side capture; Whonix documents de-anonymisation across sites, even over Tor. No product sells 1:N. |
| Authorship, not identity | Condrey (2026): the correlation between word difficulty and the preceding pause is ρ 0.35–0.55 for composition and < 0.15 for transcription **(verify)**. Timing-forgery attacks evade simple detectors ≥ 99.8%. | Exam proctoring and LLM-cheating detection. |

### The three shifts, and what is known about each

1. **Session (enrol today, query later).**
   - Parkinson et al. (Sep 2026), 8 weeks of fixed passwords: EER rises steadily, about 1.7 points per week, for every model including TypeNet- and TypeFormer-style ones **(verify)**.
   - Your KeyRecs result: the drift is a **step at the session boundary**, not a slope, and nothing in the training recipe moves it.
   - Nobody has published a fix for deep keystroke embeddings beyond "re-enrol" or adaptive templates (Buffalo tech report 2016, transfer learning, +13%).
2. **Task (transcription vs composition).**
   - Killourhy & Maxion 2012: 20 people, a few ms difference, classifiers unaffected, so "keep using transcription".
   - Newer, smaller studies point the other way:
     - Vietnamese non-original-text detection: temporal features hold up only when the cognitive task matches; rhythmic ones transfer.
     - Chinese Pinyin (EDM 2026): composition has more rhythm variability and longer latencies.
     - Condrey: composition pauses track content; transcription pauses don't.
   - **No one has measured the transfer for a deep embedding trained at Aalto scale.** That gap is real and confirmed by every search.
3. **Device and keyboard.**
   - Cross-device transfer learning on BB-MAS: EER 14.2% (Aug 2026) **(verify)**.
   - Federated cross-keyboard work (2024).
   - Buffalo's "rotation" subset (3 keyboards) has hardly been used with deep models.

### Where you stand (honestly)

- **Against TypeNet:** you clearly win. At equal data, +21.7 Rank-1 on TypeNet's own 100k test set, and KVC-style EER 2.6 vs 5.9.
- **TypeNet is no longer the bar.** Type2Branch and EERLoss are.
  - Your KVC-style 2.56 uses random impostors.
  - KVC draws half its impostors from the same age and gender group, so its 3.33 is a harder test. The two numbers are **not comparable**.
  - Only an official KVC-onGoing submission settles that comparison (kvc-plan Phase 2).
- **Recipe changes alone are not a paper at IJCB/TIFS level** (semi-hard from step 0, no BatchNorm, ms input, hardest-negative phase). They are good ablations inside one.
- **What you have that others don't:**
  - a measured cross-session step;
  - an open-set identification pipeline with calibrated thresholds;
  - a live system;
  - a strict timing-only design.
  - That last one becomes an advantage under shift. Key-code models like Type2Branch should suffer *more* when the text changes from copied to composed. This is testable, and a strong angle.

## 2. Insight to build on

The extractor (`features/extract.py`) feeds **raw milliseconds** with no cap. In transcription, a pause is reading time, which is short and fairly regular. In composition, a pause is thinking time: seconds long, placed at word, clause and sentence boundaries, and driven by content (Condrey's ρ, decades of writing-process research: pauses > 500 ms to 2 s mark planning).

So the task shift concentrates in **the long tail of IL/PL/RL at word boundaries**. An LSTM trained on Aalto has rarely seen a 15,000 ms IL, so these inputs are out of distribution. Hold times (HL) and within-word intervals are motor, and should transfer.

That gives a falsifiable hypothesis: *the transcription→composition gap is carried by cognitive pauses. Removing or simulating them closes most of it, and what remains is the session step.* Either outcome is publishable:
- **the gap closes:** a method;
- **the gap was never the task:** the shift people worry about is the wrong one, and session dominates.

## 3. Proposed methods (new, cheap, testable on a 4 GB GPU)

| # | Method | Idea | Attacks | Cost |
|---|---|---|---|---|
| M1 | **Pause-robust input** | Inside the network's forward pass (not the extractor, so the invariant and old checkpoints stay intact), apply `sign·log1p(x/τ)` or a cap at Aalto's 99th percentile to IL/PL/RL. Inference-only variant first (cap), trained variant second (log). | Task | Hours (cap) / one training run (log) |
| M2 | **Cognitive-pause augmentation (CPA)** | During Aalto training, insert synthetic thinking pauses into the raw (press, release) pairs *before* calling `windows_from_keystrokes` (shift every later event by Δ). Δ comes from composition pause distributions fitted per boundary type (within-word / between-word / sentence; ex-Gaussian, as in writing research) on **KLiCKe** (~5,000 essay writers, open). Place pauses with key identity at training time only (after Space or punctuation). The model never sees keys. Anchor stays plain, positive gets augmented, so the loss teaches invariance to cognition while Aalto's 168k people keep the identity signal. Variant M2b: place pauses by LM surprisal of the next word, as a mechanism study. | Task | 1–2 training runs |
| M3 | **Session-compensating back-end (PLDA / WCCN)** | Borrowed from speaker recognition, which solved session/channel variability at the embedding level: train a PLDA (or WCCN + LDA) back-end on frozen embeddings of a *small multi-session* set (KeyRecs, Buffalo dev people). It learns the within-person session covariance and scores without retraining the encoder. Almost absent from keystroke work, where cosine or Euclidean distance is the norm. | Session | Hours, CPU |
| M4 | **Calibrated likelihood ratios, one score for three decisions** | Calibrate scores (PLDA gives log-LR, or logistic calibration on cosine). One-time auth: threshold the LR. Continuous auth: a **sequential probability ratio test** that accumulates log-LR per 50-key window (this formalises your k-pooling lever and gives ANGA/ANIA). Forensics: report the LR itself with Cllr and Tippett plots, a first for keystrokes. | All three uses | Days, no GPU |

Baselines to beat:
- your v1 / C1 zero-shot;
- k-pooling and s-norm (already measured);
- a TypeNet rebuild;
- **Type2Branch from its released code (verify).** This is where the key-code-vs-timing-only claim under task shift gets tested.

## 4. Data

| Set | Role | Shift it measures | Access |
|---|---|---|---|
| Aalto (yours) | Training (transcription, 168k) | — | Have |
| **KLiCKe** | Pause statistics for CPA; within-essay composition identification (one session, long) | Task (composition only) | Open, github.com/terryyutian/KLiCKe-Corpus. **Check it has key-up times.** |
| **Buffalo** | Main test: 148 people × 3 sessions 28 days apart × transcription + free answers; 73 people swap keyboards | Session × task × keyboard, full factorial | Email the authors |
| **SBU (Stony Brook)** | ~196 people **compose, then retype the same text**: the cleanest task contrast, content held constant | Task, content controlled | Request from the authors (verify) |
| **BB-MAS** | Fixed + free text, same people, desktop (and phone, out of scope) | Task, one session | Open, IEEE DataPort |
| KeyRecs (yours) | Already used to choose fixes, so a dev set now, not a test set | Session (minutes) | Have |

Final test protocol: Buffalo people split in two, dev half for M3/M4 fitting, test half untouched until the end. SBU and BB-MAS are zero-shot tests, never used for tuning.

## 5. The paper

**Working title:** *Transcribe Once, Recognise Anywhere? Task, Session and Keyboard Shift in Open-Set Keystroke
Identification, and How to Close Them.*

**Research questions**
1. How much does a transcription-trained deep embedding lose on composed text, after another session and on another keyboard? Which shift dominates? (Buffalo factorial, SBU content-controlled.)
2. Is the task gap carried by cognitive pauses? (M1/M2 vs a control that adds the same amount of random timing noise; per-feature ablations: HL-only vs within-word vs boundary intervals.)
3. Do key-code models (Type2Branch) degrade more under task shift than timing-only ones?
4. Can a back-end trained on ~100 multi-session people remove the session step? (M3.)
5. Do calibrated LRs support all three uses: one-time EER, continuous ANGA/ANIA, and forensic Cllr / DIR@FPIR? (M4.)

**Contributions (in order of strength)**
1. The first large-scale, factorial measurement of task × session × keyboard shift for deep keystroke embeddings. This alone is publishable.
2. Cognitive-pause augmentation: a physically motivated augmentation trained at Aalto scale.
3. A speaker-recognition-style PLDA back-end with LR calibration, plus the first keystroke Cllr and SPRT continuous-auth results.
4. Supporting results: open-set identification at 10k-person scale with a "no match" threshold; your recipe ablations (mining, readout, BatchNorm) as a compact table; the official KVC-onGoing score if Phase 2 is done.

**Metrics:**
- Identification: CMC / Rank-N, DIR@FPIR (open set);
- Verification: global-threshold EER and FNMR@1% FMR (KVC-style);
- Continuous auth: ANGA / ANIA;
- Forensics: Cllr and Tippett plots.
- Paired bootstrap CIs over people, as in `comparison/paired.py`.
- Seed spread for every trained model (at least 2 seeds).

**Venue:**
- Arxiv preprint first.
- Conference: **IJCB** or **BIOSIG** (Darmstadt, Fraunhofer IGD co-runs KVC; friendly to single authors and students); IWBF as a fallback.
- Journal extension: IEEE T-BIOM.
- Ethics paragraph: GDPR Art. 9, the de-anonymisation risk, and the "attribution layer on lawfully collected data" framing from FUTURE_PROSPECTS §6.

## 6. Order of work

1. **Data requests now** (they take weeks): email Buffalo and SBU; download BB-MAS and KLiCKe and check that they carry press *and* release times.
2. **Zero-shot measurement (RQ1):** loaders through `windows_from_keystrokes` only, then v1 / C1 / TypeNet on BB-MAS and KLiCKe (within-essay). Buffalo and SBU when access arrives. Decision point: is the task gap > seed noise (~0.2 EER)? If not, the paper pivots to session (M3/M4) as the main method and reports "task is not the problem" as finding 1.
3. **M1 cap (inference only)**: an hour, and it tells you whether pauses carry the gap before any training.
4. **M3 / M4 on KeyRecs (dev), then Buffalo dev half.** CPU only, so they can run while the GPU trains.
5. **M2 CPA training run(s)**: new checkpoint in `runs/`, never `model/encoder_hard.pt`. One GPU job at a time.
6. **Type2Branch baseline** under the same protocols.
7. **Final test:** Buffalo test half, SBU, BB-MAS, scored once.
8. **Write.** Optionally submit the best model to KVC-onGoing in parallel for an official same-session number.

Fits after `best-model-plan.md` levers 1–3 (done). It replaces lever 4 "bigger architecture" as the research focus (the field's architecture race is what Type2Branch/EERLoss already win) and turns lever 6/7 into the paper.

## 7. Risks

- **Data access** (Buffalo, SBU) is the critical path. Fallback: BB-MAS + KLiCKe + KeyRecs give task and session separately, not crossed.
- **KLiCKe's keystroke format** (hidden characters, possibly no key-up times) could limit CPA placement. Fallback: place pauses after inter-key gaps that look like word boundaries (the long-IL heuristic) or use Aalto's own pause distribution.
- **The task gap may be small** (Killourhy 2012 replicated). That's still a result, see §2.
- **Rule tension:** M2b uses text content at *training* time only, to place pauses. The model input stays timing-only. A content-leak test must confirm it: the embedding must not predict the sentence or word identity better than the baseline.

## Sources

- KVC-onGoing: https://arxiv.org/abs/2412.20530 · Type2Branch: https://arxiv.org/abs/2405.01088 · EERLoss:
  https://tldr.takara.ai/p/2606.24586 · TypeFormer: https://arxiv.org/abs/2212.13075 · TempCharBERT:
  https://arxiv.org/pdf/2411.07224
- Template ageing, 8 weeks (Parkinson et al.): https://arxiv.org/pdf/2609.29851 · Cross-device transfer (BB-MAS):
  https://arxiv.org/pdf/2608.16334 · Federated cross-device: https://link.springer.com/article/10.1007/s00779-024-01832-6
  · Buffalo adaptive techniques: https://cse.buffalo.edu/tech-reports/2016-06.pdf
- Free vs transcribed (Killourhy & Maxion 2012): https://www.cs.cmu.edu/~keystroke/laser-2012 · Nonoriginal text
  (cognition-aware): https://www.researchgate.net/publication/390519828_Using_Keystroke_Dynamics_to_Detect_Nonoriginal_Text
  · Pinyin transcription vs composition: https://educationaldatamining.org/edm2026/proceedings/2026.EDM.industry-papers.358/index.html
  · Cognitive signatures (Condrey): https://arxiv.org/abs/2603.00177 · Timing forgery: https://arxiv.org/abs/2601.17280
- Free-text transformers: https://link.springer.com/article/10.1007/s00500-025-10524-z · Clarkson II TKCA:
  https://cybersecurity.springeropen.com/articles/10.1186/s42400-021-00075-9 · Survey (ACM CSUR 2025):
  https://arxiv.org/pdf/2303.04605 · Continuous auth metrics: https://arxiv.org/pdf/2006.09337
- Datasets: KLiCKe https://github.com/terryyutian/KLiCKe-Corpus · BB-MAS https://arxiv.org/pdf/1912.02736 · SBU
  https://www3.cs.stonybrook.edu/~rbanerjee/project-pages/keystrokes/keystrokes.html · Buffalo
  https://cse.buffalo.edu/tech-reports/2016-07.pdf
- Forensics: LR / Cllr for authorship https://arxiv.org/pdf/2607.09501 · keystroke forensic study (9 people)
  https://www.researchgate.net/publication/386743855 · de-anonymisation https://whonix.org/wiki/Keystroke_Deanonymization
- Industry: TypingDNA https://findbiometrics.com/topics/typingdna-verify · BehavioSec
  https://blogs.cisco.com/innovation/behaviosec-fires-up-next-gen-cybersecurity-solutions-with-behavioral-biometrics
- Writing-process pauses: https://research.birmingham.ac.uk/en/publications/pauses-in-written-composition-on-the-importance-of-where-writers-
