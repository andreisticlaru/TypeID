# Future Prospects — Free-Text Composition & Related Open Questions

Notes from a design discussion on extending TypeID beyond random-prompt
transcription toward genuine free-composition forensic realism (see
CLAUDE.md's "Forensic-realism caveat" for the framing this builds on).

## 1. Can identification run on freely composed text while enrollment stays transcription-based?

**Yes, with no pipeline changes required.** The feature extractor
(`features/extract.py`) only ever consumes raw keydown/keyup events — it
never looks at *what* was typed, by design (no key-identity features).
So the identify screen could show an open prompt ("write 2-3 sentences
about your day") instead of a fixed sentence, and the same
`extract_features()` → `embed()` → cosine-similarity flow would run
unchanged. The only real work is a frontend UI change plus enforcing a
minimum-keystroke floor against free-form input instead of a known
prompt length.

## 2. The transcription → composition domain gap: is it settled?

**No — the literature is split, and this is a live research area, not
a solved problem.**

- **Older, statistical-feature era (2012):** found transcription vs.
  free-text timing differences were small (~2-3ms) and didn't
  meaningfully hurt authentication accuracy. Conclusion at the time:
  transcription is a fine, easier-to-collect proxy — don't bother with
  free text.
- **Newer, deep-learning era:** treats this as a real, nuanced effect.
  Work distinguishing "Cognition-Aware" vs. "cross-cognition"
  evaluation finds that *different feature types transfer differently*
  — temporal features hold up when train/test cognitive load matches,
  rhythmic features generalize better across mismatched cognitive load
  (transcription = low load; composition = high load, due to word
  retrieval/planning pauses). Current work (2024-2025) even treats
  transcription, bona-fide composition, and LLM-paraphrased text as
  three distinct scenarios worth separating.
- **Takeaway:** nobody has published a clean answer for *modern
  triplet-loss sequence-encoder embeddings* specifically (the 2012
  result used shallow statistical features on smaller data). Measuring
  this gap directly for this project's architecture would be a
  legitimate, citable contribution — not a rediscovery of known fact.
  CLAUDE.md's cautious default (assume a gap exists until measured) is
  the right prior going in.

## 3. Candidate datasets for free-text composition

| Dataset | Fit | Notes |
|---|---|---|
| **Buffalo Keystroke Dataset** (Sun, Ceker & Upadhyaya, 2016) | **Best fit** | 148 subjects, 3 sessions over ~4 months, mixes transcription (Steve Jobs speech) *and* genuine free-text (answering questions) from the *same* people — enables direct within-subject transcription-vs-composition comparison. Has same-keyboard and cross-keyboard sections, also covering the cross-device gap CLAUDE.md flags. Access typically requires a request to UB CSE, not instant public download. |
| **KeyRecs** | Not usable | Despite the name, it's transcription-exercise data — same category as Aalto, not free composition. |
| **Mendeley "Human-written and Synthesized Free-Text"** | Narrow fit | Real free-text sentences, but built for liveness/bot-detection, likely too small a subject pool for gallery-search training. |
| **Romanian free-text dataset** (80 users) | Not usable as-is | Free text, but single session per user — no cross-session anchor/positive pairs, which triplet training requires. |

**Recommendation:** Buffalo, either as a fine-tune/eval layer on top of
an Aalto-pretrained model, or as a held-out-subject eval set to
directly quantify the transcription→composition gap for this project's
specific model.

## 4. Can LLMs help close the gap?

**Plausible mechanism, not an established solution — a real research
opportunity.** Composition-specific pauses are largely driven by lexical
retrieval difficulty (surprisal) — a well-studied psycholinguistic
effect: people pause longer before words that are less predictable
given prior context. Transcription doesn't have this, since the person
is copying, not generating.

Proposed approach: run an LLM over the (partial) text being produced,
extract per-token surprisal/perplexity, and feed it as an auxiliary
covariate alongside the HL/IL/PL/RL timing sequence. The network could
then learn to normalize out content-driven pause variance, isolating a
purer person-specific rhythm signal that (hypothesically) transfers
better across transcription and composition.

**Tension to manage:** this cuts against CLAUDE.md's explicit
content-blind design principle (used to prevent the network from
shortcutting to memorized sentence/content patterns instead of learning
rhythm). Using LLM-derived content features reopens that exact risk
unless surprisal is used strictly as a normalization term and validated
on subjects unseen during training to confirm it still generalizes
rather than becoming a content-based shortcut.

## 5. Does the framework generalize to unseen sentences at identify time (within transcription mode)?

**Yes — by design, this is the default case, not an open question.**

- Features are timing-only, with no key identity, specifically so nothing
  depends on which sentence was typed.
- Training already enforces this: anchor/positive triplet pairs are
  deliberately drawn from *different* sessions/sentences of the same
  subject, forcing the network to learn subject-specific rhythm rather
  than sentence-specific timing (CLAUDE.md's triplet construction spec).
- So identify-time prompts differing from enrollment-time prompts is
  the base case the architecture was built for — matching prompts would
  be the unusual, easier case, not the norm.
- What this *does* depend on: enough sentence-pair diversity during
  Aalto training for the network to have actually learned
  content-invariant rhythm rather than overfitting to a narrow sentence
  style. CLAUDE.md's own suggested check — evaluating on prompts never
  seen in training/enrollment — is exactly how to catch it if this
  breaks down in practice.

## 6. Keylogging: should the system capture WHAT was typed, or stay timing-only?

**Recommendation: stay timing-only (as currently designed) — capturing
content turns this into a legally and architecturally different tool.**

This question splits into two separate real-world capabilities that
forensic/law-enforcement use conflates at a glance but treats very
differently in practice:

- **Content capture (a keylogger)** — records *what* was typed. Legally,
  this is an interception of electronic communications. In the US, covert
  deployment by law enforcement requires a full **Title III wiretap order**
  (18 U.S.C. §§2510-2522) — probable cause, judicial authorization,
  time-limited, senior DOJ sign-off for federal cases. This is the same
  heavy legal bar as a phone wiretap, not a lightweight tool.
- **Timing-only capture (behavioral attribution, this project's current
  design)** — records *how* something was typed, never *what*. It doesn't
  intercept communications content at all, so it sits in a fundamentally
  lighter-weight legal category — closer to a biometric identification
  technique (fingerprint, face match) than to a wiretap.

**How this maps to a realistic investigative workflow:** in practice, a
keylogger or already-seized device would be the tool that lawfully
captures the actual evidentiary content (the ransom note, the threatening
email) under whatever legal authority applies. This project's
keystroke-biometric layer would run *on top of* that already-lawfully-
obtained keystroke stream to answer a different question: **"of the
people with access to this device, which specific person was behind the
keyboard for this session?"** — useful when multiple people share a
device or account and content alone doesn't establish who typed it. That
is a genuinely useful, narrower, and legally cleaner role than trying to
build an all-in-one covert content-logging tool.

**Separately — admissibility is still an open bar.** Any novel forensic
identification technique (this one included) would need to survive a
**Daubert** reliability challenge (FRE 702: tested methodology, known
error rate, peer review, general acceptance) before a court would weigh
it. No case law specifically addressing keystroke-dynamics-as-identification
turned up in this research — meaning the technique is largely untested in
US courts, not an established one. Worth stating explicitly in any
write-up rather than implying courtroom-readiness.

**Practical takeaway for this project:** keep the architecture
content-blind (per CLAUDE.md's existing no-key-identity design) and frame
it explicitly as an *attribution layer for already-lawfully-collected
data*, not a standalone surveillance tool. This is both the better ML
design (forces rhythm-learning over content-memorization) and the more
honest forensic scope.

## 7. Data protection: is typing-rhythm data personal data, and could client-side processing avoid sending it off-device?

**Yes to both questions — worth stating precisely, since overclaiming here
would be inaccurate.**

- GDPR's personal-data definition (Art. 4(1)) is broad: anything relating
  to an identifiable person. Timing data clears that bar trivially, since
  discriminating identity is the whole point of the system.
- It likely goes further, into **special-category biometric data** (Art. 9):
  Art. 4(14) defines biometric data as data resulting from technical
  processing of physical, physiological, **or behavioural** characteristics
  used for unique identification. Typing rhythm is explicitly behavioural,
  and unique identification is this system's explicit purpose. Special-category
  data needs explicit consent (or another Art. 9 exemption) and typically a
  DPIA before real deployment.
- Asymmetry worth noting against section 6 above: content capture
  (keylogging) is regulated as communications interception (wiretap law);
  timing-only capture is regulated as biometric data. Different legal
  regime, not a lighter one — dropping key identity doesn't dodge privacy
  law, it just changes which law applies.

**Could running feature extraction and embedding entirely client-side avoid
sending biometric data off-device?**

Partially, not fully:

- Feature extraction and the frozen embedding model could both run in the
  browser — Pyodide (real CPython in WASM) to reuse the *exact* Python
  extraction code without a JS reimplementation, and ONNX/TF.js for the
  frozen LSTM — so raw keydown/keyup timestamps would never leave the
  device, only the resulting 128-dim embedding would be sent to the
  backend for gallery storage/comparison.
- This meaningfully shrinks exposure: raw timing is a richer,
  more-reconstructable signal than a compressed embedding.
- It does not exit the regulatory category, though. The embedding is still
  "data resulting from specific technical processing" of a behavioural
  characteristic — the same reasoning that makes a face-recognition
  embedding or fingerprint minutiae template count as biometric data, not
  just the raw scan. Consent, HTTPS in transit, and a deletion policy would
  still be needed for the embedding.
- A fully "nothing biometric ever leaves the client" version would require
  local-only enrollment/identification against a gallery stored in the
  browser — workable only for a single shared device, and it abandons the
  growing multi-person gallery premise the project is built around.

**Decision for now:** keep the current architecture (raw events -> backend
-> canonical Python extractor, per CLAUDE.md's single-source-of-truth
principle). Client-side extraction/embedding via Pyodide/ONNX is a valid
future direction if the project needs a stronger privacy story, but isn't
required at portfolio-demo scope, which isn't processing real users' data
under a live consent regime.

## Sources

- [Observations on Typing from 136 Million Keystrokes (Dhakal et al., CHI 2018)](https://dl.acm.org/doi/10.1145/3173574.3174220) — the Aalto dataset this project trains on. [Free PDF](https://acris.aalto.fi/ws/portalfiles/portal/21495207/ELEC_Dhakal_et_al_Observations_CHI2018.pdf), [dataset download](https://userinterfaces.aalto.fi/136Mkeystrokes/).
- [Free vs. transcribed text for keystroke-dynamics evaluations (LASER 2012)](https://dl.acm.org/doi/10.1145/2379616.2379617)
- [TypeNet: Deep Learning Keystroke Biometrics (arXiv)](https://arxiv.org/pdf/2101.05570)
- [LLM-Assisted Cheating Detection in Korean Language via Keystrokes (arXiv)](https://arxiv.org/html/2507.22956v1)
- [Representation Learning and Pattern Recognition in Cognitive Biometrics: A Survey (PMC)](https://pmc.ncbi.nlm.nih.gov/articles/PMC9320620/)
- [Shared keystroke dataset for continuous authentication (Buffalo, ResearchGate)](https://www.researchgate.net/publication/312568278_Shared_keystroke_dataset_for_continuous_authentication)
- [User Authentication with Keystroke Dynamics in Long-Text Data (Buffalo tech report, PDF)](https://cse.buffalo.edu/tech-reports/2016-07.pdf)
- [Dataset of Human-written and Synthesized Samples of Free-Text Keystroke Dynamics (Mendeley Data)](https://data.mendeley.com/datasets/mzm86rcxxd/2)
- [KeyRecs: A keystroke dynamics and typing pattern recognition dataset (PMC)](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10474054/)
- [Shared Data Set for Free-Text Keystroke Dynamics Authentication Algorithms (Romanian dataset, ResearchGate)](https://www.researchgate.net/publication/351519662_Shared_Data_Set_for_Free-Text_Keystroke_Dynamics_Authentication_Algorithms)

**Forensics / law enforcement:**
- [Keystroke Dynamics Features in Forensic Identification: theoretical and experimental approaches (ResearchGate)](https://www.researchgate.net/publication/386743855_KEYSTROKE_DYNAMICS_FEATURES_IN_FORENSIC_IDENTIFICATION_theoretical_and_experimental_approaches)
- [User attribution based on keystroke dynamics in digital forensic readiness process (IEEE)](https://ieeexplore.ieee.org/document/8270436/)
- [Biometric keystroke attribution (US Patent, USPTO)](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11436310)
- [Insider Threat Detection Based on Stress Recognition Using Keystroke Dynamics (arXiv)](https://arxiv.org/pdf/2005.02862)
- [Title III of the Omnibus Crime Control and Safe Streets Act of 1968 / Wiretap Act (Bureau of Justice Assistance)](https://bja.ojp.gov/program/it/privacy-civil-liberties/authorities/statutes/1284)
- [18 U.S. Code § 2516 — Authorization for interception of wire, oral, or electronic communications (Cornell LII)](https://www.law.cornell.edu/uscode/text/18/2516)
- [DOJ Electronic Surveillance Manual (justice.gov, PDF)](https://www.justice.gov/sites/default/files/criminal/legacy/2014/10/29/elec-sur-manual.pdf)
- [Daubert Standard (Wikipedia)](https://en.wikipedia.org/wiki/Daubert_standard)
- [Forensic Evidence Admissibility & Expert Witnesses: Daubert Standard (Forensic Science Simplified)](https://www.forensicsciencesimplified.org/legal/daubert.html)

**Data protection:**
- [GDPR Article 4 — Definitions, incl. "personal data" and "biometric data" (gdpr-info.eu)](https://gdpr-info.eu/art-4-gdpr/)
- [GDPR Article 9 — Processing of special categories of personal data (gdpr-info.eu)](https://gdpr-info.eu/art-9-gdpr/)
