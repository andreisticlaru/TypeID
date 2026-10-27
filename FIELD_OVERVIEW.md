# Keystroke biometrics: the field at a glance

*Compiled 2026-10-27. Market figures are analyst estimates, so treat them as order-of-magnitude.*

## TL;DR

- **Research:** a small, active niche inside biometrics, with a few hundred papers a year. A handful of groups lead it, most prominently BiDA Lab (Madrid).
- **Industry:** keystrokes are one signal inside **behavioral biometrics**, a market of about **$3B (2025) growing 20–27% a year**. The money is in bank fraud detection and continuous verification.
- **TypeID's niche:** open-set 1:N *identification*. This is mostly research and forensics. Industry and benchmarks focus on 1:1 *verification*.

## Research timeline

| Year | Milestone |
|---|---|
| 1890s–WWII | Telegraph operators recognized each other by sending rhythm (their "fist") |
| 1977–80 | First formal studies (Forsen; Gaines et al., RAND) |
| 1997–2000 | Monrose & Rubin: statistical keystroke authentication |
| 2009 | Killourhy & Maxion (CMU): 51-subject fixed-password benchmark |
| 2018 | **Aalto dataset** (Dhakal et al., CHI): about 168k participants and 136M keystrokes |
| 2021–22 | **TypeNet** (Acien, Morales, Fierrez et al.): LSTM plus triplet loss, galleries up to about 100k |
| 2023 | **KVC** (Keystroke Verification Challenge): Aalto benchmark with a fairness metric (SIR) |
| 2024 | **KVC-onGoing**: 185k+ subjects, free text, desktop and mobile. **TypeFormer**: transformer, 3.25% EER on mobile |
| 2023+ | **BehaveFormer**: keystrokes plus phone motion sensors (IMU) |

**Open problems:** accuracy drops as the gallery grows, typing drifts over time, free text is harder than fixed text, and models transfer poorly across devices. One more caveat: on small datasets, classic ML sometimes beats deep learning. In one SJSU paper, a Random Forest beat a CNN on 148 users.

**Venues:** IJCB, ICB, IEEE T-BIOM, IEEE TIFS.

## Industry

| Company | Focus | Status |
|---|---|---|
| **BioCatch** | Behavioral fraud detection for banks | **Visa is buying it for $2.4B cash** (agreed 2026-08-03, closing by Visa FQ2 2027). Previously Permira at $1.3B (2024). About $160M annual recurring revenue (mid-2025); 350+ banks in 21 countries |
| **TypingDNA** | Typing-only authentication API | $7M Series A (2020, Gradient Ventures); no newer news found |
| **BehavioSec** | Continuous authentication | Acquired by LexisNexis Risk Solutions (2022) |
| **NuData** | Behavioral risk scoring | Acquired by Mastercard (2017) |
| Others | Callsign, Plurilock, ThreatMark, Zighra | Banking, enterprise, continuous authentication |

**Market estimates (2025):** TBRC $2.9B, Mordor $2.72B (→ $11.4B by 2031), Market Research Future $18.6B (outlier, broader scope).

**Trend:** multi-signal fraud platforms (keystrokes plus mouse, touch, sensors and navigation) attract the money. Typing-only products don't.

## Where TypeID fits

- **Identification vs. verification:** almost all recent work and products are verification (EER). Large-scale open-set 1:N identification has much less recent coverage, which leaves a gap TypeID can point at.
- **Privacy and regulation:** typing rhythm can de-anonymize people across sites. GDPR treats identifying biometrics as a special category, and Illinois's BIPA covers them too. That is one reason vendors frame their products as verification.
- **Transferable skills (for a CV):** metric learning, subject-disjoint evaluation, open-set thresholding, CMC/EER, and the fraud and identity-security domain.

## To follow up

- Recent (2025–26) papers: the searches came up thin. Check arXiv for "KVC" or "Type2Branch", and the Google Scholar "cited by" list under TypeNet.

## Sources

- Visa / BioCatch: [FinTech Futures](https://www.fintechfutures.com/m-a/visa-acquires-biocatch), [Nasdaq](https://nasdaq.com/articles/visa-acquire-biocatch-strengthen-ai-powered-fraud-protection), [ID Tech Wire](https://idtechwire.com/visa-agrees-to-acquire-biocatch-for-2-4-billion/), [Sacra](https://sacra.com/c/biocatch)
- TypingDNA: [Romania Insider](https://www.romania-insider.com/business/romanian-typingdna-financing-google-venture-fund)
- Market: [TBRC](https://cn.gii.tw/report/tbrc1970308-behavioral-biometrics-global-market-report.html), [Mordor](https://www.giiresearch.com/report/moi2123103-behavioral-biometrics-market-share-analysis.html), [MRFR](https://www.marketresearchfuture.com/reports/behavioral-biometric-market-24076)
- Papers: [KVC](https://arxiv.org/abs/2311.06000), [KVC-onGoing](https://arxiv.org/pdf/2412.20530), [TypeFormer](https://arxiv.org/abs/2212.13075), [BehaveFormer](https://arxiv.org/pdf/2307.11000), [Stamp et al. free-text](https://arxiv.org/abs/2107.07409), [Keystroke identification (RF vs CNN)](https://astro.paperswithcode.com/paper/keystroke-dynamics-for-user-identification)
