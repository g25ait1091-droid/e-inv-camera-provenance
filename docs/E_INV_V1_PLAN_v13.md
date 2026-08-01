# E-INV → Paper v1: PLAN v11 (final experimental plan)
**2026-07-31.** Supersedes v1–v10. Numbers in `E_INV_RESULTS_v2.md`; citation checks in
`E_INV_REFCHECK.md`.

> **Tracks A, B and C are complete. No further experiment is required to submit.**
> What remains is one statistical reconciliation, reference verification, and writing.

---

## 1. Position

**Not a bare null.** A stage-localised negative result with positive channel controls, a
simultaneous upper bound, two independent replications, and — arriving last — a **validated
additive decomposition** that accounts for the device-level offsets, the generic correlations and
the wrong-direction cross-model results. A separate shifted-template artifact is experimentally
localised but **its underlying mechanism remains unresolved**.

| stage | object measured | conclusion |
|---|---|---|
| **Encoder** | device-specific identity | **survives** — η = 0.3661, post-reconstruction same-model AUC 0.9814 |
| **Objective** | generic fixed-pattern energy with PRNU-like spectrum | **responds** from ≈3× nominal — **not device-specific, not alignment-specific** |
| **Generation** | device-specific identity | **bounded** — λ_U = 0.32%, τ_U = 0.91% |
| **Detector** | validity of the measurement | **calibrated through the full generation-and-quantisation pipeline**; quantisation costs 1.4% |
| **Replication** | — | second system (FLUX.1-dev), five same-model devices, **and a second device representation (low/mid DCT band)** |
| **Mechanism** | — | **validated** additive decomposition — held-out cross-seed R²_cv = 0.870, exact permutation p = 0.0167; leakage lives only in the interaction, which is null (+1.265e-05, CI includes 0) |

**The three stages do not measure the same object, and the paper must say so.** Encoder and
generation concern device-specific identity; the objective concerns pattern energy. Stating it
openly is what stops a reviewer arguing the localisation switches quantities mid-argument — and the
honest version is the more interesting result: *the signal is physically present, generic pattern
energy does move the optimiser, and device identity is nonetheless neither selected nor emitted.*

### Central claim
> **PRNU-derived** device-specific identity survives latent encoding, retaining 36.6% of the same-model device
> contrast. The adaptation objective responds only non-specifically to the associated fixed-pattern
> energy — blind to spatial alignment, indifferent between the true fingerprint and a
> spectrally-matched Gaussian field. After adaptation and text-to-image generation, device-specific
> identity remains undetectable within a simultaneous adapter-clustered upper bound of 0.32% of the
> real-image contrast, 0.91% of the signal the encoder transmits. The null replicates on a second
> generative system and across five same-model camera bodies, and an additive decomposition shows
> that the large apparent per-device correlations are fingerprint main effects, not device transfer.

---

## 2. Contributions

1. **Stage localisation** — encoder passes, objective responds non-specifically, generation bounded.
2. **A real inferential bound** — simultaneous one-sided upper confidence limits, adapter-clustered,
   device-specific denominator, conservative limits, and an honest ceiling (the encoder's own 36.6%).
   Not a power calculation.
3. **A detector calibrated through the full pipeline** — dither-reconstructed pre-quantisation
   float, inject, clip, 8-bit store; direct detection at nominal α = 0.02; quantisation costs 1.4%.
4. **Null-calibration methodology** — empirical null from cross-correlation surfaces, the
   μ-decomposition, measured 1.50× dependence inflation over the analytic 1/√(mG).
5. **A validated additive decomposition of the nuisance structure** — μ + a_x + b_y + i_xy predicts
   held-out device contrasts (cross-seed R²_cv = 0.870; exact permutation p = 0.0167 over 120
   assignments; off-diagonal R² = 0.775), shows the pre-registered contrast was already isolating
   the interaction, and unifies the wrong-direction cross-model null, the generic offset, the
   per-device offsets and the A>B asymmetry. **It does not cover the shifted-template artifact**,
   which is localised but mechanistically unexplained.
6. **Two independent device representations tested, both null** — PRNU (η = 0.366, λ_U 0.32%) and a
   low/mid-band DCT signature (η = 0.554, λ_U 9.7%). The band **trades absolute strength for
   robustness**: 17× less device signal, but a larger fraction survives the encoder. The
   *Beyond PRNU* objection is retired by measurement rather than conceded, and the decomposition is
   confirmed out of sample on a representation it was never fitted to (additive term 9.7× the
   interaction in low/mid, ~zero in PRNU; symmetric part null in both).
7. **Shifted-template controls are not self-calibrating** — a rolled fingerprint acquired a
   systematic correlation advantage in arms that never received it. Transferable caution for
   watermark- and coating-verification work.
8. *(supporting)* **A five-autoencoder retention screen consistent with the predicted role of
   latent bottleneck density** — the three 4-channel autoencoders cluster at low retention while
   the two 16-channel ones retain substantially more. **Not a causal architecture claim**: SDXL and
   PixArt-Σ share a VAE and are not independent observations, the two 16-channel systems differ
   substantially from each other (0.366 vs 0.554), and n = 5.

---

## 3. Remaining work

### 3.1 Blocking
| # | item | cost |
|---|---|---|
| **B0** | **Cross-validate the additive decomposition** — **DONE**: R²_cv 0.870, exact permutation p = 0.0167. Report cross-validated figures, not the in-sample 0.959 | done |
| **B1** | **Reconcile the three routes to "fraction of natural"** (RESULTS §5.2) — they disagree by ~1.6×. Resolution wording is already drafted: λ_U is *the* fraction-of-natural statement; α_equiv is a nominal injection coefficient and is never labelled "% of natural" | analysis |
| **B2** | **Reference verification** — 4 of ~23 done, **2 problems already found**. Tier 1: SIREN (the sweep cites only the S&P program page), DiffusionShield, Beyond PRNU, three patents | ~1 day |

### 3.2 Cheap and worth doing
- Bootstrap the own/off interaction t over adapters (its residuals share constrained row and column
  sums, so the plain t is approximate), or state the approximation.
- Re-run `E_INV_CONSOLIDATE.ipynb` once more; it recomputes every number from the per-row CSVs and
  self-audits against the cached JSONs.

### 3.3 Optional generality — none blocking
| | what | cost | note |
|---|---|---|---|
| E-PROMPT | diverse prompt bank | ~2 h | generation + measurement only; keep the uniform caption as primary |
| E2 | dataset size N ∈ {10, 25, 100, 200} | ~4 h | needs **constant-compute and constant-exposure** arms; N and exposures-per-image confound at fixed steps |
| E3 | random crop | ~3 h | measurement switches to translation-tolerant PCE_max. A **scope limit**, not an a fortiori case |

---

## 4. Drafting guidance

### 4.1 Structure the results around the mechanism, not around chronology
The artifact findings currently read as a list of separate oddities. **They are one phenomenon with
one model.** Present §7 of RESULTS *before* the individual artifacts, then show each falling out of
it. That is substantially more convincing than four independent curiosities and it converts the
study's messiest thread into its cleanest.

### 4.2 Related work — five subsections
2.1 Physical camera and PRNU fingerprints (Lukáš et al. 2006; Chen et al. 2008; Dresden;
non-unique artifacts; Beyond PRNU) — *end: these attribute captured images to cameras; none examine
propagation through generative adaptation.*
2.2 Fingerprints of generative models (Marra et al.; Yu et al. ICCV'19; Latent Fingerprints;
ManiFPT; Whodunit) — *end: their attribution target is the generator, not the physical device in
its fine-tuning images.*
2.3 **Proactive marking of generative training data** — the load-bearing comparison (Yu et al.
ICCV'21; DiffusionShield; ProMark; SIREN; CoprGuard) — *end: these deliberately design or optimise
a signal to survive training; we study an unmodified physical trace that exists before any ML
pipeline.*
2.4 Memorisation and data attribution (Carlini et al.; Evaluating Data Attribution; dataset
inference; FineXtract).
2.5 Position — **state explicitly that this is not another active watermark.** It evaluates whether
natural physical provenance already forms an unintended attribution channel.

### 4.3 Claims not to make
| claim | why |
|---|---|
| fingerprints in training images can transfer to outputs | **not novel** — Yu et al. ICCV'21 |
| active watermarks establish training-data attribution | **not novel** — ProMark, SIREN, CoprGuard, DiffusionShield |
| no fixed-coordinate pattern can transfer | contradicted by the active-watermark literature |
| physical camera identity disappears | too broad — say **PRNU-derived** |
| the signal entered the weights | not established; the objective responded |
| first to study camera fingerprints and autoencoders | prior work exists — limit to the exact generative VAE |
| universal firstness for the shifted-template finding | say "we demonstrate a failure mode" |

### 4.4 Figures
| # | content |
|---|---|
| F1 | every adapter's θ, both arms, with the simultaneous upper limit drawn across |
| F2 | stage localisation on a log axis: R_real → R_VAE → bound |
| F3 | detector calibration — real images and generated images on shared axes, bound drawn on |
| F4 | the paired design working: generic offset versus the contrast that survives it |
| **F5** | **the additive decomposition — predicted vs observed θ_d, R² = 0.959** *(new, and the most important)* |

---

## 5. Venue

**arXiv immediately** once written — free, establishes priority, not patent-gated for a measurement
paper. Then **TIFS**, with **IH&MMSec 2027** as the named fallback.

| venue | status |
|---|---|
| **TIFS** | primary. With two systems, five devices, six seeds per arm, a calibrated detector and the mechanism, this is a substantial paper. Lead the cover letter with distinctness from the two in-flight TIFS submissions. Realistic acceptance **35–45%**; plan for revise-and-resubmit |
| IH&MMSec 2027 | named fallback, ~Jan–Feb deadline |
| IEEE Access | viable with scope explicit; declined in favour of TIFS — generalist reviewers apply "what did you find?", and it would be a third Access paper adjacent to two TIFS submissions |
| WIFS 2026 | **closed 15 July 2026** |
| SPL / ICASSP | unsuitable — the methods, null calibration and controls do not compress into four or five technical pages. *(An OJSP-at-ICASSP route allowing 8+1 pages is worth verifying; ICASSP 2027 deadline 16 Sep 2026)* |

### Working titles
- *Where Physical Camera Provenance Disappears in Diffusion Personalisation: Encoder Survival,
  Objective Sensitivity, and Generation-Side Bounds*
- *From Sensor to Latent to Generator: Localising Device-Fingerprint Loss in Text-to-Image
  Personalisation*

---

## 6. Limitations to state, not hide
- **One dataset**, 2010-era CCD. **No a fortiori extension** to modern computational-photography
  devices, whose pipelines may carry *different* persistent signatures rather than weaker PRNU.
- **Fixed-coordinate protocol** until E3.
- **Matched-noise investigation used a small finite set of Gaussian fields.**
- **Adapter-conditional inference**: the clustered bound generalises over seeds (n = 6, two
  devices) and separately over devices (n = 5, one camera model) — two different claims.
- The objective-stage measurement concerns **pattern energy, not device identity**.

---

## 7. Method notes worth carrying into any successor study
1. **Include never-injected arms in every control.** Decoys establish rank *within* an arm; only
   arms that never received a field can separate field-specific from alignment-specific.
2. **Cross the design.** One adapter per device confounds device with seed; 5 × 2 made ICC = 0.943
   with mixed signs visible, which two devices never could.
3. **Fit main effects before interpreting a contrast.** Fingerprint main effects were 2.58× larger
   than arm effects and explained R² = 0.959 of the apparent device signal.
4. **A non-significant test plus an MDE is not a bound.** Use upper confidence limits.
5. **Calibrate the detector on the actual output medium**, including quantisation — and check that
   the calibration is not a no-op.
6. **Watch the estimator against the estimand**: nominal injection amplitude is not natural
   amplitude; here a nominal 1× delivered 0.39×.
