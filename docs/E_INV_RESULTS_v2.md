# E-INV — RESULTS (v2, consolidated)

> **SUPERSEDED IN PART.** This document records results as of 1 August 2026, when the
> primary analysis used six adapters per arm. The design was subsequently extended to twelve
> (see `config/results_corrections.json`), and the headline limits are now
> λ_U = 0.1507 % and τ_U = 0.4117 %, not the 0.32 % / 0.91 % below. Everything else here
> stands. `analysis/FINAL_LEDGER.json` is authoritative.


**2026-07-31.** Every measured number, with the wording each requires.
Supersedes all earlier revisions. Companion: `E_INV_V1_PLAN_v11.md`.
Status: **Tracks A, B and C complete.** Remaining work is drafting and reference verification.

---

## 0. The claim, and the wording rules

> **Device-specific camera identity survives the latent autoencoder** of a modern text-to-image
> model: reconstructed images still separate two bodies of the same camera model at AUC 0.981,
> retaining **36.6%** of the device-specific contrast. **PRNU-scale fixed-pattern perturbations
> measurably affect the personalisation objective** from ≈3× nominal amplitude — but that response
> is **non-specific**: blind to spatial alignment, and indifferent between the true fingerprint and
> a spectrally-matched Gaussian field. **After adaptation and text-to-image generation,
> device-specific transfer remains below λ_U = 0.32%** of the real-image device contrast,
> equivalently **τ_U = 0.91%** of the signal the encoder transmits. The complete
> generation-and-quantisation measurement pipeline **directly detects an injected template at a
> nominal injection amplitude of 0.02**. The null **replicates on a second generative system**
> (FLUX.1-dev, whose autoencoder transmits 1.5× more device signal) and **at device level across
> five same-model bodies**.

**This is not a bare null**: a stage-localised negative result with positive channel controls, a
simultaneous upper bound, two independent replications, and a mechanistic account of every apparent
positive signal.

### Binding wording rules
| write | never write |
|---|---|
| "device identity in the two tested frequency representations" (§6.3) | "camera identity" unrestricted — learned ISP signatures, lens effects and dark-current FPN/DSNU remain untested |
| "no detectable transfer for the tested signals and protocol" | "fixed patterns cannot transfer" — the active-watermark literature contradicts it |
| "does not survive under this protocol" | "does not occur" |
| "the objective responds to fixed-pattern energy" | "the objective registers device identity" |
| "the objective response is measurable" | "the signal entered the weights" — loss sensitivity is not persistent parameter encoding |
| "the principal detector-sensitivity objection is substantially addressed" | "validity is closed" |
| "the tested spectrally matched Gaussian control" | "matched noise" in general — few realisations were tested |
| "≈3.5× effective amplitude (nominal 12×)" | "12× natural amplitude" |
| the *nominal injection coefficient* equivalent to the bound | α_equiv as "% of natural" |

λ̂ always means the **paired-contrast device-specific** value, never a generic offset.
**No a fortiori reasoning** — caught three times (random crop, modern cameras, shared-model κ).

---

## 1. Apparatus

| Quantity | Value |
|---|---|
| Devices | A = Nikon_D200_1 (380 imgs), B = Nikon_D200_0 (372) — same model; nulls C = Agfa_DC-733s_0, D = Nikon_D70_1, D2 = Agfa_DC-830i_0 |
| Splits | E1 = 80, E2 = 140 (A,B); E2 = 60 (aux); T = 50, H = 40, guards 10 |
| Worst cross-split scene similarity | 0.789 (B, E1–E2); halt threshold 0.95 |
| **κ_model** (NCC K̂_A, K̂_B post-clean) | **0.0073** (was 0.431 with the inverted filter — §9) |
| K̂ variance split | device 0.355 / estimation noise 0.638 |
| Real-image AUC, pooled / same-model A vs B | 0.9983 / **1.0000** |
| ρ̂_real (naive) | A 0.03806, B 0.03241 — **not** the denominator |
| **R_real** (device-specific paired contrast) | **3.56703e-02** — *this* is the denominator |
| R_natural(B) | 3.14047e-02 |
| Suppression / injection gain (S1c) | 7.0× / 201×, α = 1.6, PSNR 62.0 / 59.8 dB |

**ρ̂_real reconciliation (closed):** 0.03806 is the **A-only** median; 0.03616 is the **A+B pooled**
median (B alone 0.03241, so the pooled value sits between the arms). Both correct, different
quantities. Neither is the denominator.

---

## 2. Stage 1 — the encoder passes

SD-3.5-medium VAE round trip on the H split: same-model AUC **1.000 → 0.9814**; scalar retention
0.358; **device-specific retention η = 0.3661**; R_VAE post = **1.30593e-02**.

**Retention tracks latent channel count across five autoencoders** (device-specific paired
contrast, H split, pre-registered screen):

| model | latent ch | post-VAE AUC | **η** |
|---|---|---|---|
| SD-1.5 | 4 | 0.8130 | 0.0749 |
| SDXL | 4 | 0.8155 | 0.0929 |
| PixArt-Σ (SDXL VAE) | 4 | 0.8159 | 0.0957 |
| SD-3.5-medium | 16 | 0.9816 | 0.3661 |
| **FLUX.1-dev** | **16** | **0.9875** | **0.5540** |

Three 4-channel autoencoders cluster at η ≈ 0.075–0.096; two 16-channel at 0.366 and 0.554. **A
standalone finding and a five-autoencoder empirical confirmation of the density law.**
**The pre-registered 0.75 AUC gate excluded nobody** (worst: SD-1.5 at 0.813) — report that; it
shows the rule was not reverse-engineered from the outcome.

---

## 3. Stage 2 — the objective responds, non-specifically

All amplitude arms share seed 0, hence batch order and noise stream: the comparison is paired, and
the ~0.013 seed-to-seed spread is removed by construction.

| arm | α nominal | tail loss | Δ vs α = 1 | ‖lora_B‖ |
|---|---|---|---|---|
| amp_1p0 | 1.0 | 0.17085 | — | 72.08 |
| amp_1p6 | 1.6 | 0.17086 | +1.72e-05 (noise) | 71.56 |
| amp_3p0 | 3.0 | 0.17071 | −1.31e-04 | 71.40 |
| amp_6p0 | 6.0 | 0.16953 | −1.31e-03 | 72.38 |
| amp_12p0 | 12.0 | 0.16618 | −4.66e-03 | 73.13 |
| **ctrl_Q** | 12.0 | 0.16494 | **−5.91e-03** | 72.80 |
| **ctrl_shift** | 12.0 | 0.16614 | **−4.70e-03** | 73.02 |

Not a clean power law — α^3.3 from 3 to 6, α^1.8 from 6 to 12. **Do not fit an exponent.** State:
undetectable at ≤1.6×, onset ≈3× **in one paired sweep**, saturating by 12×.

**Two controls make the stage non-specific:**
- **ctrl_shift ≈ amp_12p0** (0.9% of the effect). A circular shift preserves spectrum and
  amplitude, so the objective cannot distinguish alignment. **The loss curve carries no information
  about whether spatial alignment survives.**
- **ctrl_Q > amp_12p0 by 27%.** The objective has *no preference* for the true PRNU over a
  spectrally-matched Gaussian field — it prefers the smoother one.

**Therefore the three stages do not measure the same object, and the paper must say so.** Encoder
and generation concern *device-specific identity*; the objective concerns *generic fixed-pattern
energy with PRNU-like spectral content*. Conflating them would let a reviewer argue the
localisation switches quantities mid-argument.

**Also not established, and not to be implied:** that the pattern is persistently encoded in the
adapter weights. Loss sensitivity at a training step is not durable parameter encoding.

---

## 4. Stage 3 — generation is bounded

### 4.1 Hypothesis tests
Primary intersection–union test fails at every seed and pooled.

| | pooled p | per-seed p |
|---|---|---|
| S_A (A-adapters, K̂_A − K̂_B) | 0.494 (d = 1.60e-06) | 0.454 / 0.525 / 0.486 |
| S_B (symmetric replication) | 0.403 (d = 9.56e-06) | 0.689 / 0.435 / 0.250 |

No secondary survives Holm at α = 0.05 over nine tests (requires p ≤ 0.0056) even ungated; smallest
observed 0.0244.

### 4.2 The bound — three estimators, report the largest

| quantity | plug-in | conservative limit | **BCa bootstrap** | **REPORT** |
|---|---|---|---|---|
| λ_U | 0.2490% | 0.2777% | **0.3236%** | **0.32%** |
| τ_U | 0.6800% | 0.8093% | **0.9077%** | **0.91%** |

Hierarchical bootstrap, 2000 iterations, resampling adapters → generations → held-out reals, BCa
acceleration from a jackknife over adapters. **The raw 99th percentile (0.8474% / 2.2911%) is a
six-cluster coarse-tail artefact and is not reportable.** BCa landing near the analytic
conservative value is the reassurance it is behaving.

λ̂ = **+0.0419%**, bootstrap 95% CI **[−0.0850%, +0.1705%]**. U_device = **8.8802e-05**.

Per arm at n = 6 adapters: A θ̂ +1.549e-05 (sd 4.453e-05, U 8.880e-05); B θ̂ +1.438e-05
(sd 3.332e-05, U 6.924e-05). t₀.₉₉₅,₅ = 4.03. The seed extension took λ_U from 0.6993% at n = 3 to
0.2490% — a 2.8× tightening, better than the 2.46× the t-ratio alone predicts.

**Bootstrap limitation, stated:** a fully nested resample would re-estimate K̂ from a resampled E
split and re-measure all 37,500 correlations per iteration — not tractable. Fingerprint uncertainty
is handled analytically (K̂ enters numerator and denominator identically and cancels to first order
in the ratio).

**Sensitivity:** arm A's bound is carried by one adapter. Leave-one-out dropping A_raw_s5 takes
sd 4.453e-05 → 1.296e-05 and U 8.880e-05 → 2.462e-05. Arm B is stable. **Report the full-n bound as
primary and this as disclosed sensitivity; do not drop s5.**

### 4.3 Deliberate injection does not transfer either
| arm | α nominal | ρ(→K_B) | contrast vs K_A | p |
|---|---|---|---|---|
| amp_1p0 | 1.0 | 8.063e-05 | −1.520e-05 | 0.577 |
| amp_1p6 | 1.6 | 7.069e-05 | −2.862e-05 | 0.656 |
| amp_3p0 | 3.0 | 5.643e-05 | −6.868e-05 | 0.837 |
| amp_6p0 | 6.0 | 1.231e-04 | −3.233e-05 | 0.674 |
| amp_12p0 | 12.0 | 1.619e-04 | −2.101e-05 | 0.618 |

The doubling of ρ(→K_B) is a generic residual-statistics shift affecting **both** fingerprints; the
paired design removes it. All five contrasts are negative, but the arms share seed 0 and source
images — five correlated readings, not five draws. Descriptive only, no sign test.

**Neither did the tested spectrally-matched Gaussian control**: ctrl_Q vs K_Q −2.559e-04
(comparator-affected — §7.1).

---

## 5. Detector calibration *through the full pipeline*

Injecting K̂_B (E1) into **existing generated images** and measuring with K̂_B (E2). No training,
no generation.

**The quantisation path had to be got right.** A first attempt injected into the loaded 8-bit PNG
and re-rounded — **a no-op**: an 8-bit PNG is exactly integer-valued, so a sub-0.5-LSB perturbation
survives rounding unchanged (0.000% of pixels change up to α = 0.5; all seven amplitudes returned
bit-identical means). The correct construction models the decoder's **pre-quantisation float** as
`stored_integer + U(−0.5, 0.5)`, injects, then rounds **once**, with dither seeded per image so it
is identical across α.

| α | n | mean | SE | t | CI excl. 0 |
|---|---|---|---|---|---|
| 0 | 2500 | −6.6188e-05 | 3.190e-05 | −2.07 | — |
| 0.01 | 2500 | +7.5227e-05 | 3.196e-05 | 2.35 | — |
| **0.02** | 2500 | **+2.1720e-04** | 3.211e-05 | **6.76** | **yes** |
| 0.05 | 2500 | +6.4109e-04 | 3.321e-05 | 19.30 | yes |
| 0.10 | 2500 | +1.3484e-03 | 3.684e-05 | 36.60 | yes |
| 0.25 | 2500 | +3.4691e-03 | 5.609e-05 | 61.85 | yes |

Weighted fit **with intercept**: C = −6.598e-05 + **1.4142e-02**·α; lower-99% slope 1.3613e-02.
The negative intercept is the generic offset of these generations, which is why the intercept must
be fitted.

**Quantisation costs 1.4% of sensitivity** (float slope 1.4345e-02 → quantised 1.4142e-02).
Sub-LSB signals survive rounding because image content acts as dither.

### 5.1 Effective versus nominal amplitude — a claim that changed
α_effective = [ρ(→K_B at α) − ρ(→K_B at 0)] / R_natural(B):

| α nominal | 1 | 1.6 | 3 | 6 | 12 |
|---|---|---|---|---|---|
| **α effective** | **0.390** | 0.626 | 1.154 | 2.174 | **3.483** |

**A nominal 1× injection delivers only 0.39 of natural**, because K̂_B^E1 is an estimate —
NCC(K̂_B^E1, K̂_B^E2) = 0.246 — so most injected energy lands off the true fingerprint direction.
**"No transfer at up to 12× natural" must read "up to ≈3.5× effective (nominal 12×)."**

Clipping is **not** the cause of the saturation (0.0225% of pixels at nominal 12×, PSNR 45.6 dB);
the perturbation dilutes every correlation — ρ→K_A itself drops **21%** (0.038063 → 0.030228). A
high-amplitude injection measurably degrades the host device's own fingerprint.

### 5.2 OPEN — reconcile before drafting
| route | value | implied margin |
|---|---|---|
| U / R_real | 0.28% | 357× |
| U / R_natural(B) | 0.28% | 353× |
| E-POST α_equiv in effective units (1.137% × 0.39) | 0.44% | ~225× |

The ~1.6 gap is not noise. Candidates: the intercept subtraction; R_real pooling both arms versus
R_natural(B) alone; E-POST calibrating on E-AMP generations while U comes from base-study
generations. **Resolution wording:** λ_U is *the* fraction-of-natural statement; report α_equiv as
the **nominal injection coefficient** whose measured effect equals the bound, never as "% of
natural." Direct detection then sits at nominal 0.02 = **0.78% effective**, and the bound at
0.28–0.32% is a factor ≈2.8 below it.

---

## 6. Replication

### 6.1 Cross-system — FLUX.1-dev
Selected by the pre-registered screen (§2). Trained at 1024 with base-study fingerprints,
extractor, seed bank and statistic unchanged; only the generative system differs.

| arm | θ per seed | θ̂ | sd | p_adapter |
|---|---|---|---|---|
| A | −7.949e-06 / +6.368e-05 / −1.548e-05 | +1.342e-05 | 4.369e-05 | 0.498 |
| B | −6.402e-05 / +9.713e-06 / +3.070e-05 | −7.867e-06 | 4.974e-05 | 0.632 |

**Primary IUT fails: the null replicates.** U_device 2.7717e-04 → **λ_U = 0.777% at n = 3.** The
comparable primary-study figure is **0.699%, also at n = 3** — agreement within 11%. **Never
compare 0.777% against the 0.32% headline**: that contrasts t₀.₉₉₅,₂ = 9.92 with t₀.₉₉₅,₅ = 4.03
and reads a sample-size artefact as a system difference.

**Call this a cross-system robustness test, not an architecture ablation** — VAE, backbone, latent
dimensionality, text encoder and scale all co-vary.

### 6.2 Device-level — Kodak_M1063, crossed 5 devices × 2 seeds
| device | θ (s0, s1) | mean | real contrast |
|---|---|---|---|
| D0 | +1.822e-04, +2.385e-04 | **+2.104e-04** | 0.04586 |
| D1 | −5.363e-07, −1.877e-05 | −9.655e-06 | 0.01784 |
| D2 | −1.729e-04, −1.072e-04 | **−1.400e-04** | 0.03522 |
| D3 | −8.476e-05, −4.956e-05 | −6.716e-05 | 0.01713 |
| D4 | +5.025e-05, +8.919e-05 | +6.972e-05 | 0.03065 |

**ICC = 0.943** (between-device variance 1.760e-08 vs within-device 1.057e-09, a 17× ratio), and
**5 of 5 devices have both seeds on the same side** (chance ≈ 2.5). Device identity dominates θ
variation and is reproducible across independent training runs.

**And it is not leakage: the signs are mixed** — two positive, three negative, grand mean
+1.265e-05, **sign-flip p = 0.434**. Leakage predicts every device favours its *own* fingerprint,
i.e. all θ > 0. **One adapter per device would have confounded device identity with training seed**
and this would have been invisible.

λ̂ = +0.0431%; **λ_U = 1.74%** at device level (n = 5, t₀.₉₉₅,₄ = 4.60). Looser than the headline
mainly because of the **denominator**: the five bodies span 0.0171–0.0459 in real contrast (2.7×,
against the D200 pair's 1.2×), so SE = 5.43e-03 and the lower-99% limit (1.670e-02) sits **43%
below** the mean. **This bound generalises over DEVICES; 0.32% generalises over SEEDS on two
devices. Different claims — report both, neither substitutes.**

**Leave-one-device-out changes nothing** — spread 3.504e-04 → 3.515e-04 (−0.3%), ICC unchanged at
0.943. The per-device offsets are **not** the shared camera-model component. **Report the refuted
hypothesis; it is informative.**

---

## 6.3 A second device representation — the low/mid frequency band

*Beyond PRNU* reports device identity in low/mid frequency bands more robust than classical PRNU.
Rather than concede that as a limitation, we tested it: an 8×8 block-DCT signature keeping
`1 ≤ i+j ≤ 5` (DC excluded, 20 of 64 coefficients), built from the **same** E1/E2 splits with the
**same** zero-mean and Wiener-in-DFT cleaning, measured on the **same** generations with the
**same** paired statistic and the same six seeds per arm. Only the representation changes.

### It is a real device representation, but a much weaker estimator
| | PRNU | low/mid |
|---|---|---|
| same-model AUC, held-out real images | 1.0000 | **0.8433** (gate was 0.75) |
| NCC(L_A, L_B) — shared camera-model component | 0.0073 | **−0.0018** (cleaner) |
| **split-half reliability** NCC(E1, E2) | 0.362 | **0.0812** |
| R_real | 3.567e-02 | **2.120e-03** (16.8× smaller) |
| R post-VAE | 1.306e-02 | 1.175e-03 |
| **device-specific VAE retention η** | 0.3661 | **0.5544** |

**The band trades absolute strength for robustness**: it carries ~17× less device signal but
retains a *larger fraction* of it through the encoder. To our knowledge this separation — how much
device identity a representation carries versus how well it survives latent compression — has not
been measured before, and the two are anti-correlated here.

**Estimation cost, quantified.** At reliability 0.0812 only ~8% of the variance in L̂ is device
signal. Per image the low/mid signature is ~5× weaker as an estimator than PRNU: matching PRNU's
0.362 reliability would need **~680 images per device**, where Dresden's device A has 380 (best
attainable reliability 0.241). Report this; it is a concrete limit *Beyond PRNU* does not quantify.

### The null holds — and confirms the decomposition out of sample
| arm | θ̂ (n = 6) | sd |
|---|---|---|
| A | **−1.110e-04** | 8.832e-05 |
| B | **+9.032e-05** | 1.327e-04 |

**θ_A and θ_B are nearly equal and opposite** — exactly what the additive decomposition (§7)
predicts when a fingerprint main effect dominates, since the additive part is antisymmetric and
cancels in the mean:

| | interaction (leakage) | additive (b_A − b_B)/2 |
|---|---|---|
| **low/mid** | **−1.036e-05**, t = **−0.27** | −1.007e-04, t = **−4.06** |
| **PRNU** | +1.493e-05 | +5.55e-07 |

**The additive term is 9.7× the interaction in the low/mid band and essentially zero in PRNU.** Two
representations with qualitatively different nuisance structure, the same two devices — and in both
the symmetric part is null. **This is an independent confirmation of the decomposition on a signal
type it was never fitted to**, and stronger than resampling the data it came from.

**λ_U(lowmid) = 9.72%**, from the symmetric statistic (θ_sym = −1.036e-05, U = 1.459e-04, divided
by the lower-99% denominator 1.501e-03). *A per-arm bound would give 20.56% and be wrong* — in this
band the per-arm statistic is dominated by the additive main effect, not by leakage. The looser
bound reflects a 17× weaker signal estimated at 0.081 reliability, **not a weaker null**; say so
explicitly or a reviewer will read it as evidence of more transfer.

### What this buys the paper
> Device identity was tested in **two distinct frequency representations**. Classical PRNU retains
> 36.6% through the encoder and is bounded at 0.32%; a low/mid-band signature retains **more**
> (55.4%) but carries 17× less signal and is bounded at 9.7%. **Neither transfers.**

The *Beyond PRNU* constraint is now retired **by measurement rather than conceded**. The scope
sentence becomes "two independent device-identity representations", not "PRNU-derived only" —
while still excluding learned ISP signatures, lens effects and dark-current FPN/DSNU, which remain
untested.

*Note:* seed **s5** is the outlier in both representations (low/mid symmetric +1.579e-04; it also
carried arm A's spread in the PRNU analysis). Two representations, one adapter — an **adapter**
property, not a representation one. It belongs in the leave-one-out sensitivity discussion.

---

## 7. The mechanism — one model explains every apparent signal

Fit ρ(gens from arm *x* → K_*y*) = μ + a_*x* + b_*y* + i_*xy* over the 10 × 5 Kodak matrix:

- arm effects sd **3.771e-05**; **fingerprint effects sd 9.730e-05** — fingerprints dominate 2.58×
- b = {D0 +1.3568e-04, D1 −2.63e-05, D2 −1.2789e-04, D3 −2.422e-05, D4 +4.272e-05}
- predicted θ_d = b_d − mean_{d′≠d} b_d′ reproduces the observed device means at
  **r = +0.979, R² = 0.959 in-sample** — and, critically, **out of sample** (§7.3)

**Under additivity the arm term cancels algebraically**, so θ_d is fixed by fingerprint properties
alone — accounting for the stable arbitrary-sign offsets, the 0.943 ICC and the inertness of LODO
in one model.

### 7.1 The pre-registered statistic was already the interaction
θ_A = (a_A+b_A+i_AA) − (a_A+b_B+i_AB) = (b_A−b_B) + i_AA − i_AB
θ_B = (a_B+b_B+i_BB) − (a_B+b_A+i_BA) = (b_B−b_A) + i_BB − i_BA
**θ_sym = ½(θ_A+θ_B) = ½(i_AA − i_AB + i_BB − i_BA)** — the additive terms cancel **exactly**.
Likewise Σb_d = 0 by construction, so the Kodak grand mean over devices is pure interaction, which
is why it equals the measured own−off difference to four figures.

**Two independent device sets agree on the interaction — where leakage would live:**

| | interaction | significant |
|---|---|---|
| D200 pair, θ_sym (n = 6 adapters/arm) | +1.494e-05 | no |
| Kodak, own (+1.012e-05) − off (−2.530e-06) | +1.265e-05 | no, t = +1.12 |

Agreement to 15%. **A replication of the estimate, not merely of the verdict.**

### 7.3 Held-out validation — the decomposition is not merely in-sample
The in-sample R² = 0.959 fits and assesses on the same 10 × 5 matrix. Two genuinely held-out tests:

| test | result |
|---|---|
| **Cross-seed** — fit μ + a_x + b_y on one seed's adapters, predict the other seed's device contrasts, both directions | **R²_cv = 0.870**, RMSE 4.43e-05 (10.8% of the observed spread) |
| Same, aggregated to devices (the honest n = 5) | **r = +0.979, p = 0.0036, R² = 0.959** |
| **Exact permutation** over all 120 device assignments | **p = 0.0167** — only 2 of 120 relabellings do as well or better (floor 1/120 = 0.0083) |
| **Off-diagonal** — estimate b_y from off-diagonal cells only, predict the held-out own-device cell | **R² = 0.775** |
| Own-device residual with **all** additive structure removed | **+1.265e-05, t = +0.69**, 99% cluster-bootstrap CI **[−3.23e-05, +5.87e-05]** — includes zero |

That last row is the cleanest leakage estimate in the study, and it matches the Kodak grand mean
and the D200 θ_sym (+1.494e-05). **Three routes, one estimate, all null.**

**Report the cross-validated figures, not the in-sample 0.959 alone.** With five devices the exact
permutation floors at 0.0083, so p = 0.0167 is near the resolution limit — state that.

**Call it a validated additive decomposition, not a mechanistic theory.** And note what it does
*not* cover: the shifted-template artifact (§8.1) is experimentally localised but its mechanism
remains **unexplained**. Do not write "accounts for every apparent positive signal."

**Cautions.** The own/off residuals come from one fit with constrained row and column sums, so that
t is approximate. **The paired contrast stays the pre-registered primary**; promoting the
interaction to headline after seeing the data would undo the study's pre-registration credibility.

### 7.2 What this explains, all at once
| observation | explanation |
|---|---|
| Generic offset two orders larger than the contrast (X2: A-gens K_A 7.472e-05 vs K_B 7.312e-05, difference 1.60e-06) | fingerprint main effect b_y; cancels to ~2% in the paired design |
| Cross-model null runs backwards (`existence_vs_D` p = 0.9864) | D-gens correlate with K_A — a device D never saw — as strongly as with K_D: b_y again, not device leakage |
| Per-device offsets, ICC 0.943, mixed signs | b_d − mean of others, R² = 0.959 |
| LODO inert | offsets are fingerprint main effects, not shared κ |
| A > B adaptation-strength asymmetry (SD-3.5 ≈5%, FLUX ≈3.3%) | property of the training images, persists across architectures |

---

## 8. Controls, artifacts and sanity

### 8.1 Shifted-template controls are not self-calibrating
`ctrl_shift` vs K_Bshift gave +1.273e-04, p = 0.031 — the only positive-pointing contrast in the
study. Resolved as an artifact in four steps:

1. **The comparator is uncalibrated.** Four *independent* spectrally-matched Gaussian fields, never
   injected, measured against ctrl_Q generations with K_A as comparator gave **t from −4.54 to
   +1.56**. Injected K_Q sits at **z = −0.68** in that spread.
2. **Displacement decoys favoured the injected shift.** On the comparator-free symmetric statistic
   ρ(→K_Bshift) − ρ(→K_B): +2.104e-04, t = +3.07 against 30 decoys (mean t −0.96, sd 1.04, max
   +0.44) — largest of all 31, empirical p = 0.032. *This looked like alignment-specific transfer.*
3. **The arms that never saw it rank it first anyway.** K_Bshift ranks **1 of 32 in every E-AMP
   arm** — including `amp_1p0` (unshifted K_B at 1× only) and `ctrl_Q` (Gaussian field only).
   **A field never injected cannot be re-emitted.**
4. **Baselines localise it to adapted generations.** In base-model generations K_Bshift falls to
   rank 7/36 inside the decoy range while unshifted real fingerprints take the top three; on real
   device-A photographs K_A is correctly dominant at 4.00e-02 and K_Bshift is rank 10/36.

**Mechanism unexplained**, and stated as such: the obvious candidate — that every E-AMP arm received
a K_B-spectrum perturbation, including Q — **fails**, because the decoys are rolls of K_B with
identical spectra and stay flat.

> **A circularly shifted fingerprint can acquire a systematic correlation advantage with adapted
> generations that is unrelated to whether it was ever injected. Shifted-template controls are not
> self-calibrating.** Any study using one must include arms that never received the shifted field.

**Second methods point:** per-image t against a mismatched comparator is uncalibrated at this scale
— four never-injected fields produced t from −4.54 to +1.56.

### 8.2 The θ ~ ‖lora_B‖ covariate — exploratory only
| arm | Pearson | Spearman |
|---|---|---|
| A | −0.685 (p = 0.133) | **−0.257 (p = 0.62)** — no relationship; its Pearson is one leverage point |
| B | **−0.892 (p = 0.017)** | **−0.829 (p = 0.042)** — rank-robust |

The sign is **backwards for leakage**. But the stronger argument is cross-system: FLUX arm A norms
span 79.93–80.77 (0.84) versus SD-3.5's 69.49–72.68 (3.19) — 3.8× tighter — yet θ sd is essentially
identical (4.369e-05 vs 4.453e-05). **Tightening adaptation strength did not tighten θ.** Report as
an exploratory diagnostic motivating pre-registered strength controls, not as a contribution.

Suggestive but not significant: θ_d correlates with the device's own real contrast (Pearson +0.56,
Spearman +0.40, n = 5, p = 0.33); D2 breaks it.

### 8.3 Integrity and sanity
| Check | Result |
|---|---|
| Mode collapse | none — zero duplicate pairs at 0.95; max within-tag similarity 0.84–0.92 |
| √G scaling | half/full off-peak SD ratio 1.201 vs 1.414 ⇒ a persistent structured component σ_μ = 5.23e-05 alongside image noise SE(G=500) = 4.67e-05; off-peak SD **overstates** SE by ~1.5× |
| Dependence inflation | measured **1.50×** over analytic 1/√(mG) — report the measured value (a synthetic model predicted 5.8×) |
| PCE argmax mod-8 | uniform 0.042–0.070 across all 15 arm × fingerprint cells **including base-model generations with no adapter** ⇒ generic VAE 8-px decoder grid. One sentence |
| Gen vs real regime | PCE sd 2.66 vs 2.74 — same regime |
| **Copy-flag rate** | **0.0% on all A arms**; 0.2–0.6% on D. No memorisation despite adaptation shifting images by \|Δ\| 31–59/255 |
| Second extractor (E5) | U_device 1.970e-04 (db8/L4) vs 1.918e-04 (sym8/L3) on identical images at n = 250/tag — within 3%. **Robustness pair, not the headline** |
| Generation determinism | sequential-vs-fresh pipeline output bit-identical under matched seeds |
| GPU measurement equivalence | max \|Δρ\| 4.4e-10 vs empirical SE ~6.5e-05; TF32 disabled |

---

## 9. Pre-registration deviations

1. **`wiener_dft` was inverted** — it preserved the strong spectral peaks (JPEG blocking, CFA, ISP
   shading) shared across a camera model and zeroed the noise floor where device PRNU lives. Caught
   by the S1 K-quality gate at κ_model = 0.431; corrected filter gives 0.0073. Fingerprints
   recomputed from scratch. **Report both κ values** — the failure and its detection are evidence
   the gates work.
2. **α = 1.6** for the causal arms, selected to maximise suppression on **held-out real images**,
   never on generated images or any test statistic.
3. **λ_min from an empirical null**, superseding the analytic 1/√(mG); then superseded again by
   upper confidence limits, because a non-significant test plus an MDE is not a bound.
4. **A_raw_s0** trained under an earlier cell revision; not noise-stream-paired with the causal
   arms; causal analysis uses the pooled reference. Its `lora_B_norm` was recovered post hoc
   (71.995) from the safetensors.
5. **Measurement convolutions and FFTs on GPU float32, TF32 disabled**, verified equivalent to the
   CPU path (§8.3).

---

## 10. Open before submission
- **§5.2 three-route reconciliation** — blocking.
- **Reference verification** — 4 of ~23 done, 2 problems found (`E_INV_REFCHECK.md`). Binding.
- Bootstrap the own/off interaction t over adapters, or state it as approximate.
- Optional generality: E-PROMPT, E2 (dataset size, with constant-compute *and* constant-exposure
  arms), E3 (random crop — a **scope limit**, not an a fortiori case).

## 11. Limitations to state, not hide
- **One dataset.** Dresden is 2010-era CCD. **No a fortiori argument** extends this to modern
  computational-photography devices, whose pipelines may carry *different* persistent signatures
  rather than merely weaker PRNU. Daxing is behind Baidu Pan (not scriptable); FODB and VISION are
  directly downloadable but weak on same-model pairs.
- **Fixed-coordinate protocol** until E3.
- **The matched-noise investigation used a small finite set of Gaussian fields** and does not cover
  every stationary or optimised pattern family.
- Inference is **adapter-conditional**: the per-image test is conditional on the adapters trained;
  the clustered bound generalises over seeds (n = 6, two devices) and, separately, over devices
  (n = 5, one camera model).
