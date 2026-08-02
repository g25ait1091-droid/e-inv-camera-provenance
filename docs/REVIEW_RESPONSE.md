# Response to the independent audit

Every numbered finding below was actioned. Where a finding was accepted only in part, the
reason is given.

## Blocking statistical and technical

| # | Finding | Action |
|---|---|---|
| 1 | Sign-flip test cannot reject at $\alpha=0.01$ with $k=6$ ($p_{\min}=1/64=0.0156$) | **Accepted in full.** Section III now discloses the defect explicitly, keeps the registered test as a diagnostic, refuses to relax $\alpha$ post hoc, and names the upper confidence limit as the primary inference. Also added to Limitations |
| 2 | Inference hierarchy unclear; image-level and adapter-level mixed | **Accepted.** Section V-C separates them and labels the per-seed values as conditional image-level tests that do not generalize over training runs |
| 3 | "Localized downstream of where the signal reaches the optimizer" overclaims | **Accepted.** Discussion now rules out only the two intuitive explanations and states plainly that the point of loss is *not* identified — it could be latent conditioning, parameter updates, spatial averaging, the decoder, or this detector |
| 4 | "The objective cannot distinguish alignment" is stronger than the measurement | **Accepted.** Every instance rewritten as "the measured scalar loss response does not distinguish", with an explicit note that gradients, layerwise updates and representations were not measured |
| 5 | $\eta$ measures an autoencoder **round trip**, not the encoder | **Accepted.** Terminology changed throughout, including the section heading. Added that $\tau$ is a ratio of image-domain contrasts, not a latent information measure |
| 6 | Calibration uses a *modelled* pre-quantization float | **Accepted.** Retitled, the uniform residual is labelled an assumption, and the concrete route to replacing it (capture decoder floats under the recorded seeds) is stated. Added to Limitations |
| 7 | "Direct detection" uses absolute sign, not the increment over $\alpha=0$ | **Accepted.** Section V-E now separates the two: absolute contrast excludes zero at $\alpha=0.02$; the incremental reading is the fitted slope, increment $2.83\times10^{-4}$, slope lower-99\% $1.3613\times10^{-2}$ |
| 8 | Figure 2 plots $U=8.88\times10^{-5}$ but annotates 0.32\%/0.91\% | **Accepted — this was a real inconsistency.** The annotation now reads 0.680\%, which is what the plotted analytic limit gives, and the caption explains why the BCa headline has no single absolute value to plot |
| 9 | BCa simultaneity not demonstrated | **Accepted, downgraded rather than asserted.** The analytic limits are simultaneous by construction; the BCa is described as a bootstrap interval on the same max-arm statistic with no additional familywise claim |
| 10 | "Cancels identically to first order" too strong | **Accepted.** Replaced with partial cancellation expected, extent unquantified, with the three reasons the regimes differ. Added to Limitations |
| 11 | Text claims adapters trained "from 1 to 24"; only 1–12 have results | **Accepted — this was factually wrong.** Corrected: adapters at 1, 1.6, 3, 6, 12; injection-integrity on real images extended to 18 and 24 with no adapters trained there |
| 12 | "A bound on any injection-based verification scheme" | **Accepted.** Scoped to this estimator pair, injection rule and detector, with four named alternatives that could exceed it |

## Uncertainty and method detail

| # | Finding | Action |
|---|---|---|
| 13 | No interval for $\eta$ or AUC | **Accepted, honestly.** The per-arm spread is reported and the absence of an interval is stated outright: the archived summaries keep arm-level rather than per-image paired values, so an image-level bootstrap cannot be reconstructed. Listed in Limitations rather than papered over |
| 14 | ICC form and interval unspecified | **Accepted.** Now ICC(1,1), one-way random effects, single measurement, from between/within mean squares, cited, used descriptively with no interval at $n=5$ |
| 15 | 5/5 seed agreement is dependent | **Accepted.** Explicitly refuses to convert it to a $p$-value and names the shared factors |
| 16 | Tail loss underspecified | **Accepted.** Defined as the mean of the final 100 optimizer steps of a 2000-step run, single shared-noise sweep |
| 17 | Similarity audit metric unnamed | **Accepted.** Cosine similarity of self-supervised embeddings over all cross-split pairs, aggregated by maximum |
| 18 | "Maximum-sensitivity configuration" too strong | **Accepted.** Now "sensitivity-favouring, spatially registered", with the reasons it is not a proven global maximum |
| 19 | Copy audit needs a calibrated positive control | **Accepted.** Now "no evidence of memorization under the ensemble's thresholds", with detector power explicitly not calibrated |

## Low/mid representation

| # | Finding | Action |
|---|---|---|
| 20 | Not an implementation of *Beyond PRNU* or the dark-image DCT work | **Accepted — an important correction.** Now stated as a hand-crafted band signature *motivated by* their shared finding, with both methods' actual approaches described and the claim narrowed to one representation |
| 21 | "8\% of variance is device signal" needs qualification | **Accepted.** Framed as split-half reliability under classical assumptions, with the Spearman–Brown-type extrapolation and its effective split size given |
| 22 | "Inversely related" from two representations | **Accepted.** Reduced to "the two orderings differ"; no general relationship claimed |
| 23 | 9.72\% uses a different construction, unmarked in Figure 6 | **Accepted.** Marked in the text, the figure caption, and Table IV with a dagger footnote |

## Novelty, references, compliance

| Finding | Action |
|---|---|
| Abstract's "existing work establishes only…" too absolute | Rewritten as "the closest signal-transfer studies rely on deliberately designed or optimized marks" |
| Title broader than the estimand | Retitled: *PRNU-Derived Device Provenance Through Diffusion Personalization* |
| Seven contributions, several supporting | Condensed to four |
| 12 incorrect or incomplete bibliography entries | All corrected from the supplied verified author lists. **13 `REF-VERIFY` markers → 0** |
| Missing method references | Added Mihçak (denoiser), Hu (LoRA), Esser (SD3), SD3.5 and FLUX.1-dev model cards, Efron (BCa), Koo & Li (ICC), Ruiz (DreamBooth), FT-Shield — all cited in text |
| Table IV unfinished row | Removed |
| AI-use disclosure missing | **Restored** in the Acknowledgment, naming the system and the affected sections |
| Abstract at the 250-word limit | Reduced to 233 |

## One finding not adopted as stated

**Reference [21] pagination.** The audit gives 7192–7203 for *Evaluating Data Attribution*. Crossref
returns 7158–7169 for DOI 10.1109/ICCV51070.2023.00661. Both are correct — this is the
CVF-versus-Xplore split that also affects Yu et al. 2021. The bibliography uses **IEEE Xplore
pagination with DOIs throughout** for internal consistency, and the entry now carries a parenthetical
noting the CVF alternative.

## Remaining before submission
1. One `TODO-AUTHOR`: archival DOI for the repository (cut a release, connect Zenodo).
2. Supplementary material: per-adapter tables, exact test code, bootstrap pseudocode, control
   matrices, detector thresholds, five-autoencoder screen detail, ICC computation, calibration
   paired differences, split manifests.
3. Optional but valuable: capture decoder float tensors under the recorded seeds and repeat the
   calibration, replacing the uniform-dither assumption with a measurement.

---

# Second audit — response

The follow-up audit raised ten items. All ten actioned.

| # | Finding | Action |
|---|---|---|
| 1 | Section II still said the paper "quantifies where the channel closes" while the Discussion denies localization | Rewritten to "evaluates whether provenance remains recoverable at successive observable endpoints and bounds what reaches generated output"; §VI-A renamed to *What the Stage-Wise Endpoints Establish* |
| 2 | Three stale objective-stage overclaims outside Results | All three rewritten: §III-C now says the measured scalar tail-loss response does not distinguish; §VI-B no longer calls it "the mechanism"; §VI-C no longer claims the response could not support alignment-dependent recovery, and states that gradients and parameter updates were not measured |
| 3 | "Complete generation and eight-bit storage pipeline" survived in the Introduction and Results | One description everywhere: injection into generated images under a uniform model of pre-quantization fractional values, then clipping, quantization, storage, reload and measurement |
| 4 | The universal 4× claim returned in §VI-B | Removed; scoped to the tested configuration with alternatives named |
| **5** | **Contradiction over whether per-image VAE records exist** | **The auditor was right and the manuscript was wrong.** `s1b_vae_roundtrip.csv` holds 40 per-image paired values per device at both stages. Recomputing from it reproduces the archived summaries exactly ($R_{\mathrm{real}}=3.567034\times10^{-2}$, $R_{\mathrm{VAE}}=1.305928\times10^{-2}$). A paired image-level bootstrap ($10^4$ draws) now gives **$\eta = 0.3661$, 95 % CI $[0.3437, 0.3870]$**, $R_{\mathrm{real}} \in [3.262, 3.873]\times10^{-2}$, $R_{\mathrm{VAE}} \in [1.137, 1.480]\times10^{-2}$, and **post-reconstruction AUC $0.9814$, 95 % CI $[0.9647, 0.9931]$**. The false "cannot be reconstructed" sentence is deleted, the Limitations entry is reduced to the ICC alone, and §VIII now says "most reported quantities" and names the two classes that cannot be recomputed |
| **6** | **Registered adapter-level $p$-values claimed but never shown** | **Computed exactly** by enumerating all $2^6=64$ sign assignments: $p_A = 19/64 = 0.297$, $p_B = 12/64 = 0.188$. Both reported, with the note that neither could fall below the $0.0156$ floor |
| 7 | AI disclosure named only one system | Now names Anthropic Claude and OpenAI ChatGPT, with the actual roles. The premature "all citations independently verified" claim is replaced by an accurate statement that metadata was checked against publisher records and remaining items are flagged in the source |
| 8 | Submission placeholders | The invented `ACCESS.2025.DOI` is reverted to the template's own `ACCESS.2017.DOI` placeholder. The `VOLUME 4, 2016` footer is hardcoded in `ieeeaccess.cls` and replaced by IEEE at production — not an author-editable field, and left alone deliberately |
| 9 | Reference [23] carried two page ranges | Parenthetical removed; IEEE Xplore pagination retained, consistent with the cited DOI and with every other CVF entry |
| 10 | FT-Shield metadata thin; model cards lacked access dates | Author list completed and flagged for a final check against the ACM record; access dates added to the SD3.5 and FLUX.1-dev model cards |

## Figure captions
Figure 1 now says autoencoder round-trip. Figure 2 carries the $\eta$ confidence interval.
Figure 4 distinguishes the absolute contrast excluding zero at $\alpha=0.02$ from the incremental
sensitivity established by the fitted paired slope. Figure 6 gives the low/mid point a **third
marker class** (square, teal) so the symmetric-interaction construction is visually distinct from
the seed- and device-generalized max-arm bounds.

## State
17 pages, zero LaTeX errors, zero undefined references, abstract 233 words.
One `TODO-AUTHOR` (archival DOI) and one `REF-VERIFY` (FT-Shield author list) remain.

---

# Third audit — bibliography and figure 2

## Reference checker warnings adjudicated

The auditor's finding that most "year mismatch" warnings are **false positives** is correct: the
automated checker was comparing preprint, technical-report and publisher-online dates against final
proceedings dates. Left unchanged, with the reason:

| Ref | Warning | Verdict |
|---|---|---|
| [10] Yu et al., GAN fingerprints | 2018 vs 2019 | arXiv preprint vs ICCV proceedings. **2019 and pp. 7555–7565 retained**; the IEEE pagination matches the cited DOI (CVF uses 7556–7566) |
| [25] DreamBooth | 2022 vs 2023 | preprint vs CVPR proceedings. **2023 retained** |
| [30] Efron, BCa | 1984 vs 1987 | Stanford technical report vs the JASA article actually cited. **1987 retained** |
| [12] Noise-based imprint | author order | Manuscript order was already correct; the checker reordered the display |

## Corrections applied

| Ref | Change |
|---|---|
| [16] DiffusionShield | Added **Dec. 2024** so the Dec-2024 issue / Jan-2025 online date no longer reads as a discrepancy |
| [17] FT-Shield | Full author list (Cui, Ren, Lin, Xu, He, Xing, Lyu, Fan, Liu, Tang), **pp. 76–88**, Dec. 2024. The last `REF-VERIFY` marker is gone |
| [23] Data attribution | Alternate-pagination parenthetical removed; IEEE pagination retained, consistent with the cited DOI |
| [27] Esser et al., SD3 | Added **pp. 12606–12633** and the full venue name |
| [28] SD 3.5 Medium | Formatted as a Hugging Face model card with an access date |
| [29] FLUX.1-dev | Now points at the **exact model-card file** rather than the repository root, with an access date |

With no `REF-VERIFY` markers left, the AI-use disclosure's statement that bibliographic metadata was
verified against publisher records is now accurate and the hedge has been removed.

## Figure 2 — arrows touching the labels

Reported by the author and confirmed by extending the collision detector to **sample arrow paths**,
not only text and legend boxes. The previous version had two annotation arrows crossing three bar
value labels: the $\eta$ leader passed over the $3.57\times10^{-2}$ and $1.31\times10^{-2}$ labels,
and the vertical $\tau$ leader ran straight through $8.88\times10^{-5}$.

Both arrows are removed. The $\eta$ relationship is now shown by a **bracket** spanning the first two
bar tops with the label above it, and the round-trip ratio sits directly above the third bar without
a leader. Figure 4's callout leader was also straightened and shortened.

`overlapcheck.py` now samples every arrow path and exempts only a label's own leader, which
necessarily originates inside its box. All seven figures report **OK**.

## State
17 pages, zero LaTeX errors, zero undefined references, abstract 233 words.
**`REF-VERIFY` 0. One `TODO-AUTHOR` remains: the archival DOI.**

---

# Full fine-tuning result (F4)

**The null holds when every transformer parameter is free to move.**

| | $\theta$ | sd | $U$ | $p$ |
|---|---|---|---|---|
| arm A | $-1.134\times10^{-4}$ | $6.963\times10^{-5}$ | $+2.856\times10^{-4}$ | 1 |
| arm B | $+6.829\times10^{-5}$ | $1.571\times10^{-4}$ | $+9.687\times10^{-4}$ | 0.251 |
| **symmetric** | $-2.255\times10^{-5}$ | $5.042\times10^{-5}$ | $+2.663\times10^{-4}$ | $t=-0.77$ |

$\lambda_U = 0.747\,\%$ at $n=3$.

## Why this matters more than another replication

It removes the sharpest remaining objection: that the bound reflects LoRA's rank-16 attention
subspace rather than the pipeline. **At matched $n=3$ the three adaptation settings agree to within
11 %** — LoRA on SD 3.5 $0.699\,\%$, FLUX.1-dev $0.777\,\%$, full fine-tuning $0.747\,\%$.

## Two findings inside the result

**Full fine-tuning produces a far larger additive nuisance term than LoRA.** The arms are again
near equal and opposite; the additive fingerprint main effect is $4.0\times$ the interaction, against
roughly $0.04\times$ in the LoRA arms. Freeing every parameter amplifies the nuisance structure the
paper is built to cancel.

**This is the third case where the construction decides the answer.** A max-arm bound here gives
$2.72\,\%$ — a factor $3.6$ looser and dominated by the additive term. The symmetric statistic gives
$0.747\,\%$. The same thing happened in the low/mid band ($20.56\,\%$ versus $9.72\,\%$).

## Interaction estimate, all four measurements

| measurement | interaction |
|---|---|
| D200 LoRA, $n=6$ | $+1.494\times10^{-5}$ |
| Kodak, 5 devices | $+1.265\times10^{-5}$ |
| Low/mid band, $n=6$ | $-1.036\times10^{-5}$ |
| Full fine-tuning, $n=3$ | $-2.255\times10^{-5}$ |

Four independent measurements across two adaptation methods, two generative systems, seven physical
devices and two device representations, spanning $-2.3$ to $+1.5\times10^{-5}$, none significant.

## Comparison fairness, verified before the result was read
Relative parameter drift across the six runs has a coefficient of variation of 2 %, and generated
images differ from base-model generations at matched seeds by 50–63 gray levels against 31–59 for
LoRA. Full fine-tuning adapted at least as strongly, so a null cannot be attributed to
under-training.

## Outstanding
`TODO-AUTHOR` in Section V-G: run the copy-detection audit on these generations. 2.24 billion
parameters fine-tuned on 50 images is the most memorization-prone configuration in the study.

## State
18 pages, zero LaTeX errors, zero undefined references, abstract 242 words, `REF-VERIFY` 0,
two `TODO-AUTHOR` (the copy audit and the archival DOI).
