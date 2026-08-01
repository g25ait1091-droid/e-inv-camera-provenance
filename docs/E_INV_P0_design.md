# E-INV-P0 — Inverse-Channel Pilot: Device Fingerprint Leakage into Generative Model Weights
**Design document v1.0 — mathematics, adversarial confound registry, pre-registered decision tree.**
Companion notebook: `E_INV_P0_pilot.ipynb` (single file, staged S0–S7, hard gates, resumable).

---

## 0. Objective and claims ladder

**Question.** Does fine-tuning a text-to-image model on images from one physical camera leave a
detectable trace of that camera's *device-level* sensor identity (PRNU / defect map) in images
generated from pure noise?

**Claims ladder** (pre-registered — we commit to reporting whichever rung the data reaches, including the bottom):

| Rung | Claim | Evidence required |
|---|---|---|
| L4 | Distributional device-identity leakage | G1 ∧ G2 ∧ G3 pass |
| L3 | Memorization-mediated device leakage | G1 ∧ G2 pass, G3 fails |
| L2 | Camera-**model**-level leakage only | G1 passes, G2 fails |
| L1 | Bounded null: λ < λ_min at power 0.99 | all gates fail; λ_min printed a priori |

Every rung is publishable. L1 is the honest-null paper ("PRNU-based training-set attribution is
unsupportable at effect sizes ≥ λ_min under a maximal-sensitivity protocol"). L4 is the TIFS paper
and the seed of the certified training-set-attribution detector (→ patent gate, §10).

---

## 1. Formal channel

Sensor model (standard multiplicative PRNU):

    Y = (1 + K_d) ⊙ X + Θ,     K_d ∈ R^m,  E[K_d]=0,  ||K_d|| small

restricted throughout to a **fixed sensor window** Ω (center 1024×1024 native crop, no resampling),
m = |Ω| = 1024² = 1,048,576.

Inverse channel:

    T_A = {Y_i}_{i=1}^{N}  (device A, split T)
      → θ = FineTune(θ_0; Enc_VAE(T_A))        [LoRA, base weights θ_0 frozen]
      → Z_j ~ p_θ(· | c),  j = 1..G            [ancestral sampling from Gaussian noise;
                                                 NO image conditioning — asserted in code]

**Leakage question:** does Law(Z) depend on K_A beyond camera-model statistics? Formally, the
binary hypothesis test between generation laws P_{θ(T_A)} and P_{θ(T_B)} where A,B are distinct
devices of the **same camera model** trained on scene-matched data. Detection-theoretic framing
now; the Fano/MI lower bound I(K_A; Z^G) is the theorem for the full paper, not the pilot.

**Leakage coefficient (headline dimensionless quantity):**

    λ := E[ρ(Z)] / E[ρ(Y^H)]

where ρ(·) is the matched-filter correlation (§2) against K̂_A estimated from split E ⊥ T, and
Y^H are held-out real images of device A (split H). λ = 1 means gens carry the fingerprint as
strongly as real photos; λ = 0 means null.

**Why the channel can carry it at all — two preconditions we control by design:**
1. **VAE encoder pass-through.** Training images enter the UNet/DiT only through the VAE encoder.
   By the density law (your TIFS density paper), PRNU survives the autoencoder iff latent
   bottleneck density r = C/f² is high. SD-1.5 (r=0.0625) attenuates; SD-3.5/FLUX (16-ch, r=0.25)
   pass at AUC 0.88–0.91. **Base model = SD-3.5-medium** — the density law is used as a
   channel-capacity precondition. (Bonus prediction H-DENS: λ(SD3.5) > λ(SD1.5); optional arm.)
2. **Absolute-position representability.** A pixel-locked pattern requires the network to encode
   absolute position. DiT (SD-3.5) has explicit positional embeddings → natural. Conv UNets only
   via padding-induced position cues (Islam et al. 2020) → harder. SD-3.5 is therefore also the
   maximal-sensitivity choice architecturally. (Confound: density and architecture co-vary between
   SD1.5 and SD3.5 — pilot notes it; 2×2 disentangling {UNet,DiT}×{4ch,16ch} via SDXL and
   PixArt-α is stage-2, not pilot.)

**Fixed-crop protocol.** All training images present the *same* sensor window at the *same* image
coordinates. This maximizes memorizability of a pixel-locked pattern. It is the **existence**
design: if leakage does not occur here, it does not occur anywhere. Random-crop (realism arm) is
stage-2. Consequence: if the model memorizes the pattern, it emits it aligned to the image frame →
**zero-lag** statistics are primary; translation-tolerant argmax-PCE is secondary.

---

## 2. Estimators and statistics

**Residual.** W(I) = I − F(I), F = Mihcak-style adaptive wavelet Wiener denoiser (db8, 4 levels,
σ₀ = 2.0 on 0–255 scale, local-variance min over windows {3,5,7,9}), luminance channel, **fp32
end-to-end on the measurement path** (house rule). Each residual standardized to zero mean, unit
variance before use (prevents heavy-tail dominance in pooling).

**Fingerprint estimate (MLE, Chen–Fridrich–Goljan):**

    K̂_d = ( Σ_{i∈E_d} W_i ⊙ Y_i ) / ( Σ_{i∈E_d} Y_i² )

followed by NUA cleaning: zero-mean over rows and columns, then Wiener filtering in the DFT
domain (noise floor from MAD of |F|). This strips CFA/JPEG/ISP periodicities shared within a
camera model — mandatory, otherwise the "device" test is contaminated (§7.1).

**Identity decomposition** (what cleaning does and does not remove):

    K̂_A = K_A + κ_model + ε_A ,   K̂_B = K_B + κ_model + ε_B

κ_model = residual model-shared component after cleaning (reported as NCC(K̂_A, K̂_B), a nuisance
parameter). The **paired contrast** Δρ = ρ(Z→K̂_A) − ρ(Z→K̂_B) cancels κ_model to first order —
this is why device-level claims survive imperfect cleaning (§7.1).

**Defect decomposition.** K̂ = K_defect + K_prnu via median filter (5×5): salt-pepper outliers
(hot/dead pixels, dust shadows — high-amplitude, low-count) vs the dense weak PRNU field. Leakage
is measured against K̂, K_defect, K_prnu separately. **Pre-registered prediction H-COMP:**
leakage, if any, appears in K_defect first (per-component SNR ordering). This converts "which
part leaked?" from a reviewer attack into a result.

**Matched-filter statistic (per probe image Z).** Under the multiplicative model the expected
fingerprint term in W(Z) is Z ⊙ K̂, so:

    ρ_mult(Z, K̂) = corr( W(Z), Z ⊙ K̂ )        [primary — mirrors forensic practice]
    ρ_add (Z, K̂) = corr( W(Z), K̂ )            [secondary — mechanism probe]

If ρ_add ≥ ρ_mult on gens while reals show the reverse, the learned pattern is an additive
texture overlay, not scene-modulated PRNU behavior — mechanism-informative either way.

**PCE₀ (signed, zero-lag).** With FFT cross-correlation surface C(s):

    PCE₀ = sign(C(0)) · C(0)² / ( (1/(m−|N|)) Σ_{s∉N} C(s)² ),   N = 11×11 exclusion.

Zero-lag is primary (fixed-crop rationale). **PCE_max** = same over argmax shift, secondary, with
empirical null from N2 (never a textbook threshold — no hardcoded baselines).

---

## 3. Pooling and a-priori power

Per-image ρ under H0 has sd ≈ 1/√m (null correlation of standardized fields). Primary pooled
statistic = mean of per-image ρ_mult over G gens:

    SE_H0( ρ̄ ) = 1/(√m · √G)

Minimum detectable per-image effect at one-sided α = 0.01 (z = 2.326):

    MDE(ρ̄) = 2.326 / (√m √G) = 2.326 / (1024 · √500) ≈ 1.02 × 10⁻⁴

Minimum detectable leakage ratio:

    λ_min = MDE(ρ̄) / ρ̂_real

ρ̂_real is **measured** in S1 (positive control on split H), then λ_min is printed before any
GPU training spend. Expected ρ̂_real for 1024² natural-scene crops ≈ 0.02–0.05 → **λ_min ≈
0.2–0.5%.** If the experiment nulls, the paper states λ < λ_min — a *bounded* null, not an
absence claim. (Equivalent view: pooling raises achieved NCC of the mean-residual by √G while
the null sd stays 1/√m; forms (a) pooled-residual-then-correlate and (b) mean-of-ρ are
first-order equivalent; (b) is primary because inference is a clean permutation test, (a) is
confirmatory.)

---

## 4. Hypotheses (pre-registered, exact statistics)

All tests one-sided α = 0.01 unless stated; permutation (10⁴) for group differences; BCa
bootstrap (10⁴) for CIs; a gate passes only if it passes in ≥2/3 training seeds individually
AND pooled across seeds.

- **H-EX (existence).** ρ̄(A-gens → K̂_A) > ρ̄(D-gens → K̂_A).
  D = LoRA trained identically on a disjoint device (different model). This null (N2) absorbs
  every generation-side artifact: denoiser–diffusion-texture interaction, VAE decoder
  fingerprint, PNG statistics. Gate **G1**.
- **H-DEV (device > model).** Paired per-image Δρ = ρ(A-gen→K̂_A) − ρ(A-gen→K̂_B), same image
  scored against both fingerprints; BCa CI of mean Δρ excludes 0. B = same camera model as A.
  Symmetric replication with LoRA-B (B-gens: K̂_B vs K̂_A). Gate **G2**. Fail with G1 pass ⇒
  model-level leakage only (L2).
- **H-MECH (distributional vs memorization).** Recompute G1–G2 excluding copy-flagged gens
  (§7.7). Survives ⇒ distributional (L4). Gate **G3**.
- **H-COMP (component ordering).** ρ̄ against K_defect vs K_prnu; report ordering + CIs. Not a
  gate — a pre-registered prediction.
- **H-DENS (optional arm, flag PILOT_EXT).** λ(SD3.5) > λ(SD1.5) on identical T_A. Validates the
  density law in a new channel; skip if compute-constrained.
- **G4 (stage-2, not pilot).** Dose–response: λ monotone in N ∈ {10,25,50,100} and LoRA rank.
  A flat-positive curve is the artifact signature; monotone is the money figure.

---

## 5. Null architecture

| Null | Construction | Kills |
|---|---|---|
| **N2 (primary)** | D-gens (LoRA on disjoint device, different model) scored vs K̂_A | denoiser–texture interaction, decoder fingerprint, PNG stats, generic gen artifacts — the honest FAR floor |
| N1 | circular shifts of K̂ (500 draws) vs pooled residual | alignment-free distributional null for the pooled statistic |
| N3 | base-model gens (no LoRA) vs K̂_A | base-training contamination audit (Dresden is public since 2010 and may be in web-scale crawls — §7.4) |
| N4 | A-gens vs K̂_C (third camera model, fingerprint only, no LoRA) | cross-model specificity floor |

Primary hypothesis tests use N2. N1/N3/N4 are audits; N3 > null forces all effects to be
reported as Δ over base and is itself a finding (web-crawl fingerprint contamination).

---

## 6. Adversarial confound registry

| # | Threat (reviewer voice) | Kill / control | Residual risk |
|---|---|---|---|
| 1 | "It's the camera **model** (ISP/CFA/JPEG), not the device." | Same-model pair A,B; NUA cleaning (ZM+Wiener); paired Δρ cancels κ_model; NCC(K̂_A,K̂_B) reported | κ_model 2nd-order terms — bounded by reported NCC |
| 2 | "K̂ and the LoRA saw the **same images** — content leaked into K̂ correlates with content the LoRA memorized." (fingerprint estimates leak scene content — known result) | E ⊥ T ⊥ H splits, filename-level assert, hard halt on overlap | none |
| 3 | "Same **scenes** across splits within a device (Dresden shoots repeated scenes)." | Contiguous-block split by sorted filename (temporally adjacent shots of a scene stay together); DINOv2 cross-split similarity audit printed; Dresden's same-scenes-across-devices property makes residual content leakage symmetric in K̂_A, K̂_B ⇒ cancelled by the paired contrast | small; quantified by the audit |
| 4 | "**Dresden is in LAION** — the base model already saw device A." | N3 audit measured first; primary contrast is LoRA-A-gens vs LoRA-D-gens (shared base cancels); if N3 > null, report as finding + Δ-over-base analysis | none for H-EX; N3-positive changes narrative, not validity |
| 5 | "The VAE killed the fingerprint before the DiT ever saw it." | Base = SD-3.5-medium (16-ch, high density) chosen **by your own density law**; optional SD1.5 arm turns the confound into prediction H-DENS | none |
| 6 | "Conv nets can't encode absolute position — the design is unfalsifiable." | DiT has explicit positional embeddings; fixed-crop maximizes representability; documented as existence design | arch×density confounded across SD1.5/SD3.5 — stage-2 2×2 |
| 7 | "The LoRA just **memorized whole images**; this is Carlini-style extraction, not fingerprint learning." | Copy audit: DINOv2 NN similarity of every gen vs T∪E, flag at cos ≥ 0.90 (sensitivity 0.85/0.95); G1–G2 rerun copy-excluded (gate G3); fig: NN-sim vs PCE₀ scatter | threshold choice — pre-registered + sensitivity band |
| 8 | "Positive PCE is 20 **hot pixels**, not PRNU." | Defect decomposition; component-wise leakage; H-COMP converts to result; claim wording keyed to component ("device-identity" vs "PRNU proper") | none |
| 9 | "Your denoiser responds to **diffusion textures** — inflated false positives." | N2 is the primary null: identical generation process, wrong device | none |
| 10 | "JPEG (reals) vs PNG (gens) asymmetry." | All gens PNG, no JPEG anywhere post-generation (asserted); affects λ calibration slightly (noted), not hypothesis tests (all gen-vs-gen) | λ point estimate only |
| 11 | "Fixed-crop training is unrealistic." | Stated as maximal-sensitivity existence protocol; random-crop = stage-2 realism arm; a fixed-crop-only positive is itself novel ("leakage requires pixel-locking") | scope of claim, not validity |
| 12 | "Attention-rank-16 LoRA can't store per-pixel patterns — absence of evidence." | Capacity ceiling arm: r=64 + FF modules (A and D, seed 0). Null must hold at ceiling to count | full-FT ceiling deferred to stage-2 |
| 13 | "One training run — stochastic fluke." | 3 training seeds per config; per-seed gate + pooled | — |
| 14 | "Zero-lag assumption smuggles in the conclusion." | Fixed-crop makes zero-lag the H1-consistent readout; PCE_max secondary with N2-empirical null covers translating patterns | — |
| 15 | "Template choice arbitrary." | ρ_mult primary (forensic standard), ρ_add secondary; divergence is a mechanism readout, §2 | — |
| 16 | "Orientation/resolution mixing within device." | Integrity gate: uniform resolution+orientation enforced, drops logged, halt below split minimums | — |

---

## 7. Frozen design parameters

| Parameter | Value |
|---|---|
| Base model | `stabilityai/stable-diffusion-3.5-medium` (bf16 train; measurement fp32) |
| Crop window Ω | center 1024×1024, native resolution, recorded in manifest |
| Devices | **A=Nikon_D200_1, B=Nikon_D200_0** (same model, 3872x2592); C=Agfa_DC-733s_0 (control); **D=Nikon_D70_1 (primary null LoRA — DSLR, matched class)**; D2=Agfa_DC-830i_0 (secondary null, seed 0, null-choice robustness) |
| Splits per device | **E=200 (A,B) / 60 (C,D,D2)**, T=50 (LoRA), H=40 (positive control); contiguous blocks by sorted filename with a **10-image guard band** at each boundary; DINOv2 cross-split scene audit, halt at sim>=0.95 |
| LoRA (primary) | attn {to_q,to_k,to_v,to_out.0}, r=16, α=16, lr 1e-4 cosine, 2000 steps, eff. batch 4, grad ckpt, uniform caption "a photograph, sks style" |
| Capacity ceiling | + FF modules {ff.net.0.proj, ff.net.2}, r=64, 3000 steps — devices A, D, seed 0 only |
| Seeds | training {0,1,2}; generation seeds derived per adapter |
| Generation | G=500/adapter + 500 base-null; 28 steps, CFG 4.5, 1024², PNG, txt2img only (asserted) |
| Statistics | α=0.01 one-sided; permutation 10⁴; BCa 10⁴; PCE exclusion 11×11; copy flag cos≥0.90 (±0.05 sens.) |
| Compute | 9 primary + 1 secondary-null + 2 ceiling adapters ≈ 11–16 A100-h; ~6000 gens ≈ 4–5 h; residuals CPU/GPU-light. **≈ 1 GPU-day total.** Fallback chain: SD3.5-m@1024 → **native 512 centre crop** (no resampling; power recomputed, MDE doubles) → SDXL@1024 (density caveat printed). Path taken enters the config hash. |

Config is a frozen dataclass; its SHA-256 is printed at every run (CBLB protocol-hash practice).

## 8. Decision tree

    S1 gates fail (K̂ quality: real-vs-K̂ AUC < 0.99, or ρ̂_real too low → λ_min > 5%)
        → HALT. Fix extractor / add E images. No GPU training spend.
    G1 fail (all seeds, incl. ceiling arm)          → L1 bounded null. Write-up decision point.
    G1 pass, G2 fail                                → L2 model-level leakage paper.
    G1 ∧ G2 pass, G3 fail                           → L3 memorization-mediated.
    G1 ∧ G2 ∧ G3 pass                               → L4. Proceed to stage-2 (dose–response G4,
                                                      2×2 arch/density, random-crop realism,
                                                      certified detector → §9 gate first).

## 9. Disclosure / patent gate

Measurement findings (any rung) are publishable. The moment the work turns toward a **certified
training-set-attribution detector with an FAR guarantee** — the commercially valuable form — it
falls under the standing rule: Indian provisional **before** any public disclosure. The pilot
deliberately measures and does not construct the certified detector.

## 10. Thesis fit

A2 (Conservation Asymmetry): generation destroys I(source; output) and creates I(generator; output).
This pilot is the inverse arrow: **training creates I(source; weights)**. Together they close a
provenance triangle — source ↔ generator ↔ output — with your density law as the capacity
precondition on the source→weights edge. That is a thesis-level unification, not a bolt-on topic.

## 11. Known unknowns (stated, not hidden)

- Exact Dresden device instances/counts on your Drive — runtime discovery with printed table and
  hard halt + suggestions if no same-model pair ≥150 images.
- SD-3.5-medium gated repo: requires HF token + accepted license (hf_env bootstrap).
- diffusers/peft API drift: training cell mirrors the official SD3 LoRA recipe; pin versions on
  first run and record them in the config hash.
- Colab VRAM at 1024: gradient checkpointing on; fallback chain above.

## 12. Notebook stage map

S0 env+manifest (discovery, splits, integrity halts) → S1 fingerprint bank (K̂, cleaning,
decomposition, quality gates, ρ̂_real, **power box prints λ_min**) → S2 LoRA training (resumable,
skip-if-exists) → S3 generation (resumable shards, base null) → S4 copy audit → S5 measurement
(per-row CSV: every gen × every K̂ × both templates) → S6 integrity audit (NaN/variance/parity —
halt before any figure) → S7 statistics, gates, verdicts, consolidated JSON, figures.


---

## 13. v2 amendments (post-S0 run, 2026-07-27)

S0 discovered 74 Dresden devices and picked Nikon_D200_1/_0 (380/372 imgs, 3872x2592, zero
non-uniform drops). Six design/code corrections applied:

1. **Null device D swapped** Agfa_DC-830i_0 -> Nikon_D70_1. The auto-picker satisfied
   "different camera model" but broke N2's intent: a compact vs a DSLR differ in JPEG character,
   noise structure and scene content, so LoRA-D would differ from LoRA-A in training-set image
   statistics as well as device identity — exactly the confound N2 exists to remove. D70 is a
   different model on comparable DSLR class; shared Nikon ISP lineage makes the null *more*
   conservative. Agfa retained as secondary null D2 (seed 0) to answer "your result depends on
   null choice".
2. **N_E raised to 200 for A,B.** K̂ estimation noise attenuates ρ̂_real and λ_min = MDE/ρ̂_real,
   so a noisier fingerprint costs detection power for zero compute. A and B held equal (200 each)
   because G2's paired contrast is biased if their estimation quality differs.
3. **Guard band (10 imgs) + cross-split scene audit.** Dresden shoots ~83 scenes per device in
   sequence; a contiguous block boundary can cut a scene and put near-duplicate frames in E and T.
   Guard drops the boundary images; DINOv2 audit (thumbnail fallback) halts above 0.95.
4. **λ_min now uses device A's own ρ̂_real**, not the median pooled over all roles.
5. **All resampling removed.** The 512 fallback is a native centre crop, not a resize — v1
   resized training images but not K̂, breaking pixel alignment (and shape) in S5.
6. Code fixes: `uniform_filter` arity bug, dead crop-offset lines, `np.seterr(all="raise")`
   (raised on benign underflow), silent stage no-op, protocol SHA polluted by session fields,
   four duplicated tag lists -> `all_tags()`.

**Numeric verification of the measurement core** (synthetic PRNU, 256px, 40 estimation images):
NCC(K̂, K_true) = 0.877; matched-filter ρ same-device 0.510 vs wrong-device 0.0004 (1237x);
PCE peak at shift (0,0) as the fixed-crop design requires. Power box: MDE = 1.016e-4 at
MEAS=1024/G=500, so λ_min = 0.34% at ρ̂_real = 0.03.

**Split feasibility against the actual S0 counts:** A 380/310 (slack 70), B 372/310 (62),
C 562/170 (392), D 189/170 (19 — tight; if the uniform-size filter drops any D70 images, lower
N_E_AUX to 40 rather than switching device), D2 363/170 (193).


---

## 14. v3 amendments — survey response (2026-07-28)

An external novelty/design survey was run against v1. Its verdict (proceed; novelty defensible
only as *passive, natural, device-level* leakage, bounded by Yu et al. ICCV'21 artificial
fingerprints, ProMark CVPR'24, CoprGuard CVPR'25 and the Adobe diffusion-watermarking patents)
is adopted. Its design criticisms are adopted with three corrections.

### 14.1 Power — the survey was right, and its own fix is superseded
v1 §3 computed MDE = z_{1-α}/√(mG) and called it the minimum detectable effect while the L1 rung
claimed power 0.99. Those are inconsistent: that expression is the α-threshold, i.e. **50% power**.
The survey's correction, MDE = (z_{1-α}+z_{1-β})/√(mG), is arithmetically right but still assumes
independent pixels — which wavelet residuals, CFA structure and JPEG blocking violate.

**v3 replaces the analytic route entirely.** For each generated image the full normalized
cross-correlation surface C(s) is computed by FFT (already required for PCE, so this is free), and
surfaces are averaged over the G images of an adapter. C̄(0) is the statistic; the off-peak values
of C̄ are draws from the true null of that same statistic, under the real spatial dependence.
Then

    SE_emp = sd{ C̄(s) : s outside the 11×11 exclusion }
    λ_min  = (z_{1-α} + z_{1-β}) · SE_emp / ρ̂_real(A)

A synthetic check (256², G=60, spatially correlated scenes) gives SE_emp/SE_analytic = **5.8×** —
i.e. the analytic bound would have overstated sensitivity by nearly a factor of six. The
inflation factor is printed at run time and belongs in the paper.

Note the error was confined to the a-priori power claim: v2's inference was always permutation
against an empirical null, so test calibration was never affected.

### 14.2 Primary statistic — intersection–union
Primary is now the symmetric same-model contrast, per generated image:

    S_A(Z) = ρ_mult(Z, K̂_A) − ρ_mult(Z, K̂_B)   on device-A adapters
    S_B(Z) = ρ_mult(Z, K̂_B) − ρ_mult(Z, K̂_A)   on device-B adapters
    H1: E[S_A] > 0  ∧  E[S_B] > 0

tested by sign-flip permutation. Framing it as an **intersection–union test** matters: an IUT is
exact at level α for the intersection null, so combining the two contrasts needs no multiplicity
adjustment, and requiring both kills the fingerprint-norm-asymmetry explanation that either one
alone admits. Cross-model device D is retained as a secondary null — it absorbs
generation-process artefacts (denoiser response to diffusion texture, decoder fingerprint) that a
same-model contrast cannot, which is why the survey's suggestion to drop it is declined.

### 14.3 Causal arms — with a correction the survey missed
Three training sets from the *same* T-split images: raw, clean (Y·(1−αK̂)), swap
(Y·(1−αK̂_A)·(1+αK̂_B)). The survey did not note that cleaning and measuring with the same K̂
manufactures an artefact: cleaned images carry a negative imprint of that estimate's error, so a
spurious negative correlation follows by construction. **v3 splits E into E1 (80 images, used only
to modify images) and E2 (140, used only to measure).** Suppression and injection efficacy are
verified on the held-out H split before any training, with a 5× suppression gate.

Second correction: the survey presents the swap arm as raising novelty. It does the opposite — an
injected fingerprint is exactly the Yu et al./ProMark setting. Its real value is as a **pipeline
positive control**: it converts a null from "we detected nothing" into "the chain detects an
injected fingerprint at amplitude X and not the natural one at amplitude Y", which is what makes
a C0 result publishable rather than unfalsifiable. Synthetic verification of the operators:
suppression 98.6×, injection gain 226×.

### 14.4 Copy detection — peakedness, not maximum
Ensemble: DINOv2 full-image, DINOv2 on 4 quadrant crops (partial copies), pHash (near-exact), and
a residual-space test. The survey proposed residual **nearest-neighbour maximum**; that statistic
would flag the leakage signal itself as copying, because a shared fingerprint lifts the
correlation against *every* training image. v3 uses **peakedness** — (max − median)/MAD over the
training residual bank — which is elevated by a memorized copy and flat under distributional
leakage. Reference bank restricted to the T split (only trainable images can be memorized) and
stored fp16.

### 14.5 Multiplicity plan
The survey added ~9 statistics without one. v3: one primary IUT at α=0.01 (no correction),
then a 9-test secondary family at α=0.05 Holm-corrected and **hierarchically gated** — evaluated
only if the primary passes. Everything else is descriptive.

### 14.6 Other adopted items
Exact SD-3.5 VAE round-trip gate as S1b (measure survival on the actual autoencoder; AUC floor
0.75) — the density law is now a design rationale, not evidence. Paired generation seeds: every
adapter draws from one pre-registered (prompt, ε) bank. Sign-agnostic secondaries (two-sided
permutation + energy distance) in case the model learns an inverted pattern. C0–C4 ladder
replacing L1–L4, with C3 (causal) above C2 (device-level). Fixed-crop wording softened: a null is
evidence against **zero-lag pixel-locked leakage for this architecture and adaptation method**,
not against leakage in general.

### 14.7 Declined
50-prompt primary bank — varied prompts pull generations away from the mode the LoRA was fit to
and would likely *reduce* leakage; kept as an optional secondary bank. Dropping device D (14.2).
Smartphone replication cannot come from Dresden (a 2010 dataset) — VISION/FODB, at Phase 4.

### 14.8 One citation defect in the survey
Its claim that convolutional networks encode absolute position via padding is attributed to
Peebles & Xie's DiT paper, which is not about that. The correct source is Islam et al., ICLR 2020.
All of the survey's URLs carry `utm_source=chatgpt.com`; every citation needs verification before
it enters a related-work section.

### 14.9 v3 verification performed
AST parse of all cells; cross-cell undefined-name and call-before-definition analysis (both
clean); numeric identity check C̄(0) = ncc0; empirical-null construction; clean/swap operator
efficacy; Holm implementation against a hand-computed example; split feasibility against the real
S0 counts (A 380/340, B 372/340, C 562/170, D 189/170, D2 363/170); and a **full mock execution of
S6+S7** on synthetic measurement CSVs in two worlds — a planted-effect world returns C4, a
no-effect world returns C0 with the primary IUT correctly failing and Holm thresholds withheld.

### 14.10 Budget
14 adapters (12 × 2000 steps, 2 × 3000), 15 tags × 500 = 7500 generations, 37 500 measurement
rows. ≈ 20–26 A100-h training, ≈ 6 h generation, ≈ 4–6 h measurement.
