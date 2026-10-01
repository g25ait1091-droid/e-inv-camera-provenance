# E-INV v2 — results record (append-only)

Entries are appended, never edited. A quantity that is later withdrawn or superseded is listed in
the register below with the entry that first reported it; the entry itself stays as written.

## Register of withdrawn and superseded quantities

| # | quantity | first reported in | status | replaced by |
|---|---|---|---|---|
| R1 | "DiffusionShield → NeurIPS 2023; FT-Shield → SIGKDD Expl. 2025" (two bibliography corrections) | Entry 08 | **withdrawn** | Entry 12: entries were correct (SIGKDD Explorations 26(2), Dec. 2024) |
| R2 | Pre-declared reading "colab ≈ nomark → v2 stack transmits the natural fingerprint at λ ≈ 0.2 %, the study's first positive PRNU-transfer result" | Entry 31 | **withdrawn** (D6) | Entry 35: F6 shows only that the decoder is not the cause; transfer needs the symmetric statistic (F7, Entry 35a) |
| R3 | "λ = 0.216 % [0.097, 0.336]" and "above v1's upper limit" for the nine v2-trained adapters | Entry 24 (addendum) | **not a transfer quantity** — an unpaired contrast of A-trained adapters only | Entry 35a / F7 θ_sym |
| R4 | Explanation "the pipeline is a low-pass channel; the DiffusionShield pattern lives at 1/8–1/64 cycles per pixel" (Entry 33 reading (2); carried into the manuscript) | Entry 33 | **withdrawn** — the mark's spectrum is high-frequency (Entry 40); the autoencoder is low-pass (Entry 42) but does not select the watermark | Entry 45: transfer is set by spatial structure (repetition, and repetition on the latent grid) |
| R5 | "Transmission is governed by spatial structure rather than by frequency content; repetition lifts a fixed pattern from undetectable to 0.17 %" | Entry 45 (and the manuscript of 2026-09-11) | **refined** — frequency sets a floor (the finest octave passes nothing, no octave more than about 0.4 %); against a spectrum-matched non-repeating prediction, off-grid repetition gives about 3x and on-grid repetition about 85x | Entry 46 |
| R6 | "The design excludes transfer above the limit but would not resolve transfer at the level the channel predicts" (manuscript v4, Limitations; "just under the limit", Sections V-D and I) | manuscript 2026-09-12/13 | **withdrawn** — true only for the max-arm limit; the symmetric statistic's one-sided 99 % limit is 0.076 %, 3.4 SE below the 0.104 % prediction (power 0.92 at 0.10 %) | Entry 50 |
| R7 | Power translation with SE_img(500) = 4.67e-05 (TPR 0.059 at 500 images; 0.12 at sigma_mu = 0) | Entry 03 | **superseded** — the per-image rows give 7.0e-05 (D7) | Entry 50 (`t3_power_v4.json`) |
| R8 | "The autoencoder's frequency response predicts the retention" (39–41 %) | Entry 42 | **refined** — body A 39–41 %, body B 36–38 %, against 36.6 % measured (95 % CI 34.4–38.7 %); an approximation, not a prediction inside the interval for A | Entry 50 |
| R9 | "The fingerprint passes less than a non-repeating pattern with its spectrum" (Entry 50/R6 era; withdrawn in Entry 63 for estimator sensitivity) | Entries 50, 63 | **settled** — direct injection gives parity: at matched amplitude the fingerprint is 0.78x and 1.33x a spectrum-matched random field; the band-response prediction itself is high by ~2.7x | Entry 72 |
| R10 | "The learned detector's body-specific signal is not the fingerprint template and its source is unidentified (training-set content an open alternative)" | Entries 45, 50, 64 | **resolved** — content-matched training sets leave 21 % of the unmatched interaction, unresolved (p 0.28): the signal tracks training-set content | Entry 71 |
| R11 | "At 16000 steps the memorization regime is unresolved" (manuscript v4, Entry 54) | Entry 54 | **superseded** — the registered replication resolves it: theta_sym 0.16 %, one-sided p 0.010 (new three adapters per body), p 0.002 pooled over six | Entry 68 |

## Defect log

| # | defect | found in | consequence |
|---|---|---|---|
| D1 | v2 materialisation wrote injected PNGs by truncation (`astype`), recording sign(M) at −1 LSB | Entry 04 | fixed before training; rebuilt with `np.rint` |
| D2 | `pkill -f` did not kill the serial panel on Windows; two processes appended to one CSV | Entry 10 | duplicates verified identical and dropped; kill by PID now |
| D3 | reference audit trusted API 'year' over the cited issue date | Entry 12 | two spurious corrections withdrawn |
| D4 | a Figure 1 caption written through a shell heredoc had its backslashes collapsed: `\rho`/`\to` became LF+`ho`/`o` in the tex | (this session) | caught by the SyntaxWarning, repaired, verified no control characters remain; rule added to PLAN |
| D5 | re-training the learned detector under the same seed did not reproduce the first network (cuDNN nondeterminism); the first network was never saved | Entry 18 | saved weights `t2_learned_net.pt` are the detector of record; both draws disclosed |
| D6 | the F6 registration pre-declared a transfer reading on single-body (A-trained) arms, which the study's own paired design cannot support | Entry 35 | reading withdrawn (R2, R3) before it reached the paper; F7 B-body arms registered |
| D7 | the power translation used SE_img(500) = 4.67e-05, which does not match the per-image rows of the primary adapters (7.2e-05 and 6.9e-05) | Entry 50 | recomputed (`src/t3_power_v4.py`); attribution power falls further |

---

## Entry 00 · 2026-09-08 · Programme opened

Context: IEEE Access rejection (Access-2026-38689, 7 Sep 2026). Plan v14 written. Compute and
data inventory recorded in `PLAN.md`. Frozen v1 quantities that v2 must not disturb (from
`FINAL_LEDGER.json`, k = 12): U_device 5.3761e-05, λ_U 0.1507 %, τ_U 0.4117 %, θ_sym +4.5996e-06,
R_real 3.56703e-02, R_VAE 1.30593e-02, η 0.3661 [0.3437, 0.3870].

Nothing measured yet.

---

## Entry 01 · 2026-09-08 · Tier 1 pre-registration — the designed-mark ladder

Registered before any adapter is trained and before any Tier 1 measurement is examined.

**Question.** At what amplitude does a *designed* fixed-coordinate pattern — the signal class the
active-marking literature (Yu 2021, DiffusionShield, ProMark, CoprGuard) shows to reach
generated output — actually come out of *this* pipeline, measured with *this* study's statistic?
Natural PRNU is then placed on that ladder.

**Fields.**
- `M_rand`: pseudo-random ±1 field, 1024×1024, `numpy` seed 20260908, zero-mean, then scaled so
  that RMS(M_rand) = RMS(K̂_A^E1) = 1.0437e-3 (`out/fp/K_stats_A.json`). One nominal unit of
  `M_rand` therefore carries the same multiplicative energy as one nominal unit of the estimated
  fingerprint, and the α axis is the same as v1 Section V-D.
- `M_lowmid`: `M_rand` band-limited to 1 ≤ i+j ≤ 5 of the 8×8 block DCT (DC excluded), re-scaled to
  the same RMS.
- `M'`: a second independent ±1 field (seed 20260909), same normalisation, **never injected** — the
  null comparator for every mark measurement.
- `M_opt` (optional, only if the released DiffusionShield or ProMark code runs unmodified):
  the published mark at its published strength.

**A deliberate asymmetry, stated up front.** The designed field is *known exactly* to the
detector; the natural fingerprint is only *estimated* (NCC(K̂^E1, K̂^E2) = 0.362), which capped v1's
effective amplitude near 4×. The ladder therefore gives the designed mark every advantage a real
watermark has. That is the fair comparison — the question is whether an exactly-known, fixed,
non-semantic pattern transfers when a natural one does not — and the advantage must be named in
the write-up.

**Injection.** Multiplicative, identical to v1: Y_α = clip[Y·(1 + α·M)], applied to device A's
T split (50 training crops, 1024² native centre crop), 8-bit PNG.

**Arms** (rank-16 attention LoRA, lora_alpha 16, lr 1e-4 cosine, 2000 steps, effective batch 4,
bf16 training, caption `"a photograph, sks style"` via the archived prompt embedding
`_prompt_embeds_a1759453b6c6cd42.pt`, seed as stated):

| arm | field | α | seeds |
|---|---|---|---|
| `mark_rand_a1` | M_rand | 1 | 0 |
| `mark_rand_a3` | M_rand | 3 | 0, 1, 2 |
| `mark_rand_a12` | M_rand | 12 | 0 |
| `mark_lowmid_a12` | M_lowmid | 12 | 0 |
| `mark_opt` | M_opt | published | 0 (optional) |

Six or seven adapters. Generation: 500 images per arm from the v1 paired seed bank
(GEN_SEED_BASE 770000, uniform caption), 28 steps, CFG 4.5, 1024², PNG. Hardware: L40S 44.7 GB;
measurement fp32, TF32 disabled.

**Never-injected arms.** Every mark is also measured on the v1 archive's `base`, `A_raw_s0_r16`
and `B_raw_s0_r16` generations, which never received it. A mark that ranks first there is an
artifact, not transfer (v1 §V-K).

**Statistics.**
1. Per image: ρ(gen → M) − ρ(gen → M'), zero-lag NCC of the wavelet residual as in v1.
2. Displacement decoys: 30 circular rolls of M (spectrum-preserving), rank of the true M among 31.
3. Cluster-level: at α = 3, mean and sd over the three seeds; one-sided t against zero, α = 0.01.
4. Normalisation: R_mark = mean over the 50 injected training crops of [ρ(Y_α → M) − ρ(Y_α → M')]
   at that α — what the mark contributed in the inputs. λ_mark = contrast / R_mark, with the same
   plug-in upper limit construction as v1.
5. The natural paired contrast ρ(gen → K̂_A) − ρ(gen → K̂_B) is also recorded, to check that the
   mark does not disturb the v1 bound.

**Pre-declared outcomes and wording.**
- (a) `mark_rand_a3` contrast > 0 at cluster level *and* the true M outranks all 30 decoys *and*
  M does not rank first in any never-injected arm → *"A designed fixed pattern at three times one
  natural fingerprint's energy transfers at λ_mark = X %; the natural fingerprint at the same
  nominal energy is bounded below 0.15 %."* Report the full ladder.
- (b) no arm meets (a) → *"For fixed-coordinate patterns at PRNU-scale energy, this pipeline
  transmits neither the natural fingerprint nor an exactly-known designed one; the transfer
  reported by active-marking methods depends on optimisation or on amplitudes above this
  range."* This is reported as the result, not as a failure of the experiment.
- Mixed (transfer at α = 12 only) → report the threshold and say the channel is amplitude-gated.

**Confound registry additions.**
- *"The designed mark was learned because it is semantic."* — M_rand has no semantic structure;
  the 30 rolled decoys share its spectrum exactly and act as scrambled-phase controls.
- *"The mark leaked through the cached prompt embedding."* — the embedding is the v1 file, fixed
  before any mark existed.
- *"Detection is the injection template correlating with itself through clipping artefacts."* —
  the never-injected arms and M' bound this; clipped-pixel fraction is logged per α.

Nothing in this entry may be changed once `mark_rand_a1` begins training.

---

## Entry 02 · 2026-09-08 · Instrument reproduced; prior-art sweep, first pass

**Instrument.** The v1 forensic core was re-run on this machine from the Dresden copy on disk
with the deterministic S0 splits (E1 80 / E2 140 / T 50 / H 40, guard 10). `out/fp/gates.json`:

| quantity | v2 here | v1 paper | 
|---|---|---|
| κ_model = NCC(K̂_A^E2, K̂_B^E2) | 0.00745 | 0.0073 |
| split-half NCC(K̂_A^E1, K̂_A^E2) | 0.3619 | 0.362 |
| split-half NCC(K̂_B^E1, K̂_B^E2) | 0.2459 | 0.246 (§V-D, the effective-amplitude cap) |

All three agree to the precision the paper quotes. Every v2 measurement below is therefore made
with the same instrument the paper used, not a re-implementation of it.
RMS(K̂_A^E1) = 1.0437e-3; one nominal α = 1 injection is 0.125 grey levels RMS on the H split.

**Sweep** (`src/oa_sweep.py`, OpenAlex, citing works dated 2025-01-01 onward, filter = any
sensor-fingerprint term AND any generative term in title+abstract; `out/sweep_oa.json`):

| anchor | citing works since 2025 | matching | new to the bibliography |
|---|---|---|---|
| Yu et al. 2021, *Artificial Fingerprinting* | 79 | 0 | — |
| SIREN 2025 | 12 | 0 | — |
| ProMark 2024 | 17 | 0 | — |
| Klier & Baier 2026 | 0 | 0 | — |
| Chen et al. 2008 | 58 | 2 | none (both already cited) |
| Lukáš et al. 2006 | 114 | 4 | **"Spoofing Camera Source Attribution via PRNU Transfer Attacks on Physical and AI Generated Images" (LNCS, 2025)** — counter-forensic PRNU *injection onto* AI images; adjacent to SpoC, not passive transfer. Cite and distinguish. One further hit (Polish security journal, deepfake detection) is marginal. |

Web passes (Semantic Scholar rate-limited; plain web search) found no PRNU × LoRA, PRNU ×
DreamBooth or PRNU × personalisation study. **The WIFS 2026 accepted-paper list is not yet
public** (placeholder page dated 28 Jan 2026; conference 7–11 Dec 2026), so that scoop channel
stays open and must be re-checked before submission. Klier & Baier (DFRWS EU 2026 / FSI:DI,
24 Mar 2026) remains the nearest empirical neighbour; their reproduction target for Tier 2:
standard PCE threshold, false-positive rates 61 % (Adobe Firefly Image 4) and 100 % (ChatGPT 5)
against smartphone fingerprints, eliminated by centre-cropping, and the finding that generated
noise is not predominantly additive.

**Status of the novelty claim after the sweep:** holds, with one adjacent citation to add.

---

## Entry 03 · 2026-09-08 · Tier 3 — what the bound means for attribution (power translation)

`src/t3_power.py`, `out/t3_power.json`. Pure arithmetic on frozen v1 quantities; nothing
re-estimated. Model: one adapter's calibrated paired contrast over G generated images is
θ̂ ~ N(θ_true, σ_μ² + SE_img(500)²·500/G), with θ_true = λ·R_real; a two-candidate attribution
(own body vs the other body of the same model) at FPR 1 %, Bonferroni for M candidates. This is
the *most favourable* case for an examiner — every candidate fingerprint known exactly, main
effects calibrated out, no scene confound.

Inputs: U_device 5.3761e-05, R_real 3.56703e-02 (ledger); SE_img(G = 500) 4.67e-05 and the
persistent per-adapter structured component σ_μ = 5.23e-05, both from v1 §8.3 (sanity block,
√G-scaling test). **The interpretation of σ_μ as adapter-level noise of the paired contrast must
be re-checked against the sanity-block source before this enters the paper** — it was measured
as the persistent off-peak component of the correlation surface.

| true transfer | λ | M | TPR at G = 500 | TPR at G = 5000 | TPR at G → ∞ | images for TPR 0.5 |
|---|---|---|---|---|---|---|
| = upper limit | 0.151 % | 2 | 0.059 | 0.091 | **0.097** | never |
| = upper limit | 0.151 % | 5 | 0.021 | 0.035 | 0.038 | never |
| = upper limit | 0.151 % | 50 | 0.003 | 0.005 | 0.006 | never |
| 10× upper limit | 1.51 % | 2 / 5 / 50 | 1.000 | 1.000 | 1.000 | 50 / 50 / 100 |
| 100× upper limit | 15.1 % | any | 1.000 | 1.000 | 1.000 | 10 |

**Reading.** At the upper limit the signal is 1.03 σ_μ. Because σ_μ does not average out with
more images, the best attainable two-way true-positive rate at 1 % false positives is 9.7 %
*with unlimited generations*, and 0.6 % among fifty candidates. Attribution becomes practical
only if transfer were roughly ten times the bound (λ ≈ 1.5 %), where 50 images suffice. So the
operational meaning of 0.15 % is: **below the floor at which single-adapter attribution is
possible at all, by about an order of magnitude.** This is the sentence Reviewer 1.3 asked for.

Wording to carry: *"the bound lies an order of magnitude below the transfer at which
attribution of a personalised model to its source camera would become feasible under ideal
conditions."* Never: "attribution is impossible" — the statement is conditional on the
detector family and on σ_μ.

---

## Entry 04 · 2026-09-08 · Defect D1 caught before training; materialisation statistics; one amendment

**Defect D1 (v2, fixed).** The first materialisation wrote injected PNGs via
`astype(np.uint8)`, which truncates toward zero. For a multiplicative perturbation smaller than
one grey level, truncation records −1 LSB wherever the field is negative and nothing where it
is positive — the *sign* of the field, at full LSB strength, identically at every α below the
point where p·α·|M| ≥ 1. Evidence: `rand_a1` and `rand_a3` were byte-identical in 50/50 files;
on one crop, truncation changed 49.97 % of pixels (all by −1) at both α = 1 and α = 3, while
correct rounding changes 0.00 % at α = 1 and 13.28 % at α = 3. Fix: `np.rint` before the cast
(`src/t1_ladder.py:inject`). Both training sets were deleted and rebuilt. No adapter had been
trained, so Entry 01 is untouched.

**Materialisation, corrected** (`out/t1/train_png/materialise.json`, 50 crops each):

| arm | clipped pixels | R_mark = ρ(Y→M) − ρ(Y→M′) on the stored PNG | sd | natural A−B contrast on the same PNGs |
|---|---|---|---|---|
| rand α = 1 | 0.16 % | **0.0005** | 0.0014 | 0.0436 |
| rand α = 3 | 0.16 % | 0.234 | 0.152 | 0.0410 |
| rand α = 12 | 0.23 % | 0.661 | 0.106 | 0.0257 |
| lowmid α = 12 | 0.21 % | 0.504 | 0.126 | 0.0255 |

Field statistics (`out/t1/fields/field_stats.json`): all three fields at RMS 1.0437e-3 exactly;
NCC(M_rand, M′) = 9.5e-5; NCC(M_rand, K̂_A^E2) = 5.1e-4; NCC(M_rand, K̂_B^E2) = −6.0e-4;
NCC(M_rand, M_lowmid) = 0.559.

**What the α = 1 row means.** A 0.125-LSB-RMS field added to pixels that are *already
integers* rounds away almost completely — there is no analogue noise to dither it, unlike a
real sensor's PRNU, which is imprinted before quantisation. The stored α = 1 training set
therefore carries essentially no mark (R_mark ≈ 0.0005, i.e. 0.2 % of the α = 3 value), and
λ_mark is undefined for it (division by ~0).

**Amendment to Entry 01, made before any training.** `mark_rand_a1_s0` is retained and trained
as registered, but is re-designated a *materialisation null* (an arm whose training images
demonstrably carry no mark); it contributes to the never-injected comparisons and not to the
ladder. The ladder's lowest informative rung is α = 3. Outcome wording (a) already keys on α = 3
and is unchanged. Nothing else in Entry 01 changes.

**Consequence for the design axis.** For a field added after quantisation, nominal α is not
linear in stored energy: R_mark rises 0.0005 → 0.234 → 0.661 for α = 1 → 3 → 12. Every Tier 1
result will be reported against R_mark (the energy actually stored), never against nominal α
alone. This is the same lesson v1 recorded for estimated fields (nominal ≠ effective), arriving
from a different mechanism.

**Also recorded here, pending Entry 05.** v1's E-AMP A1 cell and the pilot's S1c cell write
injected training PNGs with the same `astype(np.uint8)`; whether v1's low-α arms carry a sign
pattern rather than a graded amplitude is being tested on the archived files themselves.

---

## Entry 05 · 2026-09-08 · v1 defect confirmed on the archived training images (disclosure required)

**Test.** Four archived training PNGs per arm were fetched from `E_INV_P0_v3/train_png/A_raw`
(uninjected, Colab decode) and `E_AMP/train_png/{amp_1p0, amp_1p6, amp_3p0, amp_12p0, ctrl_Q}`
(`data/v1_train_png/`). Differences are pixel-wise, injected minus uninjected, same decode.

| v1 arm | pixels changed vs `A_raw` | histogram of the change |
|---|---|---|
| `amp_1p0` | 50.08 % | −1: 50.08 %, +1: 0 % |
| `amp_1p6` | 50.08 % | −1: 50.07 %, +1: 0 % |
| `amp_3p0` | 50.3–50.6 % | −1: 49.6–49.8 %, ±2 and +1: 0.2–0.5 % |
| `amp_12p0` | 62–66 % | −1: 34–38 %, +1: 8.8–10.9 %, ±2: 9–11 %, tails to ±5 |

(Three T-split images each; `ctrl_Q` behaves like `amp_12p0`.) **Every injected-image arm in
v1 was stored by truncation.** The E-AMP A1 cell and the pilot's S1c cell write
`np.clip(...).astype(np.uint8)`; for a sub-LSB multiplicative perturbation on integer-valued
pixels this records −1 wherever the field is negative and nothing where it is positive.

**What the v1 arms actually contained.**
- `amp_1p0` and `amp_1p6`: the same image — raw minus a −1 LSB step on the half of the pixels
  where K̂_B^E1 < 0. That is a 1-bit sign map of the fingerprint at ~0.7 LSB RMS, roughly 5.7×
  the intended 0.125 LSB, and its correlation with K̂_B is that of a Gaussian variable with its
  own sign (≈ 0.8) — not "one natural amplitude" and not "1.6 natural amplitudes".
- `amp_3p0`: the same sign map plus amplitude-bearing changes on 0.2–0.5 % of pixels.
- `amp_6p0`, `amp_12p0`, `ctrl_Q`, `ctrl_shift`, and the E-AMPHI ceiling arms: the sign map
  plus a genuinely graded field.
- Pilot `A_clean` (Y·(1−K̂_A)) and `A_swap`: sign maps of the respective fields, not suppression
  or substitution. The 7.0× suppression and 201× injection-gain gate (§V-A) was measured on
  float arrays and describes images that were never stored.

**What this changes in v1's claims.**
1. Stage 2's "undetectable at ≤ 1.6×, onset ≈ 3×" is an artefact of materialisation: the two
   lowest arms are the same image, so their equal losses (0.17085 vs 0.17086) are trivial, and
   the "onset" is where amplitude-bearing pixels first appear. The dose–response x-axis is
   invalid below roughly α = 6 and must be re-stated on the energy actually stored.
2. The α_effective axis (0.39, 0.63, 1.15, 2.17, 3.48) was measured on unquantised float
   images (A1 real dose-response) and does not describe the training sets at any α.
3. The high-α generation-side null (§V-D, "deliberate amplification does not transfer") stands:
   those arms carried a strong field, and nothing came out. Its stated amplitude needs
   correcting; its verdict does not.
4. The causal clean/swap arms are uninterpretable as suppression/substitution; they were
   reported as controls that showed nothing, and that report is still true, but the wording
   "suppressed" must go.
5. **Untouched:** Stage 1, the primary bound and every replication (all on raw arms, which are
   lossless integer crops), the detector calibration (E-POST used a dither model and rounded
   once — v1 found exactly this bug there and fixed it), and the additive decomposition.

**Two other findings from the same test.**
- The Colab JPEG decode differs from this machine's: `A_raw` vs my decode of the same file,
  28 % of pixels differ by up to ±4, symmetric (chroma-upsampling / IDCT implementation).
  The forensic core is insensitive to it (κ and split-half reproduced to three figures, Entry 02),
  but v2's Tier 1 training images are built on the local decode. Registered statistic 5
  (natural contrast on mark arms vs v1 raw arms) therefore carries a decode difference and
  will be read within v2 only.
- Rounding does not rescue the low rungs: a 0.125-LSB field on integer pixels rounds away
  (Entry 04). Sub-LSB injection into stored training images needs an explicit dither before
  the single rounding — the E-POST construction — and that would be a new, separately
  registered arm (`mark_rand_a1_dither`), not a change to this one.

**Disclosure text to carry into the rewrite, verbatim or close:** *"The injected training-image
arms of the amplitude sweep, the amplitude ceiling, and the clean/swap controls were stored with
truncation rather than rounding. For α ≤ 3 the stored perturbation is a one-bit sign map of the
fingerprint at −1 LSB, identical across those arms, rather than a graded amplitude; the reported
onset near 3× nominal is where amplitude-bearing pixels first appear. The high-amplitude null
and all raw-arm results are unaffected. We report the sweep against the energy measured on the
stored images."* This is an instance of the paper's own §7 lesson (calibrate on the actual
medium, check the calibration is not a no-op) that the paper applied to E-POST and missed here.

---

## Entry 06 · 2026-09-08 · Tier 3 pre-registration — closed-set attribution on the archived five-body generations

Registered before any attribution number is computed. Data: the archive's own generations
for the Kodak M1063 group (`E_MULTIDEV/gens`, five bodies D0–D4 × seeds 0,1) and the Huawei
P20 group (`E_DAXING/gens`, bodies 1101–1105 × seeds 0,1), 250 images per adapter (the first
250 of each 500-image paired seed bank), with the archive's own E2 fingerprints
(`K_D*_E2.npy`, and the leave-one-device-out residualised `K_D*_E2_lodo.npy`; for P20 the
residualised `KR_*_E2.npy`).

**Question.** Given G generated images from one personalised model and the five candidate
bodies of its camera model, how often is the model attributed to the body whose photographs
trained it? This is the examiner's task; Entry 03 predicts it is near chance.

**Scores** (per image, then averaged over the G images of an adapter):
1. *Raw:* s_d = ρ(gen → K_d), the paper's zero-lag statistic. Argmax over d.
2. *Main-effect-corrected (the honest score):* s_d − b̂_d, where b̂_d is the mean of ρ(gen′ → K_d)
   over all generations of the *other four devices'* adapters (leave-the-candidate-out estimate
   of the fingerprint main effect, v1 §V-H). Argmax over d.
3. Real-photo ceiling: the same two scores on each device's held-out real H images
   (Dresden, deterministic C0 splits) — what attribution looks like when the signal is there.

**Reported:** top-1 accuracy at G ∈ {1, 10, 50, 250} for each score, per device group, with
the adapter as the unit (10 adapters per group; exact binomial interval against chance 0.2);
the confusion matrix at G = 250; and the same for real photographs at G ∈ {1, 10, 40}.

**Pre-declared reading.**
- Raw argmax will *not* be at chance but will *not* track the training device either — it
  will select whichever fingerprint has the largest main effect, for every adapter alike. That
  is the additive decomposition's prediction and is the practical hazard Klier & Baier
  describe. Report the per-device selection frequency to show it.
- Corrected argmax at G = 250 within the binomial 95 % interval of 0.2 → *"attribution of a
  personalised model to its source body is at chance under the most favourable closed-set
  conditions."* Above the interval → report the accuracy and revisit Entry 03's σ_μ.
- Real-photo ceiling ≥ 0.95 at G = 10 is required for the generated-image number to be
  interpretable; if the ceiling fails, the group is reported as an instrument failure, not a
  result.

**Not done here:** no new fingerprints, no new generations, no threshold tuning. The 250-image
subset is fixed as the first 250 seeds of the paired bank.

---

## Entry 07 · 2026-09-08 · Tier 3 result — closed-set attribution is at chance; raw argmax follows the fingerprint main effect

`src/t3_attrib.py`, `out/t3_attrib.json`. Computed on the archive's own per-row measurements
(`c3_measure_raw.csv`, `c3_measure_lodo.csv`, `d5_measure.csv`, and the real-photo controls
`c0_positive_control.csv`, `d1_positive_K.csv`), first 250 generations per adapter, exactly as
registered in Entry 06. Chance = 0.2.

**Real-photo ceiling (the instrument works).** Kodak: 0.99 at G = 1 (200 images), 1.00 at
G = 10 and 40. P20: 1.00 at G = 1 (150 images), 10 and 30. Both groups pass the ≥ 0.95 gate.

**Generated images — top-1 accuracy (block accuracy; adapters majority-correct out of 10).**

| group / score | G = 1 | G = 10 | G = 50 | G = 250 (one block per adapter) | G = 250: which fingerprint gets picked |
|---|---|---|---|---|---|
| Kodak, raw K, raw argmax | 0.206 | 0.228 | 0.180 (2/10) | 0.300 (3/10; CI 0.07–0.65) | **D0 ×7**, D1 ×2, D4 ×1 |
| Kodak, raw K, main-effect-corrected | 0.207 | 0.220 | 0.240 (1/10) | 0.300 (3/10) | D0 3, D1 3, D2 1, D3 1, D4 2 |
| Kodak, LODO K (both scores) | identical to raw K to the third decimal | | | | |
| P20, residualised K, raw argmax | 0.216 | 0.256 | 0.280 (2/10) | 0.200 (2/10) | **1102 ×8**, 1105 ×2 |
| P20, residualised K, corrected | 0.215 | 0.208 | 0.280 (0/10) | 0.300 (3/10) | 2 / 3 / 3 / 1 / 1 |
| P20, low/mid L, raw argmax | 0.210 | 0.196 | 0.300 (3/10) | 0.400 (4/10; CI 0.12–0.74) | 1103 ×6, 1105 ×3 |
| P20, low/mid L, corrected | 0.209 | 0.200 | 0.260 (1/10) | 0.400 (4/10) | 1102 4, 1103 4 |

Fitted main effects b̂_d (leave-the-candidate-out): Kodak D0 +6.4e-05, D1 +0.7e-05, D2 −22.5e-05,
D3 −4.2e-05, D4 −2.9e-05; P20 1101 −12.3e-05, **1102 +15.7e-05**, 1103 −4.2e-05, 1104 −16.2e-05,
1105 +8.2e-05.

**Reading, as pre-declared.**
1. Raw argmax is not at chance in *which fingerprint it picks*, and it does not track the
   training body: it selects the fingerprint with the largest main effect for almost every
   adapter — D0 (b̂ = +6.4e-05) for 7 of 10 Kodak adapters, 1102 (b̂ = +15.7e-05) for 8 of 10
   P20 adapters. That is the additive decomposition's prediction realised as an examiner's
   error, and it is the same hazard Klier & Baier report from the other side (false positives
   of generated images against real phones under an unpaired statistic).
2. With the main effect removed, accuracy at G = 250 is 3/10 (Kodak), 3/10 (P20 K) and 4/10
   (P20 L) against an expected 2/10 — all inside the exact binomial interval for chance
   (P(X ≥ 4 | n = 10, p = 0.2) = 0.12). Block accuracy at G = 1 is 0.206–0.216 ≈ 0.2.
   *Under the most favourable closed-set conditions — every candidate fingerprint known, main
   effects calibrated out — a personalised model cannot be attributed to its source body.*
3. The P20 low/mid 4/10 is the same near miss the paper already reports (four of five devices
   positive in that band, sign-flip p = 0.094). It is reported here; it is not significant.
4. Leave-one-device-out residualisation changes nothing (LODO ≡ raw to three decimals),
   reproducing v1's "LODO inert" finding on a different statistic.

Consistent with Entry 03: the achievable true-positive rate at the upper limit was predicted
to be ≤ 0.097 at FPR 0.01 with unlimited images; the observed closed-set accuracy at 250
images is indistinguishable from 0.2 chance.

**Reproduction check pending:** the same computation on locally re-measured PNGs
(`data/gens_kodak`, `data/gens_p20`, 250 per arm, fetching) to confirm the archived CSVs.

---

## Entry 08 · 2026-09-08 · Tier 4 — reference audit completed (35 of 35 checked)

`src/refaudit.py`, `out/refaudit.json`: every `\bibitem` resolved by title on Crossref and
OpenAlex, then year / DOI / pages compared; flags adjudicated by hand where the fuzzy match
was wrong (the v1 REFCHECK warned that Crossref returns unrelated items for venues it indexes
poorly, and it did so twice here).

| key | verdict | action |
|---|---|---|
| `diffusionshield` | **wrong year/venue.** Cited as 2024; the peer-reviewed version is NeurIPS 2023 (arXiv 2306.04642); an ACM SIGKDD Explorations version exists, 2025, DOI 10.1145/3715073.3715079 | cite NeurIPS 2023 |
| `ftshield` | **wrong year/venue.** Cited as 2024; published in ACM SIGKDD Explorations, 2025, DOI 10.1145/3715073.3715080 (arXiv 2310.02401, 2023) | cite SIGKDD Explorations 2025 |
| `carlini` | correct (USENIX Security 2023, pp. 5253–5270); Crossref matched an unrelated item | none |
| `sd3` | correct (ICML 2024, PMLR 235:12606–12633); Crossref matched a 2021 item | none |
| `efron` | correct (JASA 82(397):171–185, 1987); Crossref matched the 1993 reprint | none |
| `lora` | correct (ICLR 2022); OpenAlex carries a 2023 record | none |
| `spoc` | title parse broke on `R\"{o}ssler`; CVPR Workshops 2021 per v1 REFCHECK | verify by hand |
| `sd35`, `flux`, `claude`, `chatgpt` | web resources, not resolvable | none |
| remaining 24 | resolved and consistent | none |

Two corrections to make in the bibliography; the v1 audit stood at 4 of ~23 verified.

Also resolved for the comparison table (`out/comparison_table.md`): DiffusionShield → NeurIPS
2023; FT-Shield → SIGKDD Explorations 2025; Khan, Islam & Hasan → ISC 2025, LNCS 16186,
pp. 279–299, 85.5 % compromise rate against PRNU verifiers including Amped Authenticate.

---

## Entry 09 · 2026-09-08 · Tier 2 registrations — the detector panel and the learned detector

Registered before any generated-image number from either is read (the panel's real-photo
positive controls were computed first, as the standing rules require; its generated-image
rows are being computed but have not been summarised).

**Panel** (`src/t2_panel_par.py`, `out/t2_rows.csv`, `out/t2_summary.json`), on the archive's
own 3,500 base-study generations (base, A_raw s0–s2, B_raw s0–s2) and the 80 held-out real
photographs:
1. `ncc` — the paper's statistic, zero-lag NCC of the wavelet residual against Y·K̂, K̂ from E2.
2. `pce` — classical Goljan PCE via the Binghamton-port `prnu-python` (its own extractor, its
   own K̂ from the same E2 images, neighbourhood radius 2), threshold 60 for the Klier & Baier
   reproduction. `prnu-python` was patched for numpy 2 (`inten_scale` computed in float32; the
   original kept as `functions.py.orig-2018`; semantics unchanged).
3. `lowmid` — the 8×8 block-DCT low/mid signature of notebook 10 (1 ≤ i+j ≤ 5, DC excluded),
   L̂ from E2 with the same cleaning.
4. Noiseprint — **not run**: TensorFlow 1.x only, no PyTorch port; recorded as a scope limit.

Reported per detector: real-photo same-model AUC and paired contrast (gate: AUC ≥ 0.95 or the
detector is reported as failing its positive control), per-arm means against K̂_A and K̂_B, the
paired contrast toward the training device with its SE, and for PCE the fraction of
generations above 60 against each real fingerprint (the false-positive rate an examiner using
the standard threshold would see) and the median PCE.

**Learned detector** (`src/t2_learned.py`, runs when the GPU is free): a four-block CNN on
256×256 patches of the wavelet residual, 16 patches per image, trained to separate real A from
real B on E1 ∪ E2 (220 images per body), 20 epochs, AdamW 1e-3, seed 0; validated on the H
split (gate AUC ≥ 0.95); score per image = patch-mean of logit(A) − logit(B); paired contrast
toward the training device = +score for A-arms, −score for B-arms; base reported as is.
No test-time augmentation, no threshold tuning, no re-training after seeing generations.

**Pre-declared reading for the panel.** Under an unpaired PCE at threshold 60, generations are
expected to exceed the threshold against real fingerprints at a non-trivial rate (Klier &
Baier report 61 % and 100 % for two commercial generators at native resolution; ours are
centre-cropped, which they report removes the effect, so the rate here may be low — either
outcome is informative and both are reported). The paired contrast is expected near zero for
every detector; a detector whose paired contrast toward the training device exceeds three
of its SEs in both arms would be the first positive in the study and would be reported as such.

---

## Entry 10 · 2026-09-08 · Tier 2 result — three detector families, one null; the main effect shows up in PCE units; zero PCE-60 false positives

`src/t2_panel_par.py`, `out/t2_rows.csv` (3,580 rows: 40 real per body, 500 per arm),
`out/t2_summary.json`. Read only after Entry 09 was written.

**Defect D2 (process, fixed).** `pkill -f` from Git Bash did not terminate the serial panel
run on Windows; it and the crashed first parallel launch kept appending to the CSV alongside
the successful run, producing 7,037 rows for 3,580 images. Duplicates were verified
value-identical (max within-key spread: NCC 7e-10, PCE 1.5e-5, low/mid 0) and dropped, keeping
the first; the original file is kept as `t2_rows.csv.with_dupes`. A summary printed to the log
before dedup (never entered here) had understated SEs; the numbers below are from the clean
rows. Stray processes are now killed by PID via PowerShell.

**Positive controls (real held-out photographs, 40 per body).**

| detector | same-model AUC | paired contrast (own − other) | v1 paper |
|---|---|---|---|
| `ncc` (paper's) | 1.0000 | 0.03566 | 1.0000, 0.035670 |
| `pce` (Goljan, prnu-python) | 0.9997 | 367.8 PCE units | — (not in v1) |
| `lowmid` (8×8 DCT band) | 0.8389 | 0.002154 | 0.8433, 0.002120 |

All three pass the ≥ 0.95 gate except low/mid, which v1 already reported at 0.84 and which is
retained as the weaker representation it is.

**Generations, paired contrast toward the training body (mean ± SE over 500 images; t).**

| arm | `ncc` | `pce` (PCE units) | `lowmid` |
|---|---|---|---|
| A_raw_s0 | +6.5e-06 ± 7.2e-05 (t +0.09) | −0.317 ± 0.19 (t −1.65) | −9.2e-05 ± 1.1e-04 (t −0.85) |
| A_raw_s1 | −9.5e-06 ± 7.5e-05 (t −0.13) | −0.329 ± 0.19 (t −1.76) | −6.8e-05 ± 1.1e-04 (t −0.64) |
| A_raw_s2 | −3.0e-06 ± 7.3e-05 (t −0.04) | −0.233 ± 0.16 (t −1.42) | −1.4e-04 ± 1.0e-04 (t −1.38) |
| B_raw_s0 | −4.2e-05 ± 6.9e-05 (t −0.61) | **+0.477 ± 0.17 (t +2.79)** | −1.5e-04 ± 1.1e-04 (t −1.38) |
| B_raw_s1 | +1.3e-05 ± 7.2e-05 (t +0.18) | +0.170 ± 0.17 (t +1.01) | +1.8e-05 ± 1.1e-04 (t +0.16) |
| B_raw_s2 | +5.3e-05 ± 7.2e-05 (t +0.74) | +0.292 ± 0.17 (t +1.75) | +2.1e-04 ± 1.1e-04 (t +1.92) |
| θ_A / θ_B (seed means) | −2.0e-06 / +8.1e-06 | −0.293 / +0.313 | −1.0e-04 / +2.5e-05 |
| **θ_sym** | **+3.1e-06** | **+0.010** | **−3.8e-05** |
| additive part (b_A − b_B)/2 | −5.1e-06 | **−0.303** | −6.3e-05 |
| base, A vs B | 1.63e-04 vs 2.03e-04 | 24.85 vs 25.10 | 8.2e-05 vs 1.2e-04 |

**Reading.**
1. *No detector meets the registered positive criterion* (both arms > 3 SE toward the training
   body). The paper's statistic on the local instrument gives θ_sym = +3.1e-06, the same
   order as v1's k = 12 value (+4.6e-06).
2. *The additive decomposition reappears in PCE units.* Every arm — including the base model,
   which saw no photograph — scores fingerprint B about 0.3 PCE higher than fingerprint A. So
   A-arms read −0.3 "against" their own body and B-arms read +0.3 "toward" theirs, and one
   B-arm reaches t = +2.8. An unpaired reader would call that a detection; the symmetric
   statistic gives +0.010 PCE, i.e. 0.003 % of the 367.8 real-photo contrast. This is v1's
   §V-H main-effect result, now demonstrated on the field's standard statistic.
3. *Klier & Baier reproduction:* under the standard PCE threshold of 60, **0 of 3,500 generations
   exceed it against either real fingerprint** (median PCE 24.4–25.1). Their 61 % / 100 % were
   for commercial generators at native resolution; they report centre-cropping eliminates the
   false positives, and these are 1024² centre-frame generations measured whole, so the zero
   rate is consistent with their remedy rather than in tension with their finding. What the
   PCE *does* show is the generic offset: a median of ~24.5 against fingerprints the generator
   never saw, an order of magnitude above a true null, which is the same nuisance in their
   units.
4. *Low/mid* reproduces v1's picture: the additive term (−6.3e-05) exceeds the interaction
   (−3.8e-05) and no arm is significant.

**Sentence to carry:** *"Under three detector families — the study's zero-lag correlation,
classical peak-to-correlation energy, and a low/mid-band DCT signature — the paired contrast
toward the training body is null in every arm, while the unpaired reading is dominated in all
three by a fingerprint main effect present even in the unpersonalised base model."*

---

## Entry 11 · 2026-09-08 · Tier 5 pre-registration — E-PROMPT, the diverse prompt bank

Registered before any diverse-prompt image is generated. Motivation: every one of the 7,500
v1 base-study generations was produced from the single caption `"a photograph, sks style"`, and
the token `sks` collides with the SKS carbine, so the generated population is one semantic
family (rifles). A reviewer who opens the archive will ask whether the null is a property of
that prompt. v1 declined this arm (run list D1) on the argument that varied prompts pull
generations off the fine-tuned mode; that argument is now the reason to run it.

**Design.** No retraining. The six archived v1 adapters (A_raw s0–s2, B_raw s0–s2, exactly the
`pytorch_lora_weights.safetensors` in `E_INV_P0_v3/adapters/`) and the base model, each
generating 500 images from v1's own pre-declared `DIVERSE_PROMPTS` bank — five captions
("a photograph of a street / a room interior / trees / a building facade / a table with
objects"), prompt j % 5 with seed GEN_SEED_BASE + j — so image j is paired across arms and
across the uniform bank. 28 steps, CFG 4.5, 1024², PNG. `src/t5_prompt.py`.

**Statistic.** The paper's paired contrast against K̂_A^E2 / K̂_B^E2 on the local instrument
(Entry 02), per arm with SE; θ_A, θ_B over the three seeds; θ_sym. Compared directly with the
same six adapters' uniform-bank readings in Entry 10.

**Pre-declared reading.**
- θ_sym within ±2 SE of zero and no arm > 3 SE toward its body → *"the null is not a property
  of the prompt; it holds across five semantically unrelated captions."*
- A positive that appears only under diverse prompts would be reported as such and would
  reopen the question of whether the uniform-prompt null reflects a collapsed mode.
- Whether the diverse generations are visibly less "adapted" (adaptation strength) is a
  secondary observation, reported descriptively from the mean absolute difference to the base
  arm at matched seeds, not tested.

Cost: 7 × 500 generations, ~25 min each on the L40S; runs after Tier 1 releases the GPU.

---

## Entry 12 · 2026-09-08 · Correction to Entry 08 (reference audit): the two "wrong year" flags were artefacts

Entry 08 recorded DiffusionShield and FT-Shield as cited with the wrong year (2024) and
prescribed corrections. On inspection of the entries themselves, both cite *ACM SIGKDD
Explorations Newsletter*, vol. 26, no. 2, December 2024, with DOIs 10.1145/3715073.3715079 and
.3715080 — the issue is dated December 2024; Crossref and OpenAlex carry the DOI registration
year, 2025. IEEE style cites the issue date. **The submitted entries were correct and are
restored verbatim.** The DiffusionShield NeurIPS 2023 version exists but the SIGKDD
Explorations version is the one the entry cites, consistently. Entry 08's action column for
these two rows is withdrawn; the audit outcome is 35/35 consistent with zero corrections
required (spoc still to be checked by hand for the same reason — a parse failure, not a
discrepancy). Registered in the register below.

Manuscript working copy `paper/einv_v2.tex` after this entry: 0 citation inversions, all 7
figures and 5 tables referenced, hardware/software subsection and device table added, the
"none of these things" sentence replaced. Not yet compiled — no LaTeX on this machine.

---

## Entry 13 · 2026-09-08 · Tier 3 reproduction check — the archived per-row measurements are exact

`src/t3_repro.py`, `out/t3_repro.json`. The first 250 generations of every Kodak and P20 adapter
(20 arms, 5,000 PNGs) were fetched from the public archive and re-measured on this machine
with the v1 core against the archive's own E2 fingerprints (`K_D*_E2.npy`; residualised
`KR_*_E2.npy` for P20), then compared row by row with `c3_measure_raw.csv` and `d5_measure.csv`.

| group | rows compared | images | max |Δρ| | median |Δρ| | Pearson | sd of archived ρ |
|---|---|---|---|---|---|---|
| Kodak M1063 | 12,500 | 2,500 | 4.6e-08 | 3.2e-09 | 0.99999999999878 | 1.33e-03 |
| Huawei P20 | 12,500 | 2,500 | 2.7e-08 | 2.5e-09 | 0.99999999999978 | 1.11e-03 |

The largest discrepancy is five orders of magnitude below the spread of the statistic — float32
accumulation order, nothing else. Consequences: (i) the Tier 3 attribution in Entry 07, which
was computed from the archived CSVs, stands on measurements that reproduce from the images;
(ii) the archive's per-row records — the reviewers' reproducibility concern — are shown to be
faithful for 25,000 of the 50,000 Kodak/P20 rows and, by the same instrument agreement in
Entry 02, for the primary pair; (iii) the Colab-versus-local decode difference noted in Entry 05
does not affect generated PNGs (they are lossless), only re-decoded JPEG training images.

Tier 3 is complete: Entries 03 (power translation), 06–07 (attribution), 13 (reproduction).

---

## Entry 14 · 2026-09-08 · Toolchain for the rewrite, and a defect in the submitted PDF

**Toolchain.** Tectonic 0.15 (XeTeX engine, self-fetching packages) installed at
`D:\A4\tools\tectonic\`. The manuscript working copy `paper/einv_v2.tex` compiles to 19 pages
with zero errors after two local patches to `paper/ieeeaccess.cls`, both guarded and commented:
(i) the PANTONE spot-colour branch needs pdfTeX primitives, so under XeTeX a CMYK stand-in for
PANTONE 3015 C is used; (ii) see below. Figure 1 is now TikZ (`paper/fig1_pipeline_tikz.tex`,
newtx Times to match the body; 7.16 in wide); Figures 2–7 are regenerated by
`src/make_figures_v2.py` — the repository's `make_figures.py` with the STIX/Times face, the
Figure 1 palette, and Figure 1 removed — reading the same frozen ledger.

**Submission defect S1 (v1, visible in the filed PDF).** `ieeeaccess.cls`'s `\@makecaption`
sets a wide-figure caption inside `\begin{tabular}{@{}l@{}}…\end{tabular}`, a cell that cannot
break lines. Every full-width figure caption longer than one line was therefore cut off at the
right margin in the compiled submission — Figure 1's caption ends mid-word on page 5 of
`einv_access.pdf`. Neither reviewer mentioned it; it is nonetheless a reader-visible defect of
the filed PDF. Patched locally with a `\parbox[t]{\textwidth}`; the official IEEE class used at
production would need the same check.

**Figure QA (same day, later).** All seven figure pages of `build/einv_v2.pdf` were rendered
and inspected. Figure 1 was redrawn once more: fixed-height boxes with top-pinned content (the
first draft let a four-line box sit taller than its neighbours), hyphenation disabled inside
boxes, arrow labels ("measured here"; "same comparator at every endpoint"), and the drawing
sized to exactly 7.16 in so it is placed at 1:1 and no font is scaled. Figures 2-7: fonts, tick
labels, legends and captions legible; no overlaps (checker output "OK" for each); the wrapped
wide caption of Figure 1 now fills the text width on page 5. Manuscript: 19 pages, 0 errors,
`build/einv_v2.pdf`; copy at `paper/einv_v2_draft_2026-09-08.pdf`. This is a layout draft: the
v2 results sections (Tier 1 ladder, E-PROMPT, learned detector, comparison table) are not yet
written into it because their measurements are still running.

---

## Entry 15 · 2026-09-08 · Manuscript v2: the results-independent text is written and compiled

`paper/einv_v2.tex`, compiled to `paper/build/einv_v2.pdf` (snapshot
`paper/einv_v2_draft_2026-09-08b.pdf`), 23 pages, 0 errors. What went in, and from which entry:

| manuscript location | content | source |
|---|---|---|
| Related Work, new `table*` `tab:priorart` (first float, p. 5) | ten-row comparison: signal, designed/optimised, system and adaptation, unit of attribution, reported outcome; Khan et al. row; this work last | `out/comparison_table.md`, Entries 02, 08, 12 |
| Related Work text | Khan et al. sentence (85.5 %) beside SpoC; Klier & Baier cross-reference to the panel; the "four differences" paragraph kept and pointed at the ladder | Entry 02 |
| Bibliography | `khan2025`: S. R. Khan, T. Islam, R. Hasan, ISC 2025, LNCS 16186, pp. 279–299, doi 10.1007/978-3-032-08124-7_17 (authors and pages from OpenAlex, 2026-09-08) | Entry 08 |
| Methods, new §IV-J "Detector Panel, Closed-Set Attribution, Designed-Mark Ladder, and Prompt Diversity" | the four registrations, instrument agreement (κ 0.00745 vs 0.0073; 0.3619/0.2459) | Entries 01, 02, 06, 09, 11 |
| Methods, Deviations (now six) | "Truncated storage of the injected training arms": sign map at α ≤ 1.6 (50.1 % of pixels at −1), 0.2–0.5 % amplitude-bearing at 3, graded from 6; α_eff scale measured on float images | Entry 05 |
| Methods, Hardware | L40S / torch 2.11 / diffusers 0.39 / peft 0.20; 39–53 min per adapter (from `logs/t1_train.log`) | Entry 04 + logs |
| Results V-B (objective) | restated on stored images: α = 1 and 1.6 identical, so +1.7e-5 is the run-to-run floor | Entry 05 |
| Results V-D (amplification) | null re-read as three stored perturbations, not a dose–response below nominal 6 | Entry 05 |
| Results, new V-N "Three Detector Families and the Standard PCE Threshold" + `tab:panel` | ncc/pce/lowmid positive controls, θ_sym, additive part, |t|max, 0 of 3,500 above PCE 60; learned detector marked pending | Entry 10 |
| Results, new V-O "What the Bound Means for an Examiner" + `tab:attrib` | power translation at σ_μ measured and σ_μ = 0 (TPR 0.12 at G = 500; 0.5 near 3,000; 0.9 near 5,000, M = 2); closed-set accuracy table; raw-argmax main-effect finding | Entries 03, 07; `out/t3_power.json` (extended today with `rows_sigma_mu_zero`) |
| Results, new V-P "A Designed Mark on the Same Ladder" | R_mark 0.0005 / 0.234 / 0.661 / 0.504; result marked pending | Entry 04 |
| Results, new V-Q "Prompt Diversity" | question only; pending | Entry 11 |
| Discussion | PCE main effect and raw-argmax sentences; operational reading of the bound | Entries 03, 07, 10 |
| Limitations | "Stored injected arms"; "The power translation" (σ_μ measured on the pooled surface, reported as a range) | Entries 03, 05 |
| Reproducibility | archive re-measurement: max |Δρ| 4.6e-08 / 2.7e-08 over 25,000 rows; decode difference 28 % of pixels, ±4 | Entries 05, 13 |
| Abstract, Introduction (findings + contribution 5), Conclusion | one added passage each | as above |

**Addition to `out/t3_power.json`.** `src/t3_power.py` now also reports the σ_μ = 0 bracket
(only image noise, averaging out with G) on a finer image grid; at the upper limit and
M = 2: TPR 0.120 at G = 500, 0.5 first reached at G = 3,000, 0.9 at 5,000; M = 5: 0.049,
3,000, 10,000; M = 50: 0.009, 5,000, 10,000. The σ_μ-as-measured rows are unchanged.

**Visible placeholders in the draft** (bold, bracketed, three): learned detector, ladder
result, prompt-diversity result. They are removed only by their entries.

**Layout.** `tab:priorart` and `tab:devices` column widths reduced so neither `table*`
overfills the text width; `tab:panel` transposed (detectors as columns) and `tab:attrib`
set at `\scriptsize` so both fit one column. Remaining overfull boxes are the class's own
header/footer warnings present in the submitted build.

---

## Entry 16 · 2026-09-08 · Tier 2 result — the learned detector: positive control passes, no transfer under the registered criterion, and the largest main effect in the study

`src/t2_learned.py` (run 10:35–10:55 alongside Tier 1 generation; `logs/t2_learned_early.log`),
`out/t2_learned.json`; adapter-level summary `src/t2_learned_summary.py` →
`out/t2_learned_summary.json`. Exactly the Entry 09 registration: four-block CNN on 256² residual
patches, 16 per image, trained on E1 ∪ E2 (440 real images, 7,040 patches), 20 epochs, AdamW
1e-3, seed 0; score per image = patch-mean of logit(A) − logit(B); no test-time adaptation.

**Positive control.** Held-out real photographs: AUC 0.9938 (A vs B, 40 each); mean score
+1.97 (A) vs −6.46 (B); real paired contrast R = 8.43 score units. Gate ≥ 0.95 passed.

**Generations, mean score (A minus B) ± SE over 500 images.**

| arm | score | toward own body |
|---|---|---|
| base (no adapter) | −5.44 ± 0.13 | — |
| A_raw s0 / s1 / s2 | −4.59 ± 0.16 / −3.84 ± 0.17 / −5.39 ± 0.14 | −4.59 / −3.84 / −5.39 |
| B_raw s0 / s1 / s2 | −5.50 ± 0.15 / −4.84 ± 0.16 / −5.59 ± 0.16 | +5.50 / +4.84 / +5.59 |

**Decomposition (adapters as unit, three per arm).**
- Additive part (b_A − b_B)/2 = **−4.96 score units = 59 % of R.** Every arm, the base model
  included, reads as body *B* to the learned detector at more than half the strength of a real
  B photograph. This is the fingerprint main effect of v1 §V-I in its most extreme form: a
  learned detector that is perfect on real images attributes *every* generation — including
  those from a model that never saw a photograph — to one body.
- θ_A = −4.61, θ_B = +5.31; **θ_sym = +0.35**, adapter-level SE 0.25 (Welch df 3.0), t = 1.39,
  one-sided p = 0.13; image-level SE 0.064 for reference (not the unit). λ_sym = 4.2 % of R;
  plug-in one-sided 99 % upper limit **λ_U = 17.6 %** at n = 3 per arm.
- Shift relative to the base model: A arms +0.84, B arms +0.14, all six +0.49 — adaptation
  of either body moves the score toward A; the A-arm excess over the B-arm shift (0.70) is the
  interaction, and it rests on three adapters per arm with A-arm spread 0.77.
- **Registered positive criterion** (both arms > 3 SE toward own body): **not met** — the A arms
  are far on the wrong side, exactly as the main effect dictates.

**Reading.** (1) The learned detector is the fourth detector family and, like the other three,
does not detect transfer under its registered criterion. (2) Its point estimate of the symmetric
interaction is positive (4 % of its own real contrast) but is one-and-a-half adapter-level
standard errors from zero, and the limit it can set at three adapters per arm, 17.6 %, is two
orders of magnitude looser than the NCC limit at twelve — the learned score is noisier across
adapters than across images by a factor of four (0.25 vs 0.064). It is not evidence of transfer
and it is not a tight null. (3) The 59 % additive term is the single most forensically
important number the panel produced: a learned same-model detector, deployed unpaired, would
attribute generations of any provenance to body B with high confidence.

**What this does not settle:** whether θ_sym > 0 at twelve adapters per arm. That is a
registered extension (Entry 17), not a re-analysis.

---

## Entry 17 · 2026-09-08 · Tier 2 registration — extending the learned detector to twelve adapters per arm

Registered before any seed-3–11 generation is scored by the learned detector.

**Why.** Entry 16 leaves the learned detector's symmetric interaction at +0.35 ± 0.25 with n = 3
per arm; the primary design's declared unit count is twelve per arm (v1 seeds 0–11, archived as
`E_INV_P0_v3` s0–s2, `E_SEEDEXT` s3–s5, `E_SEEDEXT2` s6–s11). Extending the same detector to the
same twelve adapters is the study's own k = 6 → 12 move applied to a fourth detector family.

**Design.** Fetch the first 250 generations of each of the 18 remaining arms (A_raw s3–s11,
B_raw s3–s11) from the public archive (`data/gens_ext/`). Re-train the Entry 09 network once
under the same seed, **save its weights** (`out/t2_learned_net.pt`), and score all 24 arms from
that single model, using the first 250 images of every arm (s0–s2 included) so the unit is
uniform; the 500-image s0–s2 readings of Entry 16 are re-reported beside the 250-image ones from
the re-trained model, and any difference is GPU nondeterminism to be stated, not hidden.
Script: `src/t2_learned_ext.py` → `out/t2_learned_ext.json`.

**Statistics, unchanged from Entry 09 plus the primary construction.** θ_A, θ_B over twelve
adapters each; θ_sym; additive part; adapter-level SE and Welch t; the exact sign-flip test of
the paper's intersection–union procedure at k = 12 (floor 1/4096) applied to the twelve
per-adapter own-body contrasts after removing the additive part; plug-in one-sided 99 %
λ_U as in Entry 16; positive control AUC of the re-trained model on the H split.

**Pre-declared reading.**
- Intersection–union rejects at α = 0.01 with θ_sym > 0 → the first positive in the study, to
  be reported as such, with the caveat that the detector's real-image contrast is a
  logit-difference and λ is relative to that scale.
- Does not reject → report λ_U at n = 12 as the learned-detector bound beside the NCC bound.
- Whatever the outcome, the additive part is reported at twelve adapters.

Nothing in this entry changes once scoring starts.

**Addition to Entry 17, made before any seed-3–11 image was scored (11:10):** besides the
intersection–union sign-flip, an exact label-permutation test — enumerate all C(24, 12) =
2,704,156 assignments of the 24 adapter scores to the A and B labels and report the one-sided
p of mean_A − mean_B. Under no device-specific transfer the A/B labels are exchangeable given the
shared additive part, so this is the cleanest test of the interaction on the learned score.
Script `src/t2_learned_ext.py`; fetch of the 18 arms in `src/fetch_ext.sh` → `data/gens_ext/`.

---

## Entry 18 · 2026-09-08 · Tier 2 result — the learned detector at twelve adapters per arm: a small device-associated interaction that the template detectors do not see

`src/t2_learned_ext.py` (11:30–12:05), `out/t2_learned_ext.json`, weights `out/t2_learned_net.pt`;
18 arms fetched by `src/fetch_ext.sh` (`logs/fetch_ext.log`: 18 × 250, 0 failures). First 250
generations of every arm; the Entry 17 registration, including its permutation-test addition,
was followed as written.

**Defect D5 (process, disclosed).** Re-training the network under the same seed did not reproduce
the Entry 16 network: cuDNN nondeterminism gives a different model with a different logit scale
(held-out AUC 0.9894 vs 0.9938; R = 8.10 vs 8.43 score units; base-model score −13.10 vs −5.44).
Entry 16's network was never saved. The saved weights `t2_learned_net.pt` are therefore the
detector of record from here on; Entry 16's numbers stand as the first, unreproducible draw of
the same procedure. The per-seed *ordering* agrees between the two models on s0–s2 (A above B
at every seed; 500-image means from the saved model: A −12.57 / −12.06 / −13.80, B −13.60 /
−13.03 / −14.29), and the interaction estimates agree (θ_sym +0.35 there, +0.38 here).
Consequence for the write-up: learned-detector results are reported from the saved model only,
with the nondeterminism stated; `t2_learned.py` should save weights in any re-run.

**Positive control (saved model).** Held-out real photographs: AUC 0.9894; R = 8.10.

**Per-adapter mean score (A minus B logit), 250 generations each.**

| seed | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A arm | −12.60 | −12.07 | −13.71 | −12.95 | −12.64 | −12.16 | −13.42 | −13.27 | −12.54 | −12.05 | −12.84 | −12.66 | **−12.74** |
| B arm | −13.54 | −12.96 | −14.00 | −13.59 | −13.35 | −13.94 | −13.15 | −12.90 | −13.99 | −13.44 | −12.88 | −14.33 | **−13.50** |
| A − B | +0.94 | +0.89 | +0.29 | +0.64 | +0.71 | +1.78 | −0.27 | −0.37 | +1.45 | +1.39 | +0.04 | +1.67 | +0.76 |

Base model (no adapter): −13.10 ± 0.23. Image-level SE per arm 0.26–0.35.

**Statistics (adapters as unit, twelve per arm).**
- Additive part (b_A − b_B)/2 = −13.12 = **162 % of R**: every generation, base included,
  reads as body B *beyond* the range of real B photographs. The detector is far out of
  distribution on generations, and an unpaired reading is meaningless.
- θ_A = −12.74, θ_B = +13.50; **θ_sym = +0.381**; adapter-level SE 0.103 (Welch df ≈ 21);
  **t = 3.69, one-sided p = 0.0006.**
- **Exact label permutation** (C(24,12) = 2,704,156 assignments): one-sided **p = 0.0008**
  (2,180 of 2.7 M assignments reach the observed mean_A − mean_B = +0.76).
- **Registered intersection–union sign-flip** on the additive-corrected own-body contrasts:
  p_A = 65/4096 = 0.0159, p_B = 41/4096 = 0.0100. **Does not reject at α = 0.01** (p_B misses
  by one assignment; p_A by 24). Both arms are positive at α = 0.02.
- Relative to base: A arms +0.36, B arms −0.40 — the two arms move *symmetrically* away from
  the base model, each toward its own body. (In Entry 16's model both moved up; the symmetric
  pattern here is the cleaner signature of an interaction.)
- λ_sym = **4.7 %** of the detector's real contrast; plug-in one-sided 99 % **λ_U = 7.9 %**.
- Seeds 6, 7 and 10 are the three adapters with A − B ≤ 0; nine of twelve are positive.

**Reading, against the pre-declared wording.**
1. Under the registered primary criterion (intersection–union at 0.01) the result is *not* a
   rejection, and it is reported that way. Under the registered secondary tests it is: Welch
   t p = 0.0006 and the exact label permutation p = 0.0008. The honest summary is that a
   learned residual detector resolves a small, consistent, device-associated interaction at
   twelve adapters per arm — about 4.7 % of what it measures between real photographs of the
   two bodies, bounded above by 7.9 % — where the three template detectors (NCC, PCE, low/mid)
   bound the PRNU-specific contrast below 0.15 %.
2. **What it is not yet shown to be.** The network was trained to separate body A from body B
   on wavelet residuals of real Dresden photographs; the two bodies photographed the same
   motifs, so scene content is not the discriminant, but the residual carries every per-body
   property — PRNU, dark-signal non-uniformity, defective pixels, noise level and in-camera
   processing differences — not PRNU alone. The experiment shows that *something body-specific
   in the residual* survives personalisation and generation at a few per cent of its real
   strength; it does not show that this something is the PRNU template, and the template
   detectors say that the spatially registered PRNU component of it is < 0.15 %.
3. This is the result Reviewer 2's "learned detectors could see more" pointed at, and it is
   reported as a finding, not a failure: the study's claim narrows from "device identity does
   not transfer" to "the PRNU template does not transfer at a detectable level; a learned
   residual detector resolves a small device-associated residue whose composition is open."

**Registered follow-ups (before computing either).**
- F1 · *Is the learned interaction PRNU-driven?* Adapter-level correlation (Pearson and
  Spearman, n = 24) between the learned own-body contrast (this entry) and the archived NCC
  own-body paired contrast of the same adapter (v1 per-row CSVs, 250-image means, seeds 0–11).
  If the learned signal is the PRNU template, the two rank the adapters alike (ρ > 0); if it is
  another residual property, no correlation is expected. Either outcome is reported.
- F2 · *Does it persist across prompts?* The saved model applied to the E-PROMPT generations
  (Entry 11; six adapters s0–s2) once they exist: paired contrast toward own body, θ_sym.
- F3 · *Descriptive only:* the saved model applied to the six Tier 1 ladder arms (all trained
  on body A images): reported as the A-side reading, no test.

---

## Entry 19 · 2026-09-08 · F1 result — the learned interaction does not demonstrably track the PRNU contrast (inconclusive at n = 24)

`src/t2_learned_f1.py`, `out/t2_learned_f1.json`; archived per-row CSVs fetched to
`data/csv_primary/` (`s5_measure`, `b3_measure`, `sx2_measure`; ledger 500-image per-adapter
values from `FINAL_LEDGER.json` primary.per_adapter_A/B, seeds 0–11).

| x (learned own-body contrast) | y (NCC own-body contrast) | Pearson r (one-sided p) | Spearman ρ (p) |
|---|---|---|---|
| additive part removed per arm | same first 250 generations | +0.31 (0.07) | +0.32 (0.06) |
| additive part removed per arm | ledger, 500 generations | +0.02 (0.46) | −0.03 (0.56) |
| within A arms only (n = 12) | 250 / 500 | +0.41 / +0.29 | +0.45 / +0.16 |
| within B arms only (n = 12) | 250 / 500 | +0.38 / −0.34 | +0.33 / −0.27 |

(The "raw" learned contrast, which still contains each arm's additive part, is bimodal by arm
and its correlations are meaningless; recorded in the JSON, not read.)

**Reading.** A weak positive association appears only when both measures are computed on the
identical 250 images, and vanishes against the 500-image values. At 250 images the NCC
per-adapter contrast has an image-noise SE (~6.6e-05) larger than the adapter-level spread
(~4.5e-05), so shared image noise between the two measures can produce a correlation of this
size without any adapter-level link. **No evidence that the learned interaction is the PRNU
template; no evidence that it is not.** Carried into the paper as "inconclusive at n = 24".
F2 (prompt bank) and F3 (ladder arms) remain queued on the E-PROMPT and Tier 1 outputs.

**Manuscript (same day, later).** Entries 16–19 written into `paper/einv_v2.tex`: §V-N learned
detector paragraphs (both draws disclosed, D5), `tab:panel` fourth column (learned CNN at
twelve adapters), abstract / introduction findings / contribution 5 / conclusion / limitations
("Composition of the learned residue") / prior-art table last row. Compiled: 24 pages, 0 errors;
snapshot `paper/einv_v2_draft_2026-09-08c.pdf`. Remaining placeholders: ladder result (§V-P),
prompt diversity (§V-Q).

---

## Entry 20 · 2026-09-08 · Tier 1 result — the designed-mark ladder: detected at 12×, cluster-significant but not decoy-separated at 3×, transmission of order 10⁻³, and the natural fingerprint sits on the same line

`src/t1_measure.py` (17:13–17:34, 4,500 images, 6 workers) → `out/t1/measure_rows.csv`,
`out/t1/summary.json` (registered quantities); `src/t1_derived.py` → `out/t1/summary_derived.json`
(secondary quantities, labelled). Generation 10:03–17:13 (six arms × 500, 70 min per arm on the
L40S). Training 05:56–10:02. Everything as registered in Entry 01 / amended in Entry 04.

**Per-arm results** (ρ_mult against Y·M, Y·M′, Y·K̂; 500 images; SE over images).

| arm | nominal α | R_mark (stored) | R_mark / R_real | contrast M − M′ | t (images) | rank of true M among 31 | λ_mark raw | λ_mark offset-corr. | natural K̂_A − K̂_B |
|---|---|---|---|---|---|---|---|---|---|
| mark_rand_a1_s0 | 1 | 0.0005 | 0.01 | +7.0e-05 ± 5.9e-05 | 1.2 | 12 | — (null) | — | +2.8e-05 |
| mark_rand_a3_s0 | 3 | 0.234 | 6.6 | +2.14e-04 ± 6.0e-05 | 3.6 | 7 | 0.091 % | 0.057 % | +1.24e-04 |
| mark_rand_a3_s1 | 3 | 0.234 | 6.6 | +1.70e-04 ± 5.8e-05 | 2.9 | 2 | 0.073 % | 0.039 % | +4.0e-05 |
| mark_rand_a3_s2 | 3 | 0.234 | 6.6 | +2.04e-04 ± 6.1e-05 | 3.4 | 2 | 0.087 % | 0.053 % | +8.1e-05 |
| mark_rand_a12_s0 | 12 | 0.661 | 18.5 | +3.59e-04 ± 6.1e-05 | 5.9 | **1** | 0.054 % | 0.042 % | +9.3e-05 |
| mark_lowmid_a12_s0 (M_L stat.) | 12 | 0.504 | 14.1 | +5.23e-04 ± 6.6e-05 | 7.9 | **1** | 0.104 % | 0.083 % | +1.44e-04 |
| base (never injected) | — | — | — | +8.8e-05 ± 5.1e-05 | 1.7 | 4 | | | −4.0e-05 |
| A_raw_s0 (never injected) | — | — | — | +7.9e-05 ± 5.4e-05 | 1.4 | 8 | | | +0.7e-05 |
| B_raw_s0 (never injected) | — | — | — | +7.3e-05 ± 5.6e-05 | 1.3 | 16 | | | +4.2e-05 |

**Offset.** The M − M′ contrast is +8.0e-05 (SE over the three never-injected arms 0.4e-05;
pooled image SE 3.1e-05) in arms that never saw M: M′ happens to correlate negatively with
generations in every arm (ρ(→M′) −2e-05 to −1.5e-04). This is a comparator main effect of the
same kind as the fingerprint main effects of v1 §V-I; it is what the never-injected arms exist
to measure, and the offset-corrected λ (secondary) subtracts it.

**Registered cluster test at α = 3** (three seeds): mean +1.96e-04, sd 2.3e-05, **t = 15.0**
against zero (crit 9.92 at df 2, α = 0.01; p = 0.0022); against the never-injected offset
t = 8.9, p = 0.0062 — also < 0.01. λ_mark = 0.084 % (U 0.139 %); offset-corrected 0.050 %
(U 0.105 %).

**Registered decoy test.** At α = 3 the true M ranks 7, 2, 2 among 31 — near the top in every
seed but never first; the spectrum-preserving rolled copies beat it. At α = 12 the true field
ranks **first of 31 in both arms** (P = 1/31 each under exchangeability), and in the
never-injected arms it ranks 4, 8, 16 (chance).

**Verdict against the pre-declared outcomes (Entry 01).**
- Outcome (a) at α = 3: cluster test passes, decoy criterion fails, never-injected criterion
  passes → **not (a)**.
- α = 12 (single adapters, both fields): detected, decoy rank 1, never-injected clean.
- → **the mixed case**: *"A designed, exactly known fixed pattern at twelve times one natural
  fingerprint's energy is detected in generated images (rank 1 of 31 among spectrum-preserving
  decoys, both fields; image-level t 5.9 and 7.9) and transmits at 0.04–0.08 % of its stored
  input contrast; at three times it is cluster-significant (t = 15) but not separated from
  its decoys (ranks 7, 2, 2). The channel is amplitude-gated at this detector's floor."*

**The reading that matters (secondary, not pre-declared, labelled so).** Output contrast per
unit of stored input contrast is **0.04–0.08 %** for every detected mark, random or band-limited,
at 3× or 12×. The natural fingerprint's plug-in *point* estimate on the same statistic is
λ̂ = 0.013 % with upper limit 0.15 % (ledger). One transmission coefficient of order 10⁻³
therefore describes the designed marks *and* brackets the natural fingerprint: at 1× natural
energy a 0.05 % transmission predicts an output contrast of ~1.8e-05, below the study's
per-adapter noise (U_device 5.4e-05) and inside the natural CI. The natural null is not a
special property of PRNU; it is what a 10⁻³ channel does to a signal that starts at 1× and
cannot be amplified past ~4× effective. This is the quantitative content of the "four
differences": the active-marking literature's success rests on amplitudes and optimisation that
lift a designed mark off this floor.

**Observation, not tested (registered follow-up F4 below).** The natural paired contrast
K̂_A − K̂_B on the six mark arms — all trained on body A images with the local decode — is
positive in all six: mean +8.5e-05, SE over adapters 1.9e-05, t = 4.6; the five marked arms
mean +9.6e-05, the effectively unmarked α = 1 arm +2.8e-05. Under v1's A-arm distribution
(mean −1.0e-05, sd 4.8e-05, k = 12, Colab decode) the chance that six draws are all positive is
0.006. Two candidate explanations, confounded here: (i) the local JPEG decode (Entry 05) leaves
training images whose PRNU the adapter picks up more than the Colab-decoded ones did; (ii) the
mark itself, by adding a strong fixed-coordinate component to every training residual, makes
the adapter learn fixed-coordinate texture and carries the natural fingerprint with it. Either
would be a positive result for the primary question and neither is claimed on six adapters
without an unmarked local-decode control.

**Registered follow-up F4 (before any run).** Three rank-16 adapters on device A's local-decode
training crops with *no* mark (`nomark_s0/s1/s2`; identical protocol, seeds 0–2), 500
generations each from the primary seed bank; statistic: natural paired contrast toward K̂_A per
adapter, mean over three, compared (i) with the five marked arms (two-sample t, one-sided H1:
marked > unmarked) and (ii) with v1's twelve A arms (one-sided t against −1.0e-05, sd 4.8e-05).
Pre-declared reading: nomark ≈ v1 and marked > nomark → mechanism (ii), report "a strong
fixed-coordinate mark in the training images carries the natural fingerprint through at a level
the unmarked pipeline does not"; nomark ≈ marked > v1 → mechanism (i), the decode, report as a
v1 limitation and a v2 finding to be replicated; nomark ≈ v1 ≈ marked within noise → the six
positives were chance. GPU cost ≈ 2 h training + 3.5 h generation; queued after E-PROMPT.
F3 (learned detector on the ladder arms) partial: a1 −12.58 (as the primary A arms), a3 arms
−14.3 / −15.2 / −15.2 — the mark moves the learned score *toward B*; full set when the run ends.

---

## Entry 21 · 2026-09-08 · F3 result — the learned detector on the ladder arms (descriptive)

`src/t2_learned_f3.py ladder` → `out/t2_learned_f3.json` (saved model of Entry 18; 500 images per arm).

| arm | learned score (A − B logit) |
|---|---|
| mark_rand_a1_s0 (effectively unmarked, local decode) | −12.58 ± 0.20 |
| mark_rand_a3_s0 / s1 / s2 | −14.27 ± 0.19 / −15.16 ± 0.19 / −15.23 ± 0.19 |
| mark_rand_a12_s0 | −17.99 ± 0.22 |
| mark_lowmid_a12_s0 | −14.66 ± 0.15 |
| reference: primary A arms (12, Entry 18) / base | −12.74 / −13.10 |

The unmarked local-decode adapter reads exactly like the primary A adapters (so the decode does
not move the learned score). Every mark pushes the score *toward B*, monotonically in the
random field's amplitude (−1.6 at 3×, −5.3 at 12× relative to unmarked) — several times the
device interaction of Entry 18 (+0.38). A fixed-coordinate pattern belonging to neither body
moves the learned reading more than the bodies do: the detector responds to residual texture,
not to the PRNU template alone. Written into §V-P as one paragraph. No test; none registered.

**Manuscript.** Entries 20–21 written: §V-P (three findings + the recorded observation +
F3), `tab:ladder`, Figure 8 (`src/fig8_ladder.py` → `paper/fig8_ladder.pdf`), abstract /
introduction findings / Discussion VI-B (transmission paragraph) / conclusion / prior-art row.
Compiled 26 pages, 0 errors; snapshot `paper/einv_v2_draft_2026-09-08d.pdf`. One placeholder
left: §V-Q prompt diversity.

---

## Entry 22 · 2026-09-09 · Tier 5 result — E-PROMPT: the null is not a property of the caption; the fingerprint-B main effect grows under diverse prompts

`src/t5_prompt.py` generate (17:14–01:24, 7 × 500 at 1024², five captions, seed j → caption j mod 5)
→ `out/t5/gens/`; measure → `out/t5/rows.csv`, `out/t5/summary.json`; `src/t5_derived.py` →
`out/t5/summary_derived.json`. Exactly the Entry 11 registration.

**Own-body paired contrast per adapter (500 images; image-level t).**

| arm | diverse bank | t | uniform bank, same adapter (Entry 10) |
|---|---|---|---|
| A_raw s0 / s1 / s2 | −9.2e-05 / −6.8e-05 / −10.0e-05 | −1.2 / −0.9 / −1.3 | +0.7e-05 / −1.0e-05 / −0.3e-05 |
| B_raw s0 / s1 / s2 | +5.6e-05 / +13.3e-05 / +19.8e-05 | +0.7 / +1.8 / +2.6 | −4.2e-05 / +1.3e-05 / +5.3e-05 |
| θ_A / θ_B | −8.7e-05 / +12.9e-05 | | −0.2e-05 / +0.8e-05 |
| **θ_sym** | **+2.1e-05**, adapter-level SE 2.1e-05, t = 1.0 (Welch df 2.2) | | +0.3e-05 |
| additive part (b_A − b_B)/2 | **−10.8e-05** | | −0.5e-05 |
| base model K̂_B − K̂_A | **+13.6e-05** | | +4.0e-05 |

λ_sym = 0.06 % of R_real; n = 3 plug-in limit 0.56 % (as loose as v1's matched-n limit, 0.70 %).
Adaptation strength (registered descriptive): mean |arm − base| at matched seeds 28–33 gray levels
under the diverse bank vs 38–45 under the uniform bank — the adapters act on the diverse
generations at roughly 70 % of their uniform-bank effect, so the bank is not "off the adapted mode".

**Pre-declared reading holds:** θ_sym within ±2 SE of zero (t = 1.0) and no arm beyond 3 SE
toward its own body (largest +2.6, B_s2). *"The null is not a property of the prompt; it holds
across five semantically unrelated captions."*

**What changed, and it is the additive term.** Under diverse prompts every generation — the base
model's included — correlates with K̂_B more than with K̂_A by 1.4e-04, three times the offset
under the single caption (0.4e-04). A arms therefore read −0.9e-04 "against" their body and B arms
+1.3e-04 "toward" theirs, and B_s2 reaches t = 2.6 on an unpaired reading. The symmetric statistic
removes it and lands at +2.1e-05. This is the third time in v2 the same hazard appears in a new
place (PCE units, the learned detector, and now prompt content): the main effect is content- and
detector-dependent, the paired interaction is not. Written into §V-Q.

---

## Entry 23 · 2026-09-09 · F2 result — the learned detector on the diverse-prompt generations (three adapters per arm)

`src/t2_learned_f3.py prompt` (saved model of Entry 18) → `out/t2_learned_f2.json`; 500 images per arm.

| arm | diverse bank | uniform bank, same adapters, same model (Entry 18, 500-image means) |
|---|---|---|
| base | −12.12 ± 0.20 | −13.10 |
| A_raw s0 / s1 / s2 | −10.80 / −10.15 / −10.83 | −12.57 / −12.06 / −13.80 |
| B_raw s0 / s1 / s2 | −10.98 / −10.67 / −11.63 | −13.60 / −13.03 / −14.29 |
| mean A − mean B | **+0.50** | +0.83 |
| θ_sym (adapter-level SE) | **+0.25 (0.18)**, t = 1.4 | +0.42 at these three seeds; +0.38 (0.10) at twelve |
| additive part | −10.84 | −13.12 (twelve adapters) |
| shift vs base: A / B | +1.53 / +1.03 | +0.36 / −0.40 (twelve adapters) |

**Reading (no threshold was pre-declared for F2; descriptive).** The sign of the learned
interaction persists under five semantically unrelated captions — the A adapters' generations
read more A-like than the B adapters' in all three seed pairs — at roughly two-thirds of the
uniform-bank magnitude for the same six adapters (0.50 vs 0.83 in A − B), and it is not resolved
at three adapters per arm (t = 1.4). Under the diverse bank both arms move toward A relative to
the base model (+1.5, +1.0), so the additive shift caused by adaptation itself is larger than
under the single caption; the A − B difference is what survives. Consistent with Entry 18;
not an independent confirmation at this n. Written into §V-Q as one sentence.

**Figure 1, third version (2026-09-09, on the user's report that it still did not look
professional).** Redrawn as a schematic rather than text boxes: pictograms for the camera body,
photograph stack, latent encoder / transformer / decoder, LoRA low-rank factors on a frozen weight,
the text prompt, and the generated-image grid; numbered teal taps mark where endpoints 1–3 are
measured; the body-B comparator carries its own pictogram. `paper/fig1_pipeline_tikz.tex` →
`fig1_pipeline_v2.pdf` (7.12 × 2.75 in, fonts embedded). Manuscript recompiled (26 pp., 0 errors),
snapshot `paper/einv_v2_draft_2026-09-09f.pdf`.

---

## Entry 24 · 2026-09-09 · F4 result — the unmarked local-decode controls are positive too; the difference is between environments, not between marked and unmarked

`orchestrate2.sh` chain: `nomark_s0/s1/s2` trained 01:53–03:50, generated 03:50–07:23 (500 each),
measured 07:23–07:28 (`out/t1/summary_nomark.json`), `f4_stats.py` → `out/t1/f4_stats.json`;
`f4_derived.py` → `out/t1/f4_derived.json`. Exactly the Entry 20 F4 registration.

**Natural paired contrast ρ(→K̂_A) − ρ(→K̂_B), 500 generations per adapter.**

| group | values (×10⁻⁵) | mean ± SE (×10⁻⁵) |
|---|---|---|
| nomark s0 / s1 / s2 (unmarked, local decode, v2 stack) | +8.7, +0.9, +8.9 | +6.2 ± 2.6 |
| α = 1 arm (effectively unmarked, same environment) | +2.8 | |
| five marked ladder arms | +12.4, +4.0, +8.1, +9.3, +14.4 | +9.6 ± 1.8 |
| **all nine v2-trained A adapters** | all positive | **+7.7 ± 1.5**, t = 5.1 (df 8), p = 0.0005 |
| v1's twelve A adapters (ledger; Colab decode and stack) | 7 of 12 negative | −1.0 ± 1.4 (sd 4.8) |
| v1's A_raw s0–s2 re-measured on the *local* instrument (Entry 10) | +0.7, −1.0, −0.3 | ≈ 0 |

**Registered tests.** (i) marked > nomark: Welch t = 1.08, p = 0.17 — *no difference*. (ii) nomark
> v1 mean: one-sample t = 2.71 (df 2), p = 0.057; z against v1's adapter sd, 2.55, p = 0.005.
Nine v2 adapters vs twelve v1 adapters: Welch t and Mann–Whitney in `f4_derived.json`; the chance
that nine draws from v1's A distribution are all positive is 4e-4. Pre-declared reading:
**nomark ≈ marked > v1 → the mark is not the cause.**

**What it means, and what it does not yet mean.** Adapters trained on body A in the v2
environment carry a positive natural contrast toward K̂_A of about +8e-05 — λ ≈ 0.2 % of the
real-image contrast, above v1's upper limit of 0.15 % — whereas v1's adapters, re-measured on the
same instrument, sit at zero. The measurement is not the cause (Entries 02, 10, 13 reproduce v1's
numbers locally); the mark is not the cause (F4). What differs is the *training environment*:
the JPEG decode of the training crops (Entry 05: 28 % of pixels differ by ±1–4) and the training
stack (torch 2.11 / diffusers 0.39 / peft 0.20 / bf16 on an L40S versus v1's Colab A100 stack), and
one more thing that is not training at all: **the generations were made by the local pipeline**,
whose main effect (Section V-I: the generic K̂_A/K̂_B affinity of a generator's residual) has never
been measured under the uniform caption — the "base" comparator in every v2 measurement is
v1's Colab-generated base arm. E-PROMPT (Entry 22) showed the base-model main effect moving by
1e-04 with prompt content alone, so a local-pipeline main effect of +6e-05 toward K̂_A is entirely
possible and would produce exactly this pattern without any adapter carrying anything.
**Until that is excluded, this is not a transfer result.** Registered as F5 below; the paper's
§V-P sentence stays "recorded, not claimed" with the environment reading added.

---

## Entry 25 · 2026-09-09 · F5 registration — local-pipeline main effect versus adapter effect

Registered before any F5 image is generated (07:40).

**Design.** With the local generation pipeline (`src/f5_localgen.py`, identical code path to the
ladder and nomark arms: uniform caption, v1 paired seed bank, 500 images, 28 steps, CFG 4.5),
generate three arms that involve **no v2 training**: `local_base` (no adapter), `local_A_raw_s0`
and `local_B_raw_s0` (v1's archived adapters, exactly the files fetched from the public archive).
Measure with the same script (`T1_ARMSET=f5 t1_measure.py`): natural contrast ρ(→K̂_A) − ρ(→K̂_B)
per image, per arm.

**Pre-declared reading.**
- `local_base` ≈ +6e-05 toward K̂_A (i.e. within 2 image-SE of the nine v2 adapters' mean, and
  far from v1's Colab base −4e-05) → the v2 positives are a **generation-environment main
  effect**; Entry 24's finding is withdrawn as evidence of transfer and reported as one more
  instance of the main-effect hazard (a change of decoder/sampler build moves the unpaired
  reading by more than the study's upper limit).
- `local_base` ≈ v1's base (≈ −4e-05 or ≈ 0), `local_A_raw_s0` ≈ 0 (as v1's A adapters read on the
  Colab generations), and `local_B_raw_s0` ≈ 0 → the environment does not create the offset and
  the nine v2-trained adapters' +8e-05 is an **adapter property of the v2 training environment**
  (decode and/or stack); reported as a positive finding requiring a controlled decode-only
  replication (train on Colab-decoded PNGs from the archive with the v2 stack) before it is
  claimed as PRNU transfer.
- Intermediate: report both numbers and the difference; the paired local statistic
  (`local_A` − `local_B` own-body contrasts, symmetric) is the quantity that survives either way.

Cost: 3 × 70 min generation + 5 min measurement; GPU is free. Nothing here changes once generation starts.

**Addendum to Entry 24 (`f4_derived.json`).** Nine v2-trained A adapters vs v1's twelve: Welch
t = 4.27, one-sided p = 0.0002; Mann–Whitney p = 0.0006; difference of means +8.7e-05 = 0.24 % of
R_real. Nine of nine positive: mean +7.7e-05, SE 1.5e-05, λ = 0.216 % [0.097, 0.336] (99 %).
Marked (5) vs all unmarked (4, including the α = 1 arm): Welch t = 1.58, p = 0.08 — still no
mark effect at the registered level. v1's A_raw s0–s2 on the local instrument: +0.7, −1.0,
−0.3 (×10⁻⁵). F5 launched 07:44 (`logs/f5_generate.log`).

---

## Entry 26 · 2026-09-09 · Registration — the published-watermark arm (DiffusionShield at released strength) as the seventh rung of the ladder

Registered before any watermarked training image is written (09:20). Fulfils the optional `M_opt`
arm of Entry 01 ("the published mark at its published strength, only if the released code runs
unmodified").

**Mark.** DiffusionShield (Cui et al., SIGKDD Explorations 2024; github.com/Yingqiancui/DiffusionShield,
cloned 2026-09-09 into `D:\A4\ext\DiffusionShield`): the released pixel-domain codebook
`trained_patches/wm_patches.pt` — three 4×4×3 patches of amplitude ±8/255 (RMS 0.0310 in [0,1],
i.e. 7.9 gray levels) — and the released 64-symbol message `example.pt` (B = 4), laid out as the
authors' `add_watermark.py` does on an 8×8 block grid of 4-px blocks (a 32×32-px tile), and tiled
32 × 32 times to cover the 1024² training crop. Injection is the authors' rule, **additive in
[0,1] RGB, clipped**, then rounded once to 8-bit PNG (the v2 materialisation rule). Amplitude is
the released "budget 8": no scaling. The released block classifier `trained_model/classifier.pt`
(their detector) is applied unmodified per 4×4 block.

**Arms.** `wm_ds_s0/s1/s2`: rank-16 LoRA, 2000 steps, seeds 0–2, otherwise the primary protocol;
500 generations each from the primary seed bank. Field files `out/t1/fields/W_ds.npy` (RGB) and
`W_ds_lum.npy` (zero-mean luminance of the field).

**Stored energy.** R_wm = mean over the 50 stored crops of [ρ_add(Y → W_lum) − ρ_add(Y → M′)], where
ρ_add is the zero-lag NCC of the wavelet residual against the field itself (the additive
analogue of the multiplicative ρ_mult used for the natural fingerprint and the ±1 marks); the
RMS of the stored pixel change in gray levels is recorded beside it.

**Statistics (`src/t1_wm_measure.py`).** Per generation: ρ_add(→W_lum) − ρ_add(→M′); decoys: 30
circular rolls of W_lum whose row and column shifts are both non-multiples of the 32-px period
(a shift by a multiple of 32 reproduces the field exactly); rank of the true field among 31;
cluster t over the three seeds against zero and against the never-injected offset; λ_wm =
contrast / R_wm raw and offset-corrected; never-injected arms: the archived base / A_raw_s0 /
B_raw_s0 **and** the three v2-environment `nomark` arms; natural K̂_A − K̂_B contrast on every arm.
The authors' detector: each generation is cut into 1,024 tiles of 32 × 32 px, each tile into 64
blocks of 4 × 4, the classifier's argmax per block is decoded to two bits and compared with the
tiled message → bit accuracy per image (their metric), reported for the stored training crops
(should be ≈ 1), the wm arms, and the never-injected arms (chance 0.5).

**Pre-declared reading.** Detection ⇔ true field ranks first of 31 in every wm arm AND the
cluster t exceeds the 0.01 criterion (df 2) AND rank ≤ 2 in no never-injected arm. λ_wm goes on the
ladder of Entry 20 whatever it is. Bit accuracy on generations above 0.5 by more than three of
its never-injected spread → "the released detector recovers the released message from
generations"; otherwise → "the mark transfers at the correlation level but the released
detector, trained on 32-px CIFAR images, does not decode it from 1024² generations" — the two
are reported separately. The mark's RMS (≈ 7.9 gray levels) is ≈ 63× the natural fingerprint's
0.125 and ≈ 5× the α = 12 rung; that ratio is part of the reading, not a confound.

---

## Entry 27 · 2026-09-09 · Registration — E3, random-crop training (the untested translation-tolerance limitation)

Registered before any training (09:20). The Limitations section names random-crop training as
untested; v1's fixed centre crop is the sensitivity-favouring, spatially registered case.

**Arms.** `rcrop_s0/s1/s2`: device A's 50 T-split frames decoded at native 3872 × 2592 (local
decode); at **every training step** each sampled image is cropped at a uniformly random 1024²
position (seeded by the training generator), so the fixed-coordinate fingerprint the adapter sees
is a different spatial piece of K_A on every step. Everything else as the primary protocol
(rank-16, 2000 steps, seeds 0–2, uniform caption); 500 generations each from the primary bank.

**Statistic.** The natural paired contrast ρ(→K̂_A) − ρ(→K̂_B) of the 1024² generation against the
*centre-crop* fingerprints, as everywhere in the study. Comparator: the three `nomark` arms
(Entry 24) — same environment, same decode, same stack, fixed centre crop — so the comparison is
within the v2 environment and does not depend on F5.

**Pre-declared reading.** rcrop ≈ nomark → translation of the training crop does not change what
reaches the output (the fixed-crop protocol was not what produced either the v1 null or the v2
offset). rcrop < nomark by more than the adapter-level SE → the centre-crop fingerprint reaches
the output in proportion to how often it was seen (a registered signal, consistent with a
fixed-coordinate channel). rcrop > nomark → a translation-tolerant representation carries more
of the device signal than the registered one — the possibility the limitation names — reported
as such. Cost: 3 × 40 min + 3 × 70 min, queued behind the watermark arm (`src/orchestrate3.sh`).

---

## Entry 28 · 2026-09-09 · Registration — Noiseprint as a fifth detector family (positive-control gate first)

Registered before any Noiseprint number is read (09:20). Entry 09 recorded Noiseprint as not run
(TensorFlow 1.x); the A4 programme's TF 1.15 environment (`a4tf1`, Python 3.7) and the
`grip-unina/noiseprint` checkout with the published per-QF weights (`D:\A4\ext\noiseprint`) are on
this machine, with the session-reuse wrapper `D:\A4\src\a4_noiseprint.py` (verified bit-identical
to the reference call). CPU only.

**Design.** Noiseprint of every image (QF-matched net for the JPEG originals; the QF-101 net for
PNG generations), centre 1024² crop; per body a Noiseprint "fingerprint" = zero-mean average over
its E2 images; score = zero-lag NCC of an image's zero-mean Noiseprint against each fingerprint;
paired contrast = own minus other, exactly the panel's construction (Entry 09).

**Gate.** Real held-out photographs (H, 40 per body): same-model AUC ≥ 0.95, otherwise the detector
is reported as *failing its positive control on two bodies of one model* and generations are not
scored. Noiseprint is trained to extract camera-**model** artefacts and to suppress content; it is
not designed to separate bodies of one model, so failure is the expected outcome and is itself the
result: the field's deep model-fingerprint extractor is not a device-fingerprint extractor on this
pair. If the gate passes, the 3,500 base-study generations are scored and the panel table gains
a column.

---

## Entry 29 · 2026-09-09 · Noiseprint gate result — passes, narrowly: a model-level extractor keeps a small body-specific residue on this pair

`src/t2_noiseprint.py gate` in `a4tf1` (09:09–09:34, CPU, QF-99 net for the Dresden JPEGs) →
`out/t2_noiseprint_gate.json`; fingerprints `out/np_fingerprint_{A,B}.npy` (not for release).

| quantity | value |
|---|---|
| same-model AUC on H (40 + 40) | **0.9519** — gate ≥ 0.95 passed by 0.002 |
| real paired contrast (own − other), mean ± SE over 80 | 0.00339 ± 0.00033 (t ≈ 10) |
| NCC between the two bodies' Noiseprint fingerprints | **0.971** |

**Reading.** As expected of a camera-*model* extractor, the two bodies' Noiseprint fingerprints
are almost the same array (0.971), but the 3 % that differs is body-specific enough to separate
the held-out photographs at AUC 0.95 — a fifth working instrument on this pair, the weakest of the
five on real images (PRNU NCC 1.000, PCE 0.9997, learned CNN 0.989, low/mid 0.839 — Noiseprint
sits between the learned CNN and low/mid). Per the registration the 3,500 base-study generations
are now being scored (`t2_noiseprint.py gens`, ~4 h on CPU); the panel table gains a column
either way.

---

## Entry 30 · 2026-09-09 · F5 result — the local generation environment does not create the offset; the v2 positive is a property of the v2-trained adapters

`src/f5_localgen.py` (07:44–10:17: three arms × 500, local pipeline, uniform caption, v1 seed bank),
`T1_ARMSET=f5 t1_measure.py` → `out/t1/summary_f5.json`, `measure_rows_f5.csv`. Exactly Entry 25.

| arm (no v2 training) | K̂_A − K̂_B, mean ± image SE | the same adapter's Colab generations (Entry 10) |
|---|---|---|
| local_base (no adapter) | **−7.4e-05 ± 6.0e-05** | −4.0e-05 |
| local_A_raw_s0 (v1 adapter) | −4.1e-05 ± 7.1e-05 | +0.7e-05 |
| local_B_raw_s0 (v1 adapter) | +4.8e-05 ± 6.7e-05 | +4.2e-05 |
| symmetric statistic of the local A/B pair (own-body) | −4.4e-05 | ≈ 0 |

**Pre-declared reading, branch 2.** The local pipeline's base-model reading is on the *other*
side of zero from the nine v2-trained adapters (−7.4e-05 vs +7.7e-05; the difference is 2.4
image-SE of the single base arm and 10 adapter-SE of the nine), and v1's own adapters generated
locally read as they did on Colab — null. **The generation environment does not create the
offset.** The +8e-05 of Entry 24 is therefore a property of adapters *trained* in the v2
environment: the JPEG decode of the training crops and/or the training stack (torch 2.11 /
diffusers 0.39 / peft 0.20, bf16 casts of x_t and the timestep, L40S). Per the registration this
is reported as a positive finding that requires a decode-only replication before it is claimed as
PRNU transfer. It is that replication which decides whether v1's null was, in part, a property
of v1's training stack.

Caveat carried: each F5 arm is one 500-image draw (SE 6–7e-05); the exclusion of a +6e-05
environment offset rests on the base arm at 2.2 SE and on the v1 adapters behaving as before.

---

## Entry 31 · 2026-09-09 · F6 registration — decode-only replication (v2 stack on v1's Colab-decoded training crops)

Registered before any run (10:25).

**Design.** Fetch v1's archived training crops `E_INV_P0_v3/train_png/A_raw` (the 50 Colab-decoded
8-bit PNGs that trained v1's A adapters) into `out/t1/train_png/colab_a0/`; train three rank-16
adapters `colab_s0/s1/s2` on them with the v2 stack (identical to `nomark`, only the input PNGs
differ); 500 generations each; natural paired contrast toward K̂_A as everywhere.

**Pre-declared reading.**
- colab ≈ nomark (≈ +6e-05 to +9e-05, all positive) → the decode is not the cause; **v1's null
  was a property of v1's training stack**, and the v2 stack transmits the natural fingerprint at
  λ ≈ 0.2 % — reported as the study's first positive PRNU-transfer result, with the stack
  difference identified as the variable and v1's number kept as the other stack's result.
- colab ≈ v1 (≈ 0) with nomark ≈ +6e-05 → the decode is the cause; the local JPEG decode leaves
  training images from which the adapter picks up more of the fingerprint; reported as a decode
  effect, with the v1 null intact for v1's images.
- in between → both contribute; report the split.
Cost 3 × 40 + 3 × 70 min, queued after chain 3 (`src/orchestrate4.sh`).

**Manuscript v3 (2026-09-09, on the user's request to change the title and abstract and reformat).**
`src/paper_restructure.py` builds `paper/einv_v3.tex` from `einv_v2.tex` without touching v2:
IEEEtran journal class (TIFS-compatible; Access macros converted), new title *"PRNU Transfer Through
Diffusion Personalization: Upper Limits, Calibration by a Designed Mark, and Attribution Power"*,
new abstract (question → method → the three stages → the 10⁻³ channel → detector families → the
one positive → attribution power), rewritten findings paragraph and five contributions, and the
Results reordered: stages (V-A–E) → ladder (V-F) → detector families (V-G) → attribution (V-H) →
replications (V-I–M) → additive / shifted-template / prompt / integrity (V-N–Q). All body text
carried verbatim; cross-references by label; figures and tables re-checked to be in citation
order. Compiled 25 pp., 0 errors; snapshot `paper/einv_v3_ieeetran_2026-09-09a.pdf`. Placeholders:
two "[Noiseprint: pending]" (Entry 29's generations run finishes ~14:30).

---

## Entry 32 · 2026-09-09 · Noiseprint result — a fifth null on the paired statistic, and the largest main effect on any template detector: t ≈ 100 toward body B in every arm

`src/t2_noiseprint.py gens` (09:35–13:40, CPU) → `out/t2_noiseprint_gens.json`; decomposition →
`out/t2_noiseprint_summary.json`. Exactly Entry 28; read after Entry 29's gate.

| arm | NCC toward F_A / F_B | own-body contrast ± image SE |
|---|---|---|
| base | 0.1137 / 0.1165 | — (K_B − K_A = +0.00272) |
| A_raw s0 / s1 / s2 | 0.119 / 0.122 · 0.125 / 0.128 · 0.117 / 0.120 | −0.00295 / −0.00299 / −0.00295, SE 0.00003 |
| B_raw s0 / s1 / s2 | 0.115 / 0.118 · 0.120 / 0.123 · 0.120 / 0.123 | +0.00292 / +0.00290 / +0.00304, SE 0.00003 |

Real paired contrast (Entry 29) R = 0.00339. **Additive part −0.00296 = 87 % of R**;
**θ_sym = −5.7e-06**, adapter-level SE 2.3e-05, t = −0.24 (null); θ_sym/R = −0.17 %.

**Reading.** (1) Fifth detector family, fifth paired null. (2) Every generation correlates with
*both* bodies' Noiseprint fingerprints at ≈ 0.12 — thirty-five times the real paired contrast —
because Noiseprint extracts a camera-model signature and the generator's residual carries a
model-like signature of its own; on top of that every arm, base included, prefers body B by
0.003, i.e. 87 % of what separates real photographs of the two bodies, with an image-level t
near 100 because Noiseprint's score is so stable. An examiner reading Noiseprint unpaired would
attribute every one of the 3,500 generations — including the 500 from a model that never saw a
photograph — to body B with overwhelming confidence. This is the most extreme instance of the
main-effect hazard in the study (PRNU NCC: 0.01 % of R; PCE: 0.08 %; low/mid: 2.9 %; learned
CNN: 162 %; Noiseprint: 87 %). The paired statistic removes it exactly. Written into §V-N /
`tab:panel` (fifth column) and the abstract/introduction placeholders.

---

## Entry 33 · 2026-09-09 · Published-watermark arm result — DiffusionShield at released strength transfers at 3 % of its input contrast: the channel is a low-pass channel, and the released detector decodes nothing from generations

`orchestrate3.sh`: materialised 11:17 (`train_png/dswm_a1`, R_wm = 0.786 on the additive statistic,
stored change 6.5 gray levels RMS, 2.3 % of pixels clipped), trained 11:18–13:17, generated
13:17–16:52, measured 16:52–17:25 (`out/t1/wm_rows.csv`, `out/t1/wm_summary.json`). Exactly Entry 26.

**Correlation statistic, ρ_add(→W_lum) − ρ_add(→M′), 500 generations per arm.**

| arm | contrast ± SE | t (images) | rank of true W among 31 | decoy max | natural K̂_A − K̂_B |
|---|---|---|---|---|---|
| wm_ds_s0 / s1 / s2 | +0.0309 / +0.0350 / +0.0224 (SE ≈ 0.001) | 32 / 33 / 23 | **1 / 1 / 1** | 0.020 / 0.022 / 0.015 | −2.1e-04 / −1.3e-04 / −1.3e-04 |
| base / A_raw_s0 / B_raw_s0 (archived) | +0.0040 / +0.0041 / +0.0036 (SE 1e-04) | 45 / 40 / 30 | 2 / 4 / 5 | 0.004–0.007 | |
| nomark s0 / s1 / s2 (v2 env.) | +0.0033 / +0.0036 / +0.0038 | 27 / 31 / 28 | 7 / 4 / 5 | 0.006–0.007 | |

Never-injected offset **+0.0037** — every generation, base included, correlates with the released
32-px-periodic block pattern at t ≈ 30–45, and the true (grid-aligned) phase outranks most of its
rolled decoys even where it was never injected (ranks 2–7). This is the decoder-grid artefact of
v1 §V-M (PCE argmax at multiples of 8 px) seen from the other side: a periodic 4-px block pattern
aligned to the 8-px latent grid is exactly what the decoder emits generically. The never-injected
arms were registered to catch precisely this, and the offset is subtracted.

**Cluster test (three seeds):** mean +0.0294, sd 0.0064; t = 7.90 against zero (one-sided p =
0.0078 < 0.01, registered criterion met), t = 6.90 against the offset (p = 0.0102, at the
criterion). Rank 1 of 31 in all three arms; never-injected rank ≤ 2 in one arm (base, rank 2),
which the Entry 26 criterion ("rank ≤ 2 in no never-injected arm") does *not* satisfy — the
decoy test is confounded for a grid-aligned periodic mark, and that is reported rather than
waived. **λ_wm = 3.74 % raw, 3.27 % offset-corrected, plug-in U 8.4 %.**

**Released detector (their bit accuracy).** Stored training crops: **0.9989** (the detector reads
its own mark from the 8-bit PNGs). Generations: **0.5703 in every arm, wm and never-injected
alike, with zero spread** — the classifier emits the same block prediction everywhere (0.57 is the
agreement of that constant with the tiled message). Trained on 32-px CIFAR images with the mark at
8/255, it does not decode anything from 1024² generations; the mark transfers at the correlation
level and the released detector is blind to it. Both facts were pre-declared as separate
readings, and both are reported.

**Reading.** (1) The published mark at published strength comes out of the pipeline at
**3.3 % of its stored input contrast** — forty to eighty times the 0.04–0.08 % of the ±1 fields
of Entry 20 — with an input energy (6.5 gray levels RMS) ≈ 4× the α = 12 rung. (2) The reason
is spectral, not amplitude: the ±1 fields are white (PRNU-like), M_lowmid keeps DCT indices
i + j ≤ 5 of 8×8 blocks and gained only 0.083 %, and the DiffusionShield pattern is 4-px blocks
on a 32-px period — energy at 1/8 to 1/64 cycles per pixel. **The personalisation pipeline is a
low-pass channel for fixed-coordinate patterns**: transmission rises from ≈ 5 × 10⁻⁴ in the PRNU
band to ≈ 3 × 10⁻² for block patterns, and the active-marking literature's marks live where the
channel passes. This is the single most useful number for the "four differences" argument and it
goes into the paper as such. (3) The natural K̂_A contrast on the three wm arms is *negative*
(−1.3 to −2.1e-04) where every other v2-trained A adapter is positive (+0.8e-04): a strong
additive periodic mark in the training images reverses the natural fingerprint's sign in the
output. Observation; not tested; recorded for the F6 discussion.

Figure 8 is redrawn on log–log axes with the watermark as a seventh rung; `tab:ladder` gains a row.

**Manuscript (2026-09-09, evening).** Entries 30, 32, 33 written into `einv_v2.tex` (source) and
rebuilt into `einv_v3.tex`: §V-F seventh-rung paragraph (low-pass channel), `tab:ladder` row,
Figure 8 redrawn on log–log axes with the DiffusionShield rung and its confounded decoy ranks,
Discussion VI-B updated (channel passes block patterns at 3 × 10⁻²), §V-N Noiseprint paragraph
and `tab:panel` fifth column, §V-F F5 paragraph replacing "pending". Abstract and findings of v3
updated by the builder. v2 27 pp. / v3 25 pp., 0 errors, figure and table order verified.
Snapshots `einv_v2_draft_2026-09-09j.pdf`, `einv_v3_ieeetran_2026-09-09c.pdf`.

---

## Entry 34 · 2026-09-09 · E3 result — random-crop training: lower than the fixed-crop controls by less than one SE; excludes "translation-tolerant training carries more"

`orchestrate3.sh`: `rcrop_s0/s1/s2` trained 17:12–18:52 (random 1024² crop per step from the native
frames), generated 18:52–22:25, measured 22:25–22:30 (`out/t1/summary_rcrop.json`,
`measure_rows_rcrop.csv`); `out/t1/e3_stats.json`. Exactly Entry 27.

| adapters | natural K̂_A − K̂_B (×10⁻⁵) | mean ± SE |
|---|---|---|
| rcrop s0 / s1 / s2 | +9.8 / +2.5 / −7.4 | **+1.6 ± 5.0** (λ = 0.05 %) |
| nomark s0 / s1 / s2 (fixed centre crop, same environment) | +8.7 / +0.9 / +8.9 | +6.2 ± 2.6 |
| difference rcrop − nomark | | −4.6 ± 5.6 (Welch t = −0.82, p = 0.47) |

The ±1-field statistics on these arms (never injected) sit at the known offset (M − M′ +0.8 to
+1.1e-04, ranks 13–19): as expected, nothing to do with the mark.

**Pre-declared reading.** Neither "rcrop ≈ nomark" nor "rcrop < nomark by more than the SE" is
established at n = 3; the point estimate is lower, the interval covers both. What the result does
exclude is the third branch — a translation-tolerant regime carrying *more* device signal to the
output — which is the possibility the Limitations section raised. Written into §V-F and the
Limitations. F6 (chain 4) started 22:30.

---

## Entry 35 · 2026-09-10 · F6 result — the decoder is not the cause; defect D6 in the F6 registration; F7 registered (B-body arms in the v2 environment)

`orchestrate4.sh`: `colab_s0/s1/s2` (v2 stack, v1's archived Colab-decoded A_raw crops) trained
22:32–00:29, generated 00:29–04:02, measured 04:02–04:07 (`out/t1/summary_colab.json`);
`out/t1/f6_stats.json`.

| adapters (all trained on body A, v2 stack, unmarked) | natural K̂_A − K̂_B (×10⁻⁵) | mean ± SE |
|---|---|---|
| colab s0 / s1 / s2 (v1's Colab-decoded crops) | +10.0 / +0.9 / +6.6 | **+5.9 ± 2.7** |
| nomark s0 / s1 / s2 (local decode of the same frames) | +8.7 / +0.9 / +8.9 | +6.2 ± 2.6 |
| difference | | −0.3 ± 3.7 (Welch t = −0.08, p = 0.94) |
| all six | | +6.0 ± 1.7 |
| v1's twelve A adapters (Colab stack) | | −1.0 ± 1.4 |

**The decoder is not the cause** (same crops, two decoders, identical result to 3e-06).

**Defect D6 (registration, v2 — mine).** Entry 31 pre-declared "colab ≈ nomark → v1's null was a
property of v1's training stack, and the v2 stack transmits the natural fingerprint at λ ≈ 0.2 % —
reported as the study's first positive PRNU-transfer result." That reading is wrong by the study's
own construction. Every adapter trained in the v2 environment so far — nine unmarked or marked,
three Colab-decoded, three random-crop — was trained on **body A**. The unpaired contrast
ρ(→K̂_A) − ρ(→K̂_B) of A-trained adapters contains the fingerprint main effect (b_A − b_B) of *those
adapters' generations*, and that main effect is known to move with the generator (F5: local base
−7.4e-05; Entry 22: ×3 with prompt content; Entry 33: sign reversed by a strong periodic mark). A
training stack that shifts every adapter's generations toward K̂_A regardless of the training body
would produce exactly Entries 24, 30 and 35. The paper's primary statistic exists to cancel this
and requires B-trained adapters from the same environment. The Entry 31 wording is therefore
**withdrawn** (added to the register) and not carried into the paper; F6's valid conclusion is only
"the decoder is not the cause."

## Entry 35a · F7 registration — unmarked B-body adapters in the v2 environment (registered 04:20, before any run)

**Design.** `nomarkB_s0/s1/s2`: rank-16 LoRA, 2000 steps, seeds 0–2, on body B's T split (50 native
centre crops, local decode), identical to `nomark` in every other respect; 500 generations each from
the primary seed bank; `T1_ARMSET=nomarkB` in `t1_ladder.py` / `t1_measure.py`; `src/f7_stats.py`.

**Statistic.** θ_A = mean of the three `nomark` A adapters' own-body contrast; θ_B = mean of the
three `nomarkB` adapters' own-body contrast (ρ(→K̂_B) − ρ(→K̂_A)); θ_sym = (θ_A + θ_B)/2; additive
part = (θ_A − θ_B)/2; Welch adapter-level SE, one-sided t; plug-in 99 % limit; λ against R_real.
Secondary: θ_A from all six unmarked A adapters (nomark + colab).

**Pre-declared reading.**
- nomarkB own-body contrast positive (their natural K̂_A − K̂_B negative) **and** θ_sym > 2 SE →
  a device-specific interaction in the v2 environment; report λ_sym with its limit as a positive
  result, and state that v1's twelve-adapter null (Colab stack) and this result differ by
  training stack — requires a twelve-adapter replication before being claimed as the paper's
  headline.
- nomarkB natural K̂_A − K̂_B also positive (≈ +6e-05) → a v2-stack main effect; θ_sym ≈ 0; the
  positives of Entries 24/30/35 are one more instance of the main-effect hazard, and the paper says so.
- In between → report θ_sym, the additive part, and the limit.
Cost ≈ 2 h training + 3.5 h generation + 5 min measurement (`src/orchestrate5.sh`).

---

## Entry 36 · 2026-09-10 · F7 result — the B-body arms do not lean toward B: about half of the v2 A-arm lean is a training-stack main effect, the remainder is unresolved, and it sits inside the primary limit

`orchestrate5.sh`: `nomarkB_s0/s1/s2` (body B's T split, local decode, v2 stack, unmarked) trained
10:28–12:27, generated 12:27–16:04, measured 16:04–16:09 (`out/t1/summary_nomarkB.json`);
`src/f7_stats.py` → `out/t1/f7_stats.json`. Exactly Entry 35a.

| adapters (v2 environment, unmarked, fixed centre crop) | own-body contrast (×10⁻⁵) | mean |
|---|---|---|
| A: nomark s0 / s1 / s2 | +8.7 / +0.9 / +8.9 | θ_A = +6.2 |
| A (secondary): + colab s0 / s1 / s2 | +10.0 / +0.9 / +6.6 | θ_A = +6.0 (six) |
| **B: nomarkB s0 / s1 / s2** | **−3.4 / +3.1 / −1.2** | **θ_B = −0.5** |

| statistic | primary (3 A v 3 B) | secondary (6 A v 3 B) |
|---|---|---|
| θ_sym = (θ_A + θ_B)/2 | **+2.8e-05**, SE 1.6e-05, Welch t = 1.73 (df 3.7), p = 0.082 | +2.8e-05, SE 1.3e-05, t = 2.16 (df 5.0), p = 0.042 |
| additive part (θ_A − θ_B)/2 | **+3.3e-05** (0.09 % of R_real) | +3.3e-05 |
| λ_sym, plug-in 99 % limit | 0.08 %, U 0.26 % | 0.08 %, U 0.20 % |

**Against the pre-declared readings (Entry 35a).** Branch 1 (B arms lean toward B *and* θ_sym > 2 SE)
— not met: the B arms' own-body contrast is −0.5e-05, and t = 1.73. Branch 2 (B arms lean toward A
as much as the A arms, θ_sym ≈ 0) — not met either: the B arms sit near zero, not at +6e-05.
**Branch 3, in between:** report θ_sym, the additive part and the limit.

**Reading.**
1. About half of the A arms' +6e-05 lean is shared by adapters of either body — a main effect of
   the v2 training stack (+3.3e-05 toward K̂_A). Without the B arms it would have been read as
   transfer; D6 was the right call.
2. The remaining device-specific interaction, +2.8e-05, is not resolved from zero at three adapters
   per arm (p = 0.08 primary; 0.04 with six A adapters, not at the study's 0.01 level).
3. Its point estimate lies **inside the primary twelve-adapter limit** (U_device = 5.38e-05,
   λ_U = 0.15 %), and its own plug-in limit (0.26 %; 0.20 % secondary) is looser. The v2
   environment is consistent with the primary bound and does not revise it. v1's k = 12 θ_sym was
   +0.46e-05; the two estimates differ by 2.4e-05, about 1.4 of this entry's SE.
4. Paper: §V-F paragraph replaced ("pending" → this result); headline unchanged. R3 in the register
   stands as written; Entry 24's "above v1's upper limit" was an unpaired number.

**Manuscript (2026-09-10).** Entry 36 written into `einv_v2.tex` §V-F (the "pending" paragraph
replaced; headline unchanged). v3 abstract trimmed to 231 words by `src/abstract_trim.py` (the
250-word cap of both candidate venues); v3 26 pp., 0 errors, figures and tables in citation
order, no "pending" left. Snapshot `paper/einv_v3_ieeetran_2026-09-10c.pdf`. Evidence deposit
`deposit/einv_v2_evidence_2026-09-10.zip` (132 files, 14.9 MB, SHA-256 manifest) built by
`src/make_deposit.py` and checked: no fingerprint arrays, caches, images or adapters. All
registered experiments are now read; no chain is running.

---

## Entry 37 · 2026-09-11 · Registration — the transfer function of personalization by spatial-frequency band (A1 autoencoder only; A2 full pipeline)

Registered before any band field exists (the user asked for all three follow-ups on 2026-09-11).

**Question.** Entry 33 inferred "low-pass" from three pattern types. How does the pipeline transmit a
fixed-coordinate pattern as a function of spatial frequency, and is each band lost at the autoencoder
or during adaptation and generation?

**Fields.** Six octave bands of radial spatial frequency (cycles per pixel): b0 [0.25, 0.5],
b1 [0.125, 0.25), b2 [0.0625, 0.125), b3 [0.03125, 0.0625), b4 [0.015625, 0.03125),
b5 [0.0078125, 0.015625) — periods from 2–4 px to 64–128 px. F_b = ideal radial band-pass of a white
Gaussian field (numpy seed 20260912), zero-mean, unit RMS; comparator F'_b the same construction from
seed 20260913, never injected. Also reported: energy fraction per band of K_A (E2) and of the
DiffusionShield luminance field.

**Injection.** Additive on all three channels at 4.0 gray levels RMS for every band, rounded once to
8-bit PNG, device A's T split (50 crops) -> `train_png/band{b}_a4`. (4.0 lies between the alpha = 12
rung's 1.5 and DiffusionShield's 6.5.)

**Statistic.** Band-matched correlation s_b(image) = NCC(P_b Y, F_b) - NCC(P_b Y, F'_b), where P_b is
the same band-pass applied to the luminance image. No wavelet residual: the residual extractor is itself
high-pass and would bias the curve. All six s_b are computed on every image (a 6 x 6 matrix whose
off-diagonal is the leakage control).

**A1 (autoencoder).** SD-3.5-medium VAE, fp32, encode (mode) then decode each band's 50 stored crops and
the uninjected crops (`none_a0`), rounded to 8 bit.
T_vae(b) = [s_b(after, band b) - s_b(after, uninjected)] / [s_b(before, band b) - s_b(before, uninjected)].

**A2 (full pipeline).** One rank-16 adapter per band (`band{b}_s0`, seed 0, primary protocol, 2000 steps),
500 generations each. Never-injected arms: `nomark_s0..s2` and `nomarkB_s0..s2` (same environment).
T(b) = (C_b - offset_b) / R_b, with C_b the arm's mean s_b, offset_b the mean over the six never-injected
arms, and R_b the stored input contrast. Band b is detected if (C_b - offset_b) > 3 x its combined SE.

**Pre-declared reading.**
- T(b) decreasing from b5 to b0 over at least one order of magnitude -> "measured low-pass transfer
  function". The knee is reported as the band where T falls below 10 % of its maximum.
- Location: T(b) / T_vae(b) near 1 in a band -> the loss there is at the autoencoder; much less than 1
  -> the loss is at adaptation and generation.
- Any other shape (flat, band-pass) is reported as measured and replaces Entry 33's "low-pass" wording
  in the paper.
- With one adapter per band, the shape across bands is the result; per-band adapter variance is not
  estimated (the alpha = 3 seeds of Entry 20 varied by 11 % CV).

Cost ≈ 6 x (40 + 70) min on the L40S; A1 takes minutes.

---

## Entry 38 · 2026-09-11 · Registration — the learned detector: (a) with the PRNU template projected out, (b) five-body closed-set attribution

Registered before either network is trained.

**(a) Ablation.** Same four-block CNN, data, recipe and seed as Entry 18. Before patching, each residual
has its least-squares projection onto four PRNU templates removed: Y*K_A(E1), Y*K_A(E2), Y*K_B(E1),
Y*K_B(E2). This applies to real and generated images alike. Weights are saved
(`out/t2_learned_noprnu_net.pt`). The network scores the 24 primary adapters (250 generations each;
seeds 3–11 re-fetched from the archive) and the base model. Statistics are identical to Entry 17:
theta_sym, Welch t, exact label permutation over C(24,12), and the intersection-union sign-flip. Also
reported: the residual's NCC with Y*K_A(E2) on the H images, before and after projection.

Pre-declared reading:
- Ablated real AUC >= 0.95 *and* interaction resolved (permutation p < 0.01), with lambda within a factor
  of 2 of Entry 18's 4.7 % -> "the residue is not the PRNU template; a body-specific non-template
  component survives personalization".
- Real AUC >= 0.95 but interaction not resolved (permutation p > 0.05) -> "the learned interaction
  depended on the template component; the learned detector saw PRNU below the template detectors'
  floor".
- Real AUC < 0.95 -> the non-template residual does not identify the bodies on this pair; the ablation
  says nothing about the residue, and is reported as such.
- Caveat: the projection removes the zero-lag component along two estimates of each fingerprint;
  estimation error leaves some PRNU behind.

**(b) Five-body attribution.** One 5-way CNN per group (Kodak M1063 D0–D4; Huawei P20 1101–1105), the
same architecture with 5 outputs. Trained on E1 and E2 residual patches (16 per image), 20 epochs,
AdamW 1e-3, seed 0; weights saved. Gate: top-1 accuracy on the H split >= 0.90. Generations: the
archived 250 per adapter, 10 adapters per group. Score = patch-mean logits. Two variants, exactly as in
Entry 06: raw, and main-effect-corrected (for each candidate d, subtract its mean logit over generations
of adapters not trained on d). G in {1, 10, 50, 250}; the adapter-level decision is at G = 250.

Pre-declared reading:
- Corrected score correct for >= 6 of 10 adapters in a group (P(X >= 6 | 10, 0.2) = 0.0064) -> "the
  learned detector attributes personalized models to their source body above chance". That would be
  the study's first positive attribution.
- <= 4 of 10 -> chance; this extends Entry 07 to a learned detector.
- 5 of 10 -> reported, not significant at 0.01.
- Both groups pooled (20 adapters): the smallest k with exact p < 0.01 is reported alongside.
- If a group fails the gate, its generations are scored descriptively only.

---

## Entry 39 · 2026-09-11 · Registration — the adaptation-dose axis (2000 / 8000 / 16000 steps), paired by body

Registered before any run.

**Question.** Does the bound hold under stronger adaptation, towards memorization?

**Arms.** Unmarked, local decode, v2 stack, fixed centre crop — identical to `nomark` / `nomarkB` except
for the step count, with the cosine LR over the full run:
- 8000 steps: `dose8k_A_s0`, `dose8k_A_s1`, `dose8k_B_s0`, `dose8k_B_s1`
- 16000 steps: `dose16k_A_s0`, `dose16k_B_s0`

250 generations each from the primary seed bank. Reference at 2000 steps: `nomark` s0–s2 and `nomarkB`
s0–s2 (Entry 36).

**Statistic.** theta_A, theta_B, theta_sym and the additive part at each dose, paired by body (the D6
lesson). Memorization, descriptive only:
- per generation, the maximum NCC of its 256^2 grayscale thumbnail against the 50 training crops of
  its body; reported as mean, 95th percentile, and fraction above 0.5;
- adaptation strength = mean |generation - local_base| at matched seeds.

**Pre-declared reading.**
- theta_sym at 8000 within +/- 2 SE of the 2000-step estimate and below U_device -> "the bound holds
  into four times the adaptation".
- theta_sym at 8000 exceeding U_device by > 2 SE, with the memorization metric rising -> "transfer
  appears under stronger adaptation", reported with the dose at which it appears.
- 16000 steps is one pair per body; descriptive only.

Cost ≈ 25 GPU-hours.

---

## Entry 40 · 2026-09-11 · The watermark's spectrum contradicts Entry 33's explanation; registration of periodicity and grid-alignment arms

**Observation (from `out/t1/band_fields.json`, written by Entry 37's field stage before any band arm
was trained).** Energy fraction by octave band, b0 (0.25–0.5 cycles/px) to b5:

| field | b0 | b1 | b2 | b3 | b4 | b5 | beyond radial 0.5 (corners) |
|---|---|---|---|---|---|---|---|
| K_A (E2), the natural fingerprint | 57.9 % | 26.0 % | 5.2 % | 0.8 % | 0.03 % | 0.01 % | ≈ 10 % |
| DiffusionShield luminance field | 54.1 % | 2.4 % | 0.8 % | 0.5 % | 0 | 0 | ≈ 42 % |
| white field (area fractions, reference) | 58.9 % | 14.7 % | 3.7 % | 0.9 % | 0.2 % | 0.1 % | 21.5 % |

The DiffusionShield pattern's energy is concentrated at the *highest* spatial frequencies (its 4-px block
edges), not at 1/8–1/64 cycles per pixel as Entry 33 stated. It carries *less* mid-band energy than a
white field. Entry 33's reading (2), "the channel is low-pass and the active-marking literature's marks
live where it passes", is therefore not supported by the mark's own spectrum. It is suspended (R4). The
transmission numbers of Entry 33 (3.3 % vs 0.04–0.08 %) stand.

What does distinguish the released mark from every field of Entries 20 and 37:
- it is **periodic** — one 32 x 32 px tile repeated 1,024 times;
- its period is a multiple of the 8-px latent grid and the 16-px transformer patch, so every token sees
  the same sub-pattern.

A periodic pattern can be learned as a translation-invariant texture; a non-repeating fixed-coordinate
field such as PRNU can only be reproduced by memorizing absolute positions. Entry 33's never-injected
offset (+0.0037 at t 30–45) already showed the decoder emitting generic grid-periodic structure.

**Registration (before any periodic field exists).**

Fields:
- P32: a random Gaussian 32 x 32 tile (seed 20260914), tiled over 1024^2. Periodic, and aligned with
  both grids.
- P36: a random 36 x 36 tile (seed 20260915), tiled and cropped. Periodic, aligned with neither grid.
- Comparators: the same construction from seeds +100, same periods, never injected.
- All zero-mean, unit RMS, additive at 4.0 gray levels RMS on all channels, rounded once, on device A's
  T split -> `train_png/per32_a4`, `per36_a4`.

Arms: `per32_s0`, `per36_s0` (rank-16, 2000 steps, seed 0, primary protocol, 500 generations). The
non-periodic comparator is Entry 37's `band0_s0` (a top-octave random field, same amplitude), read with
the same statistic.

Statistic: the DiffusionShield statistic of Entry 33, rho_add = NCC(W, F) - NCC(W, F'), with W the
wavelet residual. Decoys: 30 circular rolls whose row and column shifts are not multiples of the field's
period. Never-injected arms: `nomark_s0..s2` and `nomarkB_s0..s2`. lambda = (C - offset) / R, with R
measured on the stored crops. Also the autoencoder-only transmission (as Entry 37 A1) for all three
fields. Script: `src/t1_periodic.py`.

Pre-declared reading (DiffusionShield: lambda_wm = 3.27 % on this statistic):
- lambda(P32) and lambda(P36) both much greater than lambda(band0) -> **periodicity** explains the
  published mark's transfer, whatever its alignment.
- lambda(P32) much greater than lambda(P36), with P36 near band0 -> **grid alignment** is required.
- lambda(P32) near lambda(band0) -> neither; the mark's advantage comes from its specific patch design
  or from its larger amplitude (6.5 vs 4.0 gray).
- "Much greater" means a factor of at least 5 with non-overlapping plus/minus 2 SE intervals.

The paper's "low-pass" wording is replaced by whichever reading holds, together with Entry 37's band
curve. These arms run **before** the band and dose arms (chain 6 restarted as `orchestrate6b.sh`), so
the correction is settled first.

---

## Entry 41 · 2026-09-11 · Entry 38(b) result — the five-way learned detector fails its positive control in both groups; generation scores are descriptive only

`src/t2_learned_5body.py` (chain 7, 05:33–06:03) -> `out/t2_learned_5body_{kodak,p20}.json`,
`_logits.npz`, `_net.pt`; `out/t2_learned_5body_pooled.json`.

**Gate (registered: H top-1 >= 0.90).** Kodak **0.205**: every held-out image is labelled D3. Huawei P20
**0.207**: every image is labelled 1105. **Both fail.** Per Entry 38 the generation scores are
descriptive only.

**Why it fails (diagnosed on the saved weights before anything else was changed; no re-run).**
- Train-mode and eval-mode accuracy are the same (0.175–0.225), so this is not a BatchNorm
  running-statistics fault.
- Residual RMS is equal across bodies (0.98–1.03), so the network is not keying on noise level.
- Training loss: Kodak went from 1.61 (chance for 5 classes) only to 1.47 — it never learned the
  task. Huawei reached 0.33, yet it labels even *its own training images* as 1105 when shown different
  patch positions — it memorized the content of the specific training patches.
- The binary Nikon network (Entry 18) reached loss 0.18 and AUC 0.99 with the same recipe.
- Reading: a convolutional network with global average pooling is invariant to *where* a feature
  occurs. It can learn stationary residual texture (noise character, processing), but not a
  position-specific, non-repeating pattern such as a sensor fingerprint. It separated the two Nikon
  D200 bodies on texture; the five Kodak compacts and the five P20 phones do not differ enough in
  texture for it to learn. A position-aware detector would amount to the fingerprint template, which
  the paper already tests.

**Descriptive generation scores (not interpretable as attribution, because the detector fails on real
photographs).** Adapter-level correct at G = 250:
- Kodak: raw 4/10 (picks D0 5, D3 5); corrected 5/10.
- P20: raw 1/10 (picks 1103 4, 1105 6); corrected 3/10.
- Pooled corrected: 8/20, one-sided p = 0.032; 9 were needed for p < 0.01.

**Consequence.** Entry 38(b)'s question — can a learned detector attribute personalized models to one of
five bodies — cannot be answered with a detector that does not identify those bodies from real
photographs. What is established is narrower and still useful: a position-invariant learned residual
detector does not separate five same-model bodies on either device class. This also bears on Entry 18.
The learned detector's Nikon-pair interaction (4.7 %) must be a stationary-texture difference, not the
fingerprint pattern, unless Entry 38(a) shows otherwise.

---

## Entry 42 · 2026-09-11 · Entry 37 A1 and Entry 40 A1 results — the autoencoder is a sharp low-pass filter with its cut in the top octave, and it predicts the Stage-1 retention of the natural fingerprint

`src/t1_band.py vae` and `src/t1_periodic.py vae` (chain 6b; SD-3.5 VAE fp32, encode mode, decode,
rounded to 8 bit; 50 stored crops per field) -> `out/t1/band_vae.json`, `periodic_vae.json`.
Derived (secondary, not pre-registered): `src/a1_derived.py` -> `out/t1/a1_derived.json`.

**Autoencoder transmission by band (band-matched statistic).**

| band | cycles/px | period (px) | T_vae |
|---|---|---|---|
| b0 | 0.25–0.5 | 2–4 | **0.155** |
| b1 | 0.125–0.25 | 4–8 | 0.918 |
| b2 | 0.0625–0.125 | 8–16 | 1.017 |
| b3 | 0.031–0.0625 | 16–32 | 1.018 |
| b4 | 0.016–0.031 | 32–64 | 1.011 |
| b5 | 0.0078–0.016 | 64–128 | 1.006 |

The 6 x 6 matrix is diagonal to within 0.002 (no leakage between bands). With the residual statistic of
Entry 40: per32 0.166, per36 0.173, band0 0.130. The periodic fields pass the autoencoder no better
than a non-periodic field in the same band.

**Reading.** The SD-3.5 autoencoder is a sharp low-pass filter. It passes every band below 0.25
cycles/px essentially unchanged and removes about 85 % of the top octave — periods of 2–4 px, below
the 8-px latent grid.

**Derived check (secondary).** Weighting T_vae(b) by the fraction of K_A's energy in each band predicts
a Stage-1 retention of the natural fingerprint of **0.391–0.406**. The range covers the unmeasured
diagonal corners (10 % of K_A's energy), bracketed between 0 and T_vae(b0). The value measured
independently in v1 on real photographs is **eta = 0.366**. The band curve accounts for Stage 1 to
within about 7–10 %: the autoencoder loses most of the fingerprint because 58 % of the fingerprint's
energy sits in the one octave it cuts.

**What this says about the DiffusionShield result.** The same weighting gives the published mark an
autoencoder retention of 0.12–0.18, no higher than the top-band random field. The autoencoder does
not favour it. Its end-to-end transmission of 3.3 %, 40–80 times that of the ±1 fields, must therefore
arise *after* the autoencoder, in adaptation and generation. This is where Entry 40's periodicity arms
test it. The ±1 and DiffusionShield numbers use different statistics (multiplicative vs additive
residual), so the factor is indicative only; Entry 40 compares on one statistic.

**For the paper.** The "low-pass" description is correct *for the autoencoder* and now measured. What
remains suspended (R4) is the claim that low-pass behaviour explains why published marks transfer; that
waits for Entries 37 A2 and 40.

---

## Entry 43 · 2026-09-11 · Entry 38(a) result — with the PRNU template projected out of every residual, the learned detector's interaction is unchanged: the residue is not the fingerprint template

`src/t2_learned_noprnu.py` (chain 7, 06:03–06:56) -> `out/t2_learned_noprnu.json`, weights
`out/t2_learned_noprnu_net.pt`. Derived (secondary): `src/t2_noprnu_derived.py` ->
`out/t2_learned_noprnu_derived.json`. Seeds 3–11 were re-fetched from the archive (18 x 250, 0 failures).

**Projection check.** Mean |NCC(W, Y*K_A(E2))| on the H images: 0.0205 before projection, 2.1e-08 after.
The template component is removed.

| | original network (Entry 18) | template projected out |
|---|---|---|
| real H AUC (A vs B) | 0.989 | **0.998** |
| real paired contrast R (logit units) | 8.10 | 10.35 |
| theta_sym (12 adapters per arm, 250 generations each) | +0.381 | **+0.443** |
| Welch t (p) | 3.69 (0.0006) | 3.53 (0.0010) |
| exact label permutation p, over C(24,12) | 0.0008 | **0.0011** |
| IU sign-flip p_A / p_B | 0.016 / 0.010 | 0.023 / 0.0105 |
| lambda_sym (plug-in U) | 4.7 % (7.9 %) | **4.3 % (7.3 %)** |
| additive part, % of R | 162 % | 118 % |

**Pre-declared reading: branch 1 is met.** The ablated real AUC is 0.998 (>= 0.95); the permutation p is
0.0011 (< 0.01); lambda is 4.3 %, within a factor of 2 of 4.7 %. *"The residue is not the PRNU template;
a body-specific, non-template component survives personalization."* As in Entry 18, the registered
intersection-union sign-flip does not reject at 0.01, and that is reported.

**Derived (secondary).** The two networks see the same per-adapter structure:
- adapter-level own-body contrasts correlate at Pearson r = 0.992 (Spearman 0.987, n = 24);
- per-seed A - B differences correlate at r = 0.990 (n = 12);
- the same two seeds are negative under both (6 and 7), and 10 of 12 are positive under both.

Removing the template changed essentially nothing: the network was never using it.

**Reading, with Entry 41.** A position-invariant network learns stationary residual texture, not a
fixed-coordinate pattern (Entry 41). With the template removed it separates the two Nikon bodies even
better, and the interaction survives unchanged. What passes through personalization at a few per cent
of its real-photograph strength is therefore **camera-specific noise texture**. Candidates include
noise character, demosaicing and processing signatures, and dark-signal statistics. It is not the PRNU
pattern that forensic PRNU detection uses. It is also not strong enough in same-model compact CCDs or
smartphones for this detector to separate five bodies (Entry 41).

This replaces "composition unresolved" in the manuscript with "not the fingerprint template; a
stationary camera-specific texture". Which texture property it is remains untested.

**Manuscript consistency and flow pass (2026-09-11, user request; Reviewer 1's flow comment in mind).**
`src/paper_consistency_0911.py`, `src/paper_split_env.py`, plus edits to the builder.

Consistency fixes (each carried to v3 by the builder):
- The suspended "low-pass explains watermarks" wording (R4) is removed from the abstract, the
  intro findings, §V-F, the Figure 8 caption, panel title and legend, and the Discussion.
- §V-F's statement that the watermark "lives at 1/8 to 1/64 cycles per pixel" is replaced by its
  measured spectrum and a bold pending marker for Entry 40.
- The panel subsection is retitled "Five Detector Families"; Methods now says five families, three of
  them template detectors.
- The Conclusion says four template families, and no longer claims attribution stays at chance "with
  unlimited generations".
- The intro's attribution-power sentence now carries the 0.06–0.12 bracket used in §V-H.
- The prior-art table's last row is updated.
- The reviewer map is updated.

Flow additions:
- a Results roadmap naming what each group of subsections answers;
- a "Findings at a glance" table (question, answer, section) opening the Discussion;
- Methods paragraphs for every added experiment already reported (published watermark, autoencoder
  frequency response, environment controls, learned-detector ablation and five-body attribution);
- the autoencoder frequency-response paragraph in §V-A;
- §V-F retitled "Calibrating the Pipeline With Designed Patterns";
- the environment controls (F4–F7) and random crop moved out of §V-F into a new replication
  subsection, "Replication in a Second Training Environment" (`sec:res:env`), placed after the other
  replications in v3.

Verified on the rebuilt v3: 0 errors, 0 undefined references, figures and tables in citation order,
abstract 226 words, no leftover suspended wording (scripted scan of the bound, retention, learned
detector, transmission factors, detector counts and adapter counts).

**Manuscript tone pass (2026-09-11, user request: the paper should not read as an answer to reviewers
or as defensive).** `src/paper_tone_0911.py`: 44 sentences in the source and one in the v3 builder
rewritten as plain statements of the finding. Caveats were kept but phrased as scope rather than
defence. Examples of the voice removed: "which we report because a rule that does not bind…", "We state
this explicitly because…", "reported as such", "the correct statement is", "the honest unit", "we make no
claim there", "registered before any new image was scored". Registration stays in the Methods.
Scanners `src/_tone_scan.py` and `src/_tone_sentences.py` are kept for later edits; the 13 remaining
flagged sentences are ordinary methodological statements. The preference is saved to memory
(`einv-paper-tone`). v3 rebuilt: 0 errors, 0 undefined references, citation order intact, abstract
226 words; snapshot `paper/einv_v3_ieeetran_2026-09-11c.pdf`. The reviewer-by-reviewer mapping
stays in `paper/REVIEWER_RESPONSE.md`, outside the manuscript.

---

## Entry 44 · 2026-09-11 · Entry 40 interim result — grid alignment, not repetition, lets a fixed pattern through: the 32-px tile transmits 28 times better than the 36-px tile

`src/t1_periodic.py measure` (interim, 11:09–11:31, before the band0 comparator exists) ->
`out/t1/periodic_summary_interim.json`, `periodic_rows.npz`. Arms trained 06:06–07:26 and generated
07:26–09:50 (chain 6b). Additive residual statistic, exactly Entry 40.

| field (additive, 4.0 gray RMS) | R (stored) | excess over never-injected | t | lambda (±2 SE) | rank of true among 31 | never-injected ranks | T_vae |
|---|---|---|---|---|---|---|---|
| P32 (32-px period; aligned with the 8-px latent grid and 16-px patch) | 0.828 | 0.0401 ± 0.0016 | 24.8 | **4.84 %** (4.45–5.23) | **1** | 4–5 | 0.166 |
| P36 (36-px period; aligned with neither) | 0.837 | 0.0014 ± 0.0001 | 11.8 | **0.17 %** (0.14–0.20) | 8 | 17 (chance) | 0.173 |
| DiffusionShield (Entry 33, 6.5 gray RMS, reference) | 0.786 | — | — | 3.27 % | 1 | 2–7 | 0.12–0.18 (derived) |

**Reading (pre-declared criteria of Entry 40).** lambda(P32) / lambda(P36) = 28, with 2-SE intervals
far apart. That is "much greater" by the registered definition (a factor >= 5 with non-overlapping
intervals). The first half of the "grid alignment is required" branch is met. The second half —
P36 near band0 — needs the non-repeating comparator; `band0_s0` is generated after the band adapters
and re-measured by the final waiter.

What is already clear:
- An equally periodic, equally strong, equally fine-scale pattern transfers 28 times less when its
  period does not divide the model's latent grid.
- The grid-aligned random tile transfers slightly *better* than the published mark (4.8 % vs 3.3 %).
  DiffusionShield's success is therefore a property of grid-aligned periodic structure, not of its
  patch design.
- The autoencoder does not discriminate (T_vae 0.166 vs 0.173). The selection happens in adaptation
  and generation.
- The aligned tile correlates weakly with never-injected generations at its true phase (ranks 4–5,
  offset 0.0006). This is the same decoder-grid preference seen with DiffusionShield, far smaller than
  the injected effect (0.040).

Interpretation (not tested): a pattern that repeats on the model's own grid is the same texture in
every latent token, which a LoRA on the attention projections can learn once and emit everywhere. A
non-repeating fixed-coordinate pattern, such as a sensor fingerprint, has to be memorized
position by position.

---

## Entry 45 · 2026-09-11 · Entry 40 final result — transfer through personalization is governed by spatial structure: a non-repeating field is not detected, repetition gives 0.17 %, repetition on the latent grid gives 4.8 %

`src/t1_periodic.py measure` (final, 15:01–15:25, after `band0_s0` reached 500 generations) ->
`out/t1/periodic_summary_final.json` (per32 and per36 rows identical to Entry 44; band0 added).
Additive residual statistic, exactly Entry 40.

| field (additive, 4.0 gray RMS, one adapter, 500 generations) | R (stored) | excess over never-injected | t | lambda (±2 SE) | rank among 31 | T_vae |
|---|---|---|---|---|---|---|
| band0: non-repeating random field, top octave | 0.728 | 0.00007 ± 0.00007 | 0.96 | **0.010 %** (−0.010 to 0.030) | 3 | 0.130 |
| P36: tile repeated every 36 px (off grid) | 0.837 | 0.00144 ± 0.00012 | 11.8 | **0.17 %** (0.14–0.20) | 8 | 0.173 |
| P32: tile repeated every 32 px (on the 8-px latent grid) | 0.828 | 0.0401 ± 0.0016 | 24.8 | **4.84 %** (4.45–5.23) | 1 | 0.166 |
| DiffusionShield (Entry 33, 6.5 gray RMS) | 0.786 | — | — | 3.27 % | 1 | — |

**Against the pre-declared readings (Entry 40).**
- Branch "periodicity explains the published mark's transfer, whatever its alignment" is **met**. P32
  and P36 are each far above band0: P36 / band0 = 17.7 on the point estimates (5.7 against band0's
  upper 2-SE limit), with non-overlapping intervals; P32 / band0 is larger still.
- Branch "grid alignment is required" (P32 much greater than P36 *and* P36 near band0) is only
  half met: P32 / P36 = 28, but P36 is not near band0.
- The registration treated the two properties as alternatives. The data show that both act, and
  that they multiply: repetition lifts a fixed pattern from undetectable to 0.17 %, and repetition
  on the latent grid lifts it a further 28 times, to 4.8 %.

**Reading.**
1. What passes through personalization is set by a pattern's spatial structure, not by its
   frequency content. All four fields sit at the same fine scale and pass the autoencoder alike
   (T_vae 0.13–0.17).
2. The published mark's 3.3 % is explained by its grid-aligned repetition. A random tile on the
   same grid does as well or better (4.8 %).
3. The natural fingerprint is non-repeating, the one structure that does not come through. Its
   point estimate (0.013 %, Entry 20) and the non-repeating fields of this entry and Entry 20
   (0.01 % to 0.08 %) sit together at the bottom of the scale.
4. Interpretation (not tested): a pattern that repeats on the latent grid is the same texture in
   every latent token, so a LoRA on the attention projections can learn it once and emit it
   everywhere. A non-repeating pattern would have to be stored position by position.

Register R4 is withdrawn and replaced by this reading. The autoencoder is low-pass (Entry 42), but
it does not select the watermark. Written into the manuscript by `src/paper_periodic_final.py`, with
Figure 8 redrawn to include the three tiles.

---

## Entry 46 · 2026-09-12 · Entry 37 A2 result — the full-pipeline frequency response of non-repeating patterns; spectrum-matched predictions refine Entry 45 and place the fingerprint's expected transfer just under the limit

`src/t1_band.py measure` (chain 6b; six band adapters trained 09:50–13:50 and generated 13:50–20:58 on
2026-09-11; measured 20:58–21:03) -> `out/t1/band_summary.json`, `band_rows.npz`. Band-matched
statistic, exactly Entry 37. Derived (secondary): `src/band_derived.py` -> `out/t1/band_derived.json`.

| band | period (px) | T_vae (A1) | T_full (A2) ± SE | t | detected (3 SE) | T_full / T_vae |
|---|---|---|---|---|---|---|
| b0 | 2–4 | 0.155 | **0.00005 ± 0.00011** | 0.43 | no | 0.0003 |
| b1 | 4–8 | 0.918 | **0.00303 ± 0.00026** | 11.4 | yes | 0.0033 |
| b2 | 8–16 | 1.017 | **0.00397 ± 0.00071** | 5.6 | yes | 0.0039 |
| b3 | 16–32 | 1.018 | 0.0015 ± 0.0018 | 0.81 | no | — |
| b4 | 32–64 | 1.011 | 0.0015 ± 0.0044 | 0.33 | no | — |
| b5 | 64–128 | 1.006 | −0.0012 ± 0.0097 | −0.12 | no | — |

Standard errors grow roughly tenfold per octave below b2, because scene content dominates the coarse
bands; b3–b5 are unresolved, not measured as zero.

**Against the pre-declared reading (Entry 37).** "T decreasing from b5 to b0 over at least one order of
magnitude" cannot be established: the coarse bands are unresolved. The measured shape is nothing
detectable in the finest octave (2-SE upper limit 0.027 %), 0.30 % and 0.40 % in the two octaves
spanning 4–16 px, and unresolved below. The knee (T below 10 % of its maximum) lies between b1 and b0.
Location: where the autoencoder passes everything (b1, b2), personalization and generation keep
0.3–0.4 %. The bulk of the loss for non-repeating patterns is in adaptation and generation; in the
finest octave the autoencoder removes a further 85 %.

**Derived (secondary): spectrum-matched predictions.** prediction = sum over bands of the pattern's
energy fraction x T_full(b); the diagonal corners are assigned T_full(b0).

| pattern | predicted for a non-repeating pattern of this spectrum | measured | measured / predicted |
|---|---|---|---|
| P32 (tile, 32 px, on grid) | 0.057 % | 4.84 % | **85** (63 against the 2-SE upper prediction) |
| P36 (tile, 36 px, off grid) | 0.057 % | 0.172 % | **3.0** (2.2) |
| DiffusionShield | 0.016 % | 3.27 % | **210** |
| **K_A, the natural fingerprint** | **0.104 %** (2-SE upper 0.126 %) | point estimate 0.013 %, upper limit 0.15 % (Entry 20, ledger) | — |

**Readings.**
1. Frequency matters for non-repeating patterns. The finest octave passes nothing detectable, and no
   octave passes more than about 0.4 %.
2. Spatial structure matters far more. Repetition on the latent grid multiplies transmission about 85
   times over a non-repeating pattern of the same spectrum. Off-grid repetition does so only about 3
   times.
3. **The channel's measured response predicts 0.10 % for a non-repeating pattern shaped like the
   fingerprint.** That is below the study's 0.15 % limit, so the null is what the channel predicts. By
   the same token, the design could not have resolved transfer at the predicted level; a limit several
   times tighter would be needed. This goes into the Discussion and the Limitations, stated plainly.
4. Caveats: one adapter per band; the prediction treats the fingerprint as if it behaved like an
   injected additive pattern; the natural fingerprint enters multiplicatively and is measured with the
   multiplicative statistic; the coarse bands are unresolved.

**Refinement of Entry 45 (register R5).** Entry 45's "repetition lifts a fixed pattern from
undetectable to 0.17 %" compared P36 with a top-octave-only field. Against a non-repeating pattern with
P36's own spectrum, off-grid repetition gives about 3 times. Entry 45's "governed by spatial structure
rather than by frequency content" becomes "frequency sets a floor; spatial structure, above all
repetition on the latent grid, dominates". Carried into the manuscript by `src/paper_band_final.py`,
with a new Figure 9 (`src/fig9_bands.py`).

---

## Entry 47 · 2026-09-12 · Entry 39 interim result (8000 steps) — four times the adaptation, more memorization, no device-specific transfer; the shared stack main effect grows

`orchestrate6b.sh`: `dose8k_A_s0/s1`, `dose8k_B_s0/s1` (8000 steps, cosine over the full run, unmarked,
local decode, v2 stack) trained 2026-09-11 21:26 to 2026-09-12 07:3x; 250 generations each; measured by
`T1_ARMSET=dose8k t1_measure.py` (`out/t1/summary_dose8k.json`); `src/dose_stats.py` ->
`out/t1/dose_stats.json` (copy `dose_stats_8k_interim.json`). Exactly Entry 39.

| | 2000 steps (nomark / nomarkB, 3 + 3) | 8000 steps (2 + 2) |
|---|---|---|
| A adapters, own-body contrast (x10^-5) | +8.7, +0.9, +8.9 | +13.3, +5.6 |
| B adapters, own-body contrast (x10^-5) | −3.4, +3.1, −1.2 | −10.1, −15.9 |
| theta_A / theta_B (x10^-5) | +6.2 / −0.5 | +9.5 / −13.0 |
| **theta_sym** | +2.8 ± 1.6 x10^-5 (t 1.73) | **−1.8 ± 2.4 x10^-5 (t −0.73)** |
| additive part (shared lean toward K_A) | +3.3 x10^-5 | **+11.3 x10^-5** |
| lambda_sym (plug-in 99 % limit) | 0.08 % (0.26 %) | −0.05 % (0.47 %) |
| mean \|generation − local base\| at matched seeds | 32–36 gray levels | 41–44 |
| memorization: max thumbnail NCC with the 50 training crops, mean / share > 0.5 | 0.29–0.31 / 3–6 % | 0.31–0.38 / 8–18 % |
| (local base model) | 0.26 / 0.8 % | |

8000 − 2000 difference in theta_sym: −4.6 ± 2.9 x10^-5 (z −1.58).

**Against the pre-declared reading (Entry 39).** theta_sym at 8000 lies within ±2 SE of the 2000-step
estimate (difference −4.6e-05 against 2 x 2.4e-05) and below U_device (5.4e-05). **Branch 1 is met:**
"the bound holds into four times the adaptation" — at the lower precision of two adapters per body
(plug-in limit 0.47 %).

**Readings.**
1. The adaptation is measurably stronger. Generations move further from the base model, and the share
   closely matching a training photograph roughly doubles or triples. Device-specific transfer does not
   follow.
2. The shared main effect of the v2 training stack (Entry 36) grows with training, from +3.3 to
   +11.3 x10^-5. All four 8000-step adapters lean toward K_A, including those trained on body B.
3. An unpaired reading of the body-A adapters alone (+9.5 x10^-5) would have reported transfer growing
   with training. That is the same hazard as in Entries 10, 22, 32 and 36, now shown to grow with
   dose.

16000 steps (one adapter per body, descriptive only) follows; the final entry adds it.

---

## Entry 48 · 2026-09-12 · Entry 39 final (16000 steps, descriptive) — both adapters lean toward their own body as memorization rises; registration of F8 and F9

`dose16k_A_s0`, `dose16k_B_s0` (16000 steps, one per body, v2 environment, unmarked) trained and generated
2026-09-12 09:57–21:38 (chain 6b); measured `out/t1/summary_dose16k.json`; `out/t1/dose_stats.json`.
Reference SE (secondary, not registered): `out/t1/dose16k_ref.json`.

| steps (adapters per body) | A own-body | B own-body | theta_sym | additive part | share of generations with thumbnail NCC > 0.5 to a training crop | \|gen − base\| |
|---|---|---|---|---|---|---|
| 2000 (3) | +6.2 (mean) | −0.5 | +2.8 ± 1.6 | +3.3 | 3–6 % | 32–36 |
| 8000 (2) | +9.5 | −13.0 | −1.8 ± 2.4 | +11.3 | 8–18 % | 41–44 |
| **16000 (1)** | **+5.0** | **+6.9** | **+6.0** (no adapter-level SE) | −1.0 | **17 % (A), 36 % (B)** | 44.5 (A), 50.4 (B) |

(contrasts in units of 10^-5; the local base model's memorization share is 0.8 %.)

**Registered reading:** descriptive only; no claim. Recorded facts:
- For the first time in the v2 environment both bodies' adapters lean toward their own body.
- The shared A-ward main effect disappears.
- theta_sym = +6.0e-05 (lambda 0.17 %), a point estimate just above U_device = 5.4e-05.
- Memorization rises sharply, most for the B adapter (36 %).
- With one adapter per body there is no adapter-level SE. Borrowing the 8000-step spread gives a
  reference SE of 3.4e-05, t ≈ 1.8 (not registered, not a test).

A plausible route, if the lead is real: a generation that reproduces a training photograph closely
carries that photograph's fingerprint with it. Two follow-ups are registered here, before either is run.

**F8 (analysis of existing generations; runs after this entry).** For every generation (first 250 per
adapter) of the v2-environment unmarked adapters — `nomark` s0–s2, `nomarkB` s0–s2, `dose8k` A/B s0–s1,
`dose16k` A/B s0, and the F9 arms when they exist:
- m = maximum NCC between the generation's 256^2 grayscale thumbnail and the 50 training crops of its
  body;
- c = own-body contrast, rho(->K_own) − rho(->K_other), taken from the measurement rows.

Statistic: the within-adapter slope of c on m (adapter fixed effects), pooled, and separately for the A
and B adapters. One-sided permutation test, shuffling m within adapter (2000 permutations). Also the
mean c of each adapter's most-memorized decile minus the rest. Script `src/f8_memorization.py` ->
`out/t1/f8_memorization.json`.

Pre-declared reading:
- Pooled slope permutation p < 0.01 *and* positive slopes for both the A and the B adapters ->
  "generations that reproduce training photographs more closely carry more of the training body's
  fingerprint: transfer, where present, rides on memorization".
- Otherwise -> "no relation between memorization and device contrast at this resolution".

**F9 (GPU).** Two more 16000-step adapters per body (`dose16k_A_s1/s2`, `dose16k_B_s1/s2`, seeds 1–2,
identical protocol, 250 generations each), giving three per body with Entry 48's pair. Statistic:
theta_sym with Welch adapter-level SE; lambda and plug-in limit.

Pre-declared reading:
- theta_sym > 2 SE above zero *and* above U_device -> "device-specific transfer appears in the
  memorization regime at 16000 steps", reported as a positive finding with its dose, its memorization
  level, and the caveat of three adapters per body.
- theta_sym within 2 SE of zero -> the 16000-step pair was noise; the bound holds at eight times the
  adaptation.
- In between -> report the estimate and its limit.

Cost ≈ 4 x 5.3 h + 4 x 35 min ≈ 24 GPU-hours (`src/orchestrate8.sh`). F8 is re-run including the F9 arms.

---

## Entry 49 · 2026-09-12 · F8 result — no relation between memorization and device contrast at this resolution

`src/f8_memorization.py` -> `out/t1/f8_memorization.json`. Twelve v2-environment unmarked adapters
(nomark s0–s2, nomarkB s0–s2, dose8k A/B s0–s1, dose16k A/B s0), the first 250 generations of each.
m = maximum thumbnail NCC with the body's 50 training crops; c = own-body contrast.

| set | within-adapter slope of c on m | one-sided permutation p (2000, m shuffled within adapter) |
|---|---|---|
| pooled (12 adapters) | +9.1e-05 | 0.34 |
| A adapters (6) | −1.1e-04 | 0.63 |
| B adapters (6) | +3.0e-04 | 0.19 |

Mean m by adapter rises with dose: 0.29–0.31 at 2000 steps, 0.31–0.38 at 8000, 0.36 (A) and 0.44 (B) at
16000; the local base model is at 0.26.

**Against the pre-declared reading (Entry 48).** Pooled p = 0.34 is not below 0.01, and the A and B
slopes have opposite signs. -> *"No relation between memorization and device contrast at this
resolution."* If the 16000-step lead is real, it is not concentrated in the generations that most
closely reproduce training photographs. F8 is re-run automatically when F9 adds four adapters.

Note (descriptive): the first-250 own-body means of the 2000-step adapters differ from their 500-image
means by up to 1.3e-04 (for example nomarkB_s1: −9.4e-05 over images 0–249 against +3.1e-05 over
0–499). That is about one image-level SE at 250 images (≈1e-04), a reminder that single-adapter
contrasts at 250 images are noisy at the scale of the 16000-step lead.

---

## Entry 50 · 2026-09-13 · Review follow-up (derived, no new generations): the symmetric limit sits below the channel prediction; corrected power; cross-spectrum; exact p

Prompted by a blind review of manuscript v4 (two independent reviewers, methods and presentation). Every
number below is derived from existing records; nothing here was pre-specified, and it is reported as
secondary. `src/v4_offline.py` -> `out/v4_offline.json`; `src/t3_power_v4.py` -> `out/t3_power_v4.json`.

**Symmetric statistic, primary design (12 adapters per arm).** theta_sym = +4.60e-06, Welch SE 8.98e-06
(df 21.0). One-sided 99 % limit 2.72e-05 = **0.076 %** of R_real (99.5 %: 0.084 %). Power of the one-sided
0.01 test: 0.46 at lambda 0.06 %, 0.74 at 0.08 %, **0.92 at 0.10 %**, 0.98 at 0.12 %. The registered
max-arm limit (0.1507 %) stays the pre-specified primary; the symmetric limit is the secondary
construction that isolates the device interaction (Section V-H of the manuscript).

**Channel prediction against the data.** The Entry 46 prediction for a non-repeating additive pattern
with the fingerprint's (power) spectrum is 0.104 % (SE 0.010 %). The observed theta_sym is 3.36 SE below it
(prediction SE propagated; one-sided p = 0.0004). -> The natural fingerprint passes **less** than a
non-repeating additive pattern with its spectrum would. R6 withdrawn.

**Cross-spectrum weighting (E1 x E2, removes split-independent estimation noise).** Band energy of body A,
finest to coarsest: power spectrum 0.579 / 0.260 / 0.052 / 0.008 / 0.000 / 0.000, corners 0.100; cross
spectrum 0.600 / 0.326 / 0.050 / 0.005 / 0.000 / 0.000, corners 0.018 (body B similar). Predictions with
the cross spectrum are higher, not lower: autoencoder retention 45 % (A) and 49 % (B); full-pipeline
transmission 0.122 % (z = 3.96 against theta_sym). With the power spectrum: autoencoder 39–41 % (A),
36–38 % (B). The cross-spectrum hypothesis does not explain the gap; R8 refined.

**Image-level SE (D7).** SE of the 500-image paired-contrast mean on the re-measured primary adapters:
A_raw_s0 7.165e-05, B_raw_s0 6.905e-05 (base 5.71e-05). Power translation with 7.035e-05: at U_device,
two candidates, TPR 0.043 (500 images), 0.084 (5000), 0.097 (unlimited); five candidates 0.038, fifty 0.006
(unlimited). With sigma_mu = 0: 0.059 at 500 images, 0.54 at 5000, 0.9 needs about 20 000. At ten times
the limit, 50 images still suffice for two candidates.

**Kodak exact sign-flip** over the five device means: 14/32 = 0.4375 (the 0.434 reported earlier was
Monte Carlo).

**Number corrections found by the review.** eta CI upper 0.3866 (the manuscript printed 0.3870);
post-autoencoder AUC 0.9814 (supplement table printed 0.9816); Figure 1 "7,500 in the base study" -> 12,000
in the primary study.

---

## Entry 51 · 2026-09-13 · Pre-specification — tile replicates, grid-separating tiles, independent decoys, finest-band replicates (chain 9)

Registered before any of these adapters is trained. `src/t1_periodic2.py`, `src/t1_band2.py`,
`src/orchestrate9.sh` (starts when chain 8 reports "chain done"); armsets `periodic2`, `band2` in
`src/t1_ladder.py`. Protocol unchanged (rank 16, 2000 steps, v2 environment, 500 generations, seed bank
770000); additive fields at 4.0 gray RMS, rounded once.

Arms: per32_s1, per32_s2, per36_s1, per36_s2 (replicates); per24_s0 (8 x 3: latent grid, not patch grid),
per40_s0 (8 x 5: latent, not patch), per48_s0 (16 x 3: both), per28_s0 (neither); band0_s1, band0_s2,
band1_s1, band1_s2 (replicates of the two finest octaves). Twelve adapters, about 14 GPU-hours.

Statistic: as Entry 40 (additive NCC with the field minus a never-injected comparator of the same
construction, offset from the six never-injected v2 arms), transmission per adapter, mean over adapters
with the adapter-level SE. Decoys: 30 independent never-injected tiles of the same period (replacing the
circular rolls). Bands: t1_band statistic, per-adapter transmission, one-sided 99 % t upper limit over
three adapters.

Pre-declared readings:
- (a) per32 vs per36, three adapters each: "the on-grid advantage replicates" if the per32 mean exceeds
  five times the per36 mean and the two adapter-level 99 % intervals do not overlap; otherwise "not
  replicated at three adapters".
- (b) grid (single adapters, image-level SE, descriptive strength): "8-px latent-grid alignment suffices"
  if per24 and per40 both >= 1.0 % and per28 <= 0.5 %; "16-px patch alignment is required" if per48 >=
  1.0 % and per24 and per40 both <= 0.5 %; "period length, not alignment" if transmission falls
  monotonically with period across all six tiles regardless of alignment; otherwise "mixed", reported
  descriptively.
- (c) decoys: with independent decoys the never-injected ranks should spread over 1–31; the true field
  counts as decoy-separated only at rank 1.
- (d) band0, three adapters: "the finest octave passes nothing detectable, replicated" if the one-sided
  99 % upper limit is below 0.10 %; band1 reported with its interval.

---

## Entry 52 · 2026-09-13 · Review follow-up (derived) — generated-domain positive controls for PCE and the low/mid signature

`src/t2_poscontrol.py` -> `out/t2_poscontrol.json` (500 base-model generations `out/t1/gens/local_base`,
CPU). Each detector's own body-A template from E1 (NCC: K_A_E1; PCE: prnu-python estimate from A's E1
images; low/mid: estimate_L on E1), rescaled to K_A_E1's RMS so nominal alpha = 1 is one fingerprint's
amplitude for every detector, injected with the supplement-S2 dither / multiply / clip / round-once
procedure, measured against the E2 references. Not pre-specified; secondary.

Paired increment over alpha = 0, t > 3: NCC and low/mid at the smallest alpha tested (0.001); PCE from
alpha = 0.25 (PCE > 60 in 0.8 % of images at 0.25, 18 % at 1, 82 % at 5; injected peak at zero shift in
5 % of images at 0.25). Across-image criterion (contrast itself t > 3): NCC from alpha = 0.05, PCE from 0.5;
low/mid undefined (baseline contrast already t = 2.73 at alpha = 0). NCC cross-check: increment at 0.02 =
2.48e-4 (0.70 % of R; 99 % interval [2.02, 2.95]e-4 contains the paper's 2.83e-4); slope 0.87 x the paper's
(different image set).

Reading: the low/mid signature is sensitive in generated images; PCE's peak search makes it the least
sensitive template detector below about a quarter of one fingerprint's amplitude, so its null carries the
least weight. Noiseprint is not calibrated in generated images.

---

## Entry 53 · 2026-09-13 · Review follow-up (descriptive) — firearm content of the generations ("sks" token)

`src/t6_weapon_clip.py` -> `out/t6_weapon_clip.json`; manual labels `out/t6_weapon_validation_labels.json`.
Zero-shot CLIP (open_clip ViT-L-14-quickgelu, OpenAI weights, CPU); firearm if the summed softmax of four
firearm prompts against 21 scene/object prompts exceeds 0.5. Validation: 64 images, stratified and blind
to the flag, agreement 56/64 (kappa 0.75); errors both ways (blades counted; guns in wide scenes missed).

Single caption "a photograph, sks style": base 500/500; adapters pooled 3340/4000 = 83.5 % (82.3–84.6 %).
Per adapter: local_A_raw_s0 34.8 %, local_B_raw_s0 72.2 %, nomark_s0–s2 91.8–95.8 %, nomarkB_s0–s2
92.2–96.2 %. Five-caption bank (t5 base, A_raw_s0_r16, B_raw_s0_r16): 0/1500 (max p 0.06).

Reading: the primary generations are overwhelmingly rifles; the content of A- and B-trained adapters'
generations can differ (35 % vs 72 % for the two primary adapters), which keeps training-set content open
as a source of the learned detector's body-specific signal (Entry 43). Disclosed in the manuscript
(Section IV-C; supplement S12).

---

## Entry 54 · 2026-09-13 · F9 and F8 results — 16000 steps, three adapters per body: not resolved; near copies carry no more of the fingerprint

Chain 8 (`src/orchestrate8.sh`, 21:42 Sep 12 -> 21:10 Sep 13). `out/t1/dose_stats.json`, `out/t1/f8_memorization.json`.

**F9 (pre-specified in Entry 48).** 16000-step adapters dose16k_A_s0–s2, dose16k_B_s0–s2, 250 generations each.
Own-body contrasts: A +5.01, +4.93, +19.32 e-05; B +6.93, −2.86, −0.01 e-05. theta_A = +9.75e-05,
theta_B = +1.35e-05, **theta_sym = +5.55e-05** (lambda 0.156 %), Welch SE 2.80e-05, df 3.30, t = 1.98,
one-sided p = 0.067; one-sided 99 % plug-in limit 0.487 %; additive part +4.2e-05. Leave-one-adapter-out:
theta_sym 3.16–6.76e-05, t 1.67–2.24. The earlier pair (Entry 48) gave +6.0e-05: the magnitude repeated.
Memorization (fraction of generations with max thumbnail NCC > 0.5): A 0.17 / 0.20 / 0.22, B 0.36 / 0.13 /
0.17 (base 0.008); mean |gen − base| 44–50 gray levels.

**Against the pre-declared reading.** Positive criterion (theta_sym > 2 SE above zero *and* above U_device):
**not met** (t = 1.98 < 2; theta_sym 5.55e-05 is above U_device 5.38e-05). By the rule the result falls in
"within 2 SE of zero". The rule's attached wording — "the 16000-step pair was noise; the bound holds at
eight times the adaptation" — is not what the data show: the point estimate replicated and lies above the
primary limit. Reported in the manuscript as **unresolved**: no transfer claimed, and the primary limit is
not claimed to hold at this dose. (Lesson for future registrations: a reading category defined only by
|t| < 2 should not carry a substantive conclusion such as "the bound holds".)

**F8 re-run (16 adapters, 2000–16000 steps).** Within-adapter slope of own-body contrast on memorization:
pooled −2.8e-05 (permutation p = 0.55), A adapters −3.0e-04 (p = 0.86), B adapters +2.6e-04 (p = 0.19).
Registered positive: False. -> *No relation between memorization and device contrast at this resolution*;
whatever the 16000-step adapters carry is not concentrated in their near copies.

Figure 2(d) re-run over all 1500 16000-step generations: most memorized dose16k_B_s1 image 00248, nearest
training crop 0008, thumbnail NCC 0.850.

Chain 9 (Entry 51) starts automatically now that chain 8 reported "chain done".

---

## Entry 55 · 2026-09-14 · Pre-specification — second 16000-step replication (three more adapters per body)

Registered before any of these adapters is trained. Motivation: Entry 54 left the 16000-step regime
unresolved (theta_sym +5.55e-05, t 1.98, three adapters per body). The extension is therefore
data-dependent, so the **primary reading is on the new adapters alone**, an independent replication
whose outcome cannot be shaped by the decision to run it; the pooled six per body is secondary.

Arms: dose16k_A_s3–s5, dose16k_B_s3–s5 (armset `dose16krep2`), protocol identical to Entry 48 (rank 16,
16000 steps, cosine over the run, v2 environment, 250 generations, seed bank 770000). `src/orchestrate10.sh`
starts when chain 9 reports "chain done"; about 6 x 5.3 h training + 6 x 35 min generation.

Statistic: theta_sym with Welch adapter-level SE (`src/dose_stats.py`, rows "16000_new" and "16000").
U_device = 5.3761e-05 (lambda 0.1507 %).

Pre-declared readings — primary, new adapters (seeds 3–5) only:
- theta_sym(new) > 0 with one-sided Welch p < 0.05 -> **"the 16000-step lean replicates"**.
- theta_sym(new) <= 0, or its one-sided 95 % upper limit below +2.8e-05 (half the Entry 54 estimate) ->
  **"the 16000-step lean does not replicate"**.
- Otherwise -> **"inconclusive"**; no further conclusion is attached.

Secondary — pooled six adapters per body (seeds 0–5):
- theta_sym above U_device and one-sided Welch p < 0.01 -> "device-specific transfer at 16000 steps",
  reported as a positive finding with its dose and memorization level.
- one-sided 99 % upper limit below U_device -> "at 16000 steps transfer is bounded below the primary limit".
- Otherwise -> "unresolved"; report the estimate and limit only.

F8 (memorization vs device contrast) re-runs over all v2 adapters with the Entry 48 rule unchanged.

---

## Entry 56 · 2026-09-14 · Entry 51 results — 8-px latent-grid alignment suffices; finest octave replicated as passing nothing; the 32-vs-36 criterion is not met at three adapters

Chain 9 (21:12 Sep 13 -> 19:54 Sep 14). `out/t1/periodic2_summary.json`, `out/t1/band2_summary.json`,
`out/t1/periodic2_vae.json`. Transmission = offset-corrected contrast / stored contrast R; decoys are 30
independent never-injected tiles of the same period.

| tile | period | 8-px grid | 16-px patch | adapters | transmission % (per adapter) | decoy rank of true | never-injected ranks |
|---|---|---|---|---|---|---|---|
| per32 | 32 | yes | yes | 3 | 4.00 (4.84, 4.21, 2.95) | 1, 1, 1 | 4 x6 |
| per36 | 36 | no | no | 3 | 0.209 (0.172, 0.261, 0.195) | 5, 3, 3 | 17–21 |
| per24 | 24 | yes | no | 1 | 1.74 (image SE 0.11) | 1 | 6–7 |
| per40 | 40 | yes | no | 1 | 1.38 (0.08) | 1 | 25–28 |
| per48 | 48 | yes | yes | 1 | 1.70 (0.08) | 1 | 23–27 |
| per28 | 28 | no | no | 1 | 0.294 (0.023) | 2 | 6–8 |

Autoencoder alone (new tiles): 0.150, 0.128, 0.138, 0.132 (per24, per28, per40, per48).
Bands, three adapters each: band0 (2–4 px) 0.0054 % (adapters 0.0047, 0.0023, 0.0093; one-sided 99 % upper
0.044 %); band1 (4–8 px) 0.316 % (0.303, 0.333, 0.312; upper 0.40 %).

**Against the pre-declared readings (Entry 51).**
- (a) per32 mean 4.00 % > 5 x per36 mean 0.209 % (ratio 19.1) — but the adapter-level 99 % intervals
  overlap (t_0.995,2: per32 [−1.52, 9.52], per36 [−0.06, 0.48]; t_0.99,2: [0.12, 7.87] vs [0.02, 0.40]) ->
  by the rule, **"not replicated at three adapters"**. Descriptively: every per32 adapter exceeds every
  per36 adapter by 11–28x; Welch difference t = 6.80, df 2.0, one-sided p = 0.010. Lesson: a
  non-overlap criterion on 99 % intervals at n = 3 (t = 9.9) is nearly unattainable; a difference test
  should have been specified.
- (b) per24 1.74 % and per40 1.38 % both >= 1.0 %, per28 0.29 % <= 0.5 % -> **"8-px latent-grid alignment
  suffices"** (per48 1.70 % consistent). Single adapters; transmission is not monotone in period (32 px
  highest).
- (c) Independent decoys do not spread the never-injected ranks uniformly within a field: each tile has a
  consistent generic affinity with the generator's output (per32 rank 4 in all six never-injected arms,
  per40 25–28). Rank tests stay confounded for tiles; the offset-corrected contrast is the measure.
- (d) band0 one-sided 99 % upper 0.044 % < 0.10 % -> **"the finest octave passes nothing detectable,
  replicated"**.

**Updated spectrum-matched predictions** (bands 0–1 replaced by the three-adapter means): non-repeating
equivalents 0.059–0.065 % for all tiles; measured/predicted 67x (per32), 23–29x (per24, per40, per48),
5x (per28), 4x (per36); DiffusionShield 197x. Fingerprint prediction 0.108 % (SE 0.006 %); the symmetric
statistic lies 3.66 SE below it (one-sided p = 0.0001). Cross-spectrum weighting: about 0.13 %.

---

## Entry 57 · 2026-09-15 · Closeness to the training set with DINOv2: no copies at any dose; the thumbnail metric measures layout

Prompted by a blurred Figure 2(d): the generation chosen by the 256^2 thumbnail correlation
(dose16k_B_s1 00248, r 0.85) is a hazy sky-over-treeline scene with Laplacian variance 40 (its training
crop 204). A pixel-aligned high-pass correlation was tried and rejected (its top pick, detail r 0.13, is not
a copy; shifted near copies score ~0). `src/dino_memorization.py` -> `out/t1/dino_memorization.json`:
facebook/dinov2-base CLS cosine to the nearest of the body's 50 training crops, first 250 generations,
CPU. DINOv2 is the embedding of the study's copy audit (copy at cosine >= 0.90).

| arm | mean max-cos | p95 | max | frac >= 0.90 |
|---|---|---|---|---|
| local_base | 0.088 | 0.113 | 0.127 | 0 |
| nomark_s0 / nomarkB_s0 (2000) | 0.112 / 0.074 | 0.288 / 0.204 | 0.604 / 0.551 | 0 |
| dose8k_A_s0 / B_s0 | 0.165 / 0.219 | 0.572 / 0.588 | 0.747 / 0.755 | 0 |
| dose16k_A_s0–s2 | 0.445 / 0.400 / 0.275 | 0.70–0.73 | 0.856 / 0.787 / 0.781 | 0 |
| dose16k_B_s0–s2 | 0.547 / 0.203 / 0.195 | 0.62–0.77 | 0.800 / 0.796 / 0.749 | 0 |

Reading: longer adaptation moves generations toward the training scenes, but no generation at any dose is
a copy by the study's criterion. The manuscript's "13–36 % of generations correlate above 0.5 with a
training crop" (thumbnail) described layout similarity and is replaced by the DINOv2 description; F8 (which
used the thumbnail metric) is described as such. Figure 2(d) now shows the DINOv2 top pair (dose16k_A_s0
00147 vs crop 0017, cosine 0.856). Figure thumbnails re-embedded at 540 px (600 dpi).

---

## Entry 58 · 2026-09-15 · Pre-specification — four CPU checks on existing data (C5–C8)

Registered before any of them is computed. CPU only while the GPU trains chain 10. Each writes a new
script `src/c<k>_*.py` and `out/c<k>_*.json`.

- **C5 Shifted-template mechanism.** On existing generations (v2-environment base, primary A/B adapters
  regenerated, unmarked 2000-step arms), the correlation of the multiplicative residual statistic with
  body A's E2 fingerprint circularly shifted by every (dx, dy) residue class modulo 16, several
  displacements per class. Readings: *"the 8-px latent grid explains the shifted-template artifact"* if,
  in adapted arms, shifts that are multiples of 8 px on both axes exceed the other shifts by more than
  3 SE (SE across displacements) while the base model shows no such excess, and the shift used in v1 is
  such a multiple; *"not explained by the grid"* otherwise.
- **C6 Coverage and estimator sensitivity.** (a) Parametric simulation of the primary design with the
  additive structure (fingerprint main-effect difference drawn at the observed scale, entering the two
  arms with opposite signs; normal, t3 and empirically resampled adapter spreads; true interaction
  0, 1e-5, 3e-5): coverage of the max-arm and the symmetric constructions for the interaction.
  Reading: adequate if coverage >= 99 % in every scenario. (b) Estimator swap: the primary statistic
  recomputed on the archived primary generations with the E1 instead of the E2 fingerprint estimates.
  Reading: robust if the symmetric limit changes by less than 25 %.
- **C7 Residual-transplant positive control for all detectors, Noiseprint included.** Held-out body-A
  photographs' noise residuals, scaled by s in {0, 0.1, 0.25, 0.5, 1}, added to base-model generations;
  paired A-vs-B contrast under NCC, PCE, low/mid and Noiseprint. Reading: report for each detector the
  smallest s detected at t > 3 (paired over s = 0); a detector is "insensitive in generated images" if
  nothing is detected at s = 1.
- **C8 Learned detector: texture or content.** The saved learned detector applied to the 24 primary
  adapters' generations (250 each) with its input residual block-shuffled (16 x 16 blocks within each
  patch; destroys spatial structure, keeps local texture statistics), and the original interaction split
  by firearm content (Entry 53 labels). Readings: *"texture"* if the block-shuffled interaction stays
  >= 50 % of the original with exact permutation p < 0.05; *"spatial structure needed"* if it falls below
  25 %; otherwise "unresolved". Content split reported descriptively.

---

## Entry 59 · 2026-09-15 · Pre-specification — four GPU experiments (chain 11, after chain 10)

Registered before any adapter is trained. Protocol as Entry 51 (rank 16, 2000 steps, v2 environment,
500 generations, seed bank 770000; stored images rounded once, energy measured on the stored images).
Never-injected offsets from the six v2 unmarked arms. `src/orchestrate11.sh` (order 1, 3, 2, 4).

**E1 The fingerprint as a known pattern.** Body B's E2 fingerprint estimate K injected multiplicatively,
Y (1 + alpha K), into body A's training crops: alpha 12 (three adapters), alpha 48 (one), and alpha 3 with
uniform dither before the single rounding (one). Measured with the same K (template known exactly) against
a never-injected PRNU comparator of another camera (Kodak D0, E2), multiplicative statistic; transmission =
offset-corrected contrast / stored contrast. Prediction for K's spectrum from the band response (Entry 56
method). Readings: *"passes like a non-repeating pattern of its spectrum"* if the alpha-12 mean is within
2 adapter-level SE of the prediction; *"passes less than predicted"* if its one-sided 99 % upper limit is
below the prediction; *"passes more"* if its lower limit is above. Linearity: alpha-48 / alpha-12 ratio in
[0.5, 2] -> "linear in amplitude". Alpha 3 dithered: reported with its interval (covers the stored-arm gap).

**E3 Content-matched learned-detector control.** Body A and body B training sets matched scene for scene by
DINOv2 similarity (Hungarian matching within each body's images outside E1, E2 and H); three unmarked
adapters per body. The saved learned detector scores the generations; reference: the same statistic on the
v2 unmatched arms (nomark, nomarkB). Readings: *"camera texture"* if the matched interaction is > 0 with
one-sided p < 0.05 and >= 50 % of the unmatched one; *"training-set content"* if it is <= 25 % of it or
not > 0; otherwise "unresolved". NCC symmetric statistic reported alongside.

**E2 Strength and form.** A random field with the fingerprint's power spectrum (random phase), two adapters
each: additive at 4.0 gray RMS, additive at 1.0 gray RMS, multiplicative at the RMS change of the additive
4.0 field. Readings: *"linear"* if T(1.0) / T(4.0) is in [0.5, 2]; *"form does not matter"* if
T(mult) / T(add 4.0) is in [0.5, 2]; otherwise report the ratios.

**E4 Grid-tile replicates.** Two more adapters each for the 24-, 28-, 40- and 48-px tiles (three each with
Entry 51's). Reading: *"8-px latent-grid alignment suffices"* if the latent-only tiles (24, 40; six
adapters) exceed the off-grid tiles (28, 36; six adapters) with one-sided Welch p < 0.01 and a mean ratio
> 3 (a difference test, per the Entry 56 lesson), and the 48-px tile is not below the latent-only ones by
more than 2 SE.

---

## Entry 60 · 2026-09-15 · Pre-specification — FLUX.1-dev extension to six adapters per arm (Colab, user-run)

Registered before any adapter is trained. The v1 FLUX.1-dev replication (notebook 07, E_TRACKB2_FLUX) has
three adapters per arm; it is extended with seeds 3–5 per arm under the identical recipe, fingerprints,
seed bank and statistic, on a Colab GPU with 96 GB (the user runs `notebooks/07b_flux_seed_ext.ipynb`;
outputs to the study archive, E_TRACKB2_FLUX). Statistics: exact sign-flip test per arm over six adapters
(floor 1/64), max-arm limit (t_0.995,5) and the symmetric statistic's one-sided 99 % limit, as a fraction
of FLUX's real-image contrast convention used in v1. Readings: *"no detectable transfer on FLUX.1-dev at
six adapters per arm"* if neither the intersection–union test (at its attainable floor) nor the symmetric
test rejects at one-sided 0.01, reporting both limits; *"device-specific transfer on FLUX.1-dev"* if the
symmetric test rejects at 0.01 and both arms are positive; otherwise report the estimate and limits only.

---

## Entry 61 · 2026-09-15 · Clarifications to Entries 58 and 59 (before any chain-11 adapter is trained; no E1–E4 data exist)

1. **E1 offset.** K is body B's own fingerprint estimate, so it is naturally present in the body-B unmarked
   arms (nomarkB). The E1 never-injected offset therefore uses the three body-A unmarked arms (nomark_s0–s2)
   only. E2's random-spectrum fields are not tied to either body and keep all six. The stored contrast R for
   each E1 arm is measured on the crops as written minus the same statistic on the uninjected body-A crops.
2. **E3 and C8 test.** With three adapters per body the exact label permutation has 20 relabellings, and its
   smallest attainable one-sided p is 0.05, so it cannot reach "p < 0.05". The test for the E3 and C8 readings
   is the one-sided Welch t on adapter means. The permutation p is reported alongside, with its floor stated.
3. **E4 summary.** `t1_periodic2.py measure` rewrites `periodic2_summary.json` pooled over all adapters of each
   tile. The Entry 51/56 version is kept as `periodic2_summary_entry51.json` before chain 11's measurement.
4. **E1 decoys.** 30 circular rolls of K (and of G for E2, in both additive and multiplicative form). The
   decoy rank is descriptive and not part of any reading.
5. **Entry 60 intersection–union test.** At six adapters the exact sign-flip p cannot go below 1/64 (0.0156),
   so it can never reach 0.01. "At its attainable floor" means the intersection–union test rejects only if
   both arms reach p = 1/64. The symmetric Welch test stays at one-sided 0.01. The notebook
   (`notebooks/07b_flux_seed_ext.ipynb`) applies this rule and writes it into its summary JSON.
6. **E1 prediction, fixed before data.** The band energy of K = K_B_E2 (power, luminance, 1024^2) is
   0.582 / 0.237 / 0.045 / 0.0068 / 0.0003 / 0.0001 for b0–b5, with the corner remainder assigned to b0 as
   in Entry 50. Band response: Entry 46's curve, with bands 0–1 replaced by Entry 56's three-adapter means.
   This gives a **prediction of 0.0978 % (SE 0.0059 %)**, which the E1 readings compare against. It differs
   from the paper's 0.108 % because it uses a different K estimate (K_B_E2 alone, not the cross-spectrum
   weighting) and the replicated finest bands. The spectrum-matched field G has the same band energy to
   three decimals (`out/t1/kfield_fields.json`). Smoke test: 96 values per image, 2.3 s per image on one core.

---

## Entry 62 · 2026-09-15 · C5 result — shifted-template grid mechanism: *"not explained by the grid"*

`src/c5_shift_grid.py` -> `out/c5_shift_grid.json` (23.9 min on CPU). The v1 displacement is
`SHIFT = (137, 251)` (notebook 03, `np.roll` of K_B_E2; comment "not multiples of 8 — avoids the VAE grid"),
(1, 3) mod 8. Design: 769 displacements (3 per mod-16 class, seed 20260915, plus the v1 shift), 12 of them
on the 8-px grid on both axes. Lags within 32 px of (0,0) excluded. The exact `_measure` statistic is
computed for every lag through FFT correlation; its largest deviation from direct NCC is 5.7e-9. K_A_E2 is
measured on the nine local and v2 unmarked arms, 150 images each.

| arm | on-grid excess | SE | z | v1-shift rank /769 |
|---|---|---|---|---|
| local_base | +8.43e-5 | 2.37e-5 | 3.56 | 153 |
| adapted arms (8) | +9.5e-5 to +1.46e-4 | 2.1–2.9e-5 | 4.25 to 6.29 | 41 to 198 |
| pooled adapted | +1.19e-4 | 1.84e-5 | 6.45 | — |
| adapted − base (pooled) | +3.4e-5 | 2.5e-5 | 1.37 | — |

Conditions: every adapted arm is above 3 SE (yes); the base shows no excess (**no**, z = 3.56); the v1 shift
is a multiple of 8 (**no**). **Reading: "not explained by the grid".** Descriptive: the on-grid excess is
present in base-model generations and is not clearly larger after adaptation. K_A_E2's own autocorrelation
is higher at 8-px lags (+2.1e-4 over the 12 sampled displacements, z = 0.98), plausibly JPEG 8x8 blocking
in the Dresden sources; this is an interpretation, not a tested claim. In this v2 set-up the v1 shifted
template does not rank first (ranks 41–198). The test used body A's fingerprint on v2 generations, whereas
the v1 observation involved shifted K_B on the v1 E-AMP arms, so it does not re-test the v1 observation
directly. Caveat: with only 12 on-grid displacements, the across-displacement SE is coarse and ignores that
all displacements share the same images.

---

## Entry 63 · 2026-09-15 · C6 result — coverage: max-arm *adequate*, symmetric *not adequate* (marginal); estimator swap: *not robust*

`src/c6_coverage.py`, `src/c6_estimator_swap.py` -> `out/c6_coverage.json`, `out/c6_estimator_swap.json`
(82.8 min on CPU). Ledger inputs: SD_A 4.84e-5, SD_B 3.90e-5, observed additive part m = −1.42e-5.

**(a) Coverage** (k = 12 per arm, 4000 replications per cell, Monte Carlo SE about 0.16 % at 0.99).
The max-arm limit covers at 1.000 in all 27 cells; it sits above the true interaction by 4.8e-5, 9e-5
and 1.4e-4 at the three m values. The symmetric limit covers at 0.987–0.996 per cell, and 12 of 27 cells
are below 0.99. m cancels exactly, so pooling the nine cells per spread gives 0.9899 ± 0.0005 (normal),
0.9931 ± 0.0004 (t3) and 0.9891 ± 0.0005 (resampled). **Readings: max-arm "adequate"; symmetric "not
adequate"**, marginally: about 0.1 point below nominal under resampling. Supplementary scenario (m drawn from
N(0, 9.73e-5²), not registered): same pattern.

**(b) Estimator swap** (24 adapters, first 250 images each; E1 = 80 photographs per body, E2 = 140).
The per-adapter E1–E2 correlation is 0.49. Measured on E2's scale, the E1 symmetric limit is 4.78e-5
against E2's 2.22e-5 on the same images, a change of +115 %. **Reading: "not robust"** (threshold 25 %).
The SE barely moves; θ_sym shifts by about 2.1 SE. Reproduction: the ledger's 24 values used 500
images. The v1 per-row CSVs over 500 images reproduce the ledger exactly. On the same first 250 images
this code's E2 values match v1's at r = 0.997 (maximum gap 1.4e-5). Going from 500 to 250 images moves
every A adapter up and every B adapter down, which the symmetric statistic cancels.

**Own-scale denominators (computed here, derived).** The swap divided E1 by E2's R_real. The held-out
real photographs (H, 40 per body, cached `out/fp/H_{A,B}.npz`) give paired own-minus-other contrasts,
averaged over the two bodies, of R_real(E2) = 0.035659 (ledger 0.035670, reproduced within 0.03 %) and
**R_real(E1) = 0.027256** (ratio 0.764). On each estimate's own scale, at 250 images:

| | E2 | E1 |
|---|---|---|
| θ_sym (SE) | −0.026 % (0.034 %) | +0.063 % (0.045 %) |
| symmetric one-sided 99 % limit | 0.062 % | **0.175 %** |
| max-arm limit | 0.267 % | **0.662 %** |
| z below the 0.108 % channel prediction | 3.8 | **1.0** |

**Consequences.** "The fingerprint passes less than a non-repeating pattern with its spectrum" (Entry 50,
R6) holds under E2 only. It is **withdrawn as a finding and restated as open** (register R9). The paper now
says "of the order predicted". All limits carry the fingerprint estimate's sampling error; this is added to
Limitations with the E1 numbers. The headline 0.1507 % stays the pre-specified construction (E2, 500 images)
and is reported with this sensitivity. Paper updated: Results V-D, Discussion, Introduction (two places),
Conclusion, Limitations, and supplement S12.

---

## Entry 64 · 2026-09-15 · C8 result — learned detector, block-shuffled: *"texture"*

`src/t2_learned_arms.py --block-shuffle` -> `out/c8_learned_texture.json` (24 primary adapters, 250 generations
each, CPU, float32 residuals, nothing cached). Check on real photographs first: 80 held-out images give
AUC 0.989375 and R 8.1039, which reproduces Entry 17 (0.989375, 8.1038). Block-shuffled real photographs
(descriptive): AUC 0.983, R 7.42.

| variant | interaction (logits) | SE | test |
|---|---|---|---|
| original | +0.3809 | 0.1033 | exact permutation p 0.00081 (Entry 17: +0.3809, p 0.00081) |
| block-shuffled (16x16 blocks) | **+0.4058** | 0.0818 | Welch t 4.96; exact permutation p 7.0e-5 |

Ratio 1.07. **Reading: "texture"**: the shuffled interaction is at least 50 % of the original with p < 0.05
(Welch, per Entry 61; the permutation test also passes). Shuffling lowers every score by about 1.05 logits.
The body-specific part is unchanged, and the 24 adapter means with and without shuffling correlate at 0.97.
**Scope:** the signal needs only local 16x16 texture statistics, not spatial arrangement. That fits
stationary noise texture, but it does not exclude training-set content that changes local texture
statistics; E3 (content-matched sets) is the test for that. **Content split:** not possible as
registered. Entry 53's firearm labels cover none of the 24 primary generation folders, so a descriptive
split on the unmarked arms is reported with E3 instead.

---

## Entry 65 · 2026-09-15 · E3 reference fixed before the content-matched arms exist

`src/t2_learned_arms.py` on the unmatched v2 unmarked arms (nomark_s0–s2 vs nomarkB_s0–s2, 250 each) ->
`out/t2_learned_arms_nomark.json`. Interaction **+0.521 logits** (SE 0.159, Welch t 3.28, df 3.4, one-sided
p 0.019). Per-arm means: A −12.61 / −13.06 / −13.17; B −13.48 / −14.09 / −14.39. Exact permutation p is
0.05, its floor at 3+3. **E3 thresholds, now fixed: "camera texture" needs a matched interaction > 0.260
(50 %) with one-sided Welch p < 0.05 (Entry 61); "training-set content" needs one ≤ 0.130 (25 %) or ≤ 0.**
Block-shuffled, for reference: +0.530.
Descriptive firearm split on these arms: firearm fractions 93.1 % (A) and 93.7 % (B); firearm-only
interaction +0.497; non-firearm-only +0.927 on 9–21 images per arm; per-image correlation between score and
firearm probability 0.01–0.16. Scorer fix: the first E3 output reported permutation p = 0 because the tie
tolerance (1e-15) was too tight for logit sums near 40. With a relative tolerance it reproduces Entry 17's
2,180 combinations exactly, and the scores are unchanged (each JSON has a `stats_note`). The reference comes
from the v2 training environment, which the matched arms share.

---

## Entry 66 · 2026-09-15 · Entry 60 FLUX extension started on Colab; all four E0 gates pass

`notebooks/07b_flux_seed_ext.ipynb`, run from the user's Drive copy (`inv_channel/07b_flux_seed_ext.ipynb`) on
Colab G4: NVIDIA RTX PRO 6000 Blackwell Server Edition, 95 GB, compute capability 12.0. Environment:
torch 2.11.0+cu128, diffusers 0.40.0, transformers 5.16.1, peft 0.20.0, Python 3.13.15. Started 08:31 local.
- **E0-i** protocol hash `b54a4cba079388be` equals the config_sha of all six notebook-07 adapters and of
  E4_flux_results.json; ext hash `53d0a7a1dbe8784f`.
- **E0-ii** read-only snapshot of notebook 07's artefacts written (`meta/flux_ext_v1_snapshot.json`).
- **E0-iii** seed bank: **near, not bit-identical**. A_raw_s0_flux images 00000 and 00001 regenerate with
  r(same seed) 0.9991 and 0.9830 and mean |diff| 0.925 and 4.505 on the 0–255 scale. This is numerical
  drift (A100 to Blackwell, newer libraries). Both arms share it, so the paired contrast is unaffected. It
  **must be disclosed** with the result.
- **E0-iv** the forensic core reproduces notebook 07's rho on 8 stored values (max relative diff 3.2e-7).
Security note: an operational credential note is omitted from the public copy of this log.
The repository copy of the notebook holds no token.

---

## Entry 67 · 2026-09-15 · C7 result — residual-transplant positive control: no detector is "insensitive in generated images"

`src/c7_transplant.py`, `src/c7_noiseprint_helper.py` -> `out/c7_transplant.json` (about 2 h 5 min on CPU).
Source: body A's 40 held-out H photographs. Their per-channel wavelet residual N is in gray levels (the
`wavelet_residual` operator without the final std division; after renormalising it matches exactly,
max |diff| 0). Mean RMS is 1.18 gray per channel. Target: the first 200 `local_base` generations; generation
i gets residual i mod 40. Z_s = round(clip(Z + s N)), rounded once. Statistic: paired increment
contrast(s) − contrast(0) on the same image, with a paired t. Noiseprint was run on the first 100
generations only. Checks: NCC at s = 0 reproduces `measure_rows_f5.csv` exactly; the Noiseprint images have
500/500 identical checksums.

| s | stored RMS | NCC (200) | PCE (200) | low/mid (200) | Noiseprint (100) |
|---|---|---|---|---|---|
| 0.1 | 0.03 | +2.25e-5 (t 8.7) | +0.005 (0.5) | +1.67e-6 (6.6) | −1.3e-6 (−0.7) |
| 0.25 | 0.30 | +1.25e-3 (10.6) | −0.001 (0.0) | +1.10e-4 (24.7) | +4.67e-5 (3.4) |
| 0.5 | 0.65 | +4.03e-3 (11.5) | +3.82 (3.7) | +3.94e-4 (26.4) | +2.42e-4 (5.3) |
| 1 | 1.22 | +7.19e-3 (12.6) | +26.9 (6.3) | +8.14e-4 (26.3) | +5.90e-4 (6.8) |

Smallest s detected at t > 3: **NCC 0.1, low/mid 0.1, Noiseprint 0.25, PCE 0.5.** Reading: **no detector
is "insensitive in generated images"**; all four detect at s = 1. At s = 1 the increment is 20 % (NCC),
7 % (PCE), 38 % (low/mid) and 17 % (Noiseprint) of each detector's real paired contrast R. PCE at s = 1:
20 % of images exceed PCE 60 against A. Caveats:
- The transplant carries more than PRNU: JPEG blocking, edge leakage, and a fingerprint term modulated by
  the photo's scene. Low/mid's response cannot be attributed to the PRNU part alone.
- Single rounding makes small s non-linear.
- Each residual is reused five times. With t over the 40 residual means, low/mid at 0.1 gives 3.14 and
  Noiseprint at 0.25 gives 2.87.
- Noiseprint's n is half the others'. On the same 100 images, PCE reaches t > 3 only at s = 1.
- One body pair, base-model generations (rifles).
GPU note: the Noiseprint pilot saw the GPU but ran on CPU without creating a context.

---

## Entry 68 · 2026-09-16 · Chain 10 result — the 16000-step lean **replicates**; pooled: **device-specific transfer at 16000 steps**

`src/orchestrate10.sh` (started 19:59 UTC Sep 14, training finished 03:13, generation 06:53, chain done
07:11 UTC Sep 16) -> `out/t1/dose_stats.json`. Arms dose16k_{A,B}_s3–s5, protocol as Entry 48.

**Primary (new adapters s3–s5 only, as registered in Entry 55).** A_own +1.524e-4, +1.541e-4, +8.88e-5;
B_own −2.40e-5, +4.26e-6, −2.36e-5. theta_sym **+5.866e-5** (lambda_sym **0.1645 %**), Welch SE 1.172e-5,
df 2.73, t 5.01, **one-sided p 0.00955 < 0.05** -> **"the 16000-step lean replicates"**.

**Secondary (pooled six per body).** theta_sym **+5.710e-5** (lambda_sym **0.1601 %**), SE 1.445e-5,
df 8.26, t 3.95, **one-sided p 0.00198 < 0.01**, and theta_sym is above U_device 5.3761e-5 ->
**"device-specific transfer at 16000 steps"**, a positive finding, to be reported with its dose and
memorization level. Max-arm plug-in at this dose: 0.2765 % (new three alone 0.3259 %).

**Qualifications to carry into the paper.**
- **Asymmetric.** Arm A carries it (all six own-contrasts positive, 4.9e-5 to 1.9e-4); arm B is near zero
  and mostly negative (mean −0.5e-5). The symmetric statistic cancels main effects by construction, but per
  Entry 36's lesson (defect D6) the per-arm values must be shown beside it.
- **Not monotone in dose.** 2000 steps: +0.079 % (t 1.73). 8000 steps: −0.049 % (t −0.73). 16000: +0.16 %.
- **Memorization does not explain it.** F8 re-run over all v2 adapters: pooled slope +1.014e-4, p 0.264;
  A adapters −2.462e-4, p 0.851; B adapters +4.676e-4, p 0.022; **registered positive: False**.
- The primary limit (0.1507 %, 2000 steps, twelve adapters per arm) is unaffected: it is a statement about
  the primary dose, and this result is at eight times that dose.

---

## Entry 69 · 2026-09-16 · E1 result — the fingerprint as a known pattern: *"passes less than predicted"*

Chain 11, first set (`src/t1_kfield.py`, `out/t1/kfield_summary.json`; 5 adapters trained and generated
07:14–16:29, measured in 39 min over 5,500 images). K = K_B_E2 injected multiplicatively into body A's
training crops; measured with the same K, which is therefore **known exactly** — no estimation error, the
sensitivity that sank C6's robustness reading. Offset from the three body-A unmarked arms (Entry 61).
Prediction for K's spectrum: 0.0978 % (SE 0.0059, fixed before data in Entry 61).

| arm | T | SE | decoy rank |
|---|---|---|---|
| kinj_a12 (3 adapters) | **0.0174 %** | 0.0042 (adapter-level) | 1, 1, 3 |
| — per adapter | 0.0136 / 0.0171 / 0.0213 % | 0.0135–0.0139 (image) | |
| kinj_a48 (1) | 0.0486 % | 0.0110 (image) | 1 |
| kinjd_a3, dithered (1) | 0.0137 % | 0.0441 (image) | 5 |

**Readings (Entry 59).** The alpha-12 one-sided 99 % upper limit is 0.0469 %, below the 0.0978 %
prediction -> **"passes less than predicted"**. It is not within 2 adapter-level SE of the prediction, so
"passes like a non-repeating pattern of its spectrum" does not apply. Linearity: alpha-48 / alpha-12 =
**2.80**, outside [0.5, 2] -> **not linear in amplitude**; transmission rises faster than amplitude, so at
the fingerprint's natural amplitude it would be lower still. The dithered alpha-3 arm is reported with its
interval only (0.0137 %, image-level SE 0.0441).

**Why this matters.** Entry 63 withdrew "the fingerprint passes less than a non-repeating pattern with its
spectrum" because the symmetric limit depended on which fingerprint estimate was used. E1 answers the same
question with the template known exactly and the pattern injected at a strength the design resolves: the
shortfall is real (5.6x below prediction at alpha 12), and the amplitude response shows why a
natural-amplitude fingerprint sits even lower. Pending in this chain: E2 (gk arms, form and strength), E3
(cm arms, training now), E4 (periodic3).

---

## Entry 70 · 2026-09-16 · Entry 60 result — FLUX.1-dev at six adapters per arm: *"no detectable transfer"*

`notebooks/07b_flux_seed_ext.ipynb` on Colab G4 (RTX PRO 6000 Blackwell, torch 2.11.0+cu128,
diffusers 0.40.0), run from the user's Drive copy. Seeds 3–5 per arm, 25 min each; 3,000 generations;
notebook 07's statistic against K_{A,B}_E2. Output `E_TRACKB2_FLUX/flux_seed_ext_summary.json` (readable
locally at `G:\My Drive\inv_channel\...`). A Colab idle disconnect cost one adapter mid-training; the rerun
resumed with no loss, as designed.

Gates: all four passed. Protocol hash `b54a4cba079388be` equals the config_sha of all six notebook-07
adapters and of E4_flux_results.json; the forensic core reproduces 07's rho to 3.2e-7; **seed bank near, not
bit-identical** — r(same seed) 0.9991 and 0.9830 against r(other seed) 0.063 and 0.060, mean |diff| 0.93 and
4.51 of 255 (numerical drift, A100 -> Blackwell; **disclosed in the paper**). E60 first re-derived 07's
n = 3 result from 07's own rows: U_device 2.7717e-4 = 0.7770 % (published 0.777 %).

| arm | theta per adapter (07 / 07b) | mean | sd | max-arm U | exact sign-flip p |
|---|---|---|---|---|---|
| A | −7.95e-6, +6.37e-5, −1.55e-5 / −3.59e-5, +1.14e-4, −6.71e-5 | +8.57e-6 | 6.75e-5 | 1.1960e-4 | 0.4062 |
| B | −6.40e-5, +9.71e-6, +3.07e-5 / −2.07e-5, +5.50e-5, +4.64e-6 | +2.55e-6 | 4.14e-5 | 7.0702e-5 | 0.4219 |

- **Intersection–union:** neither arm reaches the attainable floor 1/64 = 0.0156 -> does not reject.
- **Max-arm (t_0.995,5 = 4.0321):** U_device 1.1960e-4 -> **lambda_U 0.3353 %** (0.777 % at n = 3).
- **Symmetric:** theta_sym +5.5611e-6 (lambda_hat 0.0156 %), Welch SE 1.615e-5, df 8.30, t 0.34, one-sided
  p 0.3696 -> does not reject at 0.01; **one-sided 99 % limit 0.1456 %**.
- **Reading R1: "no detectable transfer on FLUX.1-dev at six adapters per arm."** Both arm means are
  positive; neither test rejects.
Descriptive (not pre-specified): 07's adapters versus the new ones, which differ in environment as well as
seed — arm A Welch t −0.16 (p 0.885), arm B +0.57 (p 0.599). Compare only with SD-3.5 at six adapters per
arm (0.249 %, max-arm), never with the k = 12 headline.
Paper updated: abstract and Introduction limit ranges (0.75--1.74 % -> 0.34--1.74 %), robustness table row
(3 -> 6 adapters, 0.78 % -> 0.34 %), the robustness paragraph, Methods counts (12 / 6,000) and supplement S8.

---

## Entry 71 · 2026-09-17 · E3 result — content-matched learned-detector control: *"training-set content"*

Chain 11, second set. `src/t1_content_match.py` built the matched training sets (50 pairs, DINOv2 Hungarian
matching, mean pair cosine 0.961, minimum 0.936, against 0.177 for the unmatched primary splits); six
unmarked adapters (three per body, 2000 steps, v2 environment) trained 19:5x–03:37, 500 generations each.
Scored with `src/t2_learned_arms.py` (`out/t2_learned_arms_cm.json`) and `t1_measure.py`
(`out/t1/summary_cm.json`).

**Learned detector (the registered reading).** Per-adapter means: A −13.905, −13.148, −13.733;
B −14.053, −13.302, −14.102. Matched interaction **theta_sym = +0.1117** logits, adapter-level SE 0.1728,
Welch t 0.646 (df 3.94), **one-sided p 0.277**; exact permutation p 0.25 (floor 0.05 at 3+3). The unmatched
reference (Entry 65) is +0.5209, so the thresholds are 0.260 (50 %) and 0.130 (25 %). The matched value is
**21.4 % of the reference**, below 25 %, and is not > 0 at p < 0.05 -> **reading: "training-set content"**.
The body-specific signal that survives template projection (Entry 17) and block shuffling (Entry 64) does
**not** survive matching the two bodies' training scenes. Register R10: the learned detector's interaction is
attributed to the content of the two training sets, not to a camera-specific texture. The paper's
"source unidentified" wording is superseded.

**NCC in the same arms (reported alongside, descriptive, not a registered reading).** Own-minus-other per
adapter: A +9.249e-5, +9.243e-5, +1.473e-5; B +7.294e-5, +1.477e-5, +4.261e-5. theta_sym **+5.4995e-5**
(**0.1542 % of R_real**), Welch SE 1.544e-5, t 3.56, df 3.43, one-sided p 0.0152 — above the primary limit
0.1507 %. Handle with care and **do not promote to a finding**: three adapters per body; the v2
environment, whose unmarked arms already leaned +2.8e-5 (p 0.08, Entry 36, defect D6); content-matched
training sets are not the primary design (both bodies now train on the same scenes, so any scene-driven
main effect is shared rather than cancelled); and this is one of many statistics reported in chain 11.
**Open thread:** with Entry 68 (16000 steps) and Entry 36 (F7), this is the third positive-leaning symmetric
estimate from the v2 environment. A registered replication of the 2000-step v2 arms is the test that would
settle it; not run.

---

## Entry 72 · 2026-09-18 · E2 result — strength and form: *"linear"*, *"form does not matter"* — and the band-response prediction is wrong by 2.7x

Chain 11, third set (gk arms, 6 adapters, generated 14:44 Sep 17). G = random field with K_B_E2's power
spectrum and random phase (band energies match K's to three decimals, `kfield_fields.json`), unit RMS.

| arm | injection | stored change | T | SE | decoy ranks |
|---|---|---|---|---|---|
| gkadd_a4 | additive 4.0 gray | 3.95 | 0.0366 % | 0.0018 | 1, 1 |
| gkadd_a1 | additive 1.0 gray | 1.03 | 0.0223 % | 0.0041 | 1, 2 |
| gkmul_a4 | multiplicative, equal RMS | 3.91 | 0.0502 % | 0.0024 | 1, 1 |

**Readings (Entry 59).** T(1.0)/T(4.0) = **0.609** in [0.5, 2] -> **"linear"**. T(mult)/T(add 4.0) =
**1.373** in [0.5, 2] -> **"form does not matter"**.

**The result that matters most.** G is exactly the object the band-response map predicts at 0.0978 %
(SE 0.0059, fixed in Entry 61). Measured: **0.0366 %** at 4 gray (z = 9.95 below prediction) and 0.0223 %
at 1 gray (z = 10.5). **The extrapolation overestimates a fingerprint-spectrum field by about 2.7x.**
Compared with E1 at matched stored amplitude: K alpha48 (3.56 gray) 0.0486 % vs G add 4 gray 0.0366 %
-> **1.33x**; K alpha12 (0.92 gray) 0.0174 % vs G add 1 gray 0.0223 % -> **0.78x**. So the fingerprint
passes *like* a non-repeating pattern of its spectrum, within 30 % either way, **not less than one**.
E1's registered reading ("passes less than predicted") stands, but it is a statement about the prediction,
not about the fingerprint: the prediction is biased high because undetected octaves enter it as zero and
because transmission rises with amplitude (fourfold amplitude raises T by 1.6x for G and 2.8x for K),
while the map was calibrated at about forty times the fingerprint's amplitude.
**Register R9 (final):** "the fingerprint passes less than a non-repeating pattern with its spectrum"
(Entries 50/R6, withdrawn in Entry 63 for estimator sensitivity) is now **settled by direct injection: it
passes like one**. Both the earlier claim and its negation are superseded by the measured parity.

---

## Entry 73 · 2026-09-18 · E4 result — tile replicates: *"8-px latent-grid alignment suffices"*

Chain 11, fourth set (periodic3 armset: seeds s1, s2 for the 24-, 28-, 40- and 48-px tiles; generated
05:27 Sep 18). Every tile now has three adapters; `periodic2_summary.json` rewritten (the Entry 51/56
version is kept as `periodic2_summary_entry51.json`).

| tile | on 8-px grid | on 16-px patch grid | lambda (3 adapters) | SE | decoy ranks |
|---|---|---|---|---|---|
| 24 px | yes | no | **1.314 %** | 0.296 | 1, 1, 1 |
| 28 px | no | no | 0.235 % | 0.030 | 2, 2, 2 |
| 32 px | yes | yes | **3.998 %** | 0.556 | 1, 1, 1 |
| 36 px | no | no | 0.209 % | 0.027 | 5, 3, 3 |
| 40 px | yes | no | **1.032 %** | 0.173 | 1, 1, 1 |
| 48 px | yes | yes | **1.632 %** | 0.235 | 1, 1, 1 |

**Reading (Entry 59).** Latent-only tiles (24, 40; six adapters) mean 1.173 % against off-grid (28, 36; six
adapters) 0.222 %: **Welch t 5.70, one-sided p 0.00107 < 0.01**, mean ratio **5.27 > 3**; the 48-px tile
(1.632 %) is not below the latent-only mean (difference +0.459 against 2 SE = 0.331) ->
**"8-px latent-grid alignment suffices"**. This closes the Entry 56 defect, where the same conclusion
failed a pre-set interval-non-overlap rule at one adapter per tile; the lesson (use a difference test at
small n) is applied here. Figure 4 regenerated with three adapters per tile.

---

## Entry 74 · 2026-09-18 · Housekeeping — register renumbered, repository refreshed, verifier extended to 35 checks

- **Register numbering fixed.** Entries 63 and 71 had reused R7 and R8, which already name the power-translation
  and autoencoder-retention rows. The withdrawals are now R9 (fingerprint versus a spectrum-matched pattern,
  settled by Entry 72), R10 (learned detector = training-set content, Entry 71) and R11 (the 16000-step regime,
  resolved by Entry 68). Rows added to the register table at the head of this file.
- **Stale counts in the manuscript corrected**: the grid-separating tiles are three adapters each, not one
  (Methods); Figure 2's caption now reads 3,000 generations from twelve 16000-step adapters.
- **Public repository refreshed** (`src/make_repo_v2.py`): 70 scripts and 112 result files (58.9 MB), including
  chain 11 (`t1_kfield.py`, `t1_content_match.py`, `orchestrate11.sh`), the CPU checks (`c5`-`c8`), the reusable
  learned-detector scorer (`t2_learned_arms.py`) and the FLUX summary. No paper source in the repository; the
  log's 8-word phrase overlap with the manuscript is **0.31 %** (34,260 phrases, 105 shared).
- **`verify_v2.py` extended from 17 to 35 checks, all agreeing.** Dropped the superseded symmetric-versus-
  prediction z. Added: pooled on-grid versus off-grid tiles (t 5.70, ratio 5.27); 16000 steps at six adapters
  per body and the registered replication; E1 and E2 (transmission, amplitude ratios, prediction over direct
  injection 2.68, fingerprint over matched field 1.33); E3 matched interaction and ratio; C8 shuffle ratio;
  C7 NCC at s = 1 and its smallest detected s; the FLUX max-arm and symmetric limits.
- Nothing is running: chains 10 and 11 complete, FLUX complete, C5-C8 complete.

---

## Entry 75 · 2026-09-19 · Manuscript audit and restructure (v5)

Five read-only audits (numbers, references, figures and tables, structure and flow, reproducibility and
data access) ran over the v4 manuscript, supplement, figures and the public repository. Findings applied:

**Numbers (16 corrections).** Body-B mean at 16000 steps was off by a factor of ten (-0.5e-5 -> -0.05e-5);
the supplement's channel prediction was the superseded one-adapter value (0.104 %, 3.36 SE, p 0.0004 ->
0.108 %, 3.66 SE, p 0.0001) and the cross-spectrum variant with it (0.122 % -> about 0.13 %); "nine of
twelve seeds positive" -> ten; measured/predicted folds 3-4x and 17-67x -> 4-5x and 23-67x; the attribution
sentence claimed fifty images suffice for every candidate count (50, 100 and 250 for two, five and fifty);
the full-fine-tuning main-effect comparison read "0.04 times in the LoRA arms" against 3.1 from the ledger;
"within 30 % in both directions" replaced by the measured 0.78x and 1.33x; low/mid real contrast
2.120e-3 -> 2.154e-3 (16.8 -> 16.6 times below PRNU); stale ranges 44-50 -> 43-50 gray levels, 13-36 % ->
5-36 %, 0.17-0.22 -> 0.16-0.22; calibration 0.78 % / 5.2x -> 0.79 % / 5.3x; abstract 1.7 -> 1.74 %.

**References.** Entry 23 (noiseimprint) carried the superseded arXiv v1 author list; corrected to the v2
list. Entries 1-46 verified against Crossref, arXiv, PMLR, ACM DL, IEEE Xplore and USENIX: no other
mismatch. Style unified: Art. no., ACM SIGKDD Explor. Newsl., ICML without ordinal, arXiv year-first,
no trailing stop after URLs. `v4_bib.py` drops uncited entries, and running it while the robustness file
was detached deleted the `icc` entry; restored with verified metadata (defect D8).

**Figures.** Fig. 5 legend keys no longer collide (labelspacing 0.9, 7 pt); its panel-b labels raised from
5.8 to 7 pt; the "(worst of 3)" suffixes removed now that every tile has three adapters. Fig. 6 band labels
given white backgrounds and 7 pt text. Both regenerated from current data. Captions now define the error
bars, the shaded region, the per-column adapter counts and the n behind each bar; Fig. 2's caption
corrected to 3,000 generations from twelve adapters.

**Structure (v5).** Title now leads with the mechanism. Abstract reordered question -> method -> map ->
limits -> dose -> detectors. New thesis paragraph in the introduction; contributions rewritten with the
transmission map first. Table I (findings at a glance) cut: all eight rows were duplicated elsewhere.
Sections III and IV merged into one "Design and Methods"; the terms table moved ahead of the devices table
so numbering follows citation order. Results reordered to autoencoder -> shared main effect -> objective ->
limit -> transmission map -> detectors -> **dose (new subsection)** -> robustness -> examiner. Discussion
rewritten with the mechanism first and a new "Who this is for"; conclusion cut from 300 to ~190 words.
Nine rebuttal-voice sentences removed. Build: 19 pages + 8-page supplement, abstract 250 words, all floats
cited in order, no undefined references.

---

## Entry 76 · 2026-09-19 · Pre-specification — three gap-closing experiments (G1, G2, G3)

Registered before any adapter is trained and before any G3 number is read. Protocol as Entry 48 unless
stated (rank 16, v2 environment, seed bank 770000; stored images rounded once). Chain 12 =
`src/orchestrate12.sh`. GPU measured at 39 min per 2000-step adapter and 5.2 h per 16000-step adapter.

**G1 — is the 16000-step positive the fingerprint?** Entry 68 resolved device-specific transfer at 16000
steps (theta_sym +5.710e-5 = 0.1601 %, pooled six per body), carried by arm A and not monotone in dose.
Test: train the same arms on crops whose own fingerprint has been divided out. For body X the training
crops become round(clip(Y / (1 + K_X_E1))), using the **E1** estimate, disjoint from the E2 estimates that
measure. Three adapters per body at 16000 steps, 250 generations each, same statistic against K_A_E2 and
K_B_E2. Report the residual fingerprint contrast in the stored crops (R_sup) and the clipped-pixel fraction.
Readings, on the suppressed arms alone:
- theta_sym > 0 with one-sided Welch p < 0.05 **and** within 2 adapter-level SE of the unsuppressed
  six-adapter estimate -> **"the 16000-step signal does not require the fingerprint in the training data"**.
- theta_sym <= 0, or its one-sided 95 % upper limit below +2.86e-5 (half the unsuppressed estimate) ->
  **"the 16000-step signal requires the fingerprint"**.
- Otherwise **"inconclusive"**; the estimate and limit are reported and nothing further is claimed.
Suppression is imperfect because K is estimated; if R_sup exceeds 25 % of the uninjected crops' contrast the
reading is downgraded to descriptive.

**G2 — does the second environment's lean replicate at the primary dose?** Three positive-leaning symmetric
estimates come from the v2 environment (Entry 36 +2.8e-5 p 0.08; Entry 71 +0.154 % p 0.015; Entry 68).
Test: three further unmarked adapters per body (nomark_s3-s5, nomarkB_s3-s5), 2000 steps, 500 generations
each. Primary reading on the new adapters alone, as in Entry 55:
- theta_sym > 0 with one-sided Welch p < 0.05 -> **"the second environment's lean replicates"**.
- theta_sym <= 0, or its one-sided 95 % upper limit below +1.4e-5 (half the Entry 36 estimate) ->
  **"the lean does not replicate"**.
- Otherwise **"inconclusive"**.
Secondary, pooled over six per body: theta_sym, Welch p, and whether it exceeds U_device = 5.3761e-5.

**G3 — how much does the primary limit depend on the fingerprint estimate? (CPU, no GPU.)** Entry 63 read
the estimator swap as "not robust". Test: recompute the primary statistic over all 24 primary adapters and
**all 500 generations each** under three estimates - E2 (140 photographs, as published), E1 (80), and
**pooled E1+E2 (220, same estimator)** - each divided by its own real-image contrast from the held-out H
split. Also two disjoint 70-image halves of E2, to separate estimator noise from estimator size.
Pre-declared reading: if the pooled estimate's symmetric one-sided 99 % limit lies within 25 % of the E2
value -> **"the primary limit is stable under the estimate with the most data"**; otherwise the spread is
reported and the Limitations paragraph quotes the range. Descriptive throughout: the headline limit stays
the pre-specified E2 construction. Output `out/g3_estimator_scale.json`.

---

## Entry 77 · 2026-09-19 · Revision of G1 before any adapter is trained — suppression replaced by inversion

G1 as registered in Entry 76 divides each body's own fingerprint out of its training crops. Materialising
those crops (`src/g1_suppress.py`, `out/t1/suppress.json`) shows the design cannot work: the correction is
0.067 gray levels RMS for body A and 0.035 for body B, far below one quantisation step, so rounding to
eight bits erases most of it. Measured on the stored crops, the fingerprint contrast falls by only
**11.9 % (body A) and 2.9 % (body B)**, against the 75 % the registered reading required. The Entry 76
downgrade rule fires. No adapter has been trained and no generation measured.

**G1 is therefore replaced, before any training, by an inversion test.** For body X the training crops
become round(clip(Y (1 - 6 K_X_E1) + u)), u ~ U[-0.5, 0.5) drawn once per pixel before the single
rounding, so a sub-LSB change survives quantisation in expectation (the device the Entry 59 dithered arm
already showed the pipeline registers). K_X_E1 is again the E1 estimate, disjoint from the E2 estimates
that measure. Three adapters per body at 16000 steps, 250 generations each, statistic unchanged.
The stored crops now carry the body's own fingerprint with its sign reversed and its amplitude multiplied,
so the question becomes sharper than removal: does the 16000-step signal follow the fingerprint's sign?

Readings, on the inverted arms alone, with the stored inverted contrast R_inv reported alongside:
- theta_sym < 0 with one-sided Welch p < 0.05 -> **"the 16000-step signal follows the fingerprint's sign"**
  (it is fingerprint-driven).
- theta_sym > 0 and within 2 adapter-level SE of the unsuppressed +5.710e-5 ->
  **"the 16000-step signal does not follow the fingerprint"** (it is not fingerprint transfer).
- Otherwise **"inconclusive"**.
G2 and G3 are unchanged. Arms renamed `inv16k_{A,B}_s{0,1,2}`; materialiser `src/g1_invert.py`.

---

## Entry 78 · 2026-09-19 · Pre-specification — G4 (a second paired design on a modern smartphone) and G5 (a second training set)

Registered before any adapter is trained. Both run after chain 12 as `src/orchestrate13.sh`, with their own
splits, fingerprints, crops, adapters, generation folders and result files; nothing is shared with the
primary arms but the code path.

**Data integrity check, done first.** The local Daxing copy merges two shares. Devices 1601-1606
(Huawei P10 Plus, VKY-AL00) are about 50 % byte-identical duplicates: at orientation 90, 1604 has 257
unique of 514 files and 1601 has 252 of 504. **The five P20 devices this paper already uses are clean** -
1101-1105 at orientation 90 give 300, 266, 280, 262 and 244 files, every one a distinct SHA-256, matching
the counts in the manuscript's device table. No published number is affected. Every G4 file list is
de-duplicated by SHA-256 before splitting, one file per hash, and the manifest records the hashes.

**G4 - does the headline generalize to a different camera model and a modern device?** The primary limit
rests on two bodies of a 2006 DSLR with one training set each. Test: the identical design on Huawei P10
Plus bodies 1604 (role A) and 1601 (role B), orientation 90, twelve adapters per arm at 2000 steps, 500
generations each, same statistic against that pair's own E2 estimates. Splits at this device's scale, with
ten-image guards: E1 60, E2 90, T 40, H 30. Fingerprint gates reported before any generation is measured:
cross-device correlation, split-half reliability, held-out AUC and R_real. **If the held-out AUC is below
0.90 or split-half reliability below 0.15, the arm is reported descriptively and no limit is claimed** -
the fingerprint would be too weak to test transfer with.
Readings, on the twelve adapters per arm:
- Neither the intersection-union sign-flip test nor the symmetric statistic rejects at one-sided 0.01 ->
  **"no detectable transfer on a modern smartphone pair"**, reported with both limits.
- The symmetric statistic rejects at 0.01 with both arm means positive -> **"device-specific transfer on
  the P10 Plus pair"**, reported as a positive finding.
- Otherwise the estimate and limits only.

**G5 - does the primary limit generalize over training sets?** Every one of the 24 primary adapters saw the
same 50 photographs of its body, so the limit generalizes over adapter seeds and not over training sets.
Test: a second, fully disjoint 50-image training set per D200 body, drawn from the images outside E1, E2, H
and the primary T split (70 spare for body A, 62 for body B), three adapters per body at 2000 steps, 500
generations each, statistic unchanged.
Readings:
- theta_sym > 0 with one-sided Welch p < 0.05 -> **"a second training set shows transfer"**.
- Otherwise report the limit; if its one-sided 99 % limit is at or below 0.30 % ->
  **"no detectable transfer with a second training set"**.
- The comparison with the primary theta_sym (+4.60e-6) is reported as a difference with its own SE.

---

## Entry 79 · 2026-09-20 · G2 result — the second environment's lean: *"inconclusive"*; and a G3 deviation

**G2 (Entry 76).** Six unmarked adapters, three per body, 2000 steps in the v2 environment, 500 generations
each (`summary_nomarkrep.json`); trained 09:38-13:32 and generated by 20:47 on 19 Sep.

*Primary reading, the new adapters alone.* Own-minus-other per adapter, x1e5: A −2.690, −1.755, +21.518;
B +1.951, +1.109, +1.137. theta_sym **+3.544e-5** (+0.0994 % of R_real), Welch SE 3.962e-5, df 2.00,
t 0.89, **one-sided p 0.233**. The one-sided 95 % upper limit is +1.51e-4, above the +1.4e-5 that the
"does not replicate" branch required. Neither branch is met -> **"inconclusive"**. The arm-A spread is
dominated by one adapter (nomark_s5 at +2.15e-4 against two negatives).

*Secondary, pooled over six per body* (`out/g2_pooled_six.json`): A x1e5 +8.715, +0.902, +8.897, −2.695,
−1.755, +21.518; B +(-3.428), +3.130, −1.228, +1.951, +1.109, +1.137. theta_sym **+3.188e-5**
(**+0.0894 %**), SE 1.929e-5, df 5.67, t 1.65, **one-sided p 0.076**, one-sided 99 % limit
**0.2627 %**; the estimate does not exceed U_device (5.3761e-5).

**What this settles.** The v2 environment's positive lean is real enough to keep appearing (+2.8e-5 at
three adapters in Entry 36, +3.19e-5 at six here) and small enough that twelve adapters do not resolve it.
It is not evidence of transfer, and it is not excluded. The paper's robustness paragraph is updated from
three adapters per arm to six, with the same conclusion and a tighter interval. The third positive-leaning
estimate from this environment (Entry 71's +0.154 % in the content-matched arms) remains descriptive.

**G3 deviation.** Entry 76 registered "all 500 generations each". Only seeds 0-2 of each body have 500
generations archived locally; seeds 3-11 have 250 (7,500 images in total, not 12,000). The 500-image rows
exist only as measurements against the published E2 templates, so they cannot answer an estimator question.
G3 therefore uses **every generation archived locally per adapter**, and records the per-adapter counts in
its output. The adapter mean is the unit, so unequal counts enter only through each adapter's own noise.

---

## Entry 80 · 2026-09-20 · Pre-specification — G6, a third paired design (Apple iPhone 5c, VISION) with a flat-field estimator arm

Registered before any adapter is trained. Runs after chain 14. Own splits, fingerprints, crops, adapters,
generations and result files, as for G4.

**Why this pair.** The paired design so far covers one 2006 DSLR model and, with G4, one 2017 Android
smartphone. VISION device D05 and D14 are two physical bodies of the Apple iPhone 5c with 1,400 and 836
natural images, a third model, a different brand and operating system, and a third sensor generation.
FloreView was examined first and rejected: its same-model pairs (Google Pixel 3a D19/D23, Xiaomi Redmi
Note 8T D04/D10) carry only 142-206 images per device, too few for these splits at a fingerprint quality
this study would accept.

**Design.** Splits from natural images, the primary shape: E1 80, E2 140, T 50, H 40, ten-image guards.
Twelve adapters per arm at 2000 steps, 500 generations each, statistic unchanged, measured against this
pair's own E2 estimate. File lists de-duplicated by SHA-256 as in G4. The same fingerprint gates are
reported before any generation is measured, and the same rule applies: held-out AUC below 0.90 or
split-half reliability below 0.15 makes the arm descriptive with no limit claimed.

**Flat-field estimator arm (the reason this pair earns its cost).** VISION ships 113 and 130 flat-field
images for D05 and D14. A second fingerprint estimate is built from the flats alone and the *same*
generations are measured again with it. This tests the estimator dependence of Entry 63 with data rather
than statistics: flats give a far better estimate than natural photographs, so if the limit is stable
between the natural-image and flat-field estimates on one pair, the dependence is a property of weak
estimates and not of the statistic.

Readings, on twelve adapters per arm, for the natural-image estimate (primary) and the flat-field estimate
(reported beside it):
- Neither the intersection-union sign-flip test nor the symmetric statistic rejects at one-sided 0.01 ->
  **"no detectable transfer on the iPhone 5c pair"**, with both limits.
- The symmetric statistic rejects at 0.01 with both arm means positive -> **"device-specific transfer on
  the iPhone 5c pair"**.
- Otherwise the estimate and limits only.
- Estimator comparison, pre-declared: the two limits agree within 25 % -> **"the limit does not depend on
  the estimate when the estimate is good"**; otherwise the spread is reported.

---

## Entry 81 · 2026-09-20 · Diagnostics on the 16000-step arms (derived, descriptive, no new data)

Run while G1 trains, on the twelve existing 16000-step adapters and their training metadata.

**Leave-one-adapter-out.** Dropping any single adapter leaves theta_sym between **+4.93e-5 and +6.36e-5**
with one-sided p between **0.0018 and 0.0081**; the most influential is dose16k_A_s2 (+4.93e-5, p 0.0046).
The finding does not rest on one adapter, unlike the G2 estimate, which one adapter dominates.

**Adapter strength.** Body A's adapters trained harder than body B's: ||lora_B|| 17.42 (sd 0.09) against
17.11 (sd 0.17), **Welch t 3.85, p 0.005**; tail loss 0.2664 against 0.2554. Across all twelve, own-body
contrast rises with ||lora_B|| (**r +0.695, p 0.012**), but that correlation is mostly between arms: within
body A it is −0.423 (p 0.40) and within body B +0.857 (p 0.029, n 6).

**Why this matters.** The asymmetry of the 16000-step result now has two candidate explanations rather than
one: the fingerprint, or how strongly each arm's adapters adapted. The registered inversion arms (G1)
separate them, because inverting the fingerprint changes the sign of what is in the training data without
changing how much the adapter learns. Both diagnostics are descriptive and pre-date no reading.

---

## Entry 82 · 2026-09-20 · Comparison scope — PRNU-Bench checked and not run; FINDINGS.md added

**PRNU-Bench (arXiv:2509.17581) checked as a sixth detector family and rejected on availability.** It is the
newest learned PRNU identification model and would have been the strongest addition to the detector panel.
Checked 20 Sep 2026: github.com/CroitoruAlin/PRNU-Bench has **no releases**, the `trained_models/` path
referenced by its README returns 404, and no weights are published on Hugging Face; the dataset is a
subset pending acceptance and the repository carries no licence. Evaluating it would mean training their
hybrid denoising-autoencoder plus CNN ourselves on partial data - a re-implementation, not a reproduction,
the same category in which FT-Shield and SIREN were excluded. Recorded in the manuscript's related work
with the date of the check.

**Comparison scope, as the paper now states it.** Detection side: five families on the identical
generations with the identical statistic, each with a positive control (C7). Marking side: DiffusionShield
run in full through the same pipeline, because it is published as a fixed artefact; FT-Shield and SIREN
excluded because their marks are optimized against a different UNet or per collection; ProMark,
CustomMark, Tree-Ring and Stable Signature excluded because their unit of attribution is a concept, a
customized model or the generator, not a physical camera body. The justification was previously only in
`paper/REVIEWER_RESPONSE.md` and is now in Section II of the manuscript.

**`FINDINGS.md` added**, regenerated by `src/findings.py` from the result files: 43 findings with value,
source file, pre-specified reading and paper section, and the in-flight experiments listed as pending
rather than omitted. Re-run it after each result lands; `verify_v2.py` independently recomputes the
headline subset from the same files.

---

## Entry 83 · 2026-09-20 · G3 result — estimator sensitivity: *"spread reported; not stable"*

`src/g3_estimator_scale.py` -> `out/g3_estimator_scale.json`. The primary statistic recomputed over all 24
primary adapters and every generation archived locally (500 for seeds 0-2 of each body, 250 beyond;
7,500 images), under five fingerprint estimates, each divided by its own real-image contrast from the
held-out split.

| estimate | photographs | R_real | theta_sym | symmetric 99 % limit | max-arm | one-sided p |
|---|---|---|---|---|---|---|
| E2 (published) | 140 | 0.035659 | −8.29e-6 | **0.0618 %** | 0.2212 % | 0.752 |
| E1 | 80 | 0.027256 | +4.26e-6 | 0.1523 % | 0.6463 % | 0.387 |
| **pooled E1+E2** | **220** | 0.039513 | +0.78e-6 | **0.0815 %** | 0.3404 % | 0.475 |
| half 1 of E2 | 70 | 0.030473 | −4.91e-6 | 0.0931 % | 0.4772 % | 0.644 |
| half 2 of E2 | 70 | 0.025934 | −2.44e-6 | 0.1014 % | 0.2414 % | 0.585 |

**Reading.** The pooled estimate's symmetric limit is **+31.8 %** away from E2's, beyond the registered
25 % -> **"spread reported; not stable"**.

**What is and is not affected.** Every estimate gives theta_sym indistinguishable from zero
(p 0.39-0.75), and none of the five shows transfer: **the null is robust to the estimate**. What moves is
the numerical limit - symmetric 0.062-0.152 %, max-arm 0.221-0.646 % - because each estimate carries its
own sampling error, which the limit construction does not model, and because a weaker estimate lowers
R_real as well as the contrast. Note the estimates differ in their own R_real by a factor of 1.5, so these
are like-for-like comparisons only on their own scales, which is how they are computed here.
**Consequence for the paper:** the limit is quoted with its estimator spread rather than as a single
number, and the pooled estimate - the one with the most data - gives 0.0815 %, slightly wider than the
published 0.0762 %. G6's flat-field arm tests whether the spread collapses when the estimate is good.

---

## Entry 84 · 2026-09-20 · G6 revision before any adapter is trained — splits resized to the real image counts

The Entry 80 registration assumed the primary split sizes for the iPhone 5c pair. The counts it relied on
were wrong: they were taken with a path match that also caught VISION's Facebook-recompressed copies
(natFBH, natFBL). The true unique native counts in `images/nat`, by SHA-256, are **D05 350, D14 209,
D18 204** - so 340 images per body were never available and `g6_prep.py` stopped on its own assertion
before writing anything.

**Revised design, registered before training.** Devices unchanged (D05 role A, D14 role B). Splits
**E1 45, E2 70, T 35, H 25** with ten-image guards, 205 images per body against 209 available. The
fingerprint is therefore estimated from 70 natural photographs rather than 140, which is thin; the
pre-registered gate is unchanged and does the work - held-out AUC below 0.90 or split-half reliability
below 0.15 makes the arm descriptive with no limit claimed. The flat-field arm is unaffected (113 and 130
flat images), and with a natural-image estimate this thin it becomes the more informative half of the
comparison. Adapter count, statistic and readings are unchanged from Entry 80.
Other VISION same-model pairs, recorded for completeness: iPhone 5 (D29 224, D34 204), iPhone 6 (D06 132,
D15 227), iPhone 4s (D02 204, D10 178), Galaxy S3 Mini (D01 205, D26 150).

---

## Entry 85 · 2026-09-20 · Defect D9 - the secondary pairs' gates were computed with the wrong combination; G4 closes at the gate, G6 passes, G4b registered

**The defect.** `g4_prep.py` and `g6_prep.py` formed the per-image own-minus-other contrast correctly and
then combined the two bodies with a **minus**: `R_real = 0.5*(mean_A - mean_B)`, with an AUC that asked
whether body A's images outscored body B's. Neither is the study's statistic. The canonical paired contrast,
used everywhere else and defined in `real_contrast()` of `g3_estimator_scale.py`, adds the two bodies -
`0.5*(mean_A + mean_B)` - and the held-out AUC separates own-fingerprint from other-fingerprint scores on
the same photographs. The filed quantity was therefore a difference between the two bodies' contrasts, which
is near zero when the fingerprint works and large when one body carries it alone: it never tested what the
gate exists to test. Logged as **defect D9**. `src/gate_recheck.py` recomputes both readings from the stored
`H_{A,B}.npz` and `K` files and rewrites each manifest, keeping the superseded values under `_superseded`;
the prep scripts are corrected at source. No generation and no primary-design number is affected - the
defect is confined to the two secondary pairs' gate blocks, neither of which had been measured.

**Corrected gates.** Same held-out photographs, correct combination:

| Pair | estimator | R_real (filed) | R_real (correct) | AUC (filed) | AUC (correct) | body A | body B |
|---|---|---|---|---|---|---|---|
| Huawei P10 Plus (G4) | E2 | +0.0149 | **+0.0179** | 0.9922 | **0.8094** | +0.0328 | **+0.0030** |
| Apple iPhone 5c (G6) | E2 | -0.0104 | **+0.0514** | 0.3210 | **0.9936** | +0.0410 | +0.0618 |
| Apple iPhone 5c (G6) | FLAT | -0.0166 | **+0.0564** | 0.2740 | **0.9749** | +0.0398 | +0.0730 |

**G6 passes and proceeds.** Both estimates identify the bodies on held-out photographs (AUC 0.994 and
0.975, paired own-above-other 98.5 % and 96.9 %), split-half reliability is 0.365 and 0.323, and the two
bodies contribute comparably. The Entry 84 worry that 70 photographs would be too thin is answered by the
data: the natural-image estimate is if anything the better-separated of the two. Lane 3 is running.

**G4 closes at its gate, descriptively, with no limit claimed.** The corrected held-out AUC is 0.809,
below the 0.90 the Entry 78 registration set, so the pre-declared branch applies: *"the arm is reported
descriptively and no limit is claimed - the fingerprint would be too weak to test transfer with."* The
gate is reported before any generation is measured, exactly as registered, so the arm closes here and its
twelve-adapter-per-arm generation is not run. The cause is one-sided and visible: body 1604's own-minus-other
contrast is +0.0328, body 1601's is +0.0030. Split-half reliability is healthy for both (0.510, 0.568) and
the cross-device correlation is -0.002, so the estimates are self-consistent and not contaminated by a
shared model term; body 1601's held-out photographs simply do not correlate with body 1601's own
fingerprint. Its file list is dominated by multi-frame captures (`_1`, `_2` suffixes on the same second)
that SHA-256 de-duplication cannot remove because the bytes differ, and multi-frame fusion averages
independent sensor reads, which is the known mechanism for PRNU loss. This is a finding about the device,
not only about the experiment, and it is reported as one.

**G4b, registered here, before any of its data is prepared.** The scope condition G4 was to close - does
the headline hold for a modern smartphone under the identical paired design - is still open, so it is
retried on a pair that can carry it. Device selection is pre-specified and uses **real photographs only,
with no generated image involved**, so it cannot bias the transfer null: among the five Daxing P20 bodies
already verified duplicate-free (1101-1105), take the two with the highest held-out per-device contrast in
the existing five-body arm that also hold at least 250 unique images. That rule selects **1104 (role A,
contrast 0.075, 262 images)** and **1103 (role B, contrast 0.069, 280 images)**; the rejected bodies are
1101 (0.060), 1105 (0.060, only 244 images) and 1102 (0.022). The P20 is a 2018 device, one year newer than
the P10 Plus, and its five-body arm already reads AUC 0.949 with top-1 identification 1.000.
Design identical to Entry 78: splits E1 60, E2 90, T 40, H 30 with ten-image guards (250 of 262), twelve
adapters per arm at 2000 steps, 500 generations each, scored against that pair's own E2 estimates with the
same statistic. The same gate applies with the corrected definition: held-out AUC below 0.90 or split-half
reliability below 0.15 and the arm is descriptive with no limit claimed.
Readings, on the twelve adapters per arm:
- Neither the intersection-union sign-flip test nor the symmetric statistic rejects at one-sided 0.01 ->
  **"no detectable transfer on a modern smartphone pair"**, reported with both limits.
- The symmetric statistic rejects at one-sided 0.01 with both arm means positive ->
  **"device-specific transfer on a modern smartphone pair"**, which would bound the headline's scope.
- Anything else -> estimate and limits only, with no claim.

---

## Entry 86 · 2026-09-20 · Generation budget for G6 and G4b registered at 250 per adapter, before either arm generates anything

Both remaining paired designs carry twelve adapters per arm. At the primary budget of 500 generations per
adapter that is 12,000 images per pair, and with four lanes sharing one card the observed rate is about
0.3 GPU-minutes per image, so each pair would spend roughly 60 GPU-hours generating alone. **Registered
now, before either arm has produced a single image: G6 (iPhone 5c) and G4b (P20) generate 250 images per
adapter**, from the same seed bank at the same settings - the first 250 seeds of the same sequence, so the
seeds are a prefix of the primary budget rather than a different set.

The statistic is unaffected in kind. Every limit in this study is computed over adapters, not images: the
twelve per-adapter means are the units, and the Welch and sign-flip constructions read their spread. Halving
the images per adapter raises only the within-adapter component of each mean's error, which is small
against the adapter-to-adapter spread that dominates the SE - Entry 79 already measured most archived
adapters at 250 and Entry 83's five-estimate comparison ran on them. What the smaller budget costs is
precision on each individual adapter, and what it buys is that both pairs can be measured at all rather
than one of them. The count actually present is recorded per arm in each result file, and the arms remain
comparable to each other because both use the same budget.

`T1_GENS` in `t1_ladder.py` carries the budget; every other arm keeps 500 and the default is unchanged.

---

## Entry 87 · 2026-09-20 · Defect D10 (stale crops from a superseded split), the 50-image training convention, and revised splits for G6 and G4b

Three operational faults, all caught before any adapter or generation of the affected arms was kept.

**D10 - stale training crops from a superseded split.** `g6_prep.py` writes its training crops as
`0000.png ... (T-1).png` into a directory it creates if absent. The Entry 84 revision lowered G6's training
split from 50 to 35, so the second run overwrote `0000-0034` and **left `0035-0049` in place from the
superseded split**. The directory therefore held 35 images from the registered split and 15 from an
abandoned one, whose members under the new split fall in the guard bands and neighbouring splits. Lane 3
began training `p5c_A_s0` on that mixed set at 15:19 and was killed at 15:26, about five minutes in; no
adapter was saved and no generation was made, so nothing measured is affected. The mixed directory is
quarantined under `C:\D_offload\einv_20260920_repair`. The lesson is that a split revision must clear the
materialised crops, not just rewrite them: every prep now removes the directory before writing.

**The training-set size is fixed at 50 by the design, not free per arm.** `t1_ladder.py` asserts exactly 50
training images per adapter, because every arm in this study trains on 50 - that is what makes adapters
comparable across arms. G4 (T 40), G6 (T 35) and G4b (T 40) were registered with smaller training splits,
which would have made them incomparable with the primary arms even had they run; the assertion stopped
G4b's first training attempt outright. **Revised, before any of these arms trains: T = 50 for both pairs**,
with the other splits resized to fit the images each body actually holds.

| Arm | body A | body B | E1 | E2 | T | H | guards | total | available (B) |
|---|---|---|---|---|---|---|---|---|---|
| G6 (iPhone 5c) | D05 | D14 | 40 | 60 | **50** | 25 | 3x10 | 205 | 209 |
| G4b (Huawei P20) | 1104 | 1103 | 60 | 90 | **50** | 30 | 3x10 | 260 | 262 |

G6's natural-image fingerprint now rests on 60 photographs, which is thin; its flat-field arm (113 and 130
flat images) is unaffected and remains the better estimate, and the pre-registered gate decides both. All
fingerprint estimates, held-out blocks and gates for both pairs are discarded and rebuilt from the revised
splits - the Entry 85 gate table is superseded for these two pairs and is re-reported below once measured.
The superseded estimate directories are quarantined rather than deleted.

**G5's generation ran out of memory after one adapter of six.** Lane 2 finished `alt_A_s0` (500 images) and
then died: with two training processes at about 10 GB each and a 28 GB generation, the card had 1.2 GB left
when generation asked for 2 GB. *(Corrected the same day, before anything depended on this entry: the first
version of this paragraph blamed 6 GB held by a four-process measurement job. That was wrong. The 6.3 GB
belongs to an unrelated OCR service of another project, resident on this machine since 11 September and
outside this study's control. The practical consequence is that this card offers about 39.7 GB to these
experiments, not 46, which is why one generation and one training fit together and a second training does
not.)* The lane then ran its remaining stages against data that did not exist, so `summary_alt.json` was
written from a single adapter; it is quarantined as incomplete and G5 is not scored until all six adapters
have generated. Two fixes: measurement jobs run with the GPU hidden, since they are CPU work, and GPU
stages are admitted through `src/gpu_admit.sh`, which holds a lock while it waits for enough free memory
and while the job allocates, so two lanes can no longer both start into the same free space. No statistic
changes; this is scheduling, and it is recorded because it explains the gap in the run logs.

---

## Entry 88 · 2026-09-21 · Pre-specification — four CPU-only analyses (H1-H4) that close gaps the GPU queue cannot

The GPU queue is saturated: one job holds the card at 100 % utilisation, so adding lanes interleaves work
rather than adding throughput. The four analyses below need no GPU at all. They run beside the queue on
archived data and cost it nothing. All four are registered here, with their readings, before any is run.

**H1 - does the 16000-step interaction track how hard each adapter trained?** The one resolved positive in
this study is carried by one body, and arm A's adapters have larger adapted weights than arm B's
(`lora_B_norm` 17.42 against 17.11, p 0.005). That is a fourth candidate explanation beside the
fingerprint, training-set content and the training environment, and it is testable on data already
written: twelve adapters, each with a per-adapter contrast (`summary_dose16k*.json`) and its own
`train_meta.json`. Fit the per-adapter contrast on adaptation strength (`lora_B_norm`, primary;
`loss_tail`, secondary) and a body indicator, by ordinary least squares over the twelve adapters.
- Adaptation-strength slope resolved (p < 0.05) **and** the body term no longer resolved after adjustment
  -> **"the 16000-step interaction tracks adaptation strength, not body identity"**, which would make it an
  artifact of unequal training rather than a fingerprint effect, and G1's inversion must then be read
  alongside this.
- Body term still resolved and the strength slope not -> **"adaptation strength does not explain it"**; the
  three original explanations stand and G1 remains the discriminator.
- Anything else -> inconclusive, both coefficients reported.
Twelve adapters give this little power, so it is a covariate check and is reported as one, never as a test
that clears or convicts on its own.

**H2 - propagate fingerprint-estimation error into the limit.** Entry 83 showed the limit ranges
0.062-0.152 % across five estimates while the null holds under all five, and the limit construction models
no estimation error at all. Resample the photographs each body's E2 estimate is built from (24 bootstrap
replicates, the same seeds for both bodies), re-estimate each K, and re-score a fixed subsample of 200
generations per adapter - the same images in every replicate, so only the estimate moves. Recompute the
symmetric statistic and its one-sided 99 % limit per replicate.
- Report the limit's distribution over replicates and an estimation-inflated limit that adds the
  between-replicate variance to the adapter-level variance.
- If the inflated limit exceeds the filed value by more than a factor of two, **the paper quotes the
  inflated limit as its headline** and the filed one as the fixed-estimate special case.
- The null is re-read under every replicate; if any replicate rejects at one-sided 0.01, that is reported.

**H3 - is the shifted-template excess a property of generator residuals?** The field's standard inert
control is elevated in every adapted arm, including arms that never received the fingerprint, and the
autoencoder grid does not explain it. A circularly shifted K still correlates with Y*K through the spatial
autocorrelation of the residual and of the luminance field, which is a quantitative prediction rather than
a further control. Measure the autocorrelation of W and of Y*K on base-model and adapted generations,
predict the shifted-template correlation as a function of displacement from those measurements, and compare
with the observed grid in `c5_shift_grid.json`.
- Predicted tracks observed across displacements (rank correlation >= 0.8) and the on-grid excess predicted
  within its interval -> **"the shifted-template excess is explained by residual autocorrelation"**, which
  converts an open question into a characterised artifact and tells the field the control is not inert in
  generated images.
- Otherwise -> the artifact remains open and is reported as such, with the prediction's failure shown.

**H4 - coverage with the variance components the simulation omits.** Entry 58's coverage simulation models
neither fingerprint-estimation error nor training-set variance, and the symmetric limit already covers at
98.9-99.3 % against nominal 99 %. Re-run it with both components added, calibrated from H2 (estimation) and
from G5's second training set (training-set variance) once G5 lands.
- Symmetric coverage stays at or above 0.98 -> the construction is reported as adequate with the components
  included.
- Below 0.98 -> **the max-arm limit becomes the paper's primary construction** and the symmetric statistic
  is reported as a secondary estimate, which is a change to how the headline is stated.

---

## Entry 89 · 2026-09-21 · H1 result — the 16000-step interaction cannot be separated from how hard the adapters trained: *"inconclusive"*

Twelve 16000-step adapters, each with its archived per-adapter own-minus-other contrast
(`natural_paired_KA_minus_KB`, oriented as `dose_stats.py` orients it) and its own `train_meta.json`.
Regression of the contrast on a body indicator and on adaptation strength, standardised:

| model | R^2 | body A | adaptation strength |
|---|---|---|---|
| body only | 0.614 | +1.151e-04, **p 0.003** | - |
| strength only (`lora_B_norm`) | 0.483 | - | +5.337e-05, **p 0.012** |
| both | 0.634 | +8.974e-05, p 0.087 | +1.715e-05, p 0.500 |

Correlation of adaptation strength with the contrast is +0.695, and with the body indicator +0.773. The
arms differ in strength to begin with (`lora_B_norm` 17.42 against 17.11, p 0.0055), while `loss_tail`
(0.2663 against 0.2554, p 0.13) and wall-clock minutes (p 0.21) do not separate them. With `loss_tail` as
the strength measure instead, the body term stays resolved (p 0.012) and strength does not (p 0.556).

**Pre-declared reading: inconclusive - both coefficients reported, neither explanation promoted.** That is
the honest outcome and it is worse for the paper than the registered alternatives, because it says the
design cannot tell the two apart: adaptation strength alone explains the interaction at p 0.012, body
identity alone at p 0.003, the two are collinear at 0.773, and adding the second explains almost nothing
further (R^2 0.614 -> 0.634). Twelve adapters cannot separate collinear explanations, and no amount of
re-analysis of these twelve will.

**What this changes.** The manuscript said the 16000-step signal is "carried by one of the two bodies".
On this evidence it cannot say that cleanly: it is carried by adapters that trained harder, and those
adapters are mostly one body's. The claim is weakened to what the data support, in
Section~
ef{sec:limits} and in the open-questions subsection, and the confound is stated with its
numbers. This is a caveat on an already-open question, not a withdrawal: nothing in the headline limit
depends on it, since the headline is the 2000-step dose where the statistic is indistinguishable from zero.

**Registered now, before G1 lands:** G1's inversion arms are read with the same covariate. The inversion
design compares arms that differ in the sign of the fingerprint in their training images, so if the
inverted arms also differ in `lora_B_norm`, the same confound recurs and the sign reading is reported with
the strength-adjusted estimate beside it. If the inverted arms are balanced in strength, the sign reading
stands on its own. Either way both are reported.

---

## Entry 90 · 2026-09-21 · H2 result — estimation error is 43 % of the variance; limit 0.109 % inflated against 0.082 % filed, and the null is untouched

Twenty-four bootstrap replicates of each body's E2 photographs, each giving its own K, its own real-image
contrast on the held-out split, and its own symmetric limit on the same 4,800 generations (200 per adapter,
identical images in every replicate, so only the estimate moves). `estimate_K` accumulates sum(W*Y) and
sum(Y*Y) before its post-processing, so each replicate is an exactly weighted combination of one pass over
the photographs: the bootstrap is exact for this estimator rather than an approximation of it.

| quantity | value |
|---|---|
| symmetric limit, mean over replicates | **0.0675 %** of R_real |
| its spread | sd 0.0460, range -0.0522 to 0.1546 |
| variance from adapters | 56.8 % |
| **variance from the fingerprint estimate** | **43.2 %** |
| estimation-inflated limit | **0.1093 %** |
| filed limit (Entry 83, pooled estimate) | 0.0815 % |
| inflation | **1.34x** |
| null across replicates | one-sided p from **0.304 to 0.999**; 0 of 24 reject at 0.01 |

**Pre-declared reading: estimation error widens the limit by a stated factor below two, so the filed limit
stands with the inflated value reported beside it.**

Two things this settles. First, the null is not a property of one lucky estimate: resampling the
photographs the estimate is built from never produces a rejection, and the smallest p over twenty-four
replicates is 0.304. Entry 83 showed this across five estimators; this shows it across the sampling
distribution of the estimator itself. Second, the limit's uncertainty was understated by construction.
Nearly half the variance in the statistic comes from not knowing K, which the published limit treats as
zero, and one replicate's limit is negative - the estimate is small enough that resampling the photographs
can place the whole interval below zero. The honest one-sided 99 % statement is therefore **0.109 % of the
real-image contrast**, not 0.082 %, and that is what the limitations section now carries.

This is the measurement Entry 76 asked for in words: the gap between "the limit under one estimate" and
"the limit accounting for the estimate" is a factor of 1.34, which is smaller than the factor of 2.5 the
five-estimator spread suggested, because that spread mixed estimators built from different numbers of
photographs while this holds the estimator fixed and resamples its input.

---

## Entry 91 · 2026-09-21 · H3 result — *"the prediction does not account for the excess"* on its registered criterion; H3b registered on disjoint images

Nine arms, sixty generations each, the 769 displacements of Entry 58 (C5). For every image, W and
V = Y*K are split into the component that repeats on the 8-pixel grid (the mean of each residue class
modulo 8, tiled back) and the remainder; the prediction is the cross-correlation of the two periodic parts,
normalised by the full fields, so nothing is fitted.

| arm | observed on-grid excess | predicted | ratio | Spearman over 769 |
|---|---|---|---|---|
| local_base | +6.18e-05 | +7.80e-05 | 1.26 | +0.40 |
| local_A_raw_s0 | +1.09e-04 | +5.85e-05 | 0.54 | +0.34 |
| local_B_raw_s0 | +5.43e-05 | +5.96e-05 | 1.10 | +0.33 |
| nomark_s0 | +6.05e-05 | +6.27e-05 | 1.04 | +0.38 |
| nomark_s1 | +7.80e-05 | +7.28e-05 | 0.93 | +0.36 |
| nomark_s2 | +4.21e-05 | +7.40e-05 | 1.76 | +0.38 |
| nomarkB_s0 | +9.60e-05 | +8.18e-05 | 0.85 | +0.36 |
| nomarkB_s1 | +3.06e-05 | +6.39e-05 | 2.09 | +0.36 |
| nomarkB_s2 | +2.13e-05 | +6.19e-05 | 2.90 | +0.33 |

**Pre-declared reading: the prediction does not account for the excess; the artifact remains open and the
prediction's failure is reported.** The registered criterion required a rank correlation of at least 0.8
across all 769 displacements *and* the on-grid excess predicted within its interval. The median rank
correlation is 0.36, so the criterion is not met and that is the result of record.

**What the same run shows, descriptively.** Every one of the nine arms has its on-grid excess predicted
within two standard errors, with per-arm ratios whose median is 1.10, including `local_base` - generations
of the unadapted model, which never saw this fingerprint. That is the part of the phenomenon the open
question is about.

**Why the criterion and the phenomenon came apart, and what follows.** The rank correlation is taken over
769 displacements of which 12 are on the grid; the other 757 are off-grid values whose arm means are
noise at this sample size, so a model that predicts only the grid-aligned structure cannot rank them and
should not have been asked to. That is a fault in the criterion I registered, recognised only after seeing
the result, so it cannot be repaired into a pass here: the reading above stands.

**H3b, registered now and run on images disjoint from H3's** (generations 60-119 of each arm, where H3 used
0-59; same arms, same displacements, same code path with no fitted quantity). The criterion is stated
before the run: pooled across the nine arms, the predicted on-grid excess must lie within two standard
errors of the observed pooled excess, **and** the median per-arm predicted/observed ratio must fall in
[0.67, 1.5].
- Both met -> **"the on-grid excess is quantitatively accounted for by the grid-periodic components of the
  residual and of Y*K, with nothing fitted"**, which converts the open question into a characterised
  artifact and tells the field that the shifted-fingerprint control is not inert in generated images.
- Either not met -> the artifact remains open, on two independent samples, and both are reported.

---

## Entry 92 · 2026-09-21 · H3b result — *"the artifact remains open"*, on a second, disjoint sample

Generations 60-119 of each of the nine arms, disjoint from H3's 0-59; same displacements, same code path,
nothing fitted. Pooled across arms: observed on-grid excess **+1.017e-05**, predicted **+6.897e-05**, a
difference of **+1.28 SE**, median per-arm ratio **1.88**.

**Pre-declared reading: the criterion required both the 2-SE agreement and a median ratio in [0.67, 1.5].
The first is met and the second is not, so the artifact remains open on two independent samples.**

The two samples together say something the first alone did not. The *prediction* is stable - +6.90e-05 here
against +7.0e-05 in H3 - while the *observation* moves from +6.18e-05 (pooled over H3's arms) to
+1.02e-05 here, on images drawn from the same arms. The observed on-grid excess is therefore measured with
an uncertainty comparable to its own size at sixty images per arm, which is why the ratio test swings from
1.10 to 1.88 while the 2-SE test passes both times. The grid-periodic account is not excluded by either
sample; it simply cannot be confirmed against an observation this noisy.

What would settle it is not a cleverer model but more images: pinning the observed excess to a tenth of its
size needs of order a hundred times the sample, which is thousands of generations per arm rather than
sixty. That is stated in the paper as the measurement that would close the question, and the prediction's
two attempts are reported with it.

---

## Entry 93 · 2026-09-21 · H4 result — the symmetric limit does not cover once estimation error is carried: *"the max-arm limit becomes the primary construction"*

Entry 58's simulation re-run with the components it omitted. The estimation component is **measured**, from
H2's bootstrap replicates (SD 1.224e-05, which is 0.28 of the adapter SD); the training-set component is
swept, because G5 has not yet landed. Minimum coverage over all 27 scenario cells, nominal 99 %:

| scenario | symmetric | max-arm |
|---|---|---|
| C6 reproduction (neither component) | 0.9858 | 0.9998 |
| **+ estimation error (measured)** | **0.8875** | **0.9955** |
| + training-set variance at 0.5x adapter SD | 0.8155 | 0.9755 |
| + training-set variance at 1x adapter SD | 0.7165 | 0.9347 |
| + training-set variance at 2x adapter SD | 0.6178 | 0.8752 |

**Pre-declared reading: symmetric coverage falls below 0.98 once the components are included, so the
max-arm limit becomes the paper's primary construction and the symmetric statistic is reported as a
secondary estimate.**

The decisive row needs no assumption. With only the measured estimation component and no training-set
variance at all, the symmetric one-sided 99 % limit covers at **88.8 %**, while the max-arm limit covers at
**99.6 %**. The reason is structural rather than numerical: estimation error enters as a shift shared by
both arms, because the same fingerprint estimate scores both, and the symmetric statistic averages the two
arms, so a shared shift passes straight through it while its Welch standard error - built from the spread
*within* each arm - cannot see it. The max-arm construction takes the larger of two single-arm limits and
carries enough slack to absorb it.

This changes how the headline is stated, which is why it was registered in advance. The paper's leading
number was already the max-arm limit; what changes is that it is now named as the primary construction for
a stated reason, and the symmetric statistic - the smaller and more flattering number - is demoted to a
secondary estimate that assumes a known fingerprint. Nothing about the null changes: H2 re-read it under
twenty-four estimates without a single rejection.

The training-set rows are a sensitivity sweep, not a measurement, and are labelled so. When G5's six
adapters are measured, H4 re-runs with that component pinned; the queue does it automatically.

---

## Entry 94 · 2026-09-21 · Pre-specification — H5, is the body recoverable from the adapter weights themselves?

Every limit in this study is measured on generated images: a black-box channel. The manuscript lists
adapter weights among the things it does not test, and that gap is closable on archived data with no GPU.
Two families hold six adapters per body - `nomark` / `nomarkB` at the primary 2000-step dose, and
`dose16k_A` / `dose16k_B` at 16000 steps - so the same construction runs at both doses, and the second
also bears on what the 16000-step interaction is.

**Statistic.** A LoRA adapter's functional update is `dW_l = B_l A_l` per layer; the factorisation itself is
seed-dependent and not comparable across adapters, but `dW` is. Similarity between two adapters is the
cosine between their concatenated updates, computed exactly without forming any `dW`: for rank 16,
`trace((B_i A_i)^T (B_j A_j)) = trace((B_i^T B_j)(A_j A_i^T))`, two 16x16 products per layer. The statistic
is **D = mean cosine between same-body pairs minus mean cosine between different-body pairs**, over all 66
pairs of the twelve adapters.

**Null.** Exact permutation over the balanced relabellings of body: C(12,6)/2 = 462 distinct assignments,
so the smallest attainable p is 1/462 = 0.0022. One-sided, D > 0.

**Readings, identical at both doses:**
- p < 0.01 with D > 0 -> **"the training body is recoverable from the adapter weights"** at that dose. This
  would be a white-box channel that the image-domain measurement does not see, and it would be reported as
  a scope extension rather than as a change to any image-domain limit.
- p >= 0.01 -> **"the adapter weights carry no detectable body signature at this dose"**, which closes the
  scope item the limitations section currently leaves open.

**What a positive result would and would not mean, fixed before seeing it.** The two bodies' adapters are
trained on different photographs, so the two bodies are confounded with their training sets exactly as in
E3. A positive D therefore says the weights carry *the training set's* identity, of which the camera is one
component among scene content, exposure and everything else those photographs differ in. It must not be
reported as recovering the camera, and the wording above is chosen accordingly. Separating the two would
need the content-matched design of E3 rerun at the weight level, which is not registered here.

---

## Entry 95 · 2026-09-21 · H5 result — *"the adapter weights carry no detectable body signature"* at either dose; what they do carry is the training run

Twelve adapters per dose, six per body, compared by the cosine between their concatenated functional
updates dW = B A, computed exactly from 16x16 products. Exact permutation over 462 balanced relabellings.

| dose | same-body cosine | different-body cosine | D | one-sided p |
|---|---|---|---|---|
| 2000 steps | +0.0163 | +0.0967 | **-0.0804** | 0.933 |
| 16000 steps | +0.0251 | +0.0927 | **-0.0676** | 0.933 |

**Pre-declared reading, at both doses: the adapter weights carry no detectable body signature.** This
closes the scope item the limitations section listed as untested: the black-box result is not hiding a
white-box channel that a party holding the weights could exploit.

**Why D is negative, which the registration did not anticipate.** Same-body adapters are *less* alike than
different-body adapters, by about two permutation SDs at both doses. Grouping the same twelve adapters by
training batch instead of by body:

| dose | same-batch cosine | different-batch cosine | D by batch |
|---|---|---|---|
| 2000 steps | +0.1160 | +0.0136 | **+0.1024** |
| 16000 steps | +0.1121 | +0.0201 | **+0.0920** |

The batches are real and were identified from the weight files' timestamps, not assumed: seeds 0-2 of both
arms were trained 9-12 September and seeds 3-5 on 15-19 September, ten days apart, with cross-body
same-batch pairs trained within the hour. Weight space is organised by *when and where an adapter was
trained*, and the effect is larger than the body contrast and opposite in sign - a cross-body pair from one
morning resembles itself more than a same-body pair ten days apart.

**What this corroborates.** G2 found that a second training environment leaves a main effect shared by
adapters of either body, at 0.09 % of the real contrast and inconclusive on its own (p 0.076). H5
reaches the same conclusion from the weights instead of the images, and far more strongly: the training run
is the dominant structure in adapter space, and the camera is not visible in it at all. The two are
independent measurements of the same thing, which is worth more than either alone.

**Status of the batch finding.** The body test is registered and its reading stands. The batch grouping was
examined after seeing the negative D, so it is descriptive: it explains the sign and it agrees with G2, but
it is not a registered test and is reported as an observation. A registered version would fix batch
membership in advance and train adapters of both bodies in deliberately separated runs.

---

## Entry 96 · 2026-09-22 · G5 result — *"no detectable transfer with a second training set"*

Three adapters per body on a second, fully disjoint 50-image training set per D200 body, 500 generations
each, statistic unchanged. All six adapters generated (the OOM of Entry 87 cost only time; the resumed run
produced the full 500 per adapter).

| quantity | value |
|---|---|
| per-adapter A | -5.17e-05, -3.97e-05, -6.39e-05 |
| per-adapter B | +2.66e-05, -2.90e-06, +1.02e-04 |
| theta_A | -5.178e-05 |
| theta_B | +4.190e-05 |
| theta_sym | **-4.943e-06** = **-0.0139 %** of R_real |
| Welch t, one-sided p | -0.309, **p 0.608** |
| one-sided 99 % limit | **0.2629 %** (registered threshold 0.30 %) |
| max-arm limit | 0.9872 % |
| vs primary theta_sym (+4.60e-06) | difference -9.54e-06 +- 1.84e-05, z -0.52, p 0.603 |

**Pre-declared reading: theta_sym is not resolved and is negative, and the one-sided 99 % limit is below
0.30 %, so "no detectable transfer with a second training set".**

This closes the first of the three scope conditions the manuscript listed as open. The primary limit
generalised over adapter seeds only, because all 24 primary adapters saw the same 50 photographs of their
body; it now also holds when each body contributes an entirely different 50 photographs, and the two
training sets' symmetric estimates are statistically indistinguishable (p 0.60).

Two honest qualifications. The limit here is 0.26 % against the primary's 0.15 %, and the max-arm limit is
0.99 % against 0.41 %, because three adapters per arm constrain the spread far less than twelve - this
confirms the null at coarser resolution rather than tightening it. And the two arms lean in opposite
directions (theta_A negative, theta_B positive) by more than either does at the primary training set, which
the symmetric statistic cancels by construction; with three adapters per arm that pattern is well inside
what chance produces.

---

## Entry 97 · 2026-09-22 · H4 re-run with the training-set component measured rather than swept

`orchestrate18.sh` fired on G5's measurement and re-ran H4 automatically. The training-set component,
estimated from the shift between each body's two training sets and corrected for sampling, is **2.201e-05,
which is 0.50 of the adapter SD** (shift_A -4.22e-05, shift_B +2.31e-05; two contrasts only, so a bound
rather than a precise variance).

| scenario | symmetric | max-arm |
|---|---|---|
| C6 reproduction | 0.9862 | 0.9998 |
| + estimation error (measured) | 0.8922 | 0.9955 |
| **+ both, training component pinned from G5** | **0.8120** | **0.9750** |

**The Entry 88 reading is unchanged and now rests on two measured components rather than one measured and
one assumed: the max-arm limit is the primary construction, the symmetric statistic secondary.** The pinned
training component lands between the 0.5x and 1x sweep rows, closer to the optimistic end, so the earlier
sensitivity sweep did not overstate the problem. Max-arm coverage with both components is 0.975, below the
nominal 0.99 but far above the symmetric statistic's 0.812; that shortfall is itself worth stating, and it
is the reason the manuscript reports the max-arm limit as the primary bound and not as an exact one.

---

## Entry 98 · 2026-09-22 · G1 result — *"inconclusive"*, and the test could not have decided: power 0.22

Six adapters at 16000 steps on training crops carrying each body's own fingerprint inverted and amplified
(stored inverted contrast -2.85x for body A, -1.83x for body B), 250 generations each, statistic unchanged.

| quantity | value |
|---|---|
| theta_sym | **-5.089e-05** = **-0.1427 %** of R_real |
| Welch SE, df | 4.009e-05, 3.15 |
| one-sided p (theta_sym < 0) | **0.145** |
| per-adapter A | -6.58e-06, +1.097e-04, -9.44e-06 |
| per-adapter B | +3.86e-06, -2.262e-04, -1.768e-04 |
| the effect it mirrors (unsuppressed 16000-step theta_sym) | +5.710e-05 |

**Pre-declared reading: theta_sym is negative but one-sided p is 0.145, not below 0.05, and it is not
positive-and-within-2-SE of the unsuppressed value either, so "inconclusive".**

**The honest addition, which matters more than the reading: this test had no real chance of deciding.** To
resolve an effect the size of the mirrored +5.710e-05 at one-sided 0.05 it needed a standard error of
2.47e-05; it achieved 4.01e-05 on three adapters per arm. **Achieved power was 0.22.** Had the fingerprint
hypothesis been exactly true, this design would have failed to detect it four times in five. The
"inconclusive" is therefore a statement about the experiment, not about the fingerprint, and it must not be
read as evidence that the signal is not fingerprint-driven. I under-powered it: three adapters per arm was
chosen for the 12 GPU-hours each costs at this dose, without computing what three could resolve.

**What the point estimate does and does not say.** It is negative, which is the direction the fingerprint
explanation predicts under inversion, and its magnitude (-5.09e-05) is close to the mirror of the
unsuppressed +5.71e-05. That is suggestive and it is not evidence: at p 0.145 a null of zero is entirely
comfortable, and the per-adapter values scatter across two orders of magnitude within each arm.

**The Entry 89 covariate, registered before this landed.** The confound recurs. The inverted arms differ in
adapted-weight norm in the same direction as the original arms (A 17.441 against B 17.235, p 0.126), the
norm correlates with the per-adapter interaction at **+0.810**, and with the body indicator at **+0.711**.
Removing the strength trend leaves theta_sym unchanged at -5.089e-05 and moves the one-sided p from 0.145
to 0.106. So the adjustment does not rescue the test, and adaptation strength remains entangled with body
identity in this arm exactly as it was in the unsuppressed one.

**What would settle it, with its price.** Eight adapters per arm at this dose - five more per arm than were
run - which is about 190 GPU-hours, roughly eight days on this card. That is the measurement, and it is
stated in the paper as such rather than implied to be cheap.

---

## Entry 99 · 2026-09-22 · Pre-specification — G1 extended to eight adapters per arm, to give the test the power it lacked

G1 returned "inconclusive" at a power of 0.22 (Entry 98). The result is uninformative about the fingerprint
and informative about the design, so the design is extended rather than the result reinterpreted.

**Extension.** Five further adapters per body on the *same* inverted training crops, seeds 3-7, 16000 steps,
250 generations each, statistic unchanged - bringing the arm to the **eight per arm** that Entry 98's power
calculation names. Arms `inv16kext_{A,B}_s{3..7}`; no new crops are materialised, so the training input is
byte-identical to the three adapters already run.

**Reading, on all eight adapters per arm together**, unchanged from Entry 77 so that the criterion is not
moved after seeing data:
- theta_sym < 0 with one-sided Welch p < 0.05 -> **"the 16000-step signal follows the fingerprint's sign"**.
- theta_sym > 0 and within 2 adapter-level SE of the unsuppressed +5.710e-5 -> **"the 16000-step signal does
  not follow the fingerprint"**.
- Otherwise **"inconclusive"**, and this time that reading will mean something, because the design will have
  had the power to decide.

**The sequential caveat, stated because it is real.** The decision to add adapters was taken after seeing an
inconclusive result, which is optional stopping, and a nominal p from the combined eight is therefore
slightly anti-conservative. Three things bound the damage: the extension was triggered by the *power*
calculation and not by the observed sign, the threshold is unchanged, and the final report will give the
p from the five new adapters alone alongside the p from all eight, so a reader who trusts only the fresh
adapters has that number. If the two disagree in reading, both are reported and the arm stays
"inconclusive".

**Cost and priority.** Ten adapters at roughly 12 GPU-hours each is about 190 GPU-hours, some eight days on
this card. It is queued behind G6 and G4b, which are closer to completion and close a different scope
condition; the order can be changed if the 16000-step question is judged more valuable than the
third-camera-model and modern-smartphone ones.

---

## Entry 100 · 2026-09-22 · Defect D11 — the GPU admission lock never locked

Entry 87 introduced `src/gpu_admit.sh` and described it as holding a lock while a stage waits for memory
and while it allocates. It did not. The script called `flock`, which Git Bash on Windows does not ship, so
every invocation printed `flock: command not found` and proceeded unconditionally: the memory *check*
worked, the *mutual exclusion* did not. Two lanes could still have measured the same free memory and both
started into it - exactly the failure Entry 87 claimed to have fixed.

Nothing was lost to it. After Entry 87 the work was restructured into one sequential queue beside a single
training lane, so at no point since were two admissions racing; the G6 generation now running is the only
GPU job on the card. The claim in Entry 87 was wrong when written, and is corrected here rather than
quietly.

`src/gpu_admit2.sh` replaces it with a directory lock: `mkdir` is atomic on every filesystem this runs on,
the holder's pid is recorded so a lock left by a killed holder is reclaimed rather than deadlocking, and
the lock is released by trap on exit. Chain 19 uses it. Chains 17's remaining stages still call the old
script, which is a no-op passthrough - correct in the current single-job schedule, and not worth editing a
running shell to change.

---

## Entry 101 · 2026-09-23 · The queue restarts itself after a reboot

The machine has closed twice in two days, each time killing every chain. Nothing was lost either time -
adapters skip when their weights exist and generation skips arms that already hold their full image count,
so a restart costs only the hours the machine sat idle before anyone noticed. That idle time is now removed:
`src/restart_queue.sh` starts whichever chains are not running, and a Windows scheduled task
("EINV queue restart", at logon, launching `src/restart_queue.cmd`) fires it.

The wrapper is safe to run at any time and as often as it likes: it checks for each chain by process and
starts only what is missing, so firing it while the queue is healthy is a no-op - verified by running it
against a live queue, which left both chains alone. It also waits up to ten minutes for `nvidia-smi` to
answer, because the driver can lag a boot by a minute or two and a chain that starts before the card is
visible would fail its first memory check.

The trigger is logon rather than boot, so it needs someone to log in - the alternative runs as SYSTEM,
where the conda environment and GPU context are not the ones these chains were built and tested against.
For a desktop VM in daily use this is the right trade; on an unattended machine it would not be.

---

## Entry 102 · 2026-09-24 · Defect D12 — D: filled during G4b generation, and the chain scripts reported success anyway

At 01:52 on 24 Sep, `t1_ladder.py generate` for G4b died with `OSError: [Errno 28] No space left on device`
while writing image 49 of `p20b_B_s1` (the fourth of 24 arms). D: had 780 KB free: G6's 6,000 images and the
first G4b arms had consumed the ~14 GB of headroom left after the Entry 87-era offload.

**The defect is in the orchestration, not the disk.** Chains 12-19 log each stage unconditionally after it
returns, without testing its exit code. So chain 17 wrote "G4b generated", ran `g4b_measure.py` (which then
failed with `KeyError: 'p20b_A_s2'`), wrote "G4b measured" and "queue done", and chain 19 - waiting on that
string - took the card for the G1 extension. Nothing false reached a result file: the measurement crashed
rather than scoring a partial arm, and no `g4b_p20.json` was written.

**Nothing was lost.** All 49 images in `p20b_B_s1` open and decode fully (checked one by one; the failed
write left no truncated file). G6's generation had finished before the disk filled, and its measurement read
6,000 of 6,000 images. The G1-extension training that started at 02:04 never needed to write to disk before
space was freed and continued without error.

**Recovery.** 28 measured generation folders (22.2 GB) were moved to `C:\einv_gens` with robocopy `/MOVE` and
replaced by directory junctions, file counts verified through each junction; a move, never a deletion.
Junctions were also pre-created on C: for all 30 generation folders not yet written (20 remaining G4b arms,
10 G1-extension arms), so no further image lands on D:. D: now has 22 GB free, C: 448 GB. Chain 20
(`src/orchestrate20.sh`) resumes G4b generation at `p20b_B_s1` image 49 beside the G1-extension training
and, unlike its predecessors, checks every stage's exit code and the final image count (6,000) before it
measures. `restart_queue.sh` now watches chains 19 and 20.

---

## Entry 103 · 2026-09-24 · G6 result — *"no detectable transfer on an iPhone 5c pair"* under both estimators; the estimator spread is not closed by a good estimate; and the flat-field estimate removes the shared main effect

VISION D05 (role A) and D14 (role B), Apple iPhone 5c. Twelve adapters per arm at 2000 steps on 50 training
images each, 250 generations per adapter (Entry 86), 6,000 images, each scored against both bodies'
natural-image estimate (E2, 60 photographs) and flat-field estimate (FLAT, 113 and 130 flats), each on its
own held-out real contrast. Splits E1 40 / E2 60 / T 50 / H 25 (Entry 87). `out/g6_p5c.json`.

| | E2 (natural images) | FLAT (flat fields) |
|---|---|---|
| gate: held-out AUC / R_real | 0.9988 / 0.047409 | 0.9940 / 0.058578 |
| arm A mean own-minus-other | -1.287e-04 (**0 of 12 positive**) | +1.59e-06 (8 of 12) |
| arm B mean own-minus-other | +1.109e-04 (**12 of 12 positive**) | +2.575e-05 (9 of 12) |
| additive part (A - B)/2 | **-1.198e-04** | -1.21e-05 |
| intersection-union sign-flip p_A / p_B | 1.0000 / **0.0002** | 0.4651 / 0.0547 |
| theta_sym | -8.90e-06 (**-0.019 %**), SE 1.28e-05, t -0.70, **p 0.753** | +1.37e-05 (**+0.023 %**), SE 1.12e-05, t +1.22, **p 0.118** |
| symmetric one-sided 99 % limit | **0.0504 %** | **0.0715 %** |
| max-arm limit | **0.3814 %** | **0.1223 %** |
| reading | no detectable transfer | no detectable transfer |

**Pre-declared readings.** For both estimates, neither the intersection-union test (arm A does not reject)
nor the symmetric statistic rejects at one-sided 0.01 -> **"no detectable transfer on an iPhone 5c pair"**,
with both limits. **Estimator comparison:** the registered rule asked whether the two limits agree within
25 %. The symmetric limits differ by a factor of 1.42 and the max-arm limits by 0.32 -> neither agrees ->
**"the spread is reported"**: a good (flat-field) estimate does not make the limit independent of the
estimate. Per-adapter values under the two estimates correlate at only **r = 0.35** (0.49 for the D200 pair,
Entry 63), so most of an adapter's contrast is estimator-specific noise, not a property of the adapter.

**What the table says beyond the readings, descriptively.** Under the natural-image estimate every one of
the 24 adapters leans toward body B - every A adapter reads *against* its own body and every B adapter
*toward* its own - so arm B on its own "rejects" at p 0.0002. That is the shared fingerprint main effect of
Section V-B, at 0.25 % of R_real, and an unpaired reading of arm B would have reported device-specific
transfer. The symmetric statistic cancels it (theta_sym -0.019 %). Under the flat-field estimate the
additive part is **ten times smaller** (-1.2e-05): the main effect lives in the natural-image estimate, not in
the generations alone, which fits a natural-image estimate carrying scene- and processing-correlated
components that a generator's residual also correlates with. It also explains the one large difference
between the two columns: the max-arm construction reads the main effect as an arm's own-body lean and widens
to 0.38 %, while with flat fields it tightens threefold to **0.12 %**. The symmetric limit, main-effect-free
by construction, stays between 0.05 and 0.07 %.

**Consequences for the paper.** (1) A third camera model - a 2013 smartphone, different brand, operating
system and sensor generation from the D200 - gives the same null, at limits no wider than the primary's:
max-arm 0.12 % with the good estimate, symmetric 0.05-0.07 %. (2) The estimator question (gap 2) is answered
with data: the spread is intrinsic to scoring generated images against any estimated fingerprint, not a
symptom of weak estimates. (3) The measurement hazard gets its clearest demonstration and a source: the
natural-image estimate. Flat-field estimation is the practical remedy an examiner can use.

---

## Entry 104 · 2026-09-25 · Defect D13 — every chain was a child of the Claude app, and closing the app killed them; the logon task never fired

At 19:47 on 24 Sep both running chains stopped mid-stage: G4b's generation at `p20b_B_s6` image 196, and the
G1 extension at `inv16kext_B_s3` step 15,600 of 16,000. The machine did not reboot (up since 23 Sep 09:39).
The chains had been launched with `nohup` from the Claude app's shell, so on Windows they belonged to the
app's process tree, and when the app closed the whole tree went with it; `nohup` does not survive that. The
"EINV queue restart" task of Entry 101 fires only at logon and there had been no logon, so its record read
"has not run". The GPU sat idle for 13.5 hours until the next check.

**Cost.** G4b lost nothing: generation resumes image by image and restarted at 196. The G1 extension lost
about 11 GPU-hours - adapters save only on completion, so `inv16kext_B_s3` retrains from step 0. No result
file was affected; nothing was measured on partial data.

**Fix.** (1) A second scheduled task, "EINV queue watchdog", runs `src/restart_queue.cmd` every 15 minutes.
(2) Chains are now started by Task Scheduler, not from an interactive shell: verified that both run with no
parent process and outlive the task instance that started them, so closing the app cannot reach them.
(3) `restart_queue.sh` detects chains through the Windows process table (`src/chain_running.py`, psutil)
rather than Git Bash's `ps`, skips a chain whose last log line ends in "measured" (finished) or reports
"FAILED" (left for a person to look at, never retried in a loop), and restarts the status page if it is down.
The worst case from here is an interrupted adapter plus up to 15 minutes. Checkpointing inside a
16,000-step adapter would shorten that further; it would change the training code of an arm already
under way, so it is not done.

---

## Entry 105 · 2026-09-26 · G4b result — *"no detectable transfer on a modern smartphone pair"*, with a positive lean (p 0.014) and the largest shared main effect of any paired design

Huawei P20 (EML-AL00, 2018), Daxing bodies 1104 (role A) and 1103 (role B), orientation 90. Twelve adapters
per arm at 2000 steps on 50 training images each, 250 generations per adapter (Entry 86), 6,000 images,
scored against this pair's own E2 estimates (90 photographs each). Gates (Entry 87): held-out AUC 1.000,
R_real 0.039229, split-half 0.635 / 0.541, cross-device kappa 0.462. `out/g4b_p20.json` (re-run on 26 Sep to
correct its entry label; per-adapter values identical to the first run, which is kept in `C:\D_offload`).

| quantity | value |
|---|---|
| arm A mean own-minus-other | +1.395e-04 (**12 of 12 positive**) |
| arm B mean own-minus-other | -9.206e-05 (**0 of 12 positive**) |
| additive part (A - B)/2 | **+1.158e-04 = 0.30 % of R_real** |
| intersection-union sign-flip p_A / p_B | **0.00024 (floor)** / 1.000 |
| theta_sym | **+2.373e-05 = +0.060 %** of R_real, Welch SE 1.013e-05, df 22.0, t 2.34, **one-sided p 0.0143** |
| symmetric one-sided 99 % limit | **0.1252 %** |
| max-arm limit | **0.4680 %** |

**Pre-declared reading (Entry 78, applied to G4b by Entry 85): neither the intersection-union test (arm B
p 1.0) nor the symmetric statistic (p 0.0143) rejects at one-sided 0.01 -> "no detectable transfer on a modern
smartphone pair", reported with both limits.**

**What the reading does not say, and must be carried beside it.**
1. *The symmetric estimate leans positive at p 0.014*, the smallest p of any 2000-step paired design (primary
   D200 +0.013 %, G5 -0.014 %, G6 -0.019 % / +0.023 %, FLUX +0.016 %, G2 +0.089 % at p 0.076). It is not
   resolved at the pre-specified level and is not claimed as transfer. Two things weaken it further: the
   symmetric construction under-covers once fingerprint-estimation error is included (0.81 against nominal
   0.99, Entry 97), so its p-values are anti-conservative; and see 2.
2. *The shared main effect is the largest in the study* - every one of the 24 adapters leans toward body A,
   arm A alone reaches the sign-flip floor, and the additive part is five times theta_sym. The two bodies'
   fingerprint estimates share an unusually large component (kappa 0.462, against 0.007 for the D200 pair),
   the model-level signature Daxing's P20 group is known to carry (its five-body analysis used residualised
   estimates for that reason). The symmetric statistic cancels a main effect that is equal in both arms; it
   cannot cancel one that differs between the arms' generations, and with a main effect this size a small
   asymmetry in it would be enough to produce a lean of +0.06 %. That is a candidate explanation, not a
   finding, and no test here separates it from a small transfer.
3. *The max-arm limit, 0.47 %, is inflated by that main effect* - it reads arm A's shared lean as own-body
   transfer - exactly as the natural-image estimate did for the iPhone 5c pair (0.38 %, Entry 103), where
   flat fields brought it to 0.12 %. Daxing ships no flat fields, so that remedy is not available here.

**Consequences for the paper.** The paired null extends to a 2018 smartphone - a fourth device class after
the 2006 DSLR, the 2013 iPhone 5c and the five-body compact and P20 groups - at a symmetric limit of 0.13 %.
The sentence that carries it states the p 0.014 lean and the dominant main effect in the same place, not in a
footnote. The G4 pair (P10 Plus) that failed the fingerprint-quality gate and this P20 pair together make one
point about modern phones: where the fingerprint can be estimated at all, the estimate carries a large
model-level component, and the paired design is what keeps it from reading as transfer.

---

## Entry 106 · 2026-09-27 · Pre-specification — M1 (pooled transfer across device pairs), H6 (a coverage-calibrated headline limit), P1 (diverse prompts at twelve adapters per arm)

Registered before any of the three is computed. **Stated plainly: all the individual estimates M1 pools have
already been seen** (Entries 50, 70, 79, 96, 103, 105). What keeps M1 honest is that its set of designs and its
readings are fixed here by rule, not chosen by outcome, and that it is reported whichever branch it lands in.

### M1 — is there a small transfer common to the device pairs that no single design resolves?

Four results lean positive at p 0.002-0.08 (16000 steps; content-matched sets; the P20 pair; the second
environment). A reader will ask whether the null is a small common effect below each design's resolution.

**Designs, by rule:** every paired design at the primary dose (2000 steps), standard training (not
content-matched, not marked, not full fine-tuning), scored with **natural-image** fingerprint estimates, each
expressed on its own scale: lambda = theta_sym / R_real and SE = Welch SE / R_real, in per cent.
- *Primary set — one design per independent device pair:* Nikon D200 primary (12 adapters per arm, ledger),
  Apple iPhone 5c (G6, 12, E2), Huawei P20 (G4b, 12). Three pairs, disjoint bodies and fingerprints.
- *Secondary set:* the three further D200 designs - second environment (G2, 6 per arm), second training set
  (G5, 3), FLUX.1-dev (6) - share the D200 bodies and fingerprint estimates with the primary, so they are first
  combined with it into one D200 estimate by inverse variance (an optimistic SE, because they share K; stated),
  then pooled with the iPhone and P20 pairs.
- *Sensitivities:* (i) the iPhone pair with its flat-field estimate instead of E2; (ii) every design's variance
  multiplied by 1/(1 - 0.432) = 1.76, carrying the share of variance that fingerprint estimation contributed
  in H2 (Entry 90), because the symmetric construction under-covers without it (Entries 93, 97).

**Statistics.** Fixed-effect inverse-variance pooled lambda, its SE, z, one-sided p, one-sided 99 % upper limit
(lambda + 2.326 SE). DerSimonian-Laird random effects with tau^2, Cochran's Q and I^2; if I^2 > 50 % the
random-effects estimate is the one quoted. `src/m1_pooled.py` -> `out/m1_pooled.json`.

**Readings, on the primary set:**
- pooled lambda > 0 with one-sided p < 0.01 **and** p < 0.05 under the variance-inflated sensitivity ->
  **"a small transfer common to the device pairs, lambda of about X %"**, reported as a finding; the headline
  becomes "transfer of order X %, at most Y %" rather than "none detected".
- one-sided p >= 0.01 -> **"no transfer common to the device pairs; pooled one-sided 99 % limit Y %"** - the
  paper's most general limit, across three camera models and two decades of sensors.
- p < 0.01 on the primary set but not p < 0.05 once inflated -> **"a pooled lean that does not survive
  fingerprint-estimation error"**; both reported, nothing claimed.

### H6 — a headline limit that covers at its stated 99 %

With both omitted variance components measured, the max-arm construction covers at 0.975, not 0.99 (Entry 97).
Using Entry 97's simulation unchanged (27 cells, 4,000 replications, estimation SD 1.224e-5 from H2,
training-set SD 2.201e-5 from G5), find the smallest multiplier c on a grid 1.00-2.50 (step 0.01) of the
max-arm half-width, t_0.995,k-1 * s / sqrt(k), such that the **minimum** coverage over all 27 cells is at
least 0.99. Apply c to the primary D200 ledger values: the calibrated limit becomes the paper's headline 99 %
limit, with c stated; 0.1507 % is kept as the nominal construction beside it. The variance components are
measured only for the D200 pair, so the other pairs keep their nominal limits, stated as nominal. This is a
construction, not a test: there is no outcome-dependent branch. `src/h6_calibrated_limit.py` ->
`out/h6_calibrated_limit.json`.

### P1 — diverse prompts at twelve adapters per arm

The primary generations are 83.5 % rifles ("sks" token, Entry 53), and the diverse-prompt null (Entry 22) rests
on three adapters per arm. **Design:** G6's 24 adapters (iPhone 5c, twelve per arm, already trained - the
largest paired set on disk, and the one with a flat-field estimate that removes the shared main effect; the
D200 primary adapters for seeds 3-11 are not on disk), each generating 250 images from v1's own five-caption
bank - *"a photograph of a street / of a room interior / of trees / of a building facade / of a table with
objects"*, caption j mod 5 with seed 770000 + j, the same seeds as the uniform bank, so image j pairs across
banks. 6,000 images, written to `gens/p5c_*_div` (junctions to C:), scored against both estimators exactly as
G6. Runs after the G1 extension, started through Task Scheduler (D13).

**Readings, natural-image estimate primary, flat-field beside it:**
- neither the intersection-union test nor the symmetric statistic rejects at one-sided 0.01 ->
  **"the null is not a property of the caption, at twelve adapters per arm"**.
- the symmetric statistic rejects at 0.01 with both arm means positive -> **"device-specific transfer appears
  under diverse prompts"**, which would reopen whether the single-caption null reflects a collapsed mode.
- otherwise, the estimate and limits only.
Descriptive, not tested: the paired difference in theta_sym between banks on the same adapters, with its SE;
the change in the additive part; the firearm share of a 250-image sample by the Entry 53 CLIP method.

---

## Entry 107 · 2026-09-27 · M1 result — *"no transfer common to the device pairs"*, pooled limit 0.072 %; H6 result — the calibrated headline limit is 0.175 %

### M1 (registered in Entry 106)

Per design, on its own scale (lambda = theta_sym / R_real, per cent; SE likewise):

| design | adapters per arm | lambda | SE |
|---|---|---|---|
| Nikon D200 primary | 12 | +0.0129 % | 0.0252 % |
| Apple iPhone 5c (E2) | 12 | -0.0188 % | 0.0269 % |
| Apple iPhone 5c (flat field) | 12 | +0.0233 % | 0.0192 % |
| Huawei P20 (G4b) | 12 | +0.0605 % | 0.0258 % |
| D200 second environment (G2) | 6 | +0.0894 % | 0.0541 % |
| D200 second training set (G5) | 3 | -0.0139 % | 0.0449 % |
| D200 FLUX.1-dev | 6 | +0.0156 % | 0.0453 % |

| pooling | quoted model | pooled lambda | SE | one-sided p | one-sided 99 % limit | I^2 |
|---|---|---|---|---|---|---|
| **primary: three independent pairs** | random (I^2 > 50 %) | **+0.0186 %** | 0.0228 % | **0.207** | **0.0715 %** (fixed: 0.054 %) | 57 % |
| secondary: D200's four designs combined (+0.018 %), then pooled | random | +0.0201 % | 0.0205 % | 0.163 | 0.068 % | 56 % |
| sensitivity (i): iPhone flat-field estimate | fixed | +0.0301 % | 0.0131 % | **0.0109** | 0.061 % | 0 % |
| sensitivity (ii): variance x 1.76 for estimation error | fixed | +0.0191 % | 0.0199 % | 0.169 | 0.065 % | 24 % |

**Pre-declared reading: the primary pooled one-sided p is 0.207 >= 0.01 -> "no transfer common to the device
pairs; pooled one-sided 99 % limit 0.072 %".**

What this adds, and what it does not.
1. *The most general limit in the study.* Three camera models - a 2006 DSLR, a 2013 phone and a 2018 phone -
   pooled, bound any transfer they share at **0.072 %** of each pair's real-image contrast (random effects;
   0.054 % fixed; 0.065 % once estimation error is carried). That is half the Nikon-only headline, and it
   generalises over device pairs rather than over adapter seeds on one pair.
2. *The answer to the near-positives question is "not resolved, and small if present".* Every pooled estimate
   is positive (+0.019 to +0.030 %). The flat-field sensitivity comes to p 0.011, just short of the threshold
   the primary reading uses, and it is the one variant that removes the iPhone pair's main effect - but it is
   a sensitivity, it does not survive carrying estimation error (p 0.17), and the primary is not close
   (p 0.21). The paper says it plainly: a common transfer of a few hundredths of a per cent is neither
   established nor excluded; one above 0.07 % is excluded.
3. *Heterogeneity is moderate and driven by one pair* (P20 +0.06 % against the iPhone's -0.02 %; Q p 0.10).
   With three pairs, tau^2 is poorly estimated; that is why the random-effects limit is the one quoted.
4. Caveat carried: fixed-effect pooling treats each design's Welch variance as known with a normal reference;
   the three pooled designs each have twelve adapters per arm, so this matters little for the primary set.

### H6 (registered in Entry 106)

Entry 97's simulation, unchanged, with both measured components (estimation SD 1.224e-5, training-set SD
2.201e-5). Worst-cell coverage of the max-arm construction at c = 1 is 0.9795 (Entry 97's 0.975 on a
different draw); the smallest multiplier with worst-cell coverage >= 0.99 is **c = 1.25**. With estimation
alone c = 1.00 already suffices (0.995), and with neither component the construction is conservative (1.000).

**Applied to the primary D200 ledger values: the calibrated one-sided 99 % limit is 0.1752 % of R_real,
against the nominal 0.1507 %.** This is the paper's headline limit from here on, stated with c; 0.1507 % is
the nominal construction beside it. The inflation comes entirely from training-set variation (estimated from
two contrasts, Entry 97, so c is itself approximate), which is the component the primary design cannot see
because every adapter of a body trained on the same fifty photographs.

---

## Entry 108 · 2026-09-27 · Novelty re-check, repository refresh, and the verifier extended to 65 checks

**Novelty (the Entry 02 sweep, re-run).** OpenAlex forward citations since 2025 of the six anchors, screened
for a sensor term and a generative term, compared with the 8 Sep run (`out/sweep_oa_2026-09-08.json` kept):
Yu 2021 79 -> 82 citing works, 0 hits; Chen 2008 58 -> 58, 2 hits (unchanged); SIREN 12 -> 12, 0; ProMark
17 -> 19, 0; Klier & Baier 0 -> 0; Lukas 2006 114 -> 121, 4 -> 5 hits. **One new hit:** Dabool and Alashwal,
"Learning-Free Detection of AI-Generated Images Using PRNU Sensor Noise for Pairwise Verification and Forensic
Generalization", IEEE ICMLT 2026 - real-versus-generated *detection* with PRNU, the same kind as the
already-cited AI-synthesis detection work; it does not ask whether a camera's fingerprint survives
personalization. Cite and distinguish. Web searches (PRNU / sensor pattern noise x LoRA, DreamBooth,
personalization, fine-tuning; memorization x camera fingerprint) found nothing asking this study's question.
**The WIFS 2026 accepted-paper list is still unpublished** (page last updated 28 Jan; conference 7-11 Dec,
Sendai) and remains the one channel to re-check before submission. Klier & Baier is now in FSI: Digital
Investigation vol. 56 (2026); the bibliography should cite that record.
Process note: OpenAlex's search endpoint returned HTTP 503 intermittently on 27 Sep and one run silently
produced an empty result; `oa_sweep.py` now resolves the anchors by their stored OpenAlex ids and backs off
longer, and the recorded run resolved all six with no retries.

**Repository refresh (local clone only; nothing committed or pushed).** `src/make_repo_v2.py`: 108 scripts
and 142 result files (65.2 MB), no machine path left in any shipped script or result. Changes to what is
shipped: the machine's operations tools (`status_server.py`, `chain_running.py`, `restart_queue.sh`) are
excluded; the new fingerprint manifests (`fp_5c`, `fp_p20b`, `fp_p10`) are included; the Daxing root is a
setting (`EINV_DAXING` in `einv_paths.py`) instead of a drive path; shell scripts take `EINV_V2` from the
environment and call helper scripts beside them. **Two items removed from the public copy:** the OpenAlex
contact address that `oa_sweep.py` carried (it now reads `OPENALEX_MAILTO`), and the Entry 66 note giving the
location of a credential, which had been published on 19 Sep. The working-tree log keeps that note - it is
append-only - and git history still holds the earlier text, so revoking the credential is the only real remedy.
Phrase overlap of the public log with the manuscript: 0.28 % of 8-word phrases (0.01 % with the filed
submission).

**`verify_v2.py`: 35 -> 65 checks, all agreeing.** Added, each recomputed from per-adapter or per-design
values rather than read from a summary: G5 (theta_sym, limit), G6 under both estimates (theta_sym, symmetric
and max-arm limits), G4b (theta_sym, p, max-arm), G1 (theta_sym, p), G2 (theta_sym, p), H1 (the two
single-predictor p values and the strength-body correlation), H2 (estimation share, inflated limit, smallest
bootstrap p), H4 (worst-cell coverage of both constructions), H5 (D and its exact permutation p), M1 (pooled
estimate, limit, p) and H6 (c and the calibrated headline limit). G1-extension and P1 checks follow when they land.

---

## Entry 109 · 2026-09-28 · Defect D14 — the G1 extension's measurement scored other arms and reported success

Chain 19 finished the G1 extension - ten adapters trained, 2,500 images generated - at 22:26 on 27 Sep, then
ran `T1_ARMSET=inv16kext t1_measure.py`. That script selected its arms through a list of known armset names,
and `inv16kext` was not on it: the name fell through to the script's default, the Entry 20 designed-mark arms,
which it re-summarised from their cached rows and exited 0. Chain 19, which predates the exit and output checks
of chain 20, logged "measured". Chain 21 then looked for `summary_inv16kext.json`, did not find it, logged
"G1-ext scoring FAILED" and stopped, and the watchdog, as designed, did not restart a failed chain. The GPU was
idle from 22:31 until the next check at 03:16, and P1 had not started.

**No result was damaged.** The fallback rewrote `out/t1/summary.json` (Entry 20) from its unchanged per-image
rows; it is identical in every value to the copy shipped to the repository that morning (checked field by
field). All 2,500 G1-extension images are complete (250 per adapter).

**Fix.** `t1_measure.py` gains the `inv16kext` arms and now **exits with an error on any armset it does not
know** instead of measuring its default - an unknown name can no longer produce a silent success. Chain 21
measures the G1 extension itself if its summary is missing, checks that the file exists, then scores it and
runs P1; it was relaunched through the watchdog at 03:17. This is the third instance of one failure class
(D12 generation, D13 process lifetime, D14 armset): a stage that reports success without its output. The
chains written since D12 check outputs, not log lines; chains 12-19 do not, and none of them will run again.

---

## Entry 110 · 2026-09-28 · G1 extension result — *"the 16000-step signal follows the fingerprint's sign"*: the one positive result is the fingerprint

Eight adapters per body at 16000 steps on training crops carrying each body's own fingerprint with its sign
reversed (Y(1 - 6 K_E1) + dither, rounded once; stored inverted contrast -2.85x for A, -1.83x for B, Entry 98),
250 generations each, 4,000 images. Scored by `src/g1_ext_score.py` -> `out/g1_ext.json`; the first three
adapters per arm reproduce Entry 98 exactly (checked in the script).

| | theta_sym | % of R_real | Welch SE | t | one-sided p (theta < 0) | reading |
|---|---|---|---|---|---|---|
| **all eight per arm (registered)** | **-5.810e-05** | **-0.163 %** | 2.173e-05 | -2.67 | **0.0106** | follows the fingerprint's sign |
| five new adapters alone (sequential check, Entry 99) | -6.243e-05 | -0.175 % | 2.848e-05 | -2.19 | 0.034 | follows the fingerprint's sign |
| strength-adjusted (Entry 89 covariate) | -5.81e-05 | -0.163 % | | -3.03 | 0.0047 | |

Power against the mirror of the unsuppressed effect: 0.79 (0.22 at three per arm).

**Pre-declared reading (Entry 77, threshold one-sided 0.05; Entry 99 sequential rule): met, and the
five-new-alone reading agrees -> "the 16000-step signal follows the fingerprint's sign" (it is
fingerprint-driven).** Stated with its level: the registered threshold for G1 was 0.05; at the 0.01 this study
uses for its headline tests, p = 0.0106 narrowly misses. The magnitude is the mirror image of the unsuppressed
result: -0.163 % against +0.160 % (Entry 68).

**Descriptive, not registered - within each body, inverted minus normal (`out/g1_within_body.json`).** The
causal contrast is the same body trained with its fingerprint in its own photographs versus inverted: only the
sign of the fingerprint differs.

| body | normal 16k own-minus-other (6) | inverted (8) | difference | Welch t | one-sided p |
|---|---|---|---|---|---|
| A (D200 no. 1) | +1.147e-04 | +1.05e-05 | **-1.04e-04 (-0.29 %)** | -3.15 | 0.0046 |
| B (D200 no. 0) | -0.05e-05 | -1.267e-04 | **-1.26e-04 (-0.35 %)** | -3.13 | 0.0060 |
| both, averaged | | | **-1.15e-04 (-0.32 %)** | z -4.42 | 1e-05 |

Adapted-weight norm does not differ between normal and inverted within a body (A +0.056, p 0.42; B +0.033,
p 0.71), so adaptation strength does not produce the drops.

**What this settles.**
1. *The 16000-step positive is the fingerprint.* Both bodies' generations move with the sign of the
   fingerprint in their own training photographs, by similar amounts. A main effect shared by both arms - of
   the training run, the environment or the generator - moves the two bodies' own-contrasts in opposite
   directions and cannot make both drop; the averaged difference is exactly the change in theta_sym between
   conditions (-5.81e-05 - 5.71e-05).
2. *The "one body carries it" asymmetry was a masking effect, not a property of one body.* In both conditions
   the generations share a lean toward body A's fingerprint (additive part +5.0e-05 normal, +6.9e-05
   inverted). It lifts arm A's own-contrast and lowers arm B's, so under normal training A shows the transfer
   and B's is masked, and under inversion B shows it and A's is masked.
3. *H1's confound is resolved.* Adaptation strength and body identity were collinear (Entry 89); inversion
   reverses the sign of the effect while leaving strength unchanged, which strength cannot do.

**What it does not settle.** One camera pair at this dose; the dose response is not monotone (8000 steps
-0.05 %, two adapters per body); the inverted fingerprint was stored at 1.8-2.9 times its natural contrast
yet the output shift is about the size of the natural effect, so transmission is not proportional to stored
amplitude and no gain is claimed; the within-body comparison was not registered.

**Consequence for the paper.** The thesis's closing clause - that the signal at eight times the adaptation
"is not the fingerprint template" - is reversed: at eight times the primary adaptation **the fingerprint itself
passes through personalization at about 0.16 % of its real-image contrast, and follows its own sign when it is
inverted in the training photographs.** The primary-dose null (calibrated 0.175 %; pooled across three pairs
0.072 %) is unaffected. At this size the transfer is still far below what attribution needs (Entry 50).

---

## Entry 111 · 2026-09-29 · P1 result — the null holds under diverse prompts with the natural-image estimate; with the flat-field estimate the same generations resolve a small transfer

G6's 24 adapters (iPhone 5c, twelve per arm), 250 images each from v1's five-caption bank (street, room
interior, trees, building facade, table with objects; caption j mod 5, seed 770000 + j), 6,000 images, scored
against both estimators as G6 (`out/g6_p5c_div.json`), compared with the same adapters' uniform-bank scores
(`out/p1_prompts.json`). Generation 03:30-19:03 on 28 Sep; measured and scored by 19:30.

| estimate | bank | theta_sym | % of R_real | t | one-sided p | arm means (A / B) | sym 99 % | max-arm |
|---|---|---|---|---|---|---|---|---|
| natural (E2, primary) | uniform ("sks") | -8.90e-06 | -0.019 % | -0.70 | 0.753 | -1.29e-04 / +1.11e-04 | 0.050 % | 0.381 % |
| natural (E2, primary) | **diverse** | -1.72e-06 | **-0.004 %** | -0.10 | **0.539** | -8.33e-05 / +7.98e-05 | 0.088 % | 0.311 % |
| flat field | uniform ("sks") | +1.37e-05 | +0.023 % | +1.22 | 0.118 | +1.6e-06 / +2.58e-05 | 0.072 % | 0.122 % |
| flat field | **diverse** | **+3.91e-05** | **+0.067 %** | **+3.20** | **0.0020** | **+1.73e-05 / +6.08e-05** | 0.119 % | 0.198 % |

Paired difference, diverse minus uniform, on the same adapters: natural +0.015 % (p 0.75), flat field +0.043 %
(p 0.20). Firearm share (Entry 53 CLIP method, 250-image samples): **uniform bank 243/250 = 97.2 %; diverse
bank 0/250 = 0.0 %** [0.0, 1.5].

**Pre-declared reading, natural-image estimate (primary): neither the intersection-union test (arm A p 0.99)
nor the symmetric statistic (p 0.54) rejects -> "the null is not a property of the caption, at twelve adapters
per arm".** The objection that the null reflects a population of rifle images is answered: the prompt bank
removes the rifles entirely and the natural-image null is unchanged (difference p 0.75).

**The flat-field estimate, registered to be reported beside it, lands in the other branch:** the symmetric
statistic rejects at 0.01 (p 0.0020) with both arm means positive -> **"device-specific transfer appears under
diverse prompts"**, at +0.067 % of the real-image contrast. The intersection-union test does not reject (arm A
p 0.16, arm B p 0.004). The two estimators disagree on the reading.

**How to read the disagreement - stated, not resolved.**
1. It is not a prompt effect: the diverse-minus-uniform difference is +0.043 % at p 0.20 under the flat-field
   estimate. The same adapters already leaned positive under the uniform bank (+0.023 %, p 0.12).
2. The two estimates see different things in the same images. The natural-image estimate carries the pair's
   large shared main effect (additive part -8.2e-05 to -1.2e-04) and more estimation noise; the flat-field
   estimate removes most of the main effect (-1.2e-05 to -2.2e-05) and has the larger real contrast (0.0586 against
   0.0474). Per-adapter values correlate at only 0.42 between them. A transfer of a few hundredths of a per
   cent is the kind of signal that one estimate could resolve and the other could not.
3. It sits with the other flat-field evidence: M1's flat-field sensitivity pooled to +0.030 % (p 0.011,
   Entry 107), and the G1 extension established that the fingerprint does pass through personalization at
   eight times the adaptation (Entry 110).
4. Weights against it: it is a secondary estimate, one of four estimator x bank combinations on the same
   adapters (Bonferroni over four gives p 0.008); the symmetric construction is anti-conservative once
   estimation error is carried (Entry 97), although flat-field estimates carry less of that error than
   natural-image ones; and +0.067 % is above what the transmission map predicts for a non-repeating pattern at
   the fingerprint's natural amplitude (roughly 0.02 % or less, Entry 72).

**Consequence for the paper.** The primary-dose statement cannot be "no transfer". It is: **with natural-image
fingerprint estimates, no transfer is detected at the primary dose on any device pair (limits 0.05-0.18 %;
pooled 0.072 %); with the better, flat-field estimate available for one pair, a small transfer of a few
hundredths of a per cent - +0.023 % under the single caption, +0.067 % under diverse captions - leans positive
and in one case is resolved.** Together with Entry 110 the picture is coherent: the fingerprint does pass
through personalization, at a level that is tiny at the standard dose, larger under heavy adaptation, and far
below what attribution needs in every case. Its detectability depends on the quality of the fingerprint
estimate - which is itself a finding for examiners. All registered experiments are now complete.

---

## Entry 112 · 2026-09-29 · Post-hoc quantities computed for the manuscript (descriptive and sensitivity analyses, not pre-specified)

Computed on the CPU by `src/fv/fv_derived.py` -> `out/fv_derived.json`, from result files only, and exposed to the
manuscript as macros by `src/fv/num_n6.py`. **None of these quantities was registered before its inputs were
seen.** They are what the manuscript needs beside the registered results: one power evaluation, one
small-sample sensitivity interval, one homogeneity statistic read from an existing file, three counts and one
descriptive decomposition. No registered reading changes.

| id | quantity | value | source | status |
|---|---|---|---|---|
| N-A1 | two-candidate attribution TPR at 1 % FPR if transfer sat at the **calibrated** limit (U_device 6.251e-05 = 0.1752 % of R_real, c = 1.25), same power model and inputs as Entry 50 (sigma_mu 5.23e-05, SE at 500 images 7.035e-05) | 500 images **0.053**; 5,000 images **0.110**; unlimited images **0.129** | `out/h6_calibrated_limit.json` primary_d200.U_device; `out/t3_power_v4.json` inputs; `verify_v2.tpr()` imported from the repository | post hoc, descriptive |
| N-A1 check | the same function at the nominal limit (U_device 5.3761e-05) | 0.043 / 0.084 / 0.097, equal to `t3_power_v4.json` and to the existing macros (the file's unlimited column differs by 8e-08: it was evaluated at a large finite G) | as above | reproduction |
| N-A2 | Hartung-Knapp-Sidik-Jonkman one-sided 99 % upper limit of the pooled random-effects estimate over the three pairs (D200, iPhone 5c E2, P20; DL tau^2; t with 2 df, t_0.99,2 = 6.96; scale q = 1.014, so the modified interval is the same) | **0.178 %** of each pair's R_real (pooled +0.0186 %, SE 0.0229 %, t 0.81, one-sided p 0.25) | `out/m1_pooled.json` per_design lambda_pct, se_pct of primary.designs | post hoc sensitivity |
| N-A2 check | DerSimonian-Laird limit recomputed the same way (normal reference) | 0.0715 %, equal to Entry 107 | as above | reproduction |
| N-A3 | the four D200 designs combined (primary, G2, G5, FLUX): I^2 and Cochran Q | I^2 **0 %**; Q 2.29 on 3 df, **p 0.51** | `out/m1_pooled.json` secondary.d200_combination | read from the M1 file (Entry 107) |
| N-A4 | the local 2000-step arms of the dose series | **3 adapters per body** (nomark s0-2, nomarkB s0-2), **500 generations each**, **3,000** in all; every image scored | `out/t1/dose_stats.json` doses.2000; `out/t1/summary_nomark*.json` arms.*.n; folder counts | count |
| N-A5 | study totals over the designs of Table 2 block B (listed below) | **211 adapters trained, 95,500 generations scored** | count macros in `paper/fv/numbers.json`; `paper/fv/work/inv_gens.json` | count |
| N-A6 | G2 (second environment, six per arm) additive part, (mean_A - mean_B)/2, which estimates b_A - b_B | +2.743e-05 = **0.077 %** of R_real (theta_sym 0.089 %) | `out/g2_pooled_six.json` per-adapter values | descriptive |

**What N-A1 and N-A2 change in the reading.** Nothing registered; both enlarge a margin the paper already
states. At the calibrated limit the best case for two-candidate attribution is 0.129 even with unlimited images
(0.097 at the nominal limit): transfer at the headline limit could not support attribution. With three pairs the
between-pair variance is poorly estimated, and a small-sample interval that carries that uncertainty reaches
0.178 % instead of 0.072 %. The pooled result keeps its estimate and p ("no common transfer; pooled +0.019 %");
0.072 % is the pre-specified construction and is not called the tightest bound (CLAIMS T8).

**N-A5, what is summed.** Adapters count once, where they were trained; generations count once, where they were
made. Summed: primary 24 / 12,000; second training set 6 / 3,000; second environment 12 / 6,000; FLUX.1-dev
12 / 6,000; full fine-tuning 6 / 3,000; iPhone 5c 24 / 6,000; P20 pair 24 / 6,000; Kodak five bodies 10 / 5,000;
P20 five bodies 10 / 5,000; 8000 steps 4 / 1,000; 16000 steps 12 / 3,000; inverted 16 / 4,000; content-matched
6 / 3,000; random crop 3 / 1,500; transmission map tiles 18 / 9,000, octave bands 10 / 5,000, DiffusionShield
3 / 1,500, E1 known pattern 5 / 2,500, E2 matched fields 6 / 3,000; base model 0 / 500 + 0 / 500; five captions on
primary seeds 0-2 0 / 3,000 and on the iPhone adapters (P1) 0 / 6,000 (adapters re-used). Not added: the
flat-field rows (rescore the iPhone images), the 16000-step replication (a subset), the 2000-step local arms
(seeds 0-2 of G2), and every detector, attribution, memorization and firearm count (they score existing images).
Trained but outside Table 2 block B, and so not in the totals: the objective arms (7 adapters, 3,500 images;
reported in S05 on stored energy and loss; with them the totals would be 218 / 99,000), the designed-mark ladder
(6 / 3,000) and the environment chain (5 / 3,000), which the paper does not report. Every local design's
generation count agrees with its folder count.

**Q1, the learning-rate schedule of the primary adapters (a record audit, no statistic).** The claim to be
checked (QUARRY Q1) was that seeds 0-2 of the primary pair used `CosineAnnealingLR(T_max = 2000)` stepped once per
optimizer update, so the rate ended at half its initial value, while seeds 3-11 used T_max = 1000 updates and
decayed to about zero. **The records do not confirm that split.**
1. Five of the six seed 0-2 adapters (A s1, s2; B s0, s1, s2) carry `train_meta.json` records whose fields and
   field order are exactly those written by `cells/S2_FINAL_cell.py`, whose scheduler is
   `CosineAnnealingLR(T_max = ceil(steps / GRAD_ACC))` = 1000 updates: the rate decays to about zero, as for
   seeds 3-11. The S2 cell of `notebooks/01_pilot.ipynb` (T_max = steps, half decay), which Q1 cites, wrote none
   of the six records: it writes neither clean_alpha, manifest_sha, model, dtype nor gpu. The run's protocol hash
   (f0b76681...) is the notebook's configuration with CLEAN_ALPHA = 1.6, so the configuration came from the
   notebook, but the hash does not cover the training code.
2. `A_raw_s0_r16`, the first adapter trained (29 Jul, 05:47), was trained by an earlier revision of the cell
   (already recorded as deviation 4 of `docs/E_INV_RESULTS_v2.md`; its lora_B_norm was recovered afterwards by
   notebook 04). Its schedule is not recorded. Its tail loss (0.174) lies among the other archive adapters
   (0.162-0.193), not among the local adapters trained with the notebook-01-style sampler sigma = u
   (0.232-0.283, 27 adapters); this is indirect evidence only, since the stacks also differ.
3. Seeds 3-5: the records' config_sha equals the protocol hash rebuilt from `notebooks/02_seed_ext.ipynb`, whose
   scheduler is T_max = 1000 updates. Seeds 6-11: the records carry the fields written by
   `notebooks/12_seed_ext2.ipynb`, same scheduler.
4. The half-decay variant is confirmed for the local stack only (`src/t1_ladder.py`, T_max = 2000 stepped per
   update): every designed-pattern, dose, inversion, second-environment, second-training-set, content-matched,
   iPhone and P20 arm.

So: **23 of the 24 primary adapters decayed to about zero; one (A seed 0) is undetermined; no primary adapter is
recorded with the half-decay schedule.** The archive-versus-local difference is real, and it comes with a second
one: the archive cells draw the noise level through the model's shifted FlowMatch sigma table, the local stack
uses sigma = u directly (the tail-loss levels above differ accordingly). Both arms of every local design share
the local schedule and sampler. Because the split is not confirmed, the descriptive comparison by schedule
(seeds 0-2 against 3-11) was not computed, and no schedule macro exists.

## Entry 113 · 2026-09-29 · Correction of two transcription errors in Entries 110 and 111 (no result changes)

Entries 110 and 111 are left as written; this entry corrects two numbers they print, each against its result
file. No statistic, test or reading changes, and the manuscript already prints the file values.

| entry | printed there | value in the result file | source |
|---|---|---|---|
| 110, item 2 | normal-training additive part (shared lean) at 16000 steps **+5.0e-05** | **+5.756e-05** (manuscript macro `nDoseAdditiveNormal`, printed 5.8e-05) | `out/t1/dose_stats.json` doses.16000.additive_part |
| 111, table row "flat field, diverse" | nominal max-arm limit **0.198 %** | **0.19746 %**, which rounds to **0.197 %** (manuscript macro `nPoneFlatMaxArm`) | `out/p1_prompts.json` estimators.FLAT.diverse.max_arm_pct |

---

## Entry 114 · 2026-09-30 · Post-hoc checks from the manuscript review: the power model's persistent term, and the adapter weights with same-seed pairs set aside (descriptive, not pre-specified)

Computed on the CPU by `src/fv/fv_sigma.py` -> `out/fv_sigma.json` and `src/fv/fv_weights_posthoc.py` ->
`out/fv_weights_posthoc.json`, from existing measurements only, and exposed to the manuscript by `src/fv/num_n8.py`
(83 macros, entry "114"; `fv_numbers.py` rebuilt: 1,614 macros, none changed). Both analyses were prompted by the
review of the manuscript. **Neither was registered before its inputs were seen. No registered reading changes.**

### A. The persistent term of the attribution power model

**What the model used.** The power translation (Entries 03, 50; `out/t3_power_v4.json`; main eq. power; S11)
treats the mean paired contrast of G generations of one adapter as N(lambda R_real, sigma_mu^2 + SE_500^2 500/G).
Its sigma_mu = 5.23e-05 is the archive sanity block's SG2 quantity (`cells/SANITY_BLOCK_cell.py`,
`E_INV_P0_v3/sanity_block.json`): for ONE adapter (A_raw_s0) against ONE fingerprint (K_A), the off-peak standard
deviation of its 500-image mean correlation surface (7.01e-05) and of the mean over half its images (8.42e-05).
Halving the images raised it by 1.20 instead of sqrt(2), and sigma_mu^2 = 2 sd_full^2 - sd_half^2 (recomputed
5.23e-05; image part 4.67e-05). It is a property of the correlation surface at non-zero lags, never measured on the
zero-lag paired contrast; Entry 03 had flagged it for a source check. SE_500 = 7.035e-05 is the mean 500-image
standard error of the per-image paired contrast of A_raw_s0 and B_raw_s0 (local re-measurement; D7).

**Data.** The archive's per-image rows of all 24 primary adapters (Drive mirror `E_INV_P0_v3/csv/s5_measure.csv`
seeds 0-2, `E_SEEDEXT/csv/b3_measure.csv` seeds 3-5, `E_SEEDEXT2/csv/sx2_measure.csv` seeds 6-11; SHA-256 in the
output), 500 generations per adapter on the seed bank 770000 + j shared by every adapter. The 24 adapter means
reproduce `FINAL_LEDGER.json` primary.per_adapter_A/B exactly.

**Model.** Within each arm, d[a,j] = m_arm + u_a + v_j + e_aj (adapter a, seed j): sigma_u is persistent per
adapter, sigma_v the seed (content) effect shared by the arm's adapters, sigma_e the adapter-by-seed residual. The
design is balanced, so the ANOVA estimates equal REML inside the parameter space (statsmodels MixedLM agrees);
sigma_u and sigma_e are pooled over the arms. An examiner who holds ONE personalized model and generates G images
with fresh seeds sees sigma_u^2 + (sigma_v^2 + sigma_e^2)/G: u_a persists, v_j and e_aj average out. So sigma_u is
the model's persistent term and (sigma_v^2 + sigma_e^2)/500 its SE_500^2. sigma_u is measured between adapters that
share one training set and one fingerprint estimate per body, so it excludes those components (H6: training-set
component 2.2e-05 per arm), from which the model's examiner, who knows the main effects, is assumed free.

| id | quantity | value | source (`out/fv_sigma.json`) | status |
|---|---|---|---|---|
| S-1 | persistent per-adapter component of the paired contrast, sigma_u (pooled REML) | **0** (on the boundary; arm A 4.8e-06, arm B 0) | primary_crossed_model.sigma_u_estimate | post hoc |
| S-2 | test of sigma_u > 0 | F 0.88 on 22 and 10,978 df, **p 0.62** | .pooled.F_adapter, F_p_upper | post hoc |
| S-3 | exact 95 % interval for sigma_u (normal theory) | **0 to 4.1e-05**; one-sided 95 % upper 3.6e-05, 99 % upper 4.8e-05 | .sigma_u_interval_exact | post hoc |
| S-4 | bootstrap intervals, 20,000 replicates | over adapters 0 to 3.0e-05; over adapters and seeds 0 to 5.5e-05 | .sigma_u_bootstrap_* | post hoc |
| S-5 | coverage of the one-sided 95 % upper limits, data simulated from the fitted model (200 sets per cell) | exact 0.930-0.955 with normal u, down to 0.875 with t(3) u; adapter bootstrap at most 0.795 (under-covers); adapter-and-seed bootstrap at least 0.965 (over-covers: a resampled seed bank holds about 316 distinct seeds) | .interval_calibration_simulation | method check |
| S-6 | the archive sigma_mu = 5.23e-05 against these data | one-sided **p 0.005**; above the one-sided 95 % upper limit of all three methods (the largest is 4.7e-05) | .archive_sigma_mu_one_sided_p | post hoc |
| S-7 | spread of the 24 adapter means within arm | **4.4e-05** observed; image noise alone predicts 4.7e-05; with the archive sigma_mu 7.0e-05 | .sd_adapter_means | descriptive |
| S-8 | split-half correlation of the adapter means (even vs odd seeds) | -0.14 | .split_half_r | descriptive |
| S-9 | per-image sigma_v and sigma_e | 1.2e-03 and 1.0e-03: the seed carries **58 %** of one adapter's per-image variance; the arms' seed means correlate -0.84 (own-minus-other) | seed_effect_and_image_noise | descriptive |
| S-10 | SE_500 of one adapter over all 24 archive adapters | 7.21e-05 (6.88-7.55e-05); the model's 7.035e-05 is about 2 % lower (A_raw_s0 and B_raw_s0 read 7.17e-05 and 6.93e-05 in the archive rows) | .SE_500_single_adapter | check |

**Two-candidate attribution at 1 % FPR** (`verify_v2.tpr()` imported from the repository; SE_500 = 7.035e-05).
sigma only lowers the TPR, so the sigma = 0 rows bound every value of the persistent term.

| sigma | limit | TPR, 500 images | 5,000 | unlimited | images for 0.5 | for 0.9 |
|---|---|---|---|---|---|---|
| 5.23e-05 (archive) | nominal, 0.151 % | 0.043 | 0.084 | 0.097 | never | never |
| 0 (the paired estimate) | nominal | **0.059** | **0.54** | 1 | **4,634** | **11,145** |
| 4.1e-05 (paired 95 % upper) | nominal | 0.048 | 0.120 | 0.154 | never | never |
| 5.23e-05 (archive) | calibrated, 0.175 % | 0.053 | 0.110 | 0.129 | never | never |
| 0 (the paired estimate) | calibrated | **0.075** | **0.69** | 1 | **3,428** | **8,244** |
| 4.1e-05 (paired 95 % upper) | calibrated | 0.059 | 0.161 | 0.210 | never | never |

The archive rows reproduce `t3_power_v4.json` and the existing macros (nAttTprTwo*, nAttTprTwo*Cal,
nAttTprZero*, nAttZeroG*Exact, nAttTenxG*Exact) to better than 1e-6; five and fifty candidates are in the file. One
half is unattainable at any image count once sigma exceeds 2.3e-05 (nominal) or 2.7e-05 (calibrated), and the
paired interval straddles both. At ten times the nominal limit one half needs 47 / 68 / 107 images for 2 / 5 / 50
candidates at sigma = 0 (49 / 73 / 122 at the archive sigma_mu). With the 24-adapter SE_500 the sigma = 0 counts rise
about 5 % (4,864 for one half).

**Seed matching (descriptive).** The seed effect is shared by adapters of both bodies at a given seed, so an
examiner who also generates, with the same seeds and caption, from a reference adapter trained on the other
candidate camera and scores (d_A + d_B)/2 cancels most of it and needs no main-effect calibration. The 500-image
SE of that contrast is **3.5e-05** (RMS over the 144 cross-arm adapter pairs), half the single-adapter value. At
sigma = 0 one half then needs **1,178** images at the nominal limit (0.9: 2,834; TPR 0.209 at 500 images) and 872 at
the calibrated limit. The transfer at which 50 images give a two-candidate TPR of one half is 9.6 times the nominal
limit with fresh seeds (9.9 at the archive sigma_mu; 8.3 times the calibrated limit) and **4.9 times** with seed
matching (4.2 calibrated). Any difference between the suspect's and the reference's training sets is persistent and
is not in this design.

**Secondary designs (descriptive; same model, local stack).** G2 (second environment, 2000 steps, 6 + 6 adapters x
500 images): sigma_u 4.7e-05, F p 0.029, entirely from one adapter, `nomark_s5`, whose own-minus-other contrast is
+2.15e-04 (0.603 % of R_real), 4.0 image-noise SEs from the other five body-A adapters; without it sigma_u is 0
(p 0.60). 16000 steps (6 + 6 x 250): sigma_u 0 (p 0.94). The persistent term is design-dependent and not normal:
one adapter of the 48 examined deviates persistently by more than the headline limit.

**What part A changes in the paper.**
1. sigma_mu = 5.23e-05 is not the persistent component of the paired contrast: it is a correlation-surface
   quantity, and the 24 primary adapters exclude it (p 0.005). It can stay only as a named sensitivity value, not
   as "the measured spread between one body's adapters".
2. "At most 0.097 however many images" and "no number of images lifts the two-candidate rate above 0.097" (abstract,
   I, V-D, VIII, S11, the caption of Fig. S10, Table S16) are **not supported**: they hold only at sigma =
   5.23e-05. For the examiner of eq. (power) and every value of the persistent term: at the nominal limit a TPR of at
   most 0.059 with 500 images and 0.54 with 5,000, and at least about 4,600 images for one half (0.075, 0.69 and
   3,400 at the calibrated limit). Whether any number of images reaches one half is **undecided**: the primary
   design's persistent component is 0 with a 95 % upper limit (4.1e-05) above the level (2.3e-05) at which one half
   becomes unattainable, and G2 shows that one adapter can deviate persistently by more than the limit.
3. "The most favorable examiner" is not the most favorable: seed-matched scoring against a reference adapter of the
   other camera halves the image noise (one half at about 1,200 images at the nominal limit). The model's examiner
   should be named by what it assumes: both fingerprints and the main effects known, one model's fresh generations.
4. "Roughly an order of magnitude below what attribution from tens of images needs" holds with fresh seeds (9.6-9.9
   times the nominal limit for one half from 50 images), not with seed matching (4.9 times); "five to ten times"
   covers both.
5. **Not changed:** every limit and every registered reading; closed-set attribution at chance (Entry 07); the
   ten-times statements (47-49 images for two candidates for sigma from 0 to 5.23e-05).

### B. Adapter weights with same-seed pairs set aside

Entry 95 grouped the H5 adapters by training batch. The explanation is the training seed: adapter j of body A and
adapter j of body B were trained with seed j, which `t1_ladder.train_arm()` sets (line 242) before PEFT adds the
adapter with `init_lora_weights="gaussian"` (line 249) and which also drives the batch order (line 264) and the noise
(line 279); every adapter's `train_meta.json` carries its seed. Same-seed adapters therefore share their
initialization and random stream while their photographs differ. From the cosine matrices of
`out/h5_weight_signature.json`:

| quantity (`out/fv_weights_posthoc.json` doses.*) | 2000 steps | 16000 steps |
|---|---|---|
| six same-seed pairs (all cross-body), mean cosine (range) | **0.526** (0.516-0.538) | **0.480** (0.476-0.484) |
| largest cosine of any other pair | 0.018 | 0.026 |
| 30 same-body pairs (none shares a seed), mean (min) | 0.0163 (0.0146) | 0.0251 (0.0236) |
| 30 cross-body pairs of different seeds, mean (max) | 0.0108 (0.0117) | 0.0151 (0.0158) |
| seed-stratified D_strat = same-body mean - cross-body different-seed mean | **+0.0055** | **+0.0099** |
| exact one-sided p, body labels swapped within seed pairs (64 relabellings, 32 distinct partitions) | **0.031 = the floor** (next value 0.0019) | **0.031 = the floor** (next 0.0034) |
| every same-body cosine above every cross-body different-seed cosine | yes (margin 0.0029) | yes (margin 0.0078) |
| same-batch pairs of different seeds vs different-batch pairs | 0.0135 vs 0.0136 (-0.0001) | 0.0201 vs 0.0201 (-0.00008) |

The registered D (-0.080 and -0.068, p 0.93) is reproduced from the matrices. Batch adds nothing within same-body
pairs either (2000 steps 0.0162 vs 0.0164) or within cross-body different-seed pairs (0.0107 vs 0.0108).

**Reading.** (1) The registered H5 test is uninformative: the six same-seed pairs, 19-32 times as alike as
same-body pairs, enter only the cross-body mean and make D negative whatever the weights carry about the body. Its
registered reading stands as worded but says nothing about the body. (2) With same-seed pairs set aside the weights
separate the two training sets completely at both doses. The within-seed-pair relabelling reaches its floor, 1/32,
which cannot reach the 0.01 used for H5, so this is an observation, not a test. (3) The two bodies' adapters were
trained on different photographs of different scenes, so weights that separate the training sets are expected from
content and are **not evidence that the fingerprint is in the weights**. Separating the two needs content held fixed
while the fingerprint changes: a weight-level test registered before looking on the G1 normal and inverted
16000-step adapters (same photographs, fingerprint sign reversed, seeds 0-5 shared within each body; relabelling
within seed pairs gives 2^5 partitions per body, 1,024 over both, floor about 0.001), or the same scenes
photographed by both bodies (content-matched or swapped training sets) with enough seed pairs (E3's content-matched
arms have three, floor 1/4). (4) **Entry 95's batch reading is superseded by the seed explanation**: the same-batch
excess (+0.102 and +0.092) is the six same-seed pairs, which always fall within one batch; among different-seed pairs
batch adds nothing. The weights are organized by training seed, not by "when and where an adapter was trained", and
the corroboration of G2 drawn from it in Entry 95 lapses (CLAIMS O5 had already dropped it).

**What part B changes in the paper.** The numbers CLAIMS X7 lacked now exist (nWtSameSeed*, nWtCrossNoSeed*,
nWtStratD*, nWtStratP*, nWtStratFloor, nWtSameBodyMin*, nWtCrossNoSeedMax*, nWtSameBatchNoSeed*, nWtDBatchNoSeed*),
and the two "Needs macros" places in S13 can be filled. "The separation cannot be attributed to the fingerprint"
stands and is the ceiling. **Not changed:** the registered H5 reading and every image-domain result.

**Register rows for the lead** (not added to the register table: this run appends only). R12: sigma_mu = 5.23e-05 as
the persistent per-adapter term of the power model, and "TPR at most 0.097 however many images" (Entries 03, 50) -
superseded: the paired contrast's persistent component is 0 (95 % interval 0 to 4.1e-05; the archive value
p 0.005), Entry 114. R13: "the adapter weights are organised by training batch; corroborates G2" (Entry 95) -
superseded: the batch excess is the same-seed pairs, Entry 114.

### Addendum · 2026-09-30 · Corrections after the independent check of this entry

An independent recomputation from the raw inputs (`paper/fv/work/check113/`, written before the scripts above
were read) reproduced every number of this entry; the three bootstrap limits differ by Monte Carlo noise only.
It corrected four reasoning steps and three presentation details. The scripts were amended and re-run
(`fv_sigma.py`, `num_n8.py`; `fv_weights_posthoc.py` re-run unchanged); `fv_numbers.py` now builds 1,644 macros.
No registered reading changes, and no number above is altered in the result files except where named here.

1. **G2 outlier (S-table "Secondary designs").** "4.0 image-noise SEs" divided by the SE of one adapter mean and
   ignored the noise in the mean of the other five and `nomark_s5`'s larger residual variance
   (1.44 times the pooled value). With its own per-seed deviations the
   distance is **3.0 SEs** (1.87e-04, SE 6.2e-05;
   3.6 with the pooled model's SE of a difference). The deviation persists over the
   seed bank (positive in all four quarters; even seeds 1.2e-04, odd 2.6e-04), but as
   the most extreme of the 48 adapters examined its family-wise p is **0.12**,
   and its 95 % interval (6.55e-05 to 3.09e-04) barely clears the nominal
   limit (5.38e-05). "One adapter deviates persistently by more than the headline
   limit" and "the persistent term is heavy-tailed" are **not established**; the G2 F p (0.029) is somewhat
   anti-conservative for the same reason. Macro `nAttGtwoOutlierZ` now prints 3.0.
2. **S-6 needs a qualifier.** The exclusion of the archive sigma_mu holds under normal u. Simulated with heavier
   tails (200,000 data sets): one-sided p 0.005 (normal),
   0.009 (t, 5 df), 0.022 (t, 3 df). With t(3) u
   the exact one-sided 95 % limit covers 0.90 at sigma_u = 4.1e-05 and
   0.86 at 5.23e-05 (S-5's 0.875 came from a grid ending at
   4e-05). The correction to "0.097 however many images" stands, because the estimate is 0.
3. **Seed matching is not a floor.** `verify_v2.tpr()` uses one SE for both hypotheses. Under the null (suspect and
   reference trained on the same camera) the seed term cancels exactly and the 500-image SE is
   3.20e-05 to 3.41e-05 (reference trained on body B or A;
   RMS 3.31e-05), not 3.55e-05. Averaged over the two reference cameras, one half needs about
   **1024** images at the nominal limit (757 calibrated), not 1,178; the
   TPR at 500 images is **0.26**, not 0.209; the fifty-image multiple is **4.5**
   times the nominal limit, not 4.9. With many reference adapters the null SE falls to
   2.34e-05: about 512 images, 3.2
   times. A difference between the suspect's and the reference's training sets (H6 sd / sqrt 2) raises the single-
   reference count to about 1874. "Five to ten times" (part A, item 4) should
   read **"about three to ten times"** (the check's own arithmetic, which used the alternative SE for many references,
   said "about four to ten").
4. **The training-set component applies to the fresh-seed examiner too.** No examiner can know the main effect of
   the suspect's own training set, so "from which the model's examiner, who knows the main effects, is assumed free"
   (Model paragraph) holds only for an idealised examiner. With the H6 training-set sd (2.2e-05, `nLimTrainSD`) as
   the persistent term and sigma_u = 0: nominal TPR 0.055 (500 images),
   0.27 (5,000), 0.55 (unlimited); one half at about
   **49718** images; calibrated 0.70 unlimited,
   about 10409 images. The sigma = 0 rows remain upper bounds. H6's value rests
   on two contrasts.
5. **S-4 names no method.** Both bootstraps estimate sigma_u^2 as the pooled within-arm variance of the resampled
   adapter means minus sigma_e^2 sum_j w_j^2, with sigma_e^2 held at its full-data value (w_j the multiplicity of
   seed j over 500). Recomputing the two-way ANOVA on each seed-resampled matrix double-counts the image noise of a
   seed drawn twice (kept as a labelled diagnostic in the file).
6. **Rounding.** `nWtSameBatchNoSeedTwo` printed 0.013 beside `nWtTwoCrossBatch` 0.014, which reads as a 0.001 gap
   where the difference is -0.00015; it now prints three significant figures (0.0135), and the manuscript prints
   the difference. The coverage macros (S-5) print two decimals (Monte Carlo SE 0.015-0.03 at 200 sets per cell).
   S-10: the 24-adapter SE_500 (7.208e-05) is 2.5 % above 7.035e-05, not "about 2 %".
7. **Table S16** needed macros at sigma = 4.1e-05 for five and fifty candidates and sigma = 0 image counts for
   them; they now exist (`nAttTpr{Five,Fifty}*PairedHigh`, `nAttZeroGFiftyExact{Five,Fifty}`).
8. **Register row for the lead** (not added to the register table: append-only). R14: Entry 95's "this closes the
   scope item ... the black-box result is not hiding a white-box channel that a party holding the weights could
   exploit" - superseded: the registered weight test is uninformative (the same-seed pairs decide its sign), so it
   closes nothing; with same-seed pairs set aside the weights separate the training sets, which cannot be
   attributed to the fingerprint (part B above).

**Revised licence for the paper (supersedes part A items 2-4 where they differ).** For the examiner of eq. (power)
(both fingerprints and their main effects known, one model's fresh generations) at the nominal limit: TPR at most
0.059 with 500 images and 0.54 with 5,000; at least about 4,600 images for one half (0.075, 0.69, 3,400 calibrated).
Whether any number of images reaches one half is undecided: the persistent component is estimated at 0 with a 95 %
upper limit (4.1e-05) above 2.3e-05, and a training-set component of H6's size would alone push one half to about
50,000 images. An examiner who also generates the same seeds from a reference model of the other camera needs about
1,000 (about 500 with many references). Attribution from about fifty images needs about ten times the nominal limit
with fresh seeds and three to five times with seed-matched references.

---

## Entry 115 · 2026-09-30 · Firearm-classifier validation re-labelled by the author (supersedes the AI-made validation labels of Entry 53)

**Why.** Entry 53 validated the CLIP zero-shot firearm classifier on 64 generations and called the comparison
labels "manual". They were not made by a person: `out/t6_weapon_clip.json` `validation.labeller` records an AI
model ("Claude (visual inspection of 480-960px downscaled copies, blind to classifier output; grids shuffled)").
On 30 Sep 2026 the author chose to label the 64 images in person rather than report AI-made labels as data.

**Procedure.** The same 64 images (Entry 53's stratified sample, seed 20260913, strata unchanged) were shown to
the author by eye on one blind page: order shuffled with seed 20260930 and no classifier output shown, under the same written criterion as `out/t6_weapon_validation_labels.json`
("firearm = a recognisable gun (stock/receiver/trigger/barrel), including distorted gun-like objects; knives,
swords and polearms are NOT firearms"). The page export is stored verbatim and mapped to image ids by
`src/fv/fv_firearm_author.py` -> `out/t6_weapon_validation_labels_author.json` (labels keyed by image id, plus the
verbatim export) and `out/t6_weapon_validation_author.json` (the comparisons). The AI-made label file
`out/t6_weapon_validation_labels.json` and the `validation` block of `out/t6_weapon_clip.json` are left unchanged.

**Result.** The author labels 27 of the 64 images firearms.

| comparison | agree / 64 | Cohen's kappa |
|---|---|---|
| classifier vs author (**now the reported validation**) | **57** (89.1 %) | **0.777** |
| author vs the earlier AI-made labels | 59 | 0.842 |
| earlier AI-made labels vs classifier (Entry 53, as previously reported) | 56 | 0.748 |

Classifier vs author: 24 both firearm, 33 both not; the classifier calls 4 images firearms that the author does
not (two in the 0.2-0.8 score stratum at p_firearm 0.70 and 0.76, two in the classifier-positive adapter stratum
at 0.99 and 0.995) and misses 3 that the author calls firearms (p_firearm 0.44, 0.44 and 0.017; one borderline,
two classifier-negative adapter images). By stratum (agree/n): base-model positives 8/8, adapter positives 14/16,
adapter negatives 14/16, five-caption negatives 16/16, borderline 5/8. The author and the AI-made labels disagree
on 5 images, all in the borderline or adapter strata.

**What changes.** Only the validation statistics the paper prints: the macros `nMemFirearmValAgree` (56 -> 57),
`nMemFirearmValKappa` (0.75 -> 0.78), `nMemFirearmValFalsePos` (3 -> 4) and `nMemFirearmValFalseNeg` (5 -> 3) now
read `out/t6_weapon_validation_author.json` (entry "115"; `nMemFirearmValN` = 64 is unchanged). The supplement (S12)
states that the author labelled the images by eye, blind to the classifier; no statement that an AI model
labelled them remains in the manuscript.

**What does not change.** The firearm shares themselves (97.2 % of the iPhone 5c adapters' generations under the
training caption, 83.5 % across the eight D200 adapters on the local stack, 0 % under the everyday captions, and
the per-adapter shares of Entry 53) are outputs of the classifier, not of the labels, and are unchanged. The
validation is still stratified and describes the direction of the classifier's errors, not a correction to the
shares. No fingerprint result depends on it.

---

## Entry 116 · 2026-09-30 · Pre-specification — closing checks before submission (seed-bank calibration, end-to-end examiner test, cross-body scene audit, normal-versus-inverted weight test, closed-set expectation)

Registered before any of the analyses below is computed. They answer the open items of `paper/fv/REVIEW_REPORT.md`
§3.6 and §4 (items 2, 4, 6, 7, 8, 9) and six small quantities the manuscript prints as "—" or not at all.

**What has already been seen, stated plainly.** (1) A scratch probe for the review (`seedbank_probe.py`, in a
session scratchpad, nothing written to `out/`) added the symmetric part of the seed-bank term to the H6 simulation and
found c of about 1.58-1.68 and a calibrated limit of about 0.21 %; item 1 is therefore not blind to its likely
direction, and what this entry fixes is the method and the rule, with the probe's form kept as a named sensitivity.
(2) The review recomputed the G2 max-arm limit (about 0.588 %, item 6a). (3) No cosine between a normal and an inverted
16000-step adapter has ever been computed: H5 (Entry 95) and Entry 114 part B used the normal 16000-step and the
2000-step adapters only. (4) Items 2, 3 and 5 have never been run in any form.

**Common rules.** Scripts in `src/fv/`, results in new files in `out/` (names below; none exists today); any job over
~20 minutes appends its per-unit rows to disk and resumes from them. The GPU is used only for DINOv2 inference
(item 3). Numbers reach the manuscript only as macros of a new `src/fv/num_n9.py`. Status: **item 4 is the only
registered test at a level (0.01, one-sided); item 1 is a construction rule that can change the bound of record;
item 2 is a registered model check with a fixed agreement criterion; item 3 is a data-audit rule with a fixed
sensitivity; items 5 and 6 are descriptive.** Nothing here alters a registered reading of an earlier entry except
where item 1's rule says so.

### 1. Seed-bank calibration of the headline limit (hostile-r2-13)

**Why.** All 24 primary adapters generate from one 500-seed bank (770000 + j). The seed effect carries 58 % of one
adapter's per-image variance (Entry 114, S-9). A fresh bank would move every adapter of an arm by that arm's bank mean;
the within-arm SD and the Welch SE do not see this term, and H6 (Entry 107) did not include it.

**Data.** The per-image rows of the 24 primary adapters exactly as Entry 114 read them (`fv_sigma.primary_rows()`,
imported, not re-implemented; the three Drive CSVs' SHA-256 checked against `out/fv_sigma.json`; the adapter means
asserted equal to `FINAL_LEDGER.json` primary.per_adapter_A/B): d_A = rho(K_A) - rho(K_B) on arm-A images,
d_B = rho(K_B) - rho(K_A) on arm-B images, 12 x 500 per arm, seed j shared by all 24.

**Estimation.** Entry 114's crossed model per arm, d_X[a,j] = m_X + u_Xa + v_Xj + e_Xaj, with the seed effect made
bivariate across arms, (v_Aj, v_Bj) ~ (0, Sigma_v):
- sigma_vX^2 = (MS_seed - MS_resid)/12, Entry 114's ANOVA estimate per arm;
- cov(v_A, v_B) = the covariance over the 500 seeds of the two arms' seed means. It needs no image-noise correction,
  because the two arms share no adapter;
- the symmetric seed variance, the part that enters theta_sym, estimated directly as
  sigma_vs^2 = var_j(s_j) - (sigma_eA^2 + sigma_eB^2)/48, with s_j = (mean_a d_A[a,j] + mean_a d_B[a,j])/2. Its one-sided
  95 % upper limit comes from 20,000 bootstrap resamples of the 500 seeds, with sigma_e^2 held at its full-data value
  (Entry 114 addendum, item 5).
The bank term of a replication is (b_A, b_B) ~ N(0, Sigma_v/500), added to every adapter of each arm. It enters
theta_sym as (b_A + b_B)/2 and the additive part as (b_A - b_B)/2. No double counting with H6's components: the
estimation SD (H2) was measured with the generations held fixed, and the training-set SD (G5) from shifts between
designs generated on the same bank (770000 + j).

**Simulation.** `src/fv/fv_seedbank.py` -> `out/fv_seedbank.json`. H6 is kept unchanged: the 27 cells, 4,000
replications, master seed 106061 and draw order of `h6_calibrated_limit.simulate()` with `h4_coverage2.draw()`, and
the estimation SD 1.224e-05 and training-set SD 2.201e-05 read from `out/h6_calibrated_limit.json`. The only addition
is a bank shift per replication, drawn from an independent generator (seed 116001). This gives common random numbers:
before adding the term, the script asserts that with it set to zero it reproduces H6's worst-cell coverage exactly at
c = 1 (0.9795) and at c = 1.25 (0.9900).

**Target.** The minimum over the 27 cells of max-arm coverage is at least 0.99, with c on H6's grid 1.00-2.50 in
steps of 0.01, extended in the same steps if 0.99 is not reached.

**Rule (point estimate of Sigma_v, the 4,000-replication common-random-number run):**
- worst-cell coverage at c = 1.25 >= 0.99 -> **"the seed-bank term does not change the calibration"**. 0.175 %
  (c = 1.25) stays the bound of record, and the paper says the term was included.
- worst-cell coverage at c = 1.25 < 0.99 -> the smallest c* that reaches 0.99 is applied to the ledger values exactly
  as H6 applied c: U_X = mean_X + c* t_{0.995,11} s_X/sqrt(12), U = max(U_A, U_B), then divided by R_real. **That limit
  becomes the bound of record**, described as calibrated for adapter, fingerprint-estimation, training-set and
  seed-bank variation. H6's 0.175 % (c = 1.25) and the nominal 0.1507 % are printed beside it. The rates derived at
  the calibrated limit (Entry 112 N-A1, the calibrated rows of Entry 114) are recomputed at the new limit as derived
  quantities.

**Reported beside, descriptive; the rule does not use them:**
- (i) the probe's form: the symmetric part only, added as a shift shared by both arms;
- (ii) the symmetric variance at its 95 % upper limit, with sigma_vX^2 kept and the covariance raised to match;
- (iii) the symmetric construction's worst-cell coverage with the term;
- (iv) a 20,000-replication rerun at c* (fresh master seed 116002), with its Monte Carlo SE.

**Scope.** Per-image rows are analysed here for the primary D200 pair only. Every other design also shares one bank
and keeps its nominal limit, stated as such. The term concerns re-running the design with a fresh bank: an examiner
who generates with fresh seeds already carries the seed effect as image noise in SE_500, so eq. (power) is unaffected.

### 2. End-to-end examiner test (review §4 item 4)

**Question.** Does a transfer planted into real generated images at a known size come out of the paper's own scoring
and its two-candidate examiner at the rate that eq. (power) predicts?

**Images and roles** (primary archive generations on local disk, `C:/D_offload/einv_v2_data/gens` for seeds 0-2 with
500 images each, and `gens_ext` for seeds 3-11 with images 0-249):
- *Calibration of the planting amplitude:* adapters s0-s2 of both bodies, **images 250-499** (750 per body). These
  images are never examined.
- *Examined (held-out) adapters:* s3-s11, nine per body, 18 in all, **images 0-249**.
- *Main effect known to the examiner:* m_X is the mean unplanted paired contrast of the other eleven adapters of arm
  X over images 0-249, leaving the examined adapter out.

**Planting.** For each examined image of arm X, Y' = Y (1 + a_X K_X^E1), with Y the float luminance of the PNG (the
same luminance the scorer computes; planting multiplicatively in each RGB channel gives the same Y'). Scoring is on
Y' in float32, **without re-quantisation**, because the images are already 8-bit and a change far below half a grey
level would be rounded away. Real transfer happens before the decoder's rounding, so this is a stated departure.
- *Where K comes from:* K_X^E1 (`out/fp/K_A_E1.npy`, `K_B_E1.npy`), estimated from 80 photographs.
- *Scoring:* K_A^E2 and K_B^E2, estimated from 140 photographs disjoint from E1, with guard bands between the splits.
  The measurement is `c6_estimator_swap._measure`, i.e. `t1_measure._measure`: luminance, `wavelet_residual`,
  NCC(W, Y K), imported. The unplanted rows are reused from `out/c6_estimator_swap.json`, after asserting that 50
  randomly chosen images recompute to within 1e-9. Agreement with the archive rows on the same images (per-image
  correlation and mean difference) is reported, because the paper's primary numbers come from the archive
  instrument.

**Amplitude.** a_X is chosen on the calibration images so that the mean planted-minus-unplanted paired contrast
equals the target T:
- targets T = 1 x and 10 x the nominal limit, U_device = 5.3761e-05 and 5.3761e-04;
- secant iteration from two starting amplitudes, stopped at 1 % relative error (at most 8 iterations);
- the planted shift achieved on the examined images is reported, not tuned. If it differs from T by more than 10 %,
  the model curves are also shown at the achieved shift, as a diagnostic.

**Examiner.** For examined adapter a of arm X and G in **{10, 20, 50, 100}**:
- theta_hat_G = (mean of d_X over G distinct images of that adapter) - m_X;
- **1,000 random subsets per adapter, per G and per condition**, seeded (20260930 + offsets), with the same subsets
  used in every condition;
- conditions: **null** (no planting), 1 x and 10 x.
- **Threshold:** the 99th percentile of the null theta_hat_G, pooled over the 18 adapters at that G, so the FPR is
  1 % by construction; M = 2.
- **Empirical TPR:** the fraction of planted trials above the threshold.
- **Uncertainty:** a cluster bootstrap over adapters (2,000 replicates, stratified by body), with percentile
  intervals simultaneous over the eight (G, target) cells at 1 - 0.05/8.

**Model comparator.** Eq. (power) with lambda R_real = T and M = 2, evaluated at sigma_mu = 0 and at
sigma_mu = 4.1e-05, the Entry 114 exact 95 % upper limit.
- For the criterion, sigma_G^2 is the variance this design has under the model:
  sigma_mu^2 (1 + 1/11) + [sigma_vX^2 (N - G)/(N - 1) + sigma_eX^2]/G + sigma_eX^2/(11 x 250), with N = 250, and
  sigma_v, sigma_e per arm from `out/fv_sigma.json`. The seed term shrinks because subsets are drawn from a 250-seed
  bank that m_X shares; the last term is the main-effect estimate.
- Per-arm TPRs are averaged over the two arms.
- The unadjusted eq. (power) curves (SE_500 = 7.035e-05) are printed beside the adjusted ones.

**Agreement criterion and readings** (the band at each cell is [TPR at sigma_mu = 4.1e-05, TPR at sigma_mu = 0]):
- the simultaneous interval meets the band at all eight cells -> **"eq. (power) agrees end-to-end: planted transfer
  at the nominal limit and at ten times it is detected at the rates the model gives"**;
- the interval lies wholly below the band at any cell -> **"the paper's scoring detects less than eq. (power)
  predicts"**. The model's rates remain upper bounds, and the paper prints the empirical rates beside them and names
  the cells;
- the interval lies wholly above the band at any cell -> **"eq. (power) understates the examiner's power"**. The
  sigma_mu = 0 rates are then no longer upper bounds for this examiner, and the paper's "at most" statements are
  replaced by the empirical rates where those are higher;
- misses in both directions -> reported as "does not agree", with both directions named.

**Descriptive:**
- the FPR of an examiner who uses the model's own threshold, z_0.99 sigma_G (adjusted, sigma_mu = 0);
- TPR at five and fifty candidates, taking the null quantiles 1 - 0.01/(M - 1) as thresholds;
- per-body results.

At 1 x the model predicts rates of about 0.01-0.03 for G <= 100; that half is a check that planting at the limit
stays undetectable, and the 10 x half tests the curve.

**Output and runtime.** `src/fv/fv_examiner.py` -> `out/fv_examiner.json`, with per-image planted rows in
`out/fv_examiner_rows.csv` (appended and resumable). About 15,000 residual computations on 7 workers, 40-60 min.

### 3. Cross-body scene audit (review §4 item 7)

**Why.** Dresden photographs the same scenes with many cameras. If body A's training crops share scenes with body
B's E2 photographs, K_B^E2 carries leaked content that A's generations can match. That lowers d_A, likewise for
T(B) against E2(A), and makes the limit too low. The v1 scene audit (`notebooks/01_pilot.ipynb`
`scene_audit()`; manifest `scene_audit`, worst 0.732 A / 0.789 B) compared only splits within one body.

**Sets.**
- **Primary:** every T crop of body A against every E2 photograph of body B, and T(B) against E2(A), for the D200
  pair (50 x 140 each way), the iPhone 5c pair (50 x 60; `out/fp_5c/manifest.json`) and the P20 pair (50 x 90;
  `out/fp_p20b/manifest.json`).
- **Descriptive:** H(A) x E2(B) and H(B) x E2(A), which enter R_real; T(A) x T(B); and within-body T x E2 again, for
  comparison with the v1 values.

**Embedding.**
- **Primary:** facebook/dinov2-base, CLS pooler output, L2-normalised, with the default processor (shorter side 256,
  centre crop 224), cached locally. This is the embedding of the study's 0.90 copy criterion and of the content
  matching (`src/dino_memorization.py`).
- The v1 split audit used DINOv2 ViT-S/14 through torch.hub with the same resize and crop. Its weights are not on
  this machine; fetching them needs the author's approval, so that replication runs only if approved and is
  otherwise reported as not run.
- **Inputs:** T as the 1024^2 native centre crops that were trained on. E2 both as the full photograph (the v1
  audit's input) and as the 1024^2 centre crop (the pixels the fingerprint is estimated from).

**Threshold.** A cross-body pair is a **near-copy if its cosine is >= 0.90 in either E2 representation**. That is
the copy criterion; it is stricter than the v1 split audit's halt at 0.95. Counts at 0.80, 0.85 and 0.95 and the
maximum per direction are reported descriptively. `src/fv/fv_crossbody_scene.py` -> `out/fv_crossbody_scene.json`
(+ `_emb.npz`).

**Rule and readings.**
- No cross-body near-copy in any pair -> **"no scene shared between one body's training crops and the other body's
  estimate photographs"**. The manuscript's "did not compare the two bodies" (III-B) is replaced by the result.
- Any near-copy -> the affected E2 photographs are listed by file name with their cosine and matching T crop, and
  for each affected pair:
  - (a) K_X^E2 is re-estimated without them, with the estimator that built it (`fingerprints.estimate_K` for the
    D200 pair; the pair's own prep script's estimator for the iPhone and P20 pairs, imported);
  - (b) R_real is recomputed from the pair's H split with the new estimate;
  - (c) the generations are rescored. Stored per-image rows cannot be rescored against a new K; the images are
    needed. The iPhone and P20 pairs rescore on all their local generations (12 x 250 per arm). The D200 pair
    rescores on the 24 primary adapters' images 0-249 (local), with old and new K on the same images, so that the
    change in the nominal and calibrated max-arm limits is measured on one 250-image basis. The ledger's 500-image
    basis would also need images 250-499 of seeds 3-11, which are on the Drive archive only; this is stated, and
    they are not fetched.
  - A relative change of the limit of at most 10 % -> **"the scene overlap does not move the limit"**. More than
    10 % -> the recomputed limit is printed beside the limit of record as a sensitivity, and the overlap is named in
    Limitations. The limit of record is not replaced.

### 4. Normal-versus-inverted weight test (review §4 item 8)

**Adapters, all present on 30 Sep 2026.** Paths are
`D:/A4/einv/v2/out/t1/adapters/<tag>/pytorch_lora_weights.safetensors`; each file is 14,587,296 bytes, and each
`train_meta.json` gives seed = the tag's seed, steps 16000 and rank 16.
- normal, body A: `dose16k_A_s0` ... `dose16k_A_s5` (field `none`);
- normal, body B: `dose16k_B_s0` ... `dose16k_B_s5` (field `noneB`);
- inverted, body A: `inv16k_A_s0` ... `inv16k_A_s2`, `inv16kext_A_s3` ... `inv16kext_A_s7` (field `invA`);
- inverted, body B: `inv16k_B_s0` ... `inv16k_B_s2`, `inv16kext_B_s3` ... `inv16kext_B_s7` (field `invB`).
- `dose16k_*_s6` and `_s7` do not exist.

**Design facts that fix the statistic.**
- Within a body, the normal and the inverted adapter of seed j share the initialisation, batch order and noise
  stream (`t1_ladder.train_arm`), and their 50 crops are the same photographs.
- The crops differ by the reversed fingerprint estimate, Y(1 - 6 K_X^E1), plus a dither field fixed once per crop
  (`g1_invert.py`, seed 20260919). The total change is 0.84 (A) and 0.72 (B) grey levels RMS (`out/t1/invert.json`),
  part of which is the dither and rounding noise (about 0.4 grey RMS per channel, before any luminance weighting).
- Normal adapters were trained 12-16 Sep and inverted adapters 20-27 Sep.
- Same-seed pairs are "twins" and must not enter a within-versus-cross comparison.

**Extraction.** Delta W = B A per layer, with the cosine between adapters' concatenated updates computed exactly as
`h5_weight_signature.py` does (`load_pairs`, `gram`, imported).

**Statistic (primary).** For body X, over seeds 0-5 only:
- D_X = (mean cosine over the 30 within-condition different-seed pairs: 15 normal-normal, 15 inverted-inverted)
  - (mean cosine over the 30 cross-condition different-seed pairs (N_j, I_k), j != k);
- the six same-seed cross pairs (N_j, I_j) are excluded;
- the inverted seeds 6-7 are excluded, because they have no normal partner;
- D = (D_A + D_B)/2.

**Null and level.** Exact relabelling within seed pairs: for each body and each seed j, the labels of N_j and I_j
are kept or exchanged. Exchanging all six in one body maps the partition to itself, so each body has 32 distinct
relabellings and the two bodies together have **1,024. The floor is 1/1024 = 0.00098.** The one-sided p is the
fraction of the 1,024 with D >= observed, the identity included. **Level 0.01, one-sided, D > 0.** The relabelling
is valid under H0 because the twins of seed j are exchangeable if the training condition does not change the
weights' distribution.

**Readings.**
- p < 0.01 with D > 0 -> **"the adapter weights carry the fingerprint's sign"**, in the precise sense that they
  distinguish training on the photographs with the fingerprint reversed from training on the same photographs
  unchanged: a white-box trace of the stored fingerprint change.
  - The ceiling is fixed now. The two conditions also differ by the fixed dither field and by training date, so a
    positive result does not isolate the fingerprint pattern from the dither. It says nothing about whether a
    natural-amplitude fingerprint is recoverable from standard 2000-step adapters.
  - The date is not expected to matter: among different-seed pairs, batches ten days apart differ by -0.0001
    (Entry 114). That observation is cited, not tested.
- p >= 0.01 -> **"the weights do not detectably carry the fingerprint's sign"**, at six seed pairs per body and
  16000 steps.

**Descriptive, not tested:**
- per-body D_X and exact p (floor 1/32);
- the six twin cosines per body;
- a matched variant: the mean pairwise cosine among the twin differences Delta_j = dW(I_j) - dW(N_j), with its
  relabelling p over the same 1,024 partitions;
- D with the inverted seeds 6-7 kept at their fixed label (4,096 distinct relabellings);
- adapter-weight norms by condition.

`src/fv/fv_weights_invert.py` -> `out/fv_weights_invert.json`.

### 5. Closed-set expectation (review §4 item 9; descriptive)

**Question.** What closed-set accuracy would Entry 06's registered experiment have been expected to show if transfer
sat at a stated limit? The experiment has five bodies, ten adapters, the first 250 images each, and the
main-effect-corrected argmax with leave-the-candidate-out main effects. The expectation gives "at chance" (3/10,
3/10) a yardstick.

**Data and scorer.**
- Kodak: raw K rows `data/csv_kodak/c3_measure_raw.csv`.
- P20: residualised K rows `G:/My Drive/inv_channel/E_DAXING/csv/d5_measure.csv` (not on D:), with SHA-256
  recorded.
- The scorer is `t3_attrib.attribute`, imported. The script first reproduces Entry 07's corrected accuracies at
  G = 250 (3/10 and 3/10).

**Planted levels.** A shift delta is added to each image's score against its own body's fingerprint. delta is set
so that the group's own transfer statistic, as defined in the file that produced the group's device-level limit,
rises by exactly the target. The script asserts the rise and that it recomputes the ledger's limit.
- (a) the group's device-level limit: Kodak 1.74 %, P20 1.14 % of that group's own R_real (Kodak's at its lower
  99 % limit, as in the ledger);
- (b) the primary limits applied to the group's own R_real: nominal 0.1507 % and calibrated 0.1752 %, plus the
  item 1 limit if it replaces 0.1752 %;
- (c) zero.

**Expectation.**
- **Primary (empirical noise):** 2,000 two-stage bootstrap replicates. Adapters are resampled with replacement
  within each body and images with replacement within each adapter's first 250; the full scorer, including the
  main-effect estimates, is rerun in each.
- **Beside it (parametric):** each adapter's corrected five-score vector is drawn from
  N(delta e_own, sigma_mu^2 I + Sigma_img/250 + Sigma_b). Sigma_img is the pooled within-adapter per-image
  covariance of the five scores, Sigma_b the covariance of the leave-the-candidate-out main-effect estimates, and
  sigma_mu is 0 or 4.1e-05 (the D200 value, transported and stated so). 100,000 simulated experiments.

**Reported.** E[accuracy]; the distribution of correct adapters out of 10; P(X >= 6), which is above the central
95 % binomial interval of chance, 0.0064 under chance; and P(X >= 5), 0.033 under chance, for each group and level.
No reading attaches. `src/fv/fv_closedset_expect.py` -> `out/fv_closedset_expect.json`.

### 6. Small quantities (descriptive, from existing files) — `src/fv/fv_small.py` -> `out/fv_small.json`

- **(a) hostile-r2-04.** The nominal max-arm limit of the primary design retrained on the local stack, six per arm,
  from `out/g2_pooled_six.json` per_adapter_A/B: U_X = mean + t_{0.995,5} s/sqrt(6), U = max, as % of R_real
  0.0356703 (c = 1; H6's c is not transported to the local stack). Beside it, labelled post hoc, the same with
  `nomark_s5` left out (arm A at five, t_{0.995,4}), and the archive's six-per-arm 0.249 % (ledger history_k6).
- **(b) hostile-r2-14.** The 16000-step symmetric p-values with the E2 estimation shift added:
  SE_infl = sqrt(SE_Welch^2 + 1.224e-05^2), t = theta_sym/SE_infl, one-sided at the Welch df of each result.
  - Applies to the Entry 68 replication (new three per body; p 0.0096), the pooled six per body (p 0.0020) and the
    G1 inversion at eight per arm (p 0.0106, direction theta < 0).
  - Each input is read from the file that produced its entry (`out/t1/dose_stats.json`, `out/g1_ext.json`, and the
    Entry 68 summaries) and asserted to reproduce the entry's p before inflation.
  - The estimation SD was measured at 2000 steps and 500 images and is transported (stated).
  - The within-body inverted-minus-normal contrast (Entry 110) is not inflated: both conditions are scored with the
    same K, so the shift cancels exactly.
- **(c) trace-r2-12.** Total GPU-hours, split by GPU (A100 archive and Colab runs, including FLUX; L40S local), for
  the 211 adapters and 95,500 generations of Entry 112 N-A5, and separately for all work, adding the objective arms,
  the ladder and the environment chain.
  - Training uses recorded wall time where a record exists (`train_meta.json` minutes or equivalent); otherwise the
    per-unit ranges of supplement S3.
  - Generation uses S3's per-unit ranges times the counts.
  - Reported as low-high with the recorded share stated. Detector, embedding and scoring GPU time is excluded
    (stated).
- **(d) figures-r2-11.** nDoseInvThreeT: the Welch t of the first inversion reading, seeds 0-2 per arm from
  `out/g1_ext.json` per_adapter_A/B. It must reproduce Entry 98's theta_sym (-0.143 %) and p (0.145); the script
  stops otherwise.
- **(e) trace-r2-09.**
  - The verifier's check count: `python verify_v2.py` in the repository clone (CPU, no downloads); the number of
    checks executed and the number agreeing, from its output.
  - Adapters and generations by stack, exactly as Table 2(b) labels them (archive, local, archive/local), asserted
    to sum to 211 and 95,500.
  - The deposit file count waits on author actions and is not computed.

**Order of work.** Items 1, 4, 5 and 6 on the CPU first. Item 3 uses the GPU for embeddings, and item 2 is the long
CPU job. Results are logged in the next entry, reported whichever branch each lands in.

---

## Entry 117 · 2026-09-30 · Results of the Entry 116 closing checks

The six items registered in Entry 116 were run on 30 Sep - 1 Oct 2026 (CPU; the GPU only for DINOv2 inference in
item 3; nothing trained). Each was then re-derived by a separate agent that wrote its own code from the raw inputs
before reading the analysis script (`paper/fv/work/check116/<item>/`). Every number that enters a pre-specified
reading reproduced. The checks found errors only in the analysts' prose and in earlier documents (Entry 116 itself,
Entry 112, supplement S3); none changed a value in a result file. The result files were inspected for the corrected
statements (the seed-matched pairing, "median" against "mean", the band shift, the post-hoc labels): each file
carries the correctly labelled values, so no script was amended or re-run and no `_v2` file was written. The
corrections are listed under each item and govern how the numbers are worded.

Numbers reach the manuscript through `src/fv/num_n9.py` (246 macros, entry "117"; prefixes Lim, Att, Wt, Dose,
Data). `src/fv/fv_numbers.py` now builds 1,890 macros from nine parts; none of the 1,644 earlier macros changed text.

**Files of record and name deviations.** The workflow that ran the items used other file names than Entry 116
registered. Where a first attempt had already written a file, it was kept and the rerun wrote new names, so two
items have two output files. Methods are as registered in every case.

| item | registered name | file of record | other file (same numbers) |
|---|---|---|---|
| 1 | `fv_seedbank.*` | `src/fv/fv_seedbank_calib.py` -> `out/fv_seedbank_calib.json` (+ `_progress.json`) | - |
| 2 | `fv_examiner.*` | `src/fv/fv_examiner_e2e.py` -> `out/fv_examiner_e2e.json`, rows `out/fv_examiner_e2e_rows.csv` | - |
| 3 | `fv_crossbody_scene.*` | `src/fv/fv_crossbody_scenes.py` -> `out/fv_crossbody_scenes.json` (+ `_emb.npz`, `_rows.csv`; re-estimates `out/fp/K_B_E2_xbscene*.npy`, `out/fp_p20b/K_A_E2_xbscene*.npy`) | - |
| 4 | `fv_weights_invert.*` | `src/fv/fv_weights_inv.py` -> `out/fv_weights_inv.json` (30 Sep) | `out/fv_weights_inv_recheck.json` (independent recheck + post-hoc checks; differences <= 2e-17) |
| 5 | `fv_closedset_expect.*` | `out/fv_closedset_expect_run2.json` (+ `_boot_rows.csv`), `src/fv/fv_closedset_expect_run2.py` | `out/fv_closedset_expect.json` (first attempt; identical on every shared quantity) |
| 6 | `fv_small.*` | `src/fv/fv_small.py` -> `out/fv_small.json` (the registered names) | `out/fv_small_quantities.json` (earlier attempt; identical) |

Item 2's measurement ran in two invocations: the first wrote 16,450 of 16,500 rows and ended; the second resumed
from the rows file (as registered) after recomputing 50 unplanted and 12 planted rows exactly. Appending to the
rows file was that resume, not an overwrite.

### 1. Seed-bank calibration of the headline limit (rule) — **the term changes the calibration**

**Estimate** (24 primary adapters x 500 seeds; the three Drive CSVs match the SHA-256 in `fv_sigma.json`; adapter
means equal the ledger exactly):

| quantity | value |
|---|---|
| sigma_vA^2, sigma_vB^2 | 1.4375e-06, 1.5702e-06 |
| cov(v_A, v_B) | -1.3394e-06 (implied correlation -0.89; seed-mean correlation -0.84) |
| symmetric seed-bank variance sigma_vs^2 | **8.22e-08** (= (vA + vB + 2 cov)/4 exactly); bootstrap 5th / 95th percentile 6.66e-08 / 9.80e-08 |
| SD of the bank shift per replication | symmetric part 1.28e-05; additive part 5.33e-05 |

**Rule run** (H6's 27 cells, 4,000 replications, master seed 106061, bank shifts from a generator seeded 116001,
drawn per cell in H6's cell order as standard normals times the Cholesky factor of Sigma_v/500). With the term set
to zero H6 is reproduced exactly (worst-cell coverage 0.9795 at c = 1, 0.9900 at c = 1.25). With the term: 0.981
at c = 1 and **0.98875 at c = 1.25** (3,955 of 4,000; 3,960 needed); binding cell t3 spread, observed additive
part, transfer 1e-05. The smallest c reaching 0.99 is **c* = 1.31** (coverage 0.9900).

**Pre-specified reading: worst-cell coverage at c = 1.25 is below 0.99 -> "the seed-bank term changes the
calibration". The bound of record becomes 0.1811 % of R_real** (U_device 6.461e-05, arm B binds; arm A 0.133 %),
the calibrated max-arm one-sided 99 % limit, calibrated for adapter, fingerprint-estimation, training-set and
seed-bank variation. H6's 0.1752 % (c = 1.25) and the nominal 0.1507 % are printed beside it.

Registered descriptive items: (i) the review probe's form (symmetric part only, shared by both arms) needs c = 1.56,
0.206 % (the check gets 1.53 with another drawing of the shared shift; the review's "about 0.21 %" came from this
incomplete form); (ii) at the 95th percentile of sigma_vs^2, c = 1.33, 0.183 % (5th percentile: c = 1.27,
0.177 %; the 2.5th and 97.5th give the same two values of c); (iii) the symmetric construction's worst-cell
coverage is 0.803 without the term and 0.770 with it, so max-arm stays the primary construction; (iv) a
20,000-replication rerun (master seed 116002) gives 0.9917 at c* (Monte Carlo SE 0.00064) and 0.9904 at c = 1.25.

Rates at the bound of record (derived with the Entry 112/114 functions; at 0.1752 % they reproduce the filed values
exactly in 143 comparisons). Two candidates, 1 % FPR:

| examiner / persistent term | 500 images | 5,000 | unlimited | images for TPR 0.5 (0.9) | was at 0.1752 % |
|---|---|---|---|---|---|
| fresh seeds, sigma = 0 (bound) | 0.080 | 0.72 | 1 | 3,209 (7,717) | 0.075, 0.69; 3,428 (8,244) |
| fresh seeds, sigma = 4.1e-05 (paired upper limit) | 0.063 | 0.173 | 0.226 | not reached | 0.059, 0.161, 0.210 |
| fresh seeds, archive sigma_mu 5.23e-05 (named sensitivity) | 0.056 | 0.117 | 0.138 | not reached | 0.053, 0.110, 0.129 |
| training-set sd as persistent term | 0.074 | 0.40 | 0.73 | 8,621 | 0.070, 0.37, 0.70; 10,409 |
| seed-matched, null-specific SE (the corrected form) | 0.365 | | | 709 | 757 |
| seed-matched, single-SE form | 0.307 | | | 816 (1,962) | 0.286; 869 |

Critical sigma above which TPR 0.5 is unreachable: 2.78e-05. Transfer needed for TPR 0.5 from 50 images: 8.0x
(fresh seeds), 8.2x (archive sigma_mu), 4.0x (seed-matched, single SE) the bound of record.

**Check: CONFIRMED WITH CORRECTIONS.** Data, estimate, H6 reproduction, rule run, interval ends, items (iii)-(iv) and
every rate reproduce exactly; the bootstrap percentiles agree to Monte Carlo error (95th 9.802e-08). Corrections to
the analysis report: (1) the claim that drawing with `multivariate_normal` "could have landed on 'does not change'"
is wrong: that drawing also reads "changes", with c* = 1.35 (0.1851 %). Two other literal readings of "an
independent generator (seed 116001)" do read "does not change": one draw shared by all 27 cells (0.99225 at 1.25,
c* 1.17) and a child generator per cell (0.9905, c* 1.23). The registration fixed the seed but not the
transformation; the run recorded its choice in `settings.bank_draws`. Both the branch and c* depend on that choice.
(2) "c is 1.275 with the term and 1.315 without" are medians; the means are 1.2765 and 1.3255 (difference
-0.049). (3) The seed-matched figures were paired wrongly in the report: 709 images go with TPR 0.365 at 500
(null-specific SE, the corrected form), 816 with 0.307 (single SE). The result file pairs them correctly. (4) The
5-replication margin is about 0.75-0.8 binomial SE, not 0.7.

**What the result does and does not say (post hoc, carried into the paper's wording).**
- *The branch was decided inside Monte Carlo noise.* With H6's draws fixed and the bank shift redrawn from 200 other
  generator seeds, the rule reads "changes" in 50 % of draws (check: 49.5 %), with c* from 1.14 to 1.42 (median
  1.26); 12.5 % give c* >= 1.31.
- *On average the term lowers c.* Over 20 independent 4,000-replication runs c is 1.28 with the term and 1.33
  without (mean difference -0.05; Monte Carlo SD of c about 0.05, i.e. about 0.005 percentage points on the limit).
  The term is mostly an additive part (SD 5.3e-05) that moves the arms in opposite directions.
- *H6's c = 1.25 was a favourable draw.* All 20 runs without the term need more than 1.25, and the 20,000-replication
  rerun selects 1.30 without the term (0.180 %) and 1.24 with it. So 0.181 % is about what H6's model needs when
  simulated precisely; the seed-bank term adds nothing on average.
- Therefore the paper states 0.181 % as the calibrated bound of record (the rule's outcome) and must not say that the
  seed bank raised the limit. It says the seed-bank term was included and that its effect on c is within the
  simulation's Monte Carlo error.
- *Not verified:* that the bank term is not double-counted with G5's training-set SD (this needs the seed effect to
  be shared across stacks, not measured). Only the symmetric variance was varied; the per-arm variances that drive
  the additive part were held at their estimates.
- Scope: only the D200 pair was analysed. Every other design also generates from one bank and keeps its nominal limit.

**What changes in the paper.** Every place where `\nLimCal` stands as the headline (abstract s3, I ¶6-7, V-C
heading and text, V-D, VI-A, VII, VIII, Table 1, Table 5 row 1, Fig. 6 caption, Fig. 7 and S10 captions, S08,
S11, Tables S11 and S16) takes `\nLimRec` with `c = \nLimRecC`; `\nLimCal` (c = 1.25) and `\nLimNom` print beside
it where the construction is explained (V-C, S07, S08). The "Cal" rate macros become the "Rec" macros
(`nAttTprZeroTwoFiveHundredRec`, `nAttZeroGFiftyExactRec`, ... ; list in `num_n9.py`). S07 adds the seed-bank term
and its estimate, the rule, items (i)-(iv) and the Monte Carlo caveat. The calibrated power curve in Figs. 7 and S10
and the calibrated marker in Fig. 6 are regenerated at 0.181 %.

### 2. End-to-end examiner test (registered model check) — **eq. (power) agrees**

Planting Y(1 + a_X K_X^E1) in float32 without re-quantisation, scored with K^E2 by the paper's local scorer.
Amplitudes from the 1,500 calibration images (s0-s2, images 250-499), one secant step each: a_A 1.756e-03, a_B
3.604e-03 at 1x (ratio 2.05), ten times that at 10x (shift exactly ten times). Achieved shift on the 4,500 examined
images (s3-s11, images 0-249): A 0.978 T, B 1.047 T (inside the registered 10 % rule). The local scorer agrees with
the archive rows behind the paper's numbers at r = 0.9966 (6,000 images; mean difference -3.6e-06).

Two candidates, threshold the pooled null 99th percentile (FPR 0.01 by construction); 18 adapters x 1,000 subsets;
brackets = cluster-bootstrap interval simultaneous over the eight cells (1 - 0.05/8); band = [model at sigma_mu =
4.1e-05, model at sigma_mu = 0], adjusted to this design; "eq. (power)" = the unadjusted curve as printed.

| G | 1x empirical | 1x band | 10x empirical | 10x band | eq. (power) 1x / 10x | FPR at the model's threshold |
|---|---|---|---|---|---|---|
| 10 | 0.0141 [0.0122, 0.0156] | 0.0132 | 0.118 [0.095, 0.150] | 0.103-0.104 | 0.0133 / 0.106 | 0.0101 |
| 20 | 0.0156 [0.0135, 0.0178] | 0.0148-0.0149 | 0.221 [0.177, 0.277] | 0.208-0.211 | 0.0149 / 0.212 | 0.0113 |
| 50 | 0.0219 [0.0174, 0.0244] | 0.0187-0.0189 | 0.583 [0.465, 0.709] | 0.548-0.567 | 0.0185 / 0.536 | 0.0086 |
| 100 | 0.0239 [0.0220, 0.0339] | 0.0247-0.0256 | 0.858 [0.729, 0.963] | 0.900-0.925 | 0.0236 / 0.862 | 0.0100 |

**Pre-specified reading: every simultaneous interval meets its band -> "eq. (power) agrees end-to-end: planted
transfer at the nominal limit and at ten times it is detected at the rates the model gives."** It holds at the
achieved shift too, and in 20 of 20 independent subset draws with their own bootstraps (check).

**Check: CONFIRMED WITH CORRECTIONS.** Unplanted and planted scoring, calibration (from another starting point),
achieved shifts, every TPR, band and unadjusted curve reproduce (planted rows within 1.1e-07, no bias; the resumed
rows are sound). Corrections to the report's wording: (1) using this scorer's own variance components moves the band
by up to 0.004 (G = 50, 10x), not 0.002; (2) the point TPRs carry subset-sampling noise that the intervals do not
include: SD about 0.001 at 1x and 0.004-0.009 at 10x over 20 draws; the registered draw sits about 2.5 SD from the
mean of the draws in three cells (G 20 10x 0.221 vs 0.238; G 50 1x 0.0219 vs 0.0197; G 100 1x 0.0239 vs 0.0268).
Point TPRs are printed to two significant figures with that noise stated; (3) the null's spread is within 0.4-1.9 %
of sigma_G pooled over bodies, and +4.4 % to -8.9 % per body (not "1-3 %"); (4) per-adapter achieved shifts vary
about 4.8 (A) and 2.9 (B) times more than image sampling explains ("three to five times", not "four times").

**Limits of the check (carried into the wording).** It tests the curve at G <= 100; the persistent term
sigma_mu = 4.1e-05 contributes at most 8.2 % of sigma_G^2 there, so nothing here tests the statements at 500 or
5,000 images or "any number of images". Resolution is uneven: at 10x and G = 50-100 the intervals are about +-0.12
wide against bands at most 0.025 wide. The examiner here is more favourable than the paper's (its subsets share the
250-seed bank with the main-effect estimate), which is why the adjusted variance is the comparator; that the
unadjusted 0.862 sits near the empirical 0.858 at 10x, G = 100 is coincidence. The model assumes one shift for all
adapters; per-adapter achieved shifts range 0.71-1.36 T, which costs about 0.06 in TPR at 10x, G = 100 (post hoc:
measured null + constant shift 0.925, + each adapter's own shift 0.864, empirical 0.858). At low TPR the point
estimates slightly exceed the sigma_mu = 0 rate (10x, G = 10: 0.118 against 0.104, inside the interval), so "the
sigma_mu = 0 rates are upper bounds" holds strictly for a constant transfer. Planting is idealised: multiplied in
after decoding and never rounded to 8 bits. Per body at 10x, G = 100: A 0.80, B 0.92. The M = 50 thresholds rest on
about four of 18,000 null trials (descriptive only).

**What changes in the paper.** V-D and S11 no longer call eq. (power) only a model: one sentence reports that
transfer planted at the nominal limit and at ten times it into 4,500 generations of 18 held-out adapters was
detected at the rates the model gives at all eight cells (G 10-100), with the limits above. A supplement table
(macros `nAttEnd*`) carries the eight cells. REVIEW_REPORT §4 item 4 is answered.

### 3. Cross-body scene audit (data-audit rule) — **near-copies in two pairs; the overlap does not move the limit**

DINOv2-base pooler embeddings (TF32 off), cosine >= 0.90 in either E2 representation (full photograph or the
1024^2 crop the fingerprint is estimated from). Seven near-copy pairs:

| pair | T crop (body) | E2 photograph (other body) | cosine crop / full |
|---|---|---|---|
| D200 | Nikon_D200_1_17712 | Nikon_D200_0_15391 | 0.9734 / 0.8118 |
| D200 | Nikon_D200_1_17708 | Nikon_D200_0_15387 | 0.9318 / 0.5496 |
| D200 | Nikon_D200_1_17714 | Nikon_D200_0_15391 | 0.9317 / 0.7271 |
| D200 | Nikon_D200_1_17706 | Nikon_D200_0_15385 | 0.9144 / 0.6712 |
| D200 | Nikon_D200_1_17700 | Nikon_D200_0_15387 | 0.9027 / 0.5610 |
| P20 | IMG_20190505_141920 (1103) | IMG_20190505_153503 (1104) | 0.9113 / 0.6560 |
| P20 | IMG_20190505_141915 (1103) | IMG_20190505_153502_2 (1104) | 0.9005 / 0.5691 |

Largest cross-body cosine: D200 0.973 (T(A) x E2(B); full photographs at most 0.878; the other direction 0.441);
iPhone 5c 0.850 (no near-copy); P20 0.911 (next pair 0.898). H x E2 (R_real's photographs): at most 0.744, none.

Sensitivity as registered: K_B^E2 (D200) re-estimated without 15385, 15387, 15391; K_A^E2 (P20) without 153502_2 and
153503; R_real recomputed; generations rescored (D200: 24 primary adapters, images 0-249, local instrument; P20:
12 x 250 per arm).

| pair, max-arm limit (% of R_real) | before | after | change |
|---|---|---|---|
| D200 nominal (250-image basis, arm A binds) | 0.2674 | 0.2421 | **-9.5 %** |
| D200 c = 1.25 | 0.3124 | 0.2863 | -8.4 % |
| D200 c = 1.31 (item 1's c*) | 0.3232 | 0.2969 | -8.1 % |
| D200 R_real | 0.035659 | 0.035524 | -0.38 % |
| P20 nominal | 0.4680 | 0.4456 | **-4.8 %** |
| P20 R_real | 0.039229 | 0.038764 | -1.19 % |

**Pre-specified readings: D200 and P20 -> "the scene overlap does not move the limit" (every change within 10 %);
iPhone 5c -> "no scene shared between one body's training crops and the other body's estimate photographs".** The
limits of record (0.1507 / 0.1752 / 0.1811 % for D200; the P20 limits) stand, as the rule says in either branch.

**Check: CONFIRMED** (no corrections). Same near-copy set (independent embeddings; with TF32 on the borderline P20
pair is 0.9002, still >= 0.90), re-estimated fingerprints within 1.6e-09, identical R_real, all 12,000 rescored
images equal, same limits and changes.

**Descriptive (not pre-specified).** The D200 margin is narrow and inside the noise: over 2,000 seed resamples the
nominal change has SD 9.2 percentage points, centred near -2.5 %, and 27 % of resamples exceed 10 % in size (P20:
11 %); six placebo exclusions of as many kept photographs give -13 % to +5.7 %. Removing the shared scenes lowered
arm A's contrast, the opposite of the hypothesised "limit too low" bias. On the ledger's 500-image basis, where arm B
binds, adding the measured arm-mean shifts moves the D200 limits up: nominal +7.7 % (+9.3 % with per-adapter
shifts), c* +6.5 % (+8.2 %); also within 10 %. With a 0.85 threshold the D200 nominal change is -10.9 % and P20's
-12.8 % (11 photographs dropped). The D200 near-copies are real shared scenes (a church tower; a garden wall with a
tree), visible only in the crop representation, so the v1 full-photograph audit could not have flagged them; the P20
pairs are textureless walls. **The two D200 training sets share scenes: 95 cross-body training-crop pairs at
cosine >= 0.90.** Within body, the iPhone body A has two T x E2 pairs >= 0.90 (0.911, 0.902), which would bias
toward finding transfer. The v1-style within-body audit with dinov2-base gives 0.734 / 0.800 (v1: 0.732 / 0.789).
The ViT-S/14 replication was not run (weights not on this machine; fetching needs the author's approval).

**What changes in the paper.** III-B: "did not compare the two bodies" is replaced by the result (cross-body audit,
seven near-copies in two pairs, re-estimation moves the limit by less than 10 %; the iPhone pair shares no scene).
S01 adds the table above. Wherever the text says the two D200 bodies trained on different scenes, it must say
different photographs (S13: "The two training sets differ in scene content as well as in camera" needs rewording;
CLAIMS X7). Limitations may add one clause that the D200 margin (-9.5 % against 10 %) is within resampling noise.

### 4. Normal-versus-inverted weight test (registered test, level 0.01) — **the weights tell the two conditions apart**

28 adapters asserted (size, seed, 16,000 steps, rank 16, field); `dose16k_*_s6/s7` absent as stated.

| quantity | value |
|---|---|
| D = (D_A + D_B)/2 | **+8.07e-04** (D_A +9.19e-04: within 0.02562, cross 0.02470; D_B +6.95e-04: within 0.02477, cross 0.02407) |
| exact one-sided p over 1,024 relabellings | **1/1024 = 0.00098**, the floor; observed D ranks first (next 5.80e-04; z 5.4) |
| per body | p 1/32 each, each body's floor |
| twin cosines | A 0.906-0.919, B 0.925-0.933 |
| matched variant M | 0.0103, p 1/1024 (one finding with D: D ~ 1/2 r^2 M) |
| inverted seeds 6-7 kept | D 8.21e-04, p 1/4096 (floor) |
| leave one seed out | D 8.02e-04 to 8.10e-04, each at its floor 1/256 |
| weight-update norms | twin differences +0.029 on average, mixed signs |

**Pre-specified reading: p < 0.01 with D > 0 -> "the adapter weights carry the fingerprint's sign", in the
registered precise sense: they distinguish training on the photographs with the fingerprint reversed from training
on the same photographs unchanged.** The registered ceiling applies: the conditions also differ by a fixed dither
field and by training date, so the result does not isolate the fingerprint from the dither, and it says nothing
about a natural-amplitude fingerprint at 2000 steps.

**Check: CONFIRMED WITH CORRECTIONS.** Own loader and a different overlap formula (checked against `gram` and
against explicit weight updates); every registered number matches (<= 2e-18). Corrections: (1) the reading's name
must not stand alone in the paper: no +K arm exists, so the test shows that the inverted condition (-6K plus dither)
differs systematically from the unchanged one, not that the weights encode the fingerprint's sign; (2) fixed-pattern
arms align "comparably or more strongly" (0.0086-0.031), not "as much as or more than" the inverted condition (the
random +-1 field 0.0086 and the 1-gray Gaussian field 0.0096 are below 0.0107 A / 0.0099 B); these arms are at 2000
steps on body A's crops with 2-3 seeds each, so "any fixed pattern" is a post-hoc inference; (3) overlaps of the
inverted twin differences with unrelated patterns must be quoted for both bodies or as means (+0.0059 / +0.0040 with
body B's estimate at 12x; +0.0036 / +0.0024 random field; mean over 15 pattern arms +0.0023 A, +0.0014 B);
(4) the design cannot test a body-specific component at all (relabelling also flips a shared condition effect), not
"only to p 1/32"; (5) the file-name deviation and the post-hoc label of the original script's context block (it was
added after a scratch run had shown the result; the registered numbers do not depend on it).

**Post hoc (not pre-specified).** The reversed-fingerprint term is 81 % (A) and 73 % (B) of the stored change's
energy (0.77 / 0.62 gray RMS), one pattern in all 50 crops; dither plus rounding is 20 % / 28 % (0.38 gray RMS),
drawn afresh per crop; the inverted crops rebuild bit-exactly. A decoder change with no shared pattern (v1 Colab
crops, 0.66 gray RMS) aligns only 0.0014. About half the within-body alignment is shared by both bodies (cross-body
0.0054, 53 % of 0.0103, p at its floor 1/2048), although the two E1 estimates are nearly unrelated (r 0.0024).
No training-date effect is visible: identical training code in all editor snapshots, all packages and model files
predate the first adapter, same-run pairs are not more alike than different-run pairs (-0.00004 to -0.00033);
twins share their random stream (loss-trace r 0.999) and inverted twins run at a constant lower loss (-0.0057 A,
-0.0038 B). Same-seed adapters of different bodies have cosine about 0.48 against about 0.025 for different seeds.

**What changes in the paper.** VI-D / S13 and CLAIMS X7: "separating them needs content held fixed ... tested with a
test registered before looking" is now done. Ceiling: "With content held fixed, a test registered before looking
distinguishes 16000-step adapters trained on photographs with the fingerprint reversed from adapters trained on the
same photographs unchanged (exact p 0.00098, the design's floor)." Floor in the same paragraph: the inverted crops
also carry a fresh dither, so fingerprint and dither are not separated; post hoc, any fixed pattern added to every
training crop produces comparable alignment at 2000 steps, and about half the alignment is common to both bodies; it
says nothing about natural-amplitude fingerprints at 2000 steps. Do not write "the weights carry the fingerprint",
"the fingerprint's sign is recoverable from the weights" or "white-box attribution". Macros `nWtInv*`.

### 5. Closed-set expectation (descriptive; no reading attaches)

Entry 07 reproduces (3/10 Kodak, 3/10 P20 K, 4/10 P20 low/mid). Planting raises each group's statistic and U by
exactly delta; the scorer's count equals the margin count in every replicate. Chance (Binomial(10, 0.2)): E 0.20,
P(X >= 6) 0.0064, P(X >= 5) 0.033, P(X <= 3) 0.879. E = expected accuracy; bootstrap = pre-specified two-stage,
2,000 replicates; parametric = 100,000 experiments, sigma_mu = 0 (beside it 4.108e-05).

| group (observed) | level | delta | bootstrap E, P(X <= 3) | parametric E, P(X <= 3) |
|---|---|---|---|---|
| Kodak (3/10) | (a) device level 1.736 % x R_low | 2.899e-04 | 0.985, 0 | 0.975, 0 |
| | (b) nominal 0.1507 % x R 0.0293 | 4.42e-05 | 0.462, 0.22 | 0.343, 0.53 |
| | (b) H6 0.1752 % | 5.14e-05 | 0.488, 0.17 | 0.370, 0.46 |
| | (b) bound of record 0.1811 % | 5.31e-05 | 0.495, 0.16 | 0.377, 0.44 |
| | (c) zero | 0 | 0.304, 0.64 | 0.200, 0.88 |
| P20 K (3/10) | (a) device level 1.144 % x R_low | 4.086e-04 | 0.999, 0 | 0.999, 0 |
| | (b) nominal x R 0.0573 | 8.63e-05 | 0.603, 0.048 | 0.514, 0.15 |
| | (b) H6 | 1.004e-04 | 0.650, 0.025 | 0.570, 0.080 |
| | (b) bound of record | 1.037e-04 | 0.661, 0.020 | 0.583, 0.068 |
| | (c) zero | 0 | 0.302, 0.64 | 0.200, 0.88 |

At the device-level limits, 3 or fewer correct occurred in 0 of 2,000 replicates and 0 of 100,000 experiments (also
at sigma_mu = 4.1e-05). Observed data plus delta (no resampling): Kodak 10/6/8/8/3, P20 10/8/9/9/3.

**Check: CONFIRMED WITH CORRECTIONS** (second Monte Carlo realisation with its own random numbers; all values agree
within Monte Carlo error; the parametric P20 values are 0.004-0.006 higher with the analytic main-effect covariance,
the disclosed D5 variant). Corrections: (1) the report's "the closed set rules out these limits far more firmly"
is a test's conclusion; item 5 is descriptive and is stated as "had transfer sat at the device-level limit, expected
accuracy would be at least 0.95; 3 or fewer correct occurred in 0 of 2,000 replicates and 0 of 100,000 simulated
experiments" (with the lean removed first: 0.960 Kodak, 0.996 P20); (2) the seed-paired Kodak figure 0.541 is a
2,000-replicate value; 20,000 replicates give 0.532 +- 0.0014, so "about 0.53" (P20 0.697 matches); (3) sigma_mu^2 I
on each of five scores gives each two-candidate margin 2 sigma_mu^2, so the 4.1e-05 column corresponds to a paired
sigma_mu of about 5.8e-05 (conservative); say so where printed; (4) sharing the main-effect draw across the ten
adapters leaves E unchanged and moves the tails little (Kodak nominal P(X >= 6) 0.079 against 0.089); (5) the
body-by-candidate interaction is persistent only relative to seed-free noise; against the full per-adapter noise the
ratio is 1.18 (Kodak, p 0.29), 0.77 (low/mid) and 2.43 (P20, p 0.005), so only for P20 is it clearly outside even
the inflated noise; (6) using R_low for P20 level (a) is the correct reading (the ledger's 1.1443 % is U/R_low);
Entry 116's "Kodak only" parenthetical was the inaccurate part.

**Caveats (post hoc).** The bootstrap is centred on the observed data: at zero transfer it expects 0.30, not 0.20,
and planted levels stack on the closed-set images' own lean (0.16 % of R_real for Kodak's first 250 images, 0.043 %
over all 500; P20 0.093 %). With the lean removed first the nominal level gives 0.300 (Kodak, P(X <= 3) 0.65) and
0.411 (P20, 0.34). Both pre-specified methods count the shared seed effect (51-65 % of per-image variance) as noise
and so understate the accuracy expected at a given transfer (seed-paired bootstrap at the nominal level about 0.53 /
0.70). The registered scorer's chance level is about 0.22, not 0.20 (own-body main effect leaves the adapter out).
Transporting the primary limits to the five-body groups is an assumption (each body's real-photo contrast varies
2.7-fold within Kodak and 3.4-fold within P20).

**What changes in the paper.** V-D's "a test too small to tell chance from moderate accuracy" is replaced by the
yardstick: had transfer sat at either group's device-level limit, expected accuracy would be at least 0.95 and 3/10
never occurred in the simulations; at the primary limits expected accuracy is about 0.34-0.66 depending on method and
group, close enough to the 0.30 the same bootstrap gives at zero that 3/10 does not separate them (Kodak); for P20
the bootstrap puts 3/10 in the lower tail (P 0.02-0.05) but not once the images' own lean is removed. Table S17 adds
these rows (macros `nAttCs*`).

### 6. Small quantities (descriptive)

- **(a) Table 5 row 3** (primary design retrained on the local stack, six per arm, c = 1): nominal max-arm limit
  **0.588 %** of R_real, set by arm A (arm B alone 0.122 %). Post hoc, without `nomark_s5` (arm A at five): 0.404 %,
  still arm A, 1.6 times the archive's six-per-arm 0.249 % (ledger history_k6, recomputed from seeds 0-5 as
  0.24895 %). The "—" in Table 5 row 3 becomes `\nLimLocalMaxArm`.
- **(b) 16000-step p-values with the estimation shift added** (SE_infl = sqrt(SE_Welch^2 + 1.224e-05^2), Welch df;
  each uninflated p reproduced first):

  | result | theta_sym | p -> p_infl | at the SD's 95 % upper limit 1.62e-05 | level |
  |---|---|---|---|---|
  | Entry 68 replication, seeds 3-5 | +0.164 % | 0.00955 -> **0.023** | 0.034 | 0.05 |
  | Entry 68 pooled six | +0.160 % | 0.00198 -> **0.0080** | 0.015 | 0.01 |
  | G1 inversion, eight per arm (theta < 0) | -0.163 % | 0.0106 -> **0.020** | 0.027 | 0.05 |

  All three stay below their registered levels at the point SD; the pooled-six reading at 0.01 does not survive the
  SD's upper limit (0.015). The within-body inverted-minus-normal contrast is not inflated (the shift cancels).
  The estimation SD was measured at 2000 steps and **200** images per adapter (H2 file), not 500 as Entry 116 says;
  the transport is 2000 steps / 200 images -> 16000 steps / 250 images. Entry 116's "p 0.0096" is 0.0095 (exact
  0.009549; Entry 68 prints 0.00955). Welch df is the conservative choice (Satterthwaite with the estimation term
  gives smaller p).
- **(c) GPU-hours.** Under the registered rule (recorded training minutes, else S3 per-unit times; generation from
  S3's per-unit times), for the 211 adapters and 95,500 generations: A100/Colab including FLUX training 40.1 h;
  L40S 447.7-462.9 h; all GPUs 487.8-503.0 h (66-68 % recorded); all work (227 trained adapters, 105,000
  generations) 511.3-528.9 h. **The rule's Colab figure is too low:** the images' write times on the Drive mirror
  show 52 A100-40GB jobs taking 49-57 min per 500 images (median 55; the check confirmed three at 56.7-57.1 min),
  not S3's "about 20 min". Measured from records and write times (post hoc): Colab about 85 h (including 15.7 h of
  FLUX and 3.0 h of full fine-tuning generation, which the rule cannot estimate); L40S 488 job-hours, during which
  the card was occupied 399 h (jobs overlapped); the rule's L40S generation (139-154 h) is a floor (measured 180 h;
  26 % of 250-image jobs took more than 38.5 min, up to 80 min; ten 2000-step trainings ran about 130 min). Excluded
  throughout: detector, embedding, scoring and autoencoder time, model loading, re-run interrupted jobs, ten pilot
  adapters (4.1 h).
- **(d)** `nDoseInvThreeT` = **-1.27** (theta_sym -0.1427 %, SE 4.009e-05, df 3.15, p 0.145; reproduces Entry 98).
  Table S18's "—" (row "seeds 0-2, first reading", t column) is filled.
- **(e)** `python verify_v2.py` in the repository clone: **76 of 76 checks agree** (working tree, not yet pushed;
  the committed HEAD fc3a979 that the public repository serves has 35 checks, all agreeing). Table 2(b) by stack:
  archive 62 adapters / 31,500 generations; local 149 / 61,000; archive/local 0 / 3,000 (its adapters are primary
  seeds 0-2, counted under archive); total 211 / 95,500. The deposit file count waits on author actions.

**Check: CONFIRMED WITH CORRECTIONS** (both output files and the check's code agree to full precision). Corrections
are to documents, not numbers: Entry 116's "500 images" and "p 0.0096" (above); S3's A100 generation time and the
Table S4 cell must be corrected before any Colab GPU-hour figure is printed; the Colab total's label "including FLUX"
covers FLUX and full fine-tuning training only; the L40S rule range reflects only the 500-image jobs; **Entry 112
over-counts the environment chain**: of its 5 adapters only `colab_s0`-`s2` were trained (`local_A_raw_s0` and
`local_B_raw_s0` are archived v1 adapters generated again locally), so all work is 227 trained adapters, not 229.

**What changes in the paper.** Table 5 row 3 max-arm cell; S12 prints the inflated p-values (`nDoseRepPInfl`,
`nDoseSixteenPInfl`, `nDoseInvPInfl`) with the transport stated and the upper-limit values beside, and VI-A's verbal
statement of the omission is replaced; S3's per-unit A100 generation sentence is corrected to the measured time and
the GPU-hour estimate is printed as measured job-hours (`nDataGpuColabMeasured`, `nDataGpuLocalMeasured`, with
`nDataGpuLocalOccupied`), labelled as measured from records and write times, a disclosed deviation from Entry 116
(6c), whose rule rests on the per-unit times the write times contradict; Table S18 t cell; the availability section
can print the verifier count (`nDataVerifierChecks`) once the refreshed verifier is pushed (until then the public
verifier has `nDataVerifierChecksPublic` checks); Table 2(b) may use the stack macros.

### Register rows for the lead (not added to the register table: this run appends only)

- R15: "the calibrated headline limit is 0.1752 % (c = 1.25)" (Entry 107) — **superseded as the bound of record** by
  0.1811 % (c* = 1.31) under Entry 116's rule; 0.1752 % stays as H6's value beside it (this entry, item 1).
- R16: "the seed-bank term would put the calibrated limit at about 0.21 %" (REVIEW_REPORT §3.6, scratch probe) —
  withdrawn: that was the incomplete symmetric-only form (registered item (i): 0.206 %); the registered model gives
  0.181 %, and on average the term lowers c (item 1).
- R17: "the DINOv2 scene audit did not compare the two bodies" (manuscript III-B) — resolved (item 3).
- R18: Entry 112's environment chain "5 adapters" and "all work 229 adapters" — 3 trained; all work 227 (item 6).
- R19: Entry 116's "estimation SD measured at 2000 steps and 500 images" and "p 0.0096" — 200 images; p 0.0095.
- R20: S3's "500 generations about 20 min" on the A100 — about 55 min per 500 (write times; item 6c).

### What this entry decides for the paper (summary)

1. The bound of record at 2000 steps is **0.181 %** of R_real (calibrated max-arm, c = 1.31), with H6's 0.175 %
   (c = 1.25) and the nominal 0.151 % beside it; it is described as calibrated for adapter, fingerprint-estimation,
   training-set and seed-bank variation, with the statement that the seed-bank term's effect on c is within the
   simulation's Monte Carlo error. All rates quoted at the calibrated limit move to the bound of record.
2. Eq. (power) is supported end to end at G <= 100 for transfer at the nominal limit and at ten times it.
3. The cross-body scene overlap exists (D200, P20) and does not move the limits; the D200 training sets share scenes.
4. The weights distinguish fingerprint-reversed from unchanged training at the design's floor, without separating
   the fingerprint from the dither or from any fixed pattern.
5. The closed-set result gets a yardstick (device-level limits would have given near-perfect accuracy).
6. Table 5 row 3, the inflated 16000-step p-values, the GPU-hours (measured), Table S18's t and the verifier count
   are filled.

---

## Entry 118 · 2026-10-01 · Post-hoc sensitivities from the fourth review round (registered before computing; descriptive, no reading attaches)

Registered before either quantity below is computed. Both are **post hoc and descriptive**: neither changes the
bound of record (0.1811 %, Entry 117 item 1), a registered reading, or any earlier number. They answer two confirmed
findings of the fourth review round (hostile-r4-01, hostile-r4-02; the latter is REVIEW_REPORT §3.6 hostile-r2-12).

**What has already been seen, stated plainly.** The review recomputed both in a scratchpad (nothing written to
`out/`): (A) with the training-set SD doubled, c of about 2.10 and a limit of about 0.26 %; at the chi-square(2) 95 %
upper confidence limit of that SD no c up to 4.0 reached 0.99 (0.985 at c = 4), all without the seed-bank term;
(B) eq. (power) at twice the nominal limit gives TPR about 0.21 at 500 images and about 1,160 images for one half,
and at twice the bound of record about 0.31 and about 800. This entry fixes the method so that the paper prints
values from a result file, with the seed-bank term included as the bound of record requires.

**Common rules.** Script `src/fv/fv_r4_sens.py`, result `out/fv_r4_sens.json` (new names; neither exists). CPU only.
Numbers reach the manuscript only through a new `src/fv/num_n10.py`.

### A. The calibration multiplier if the training-set component is larger than measured (hostile-r4-02)

**Why.** c* is set almost wholly by the training-set component (with estimation error alone c = 1.00 suffices,
Entry 107). Its SD, 2.201e-05 (`h4_coverage2.training_sd()`), comes from two shifts (one per body) with the sampling
variance subtracted, so it has at most two degrees of freedom; Entry 97 calls it "a bound rather than a precise
variance". The paper says "calibrated for training-set variation of the measured size" without saying what a larger
component would need.

**Method.** Entry 116 item 1's simulation exactly as `src/fv/fv_seedbank_calib.py` runs it (its functions imported,
not re-implemented): H6's 27 cells, 4,000 replications, master seed 106061, estimation SD 1.224e-05; the bank shift
from the registered generator (default_rng(116001), per cell in H6's order, z L^T with L the Cholesky factor of the
point Sigma_v / 500). The training-set SD is multiplied by f in {1, 2, f95}, f95 = sqrt(2 / chi2.ppf(0.05, 2)) = 4.415
(the one-sided 95 % upper confidence limit of an SD on two degrees of freedom). For each f, with and without the bank
term (the same draws otherwise; common random numbers): worst-cell max-arm coverage at c = 1, 1.25 and 1.31; the
smallest c on the grid 1.00, 1.01, ... with worst-cell coverage >= 0.99, the grid extended in steps of 0.01 up to
c = 10 (reported as "not reached by 10" otherwise); and the limit at that c applied to the ledger values as H6 does
(% of R_real, binding arm). Assertions: f = 1 reproduces c = 1.25 without the term and c* = 1.31 with it, and their
worst-cell coverages, exactly.

**Reporting.** Descriptive. The Monte Carlo SD of c is about 0.05 at f = 1 (Entry 117) and is not re-estimated here;
values are quoted to two decimals for c and three significant figures for the limit, with that caveat. If f = 2 or
f95 needs a c that the grid does not reach, the paper says so; no value is extrapolated.

### B. One body's transfer under the max-arm limit (hostile-r4-01)

**Why.** The max-arm limit bounds the transfer averaged over the two bodies, (1/2)[(i_AA - i_AB) + (i_BB - i_BA)]
(S07 eq. armexp). Each arm mean carries +-(b_A - b_B), which the design cannot separate from unequal transfer, so one
body's transfer t_x = i_xx - i_xy is bounded only by t_A + t_B <= 2U; with the other body's transfer non-negative,
t_x <= 2U. The examiner rates of eq. (power) are printed for transfer equal to U.

**Method.** eq. (power) as `src/t3_power_v4.py` and Entry 114/117 evaluate it (two candidates, FPR 0.01,
sigma_mu = 0, SE_500 from `out/t3_power_v4.json` inputs.SE_img_500, R_real from the ledger): at theta = U and theta =
2U for U the nominal U_device (ledger) and the bound of record's U_device (`out/fv_seedbank_calib.json`), the TPR at
500 and 5,000 images and the exact G for TPR 0.5; 2U as % of R_real; and the TPR averaged over the two bodies when one
carries 2U and the other none ((TPR(2U) + 0.01) / 2 at 500 images). Assertion: theta = U reproduces nAttTprZeroTwoFiveHundred
(0.059), nAttZeroGFiftyExact (about 4,600), nAttTprZeroTwoFiveHundredRec (0.080) and nAttZeroGFiftyExactRec (about
3,200). Descriptive; it states what the limit does not bound, and does not replace a printed rate.

---

## Entry 119 · 2026-10-01 · Results of the Entry 118 sensitivities (post hoc, descriptive)

Run as registered: `src/fv/fv_r4_sens.py` -> `out/fv_r4_sens.json` (CPU, seconds). Both reproduction assertions
passed (f = 1 gives H6's c = 1.25 and item 1's c* = 1.31 / 0.1811 % exactly; theta = U gives 0.0591 / 4,634 images
nominal and 0.0796 / 3,208 at the bound of record, the printed rates). Numbers reach the manuscript through
`src/fv/num_n10.py` (entry "119"). No registered reading or earlier value changes.

### A. Calibration multiplier against the size of the training-set component

Training-set SD 2.201e-05 (measured, two contrasts); f95 = 4.415 (chi-square(2) one-sided 95 % upper limit of an SD).
Worst-cell max-arm coverage (27 cells, 4,000 replications) and the smallest c reaching 0.99, with the limit at that c
on the ledger values (% of R_real).

| training-set SD | seed-bank term | coverage c = 1 | c = 1.25 | c = 1.31 | c* | limit at c* | binding arm |
|---|---|---|---|---|---|---|---|
| x1 (2.20e-05) | without | 0.9795 | 0.9900 | 0.991 | 1.25 | 0.1752 % | B |
| x1 | with | 0.981 | 0.98875 | 0.990 | **1.31** | **0.1811 %** (bound of record) | B |
| x2 (4.40e-05) | without | 0.938 | 0.958 | 0.962 | 2.10 | 0.259 % | B |
| x2 | with | 0.954 | 0.967 | 0.9695 | **1.99** | **0.248 %** | B |
| x4.415 (9.72e-05) | without | 0.857 | 0.882 | 0.887 | 4.44 | 0.514 % | A |
| x4.415 | with | 0.8815 | 0.896 | 0.901 | **4.21** | **0.486 %** | A |

**What it says.** The bound of record is calibrated to the training-set component's point estimate. Were that
component twice its estimate, the rule (with the seed-bank term) would set c = 1.99 and a limit of 0.248 % of R_real;
at the component's 95 % upper confidence limit, c = 4.21 and 0.486 %. The review's "no c up to 4.0 reaches 0.99" at
the 95 % limit is confirmed (it is reached at 4.21-4.44). As at f = 1, the seed-bank term lowers c slightly at every f
(Monte Carlo SD of c about 0.05 at f = 1, not re-estimated at larger f). Two consequences for the wording: the
16000-step estimate (0.160 %) lies below the bound of record only for the measured component, and the paper already
says only that it is "not shown to exceed" the standard-dose bound; and the fifty-image requirements (3.2 to 9.6
times the nominal limit, 0.48-1.45 % of R_real) stay above the limit at f = 2 (0.248 %) and are met at their lower
end at f95 (0.486 %, 3.2 times nominal). The examiner rates quoted at the nominal limit do not use c.

**Not done.** f was not varied jointly with the estimation SD or Sigma_v; the 20,000-replication precision check was
not repeated at f > 1; the df of the training-set SD is at most two (the sampling correction makes it less than two in
effect), so f95 is approximate.

### B. One body's transfer under the max-arm limit

The max-arm limit bounds the transfer averaged over the two bodies; one body's transfer is bounded only by 2U (with the
other body's transfer non-negative). eq. (power), two candidates, FPR 0.01, sigma_mu = 0:

| limit | theta | % of R_real | TPR at 500 | TPR at 5,000 | images for TPR 0.5 |
|---|---|---|---|---|---|
| nominal | U | 0.151 | 0.059 | 0.54 | 4,634 |
| nominal | 2U | 0.301 | **0.212** | 0.994 | **1,158** |
| bound of record | U | 0.181 | 0.080 | 0.72 | 3,208 |
| bound of record | 2U | 0.362 | **0.312** | 1.000 | **802** |

Averaged over the two bodies when one carries 2U and the other none, the TPR at 500 images is 0.111 (nominal) and
0.161 (bound of record), above the 0.059 / 0.080 printed for equal transfer: by convexity of the TPR in the shift in
this range, any spread of transfer between bodies (or between adapters, Entry 117 item 2) with the same mean raises
the TPR. The printed rates, counts and "at most / at least" statements therefore hold for transfer equal in both
bodies and constant across adapters; the worst case for one body is the 2U row. 2U (0.301 % and 0.362 %, 2.0 and 2.4 times the nominal limit) stays below the smallest fifty-image requirement
(3.2 times the nominal limit, many seed-matched references).

**What changes in the paper.** III-E names the estimand of the max-arm limit (transfer averaged over the two bodies)
and the 2U bound for one body, and S07 states it with eq. (armexp); V-D and S11 add the one-body worst case
(`nAttOneBody*`); the abstract, I, V-D, VII-C and VIII state the examiner rates for transfer equal in both cameras;
III-E, V-D, S07, S11 and Fig. S10 qualify "rates at sigma_mu = 0 are upper bounds" as holding for a transfer equal in
both bodies and constant across adapters. V-C replaces "the multiplier is approximate" by the item A values
(`nLimTrainSens*`), S07 and Table S10 print them, and the abstract and VIII say "calibrated to measured (variance)
components". III-H lists the item A sensitivity among the descriptive analyses.

**Also printed from Entry 117 item 5's file (no new computation).** The closed-set yardstick's floor at the
device-level limits is quoted as Entry 117 prescribes, "at least 0.95" (`nAttCsDevEFloor`, the lowest of both
fingerprint scorings, both methods and both persistent terms: Kodak parametric at sigma_mu 4.1e-05, 0.954); the
4.1e-05 column is printed with its paired equivalent (about 5.8e-05, conservative; Entry 117 item 5 correction 3);
and the not-pre-specified P20 low/mid reading (observed 4/10) at its device-level limit, 0.848 (bootstrap) and 0.775
(parametric), P(X <= 4) 0.0015 and 0.012, is printed in V-D, S11 and the Table S17 note.
