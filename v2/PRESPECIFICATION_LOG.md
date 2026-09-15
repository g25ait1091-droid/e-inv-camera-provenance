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
