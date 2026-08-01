# E-INV — EXPERIMENT RUN LIST
**Everything left to run, in order.** 2026-07-31. Rationale lives in `E_INV_V1_PLAN_v8.md`;
numbers in `E_INV_RESULTS.md`.

## Summary

| # | Experiment | GPU | Wall | Blocks | Status |
|---|---|---|---|---|---|
| **A1** | Hierarchical bootstrap (bound sensitivity) | no | ~1 h | any submission | to run |
| **A2** | Injection integrity + α_effective axis | no | ~30 min | any submission | to run |
| **A3** | Scope paragraph (writing, not compute) | no | ~30 min | any submission | to run |
| **B1** | E-VAE-SCREEN — 5 candidate systems | yes | ~40 min | B2 | to run |
| **B2** | E4 — second generative system, 6 adapters | yes | ~4.5 h | TIFS | after B1 |
| **C0** | Kodak_M1063 S0/S1 — 5 devices, fingerprints | yes | ~1 h | C1 | to run |
| **C1** | E-MULTIDEV — crossed 5 devices × 2 seeds | yes | ~8 h | TIFS | after C0 |
| **D1** | E-PROMPT — diverse prompt bank | yes | ~2 h | — | optional |
| **D2** | E2 — dataset size, 2 confounding-control arms | yes | ~4 h | — | optional |
| **D3** | E3 — random-crop arm | yes | ~3 h | — | optional |

**Required core: A1–A3 (2 h, no GPU) + B1–B2 + C0–C1 ≈ 16 GPU-hours.**
Access needs only A. TIFS needs A + B + C.

Run `E_INV_CONSOLIDATE.ipynb` after every experiment — it recomputes all numbers from raw CSVs and
self-audits against the cached JSONs.

---

## TRACK A — analysis only, no GPU. **Do these first; they block everything.**

### A1 · Hierarchical bootstrap
**Produces:** λ_U and τ_U bootstrap percentiles beside the conservative-limit values.

Resample, 2000 iterations, in this nesting:
1. fingerprint-estimation images (E1, E2) → re-estimate K̂ per device
2. held-out reals (H) → re-estimate R_real and R_VAE
3. adapters at cluster level, within arm
4. generations within each resampled adapter

Recompute λ_U and τ_U **inside** the loop.
**Critical:** each resampled K̂ must be held fixed across every downstream observation in that
iteration. All correlations in an iteration share the same fingerprint; resampling already-computed
ρ values independently is wrong.

**Report:** plug-in t-based | conservative-limit (**headline**) | BCa or bootstrap-t percentile.
Six clusters give a coarse upper tail — **the bootstrap is sensitivity, not the headline.**

### A2 · Injection integrity and the α_effective axis
**Produces:** a table making the amplitude axis physical rather than nominal.

From `a1_real_dose.csv` (already collected): α_effective = R_injected(α) / R_natural(B).
Add per level: post-clipping RMS fingerprint amplitude, clipped-pixel fraction, PSNR, SSIM, and
the residual spectrum before/after injection.
**Replot every amplitude figure on α_effective.**

### A3 · Scope paragraph
Two physical Nikon D200 bodies · SD-3.5-medium · rank-16 attention LoRA · 50 training images ·
2000 steps · fixed-coordinate 1024² centre crop · txt2img only · one dataset (Dresden, 2010-era
CCD). A paragraph, not a footnote.

---

## TRACK B — cross-system robustness

### B1 · E-VAE-SCREEN
**VAE forward passes only. No training, no generation.**

Candidates: SD-1.5 · SDXL · SD-3.5-medium (incumbent) · FLUX.1-dev · PixArt-Σ.
Measure the **device-specific paired contrast** R_VAE_post / R_VAE_pre on the H split — the same
statistic as S1b, not the scalar retention.

**Selection rule, fixed before any generation result is examined:**
1. exclude any model whose post-reconstruction same-model AUC < 0.75
2. among survivors, take the highest device-specific retention η
3. **report every model screened, including rejects**

**Why not just use SD-1.5:** a tighter autoencoder makes the null easier and proves nothing.

### B2 · E4 on the selected system
6 adapters (3 seeds × 2 arms), 6 × 500 generations, measurement.
**The question is whether the null replicates**, not what the bound is on system two. Report the
IUT p-values and a clustered bound explicitly flagged as n = 3.

**Call it a cross-system robustness test, never an architecture ablation** — VAE, backbone, latent
dimensionality, text encoder, implementation and scale all co-vary.

---

## TRACK C — device-level generality

### C0 · Kodak_M1063 setup
5 same-model bodies, 438–571 images each. Run S0 discovery first to confirm counts survive the
uniform-resolution filter, then S1 fingerprints.
Splits per device: E1 = 80, E2 = 140, T = 50, H = 40, guard 10 (needs 340; all five clear it).

**κ_model by leave-one-device-out:** κ̂^(−d) = ¼ Σ_{j≠d} K̂_j. Residualise device d against the
*other four*. Report results **both raw and κ-residualised** — this is what five bodies buy that
the D200 pair could not.

### C1 · E-MULTIDEV, crossed
**5 devices × 2 seeds = 10 adapters.** One adapter per device would confound device identity with
training seed.

- Contrast per adapter: θ_{d,s} = mean over the other four devices d′ of
  ρ(gens_{d,s} → K̂_d) − ρ(gens_{d,s} → K̂_{d′})
- Model: θ_{d,s} = μ + u_d + v_{d,s}, separating between-device from adapter-training variance.
  Two seeds per device won't resolve a device × seed interaction precisely, but it prevents
  complete confounding.
- **Primary protocol stays fixed:** identical optimiser, schedule, steps, effective batch, no
  norm-dependent intervention.
- **Adaptation-strength matching is secondary and pre-specified:** ‖lora_B‖ as a covariate, *or*
  inference-time adapter scaling at matched effective norm, *or* one additional norm-targeted arm.
  Norm-targeted early stopping changes the update count and substitutes one confound for another.

---

## TRACK D — optional

| | What | Note |
|---|---|---|
| **D1** | E-PROMPT diverse bank | generation + measurement only, no retraining. Keep the uniform caption as the primary bank — varied prompts pull generations off the fine-tuned mode |
| **D2** | E2 dataset size, N ∈ {10, 25, 100, 200} | needs **constant-compute and constant-exposure** arms; N and exposures-per-image confound at fixed steps |
| **D3** | E3 random crop | measurement switches to translation-tolerant PCE_max. This is a **scope limit**, not an a fortiori case — random crop may teach translation-invariant statistics |

---

## Dropped, with reasons

| | Why |
|---|---|
| **E-SHIFT-REP** | K_Bshift ranks 1/32 in arms that never received it — template-specific artifact, not alignment-specific transfer |
| **More seeds beyond n = 6** | n = 9 buys t 4.03 → 3.36 for five hours |
| **SSCD copy detector** | copy-flag rate is 0.0% on the A arms; nothing for a better detector to find |
| **Clean-arm mirror, transition seeds** | moot — nothing transferred at any amplitude |
| **Daxing second dataset** | Baidu Pan only, not scriptable. Name the one-dataset limitation instead of arguing it away |

---

## Environment checklist — every GPU notebook

- `HF_TOKEN` in Colab Secrets; A100 runtime (bf16 required)
- peft/torchao patch: `is_torchao_available = lambda: False` in both `peft.import_utils` and
  `peft.tuners.lora.torchao`, **before** any `add_adapter`
- `torch.backends.cudnn.allow_tf32 = False` and the matmul equivalent
- `im.save(path, format="PNG")` — PIL cannot infer format from `.tmp`
- Dead-adapter check (‖lora_B‖ > 0) and a non-finite-loss halt
- New `OUT_DIR` per experiment — `TRAIN_SEEDS`, `N_T` and the arm registry are all inside the
  hashed config, so changing them invalidates existing adapters

**Timings on A100-40GB:** rank-16 adapter ≈ 23.8 min · rank-64 ceiling ≈ 38 min · 500 generations
at 1024² ≈ 20 min · GPU-batched measurement ≈ 0.19 s/image.

---

## Still owed, and I cannot do it

**Prior-art sweep before writing:** Scholar forward-citations on Chen–Fridrich–Goljan 2008 and on
SIREN; arXiv full-text for PRNU × LoRA and PRNU × personalisation; **WIFS 2026 proceedings**, which
publish before you submit and are where a short version of this experiment would appear if someone
else ran it.
