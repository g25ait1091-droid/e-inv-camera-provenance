# E-INV — repository manifest

**Does passive, PRNU-derived device identity from a physical camera survive LoRA personalisation
and text-to-image generation?**

Answer: it survives the latent autoencoder (η = 0.366), the training objective responds to
fixed-pattern energy non-specifically, and device-specific transfer into generated images is
bounded at λ_U = 0.32% of the real-image device contrast.

---

## Directory layout

```
e-inv/
├── README.md                         ← this file
├── docs/
│   ├── E_INV_RESULTS_v2.md           ← every measured number (the source of truth)
│   ├── E_INV_V1_PLAN_v11.md          ← plan, contributions, drafting guidance
│   ├── E_INV_P0_design.md            ← original design + adversarial registry
│   ├── E_INV_RUNLIST.md              ← experiment checklist
│   └── E_INV_REFCHECK.md             ← reference verification log
├── notebooks/
│   ├── 01_pilot_v3_2.ipynb           ← S0–S7, the base study
│   ├── 02_seed_ext.ipynb             ← seeds 3–5, arms A and B
│   ├── 03_amp.ipynb                  ← amplitude sweep + Q/shift controls + E-POST
│   ├── 04_quick_controls.ipynb       ← norm recovery, Q realisations, second extractor
│   ├── 05_track_a.ipynb              ← bootstrap, injection integrity, scope
│   ├── 06_track_b_screen.ipynb       ← 5-autoencoder VAE screen
│   ├── 07_track_b2_flux.ipynb        ← E4 on FLUX.1-dev
│   ├── 08_track_c_multidev.ipynb     ← Kodak 5 devices × 2 seeds
│   ├── 09_consolidate.ipynb          ← recomputes everything, self-audits
│   └── 10_dataset_fetch.ipynb        ← dataset acquisition + same-model screener
├── cells/                            ← drop-in replacements for individual stages
└── results/                          ← CSVs, JSONs, figures (see the map below)
```

---

## Notebook → result map

| notebook | stage | writes | RESULTS §ects |
|---|---|---|---|
| **01_pilot_v3_2** | S0 discovery | `manifest/manifest.{csv,json}` | §1 |
| | S1 fingerprints | `fingerprints/K_*.npy`, `csv/s1_positive_control.csv`, `s1_gates.json` | §1 — κ = 0.0073, AUC 1.0000, R_real 3.567e-02 |
| | S1b VAE round-trip | `csv/s1b_vae_roundtrip.csv`, `s1b_vae.json` | §2 — η = 0.3661, AUC 0.9814 |
| | S1c suppression | `s1c_suppression.json`, `train_png/A_{raw,clean,swap}` | §1 — 7.0× / 201× |
| | S2 adapters (14) | `adapters/*/pytorch_lora_weights.safetensors`, `train_meta.json` | §3, §8.2 — ‖lora_B‖ |
| | S3 generation | `gens/<tag>/*.png` (7500) | — |
| | S4 copy audit | `csv/s4_copy.csv` | §8.3 — 0.0% on A arms |
| | S5 measurement | `csv/s5_measure.csv`, `surfaces/*__KA.npy` | §4.1 |
| | S6/S7 verdict | `E_INV_P0_results.json`, `figs/` | §4.1 — C0, S_A p = 0.494 |
| **02_seed_ext** | B1–B4 | `csv/b3_measure.csv`, `seedext_pooled.json` | §4.2 — n = 6, λ_U 0.699% → 0.249% |
| **03_amp** | A1 materialise | `csv/a1_real_dose.csv`, `train_png/amp_*` | §5.1 — α_effective |
| | A2 adapters (7) | `adapters/{amp_*,ctrl_Q,ctrl_shift}` | §3 — loss dose-response |
| | A3 generation | `gens/*` (3500) | — |
| | A4/A5 | `csv/a4_measure.csv`, `E_AMP_results.json` | §4.3, §8.1 |
| | E-POST-LOW-Q | `csv/epost_low.csv`, `epost_lowq.csv` | §5 — slope 1.4142e-02 |
| **04_quick_controls** | Q1 norm recovery | `quick/covariates.csv` | §8.2 |
| | Q2 Q realisations | `quick/q_realisations.csv` | §8.1 step 1 — z = −0.68 |
| | Q3 second extractor | `quick/e5_extractors.csv` | §8.3 — 1.970e-04 vs 1.918e-04 |
| **05_track_a** | A1 bootstrap | `trackA/A1_bootstrap.json` | §4.2 — BCa 0.3236% |
| | A2 integrity | `trackA/A2_injection_integrity.csv` | §5.1 |
| **06_track_b_screen** | B1 | `B1_screen.json`, `csv/b1_<model>.csv` | §2 — five-autoencoder table |
| **07_track_b2_flux** | F1–F4 | `adapters/*_flux`, `csv/f3_measure.csv`, `E4_flux_results.json` | §6.1 — λ_U 0.777% at n = 3 |
| **08_track_c_multidev** | C0–C4 | `csv/c0_positive_control.csv`, `c3_measure_{raw,lodo}.csv`, `C4_multidev.json` | §6.2, §7 — ICC 0.943, R² 0.959 |
| **09_consolidate** | — | `CONSOLIDATED_NUMBERS.json`, `figs_paper/F1–F4.png` | all |

**Experiment roots on Drive:** `inv_channel/E_INV_P0_v3` (base), `E_SEEDEXT`, `E_AMP`,
`E_TRACKB2_FLUX`, `E_MULTIDEV`, `E_TRACKB`.

---

## Headline numbers and where each comes from

| number | value | notebook | file |
|---|---|---|---|
| κ_model | 0.0073 | 01 | `s1_gates.json` |
| R_real | 3.56703e-02 | 01 | `csv/s1_positive_control.csv` |
| η device-specific VAE retention | 0.3661 | 01 | `s1b_vae.json` |
| **λ_U (headline)** | **0.32%** | 05 | `trackA/A1_bootstrap.json` |
| **τ_U** | **0.91%** | 05 | same |
| λ̂ | +0.0419% | 02 + 05 | `seedext_pooled.json` |
| detector slope (quantised) | 1.4142e-02 | 03 | `csv/epost_lowq.csv` |
| quantisation cost | 1.4% | 03 | same |
| FLUX λ_U (n = 3) | 0.777% | 07 | `E4_flux_results.json` |
| Kodak ICC | 0.943 | 08 | `C4_multidev.json` |
| additive R² (in-sample) | 0.959 | 08 | same |

---

## Reproduction order

CPU-only: `09_consolidate` reproduces every number from the per-row CSVs and self-audits against
the cached JSONs. Run it after any change; a `*** DISAGREE ***` line means drift.

Full rebuild: 01 → 02 → 03 → 04 → 05 → 06 → 07 → 08 → 09. Roughly 60 GPU-hours.

## Environment
A100-40GB or better; **96 GB needed for FLUX at 1024** (peak 41 GB). bf16 required.
Known traps, all patched in the notebooks:
- peft's `is_torchao_available()` **raises** on torchao < 0.16 (Colab ships 0.10.0) — patch both
  `peft.import_utils` and `peft.tuners.lora.torchao` before any `add_adapter`.
- PIL infers format from the extension: `im.save(path, format="PNG")` when the temp name ends `.tmp`.
- **TF32 must stay disabled** on the measurement path.
- Helper defs (`adir`, `gdir`) belong in the config cell, not inside a stage cell, or selective
  stage runs raise `NameError`.

---

**Reproduction:** see [REPRODUCE.md](REPRODUCE.md) — three tiers, from verifying the numbers with no data through a full ~60 GPU-hour rerun.
