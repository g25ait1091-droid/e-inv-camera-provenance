<div align="center">

# E-INV

### Does camera sensor provenance survive diffusion personalization?

**A stage-localized negative result, with positive channel controls and a simultaneous upper bound.**

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Reproduce](https://img.shields.io/badge/reproduce-2%20min%20%E2%86%92%2060%20GPU--h-brightgreen.svg)](REPRODUCE.md)
[![Status](https://img.shields.io/badge/manuscript-in%20preparation-orange.svg)](#citation)
[![Verify v2](https://img.shields.io/badge/verify__v2.py-seconds%2C%20no%20GPU-brightgreen.svg)](#v2--extension-experiments)
[![Data](https://img.shields.io/badge/data-Google%20Drive-4285F4.svg)](DATA.md)

</div>

---

Photo-response non-uniformity (PRNU) is a manufacturing artifact that identifies an
**individual physical camera body** — not a model, a body. Personalized text-to-image
models are routinely fine-tuned on photographs from one camera. This repository asks, and
measures, whether that signature is passively inherited by the model and re-emitted in
images generated from text alone.

**Answer: no, to within a tight bound** — and the study localizes *where* the signal is
lost, rather than reporting a bare null.

> [!IMPORTANT]
> **`analysis/FINAL_LEDGER.json` is the single source of truth for every reported number.**
> If a value is not in that file, it does not belong in the paper. Any file, figure or older
> clone stating **λ_U = 0.32 %** or **τ_U = 0.91 %** predates the declared extension from six
> to twelve adapters per arm; the current values are **0.1507 %** and **0.4117 %**.
> `config/results_corrections.json` records that supersession and two others.

---

## Contents

[The result](#the-result) · [v2 extensions](#v2--extension-experiments) · [Start here](#start-here) · [Layout](#repository-layout) ·
[Notebooks](#notebooks) · [Datasets](#datasets) · [Before you run anything](#before-you-run-anything) ·
[Analysis scripts](#analysis-scripts) · [Known limits](#what-this-does-not-establish) ·
[Citation](#citation)

---

## The result

The study separates three stages and measures each one independently.

| stage | finding |
|:--|:--|
| **1 · Autoencoder round trip** | Device contrast **survives**: η = 0.3661, same-model AUC 1.0000 → 0.9814 |
| **2 · Optimization objective** | Responds to fixed-pattern energy from α ≈ 3 — but **not device- or alignment-specific**: a circularly shifted fingerprint gives the same response, and a spectrally matched Gaussian field beats the true fingerprint by 27 % |
| **3 · Generation** | The registered intersection–union test, applied after the declared extension to twelve adapters per arm, **does not reject at a cluster count where rejection was attainable**. Transfer is bounded at **λ_U ≤ 0.1507 %** of real-image device contrast and **τ_U ≤ 0.4117 %** of what survives reconstruction |

### It replicates

| replication | generalizes over | n | λ_U | construction |
|:--|:--|--:|--:|:--|
| Primary, SD-3.5-medium | adapter seeds, 2 devices | 12 | **0.1507 %** | max-arm |
| FLUX.1-dev | adapter seeds, 2 devices | 3 | 0.777 % | max-arm |
| Full fine-tuning, all 2.24 B parameters | adapter seeds, 2 devices | 3 | 0.747 % | **symmetric** |
| Kodak M1063 — legacy CCD | physical bodies | 5 | 1.74 % | max-arm |
| Huawei P20 — modern smartphone | physical bodies | 5 | 1.14 % | max-arm |
| Low/mid DCT band | adapter seeds, 2 devices | 6 | 9.72 % | **symmetric** |

> [!WARNING]
> **These rows answer different questions and are not interchangeable.** 0.1507 % generalizes
> over adapter *training seeds* on two devices; the five-device limits generalize over
> *physical bodies*. A max-arm bound and a symmetric-interaction bound also answer different
> questions — where an additive main effect dominates, the two differ by up to 3.6×.
> `analysis/FINAL_LEDGER.json` records which construction produced each row.

### Two findings that stand on their own

**Retention tracks latent channel count.** Across five autoencoders, device-specific
retention clusters by latent dimensionality: three 4-channel models at η ≈ 0.075–0.096, two
16-channel models at 0.366 and 0.554.

**A modern smartphone passes *more* device signal, not less.** The paper's Limitations used
to concede that computational-photography devices might carry weaker signatures. Measured,
it points the other way: η = 0.5601, 95 % CI [0.4742, 0.6588], against the 2010 CCD's 0.3661,
[0.344, 0.387]. **The intervals do not overlap.**

---

## v2 — extension experiments

A second round of experiments (September 2026) asks *why* the fingerprint does not come through, and
what an examiner could do with it. Code is in [`v2/src`](v2/src), the small result files are in
[`v2/workspace/out`](v2/workspace/out), and every experiment was written into
[`v2/PRESPECIFICATION_LOG.md`](v2/PRESPECIFICATION_LOG.md) — with the reading that would count as
positive or negative — before its first number was looked at. Details: [`v2/README.md`](v2/README.md).

| question | what was run | outcome |
|:--|:--|:--|
| What does personalization pass? | random fields, one-octave fields, periodic tiles (24–48 px) and the released DiffusionShield watermark, all through the identical pipeline | non-repeating patterns ≤ 0.4 % of their stored contrast, nothing detectable in the finest octave; tiles on the 8-px latent grid 1.4–4.0 %, off-grid tiles 0.2–0.3 %, the watermark 3.3 % |
| Is the fingerprint below what that channel predicts? | the band response weighted by the fingerprint's spectrum | predicted 0.108 %; the symmetric statistic's one-sided 99 % limit is 0.076 %, 3.7 standard errors below the prediction |
| Does another detector see it? | PCE at threshold 60, a low/mid DCT signature, Noiseprint, a learned CNN (and the CNN with the fingerprint projected out) | template detectors: no; the CNN resolves a small body-specific signal that survives removal of the fingerprint template |
| Could an examiner attribute a model? | power at the limit; closed-set attribution on five-body Kodak and Huawei groups | 4–6 % true positives at 1 % false positives from 500 images; closed-set accuracy not distinguishable from chance |
| Does longer training change it? | 8000- and 16000-step adapters; DINOv2 nearest-training-image similarity | 8000 steps: no; 16000 steps: estimate 0.16 % with t = 2.0, unresolved (a second replication is running); no generation is a copy (DINOv2 cosine ≤ 0.86 against the copy threshold 0.90) |

```bash
python verify_v2.py      # recomputes the numbers above from the shipped files; seconds, no GPU
```

---

## Start here

### Tier 0 — nothing but this repository (no data, no GPU)

Every reported quantity is in the ledger, and the figures are drawn from it, so a figure
cannot drift from the text.

```bash
git clone https://github.com/g25ait1091-droid/e-inv-camera-provenance
cd e-inv-camera-provenance
pip install -r requirements.txt

python -m json.tool analysis/FINAL_LEDGER.json | head -40   # every number, seconds
python analysis/make_figures.py                              # redraw all seven figures, seconds
python analysis/coverage_sim.py                              # coverage of the bound, ~9 min
```

### Tier 1 — recompute the numbers from the measurements (~3 minutes)

Download the experiment roots from [Drive](DATA.md), then:

```bash
python verify_einv.py --root /path/to/inv_channel
```

It recomputes **from the per-row CSVs** and **never reads the cached result JSONs** — those
are what it is checking. It defines **44 distinct checks** and exits non-zero on any
disagreement, so it works as a pre-submission gate. A partial download verifies whatever it
contains and reports the rest as `skip` rather than failing.

The k = 12 headline needs the `E_SEEDEXT2` root. Without it the verifier says
`E_SEEDEXT2 root absent` rather than silently falling back to the superseded k = 6 value.

`notebooks/00_verify.ipynb` is the same thing for Colab, with Drive mounting.

> [!NOTE]
> **The verifier is *expected* to disagree with two archived JSONs**, and says so in its
> output. `E_LOWFREQ`'s stored λ_U is a per-arm bound where the symmetric one is correct, and
> `XVAL`'s stored `p_perm` came from a permutation that could not reject. Both are documented
> with derivations in `config/results_corrections.json`.

### Tier 2 and 3

Re-audit from source (~20 min) or rerun everything from raw images (~60 GPU-hours).
See **[REPRODUCE.md](REPRODUCE.md)**.

---

## Repository layout

```
notebooks/        the full pipeline, in run order — see the table below
cells/            drop-in stage cells for notebooks that were built incrementally
analysis/
  FINAL_LEDGER.json   ← every reported quantity; the single source of truth
  make_figures.py     redraws fig1–fig7 from the ledger
  coverage_sim.py     frequentist coverage of the bound constructions
  motif_check.py      scene-motif disjointness with a self-calibrated threshold
  overlapcheck.py     figure collision detector used by make_figures
  fig1..fig7.pdf      the figures as published
config/
  headline_numbers.json     convenience mirror of the ledger, grouped by section
  results_corrections.json  three recorded supersessions, with derivations
docs/             results of record, plan, run order, reference check, audit
                  responses, self-adversarial review
verify_einv.py    command-line verifier (primary study)
verify_v2.py      command-line verifier (v2 experiments), runs on the shipped files
v2/
  src/                    v2 experiment, analysis and figure scripts; paths set in einv_paths.py
  workspace/out/          small v2 result files (JSON, per-image CSV, NPZ), same layout the scripts write
  PRESPECIFICATION_LOG.md dated, append-only log: each experiment's readings fixed before its results
  README.md               how to set up and run the v2 experiments
CITATION.cff      how to cite this repository
requirements.txt  CPU analysis deps; GPU deps listed inline for Tier 3
DATA.md           what is in Drive, and what is deliberately not released
REPRODUCE.md      three reproduction tiers, from 2 minutes to 60 GPU-hours
LICENSE-NOTE.md   licensing, and why source images and fingerprint arrays are withheld
```

### Notebooks

Run in **this** order, which is not notebook-number order: `12_seed_ext2` extends notebook
02 and must run straight after it.

| # | notebook | produces | wall |
|:--|:--|:--|--:|
| 00 | `00_verify.ipynb` | independent recomputation of every headline number | 3 min |
| 01 | `01_pilot.ipynb` | manifests, fingerprints, 14 adapters, 7500 generations, the C0 verdict | 14 h |
| 02 | `02_seed_ext.ipynb` | seeds 3–5; takes the bound from n = 3 to n = 6 | 5 h |
| **12** | **`12_seed_ext2.ipynb`** | **seeds 6–11; the declared extension to twelve per arm — produces the headline bound** | **11 h** |
| 03 | `03_amp.ipynb` | amplitude sweep, Gaussian and shifted controls, detector calibration | 5 h |
| 04 | `04_quick_controls.ipynb` | norm recovery, matched-noise fields, second extractor | 30 min |
| 05 | `05_track_a_bounds.ipynb` | hierarchical bootstrap, injection integrity | 1.5 h |
| 06 | `06_vae_screen.ipynb` | five-autoencoder retention screen | 40 min |
| 07 | `07_e4_flux.ipynb` | FLUX.1-dev replication | 13 h |
| 08 | `08_multidevice.ipynb` | Kodak, five bodies × two seeds, LODO κ | 9 h |
| 09 | `09_lowmid_representation.ipynb` | low/mid DCT representation | 1.6 h |
| 10 | `10_full_finetune.ipynb` | full fine-tuning of all 2.24 B parameters | 5 h |
| 11 | `11_amplitude_ceiling.ipynb` | amplitude ceiling — **run stage H0 only** | 10 min |
| 13 | `13_daxing_smartphone.ipynb` | smartphone replication, Huawei P20, five bodies | 11 h |
| 14 | `14_decoder_float_calibration.ipynb` | replaces the dither model with captured decoder floats | 1 h |
| 90 | `90_consolidate.ipynb` | recomputes and audits everything | 10 min |
| 91 | `91_paper_prep.ipynb` | reference verification, amplitude reconciliation | 20 min |
| 99 | `99_dataset_fetch.ipynb` | dataset screener: which models have enough same-model bodies | 15 min |

Notebooks 13 and 14 are extensions beyond the published study; see `docs/REVIEW_RESPONSE.md`.

---

## Datasets

The photographs this study used — and only those — are mirrored, with full credit to their
creators, in one Drive folder linked from **[DATA.md](DATA.md)**. Download it and set
`EINV_MYDRIVE` to it; the layout matches the paths the code expects. Please cite the original
datasets below, and prefer their official distributors where available. Split definitions live in
`docs/E_INV_RESULTS_v2.md` §1, and notebook 01 stage S0 rebuilds the manifests
deterministically from the image directory — same directory, same manifest.

### Dresden Image Database — primary

Place at `MyDrive/forensic_datasets/dresden/Dresden_Exp/`.

- **A** = `Nikon_D200_1` (380 images), **B** = `Nikon_D200_0` (372) — two bodies of one model.
  The paired same-model contrast is the whole design: comparing two bodies of *one* model is
  what cancels the camera-model and processing-pipeline components.
- Cross-model nulls: `Agfa_DC-733s_0`, `Nikon_D70_1`, `Agfa_DC-830i_0`.
- Five-device crossed replication on `Kodak_M1063` (bodies 0–4).
- Splits: E1 = 80, E2 = 140 (A and B), E2 = 60 (aux), T = 50, H = 40, guard bands of 10.
- Cite: Gloe & Böhme, ACM SAC 2010, [doi:10.1145/1774088.1774427](https://doi.org/10.1145/1774088.1774427),
  pp. 1584–1590. A JDFP version exists — cite the ACM SAC one.

> [!TIP]
> Only **two** Dresden camera models have two or more bodies with enough images for the
> A/B protocol: Nikon D200 and Kodak M1063. `99_dataset_fetch.ipynb` contains the screener
> that establishes this. Discovery is a recursive glob keyed on the filename pattern
> `<model>_<body>_<shot>.JPG`, so a nested or duplicated device folder is counted twice —
> check your copy's layout before trusting a device's image count.

### Daxing Smartphone Identification Dataset — second device class

Five Huawei P20 bodies, codes 1101–1105, 3968 × 2976.
Cite: Tian et al., *IEEE Access* 7:101046–101053, 2019,
[doi:10.1109/ACCESS.2019.2928356](https://doi.org/10.1109/ACCESS.2019.2928356).

Four things that cost time, all handled in notebook 13's stage D0:

- Devices 1101–1104 sit under `image/1101-1104/<dev>/<angle>` but **1105 sits under
  `image/1105/<angle>`** — discovery must try both.
- Exclude `desktop.ini` and `.tmp.driveupload`.
- **Never apply EXIF transpose.** PRNU needs sensor-native pixels.
- **Use one angle folder only.** Angle 90 is portrait, 0 and 180 are landscape. Within-angle-0
  split-half correlation is +0.269; angle-0 against angle-180 is +0.002 — they are not
  PRNU-compatible.

A partial copy of Daxing may lack stills for most devices. Stage D0 reports what it actually
finds before anything else runs.

---

## Before you run anything

**Stages are resumable and gated.** Each notebook exposes `C.STAGES`; set it to one stage to
run one step. Every stage skips completed work, so an interrupted session can simply be
rerun. Gates halt rather than produce a number you cannot trust — a fingerprint-quality gate
caught an inverted spectral filter during development (κ was 0.431 instead of 0.0073), and
**both values are reported in the paper, because the gate working is itself evidence.**

**Protocol hashes bind configuration to artifacts.** Changing any config field inside the
hash invalidates dependent results rather than silently reusing them. `TRAIN_SEEDS`, `N_T`,
`ALPHAS`, `STEPS` and the arm registry are all inside it — change one and every adapter in
that root must be retrained. Use a fresh `OUT_DIR` for a variant.

**Three environment traps, all patched in the notebooks.** `peft`'s
`is_torchao_available()` *raises* rather than returning `False` when `torchao` is older than
0.16 (Colab ships 0.10.0), which kills `add_adapter`. PIL infers format from the file
extension, so temporary names ending `.tmp` need `format="PNG"` — and `np.save` appends
`.npy` for the same reason, so atomic temp writes must pass a file handle. **TF32 must stay
disabled on the measurement path.**

**One column name worth memorising.** `s5_measure.csv` carries eight numeric columns and the
study's statistic is **`rho_mult`**; `b3_measure.csv` carries only that one. A generic
"last float column" heuristic picks the wrong one and produces NaN on concatenation.
`verify_einv.py --verbose` prints the column layout of every CSV it finds.

**Do not run** stages H1–H4 of notebook 11, the better-injection variant, Kodak low/mid,
E-PROMPT, E2, or E3. Each was declined with a reason recorded in `docs/E_INV_V1_PLAN_v13.md`.

---

## Analysis scripts

| script | what it does | cost |
|:--|:--|:--|
| `make_figures.py` | Redraws all seven figures **from `FINAL_LEDGER.json`**, so a figure cannot drift from the text. | seconds |
| `coverage_sim.py` | Measures the frequentist coverage of the upper-limit constructions by simulation, under normal, heavy-tailed and empirically resampled cluster distributions, with the denominator resampled. Plug-in *t* covers at 99.9 % worst-case; BCa alone reaches only 98.2 % under heavy tails. | CPU; nine scenarios × 4000 replications × 800 bootstrap resamples. Measured 531 s on a desktop CPU; it prints each scenario as it finishes |
| `overlapcheck.py` | Computes text, legend and arrow-path bounding boxes in display coordinates and reports collisions with plotted data. Used by `make_figures.py`. | — |
| `motif_check.py` | Calibrates a scene-similarity threshold **from the data itself** — similarity as a function of acquisition-index gap separates same-motif from different-motif pairs without needing labels — then tests whether the fingerprint and query splits share motifs and, if they do, recomputes the denominator on motif-clean subsets. | ~25 min, needs the source images, no training |

> [!NOTE]
> `coverage_sim.py` is written against the **k = 6** cluster means and the three constructions
> reported at that time. It is retained as the record behind the construction choice. The
> plug-in construction it validates is the one carried into the k = 12 headline; BCa is no
> longer reported.

---

## What this does not establish

Stated here because the paper states it, and a reader should not have to find it.

- **The bound is one detector's bound.** Zero-lag normalised correlation against a
  wavelet-denoised residual, fixed centre crop, one fingerprint estimator. A second extractor
  agrees within 3 % and the low/mid band is a second representation — but both are variations
  on the same detection philosophy.
- **Device identity in the two tested frequency representations**, not "camera identity"
  unrestricted. Learned ISP signatures, lens effects and dark-current FPN/DSNU are untested.
- **The adapter-level positive control uses designed patterns, not the fingerprint itself.**
  The v2 experiments train adapters on images carrying patterns of known strength and structure
  and measure what comes out, which calibrates the channel; the fingerprint's own expected
  transmission is then inferred from its spectrum. See `docs/SELF_ADVERSARIAL_REVIEW.md` for the
  primary study's version of this limit.
- **Effective amplitude saturates near 4×**, because the injected field is an estimate, so
  the regime above that is untested.
- **The shifted-template artifact is unexplained.** A circularly shifted fingerprint ranks
  first in arms that never received it. The paired statistic cancels it structurally, which is
  a defence — but not a mechanism.
- **No a fortiori reasoning.** "No detectable transfer for the tested signals and protocol",
  never "fixed patterns cannot transfer" — the active-watermark literature contradicts that.

---

## Citation

A manuscript describing this study is in preparation. **Its text is deliberately not in this
repository**; it will be linked here on publication.

Until then, cite this repository (also available as [`CITATION.cff`](CITATION.cff)):

```bibtex
@software{behera_einv_2026,
  author  = {Behera, Mohini Mohan},
  title   = {E-INV: PRNU-Derived Device Provenance Through Diffusion Personalization},
  year    = {2026},
  url     = {https://github.com/g25ait1091-droid/e-inv-camera-provenance}
}
```

| | |
|:--|:--|
| **Author** | Mohini Mohan Behera · [ORCID 0009-0007-3475-8230](https://orcid.org/0009-0007-3475-8230) · IIT Jodhpur |
| **Supervisor** | Dr. Navchetan Awasthi · School of AI and Data Science, IIT Jodhpur |

Code is MIT-licensed. The Dresden Image Database and the Daxing Smartphone Identification
Dataset carry their own terms; the photographs are not in this repository, and the subset
mirrored on Drive is credited to its creators. See [LICENSE-NOTE.md](LICENSE-NOTE.md) and
[DATA.md](DATA.md).
