# Reproducing E-INV

> **If you are looking at a clone that reports 0.32 % as the headline bound, it predates the
> declared extension from six to twelve adapters per arm.** The current figure is
> 0.1507 % (plug-in) with a conservative real-normalized variant of 0.1681 %.
> `analysis/FINAL_LEDGER.json` is authoritative;
> `config/results_corrections.json` records the supersession.


Data: **https://drive.google.com/drive/folders/1yHgioOdyRhEGl83dRAOgIaJhirbSfkIm**

Three tiers. Tier 1 needs nothing but this repository. Tier 3 needs about 60 GPU-hours.

---

## Tier 1 — verify every reported number (2 min, no data, no GPU)

Every headline quantity is in `config/headline_numbers.json`, and every one was checked
against the JSON written by the notebook that produced it. The audit table is
`docs/ADVERSARIAL_REVIEW.md`, Part 1.

```bash
python -c "
import json
h = json.load(open('config/headline_numbers.json'))
c = json.load(open('config/results_corrections.json'))
print('headline quantities:', sum(len(v) for v in h.values() if isinstance(v, dict)))
print('archived values superseded by later corrections:', len(c) - 1)
"
```

**Read `config/results_corrections.json` before trusting any cached JSON.** Two archived
files hold values that later analyses superseded; both are documented with derivations.

---

## Tier 1.5 — independent recomputation from the measurements (~3 min)

Download the experiment roots from Drive, then:

```bash
python verify_einv.py --root /path/to/inv_channel
```

or open `notebooks/00_verify.ipynb` in Colab, which mounts Drive itself.

It recomputes **from the per-row CSVs** and **never reads the cached result JSONs** — those
are what it is checking. It reports `OK`, `DISAGREE`, `skip` (CSV absent) or `info` per
quantity and exits non-zero on any disagreement, so it works as a pre-submission gate. A
partial download verifies whatever it contains rather than failing.

Covered: `R_real`, `R_VAE`, `eta` with its bootstrap interval, pre- and post-reconstruction
AUC, the six-adapter arm means and `U_device`, the **exact** adapter-level sign-flip
p-values over all 64 assignments, the FLUX and full-fine-tuning bounds, the low/mid
symmetric bound *and* the contaminated per-arm value it must not be confused with, the
Kodak ICC and device-level sign-flip p, and the additive decomposition's cross-seed R²,
device-level correlation and exact 120-assignment permutation p. It also runs the
leave-flagged-out copy-audit sensitivity when `E_FULLFT/csv/f5_copy.csv` is present.

**Expect it to disagree with two archived JSONs.** `E_LOWFREQ`'s stored λ_U is a per-arm
bound where the symmetric one is correct, and `XVAL`'s stored `p_perm` came from a
permutation that could not reject. The verifier recomputes both properly, so it agrees with
the manuscript and disagrees with those files. That is intended, and it says so in the
output.

---

## Tier 2 — recompute and re-audit from source (~20 min)

`notebooks/90_consolidate.ipynb` recomputes every headline quantity from the per-row CSVs,
**self-audits against the cached JSONs**, prints a `*** DISAGREE ***` line for any drift,
regenerates the figures, and writes `CONSOLIDATED_NUMBERS.json`.

`notebooks/91_paper_prep.ipynb` re-verifies the bibliography against Crossref, OpenAlex and
arXiv, and reproduces the amplitude reconciliation.

`analysis/coverage_sim.py` reproduces the coverage simulation behind the bound
construction. CPU, two minutes, no data needed.

`analysis/motif_check.py` tests whether the fingerprint and query splits share photographic
motifs, using a threshold calibrated from the acquisition-index gap curve rather than chosen.
About 25 minutes; needs the source images but no training.

`analysis/FINAL_LEDGER.json` holds every reported quantity, and `make_figures.py` reads from
it. To check that a figure matches the text, compare both against the ledger rather than
against each other.

---

## Tier 3 — rerun the experiments from raw images (~60 GPU-hours)

### Data

The Dresden Image Database is not redistributed. Obtain it from its original distributors
and place it at `MyDrive/forensic_datasets/dresden/Dresden_Exp/`. Device selection, split
sizes and guard bands needed to reproduce the manifests exactly are in
`docs/E_INV_RESULTS_v2.md` §1. `notebooks/99_dataset_fetch.ipynb` includes a screener that
lists which camera models have enough same-model bodies.

For notebook 13, the Daxing Smartphone Identification Dataset is likewise not
redistributed. Note that a partial copy may lack stills for most devices — that notebook's
stage D0 reports what it actually finds before anything else runs.

### Environment

Colab or equivalent, A100-40GB minimum. **FLUX at 1024 and full fine-tuning need 96 GB.**
bf16 required; the notebooks halt on an fp16 fallback.

```
diffusers>=0.31  transformers  peft  accelerate  safetensors
PyWavelets  scipy  pandas  numpy  torch  sentencepiece  protobuf
scikit-learn  ImageHash          # notebooks 13 and 14 only
```

A Hugging Face token with the Stable Diffusion 3.5 and FLUX.1-dev licences accepted must be
in Colab Secrets as `HF_TOKEN`.

### Order

Run in the order of this table, which is **not** notebook-number order: `12_seed_ext2` extends notebook 02 and must run straight after it, before anything that reads the
twelve-adapter bound. Each notebook exposes `C.STAGES`; set it to a single stage to run
one step. All stages are resumable and skip completed work, so an interrupted session can
simply be rerun.

| # | notebook | wall | note |
|---|---|---|---|
| 01 | pilot | 14 h | produces the manifests everything else depends on |
| 02 | seed extension, seeds 3-5 | 5 h | |
| 12 | seed extension, seeds 6-11 (`12_seed_ext2.ipynb`) | 11 h | the declared extension to twelve adapters per arm; **produces the headline bound** |
| 03 | amplitude and controls | 5 h | |
| 04 | quick controls | 30 min | |
| 05 | Track A bounds | 1.5 h | CPU |
| 06 | VAE screen | 40 min | |
| 07 | FLUX E4 | 13 h | 96 GB |
| 08 | multi-device | 9 h | |
| 09 | low/mid representation | 1.6 h | |
| 10 | full fine-tuning | 5 h | 96 GB; needs fp32 master weights |
| 11 | amplitude ceiling | 10 min | **stage H0 only** |
| 13 | Daxing smartphone | 11 h | run D0 and D1 first and stop |
| 14 | decoder float calibration | 1 h | independent of the others |
| 90 | consolidate | 10 min | |

**Notebook 11: run stage H0 only.** H0 establishes that effective amplitude saturates near
4× at any nominal amplitude, which is the result. H1–H4 would train adapters at 3.96×
effective when a null at 3.48× already exists.

**Notebook 13: run D0 and D1 first and read the gate.** Modern smartphones apply
multi-frame fusion and heavy denoising, and whether device PRNU survives is genuinely
uncertain in advance. If the gate fails, that is the result — do not widen the crop or swap
the extractor to rescue it.

**Notebook 10 needs fp32 master weights.** AdamW's step is about the learning rate
regardless of gradient scale; bf16 spacing at |θ| is |θ|/128, so at LR 2e-6 updates survive
only where |θ| < 2.6e-4 and every larger weight is frozen. Peak memory 44 GB with fp32
masters against 22 GB without — and without, most of the model does not train.

---

## v2 extension experiments

**Check (seconds):** `python verify_v2.py` recomputes the v2 headline numbers from `v2/workspace/out/`.

**Recompute the analyses (minutes, CPU):** with `EINV_V2` pointing at `v2/workspace`, the derived scripts
in `v2/src` (`band_derived.py`, `a1_derived.py`, `v4_offline.py`, `t3_power_v4.py`, `f7_stats.py`,
`t5_derived.py`) and the figure scripts run on the shipped files alone.

**Rerun the experiments (GPU):** follow [`v2/README.md`](v2/README.md) for setup. The `orchestrate*.sh`
files in `v2/src` are the run chains in the order they were executed, and
[`v2/PRESPECIFICATION_LOG.md`](v2/PRESPECIFICATION_LOG.md) records what each was meant to decide before it
ran.

---

## Layout produced on Drive

```
inv_channel/
├── E_INV_P0_v3/     base study: manifest/ fingerprints/ adapters/ gens/ csv/ meta/
├── E_SEEDEXT/       seeds 3-5
├── E_AMP/           amplitude sweep, controls, detector calibration
├── E_TRACKB/        VAE screen
├── E_TRACKB2_FLUX/  FLUX E4
├── E_MULTIDEV/      Kodak
├── E_LOWFREQ/       second representation
├── E_AMPHI/         amplitude ceiling
├── E_FULLFT/        full fine-tuning
├── E_DAXING/        smartphone replication      (notebook 13)
└── E_FLOATCAL/      decoder float calibration   (notebook 14)
```

---

## If a number does not reproduce

1. Check `config/results_corrections.json` — it may be one of the two known superseded
   values.
2. Check the protocol hash in that experiment's `meta/`. A configuration change invalidates
   dependent artifacts by design.
3. Check the column name. `s5_measure.csv` carries eight numeric columns and the study's
   statistic is **`rho_mult`**; `b3_measure.csv` carries only that one. A generic
   "last float column" heuristic picks the wrong one.
4. Run `verify_einv.py --verbose`, which prints the column layout of every CSV it finds.
