# Reproducing E-INV

Three tiers. Tier 1 needs nothing but this repository. Tier 3 needs roughly 60 GPU-hours.

---

## Tier 1 — verify every number in the paper (no GPU, no data, ~5 min)

Every quantity in the manuscript is in `config/headline_numbers.json`, and every one of them was
checked against the JSON written by the notebook that produced it. The audit table is in
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

**Read `config/results_corrections.json` before trusting any archived JSON.** Two files on Drive
hold values that a later, corrected computation superseded; the manuscript carries the corrected
ones. Both cases are documented with their derivations.

---

## Tier 2 — recompute from the measurement records (no GPU, ~20 min)

Requires the per-row CSVs from the experiment roots (see Tier 3 for layout). These are the
measurement outputs, not the images: roughly 60 MB total.

1. Place the experiment roots under `MyDrive/inv_channel/` or set `EINV_DRIVE`.
2. Run `notebooks/90_consolidate.ipynb`.

It recomputes every headline quantity from the per-row CSVs, **self-audits against the cached
JSONs**, and prints a `*** DISAGREE ***` line for any drift. It also regenerates the four figures
and writes `CONSOLIDATED_NUMBERS.json`.

3. Run `notebooks/91_paper_prep.ipynb` to re-verify the bibliography against Crossref, OpenAlex and
   arXiv, and to reproduce the amplitude reconciliation.

---

## Tier 3 — rerun the experiments from raw images (~60 GPU-hours)

### Data
The Dresden Image Database is not redistributed here. Obtain it from its original distributors and
place it at `MyDrive/forensic_datasets/dresden/Dresden_Exp/`. The device selection, split sizes and
guard bands needed to reproduce the manifests exactly are in `docs/E_INV_RESULTS_v2.md` §1.
`notebooks/99_dataset_fetch.ipynb` includes a screener that lists which camera models have enough
same-model bodies.

### Environment
Colab or equivalent, A100-40GB minimum. **FLUX at 1024 and full fine-tuning need 96 GB.** bf16
required; the notebooks halt on an fp16 fallback.

```
diffusers>=0.31  transformers  peft  accelerate  safetensors
PyWavelets  scipy  pandas  numpy  torch  sentencepiece  protobuf
```

A Hugging Face token with the Stable Diffusion 3.5 and FLUX.1-dev licences accepted must be in
Colab Secrets as `HF_TOKEN`.

Three environment traps, all already patched in the notebooks:
- `peft`'s `is_torchao_available()` **raises** rather than returning `False` when `torchao` is older
  than 0.16 (Colab ships 0.10.0), which kills `add_adapter`.
- PIL infers format from the file extension; temporary names ending `.tmp` need `format="PNG"`.
- **TF32 must stay disabled** on the measurement path.

### Order

| # | notebook | produces | wall |
|---|---|---|---|
| 1 | `01_pilot.ipynb` | manifests, fingerprints, 14 adapters, 7500 generations, the C0 verdict | ~14 h |
| 2 | `02_seed_ext.ipynb` | seeds 3–5 for arms A and B; takes the bound from n=3 to n=6 | ~5 h |
| 3 | `03_amp.ipynb` | amplitude sweep, Q and shifted controls, E-POST calibration | ~5 h |
| 4 | `04_quick_controls.ipynb` | norm recovery, four independent matched-noise fields, second extractor | ~30 min |
| 5 | `05_track_a_bounds.ipynb` | hierarchical bootstrap, injection integrity, scope paragraph | ~1.5 h (CPU) |
| 6 | `06_vae_screen.ipynb` | five-autoencoder retention screen | ~40 min |
| 7 | `07_e4_flux.ipynb` | E4 on FLUX.1-dev | ~13 h |
| 8 | `08_multidevice.ipynb` | Kodak 5 devices × 2 seeds, LODO κ, mixed model | ~9 h |
| 9 | `09_lowmid_representation.ipynb` | second device representation | ~1.6 h |
| 10 | `10_full_finetune.ipynb` | full fine-tuning arm | ~5 h |
| 11 | `11_amplitude_ceiling.ipynb` | H0 only; H1–H4 are **not** required (see below) | ~10 min |
| 12 | `12_daxing.ipynb` | second dataset. **Run stage D0 alone first** and check the device-code printout | ~10 h |
| 90 | `90_consolidate.ipynb` | recomputes and audits everything | ~10 min |

Each notebook has a `C.STAGES` tuple. Set it to a single stage to run one step; all stages are
resumable and skip completed work, so an interrupted session can simply be rerun.

**`11_amplitude_ceiling.ipynb`: run stage H0 only.** H0 establishes that effective amplitude
saturates near 4× at any nominal amplitude, which is the result. H1–H4 would train adapters at
3.96× effective when a null at 3.48× already exists.

### Protocol hashes
Every experiment hashes its configuration and binds artifacts to that hash. Changing any config
field invalidates dependent results rather than silently reusing them. `TRAIN_SEEDS`, `N_T`,
`ALPHAS`, `STEPS` and the arm registry are all inside the hash — change one and every adapter in
that root must be retrained. Use a fresh `OUT_DIR` for a variant.

---

## What is not released
Source images, and estimated device fingerprints as arrays. Fingerprints are released as derived
statistics only, since the arrays would permit device-level identification of third-party images.
Generated-image collections (roughly 25,000 files) are available on request.

## Layout produced on Drive
```
inv_channel/
├── E_INV_P0_v3/     base study: manifest/ fingerprints/ adapters/ gens/ csv/ surfaces/ figs/
├── E_SEEDEXT/       seeds 3-5
├── E_AMP/           amplitude sweep, controls, E-POST
├── E_TRACKB/        VAE screen
├── E_TRACKB2_FLUX/  E4
├── E_MULTIDEV/      Kodak
├── E_LOWFREQ/       second representation
├── E_AMPHI/         amplitude ceiling
└── E_FULLFT/        full fine-tuning
```
