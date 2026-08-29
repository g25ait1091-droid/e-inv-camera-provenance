# Data

Everything measured by this study lives in one Drive folder:

**https://drive.google.com/drive/folders/1yHgioOdyRhEGl83dRAOgIaJhirbSfkIm**

Code is here in git; measurements are there. The split is deliberate — the per-row CSVs run
to hundreds of megabytes and the generated-image collections to tens of gigabytes, neither
of which belongs in a repository.

## What is in the Drive folder

| root | holds | needed for |
|---|---|---|
| `E_INV_P0_v3/` | base study: manifests, fingerprints, adapters, 7500 generations, per-row CSVs | everything |
| `E_SEEDEXT/` | seeds 3–5, taking the primary bound from n=3 to n=6 | the k=6 analysis |
| `E_SEEDEXT2/` | **seeds 6–11, the declared extension to twelve adapters per arm** | **the headline limit** — `verify_einv.py` needs `E_SEEDEXT2/csv/sx2_measure.csv` |
| `E_AMP/` | amplitude sweep, Gaussian and shifted controls, detector calibration | Sections V-D, V-E |
| `E_TRACKB/` | five-autoencoder retention screen | Table I |
| `E_TRACKB2_FLUX/` | FLUX.1-dev replication | Section V-F |
| `E_MULTIDEV/` | Kodak M1063, five bodies × two seeds | Sections V-H, V-I |
| `E_LOWFREQ/` | low/mid DCT representation | Section V-J |
| `E_AMPHI/` | amplitude ceiling | Section V-D |
| `E_FULLFT/` | full fine-tuning of all 2.24 B parameters | Section V-G |
| `E_DAXING/` | Huawei P20 smartphone replication, five bodies | notebook 13 |
| `E_FLOATCAL/` | captured decoder pre-quantization floats | notebook 14 |

Within each root: `csv/` per-row measurements, `fingerprints/` estimates and gate JSONs,
`manifest/` split definitions, `meta/` protocol hashes, `adapters/` LoRA weights,
`gens/` generated images.

## What is deliberately not released

**Source photographs.** The Dresden Image Database is available from its original
distributors. The device selection and split definitions needed to reproduce the manifests
exactly are in `docs/E_INV_RESULTS_v2.md` §1.

**Fingerprint arrays for third-party use.** Estimated device fingerprints are released as
derived statistics within each experiment root, not as standalone arrays, because a
released PRNU array permits device-level identification of images the study never touched.

## Three recorded supersessions — read this before trusting any cached JSON

`config/results_corrections.json` records all three with derivations.

1. **The primary bound moved from k = 6 to k = 12.** Any file, figure or clone stating
   λ_U = 0.32 % or τ_U = 0.91 % predates the declared extension. The current values are
   0.1507 % and 0.4117 %, produced by `notebooks/12_seed_ext2.ipynb`.
2. `E_LOWFREQ_results.json` stores a per-arm bound of 20.56 % where the symmetric statistic
   gives 9.72 %.
3. `XVAL_additive.json` stores a permutation p of 0.156 from a test that could not reject;
   the valid test gives 0.0167.

The manuscript carries the corrected value in every case. `verify_einv.py` and
`notebooks/00_verify.ipynb` recompute from the per-row CSVs, so they agree with the
manuscript and disagree with those archived files — which is intended, and stated in their
output.
