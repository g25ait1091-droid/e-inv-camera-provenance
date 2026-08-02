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
| `E_SEEDEXT/` | seeds 3–5, taking the primary bound from n=3 to n=6 | the headline limit |
| `E_AMP/` | amplitude sweep, Gaussian and shifted controls, detector calibration | Sections V-D, V-E |
| `E_TRACKB/` | five-autoencoder retention screen | Table I |
| `E_TRACKB2_FLUX/` | FLUX.1-dev replication | Section V-F |
| `E_MULTIDEV/` | Kodak M1063, five bodies × two seeds | Sections V-H, V-I |
| `E_LOWFREQ/` | low/mid DCT representation | Section V-J |
| `E_AMPHI/` | amplitude ceiling | Section V-D |
| `E_FULLFT/` | full fine-tuning of all 2.24 B parameters | Section V-G |

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

## Two archived values that later analyses superseded

`config/results_corrections.json` records them with derivations. **Read it before trusting
any cached JSON**: `E_LOWFREQ_results.json` stores a per-arm bound of 20.56 % where the
symmetric statistic gives 9.72 %, and `XVAL_additive.json` stores a permutation p of 0.156
from a test that could not reject. The manuscript carries the corrected values; the
verification notebook recomputes both correctly.
