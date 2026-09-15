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

## The photographs used by the study

**https://drive.google.com/drive/folders/1wpsN17d3ggdVES40CR6Sk97d9tm2tc7X** (`E-INV_paper_datasets`)

This folder holds the photographs the study used and nothing else, copied from two public datasets.
All credit belongs to their creators; please cite them and follow their terms. The folder's own
README lists the licences and a SHA-256 manifest.

| dataset | devices (as in the filenames) | images | role |
|---|---|--:|---|
| Dresden Image Database | `Nikon_D200_1` (A), `Nikon_D200_0` (B) | 380 + 372 | primary pair |
| Dresden Image Database | `Kodak_M1063_0` … `_4` | 2,391 | five-body device group |
| Dresden Image Database | `Agfa_DC-733s_0`, `Nikon_D70_1`, `Agfa_DC-830i_0` | 281, 189, 363 | cross-model references |
| Daxing Smartphone Identification Dataset | Huawei P20 1101–1105, orientation `90` only | 244–300 each | five-body smartphone group |

Download it, then `export EINV_MYDRIVE=/path/to/E-INV_paper_datasets`; the layout matches the paths
the code expects.

- **Dresden:** T. Gloe and R. Böhme, "The 'Dresden Image Database' for benchmarking digital image
  forensics," *Proc. ACM SAC*, 2010, pp. 1584–1590, doi:10.1145/1774088.1774427. Distributed by
  TU Dresden for research; its original site was unreachable in September 2026, so use the images for
  non-commercial research and cite the paper.
- **Daxing:** H. Tian *et al.*, "Daxing Smartphone Identification Dataset," *IEEE Access*, vol. 7,
  pp. 101046–101053, 2019, doi:10.1109/ACCESS.2019.2928356. Official repository
  https://github.com/xyhcn/Daxing (shared for research under GPL-3.0).

The device selection and split definitions needed to rebuild the manifests exactly are in
`docs/E_INV_RESULTS_v2.md` §1.

## v2 measurements

The small v2 result files are in this repository under `v2/workspace/out/`. The v2 generated images and
adapter weights are not yet in Drive; they are available from the author on request.

## What is deliberately not released

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
