# E-INV — does camera sensor provenance survive diffusion personalization?

Code and measurement records for a study of whether naturally occurring, PRNU-derived
identity from an **individual physical camera body** is passively inherited when a
text-to-image model is personalized on that camera's photographs, and whether it is
re-emitted in images generated from text alone.

**Data:** https://drive.google.com/drive/folders/1yHgioOdyRhEGl83dRAOgIaJhirbSfkIm
— see [DATA.md](DATA.md). Code lives here, measurements live there.

---

## The result in one table

| stage | finding |
|---|---|
| Autoencoder round trip | Device contrast **survives**: η = 0.3661, AUC 1.000 → 0.9814 |
| Optimization objective | Responds to fixed-pattern energy from α ≈ 3, but **not device- or alignment-specific** |
| Generation | Transfer bounded at **λ_U ≤ 0.32 %** of real-image contrast, **τ_U ≤ 0.91 %** of what survives reconstruction |

The bound reproduces on a second generative system (FLUX.1-dev, 0.777 % at matched n),
under full fine-tuning of all 2.24 B transformer parameters (0.747 %), across five
same-model bodies (1.74 %, device-generalized), and in a second frequency representation
of device identity (9.72 %).

**These are different generalization units and are not interchangeable.** 0.32 % generalizes
over adapter training seeds for two devices; 1.74 % generalizes over five physical bodies.

---

## Start here

```bash
git clone https://github.com/g25ait1091-droid/e-inv-camera-provenance
cd e-inv-camera-provenance
python verify_einv.py --root /path/to/downloaded/inv_channel
```

That recomputes **36 headline quantities from the per-row CSVs** and compares each against
the manuscript. It never reads the cached result JSONs — those are what it is checking. On
the released data it reports 36 verified, 0 disagreements. Roughly two minutes.

`notebooks/00_verify.ipynb` is the same thing for Colab, with Drive mounting.

---

## Repository layout

```
notebooks/     01–14, the full pipeline in run order
cells/         drop-in stage cells for notebooks that were built incrementally
analysis/      coverage simulation, figure generation, figure collision checker
config/        frozen numeric ledger and the corrections file
docs/          results of record, plan, run order, reference check, audit responses
verify_einv.py command-line verifier
DATA.md        what is in Drive and what is deliberately not released
REPRODUCE.md   three reproduction tiers, from 2 minutes to 60 GPU-hours
```

### Notebooks

| # | notebook | produces | wall |
|---|---|---|---|
| 00 | `00_verify.ipynb` | independent recomputation of every headline number | 3 min |
| 01 | `01_pilot.ipynb` | manifests, fingerprints, 14 adapters, 7500 generations, the C0 verdict | 14 h |
| 02 | `02_seed_ext.ipynb` | seeds 3–5; takes the bound from n=3 to n=6 | 5 h |
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

Notebooks 13 and 14 are extensions beyond the published study; see
`docs/REVIEW_RESPONSE.md`.

---

## Things worth knowing before you run anything

**Stages are resumable and gated.** Each notebook exposes `C.STAGES`; set it to one stage
to run one step. Every stage skips completed work. Gates halt rather than produce a number
you cannot trust — a fingerprint-quality gate caught an inverted spectral filter during
development (κ was 0.431 instead of 0.0073), and both values are reported in the paper
because the gate working is itself evidence.

**Protocol hashes bind configuration to artifacts.** Changing any config field inside the
hash invalidates dependent results rather than silently reusing them. `TRAIN_SEEDS`, `N_T`,
`ALPHAS`, `STEPS` and the arm registry are all inside it — change one and every adapter in
that root must be retrained. Use a fresh `OUT_DIR` for a variant.

**Three environment traps, all patched in the notebooks.**
`peft`'s `is_torchao_available()` *raises* rather than returning `False` when `torchao` is
older than 0.16 (Colab ships 0.10.0), which kills `add_adapter`. PIL infers format from the
file extension, so temporary names ending `.tmp` need `format="PNG"` — and `np.save` appends
`.npy` for the same reason, so atomic temp writes must pass a file handle. **TF32 must stay
disabled on the measurement path.**

**Do not run** stages H1–H4 of notebook 11, the better-injection variant, Kodak low/mid,
E-PROMPT, E2, E3, or the superseded `E_FULLFT.ipynb`. Each was declined with a reason
recorded in `docs/E_INV_V1_PLAN_v13.md`.

---

## Analysis scripts

`analysis/coverage_sim.py` — simulates the frequentist coverage of the three upper-limit
constructions at k = 6 clusters, under normal, heavy-tailed and empirically resampled
cluster distributions, with the denominator resampled. Plug-in *t* covers at 99.9 %
worst-case; BCa alone reaches only 98.2 % under heavy tails; the reported maximum inherits
the coverage of its best member. CPU, two minutes.

`analysis/make_figures.py` — regenerates all seven figures from values transcribed out of
the archived result JSONs. No figure is drawn from data that does not appear in the paper.

`analysis/overlapcheck.py` — computes text, legend and arrow-path bounding boxes in display
coordinates and reports collisions with plotted data. Used by `make_figures.py`.

---

## Citation

Manuscript under review. Cite this repository as
`https://github.com/g25ait1091-droid/e-inv-camera-provenance` until publication.

Released under the MIT License. The Dresden Image Database and the Daxing Smartphone
Identification Dataset carry their own terms; neither is redistributed here.
