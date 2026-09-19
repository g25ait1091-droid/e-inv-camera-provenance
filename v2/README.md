# E-INV v2 — extension experiments

These experiments extend the primary study (the notebooks at the repository root) on one local GPU. Each
one was entered in [`PRESPECIFICATION_LOG.md`](PRESPECIFICATION_LOG.md) — with the readings that would
count as positive or negative — before its first result was read. Entries that later turned out wrong are
kept, and listed in the log's register of withdrawn statements.

## 1. Check the numbers (seconds, no GPU, nothing to download)

```bash
python verify_v2.py            # from the repository root
```

It recomputes the v2 headline quantities from per-adapter and per-band values in `workspace/out/` and from
`analysis/FINAL_LEDGER.json`, compares each with the reported value, and exits non-zero on any disagreement.

## 2. Set up

```bash
pip install -r v2/requirements.txt
export EINV_V2=$PWD/v2/workspace            # outputs, shipped results (default)
export EINV_MYDRIVE=/path/to/E-INV_paper_datasets   # the photographs; see DATA.md
export EINV_EXT=$PWD/v2/ext                 # third-party code, below
```

Every script reads its paths from [`src/einv_paths.py`](src/einv_paths.py), so these three variables are
the only machine-specific settings.

**Third-party code** (clone into `$EINV_EXT`; the commits used are pinned):

| folder | repository | commit | used by |
|---|---|---|---|
| `DiffusionShield/` | https://github.com/Yingqiancui/DiffusionShield | `f9cc205` | the published-watermark arm (`t1_ladder.py`, `t1_wm_measure.py`) |
| `prnu-python/` | https://github.com/polimi-ispl/prnu-python | `91e1585` | the PCE detector (`t2_panel*.py`, `t2_poscontrol.py`) |
| `noiseprint/` | https://github.com/grip-unina/noiseprint | `c06034e` | Noiseprint (`t2_noiseprint.py` via `a4_noiseprint.py`; needs TensorFlow 1.15 in a separate Python 3.7 environment) |

Models are downloaded by `diffusers` / `transformers` on first use: Stable Diffusion 3.5 Medium (accept
its licence on Hugging Face and log in), and `facebook/dinov2-base` for the near-copy measure.

Hardware used: one NVIDIA L40S (44.7 GB). A rank-16 adapter takes about 45 min at 2000 steps, and 500
generations at 1024 × 1024 about 70 min.

## 3. What each script does

| experiment | scripts | writes (under `workspace/out/`) |
|---|---|---|
| fingerprints and splits of the primary pair | `fingerprints.py` | `fp/manifest.json`, fingerprint estimates (kept private) |
| designed-pattern ladder, published watermark, unmarked and control arms | `t1_ladder.py` (select arms with `T1_ARMSET=`), `t1_measure.py`, `t1_wm_measure.py`, `t1_derived.py` | `t1/summary*.json`, `t1/measure_rows*.csv`, `t1/wm_*` |
| frequency response, one octave at a time | `t1_band.py` (`fields`, `materialise`, `vae`, `measure`), `t1_band2.py`, `band_derived.py`, `a1_derived.py` | `t1/band_*.json`, `t1/band2_*` |
| periodic tiles and grid alignment | `t1_periodic.py`, `t1_periodic2.py` | `t1/periodic*.json` |
| second training environment, random crops, B-body arms | `f4_*.py`, `f5_localgen.py`, `f7_stats.py` | `t1/f4_*`, `t1/f7_stats.json`, `t1/e3_stats.json` |
| adaptation dose and memorization | `dose_stats.py`, `f8_memorization.py`, `dino_memorization.py` | `t1/dose_stats.json`, `t1/f8_memorization.json`, `t1/dino_memorization.json` |
| detector panel (PCE, low/mid, NCC) | `t2_panel.py`, `t2_panel_par.py`, `t2_poscontrol.py` | `t2_summary.json`, `t2_rows.csv`, `t2_poscontrol.json` |
| learned detector and its template-projected version | `t2_learned*.py`, `t2_noprnu_derived.py` | `t2_learned*.json`, `*_net.pt` |
| Noiseprint | `t2_noiseprint.py`, `a4_noiseprint.py` | `t2_noiseprint_*.json` |
| examiner: power and closed-set attribution | `t3_power.py`, `t3_power_v4.py`, `t3_attrib.py`, `t3_repro.py` | `t3_*.json` |
| five-caption prompt bank | `t5_prompt.py`, `t5_derived.py` | `t5/summary*.json`, `t5/rows.csv` |
| firearm content of the generations | `t6_weapon_clip.py` | `t6_weapon_clip.json`, `t6_weapon_validation_labels.json` |
| secondary limits, channel prediction, exact p-values | `v4_offline.py` | `v4_offline.json` |
| figures | `fig8_ladder.py`, `fig9_bands.py`, `fig10_examples.py`, `make_figures_v2.py`, `fig_pipeline_v2.py` | `workspace/paper/*.pdf` |
| fetching the primary study's archived generations | `drive_fetch.py`, `drive_fetch_any.py`, `fetch_ext.sh` | `workspace/data/` |

The `orchestrate*.sh` files are the exact run chains used, in order; each names the log entry it serves.

## 4. What is shipped and what is not

Shipped in `workspace/out/`: every result JSON — 86 files, at the top level and under `t1/`, `t5/` and
the `fp*/` directories — together with the per-image measurement tables (CSV), per-band and per-tile
rows (NPZ), the learned detectors' weights, and derived fingerprint statistics (JSON).

Not shipped: generated images and adapter weights (tens of GB; see [`../DATA.md`](../DATA.md)), the
designed-pattern fields (regenerated from their seeds by the `fields` stages), and fingerprint arrays,
which would allow device-level identification of third-party photographs. The JSON and NPZ files that
belong to those are left out with them, and are the only ones that are: each adapter's
`t1/adapters/*/train_meta.json`, `t1/fields/field_stats.json`, `t1/train_png/materialise.json`, and the
held-out residual stacks `fp/H_{A,B}.npz`. Smoke-test and reference-audit files (`*_smoke.*`,
`refaudit.json`) are also skipped; nothing in the paper rests on them.
