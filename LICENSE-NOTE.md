# Licensing, credit, and what is not released

**Code** in this repository is MIT-licensed — see [LICENSE](LICENSE).

**Photographs are not in this repository.** The subset the study used is mirrored on Drive, credited to
its creators (see [DATA.md](DATA.md)):

- **Dresden Image Database** (Gloe and Böhme, ACM SAC 2010). Distributed by TU Dresden for research. Its
  original distribution site was unreachable in September 2026, so its full terms could not be
  re-checked; use the images for non-commercial research, cite the paper, and ask for removal if you are a
  rights holder.
- **Daxing Smartphone Identification Dataset** (Tian *et al.*, IEEE Access 2019). Shared for research
  under GPL-3.0 at https://github.com/xyhcn/Daxing; keep that licence when redistributing and cite the
  paper.

The device selection, split sizes and guard bands needed to rebuild the manifests exactly are in
[`docs/E_INV_RESULTS_v2.md`](docs/E_INV_RESULTS_v2.md) §1, and `notebooks/01_pilot.ipynb` stage S0 rebuilds
them deterministically from the image directory.

**Third-party code** used by the v2 experiments (DiffusionShield, prnu-python, Noiseprint) is not
included; it is cloned from its own repositories under their own licences (see [v2/README.md](v2/README.md)).

**Estimated device fingerprints are released as derived statistics only**, never as arrays. A released
PRNU array would permit device-level identification of photographs this study never touched.
