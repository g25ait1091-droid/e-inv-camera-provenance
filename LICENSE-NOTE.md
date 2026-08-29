# Licensing and what is not redistributed

**Code** in this repository is MIT-licensed — see [LICENSE](LICENSE).

**Source photographs are not redistributed.** The Dresden Image Database and the Daxing
Smartphone Identification Dataset each carry their own terms and must be obtained from their
original distributors. The device selection, split sizes and guard bands needed to reproduce
the manifests exactly are in [`docs/E_INV_RESULTS_v2.md`](docs/E_INV_RESULTS_v2.md) §1, and
`notebooks/01_pilot.ipynb` stage S0 rebuilds them deterministically from the image directory.

**Estimated device fingerprints are released as derived statistics only**, within each
experiment root — never as standalone arrays. A released PRNU array would permit
device-level identification of images this study never touched.
