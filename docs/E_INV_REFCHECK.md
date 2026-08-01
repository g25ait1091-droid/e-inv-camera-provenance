# E-INV — REFERENCE VERIFICATION LOG
**Started 31 July 2026.** Independent check of the prior-art sweep, reference by reference.
Status: **4 of ~23 verified in depth.** Two discrepancies found already.

## What "verified" means here
- **Metadata** — title, authors, venue, year, pages, DOI, cross-checked against ≥2 independent sources.
- **Claim** — the specific statement attributed to the work confirmed against its own text.
- Page numbers exist only where the venue assigns them. **arXiv preprints have none**; CVF
  open-access and IEEE Xplore sometimes **disagree** for the same paper (see R1).

---

## VERIFIED

### R1 · Yu, Skripniuk, Abdelnabi, Fritz — Artificial Fingerprinting for Generative Models
**⚠ PAGE DISCREPANCY — two ranges in circulation for the same paper.**

| source | pages |
|---|---|
| CVF Open Access (official proceedings HTML + BibTeX) | **14448–14457** |
| MPI-INF institutional record + DOI 10.1109/ICCV48922.2021.01418 | **14428–14437** |

Both are for ICCV 2021, same four authors, ICCV 2021 **Oral**. This is the CVF-vs-IEEE-Xplore
pagination split. **Decide one convention and apply it to every CVF paper in the bibliography.**
For IEEE venues (TIFS), cite the **IEEE Xplore** pagination with the DOI.
- Authors confirmed: Ning Yu, Vladislav Skripniuk, Sahar Abdelnabi, Mario Fritz. *(Note: some
  reference lists give "Yu, Skripniuk, Chen, Davis, Fritz" — that is a **different** paper,
  "Responsible Disclosure of Generative Models Using Scalable Fingerprinting", ICLR 2022.
  Do not conflate them.)*
- **Claim confirmed** from the abstract: they "embed artificial fingerprints into training data,
  then validate a surprising discovery on the transferability of such fingerprints from training
  data to generative models." Supports the sweep's "prior art for the general channel."

### R2 · Asnani, Collomosse, Bui, Liu, Agarwal — ProMark (CVPR 2024) ✅ CLEAN
- **pp. 10802–10811**, DOI **10.1109/CVPR52733.2024.01027**, arXiv 2403.09914.
- Four independent sources agree (CVF BibTeX, two arXiv reference lists, one with the DOI).
- **Claim confirmed**: concept information "proactively embedded into the input training images
  using imperceptible watermarks, and the diffusion models … trained to retain the corresponding
  watermarks in generated images." The sweep's ProMark/E-INV comparison table is accurate.

### R3 · Liu, Shuai, Fan, Dong, Hu, Ba, Ren — CoprGuard (CVPR 2025) ✅ CLEAN, one citation note
- **pp. 18653–18662**, arXiv 2503.11071, IEEE Xplore document 11093455.
- **Title is "Harnessing Frequency Spectrum Insights for Image Copyright Protection Against
  Diffusion Models."** *CoprGuard is the framework name, not the title* — the bibliography entry
  must use the title. Citing "CoprGuard (CVPR 2025)" in prose is fine.
- **Claim confirmed verbatim**: "we present novel evidence that diffusion-generated images
  faithfully preserve the statistical properties of their training data, particularly reflected in
  their spectral features" and robustness "even when watermarked images comprise a mere 1% of the
  training dataset."

### R4 · US12283072B2 — Zero-vision camera system ✅ VERIFIED FROM THE GRANTED CLAIMS
- Assignee **Siliconesignal Technologies**, inventor **Khalid Saghiri**. Application 18/604,683,
  filed 2024-03-14, priority 2023-09-16, **granted 2025-04-22**. 20 claims. Continuation of
  18/468,706 (US11961263B1). Family also MA67572B1.
- **The sweep's distinction is correct, and there is a stronger one it missed:**

| | US12283072B2 | E-INV |
|---|---|---|
| fingerprint type | **FPN-DSNU** — *dark-signal* non-uniformity, from covered-sensor dark frames (claims 2–5) | **PRNU** — *photo-response* non-uniformity, from illuminated images. **Different physical quantity** |
| how it enters the network | a dedicated **"fingerprint injection layer" with weights fixed from the fingerprint** (claim 1 + FIG. 3, element 312) | nothing injected; measured passively |
| the autoencoder | purpose-trained per camera, encoder on-device, decoder remote | off-the-shelf pretrained latent-diffusion VAE |
| purpose | **encryption / privacy** — never produce a visualisable stream; camera authentication via PUF | provenance measurement |
| generative model | none | text-to-image personalisation and generation |

- **Use this wording:** the patent *writes* a dark-frame FPN-DSNU pattern into a purpose-built
  camera autoencoder as fixed layer weights, for encryption and device authentication. It does not
  measure whether a PRNU fingerprint present in ordinary photographs passively survives a
  pretrained generative autoencoder.
- Incidental find: the patent's non-patent citations include **Berdich et al., "Smartphone Camera
  Identification from Low-Mid Frequency DCT Coefficients of Dark Images", Entropy 2022, 24, 1158**
  — independent corroboration that low/mid-frequency device ID is an active line, which is the
  *Beyond PRNU* wording constraint arriving from a second direction.

---

## NOT YET VERIFIED (19)

**Tier 1 — verify before drafting** (they bound the novelty claim or are patents)
SIREN (IEEE S&P 2025) — the sweep cites only the *program page*, which is weak; find the paper
record · DiffusionShield (arXiv 2306.04642) · Beyond PRNU (arXiv 2111.02144) · US20250307974A1 ·
US20250104399A1 · US20250225802A1

**Tier 2 — foundational, low risk but must be exact**
Lukáš, Fridrich & Goljan 2006 · Chen, Fridrich, Goljan & Lukáš 2008 · Dresden (Gloe & Böhme,
J. Digital Forensic Practice 3(2–4), 2010)

**Tier 3 — adjacent**
Marra et al. (GAN fingerprints) · Yu, Davis & Fritz ICCV 2019 pp. 7555–7565 (already seen in a
reference list) · Latent Fingerprints (PMLR v202) · ManiFPT · Whodunit · SpoC (CVPRW 2021) ·
PRNU-Bench (arXiv 2509.17581) · Carlini et al. USENIX Sec 2023 · CDI (CVPR 2025) · FineXtract ·
medical diffusion fingerprints (PMLR v301)

---

## Protocol for the remainder
1. Locate the **publisher record**, not a blog, aggregator or program page.
2. Record title, full author list, venue, year, pages, DOI.
3. Cross-check pages against ≥2 sources; **flag CVF-vs-Xplore splits explicitly**.
4. Confirm the **specific claim** attributed to the work against its own abstract or text.
5. For patents: read the **granted claims**, not the abstract — the abstract is not the scope.
6. Pick one pagination convention for CVF papers and apply it uniformly.

## Standing caution
Every URL in the sweep carries `utm_source=chatgpt.com`. An earlier survey in this project attached
a CNN-position claim to the DiT paper, which does not support it. Two of the first four checks here
turned up an issue (a page discrepancy and a title/framework-name conflation). **Assume nothing is
correct until checked against the publisher record.**
