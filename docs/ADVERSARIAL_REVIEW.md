# E-INV — adversarial review and consistency audit
**1 August 2026.** Written against the compiled manuscript, the frozen ledger, and the
result files read directly from Drive.

---

## Part 1 — Number verification against the stored records

Every headline quantity was checked against the JSON written by the notebook that produced it,
not against any intermediate document.

| Quantity | Manuscript | Stored record | Source file | Status |
|---|---|---|---|---|
| $R_\mathrm{real}$ | 3.567e-02 | 0.0356703416571125 | `X_reanalysis.json` | match |
| $R_\mathrm{natural}(B)$ | 3.140e-02 | 0.031404711308475 | `X_reanalysis.json` (`R_real.B_arm`) | match |
| $R_\mathrm{VAE}$ | 1.306e-02 | 0.0130592770366 | `X_reanalysis.json` | match |
| $\eta$ | 0.3661 | 0.36611023135507426 | `X_reanalysis.json` | match |
| $\kappa_\mathrm{model}$ | 0.0073 | 0.007292465306818485 | `s1_gates.json` | match |
| Pooled AUC | 0.9983 | 0.99835 | `s1_gates.json` | match |
| Same-model AUC | 1.0000 | 1.0 | `s1_gates.json` | match |
| Device / noise variance share | 0.355 / 0.638 | 0.3549086 / 0.6377989 | `s1_gates.json` | match |
| $\hat\rho_\mathrm{real}$ A / B | 0.0381 / 0.0324 | 0.03806252 / 0.032412825 | `s1_gates.json` | match |
| $U_\mathrm{device}$ | 8.8802e-05 | 8.880180005344921e-05 | `seedext_pooled.json` | match |
| $\hat\theta_\mathrm{sym}$ | 1.494e-05 | 1.4937508457778334e-05 | `seedext_pooled.json` | match |
| $\hat\lambda$ | 0.0419 % | 0.0004187654999598207 | `seedext_pooled.json`, `A1_bootstrap.json` | match |
| arm A / B sd | 4.453e-05 / 3.332e-05 | se×√6 = 4.4535e-05 / 3.3324e-05 | `seedext_pooled.json` | match |
| $t_{0.995,5}$ | 4.03 | 4.032142983557536 | `seedext_pooled.json` | match |
| $\lambda_U$ plug-in | 0.2490 % | 0.0024895135826584537 | `A1_bootstrap.json` | match |
| $\lambda_U$ BCa | 0.3236 % | 0.003236346253844063 | `A1_bootstrap.json` | match |
| $\tau_U$ plug-in | 0.6800 % | 0.006799901694754833 | `A1_bootstrap.json` | match |
| $\lambda_U$ raw pct99 (not reported) | — | 0.00847384503471565 | `A1_bootstrap.json` | correctly withheld |
| mod-8 range | 0.042–0.070 | min 0.042, max 0.070 across 15 cells | `X_reanalysis.json` `x1` | match |
| A-gens $\to K_A$ / $K_B$ | 7.472e-05 / 7.312e-05 | 7.47187e-05 / 7.31192e-05 | `X_reanalysis.json` `x2` | match |
| $n=3$ comparison figure | 0.699 % | 0.006992659222015797 | `X_reanalysis.json` | match |
| Low/mid $\eta$ | 0.5544 | 0.5543732677169075 | `E_LOWFREQ_results.json` | match |
| Low/mid arm $\theta$, sd | −1.110e-04 / +9.032e-05 | −1.1102759e-04 / +9.0323214e-05 | `E_LOWFREQ_results.json` | match |
| Low/mid $R_\mathrm{real}$ lower-99 | 1.50106e-03 | 0.0015010588516227571 | `E_LOWFREQ_results.json` | match |
| Cross-seed $R^2_\mathrm{cv}$ | 0.870 | 0.8697369920001163 | `XVAL_additive.json` | match |
| Off-diagonal $R^2$ | 0.775 | 0.7754329979273701 | `XVAL_additive.json` | match |
| Own-device residual, $t$, CI | 1.265e-05, 0.69, [−3.23e-05, +5.87e-05] | 1.26494544741798e-05, 0.6861130087900937, same | `XVAL_additive.json` | match |

**No discrepancy was found between the manuscript and the stored records.**

### Two records that disagree with the manuscript, and why

Both are cases where a *later, corrected* computation superseded a value already written to disk.
The manuscript carries the corrected value; the archived JSON does not. Left unresolved, a reader
checking the repository would find an apparent contradiction.

**(a) `E_LOWFREQ_results.json` stores `lambda_U = 0.20564658674097028` (20.56 %).**
The manuscript reports **9.72 %**. The stored value is `max(U_A, U_B)/R_lower99`, a per-arm bound.
In this representation the per-arm statistic is dominated by the additive fingerprint main effect,
which is antisymmetric and cancels only in the symmetric statistic. Recomputing from the stored
per-seed means gives $\theta_\mathrm{sym} = -1.03522\times10^{-5}$, $t = -0.267$,
$U = 1.4593\times10^{-4}$, $\lambda_U = 9.72\,\%$ — the manuscript value.
**Action:** `results_corrections.json` records both, with the derivation.

**(b) `XVAL_additive.json` stores `p_perm = 0.15596880623875226`.**
The manuscript reports **0.0167**. The stored value came from a permutation that relabelled the
correlation matrix columns and then recomputed *both* the fitted main effects and the observed
contrasts from the relabelled matrix. That is a consistent relabelling: it does not break the
correspondence being tested, so it cannot reject. The corrected test permutes which prediction is
paired with which observation, over all $5! = 120$ device assignments, giving $p = 0.0167$ (two
assignments perform as well or better; floor $1/120 = 0.0083$). The device-level statistics
($r = +0.979$, $p = 0.0036$) were also computed after the file was written and are absent from it.
**Action:** `results_corrections.json` records the corrected values and marks `p_perm` superseded.

---

## Part 2 — Adversarial review

Ordered by how much damage each objection could do.

### A1. "You have an unexplained artifact inside your measurement pipeline." — was unanswered
A shifted fingerprint ranks first in arms that never received it, and you cannot say why. A
reviewer will ask how the main result is protected from whatever mechanism produces it.

**Assessment: the strongest available attack, and the manuscript did not answer it.**
The answer is structural. The elevation is uniform across arms, so in the decomposition it is a
fingerprint main effect $b_y$ with no interaction component, and the symmetric paired statistic
cancels $b_y$ exactly. A main effect of any magnitude from any mechanism cannot manufacture a
spurious device interaction. **Fixed:** Section V-K now states this and links it to the same
property that neutralises the generic offset.

### A2. "The amplification arm cannot rule out transfer at high amplitude."
Effective amplitude saturates near $4\times$ regardless of nominal amplitude, so the experiment is
structurally silent above that point.

**Assessment: valid, and previously implicit.** **Fixed:** Section V-D now states explicitly that
the arm is silent above approximately $4\times$ effective and makes no claim there.

### A3. "$\eta = 0.3661$ carries no uncertainty."
Four significant figures from 40 images per device.

**Assessment: valid.** **Fixed:** Section V-A now reports the per-arm contrasts that bracket the
pooled value and states that four figures are for traceability, not precision.

### A4. "Did the adapters produce usable images at all?"
The manuscript reported no image-level evidence that adaptation took effect.

**Assessment: valid and easily answered.** **Fixed:** Section IV-D now reports the 31–59 gray-level
mean absolute difference against base-model generations at matched seeds.

### A5. "Six clusters is too few for a $t$-based limit, and normality is untested."
**Assessment: partly valid, already mitigated.** Three constructions are reported and the largest
is taken; the BCa bootstrap is distribution-free in the relevant sense and lands near the analytic
value. The manuscript already declines to report the raw percentile because six clusters give a
coarse tail. *No change; the objection is anticipated in Section IV-F.*

### A6. "A single caption makes the study unrepresentative of real personalisation."
**Assessment: valid as a scope limit, not as an error.** A fixed caption is what removes semantic
confounding between arms. Section VII lists prompt protocol among the conditioning factors.
*Consider one sentence in Section IV-D on the trade-off. Not blocking.*

### A7. "The low/mid bound of 9.72 % is so loose it establishes nothing."
**Assessment: rhetorically effective, substantively wrong, and the manuscript handles it.** The
looseness is a 16.8-fold weaker signal estimated at 0.081 split-half reliability, both reported.
The contribution of that section is not the bound but the demonstration that a second, qualitatively
different representation yields a null interaction while carrying a nine-fold larger additive term.
*No change.*

### A8. "Kodak ICC = 0.943 with 5/5 sign agreement looks like a strong device effect."
**Assessment: this is the trap the paper is built to expose, and it does.** Mixed signs, null grand
mean, sign-flip $p = 0.434$. *No change.*

### A9. "The full fine-tuning row of Table IV is empty."
**Assessment: valid and outstanding.** Either complete it or remove the row before submission; an
empty row invites the question of whether the experiment failed.

### A10. "Sixteen `REF-VERIFY` markers."
**Assessment: valid and blocking for submission.** Every DOI, venue and page range is verified;
the markers are author lists. They must be resolved from the papers themselves.

---

## Part 3 — Formatting audit

| Item | Status |
|---|---|
| Compiles under the genuine `ieeeaccess.cls` | yes — 13 pages |
| Undefined references or citations | none |
| Balanced environments | yes (4 tables, 14 equations, 2 biographies) |
| Abstract length | 242 words (target 220–260) |
| Abstract free of citations, equations, undefined acronyms | yes; PRNU and AUC defined at first use |
| Index terms | 9 (limit 3–10) |
| Page count | 13 before figures; 20-page guidance leaves room for the 7 planned |
| Prohibited wording | none present except in explicit negations |
| Author biographies present | yes, as `TODO-AUTHOR` stubs |
| AI-use disclosure in Acknowledgment | present |
| Consistent American spelling | yes |

**Two class-compatibility shims** are present and guarded, needed only because the TeX Live
`IEEEtran` differs from the one bundled with the IEEE author kit: a `biography` counter and the
`@biographyTOCentrynotmade` conditional. Both are no-ops with the official kit. One residual
non-fatal error comes from the class's own PANTONE `spotcolor` branch, which requires a package
absent from TeX Live; it affects only the branding colour.

**Branding images** (`logo.png`, `notaglinelogo.png`, `bullet.png`) are placeholders. Replace them
with the official files from the author kit.

---

## Part 4 — Remaining blockers

1. Resolve 16 `REF-VERIFY` author lists and 8 `TODO-AUTHOR` items.
2. Complete or remove the full fine-tuning row of Table IV.
3. Read the granted claims of the three unverified patents.
4. Add the seven figures.
5. Replace the branding placeholders and obtain `ieeeaccess.cls` from the author kit.
6. Decide one pagination convention for CVF papers; the bibliography currently uses IEEE Xplore
   pagination with DOIs throughout.
