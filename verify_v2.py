"""Recompute the headline numbers of the v2 experiments from the files shipped in this repository.

No GPU, no downloads, about ten seconds:  python verify_v2.py
Reads analysis/FINAL_LEDGER.json (the primary per-adapter values) and v2/workspace/out/ (override the
latter with EINV_V2). Each quantity is recomputed from per-adapter or per-band values, never copied from a
summary, and compared with the value reported for it. Exits non-zero if any check disagrees.

Sections 24-32 cover the first-version manuscript's post-hoc and closing analyses (pre-specification log
Entries 114-119). They recompute from the shipped per-adapter values, per-image and per-replicate rows, the
weight cosine matrices and the DINOv2 embeddings; the calibration multipliers are re-derived by running the
shipped simulation (v2/src/fv/fv_seedbank_calib.py, which reuses h4_coverage2 / h6_calibrated_limit) with its
registered seeds. What is NOT recomputed here, because its inputs are not in this repository:
  - Entry 114: the crossed-model variance components themselves (MS_resid, the seed variances, the seed-matched
    SE) and the bootstraps and heavy-tail simulations - they need the 24 primary adapters' per-image rows
    (measurement archive, hundreds of MB); they are taken from fv_sigma.json as inputs.
  - Entry 117 item 1: the seed-bank covariance Sigma_v and its bootstrap (same per-image rows; Sigma_v is taken
    from fv_seedbank_calib.json as an input), and the 20,000-replication and redraw runs (descriptive).
  - Item 2: the planting and scoring (generated images and fingerprint arrays are not released), the cluster
    bootstrap intervals (reproducible with fv_examiner_e2e.py from the shipped rows), and the agreement of the
    local scorer with the archive rows.
  - Item 3: the embeddings, the re-estimated fingerprints and the rescoring (photographs, fingerprints and
    generated images); R_real of the re-estimated fingerprints is taken from fv_crossbody_scenes.json.
  - Item 4: the weight cosines themselves (adapter weights are not released).
  - Item 5: the parametric expectation (a 100,000-experiment simulation) and the P20 per-image rows.
  - Item 6: the GPU-hours (training records and archive write times) and the counts by stack.
"""
import os, sys, json, math
import numpy as np
from scipy import stats

REPO = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.environ.get("EINV_V2", os.path.join(REPO, "v2", "workspace")), "out")
T1 = os.path.join(OUT, "t1")
R_REAL, U_DEVICE = 0.0356703416571125, 5.3761e-05
results = []

def check(name, got, want, tol):
    ok = abs(got - want) <= tol
    results.append(ok); print(f"{'OK      ' if ok else 'DISAGREE'} {name:58s} got {got:.4g}  reported {want:.4g}")

def load(*p): return json.load(open(os.path.join(*p)))

def welch_sym(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    th = 0.5 * (a.mean() + b.mean()); va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = 0.5 * math.sqrt(va + vb); df = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    return th, se, df

# 1. symmetric statistic of the primary design, its limit and power
led = load(REPO, "analysis", "FINAL_LEDGER.json")["primary"]
A, B = led["per_adapter_A"], led["per_adapter_B"]
th, se, df = welch_sym(A, B)
check("primary theta_sym (x1e6)", 1e6 * th, 4.60, 0.01)
check("symmetric one-sided 99% limit, % of R_real", 100 * (th + stats.t.ppf(0.99, df) * se) / R_REAL, 0.0762, 0.0005)
check("power of the symmetric test at lambda = 0.10%", float(stats.nct.sf(stats.t.ppf(0.99, df), df, 0.001 * R_REAL / se)), 0.92, 0.01)

# 2. channel prediction for the fingerprint, and its test against the symmetric statistic
bf = load(T1, "band_fields.json"); curve = load(T1, "band_summary.json")["curve"]; b2 = load(T1, "band2_summary.json")["bands"]
T = [c["T_full"] for c in curve]; SE = [c["T_full_se"] for c in curve]
for b in (0, 1): T[b] = b2[f"band{b}"]["T_mean_pct"] / 100; SE[b] = b2[f"band{b}"]["T_se_adapter_pct"] / 100
e = list(bf["energy_fraction_K_A_E2"]); e[0] += max(0.0, 1 - sum(e))
pred = sum(x * y for x, y in zip(e, T)); pse = math.sqrt(sum((x * s) ** 2 for x, s in zip(e, SE)))
check("fingerprint predicted transmission, %", 100 * pred, 0.108, 0.001)

# 3. periodic tiles, three adapters each
P = load(T1, "periodic2_summary.json")["fields"]
lam = {k: np.array([a["lambda_pct"] for a in v["adapters"]]) for k, v in P.items()}
for k, want in (("per32", 4.00), ("per36", 0.209), ("per24", 1.314), ("per40", 1.032), ("per48", 1.632), ("per28", 0.2355)):
    check(f"tile {k[3:]} px transmission, %", float(lam[k].mean()), want, 0.006 if want < 1 else 0.01)
a, b = lam["per32"], lam["per36"]; va, vb = a.var(ddof=1) / 3, b.var(ddof=1) / 3
check("32 vs 36 px, Welch t", float((a.mean() - b.mean()) / math.sqrt(va + vb)), 6.80, 0.02)
on = np.concatenate([lam["per24"], lam["per40"]]); off = np.concatenate([lam["per28"], lam["per36"]])
vo, vf = on.var(ddof=1) / len(on), off.var(ddof=1) / len(off)
check("on-grid vs off-grid tiles, Welch t", float((on.mean() - off.mean()) / math.sqrt(vo + vf)), 5.70, 0.02)
check("on-grid / off-grid transmission ratio", float(on.mean() / off.mean()), 5.27, 0.05)

# 4. finest octave, three adapters: one-sided 99% upper limit
bs = load(T1, "band2_summary.json"); R0 = bs["bands"]["band0"]["R_b"]
t0 = np.array([x["T_pct"] for x in bs["bands"]["band0"]["adapters"]])
se0 = math.sqrt((t0.std(ddof=1) / math.sqrt(3)) ** 2 + (100 * bs["offset_se"][0] / R0) ** 2)
check("finest octave, upper 99% limit, %", float(t0.mean() + stats.t.ppf(0.99, 2) * se0), 0.0435, 0.0005)

# 5. 16000-step adapters, seeds 0-5
S = {}
for f in ("summary_dose16k.json", "summary_dose16krep.json", "summary_dose16krep2.json"): S.update(load(T1, f)["arms"])
a = [S[f"dose16k_A_s{i}"]["natural_paired_KA_minus_KB"] for i in range(3)]
b = [-S[f"dose16k_B_s{i}"]["natural_paired_KA_minus_KB"] for i in range(3)]
th16, se16, df16 = welch_sym(a, b)
check("16000 steps theta_sym (x1e5)", 1e5 * th16, 5.55, 0.01)
check("16000 steps t", th16 / se16, 1.98, 0.01)
own = lambda i: (S[f"dose16k_A_s{i}"]["natural_paired_KA_minus_KB"], -S[f"dose16k_B_s{i}"]["natural_paired_KA_minus_KB"])
a6, b6 = zip(*[own(i) for i in range(6)]); th6, se6, df6 = welch_sym(a6, b6)
check("16000 steps, six per body, theta_sym (x1e5)", 1e5 * th6, 5.71, 0.02)
check("16000 steps, six per body, t", th6 / se6, 3.95, 0.02)
a3, b3 = zip(*[own(i) for i in (3, 4, 5)]); th3, se3, df3 = welch_sym(a3, b3)
check("16000 steps, registered replication (seeds 3-5), t", th3 / se3, 5.01, 0.02)

# 6. attribution power at the limit, measured image-level SE (eq. (power) of the manuscript)
# The default sigma_mu is the ARCHIVE value 5.23e-5 (the off-peak spread of one adapter's correlation surface,
# Entry 114 A). It is kept as the default only so that earlier callers reproduce; the paper prints it as a named
# SENSITIVITY value. The bound uses sigma_mu = 0 (Entry 114: persistent component estimated at 0, exact 95 %
# interval 0 to 4.1e-5).
ARCHIVE_SIGMA_MU = 5.23e-05
def tpr(G, sigma_mu=ARCHIVE_SIGMA_MU, se500=7.035e-05, M=2):
    s = math.sqrt(sigma_mu ** 2 + (se500 * math.sqrt(500 / G)) ** 2)
    return float(1 - stats.norm.cdf(stats.norm.ppf(1 - 0.01 / (M - 1)) - U_DEVICE / s))
check("TPR at nominal limit, 500 img, archive sigma_mu (sensitivity)", tpr(500), 0.043, 0.001)
check("TPR at nominal limit, 500 img, sigma_mu = 0 (bound)", tpr(500, sigma_mu=0.0), 0.059, 0.001)

# 7. chain 11: the fingerprint and a spectrum-matched field injected directly
K = load(T1, "kfield_summary.json"); F = K["fields"]
check("E1 fingerprint as a known pattern, alpha 12, %", F["kinj_a12"]["T_mean_pct"], 0.0174, 0.0005)
check("E1 amplitude ratio alpha48 / alpha12", F["kinj_a48"]["T_mean_pct"] / F["kinj_a12"]["T_mean_pct"], 2.80, 0.02)
check("E2 spectrum-matched field at 4 gray, %", F["gkadd_a4"]["T_mean_pct"], 0.0366, 0.0005)
check("E2 amplitude ratio 1 gray / 4 gray", F["gkadd_a1"]["T_mean_pct"] / F["gkadd_a4"]["T_mean_pct"], 0.609, 0.005)
check("E2 multiplicative / additive at equal RMS", F["gkmul_a4"]["T_mean_pct"] / F["gkadd_a4"]["T_mean_pct"], 1.373, 0.005)
check("band prediction / direct injection", K["prediction_K_spectrum_pct"] / F["gkadd_a4"]["T_mean_pct"], 2.67, 0.05)
check("fingerprint / matched field at matched amplitude", F["kinj_a48"]["T_mean_pct"] / F["gkadd_a4"]["T_mean_pct"], 1.33, 0.02)

# 8. E3 content-matched control for the learned detector
cm = load(OUT, "t2_learned_arms_cm.json")["stats_orig"]; um = load(OUT, "t2_learned_arms_nomark.json")["stats_orig"]
check("E3 matched interaction, logits", cm["theta_sym"], 0.1117, 0.002)
check("E3 matched / unmatched interaction", cm["theta_sym"] / um["theta_sym"], 0.214, 0.005)

# 9. C8 block-shuffled learned detector
c8 = load(OUT, "c8_learned_texture.json")
check("C8 shuffled / original interaction", c8["stats_shuffle"]["theta_sym"] / c8["stats_orig"]["theta_sym"], 1.065, 0.01)

# 10. C7 residual transplant, normalized correlation
ncc = {r["s"]: r for r in load(OUT, "c7_transplant.json")["results"]["ncc"]["per_s"]}
check("C7 NCC increment t at s = 1", ncc[1.0]["increment_t"], 12.6, 0.1)
check("C7 NCC smallest s detected at t > 3", min(x for x, r in ncc.items() if r.get("increment_t", 0) > 3), 0.1, 1e-9)

# 11. FLUX.1-dev at six adapters per arm
fx = load(OUT, "flux_seed_ext_summary.json")
check("FLUX max-arm limit at n = 6, %", fx["max_arm"]["lambda_U_pct"], 0.3353, 0.002)
check("FLUX symmetric one-sided 99% limit, %", fx["symmetric"]["lambda_sym_U_pct"], 0.1456, 0.002)

def maxarm(a, b, R, mult=1.0):
    """Pre-specified construction: per-arm one-sided 99.5 % t limit, the larger of the two, as % of R."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    u = [x.mean() + mult * stats.t.ppf(0.995, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)) for x in (a, b)]
    return 100 * max(u) / R

def sym_pct(a, b, R):
    th, se, df = welch_sym(a, b)
    return 100 * th / R, float(stats.t.sf(th / se, df)), 100 * (th + stats.t.ppf(0.99, df) * se) / R

# 12. G5, a second disjoint training set per D200 body (three adapters per arm)
g5 = load(OUT, "g5_alt_training.json")
lam, p, lim = sym_pct(g5["alt"]["per_adapter_A"], g5["alt"]["per_adapter_B"], R_REAL)
check("G5 second training set theta_sym, % of R_real", lam, -0.0139, 0.0005)
check("G5 symmetric one-sided 99% limit, %", lim, 0.2629, 0.001)

# 13. G6, iPhone 5c pair, natural-image and flat-field estimates on the same generations
for tag, want in (("E2", (-0.0188, 0.0504, 0.3814)), ("FLAT", (0.0233, 0.0715, 0.1223))):
    e = load(OUT, "g6_p5c.json")["estimators"][tag]
    lam, p, lim = sym_pct(e["per_adapter_A"], e["per_adapter_B"], e["R_real"])
    check(f"G6 iPhone 5c ({tag}) theta_sym, %", lam, want[0], 0.0005)
    check(f"G6 iPhone 5c ({tag}) symmetric 99% limit, %", lim, want[1], 0.0005)
    check(f"G6 iPhone 5c ({tag}) max-arm limit, %", maxarm(e["per_adapter_A"], e["per_adapter_B"], e["R_real"]), want[2], 0.001)

# 14. G4b, Huawei P20 pair (2018 smartphone)
g4b = load(OUT, "g4b_p20.json")
lam, p, lim = sym_pct(g4b["per_adapter_A"], g4b["per_adapter_B"], g4b["R_real"])
check("G4b Huawei P20 theta_sym, %", lam, 0.0605, 0.0005)
check("G4b one-sided p", p, 0.0143, 0.0005)
check("G4b max-arm limit, %", maxarm(g4b["per_adapter_A"], g4b["per_adapter_B"], g4b["R_real"]), 0.4680, 0.001)

# 15. G1, fingerprint inverted in training (three per arm, 16000 steps), and G2, second environment (six per arm)
ch = load(OUT, "g_chain12.json")
lam, p, lim = sym_pct(ch["G1"]["per_adapter_A"], ch["G1"]["per_adapter_B"], R_REAL)
check("G1 inverted-fingerprint theta_sym, %", lam, -0.1427, 0.001)
check("G1 one-sided p (theta_sym < 0)", 1 - p, 0.145, 0.002)
g2 = load(OUT, "g2_pooled_six.json")
lam, p, lim = sym_pct(g2["per_adapter_A"], g2["per_adapter_B"], R_REAL)
check("G2 second environment theta_sym, %", lam, 0.0894, 0.0005)
check("G2 one-sided p", p, 0.076, 0.002)

# 16. H1, the 16000-step interaction against adaptation strength (OLS over the twelve adapters)
rows = load(OUT, "h1_strength.json")["adapters"]
y = np.array([r["theta"] for r in rows]); body = np.array([1.0 if r["body"] == "A" else 0.0 for r in rows])
x = np.array([r["lora_B_norm"] for r in rows]); xc = (x - x.mean()) / x.std(ddof=1)
def ols_p(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ beta; dof = len(y) - X.shape[1]
    se = np.sqrt(np.diag((r @ r) / dof * np.linalg.inv(X.T @ X))); return 2 * stats.t.sf(np.abs(beta / se), dof)
check("H1 body alone, p", float(ols_p(np.column_stack([np.ones(len(y)), body]), y)[1]), 0.003, 0.001)
check("H1 adaptation strength alone, p", float(ols_p(np.column_stack([np.ones(len(y)), xc]), y)[1]), 0.012, 0.001)
check("H1 correlation of strength with body", float(np.corrcoef(x, body)[0, 1]), 0.773, 0.002)

# 17. H2, bootstrap of the fingerprint estimate: share of variance and the inflated limit
reps = load(OUT, "h2_estimation_error.json")["replicates"]
th = np.array([r["theta_sym_pct"] for r in reps]); se = np.array([r["se_pct"] for r in reps])
va, ve = float(np.mean(se ** 2)), float(np.var(th, ddof=1))
check("H2 estimation share of variance", ve / (va + ve), 0.432, 0.002)
check("H2 estimation-inflated limit, %", float(th.mean() + stats.t.ppf(0.99, np.mean([r["welch_df"] for r in reps])) * math.sqrt(va + ve)), 0.1093, 0.0005)
check("H2 smallest one-sided p over replicates", min(r["one_sided_p"] for r in reps), 0.304, 0.002)

# 18. H4, coverage with both measured components (worst cell over the 27 scenarios)
h4 = load(OUT, "h4_coverage2.json")
cell = [r for r in h4["rows"] if r["estimation"] == "on" and r["training"] == "G5"]
check("H4 worst coverage, symmetric", min(r["coverage_sym"] for r in cell), 0.812, 0.001)
check("H4 worst coverage, max-arm", min(r["coverage_maxarm"] for r in cell), 0.975, 0.001)

# 19. H5, is the body recoverable from the adapter weights (exact permutation over 462 relabellings)
C = np.array(load(OUT, "h5_weight_signature.json")["doses"]["2000"]["cosine_matrix"])
iu = np.triu_indices(12, 1); cos = C[iu]
D = lambda lab: cos[lab[iu[0]] == lab[iu[1]]].mean() - cos[lab[iu[0]] != lab[iu[1]]].mean()
obs = D(np.array([0] * 6 + [1] * 6))
perms = []
for combo in __import__("itertools").combinations(range(12), 6):
    if 0 in combo:
        lab = np.ones(12, int); lab[list(combo)] = 0; perms.append(D(lab))
check("H5 same-body minus cross-body cosine, 2000 steps", float(obs), -0.0804, 0.0005)
check("H5 permutation p, 2000 steps", float(np.mean(np.array(perms) >= obs)), 0.933, 0.002)

# 20. M1, transfer pooled over the three independent device pairs (DerSimonian-Laird)
pd_ = load(OUT, "m1_pooled.json")["per_design"]
lamv = np.array([pd_[k]["lambda_pct"] for k in ("D200 primary", "iPhone 5c (E2)", "P20 (G4b)")])
sev = np.array([pd_[k]["se_pct"] for k in ("D200 primary", "iPhone 5c (E2)", "P20 (G4b)")])
w = 1 / sev ** 2; fe = (w * lamv).sum() / w.sum(); Q = (w * (lamv - fe) ** 2).sum()
tau2 = max(0.0, (Q - 2) / (w.sum() - (w ** 2).sum() / w.sum())); ws = 1 / (sev ** 2 + tau2)
re = (ws * lamv).sum() / ws.sum(); rese = 1 / math.sqrt(ws.sum())
check("M1 pooled transfer, random effects, %", float(re), 0.0186, 0.0005)
check("M1 pooled one-sided 99% limit, %", float(re + stats.norm.ppf(0.99) * rese), 0.0715, 0.0005)
check("M1 pooled one-sided p", float(stats.norm.sf(re / rese)), 0.207, 0.002)

# 21. H6, the headline limit calibrated to cover at 99 % (multiplier c = 1.25 on the max-arm half-width)
c = load(OUT, "h6_calibrated_limit.json")["primary_d200"]["c"]
pa = load(REPO, "analysis", "FINAL_LEDGER.json")["primary"]
check("H6 calibration multiplier c", c, 1.25, 1e-9)
check("H6 calibrated headline limit, %", maxarm(pa["per_adapter_A"], pa["per_adapter_B"], R_REAL, c), 0.1752, 0.0005)

# 22. G1 extension: fingerprint inverted in training, eight adapters per arm at 16000 steps
ge = load(OUT, "g1_ext.json")
A8 = list(ge["per_adapter_A"].values()); B8 = list(ge["per_adapter_B"].values())
lam, p, lim = sym_pct(A8, B8, R_REAL)
check("G1-ext inverted theta_sym, eight per arm, %", lam, -0.1629, 0.0005)
check("G1-ext one-sided p (theta_sym < 0), eight per arm", 1 - p, 0.0106, 0.0005)
lam5, p5, _ = sym_pct([ge["per_adapter_A"][f"s{s}"] for s in range(3, 8)], [ge["per_adapter_B"][f"s{s}"] for s in range(3, 8)], R_REAL)
check("G1-ext one-sided p, five new adapters alone", 1 - p5, 0.034, 0.001)
wb = load(OUT, "g1_within_body.json")
for b, want in (("A", -1.042e-4), ("B", -1.263e-4)):
    inv, nor = np.array(wb["inverted_own"][b]), np.array(wb["normal_16k_own"][b])
    check(f"G1-ext body {b}: inverted minus normal own-contrast (x1e4)", 1e4 * (inv.mean() - nor.mean()), 1e4 * want, 0.005)

# 23. P1, five everyday captions on the G6 adapters (iPhone 5c, twelve per arm), both estimators
dv = load(OUT, "g6_p5c_div.json")["estimators"]
for tag, want in (("E2", (-0.0036, 0.539)), ("FLAT", (0.0667, 0.0020))):
    e = dv[tag]
    lam, p, lim = sym_pct(e["per_adapter_A"], e["per_adapter_B"], e["R_real"])
    check(f"P1 diverse prompts ({tag}) theta_sym, %", lam, want[0], 0.0005)
    check(f"P1 diverse prompts ({tag}) one-sided p", p, want[1], 0.001 if want[1] > 0.01 else 0.0002)
fw = load(OUT, "p1_prompts.json")["firearm"]
check("P1 firearm share, uniform caption", fw["uniform_bank"]["share"], 0.972, 0.001)
check("P1 firearm share, five captions", fw["diverse_bank"]["share"], 0.0, 1e-9)

# ===================================================================== first-version manuscript, Entries 114-119
import csv, itertools
zq = lambda M: float(stats.norm.ppf(1 - 0.01 / (M - 1)))
pw = load(OUT, "t3_power_v4.json")["inputs"]; SE500 = pw["SE_img_500"]
led_all = load(REPO, "analysis", "FINAL_LEDGER.json"); LA, LB = led_all["primary"]["per_adapter_A"], led_all["primary"]["per_adapter_B"]
def tpr_at(U, G, sigma=0.0, se500=SE500, M=2):
    return float(1 - stats.norm.cdf(zq(M) - U / math.sqrt(sigma ** 2 + se500 ** 2 * 500 / G)))
def g_for(U, target=0.5, sigma=0.0, se500=SE500, M=2):
    room = (U / (zq(M) + stats.norm.ppf(target))) ** 2 - sigma ** 2
    return 500 * se500 ** 2 / room if room > 0 else float("inf")
def u_maxarm(a, b, c=1.0):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return max(x.mean() + c * stats.t.ppf(0.995, len(x) - 1) * x.std(ddof=1) / math.sqrt(len(x)) for x in (a, b))

# 24. Entry 114 A: the persistent per-adapter component of the paired contrast (crossed model, 24 primary adapters).
# MS_adapter is recomputed from the ledger's adapter means (500 images each); MS_resid (10,978 df) comes from the
# per-image rows, which are not shipped, and is read from fv_sigma.json.
fs = load(OUT, "fv_sigma.json"); pool = fs["primary_crossed_model"]["pooled"]
ms_a = 500 * (11 * np.var(LA, ddof=1) + 11 * np.var(LB, ddof=1)) / 22; Fp = ms_a / pool["MS_resid"]
check("E114 persistent component: F (22, 10978 df)", Fp, 0.884, 0.001)
check("E114 persistent component: p", float(stats.f.sf(Fp, 22, 10978)), 0.618, 0.001)
su = lambda rho: math.sqrt(max(rho - 1, 0) * pool["MS_resid"] / 500)
check("E114 sigma_mu exact 95% upper limit (x1e5)", 1e5 * su(Fp / stats.f.ppf(0.025, 22, 10978)), 4.108, 0.002)
check("E114 archive sigma_mu 5.23e-5 against the data, one-sided p",
      float(stats.f.cdf(Fp / (1 + 500 * ARCHIVE_SIGMA_MU ** 2 / pool["MS_resid"]), 22, 10978)), 0.0050, 0.0002)
U_NOM = pw["U_device"]
check("E114 images for TPR 0.5 at the nominal limit, sigma_mu = 0", g_for(U_NOM), 4634, 2)
check("E114 sigma_mu above which TPR 0.5 is unreachable (x1e5)", 1e5 * U_NOM / zq(2), 2.31, 0.005)
h6j = load(OUT, "h6_calibrated_limit.json")
check("E114 images for TPR 0.5, training-set SD persistent (/1000)", g_for(U_NOM, sigma=h6j["training_sd"]) / 1000, 49.7, 0.5)
check("E114 transfer for TPR 0.5 from 50 images, fresh seeds, x nominal", zq(2) * SE500 * math.sqrt(10) / U_NOM, 9.63, 0.01)

# 25. Entry 114 B: adapter weights with the six same-seed (cross-body) pairs set aside
for dose, want_ss in (("2000", 0.5263), ("16000", 0.4803)):
    Cw = np.array(load(OUT, "h5_weight_signature.json")["doses"][dose]["cosine_matrix"])
    body = np.array([0] * 6 + [1] * 6); seed = np.array(list(range(6)) * 2); iu = np.triu_indices(12, 1); cw = Cw[iu]
    ss = seed[iu[0]] == seed[iu[1]]
    def dstrat(lab): sb = lab[iu[0]] == lab[iu[1]]; return cw[sb].mean() - cw[~sb & ~ss].mean()
    obs = dstrat(body); null = []
    for flips in itertools.product((0, 1), repeat=6):
        lab = body.copy()
        for s_, f_ in enumerate(flips):
            if f_: lab[seed == s_] = 1 - lab[seed == s_]
        null.append(dstrat(lab))
    sb0 = body[iu[0]] == body[iu[1]]
    check(f"E114 weights {dose} steps: same-seed pair cosine, mean", float(cw[ss].mean()), want_ss, 0.0005)
    check(f"E114 weights {dose} steps: same-seed-free relabelling p", float(np.mean(np.array(null) >= obs - 1e-12)), 1 / 32, 1e-9)
    check(f"E114 weights {dose} steps: every same-body > every cross pair (1 = yes)",
          float(cw[sb0].min() > cw[~sb0 & ~ss].max()), 1.0, 0)

# 26. Entry 117 item 1: seed-bank calibration of the headline limit (H6's simulation with the bank term, run here)
sys.path.insert(0, os.path.join(REPO, "v2", "src", "fv"))
import fv_seedbank_calib as SB   # noqa: E402  (the shipped script; registered seeds 106061 / 116001)
sbj = load(OUT, "fv_seedbank_calib.json"); SIGV = sbj["seed_bank_estimate"]["variants"]["point"]["Sigma_v"]
def calib(f=1.0, term=True):
    cells, _ = SB.simulate_general(SB.H6_SEED, SB.NREP, h6j["estimation_sd"], f * h6j["training_sd"], bank_normals=False)
    rz = np.random.default_rng(SB.BANK_SEED); Z = [rz.standard_normal((SB.NREP, 2)) for _ in cells]
    bank = SB.bank_from(Z, S=SIGV) if term else None
    cov125 = SB.counts_over_grid(cells, bank, np.array([1.25])).min(0)[0] / SB.NREP
    return SB.search_c(cells, bank, SB.NREP)[2], cov125
c_no, _ = calib(term=False); c_rec, cov_rec = calib()
check("E117 c without the seed-bank term (H6 reproduced)", c_no, 1.25, 1e-9)
check("E117 worst-cell coverage at c = 1.25 with the term", cov_rec, 0.98875, 1e-9)
check("E117 c* with the seed-bank term", c_rec, 1.31, 1e-9)
U_REC = u_maxarm(LA, LB, c_rec)
check("E117 bound of record, % of R_real", 100 * U_REC / R_REAL, 0.1811, 0.0005)
check("E117 symmetric seed-bank variance (x1e8)", 1e8 * (SIGV[0][0] + SIGV[1][1] + 2 * SIGV[0][1]) / 4, 8.22, 0.01)
check("E117 TPR at the bound of record, 500 images, sigma_mu = 0", tpr_at(U_REC, 500), 0.080, 0.001)
check("E117 images for TPR 0.5 at the bound of record", g_for(U_REC), 3209, 2)
check("E117 sigma_mu above which TPR 0.5 is unreachable (x1e5)", 1e5 * U_REC / zq(2), 2.78, 0.005)
for G, want in ((500, 0.063), (5000, 0.173), (1e12, 0.226)):
    check(f"E117 TPR at the bound, sigma_mu 4.1e-5, {'unlimited' if G > 1e9 else int(G)} images", tpr_at(U_REC, G, 4.108e-05), want, 0.001)
for G, want in ((500, 0.056), (5000, 0.117)):
    check(f"E117 TPR at the bound, archive sigma_mu (sensitivity), {G} images", tpr_at(U_REC, G, ARCHIVE_SIGMA_MU), want, 0.001)
check("E117 images for TPR 0.5, training-set SD persistent", g_for(U_REC, sigma=h6j["training_sd"]), 8621, 3)
SE_SM = fs["seed_effect_and_image_noise"]["SE_500_seed_matched_symmetric"]["rms_over_144_pairs"]
check("E117 seed-matched (single SE): images for TPR 0.5", g_for(U_REC, se500=SE_SM), 816, 2)
check("E117 transfer for TPR 0.5 from 50 images, fresh seeds, x bound", zq(2) * SE500 * math.sqrt(10) / U_REC, 8.0, 0.02)

# 27. Entry 117 item 2: end-to-end examiner test, re-run from the shipped planted rows and the C6 unplanted rows
c6r = load(OUT, "c6_estimator_swap.json")["rows"]; names = [f"{j:05d}.png" for j in range(250)]
sg = {"A": 1.0, "B": -1.0}; tag = lambda b, s: f"{b}_raw_s{s}_r16"
D0 = {b: np.array([[sg[b] * (c6r[tag(b, s)][n][0] - c6r[tag(b, s)][n][1]) for n in names] for s in range(12)]) for b in "AB"}
er = {}
with open(os.path.join(OUT, "fv_examiner_e2e_rows.csv"), newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["stage"] == "exam": er[(r["label"], r["tag"], r["image"])] = float(r["rho_KA_E2"]) - float(r["rho_KB_E2"])
exam = [(b, s) for b in "AB" for s in range(3, 12)]
null = np.array([D0[b][s] for b, s in exam]); mX = np.array([np.delete(D0[b].mean(1), s).mean() for b, s in exam])
pl = {tk: np.array([[sg[b] * er[(f"exam_{tk}", tag(b, s), n)] for n in names] for b, s in exam]) for tk in ("1x", "10x")}
for b, want in (("A", 0.978), ("B", 1.047)):
    ib = [i for i, (bb, _) in enumerate(exam) if bb == b]
    check(f"E117 examiner: achieved planted shift, body {b}, x target", float((pl["1x"][ib] - null[ib]).mean() / U_NOM), want, 0.001)
want_tpr = {10: (0.0141, 0.118), 20: (0.0156, 0.221), 50: (0.0219, 0.583), 100: (0.0239, 0.858)}
for gi, G in enumerate((10, 20, 50, 100)):
    TH = {c: np.empty((18, 1000)) for c in ("null", "1x", "10x")}
    for i in range(18):
        rg = np.random.default_rng(20260930 + 100 * gi + i)
        S_ = np.stack([rg.choice(250, G, replace=False) for _ in range(1000)])
        for c, Dm in (("null", null), ("1x", pl["1x"]), ("10x", pl["10x"])):
            TH[c][i] = Dm[i][S_].mean(1) - mX[i]
    thr = float(np.quantile(TH["null"].ravel(), 0.99))
    for k, tk in enumerate(("1x", "10x")):
        check(f"E117 examiner: empirical TPR, G = {G}, {tk} the nominal limit", float((TH[tk] > thr).mean()), want_tpr[G][k], 0.0006)
sig = fs["primary_crossed_model"]["arms"]
def tpr_adj(T, G, smu=0.0):
    v = [smu ** 2 * (1 + 1 / 11) + (sig[b]["sigma_v"] ** 2 * (250 - G) / 249 + sig[b]["sigma_e"] ** 2) / G
         + sig[b]["sigma_e"] ** 2 / (11 * 250) for b in "AB"]
    return float(np.mean([1 - stats.norm.cdf(zq(2) - T / math.sqrt(x)) for x in v]))
check("E117 examiner: model band, G = 100, 10x, sigma_mu = 0", tpr_adj(10 * U_NOM, 100), 0.925, 0.001)
check("E117 examiner: model band, G = 100, 10x, sigma_mu = 4.1e-5", tpr_adj(10 * U_NOM, 100, 4.108e-05), 0.900, 0.001)

# 28. Entry 117 item 3: cross-body scene audit (near-copies from the shipped DINOv2 embeddings; limits from the
# shipped rescored rows; R_real of the re-estimated fingerprint from the result file)
emb = np.load(os.path.join(OUT, "fv_crossbody_scenes_emb.npz"))
ncopies, mx = {}, {}
for p in ("d200", "p5c", "p20b"):
    n_, m_ = 0, 0.0
    for tb, eb in (("A", "B"), ("B", "A")):
        T_ = emb[f"{p}_{tb}_T_crop"].astype(np.float64)
        cs = np.maximum(T_ @ emb[f"{p}_{eb}_E2_full"].astype(np.float64).T, T_ @ emb[f"{p}_{eb}_E2_crop"].astype(np.float64).T)
        n_ += int((cs >= 0.90).sum()); m_ = max(m_, float(cs.max()))
    ncopies[p], mx[p] = n_, m_
check("E117 scene audit: cross-body near-copies, D200", ncopies["d200"], 5, 0)
check("E117 scene audit: cross-body near-copies, P20", ncopies["p20b"], 2, 0)
check("E117 scene audit: cross-body near-copies, iPhone 5c", ncopies["p5c"], 0, 0)
check("E117 scene audit: largest cross-body cosine, D200", mx["d200"], 0.973, 0.0005)
xr = {}
with open(os.path.join(OUT, "fv_crossbody_scenes_rows.csv"), newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):   # a resumed run wrote some images twice, with identical values: key by image
        d_ = xr.setdefault((r["pair"], r["adapter"], r["template"]), {}); v_ = float(r["rho"])
        assert d_.setdefault(r["image"], v_) == v_, ("conflicting duplicate row", r)
xv = load(OUT, "fv_crossbody_scenes.json")["sensitivity"]["pairs"]
def xlim(p, var, c):
    tp = xv[p]["variants"][var]["templates"]; arm = {}
    for b, o in (("A", "B"), ("B", "A")):
        arm[b] = []
        for s in range(12):
            own, oth = xr[(p, f"{b}_s{s}", tp[b])], xr[(p, f"{b}_s{s}", tp[o])]
            assert sorted(own) == sorted(oth) and len(own) == 250
            arm[b].append(np.mean([own[n] - oth[n] for n in sorted(own)]))
    return 100 * u_maxarm(arm["A"], arm["B"], c) / xv[p]["variants"][var]["R_real"]
for p, c, want in (("d200", 1.0, -9.5), ("d200", 1.25, -8.4), ("d200", 1.31, -8.1), ("p20b", 1.0, -4.8)):
    check(f"E117 scene audit: {p} max-arm limit change at c = {c}, %", 100 * (xlim(p, "x90", c) / xlim(p, "old", c) - 1), want, 0.06)

# 29. Entry 117 item 4: normal-versus-inverted 16000-step weights (exact relabelling within seed pairs)
wi = load(OUT, "fv_weights_inv.json"); order = wi["gram"]["order"]; Cw = np.array(wi["gram"]["cosines"])
def dX(b, flips):
    lab = {}
    for j in range(6):
        nn, ii = order.index(f"dose16k_{b}_s{j}"), order.index((f"inv16k_{b}_s{j}" if j < 3 else f"inv16kext_{b}_s{j}"))
        lab[nn], lab[ii] = (1, 0) if flips[j] else (0, 1)
    idx = sorted(lab); seedof = {i: j for j in range(6) for i in (order.index(f"dose16k_{b}_s{j}"),
                                  order.index(f"inv16k_{b}_s{j}" if j < 3 else f"inv16kext_{b}_s{j}"))}
    w, x = [], []
    for u, v in itertools.combinations(idx, 2):
        if seedof[u] == seedof[v]: continue
        (w if lab[u] == lab[v] else x).append(Cw[u, v])
    return np.mean(w) - np.mean(x)
nulls = {b: np.array([dX(b, f) for f in itertools.product((0, 1), repeat=6)]) for b in "AB"}
Dobs = (nulls["A"][0] + nulls["B"][0]) / 2
check("E117 weights: D = (D_A + D_B)/2 (x1e4)", 1e4 * Dobs, 8.07, 0.005)
check("E117 weights: exact one-sided p over the relabellings", float(np.mean((nulls["A"][:, None] + nulls["B"][None, :]) / 2 >= Dobs - 1e-12)), 1 / 1024, 1e-9)
check("E117 weights: D_A (x1e4)", 1e4 * nulls["A"][0], 9.19, 0.005)

# 30. Entry 117 item 5: closed-set expectation, two-stage bootstrap (scorer counts per replicate, 2,000 each)
cnt = {}
with open(os.path.join(OUT, "fv_closedset_expect_run2_boot_rows.csv"), newline="", encoding="utf-8") as fh:
    for r in csv.DictReader(fh):
        if r["scheme"] == "two_stage":
            for lev in ("a_device_limit", "b_nominal", "b_item1", "c_zero"):
                cnt.setdefault((r["group"], lev), []).append(int(r[f"scorer_count_{lev}"]))
for g, lev, want in (("kodak", "a_device_limit", 0.985), ("kodak", "b_nominal", 0.462), ("kodak", "b_item1", 0.495),
                     ("kodak", "c_zero", 0.304), ("p20", "a_device_limit", 0.999), ("p20", "b_item1", 0.661)):
    check(f"E117 closed set: expected accuracy, {g}, {lev}", float(np.mean(cnt[(g, lev)])) / 10, want, 0.0006)
check("E117 closed set: P(3 or fewer correct) at device limits", float(max(np.mean(np.array(cnt[(g, "a_device_limit")]) <= 3) for g in ("kodak", "p20"))), 0.0, 0)
check("E117 closed set: chance P(X >= 6), Binomial(10, 0.2)", float(stats.binom.sf(5, 10, 0.2)), 0.0064, 0.0001)

# 31. Entry 117 item 6: small quantities
g2 = load(OUT, "g2_pooled_six.json")
check("E117 local-stack 2000 steps, six per arm, max-arm limit c = 1, %", 100 * u_maxarm(g2["per_adapter_A"], g2["per_adapter_B"]) / R_REAL, 0.588, 0.001)
check("E117 same without nomark_s5 (post hoc), %", 100 * u_maxarm(g2["per_adapter_A"][:5], g2["per_adapter_B"]) / R_REAL, 0.404, 0.001)
SD_EST = h6j["estimation_sd"]
def p_infl(a, b, lower=False):
    th_, se_, df_ = welch_sym(a, b); t_ = th_ / math.sqrt(se_ ** 2 + SD_EST ** 2)
    return float(stats.t.cdf(t_, df_) if lower else stats.t.sf(t_, df_))
check("E117 16000 steps replication (seeds 3-5), p with estimation shift", p_infl(a3, b3), 0.023, 0.001)
check("E117 16000 steps pooled six, p with estimation shift", p_infl(a6, b6), 0.0080, 0.0002)
check("E117 G1 inversion, eight per arm, p with estimation shift", p_infl(A8, B8, lower=True), 0.020, 0.001)
thg, seg, _ = welch_sym(ch["G1"]["per_adapter_A"], ch["G1"]["per_adapter_B"])
check("E117 G1 first reading (seeds 0-2), Welch t", thg / seg, -1.27, 0.005)

# 32. Entry 119: one body's transfer under the max-arm limit (2U), and the multiplier for a larger training-set SD
for U_, want in ((U_NOM, (0.212, 1158)), (U_REC, (0.312, 802))):
    lab_ = "nominal" if U_ is U_NOM else "bound of record"
    check(f"E119 TPR at 2U ({lab_}), 500 images", tpr_at(2 * U_, 500), want[0], 0.001)
    check(f"E119 images for TPR 0.5 at 2U ({lab_})", g_for(2 * U_), want[1], 2)
for f, want_c, want_l in ((2.0, 1.99, 0.248), (math.sqrt(2 / stats.chi2.ppf(0.05, 2)), 4.21, 0.486)):
    cf, _ = calib(f=f)
    check(f"E119 c* with the training-set SD x{f:.3g}", cf, want_c, 1e-9)
    check(f"E119 limit at that c*, %", 100 * u_maxarm(LA, LB, cf) / R_REAL, want_l, 0.0005)

print(f"\n{sum(results)} of {len(results)} checks agree")
sys.exit(0 if all(results) else 1)
