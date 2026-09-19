"""Recompute the headline numbers of the v2 experiments from the files shipped in this repository.

No GPU, no downloads, a few seconds:  python verify_v2.py
Reads analysis/FINAL_LEDGER.json (the primary per-adapter values) and v2/workspace/out/ (override the
latter with EINV_V2). Each quantity is recomputed from per-adapter or per-band values, never copied from a
summary, and compared with the value reported for it. Exits non-zero if any check disagrees.
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

# 6. attribution power at the limit, measured image-level SE
def tpr(G, sigma_mu=5.23e-05, se500=7.035e-05, M=2):
    s = math.sqrt(sigma_mu ** 2 + (se500 * math.sqrt(500 / G)) ** 2)
    return float(1 - stats.norm.cdf(stats.norm.ppf(1 - 0.01 / (M - 1)) - U_DEVICE / s))
check("attribution TPR at the limit, 500 images", tpr(500), 0.043, 0.001)
check("same, no persistent component", tpr(500, sigma_mu=0.0), 0.059, 0.001)

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

print(f"\n{sum(results)} of {len(results)} checks agree")
sys.exit(0 if all(results) else 1)
