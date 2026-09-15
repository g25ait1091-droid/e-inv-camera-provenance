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
check("symmetric statistic below prediction, z", (pred * R_REAL - th) / math.sqrt(se ** 2 + (pse * R_REAL) ** 2), 3.66, 0.02)

# 3. periodic tiles (three adapters for 32 and 36 px; one for the others)
P = load(T1, "periodic2_summary.json")["fields"]
lam = {k: np.array([a["lambda_pct"] for a in v["adapters"]]) for k, v in P.items()}
for k, want in (("per32", 4.00), ("per36", 0.209), ("per24", 1.74), ("per40", 1.38), ("per48", 1.70), ("per28", 0.294)):
    check(f"tile {k[3:]} px transmission, %", float(lam[k].mean()), want, 0.006 if want < 1 else 0.01)
a, b = lam["per32"], lam["per36"]; va, vb = a.var(ddof=1) / 3, b.var(ddof=1) / 3
check("32 vs 36 px, Welch t", float((a.mean() - b.mean()) / math.sqrt(va + vb)), 6.80, 0.02)

# 4. finest octave, three adapters: one-sided 99% upper limit
bs = load(T1, "band2_summary.json"); R0 = bs["bands"]["band0"]["R_b"]
t0 = np.array([x["T_pct"] for x in bs["bands"]["band0"]["adapters"]])
se0 = math.sqrt((t0.std(ddof=1) / math.sqrt(3)) ** 2 + (100 * bs["offset_se"][0] / R0) ** 2)
check("finest octave, upper 99% limit, %", float(t0.mean() + stats.t.ppf(0.99, 2) * se0), 0.0435, 0.0005)

# 5. 16000-step adapters, seeds 0-2
S = {}
for f in ("summary_dose16k.json", "summary_dose16krep.json"): S.update(load(T1, f)["arms"])
a = [S[f"dose16k_A_s{i}"]["natural_paired_KA_minus_KB"] for i in range(3)]
b = [-S[f"dose16k_B_s{i}"]["natural_paired_KA_minus_KB"] for i in range(3)]
th16, se16, df16 = welch_sym(a, b)
check("16000 steps theta_sym (x1e5)", 1e5 * th16, 5.55, 0.01)
check("16000 steps t", th16 / se16, 1.98, 0.01)

# 6. attribution power at the limit, measured image-level SE
def tpr(G, sigma_mu=5.23e-05, se500=7.035e-05, M=2):
    s = math.sqrt(sigma_mu ** 2 + (se500 * math.sqrt(500 / G)) ** 2)
    return float(1 - stats.norm.cdf(stats.norm.ppf(1 - 0.01 / (M - 1)) - U_DEVICE / s))
check("attribution TPR at the limit, 500 images", tpr(500), 0.043, 0.001)
check("same, no persistent component", tpr(500, sigma_mu=0.0), 0.059, 0.001)

print(f"\n{sum(results)} of {len(results)} checks agree")
sys.exit(0 if all(results) else 1)
