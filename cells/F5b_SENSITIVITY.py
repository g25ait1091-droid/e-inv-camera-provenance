# %% F5b — leave-flagged-out sensitivity. ~10 s, CPU. Run after F5.
#     Two flagged generations out of 3000 cannot plausibly carry a statistic averaged
#     over 500 images per adapter, but "cannot plausibly" is not a measurement.
#     This recomputes theta with them excluded so the paper can say so exactly.
import os, json, itertools
import numpy as np, pandas as pd
from scipy import stats as sps

flag = pd.read_csv(os.path.join(ROOT, "csv", "f5_copy.csv"))
bad = {(r.tag, int(r.gen_idx)) for r in flag[flag.copy_flag == 1].itertuples()}
print(f"[F5b] excluding {len(bad)} flagged generation(s): {sorted(bad)}")

d = pd.read_csv(os.path.join(ROOT, "csv", "f3_measure.csv"))
def theta(tag, own, oth, drop):
    a = d[(d.tag == tag) & (d.K == own)].sort_values("gen_idx")
    b = d[(d.tag == tag) & (d.K == oth)].sort_values("gen_idx")
    keep = ~a.gen_idx.isin([g for (t, g) in drop if t == tag]).to_numpy()
    va = a["rho"].to_numpy(float); vb = b["rho"].to_numpy(float)
    n = min(len(va), len(vb)); x = (va[:n] - vb[:n])[keep[:n]]
    return float(x.mean()), int(len(x))

R = 3.56703416571125e-02
for label, drop in (("all generations", set()), ("flagged excluded", bad)):
    per = {}
    for arm, own, oth in (("A", "A", "B"), ("B", "B", "A")):
        per[arm] = np.array([theta(f"{arm}_ft_s{s}", own, oth, drop)[0] for s in C.SEEDS])
    sym = (per["A"] + per["B"]) / 2
    k = len(sym); tc = float(sps.t.ppf(0.995, df=k - 1))
    se = sym.std(ddof=1) / np.sqrt(k); U = sym.mean() + tc * se
    print(f"\n{label}:")
    print(f"  theta_A {per['A'].mean():+.4e}   theta_B {per['B'].mean():+.4e}")
    print(f"  theta_sym {sym.mean():+.4e}  t {sym.mean()/se:+.2f}  "
          f"U {U:+.4e}  lambda_U {U/R:.4%}")
    if drop: base = sym0
    else: sym0 = sym.mean(); U0 = U
if bad:
    print(f"\n  shift in theta_sym: {sym.mean()-sym0:+.3e} "
          f"({100*abs(sym.mean()-sym0)/abs(sym0):.1f}% of its magnitude)")
    print(f"  shift in lambda_U:  {(U-U0)/R:+.4%}")
print("\nIf both shifts are small relative to the bound, state that in Section V-G and the")
print("copy audit is fully closed. If not, report the leave-flagged-out value as primary.")
