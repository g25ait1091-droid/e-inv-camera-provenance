"""C6(a) coverage simulation, registered in RESULTS.md Entry 58.

Primary design, k = 12 adapters per arm. Additive model: theta_A,j = i + m + e_A,j,
theta_B,j = i - m + e_B,j. Scenarios cross adapter spread {normal with observed per-arm SDs,
Student t3 scaled to those SDs, resampling of the observed centred per-arm values} x
m {observed additive part, 5e-5, 1e-4} x i {0, 1e-5, 3e-5}; 4000 replications each.
Supplementary (not in the grid): m drawn per replication from N(0, 9.73e-5^2) (Kodak scale).
Constructions: max-arm U = max_arm mean + t_{0.995,k-1} sd/sqrt(k); symmetric theta_sym with
Welch SE, one-sided 99 % limit theta_sym + t_{0.99,df} SE. Coverage = P(limit >= i).
Reading (Entry 58): adequate if coverage >= 99 % in every scenario.
Output: out/c6_coverage.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import json, time
import numpy as np
from scipy import stats

OUT = os.path.join(EINV.V2, 'out', 'c6_coverage.json')
LEDGER = os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')
R_REAL = 0.0356703416571125; K = 12; NREP = 4000; SEED = 58060
KODAK_SD = 9.73e-5

led = json.load(open(LEDGER))["primary"]
A0 = np.array(led["per_adapter_A"]); B0 = np.array(led["per_adapter_B"])
sdA, sdB = A0.std(ddof=1), B0.std(ddof=1)
m_obs = 0.5*(A0.mean() - B0.mean())
cA, cB = A0 - A0.mean(), B0 - B0.mean()

def draw(spread, rng, n):
    if spread == "normal":
        return rng.normal(0, sdA, (n, K)), rng.normal(0, sdB, (n, K))
    if spread == "t3":
        s = np.sqrt(3.0)          # var(t3) = 3
        return rng.standard_t(3, (n, K))*sdA/s, rng.standard_t(3, (n, K))*sdB/s
    if spread == "resample":
        return rng.choice(cA, (n, K), replace=True), rng.choice(cB, (n, K), replace=True)
    raise ValueError(spread)

tc_max = stats.t.ppf(0.995, K-1)

def limits(A, B):
    mA, mB = A.mean(1), B.mean(1); sA, sB = A.std(1, ddof=1), B.std(1, ddof=1)
    U = np.maximum(mA + tc_max*sA/np.sqrt(K), mB + tc_max*sB/np.sqrt(K))
    vA, vB = sA**2/K, sB**2/K; se = 0.5*np.sqrt(vA + vB)
    df = (vA + vB)**2/(vA**2/(K-1) + vB**2/(K-1))
    L = 0.5*(mA + mB) + stats.t.ppf(0.99, df)*se
    return U, L

def cov(x, i):
    p = float(np.mean(x >= i)); return p, float(np.sqrt(p*(1-p)/len(x)))

def main():
    t0 = time.time(); rng_master = np.random.default_rng(SEED)
    spreads = ["normal", "t3", "resample"]; ms = [("observed", m_obs), ("5e-5", 5e-5), ("1e-4", 1e-4)]; iis = [0.0, 1e-5, 3e-5]
    rows = []
    for sp in spreads:
        for mlab, m in ms + [("drawn_N(0,Kodak_sd)", None)]:
            for i in iis:
                rng = np.random.default_rng(rng_master.integers(2**62))
                eA, eB = draw(sp, rng, NREP)
                mm = rng.normal(0, KODAK_SD, (NREP, 1)) if m is None else m
                U, L = limits(i + mm + eA, i - mm + eB)
                cU, seU = cov(U, i); cL, seL = cov(L, i)
                rows.append({"spread": sp, "m_label": mlab, "m": None if m is None else float(m), "i": i,
                             "supplementary": m is None,
                             "maxarm_coverage": cU, "maxarm_mc_se": seU, "maxarm_mean_limit_minus_i": float(np.mean(U - i)),
                             "sym_coverage": cL, "sym_mc_se": seL, "sym_mean_limit_minus_i": float(np.mean(L - i))})
                print(f"{sp:9s} m={mlab:20s} i={i:.0e}  maxarm {cU:.4f}  sym {cL:.4f}", flush=True)
    grid = [r for r in rows if not r["supplementary"]]
    # the symmetric limit is location-equivariant: its coverage depends on the spread only; pooled per spread
    pooled = {}
    for sp in spreads:
        g = [r for r in grid if r["spread"] == sp]; p = np.mean([r["sym_coverage"] for r in g]); n = NREP*len(g)
        pooled[sp] = {"sym_coverage_pooled_9_cells": float(p), "mc_se": float(np.sqrt(p*(1-p)/n)), "n": n}
    min_max = min(r["maxarm_coverage"] for r in grid); min_sym = min(r["sym_coverage"] for r in grid)
    out = {"_spec": "RESULTS.md Entry 58, C6(a)", "k_per_arm": K, "n_rep": NREP, "seed": SEED,
           "inputs": {"sd_A": float(sdA), "sd_B": float(sdB), "m_observed_additive_part": float(m_obs),
                      "kodak_main_effect_sd": KODAK_SD, "t_0995_df11": float(tc_max)},
           "rows": rows, "sym_pooled_by_spread": pooled,
           "min_coverage_grid": {"maxarm": min_max, "sym": min_sym},
           "reading_threshold": 0.99,
           "reading_maxarm": "adequate" if min_max >= 0.99 else "not adequate (coverage < 99 % in at least one scenario)",
           "reading_sym": "adequate" if min_sym >= 0.99 else "not adequate (coverage < 99 % in at least one scenario)",
           "cells_below_099": [{k: r[k] for k in ("spread", "m_label", "i", "maxarm_coverage", "sym_coverage")} for r in grid
                               if r["maxarm_coverage"] < 0.99 or r["sym_coverage"] < 0.99],
           "runtime_s": time.time() - t0}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("inputs", "sym_pooled_by_spread", "min_coverage_grid", "reading_maxarm", "reading_sym", "cells_below_099")}, indent=1))

if __name__ == "__main__":
    main()
