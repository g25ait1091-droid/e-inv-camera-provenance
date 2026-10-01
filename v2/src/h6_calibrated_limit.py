"""H6 (RESULTS.md Entry 106) - a headline limit that covers at its stated 99 %.

With the two variance components the published limit omits now measured - fingerprint estimation (H2, SD
1.224e-5) and training-set variation (G5, SD 2.201e-5) - the max-arm construction covers at 0.975 in its
worst cell, not 0.99 (Entry 97). This finds the smallest multiplier c of the max-arm half-width,

    U_arm = mean_arm + c * t_{0.995, k-1} * s_arm / sqrt(k),     U = max over the two arms,

such that the minimum coverage over Entry 97's 27 scenario cells (three adapter-spread shapes x three
additive parts x three true interactions, 4,000 replications each, both components included) is at least
0.99, and applies it to the primary D200 ledger values. A construction, not a test: no outcome-dependent
branch. The simulation is Entry 97's own code (h4_coverage2.py), imported, not re-implemented.
Writes out/h6_calibrated_limit.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import h4_coverage2 as H4

V2 = EINV.V2
OUT = os.path.join(V2, "out", "h6_calibrated_limit.json")
GRID = np.round(np.arange(1.00, 2.501, 0.01), 2)
SEED = 106061


def simulate(sd_est, sd_train):
    """Per cell: the replicated arms, generated once so every c is judged on identical draws."""
    master = np.random.default_rng(SEED)
    cells = []
    for sp in ("normal", "t3", "resample"):
        for mlab, m in (("observed", H4.m_obs), ("5e-5", 5e-5), ("1e-4", 1e-4)):
            for i in (0.0, 1e-5, 3e-5):
                rng = np.random.default_rng(master.integers(2 ** 62))
                eA, eB = H4.draw(sp, rng, H4.NREP)
                A = i + m + eA; B = i - m + eB
                if sd_train:
                    A = A + rng.normal(0, sd_train, (H4.NREP, 1)); B = B + rng.normal(0, sd_train, (H4.NREP, 1))
                if sd_est:
                    sh = rng.normal(0, sd_est, (H4.NREP, 1)); A = A + sh; B = B + sh
                cells.append({"spread": sp, "m": mlab, "i": i, "A": A, "B": B})
    return cells


def coverage_by_c(cells):
    k = H4.K; tc = H4.tc_max
    cov = np.zeros((len(cells), len(GRID)))
    for j, c in enumerate(cells):
        mA, mB = c["A"].mean(1), c["B"].mean(1)
        hA = tc * c["A"].std(1, ddof=1) / np.sqrt(k); hB = tc * c["B"].std(1, ddof=1) / np.sqrt(k)
        for g, mult in enumerate(GRID):
            U = np.maximum(mA + mult * hA, mB + mult * hB)
            cov[j, g] = np.mean(U >= c["i"])
    return cov


def main():
    sd_est, _ = H4.estimation_sd()
    sd_train, meta = H4.training_sd()
    assert sd_train is not None, f"G5 training-set component unavailable: {meta}"
    scen = {"both components (headline)": (sd_est, sd_train), "estimation only": (sd_est, 0.0),
            "neither (C6 reproduction)": (0.0, 0.0)}
    out = {"entry": "RESULTS.md Entry 106 (H6)", "estimation_sd": sd_est, "training_sd": sd_train,
           "grid": [float(GRID[0]), float(GRID[-1]), 0.01], "n_rep": H4.NREP, "seed": SEED, "scenarios": {}}
    for name, (se_, st_) in scen.items():
        cov = coverage_by_c(simulate(se_, st_))
        worst = cov.min(0)
        ok = np.where(worst >= 0.99)[0]
        c = float(GRID[ok[0]]) if len(ok) else None
        out["scenarios"][name] = {"c": c, "worst_coverage_at_c": float(worst[ok[0]]) if len(ok) else None,
                                  "worst_coverage_at_1": float(worst[0])}
        print(f"[h6] {name:28s} worst coverage at c=1: {worst[0]:.4f};  smallest c for >= 0.99: {c}", flush=True)

    c = out["scenarios"]["both components (headline)"]["c"]
    led = json.load(open(H4.LEDGER))
    A = np.array(led["primary"]["per_adapter_A"]); B = np.array(led["primary"]["per_adapter_B"])
    R = json.load(open(H4.LEDGER))["denominators"]["R_real"]
    k = len(A); tc = stats.t.ppf(0.995, k - 1)
    UA = A.mean() + c * tc * A.std(ddof=1) / np.sqrt(k); UB = B.mean() + c * tc * B.std(ddof=1) / np.sqrt(k)
    U1 = max(A.mean() + tc * A.std(ddof=1) / np.sqrt(k), B.mean() + tc * B.std(ddof=1) / np.sqrt(k))
    out["primary_d200"] = {"c": c, "U_A": float(UA), "U_B": float(UB), "U_device": float(max(UA, UB)),
                           "lambda_U_calibrated_pct": float(100 * max(UA, UB) / R),
                           "lambda_U_nominal_pct": float(100 * U1 / R), "R_real": R}
    json.dump(out, open(OUT, "w"), indent=2)
    p = out["primary_d200"]
    print(f"[h6] headline: nominal {p['lambda_U_nominal_pct']:.4f} %  ->  calibrated (c = {c}) "
          f"{p['lambda_U_calibrated_pct']:.4f} %", flush=True)


if __name__ == "__main__":
    main()
