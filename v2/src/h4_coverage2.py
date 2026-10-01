"""H4 (RESULTS.md Entry 88) - coverage once the two omitted variance components are in.

Entry 58's simulation (C6) models the adapter spread and nothing else. Two sources of variation that the
published limit treats as zero are now measurable:

  estimation     the fingerprint is estimated from a finite set of photographs. H2 resampled that set and
                 found it contributes 43 % of the variance in the statistic. It enters as a shift shared by
                 both arms of a replication, because the same K scores both.
  training set   each body contributes one training set, so the limit generalises over adapter seeds and
                 not over training sets. It enters as a per-arm offset. G5 gives a point estimate once its
                 six adapters are measured; until then, and in any case for sensitivity, the simulation
                 sweeps a grid of plausible sizes expressed in units of the observed adapter SD.

Everything else - the scenarios, the constructions, the reading threshold - is C6 unchanged, so the rows
are comparable to the ones already reported. CPU only. Writes out/h4_coverage2.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2
OUT = os.path.join(V2, "out", "h4_coverage2.json")
LEDGER = os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')
H2 = os.path.join(V2, "out", "h2_estimation_error.json")
ALT = os.path.join(V2, "out", "t1", "summary_alt.json")
R_REAL = 0.0356703416571125
K = 12; NREP = int(os.environ.get("H4_REP", "4000")); SEED = 88041

led = json.load(open(LEDGER))["primary"]
A0 = np.array(led["per_adapter_A"]); B0 = np.array(led["per_adapter_B"])
sdA, sdB = A0.std(ddof=1), B0.std(ddof=1)
sd_bar = 0.5 * (sdA + sdB)
m_obs = 0.5 * (A0.mean() - B0.mean())
cA, cB = A0 - A0.mean(), B0 - B0.mean()
tc_max = stats.t.ppf(0.995, K - 1)


def estimation_sd():
    """Absolute SD that fingerprint estimation adds to theta_sym, from H2's bootstrap replicates."""
    h2 = json.load(open(H2))
    th = np.array([r["theta_sym"] for r in h2["replicates"]])
    return float(th.std(ddof=1)), h2["n_replicates"]


def training_sd():
    """Per-arm offset SD between two disjoint training sets, from G5 - None until G5 is measured."""
    if not os.path.exists(ALT):
        return None, "G5 not yet measured"
    arms = json.load(open(ALT))["arms"]
    need = [f"alt_{b}_s{s}" for s in (0, 1, 2) for b in ("A", "B")]
    if not all(a in arms for a in need):
        return None, f"G5 incomplete ({len(arms)} of 6 arms)"
    nat = lambda a: arms[a]["natural_paired_KA_minus_KB"]
    altA = np.array([nat(f"alt_A_s{s}") for s in (0, 1, 2)])
    altB = np.array([-nat(f"alt_B_s{s}") for s in (0, 1, 2)])
    # the training-set effect is the shift between the two training sets of the same body, after removing
    # the adapter-level mean; with one contrast per body this is a two-point estimate and is labelled so
    dA = float(altA.mean() - A0.mean()); dB = float(altB.mean() - B0.mean())
    se_A = np.sqrt(altA.var(ddof=1) / 3 + A0.var(ddof=1) / len(A0))
    se_B = np.sqrt(altB.var(ddof=1) / 3 + B0.var(ddof=1) / len(B0))
    var = max(0.0, 0.5 * (dA ** 2 + dB ** 2) - 0.5 * (se_A ** 2 + se_B ** 2))  # sampling-corrected
    return float(np.sqrt(var)), {"shift_A": dA, "shift_B": dB,
                                 "se_A": float(se_A), "se_B": float(se_B),
                                 "note": "two contrasts only; a bound, not a precise variance"}


def draw(spread, rng, n):
    if spread == "normal":
        return rng.normal(0, sdA, (n, K)), rng.normal(0, sdB, (n, K))
    if spread == "t3":
        s = np.sqrt(3.0)
        return rng.standard_t(3, (n, K)) * sdA / s, rng.standard_t(3, (n, K)) * sdB / s
    if spread == "resample":
        return rng.choice(cA, (n, K), replace=True), rng.choice(cB, (n, K), replace=True)
    raise ValueError(spread)


def limits(A, B):
    mA, mB = A.mean(1), B.mean(1); sA, sB = A.std(1, ddof=1), B.std(1, ddof=1)
    U = np.maximum(mA + tc_max * sA / np.sqrt(K), mB + tc_max * sB / np.sqrt(K))
    vA, vB = sA ** 2 / K, sB ** 2 / K; se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (K - 1) + vB ** 2 / (K - 1))
    L = 0.5 * (mA + mB) + stats.t.ppf(0.99, df) * se
    return U, L


def main():
    t0 = time.time()
    sd_est, n_rep_h2 = estimation_sd()
    sd_train, train_meta = training_sd()
    print(f"[h4] estimation SD {sd_est:.3e} ({sd_est/sd_bar:.2f} x adapter SD, from {n_rep_h2} replicates)",
          flush=True)
    print(f"[h4] training-set SD {sd_train if sd_train is None else f'{sd_train:.3e}'} ({train_meta})",
          flush=True)

    train_grid = [("0", 0.0), ("0.5xSD", 0.5 * sd_bar), ("1xSD", sd_bar), ("2xSD", 2 * sd_bar)]
    if sd_train is not None:
        train_grid.append(("G5", sd_train))

    master = np.random.default_rng(SEED)
    spreads = ["normal", "t3", "resample"]
    ms = [("observed", m_obs), ("5e-5", 5e-5), ("1e-4", 1e-4)]
    iis = [0.0, 1e-5, 3e-5]
    rows = []
    for sp in spreads:
        for mlab, m in ms:
            for i in iis:
                for elab, use_est in (("off", False), ("on", True)):
                    for tlab, sdt in train_grid:
                        rng = np.random.default_rng(master.integers(2 ** 62))
                        eA, eB = draw(sp, rng, NREP)
                        A = i + m + eA
                        B = i - m + eB
                        if sdt:                                   # one offset per arm per replication
                            A = A + rng.normal(0, sdt, (NREP, 1))
                            B = B + rng.normal(0, sdt, (NREP, 1))
                        if use_est:                               # one shift shared by both arms
                            sh = rng.normal(0, sd_est, (NREP, 1))
                            A = A + sh; B = B + sh
                        U, L = limits(A, B)
                        rows.append({"spread": sp, "m": mlab, "i": i, "estimation": elab, "training": tlab,
                                     "coverage_maxarm": float(np.mean(U >= i)),
                                     "coverage_sym": float(np.mean(L >= i))})
        print(f"[h4] {sp} done ({(time.time()-t0)/60:.1f} min)", flush=True)

    def worst(sel):
        r = [x for x in rows if sel(x)]
        return {"min_coverage_sym": min(x["coverage_sym"] for x in r),
                "min_coverage_maxarm": min(x["coverage_maxarm"] for x in r),
                "n_cells": len(r)}

    base = worst(lambda x: x["estimation"] == "off" and x["training"] == "0")
    est_only = worst(lambda x: x["estimation"] == "on" and x["training"] == "0")
    both_1sd = worst(lambda x: x["estimation"] == "on" and x["training"] == "1xSD")
    headline = worst(lambda x: x["estimation"] == "on" and x["training"] == ("G5" if sd_train is not None else "1xSD"))
    print(f"[h4] min symmetric coverage: C6 reproduction {base['min_coverage_sym']:.4f}; "
          f"+estimation {est_only['min_coverage_sym']:.4f}; +both at 1xSD {both_1sd['min_coverage_sym']:.4f}",
          flush=True)

    if headline["min_coverage_sym"] >= 0.98:
        reading = ("the construction still covers with both components included; reported as adequate with "
                   "the components in")
    else:
        reading = ("symmetric coverage falls below 0.98 once the components are included; the max-arm "
                   "limit becomes the primary construction and the symmetric statistic is secondary")
    res = {"entry": "RESULTS.md Entry 88 (H4)", "k_per_arm": K, "n_rep": NREP, "seed": SEED,
           "adapter_sd": {"A": float(sdA), "B": float(sdB), "mean": float(sd_bar)},
           "estimation_sd": sd_est, "estimation_sd_in_adapter_sd": sd_est / sd_bar,
           "training_sd": sd_train, "training_detail": train_meta,
           "train_grid": {k: v for k, v in train_grid},
           "rows": rows, "worst_case": {"c6_reproduction": base, "estimation_only": est_only,
                                        "both_at_1sd": both_1sd, "headline": headline},
           "reading": reading, "runtime_min": (time.time() - t0) / 60}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[h4] headline min symmetric coverage {headline['min_coverage_sym']:.4f}", flush=True)
    print(f"[h4] READING: {reading}", flush=True)


if __name__ == "__main__":
    main()
