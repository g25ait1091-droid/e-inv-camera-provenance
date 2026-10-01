"""G5 scoring (RESULTS.md Entry 78) - does the primary limit generalize over training sets?

Every one of the 24 primary adapters saw the same 50 photographs of its body, so the primary limit
generalizes over adapter seeds and not over training sets. G5 trains three adapters per body on a second,
fully disjoint 50-image training set and scores them with the statistic unchanged.

Applies the Entry 78 readings verbatim: theta_sym with a one-sided Welch p, the one-sided 99 % limit
against the 0.30 % threshold, and the difference from the primary theta_sym with its own standard error.
CPU only, from the archived per-adapter summary. Writes out/g5_alt_training.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1")
OUT = os.path.join(V2, "out", "g5_alt_training.json")
LEDGER = os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')
R_REAL = 0.0356703416571125
SEEDS = (0, 1, 2)


def sym(A, B):
    vA, vB = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(B) - 1))
    tc = stats.t.ppf(0.995, len(A) - 1)
    U = max(A.mean() + tc * A.std(ddof=1) / np.sqrt(len(A)),
            B.mean() + tc * B.std(ddof=1) / np.sqrt(len(B)))
    return {"theta_A": float(A.mean()), "theta_B": float(B.mean()), "theta_sym": float(th),
            "welch_se": float(se), "welch_df": float(df), "t": float(th / se),
            "one_sided_p": float(stats.t.sf(th / se, df)),
            "limit99": float(th + stats.t.ppf(0.99, df) * se),
            "limit99_pct": float(100 * (th + stats.t.ppf(0.99, df) * se) / R_REAL),
            "theta_sym_pct": float(100 * th / R_REAL),
            "maxarm_U": float(U), "maxarm_U_pct": float(100 * U / R_REAL),
            "per_adapter_A": A.tolist(), "per_adapter_B": B.tolist()}


def main():
    arms = json.load(open(os.path.join(T1, "summary_alt.json")))["arms"]
    need = [f"alt_{b}_s{s}" for s in SEEDS for b in ("A", "B")]
    missing = [a for a in need if a not in arms]
    assert not missing, f"G5 incomplete: {missing}"
    nat = lambda a: arms[a]["natural_paired_KA_minus_KB"]
    A = np.array([nat(f"alt_A_s{s}") for s in SEEDS])
    B = np.array([-nat(f"alt_B_s{s}") for s in SEEDS])
    st = sym(A, B)
    st["images_per_adapter"] = {a: arms[a]["n"] for a in need}

    led = json.load(open(LEDGER))["primary"]
    pA = np.array(led["per_adapter_A"]); pB = np.array(led["per_adapter_B"])
    prim = sym(pA, pB)
    # difference between training sets, with its own SE from the two independent symmetric estimates
    diff = st["theta_sym"] - prim["theta_sym"]
    se_diff = float(np.sqrt(st["welch_se"] ** 2 + prim["welch_se"] ** 2))
    comparison = {"primary_theta_sym": prim["theta_sym"], "alt_theta_sym": st["theta_sym"],
                  "difference": float(diff), "se_difference": se_diff,
                  "z": float(diff / se_diff),
                  "two_sided_p_normal": float(2 * stats.norm.sf(abs(diff / se_diff))),
                  "note": "three adapters per arm against twelve; the SE is dominated by the G5 arms"}

    if st["theta_sym"] > 0 and st["one_sided_p"] < 0.05:
        reading = "a second training set shows transfer"
    elif st["limit99_pct"] <= 0.30:
        reading = "no detectable transfer with a second training set"
    else:
        reading = (f"neither branch: theta_sym not resolved and the limit is "
                   f"{st['limit99_pct']:.4f} %, above the 0.30 % threshold")
    res = {"entry": "RESULTS.md Entry 78 (G5)", "n_per_arm": len(A), "R_real": R_REAL,
           "alt": st, "primary": {k: prim[k] for k in ("theta_sym", "welch_se", "limit99_pct",
                                                       "maxarm_U_pct", "theta_sym_pct")},
           "comparison": comparison, "reading": reading}
    json.dump(res, open(OUT, "w"), indent=2)
    print(f"[g5] per-adapter A {np.round(A, 7).tolist()}")
    print(f"[g5] per-adapter B {np.round(B, 7).tolist()}")
    print(f"[g5] theta_A {st['theta_A']:+.3e}  theta_B {st['theta_B']:+.3e}")
    print(f"[g5] theta_sym {st['theta_sym']:+.3e} ({st['theta_sym_pct']:+.4f} % of R_real), "
          f"Welch t {st['t']:+.3f}, one-sided p {st['one_sided_p']:.4f}")
    print(f"[g5] one-sided 99 % limit {st['limit99_pct']:.4f} % (threshold 0.30 %); "
          f"max-arm {st['maxarm_U_pct']:.4f} %")
    print(f"[g5] vs primary theta_sym {prim['theta_sym']:+.3e}: difference {diff:+.3e} "
          f"+- {se_diff:.3e} (z {comparison['z']:+.2f}, p {comparison['two_sided_p_normal']:.3f})")
    print(f"[g5] READING: {reading}")


if __name__ == "__main__":
    main()
