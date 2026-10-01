"""Chain 12 readings (RESULTS.md Entries 76 and 77): G2 second-environment replication, G1 inversion.

Reads out/t1/summary_nomarkrep.json and out/t1/summary_inv16k.json, forms the own-minus-other contrast
per adapter, and applies the pre-registered decision rules. Writes out/g_chain12.json.
"""
import os, sys, json
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

T1 = os.path.join(EINV.V2, "out", "t1"); OUT = os.path.join(EINV.V2, "out", "g_chain12.json")
R_REAL = 0.0356703416571125
UNSUP_16K = 5.710464675763419e-05      # Entry 68, six adapters per body
F7_LEAN = 2.8e-05                      # Entry 36, the lean being replicated

def sym(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    vA, vB = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(B) - 1))
    return {"theta_sym": float(th), "welch_se": float(se), "welch_df": float(df), "t": float(th / se),
            "one_sided_p_gt0": float(stats.t.sf(th / se, df)), "one_sided_p_lt0": float(stats.t.cdf(th / se, df)),
            "upper95": float(th + stats.t.ppf(0.95, df) * se), "lambda_pct": float(100 * th / R_REAL),
            "per_adapter_A": A.tolist(), "per_adapter_B": B.tolist()}

def own(summary, tags_A, tags_B):
    a = json.load(open(summary))["arms"]
    return ([a[t]["natural_paired_KA_minus_KB"] for t in tags_A],
            [-a[t]["natural_paired_KA_minus_KB"] for t in tags_B])

res = {}
p2 = os.path.join(T1, "summary_nomarkrep.json")
if os.path.exists(p2):
    A, B = own(p2, [f"nomark_s{s}" for s in (3, 4, 5)], [f"nomarkB_s{s}" for s in (3, 4, 5)])
    st = sym(A, B)
    if st["theta_sym"] > 0 and st["one_sided_p_gt0"] < 0.05:
        st["reading"] = "the second environment's lean replicates"
    elif st["theta_sym"] <= 0 or st["upper95"] < 0.5 * F7_LEAN:
        st["reading"] = "the lean does not replicate"
    else:
        st["reading"] = "inconclusive"
    res["G2"] = st
    print(f"G2  theta_sym {st['theta_sym']:+.3e} ({st['lambda_pct']:+.4f}%)  t {st['t']:.2f}  "
          f"p {st['one_sided_p_gt0']:.4f}  ->  {st['reading']}")

p1 = os.path.join(T1, "summary_inv16k.json")
if os.path.exists(p1):
    A, B = own(p1, [f"inv16k_A_s{s}" for s in range(3)], [f"inv16k_B_s{s}" for s in range(3)])
    st = sym(A, B)
    inv = json.load(open(os.path.join(T1, "invert.json")))
    st["stored_inverted_contrast"] = {r: v["R_inv_fraction_of_uninjected"] for r, v in inv["bodies"].items()}
    if st["theta_sym"] < 0 and st["one_sided_p_lt0"] < 0.05:
        st["reading"] = "the 16000-step signal follows the fingerprint's sign"
    elif st["theta_sym"] > 0 and abs(st["theta_sym"] - UNSUP_16K) <= 2 * st["welch_se"]:
        st["reading"] = "the 16000-step signal does not follow the fingerprint"
    else:
        st["reading"] = "inconclusive"
    res["G1"] = st
    print(f"G1  theta_sym {st['theta_sym']:+.3e} ({st['lambda_pct']:+.4f}%)  t {st['t']:.2f}  "
          f"p(<0) {st['one_sided_p_lt0']:.4f}  ->  {st['reading']}")

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", OUT)
