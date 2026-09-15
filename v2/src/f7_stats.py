"""F7 (RESULTS.md Entry 35) — the symmetric statistic in the v2 environment.

theta_A = mean over A-trained unmarked adapters of rho(->K_A) - rho(->K_B)
theta_B = mean over B-trained unmarked adapters of rho(->K_B) - rho(->K_A)   (= -natural contrast)
theta_sym = (theta_A + theta_B)/2 (device-specific interaction; main effects cancel)
additive  = (theta_A - theta_B)/2 (what every adapter shares, e.g. a training-stack main effect)
Primary: A = nomark_s0..s2 (3) vs B = nomarkB_s0..s2 (3). Secondary: A = nomark + colab (6).
Welch SE, one-sided t and p on theta_sym, plug-in one-sided 99 % upper limit, lambda against R_real.
Writes out/t1/f7_stats.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, numpy as np
from scipy import stats
T1 = os.path.join(EINV.V2, 'out', 't1'); R_REAL = 0.0356703416571125
N = json.load(open(os.path.join(T1, "summary_nomark.json")))["arms"]
C = json.load(open(os.path.join(T1, "summary_colab.json")))["arms"]
B = json.load(open(os.path.join(T1, "summary_nomarkB.json")))["arms"]
nat = lambda S, a: S[a]["natural_paired_KA_minus_KB"]
A3 = np.array([nat(N, f"nomark_s{i}") for i in range(3)])
A6 = np.concatenate([A3, [nat(C, f"colab_s{i}") for i in range(3)]])
Bown = np.array([-nat(B, f"nomarkB_s{i}") for i in range(3)])       # own-body contrast of B adapters

def sym(a, b):
    thA, thB = a.mean(), b.mean(); vA, vB = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = 0.5 * np.sqrt(vA + vB); df = (vA + vB) ** 2 / (vA ** 2 / (len(a) - 1) + vB ** 2 / (len(b) - 1))
    th = 0.5 * (thA + thB); t = th / se
    return {"n_A": len(a), "n_B": len(b), "theta_A": float(thA), "theta_B": float(thB), "theta_sym": float(th),
            "additive_part": float(0.5 * (thA - thB)), "adapter_SE": float(se), "welch_df": float(df), "t": float(t),
            "p_one_sided": float(1 - stats.t.cdf(t, df)), "lambda_sym_pct": float(100 * th / R_REAL),
            "lambda_U_plugin_pct": float(100 * (th + stats.t.ppf(0.99, df) * se) / R_REAL),
            "additive_over_R_pct": float(100 * 0.5 * (thA - thB) / R_REAL)}

out = {"A_nomark": A3.tolist(), "A_colab": A6[3:].tolist(), "B_nomarkB_natural_KA_minus_KB": [nat(B, f"nomarkB_s{i}") for i in range(3)],
       "B_own_contrast": Bown.tolist(), "primary_3v3": sym(A3, Bown), "secondary_6v3": sym(A6, Bown),
       "v1_reference_k12": {"theta_A": -9.5699e-06, "theta_B": 1.8769e-05, "theta_sym": 4.5996e-06}}
json.dump(out, open(os.path.join(T1, "f7_stats.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
