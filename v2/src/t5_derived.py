"""E-PROMPT (Entry 11) — derived quantities from out/t5/summary.json and out/t5/rows.csv.

Adapter-level SE and t of theta_sym over the three seeds per arm; additive part; the base
model's K_B - K_A main effect under the diverse bank against the uniform bank (out/t2_summary.json,
same six adapters, same statistic on the local instrument); per-arm image-level t; lambda_sym
against R_real and the n = 3 plug-in limit; and the registered secondary descriptive, adaptation
strength = mean absolute pixel difference to the base arm at matched seeds, for both banks.
Writes out/t5/summary_derived.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, glob, json, numpy as np
from PIL import Image
from scipy import stats

V2 = EINV.V2; T5 = os.path.join(V2, "out", "t5")
S = json.load(open(os.path.join(T5, "summary.json"))); A = S["arms"]
U = json.load(open(os.path.join(V2, "out", "t2_summary.json")))["ncc"]["arms"]
R_REAL = 0.0356703416571125
arms_A = ["A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16"]; arms_B = ["B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16"]
cA = np.array([A[a]["paired_contrast"] for a in arms_A]); cB = np.array([A[a]["paired_contrast"] for a in arms_B])
th_sym = 0.5 * (cA.mean() + cB.mean()); vA, vB = cA.var(ddof=1) / 3, cB.var(ddof=1) / 3
se = 0.5 * np.sqrt(vA + vB); df = (vA + vB) ** 2 / (vA ** 2 / 2 + vB ** 2 / 2); t = th_sym / se
additive = 0.5 * (cA.mean() - cB.mean())             # (b_A - b_B)/2 in own-body-contrast convention
UL = th_sym + stats.t.ppf(0.995, df) * se
uA = np.array([U[a]["paired_contrast"] for a in arms_A]); uB = np.array([U[a]["paired_contrast"] for a in arms_B])
u_sym = 0.5 * (uA.mean() + uB.mean())
res = {"diverse_bank": {"per_arm_own_contrast": {a: A[a]["paired_contrast"] for a in arms_A + arms_B},
                        "per_arm_image_t": {a: A[a]["paired_contrast"] / A[a]["se"] for a in arms_A + arms_B},
                        "theta_A": float(cA.mean()), "theta_B": float(cB.mean()), "theta_sym": float(th_sym),
                        "additive_part": float(additive), "adapter_level_SE": float(se), "welch_df": float(df), "t": float(t),
                        "lambda_sym_pct": float(100 * th_sym / R_REAL), "lambda_U_plugin_n3_pct": float(100 * UL / R_REAL),
                        "base_KB_minus_KA": A["base"]["mean_KB"] - A["base"]["mean_KA"], "max_abs_image_t": float(max(abs(A[a]["paired_contrast"] / A[a]["se"]) for a in arms_A + arms_B))},
       "uniform_bank_same_adapters_entry10": {"per_arm_own_contrast": {a: U[a]["paired_contrast"] for a in arms_A + arms_B},
                                              "theta_A": float(uA.mean()), "theta_B": float(uB.mean()), "theta_sym": float(u_sym),
                                              "base_KB_minus_KA": U["base"]["mean_B"] - U["base"]["mean_A"]},
       "registered_reading_holds": bool(abs(th_sym) <= 2 * se and all(abs(A[a]["paired_contrast"]) / A[a]["se"] < 3 or
                                        (A[a]["paired_contrast"] < 0) for a in arms_A + arms_B))}
# adaptation strength: mean |arm_j - base_j| over 500 matched seeds, both banks (descriptive, registered)
def mad(bank_dir, arm, n=500):
    vals = []
    for j in range(n):
        fa = os.path.join(bank_dir, arm, f"{j:05d}.png"); fb = os.path.join(bank_dir, "base", f"{j:05d}.png")
        if not (os.path.exists(fa) and os.path.exists(fb)): continue
        a = np.asarray(Image.open(fa).convert("L"), np.float32); b = np.asarray(Image.open(fb).convert("L"), np.float32)
        vals.append(float(np.abs(a - b).mean()))
    return float(np.mean(vals)), len(vals)
strength = {}
for bank, d in (("diverse", os.path.join(T5, "gens")), ("uniform", os.path.join(V2, "data", "gens"))):
    strength[bank] = {a: mad(d, a) for a in arms_A + arms_B}
res["adaptation_strength_mean_abs_diff_to_base_gray_levels"] = strength
json.dump(res, open(os.path.join(T5, "summary_derived.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
