"""Tier 2 — adapter-level summary of the learned detector (Entry 09 registration).

Reads out/t2_learned.json (per-arm mean score A-minus-B over 500 generations) and computes the
same decomposition used for the other detector families:
  contrast toward own body   c = +score (A arms), -score (B arms)
  theta_A, theta_B           means of c over the three seeds of each arm
  theta_sym                  (theta_A + theta_B)/2  — the fingerprint main effect cancels
  additive part              (mean score_A_arms + mean score_B_arms)/2 — what every arm shares
  adapter-level SE of theta_sym = (1/2) sqrt(var_A/3 + var_B/3), Welch df; one-sided t, p
  lambda_sym = theta_sym / real paired contrast (held-out A mean minus B mean of the score)
  plug-in one-sided upper limit at 99 %: (theta_sym + t_{0.99,df} SE) / R
Writes out/t2_learned_summary.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, os, numpy as np
from scipy import stats

V2 = EINV.V2
d = json.load(open(os.path.join(V2, "out", "t2_learned.json")))
arms = d["arms"]
A = [arms[k]["mean_score_AminusB"] for k in ("A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16")]
B = [arms[k]["mean_score_AminusB"] for k in ("B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16")]
base = arms["base"]["mean_score_AminusB"]
R = d["real_H_mean_score_A"] - d["real_H_mean_score_B"]
cA = np.array(A); cB = -np.array(B)                      # contrast toward own body
thA, thB = cA.mean(), cB.mean()
th_sym = 0.5 * (thA + thB)
additive = 0.5 * (np.mean(A) + np.mean(B))
vA, vB = cA.var(ddof=1) / 3, cB.var(ddof=1) / 3
se = 0.5 * np.sqrt(vA + vB)
df = (vA + vB) ** 2 / (vA ** 2 / 2 + vB ** 2 / 2)          # Welch
t = th_sym / se
p = 1 - stats.t.cdf(t, df)
UL = th_sym + stats.t.ppf(0.99, df) * se
# image-level SE of theta_sym for reference (not the unit of inference)
se_img = 0.5 * np.sqrt(sum(arms[k]["se"] ** 2 for k in arms if k != "base")) / 3
res = {
 "real_H_auc": d["real_H_auc_A_vs_B"], "real_paired_contrast_R": R,
 "base_score": base, "A_arm_scores": A, "B_arm_scores": B,
 "theta_A_toward_own": thA, "theta_B_toward_own": thB, "theta_sym": th_sym,
 "additive_part": additive, "additive_over_R_pct": 100 * abs(additive) / R,
 "adapter_level_SE": se, "welch_df": df, "t": t, "p_one_sided": p,
 "image_level_SE_reference": se_img,
 "lambda_sym_pct": 100 * th_sym / R, "lambda_U_plugin_pct": 100 * UL / R,
 "arms_shift_vs_base": {"A_minus_base": float(np.mean(A) - base), "B_minus_base": float(np.mean(B) - base),
                        "all_six_minus_base": float(0.5 * (np.mean(A) + np.mean(B)) - base)},
 "registered_criterion_both_arms_gt_3SE_toward_own": bool(thA > 3 * np.sqrt(vA) and thB > 3 * np.sqrt(vB)),
}
json.dump(res, open(os.path.join(V2, "out", "t2_learned_summary.json"), "w"), indent=1)
for k, v in res.items(): print(f"{k:48s} {v}")
