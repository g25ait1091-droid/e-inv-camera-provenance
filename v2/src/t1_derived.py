"""Tier 1 — derived quantities from out/t1/summary.json and measure_rows.csv (Entry 20).

Registered quantities are in summary.json (t1_measure.py). This adds, clearly labelled as
secondary: the never-injected offset of the M - M' and M_L - M' contrasts (mean over base,
A_raw_s0, B_raw_s0), offset-corrected contrasts and lambda_mark per arm, the alpha = 3 cluster t
against that offset, image-level t per arm, the chance probability of each decoy rank
(P(rank <= r) = r/31 under exchangeability), and the natural K_A - K_B contrast on the mark arms
with its SE over adapters. Writes out/t1/summary_derived.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, numpy as np, pandas as pd
from scipy import stats

T1 = os.path.join(EINV.V2, 'out', 't1')
S = json.load(open(os.path.join(T1, "summary.json"))); A = S["arms"]
d = pd.read_csv(os.path.join(T1, "measure_rows.csv"))
NEVER = ["base", "A_raw_s0_r16", "B_raw_s0_r16"]
MARK = ["mark_rand_a1_s0", "mark_rand_a3_s0", "mark_rand_a3_s1", "mark_rand_a3_s2", "mark_rand_a12_s0", "mark_lowmid_a12_s0"]
offM = np.array([A[a]["contrast_M"] for a in NEVER]); offL = np.array([A[a]["contrast_ML"] for a in NEVER])
off = {"M_minus_Mp_never_injected": {"per_arm": offM.tolist(), "mean": float(offM.mean()), "se_over_arms": float(offM.std(ddof=1) / np.sqrt(3)),
                                     "se_pooled_images": float(np.sqrt(sum(A[a]["contrast_M_se"] ** 2 for a in NEVER)) / 3)},
       "ML_minus_Mp_never_injected": {"per_arm": offL.tolist(), "mean": float(offL.mean()), "se_over_arms": float(offL.std(ddof=1) / np.sqrt(3))},
       "rho_Mp_all_arms": {a: A[a]["rho_Mp"] for a in A}}
arms = {}
for a in MARK + NEVER:
    e = A[a]; g = d[d.arm == a]
    c = (g.rho_M - g.rho_Mp).values; cL = (g.rho_ML - g.rho_Mp).values
    rec = {"contrast_M": e["contrast_M"], "t_image_level_M": float(c.mean() / (c.std(ddof=1) / np.sqrt(len(c)))),
           "contrast_M_minus_offset": e["contrast_M"] - offM.mean(),
           "contrast_ML": e["contrast_ML"], "t_image_level_ML": float(cL.mean() / (cL.std(ddof=1) / np.sqrt(len(cL)))),
           "contrast_ML_minus_offset": e["contrast_ML"] - offL.mean(),
           "decoy_rank": e["decoy_rank_of_true_M"], "P_rank_le_r_under_exchangeability": e["decoy_rank_of_true_M"] / 31,
           "rho_M_minus_decoy_mean": e["rho_M"] - e["decoy_mean"], "natural_KA_minus_KB": e["natural_paired_KA_minus_KB"]}
    if e.get("R_mark") and e["R_mark"] > 1e-3:
        R = e["R_mark"]; cc = e["contrast_ML"] if a.startswith("mark_lowmid") else e["contrast_M"]
        oo = offL.mean() if a.startswith("mark_lowmid") else offM.mean()
        rec.update(R_mark=R, lambda_mark_pct=100 * cc / R, lambda_mark_offset_corrected_pct=100 * (cc - oo) / R,
                   R_mark_over_R_real=R / 0.0356703)
    arms[a] = rec
a3 = np.array([A[f"mark_rand_a3_s{i}"]["contrast_M"] for i in range(3)]); m, sd = a3.mean(), a3.std(ddof=1)
t_off = (m - offM.mean()) / (sd / np.sqrt(3)); R3 = A["mark_rand_a3_s0"]["R_mark"]
a3c = {"mean_minus_offset": float(m - offM.mean()), "t_vs_offset_df2": float(t_off), "p_one_sided_vs_offset": float(1 - stats.t.cdf(t_off, 2)),
       "lambda_offset_corrected_pct": float(100 * (m - offM.mean()) / R3),
       "lambda_offset_corrected_U_pct": float(100 * (m - offM.mean() + stats.t.ppf(0.995, 2) * sd / np.sqrt(3)) / R3),
       "decoy_ranks": [A[f"mark_rand_a3_s{i}"]["decoy_rank_of_true_M"] for i in range(3)]}
nat = np.array([A[a]["natural_paired_KA_minus_KB"] for a in MARK]); nat5 = np.array([A[a]["natural_paired_KA_minus_KB"] for a in MARK[1:]])
natural = {"mark_arms_all6": {"values": nat.tolist(), "mean": float(nat.mean()), "se_over_adapters": float(nat.std(ddof=1) / np.sqrt(6)),
                              "t": float(nat.mean() / (nat.std(ddof=1) / np.sqrt(6)))},
           "marked_only_5": {"values": nat5.tolist(), "mean": float(nat5.mean()), "se_over_adapters": float(nat5.std(ddof=1) / np.sqrt(5))},
           "a1_effectively_unmarked": A["mark_rand_a1_s0"]["natural_paired_KA_minus_KB"],
           "never_injected": {a: A[a]["natural_paired_KA_minus_KB"] for a in NEVER},
           "v1_reference": {"theta_A_mean_k12": -9.5699e-06, "theta_A_sd_k12": 4.842e-05, "note": "FINAL_LEDGER primary; Colab decode, uniform caption",
                            "P_all6_positive_under_v1_A_distribution": float((1 - stats.norm.cdf(0, -9.5699e-06, 4.842e-05)) ** 6),
                            "P_all5_marked_positive_under_v1_A_distribution": float((1 - stats.norm.cdf(0, -9.5699e-06, 4.842e-05)) ** 5)}}
# transmission coefficient reading: output contrast per unit input contrast, offset-corrected, in %
trans = {a: arms[a].get("lambda_mark_offset_corrected_pct") for a in MARK if arms[a].get("lambda_mark_offset_corrected_pct") is not None}
out = {"offsets": off, "arms": arms, "alpha3_cluster_vs_offset": a3c, "natural_contrast_on_mark_arms": natural,
       "transmission_pct_offset_corrected": trans, "natural_limit_lambda_U_pct_v1": 0.15072}
json.dump(out, open(os.path.join(T1, "summary_derived.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
for a, r in arms.items(): print(a, {k: (round(v, 6) if isinstance(v, float) else v) for k, v in r.items()})
