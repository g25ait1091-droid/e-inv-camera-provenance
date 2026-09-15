"""F4 (RESULTS.md Entry 20) — unmarked local-decode control arms versus the marked ladder arms and
versus v1's twelve primary A adapters, on the natural paired contrast rho(->K_A) - rho(->K_B).

  (i)  two-sample Welch t, one-sided H1: marked (five arms of Entry 20) > nomark (three arms)
  (ii) one-sample t of the three nomark means against v1's A-arm mean (-9.5699e-06), and the
       z of the nomark mean against v1's A-arm distribution (sd 4.842e-05 / sqrt(3))
  also: the same for the five marked arms against v1, and the a1 (effectively unmarked) arm alone.
Reads out/t1/summary.json and out/t1/summary_nomark.json; writes out/t1/f4_stats.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, numpy as np
from scipy import stats

T1 = os.path.join(EINV.V2, 'out', 't1')
S = json.load(open(os.path.join(T1, "summary.json")))["arms"]; N = json.load(open(os.path.join(T1, "summary_nomark.json")))["arms"]
V1_MEAN, V1_SD = -9.5699e-06, 4.842e-05
marked = np.array([S[a]["natural_paired_KA_minus_KB"] for a in ("mark_rand_a3_s0", "mark_rand_a3_s1", "mark_rand_a3_s2", "mark_rand_a12_s0", "mark_lowmid_a12_s0")])
nomark = np.array([N[a]["natural_paired_KA_minus_KB"] for a in ("nomark_s0", "nomark_s1", "nomark_s2")])
a1 = S["mark_rand_a1_s0"]["natural_paired_KA_minus_KB"]
t_i, p_i = stats.ttest_ind(marked, nomark, equal_var=False, alternative="greater")
t_ii, p_ii = stats.ttest_1samp(nomark, V1_MEAN, alternative="greater")
z_ii = (nomark.mean() - V1_MEAN) / (V1_SD / np.sqrt(3)); pz_ii = 1 - stats.norm.cdf(z_ii)
t_m, p_m = stats.ttest_1samp(marked, V1_MEAN, alternative="greater")
z_m = (marked.mean() - V1_MEAN) / (V1_SD / np.sqrt(5)); pz_m = 1 - stats.norm.cdf(z_m)
res = {"nomark_values": nomark.tolist(), "nomark_mean": float(nomark.mean()), "nomark_se": float(nomark.std(ddof=1) / np.sqrt(3)),
       "marked_values": marked.tolist(), "marked_mean": float(marked.mean()), "a1_unmarked": a1,
       "v1_reference": {"mean": V1_MEAN, "sd": V1_SD, "k": 12},
       "(i)_marked_gt_nomark_welch": {"t": float(t_i), "p_one_sided": float(p_i)},
       "(ii)_nomark_vs_v1": {"t_1samp": float(t_ii), "p_one_sided": float(p_ii), "z_vs_v1_sd": float(z_ii), "p_z": float(pz_ii)},
       "marked_vs_v1": {"t_1samp": float(t_m), "p_one_sided": float(p_m), "z_vs_v1_sd": float(z_m), "p_z": float(pz_m)},
       "nomark_per_arm": {a: N[a] for a in N}}
json.dump(res, open(os.path.join(T1, "f4_stats.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in res.items() if k != "nomark_per_arm"}, indent=1))
