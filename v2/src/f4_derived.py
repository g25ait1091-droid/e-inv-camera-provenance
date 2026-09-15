"""F4 derived (Entry 24): all A-trained adapters produced in the v2 environment (five marked ladder
arms, the effectively unmarked alpha=1 arm, three nomark controls) on the natural paired contrast
rho(->K_A) - rho(->K_B), against v1's twelve A adapters (Colab decode / Colab stack; ledger).
Writes out/t1/f4_derived.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, numpy as np
from scipy import stats
T1 = os.path.join(EINV.V2, 'out', 't1'); R_REAL = 0.0356703416571125
S = json.load(open(os.path.join(T1, "summary.json")))["arms"]; N = json.load(open(os.path.join(T1, "summary_nomark.json")))["arms"]
V1_MEAN, V1_SD = -9.5699e-06, 4.842e-05
led = json.load(open(os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')))["primary"]["per_adapter_A"]
groups = {"marked_5": [S[a]["natural_paired_KA_minus_KB"] for a in ("mark_rand_a3_s0", "mark_rand_a3_s1", "mark_rand_a3_s2", "mark_rand_a12_s0", "mark_lowmid_a12_s0")],
          "unmarked_4": [S["mark_rand_a1_s0"]["natural_paired_KA_minus_KB"]] + [N[a]["natural_paired_KA_minus_KB"] for a in ("nomark_s0", "nomark_s1", "nomark_s2")],
          "v1_A_12": led}
groups["v2_all_9"] = groups["marked_5"] + groups["unmarked_4"]
out = {}
for k, v in groups.items():
    v = np.array(v); n = len(v); se = v.std(ddof=1) / np.sqrt(n)
    t0 = v.mean() / se; p0 = 1 - stats.t.cdf(t0, n - 1)
    out[k] = {"n": n, "values": v.tolist(), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "se": float(se),
              "t_vs_zero": float(t0), "p_one_sided_vs_zero": float(p0), "n_positive": int((v > 0).sum()),
              "lambda_pct": float(100 * v.mean() / R_REAL),
              "lambda_lower99_pct": float(100 * (v.mean() - stats.t.ppf(0.99, n - 1) * se) / R_REAL),
              "lambda_upper99_pct": float(100 * (v.mean() + stats.t.ppf(0.99, n - 1) * se) / R_REAL)}
v2 = np.array(groups["v2_all_9"]); v1 = np.array(led)
tw, pw = stats.ttest_ind(v2, v1, equal_var=False, alternative="greater")
mw = stats.mannwhitneyu(v2, v1, alternative="greater")
out["v2_all_9_vs_v1_A_12"] = {"welch_t": float(tw), "p_one_sided": float(pw), "mannwhitney_U": float(mw.statistic), "p_one_sided_mw": float(mw.pvalue),
                              "P_all9_positive_under_v1_normal": float((1 - stats.norm.cdf(0, V1_MEAN, V1_SD)) ** 9),
                              "difference_of_means": float(v2.mean() - v1.mean()), "difference_lambda_pct": float(100 * (v2.mean() - v1.mean()) / R_REAL)}
tu, pu = stats.ttest_ind(np.array(groups["marked_5"]), np.array(groups["unmarked_4"]), equal_var=False, alternative="greater")
out["marked_5_vs_unmarked_4"] = {"welch_t": float(tu), "p_one_sided": float(pu)}
# what the v1 statistic saw on the same instrument for v1's own A adapters, locally re-measured (Entry 10)
U = json.load(open(os.path.join(EINV.V2, 'out', 't2_summary.json')))["ncc"]["arms"]
out["v1_A_s0_s2_local_instrument"] = [U[a]["paired_contrast"] for a in ("A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16")]
json.dump(out, open(os.path.join(T1, "f4_derived.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
