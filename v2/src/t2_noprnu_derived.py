"""Entry 43 (secondary, derived): does the PRNU-projected learned detector see the same per-adapter
structure as the original (Entry 18)? Per-seed A - B differences under both networks, the number of
positive seeds, and the adapter-level Pearson / Spearman correlation between the two networks'
own-body contrasts (additive part removed per arm). Writes out/t2_learned_noprnu_derived.json."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import json, numpy as np
from scipy import stats
V2 = os.path.join(EINV.V2, 'out')
n = json.load(open(f"{V2}/t2_learned_noprnu.json")); o = json.load(open(f"{V2}/t2_learned_ext.json"))
res = {}
for tag, d in (("noprnu", n), ("original", o)):
    A, B = np.array(d["A_means"]), np.array(d["B_means"])
    res[tag] = {"A_minus_B_per_seed": (A - B).tolist(), "n_positive_of_12": int(((A - B) > 0).sum()),
                "negative_seeds": [int(i) for i in np.where((A - B) <= 0)[0]]}
def own(d):
    A, B = np.array(d["A_means"]), np.array(d["B_means"]); cA, cB = A - A.mean(), -(B - B.mean())
    return np.concatenate([cA, cB])
x, y = own(n), own(o)
res["adapter_level_correlation_noprnu_vs_original"] = {"n": 24, "pearson_r": float(stats.pearsonr(x, y)[0]), "pearson_p": float(stats.pearsonr(x, y)[1]),
                                                       "spearman_rho": float(stats.spearmanr(x, y)[0])}
dA = np.array(n["A_means"]) - np.array(n["B_means"]); dO = np.array(o["A_means"]) - np.array(o["B_means"])
res["per_seed_A_minus_B_correlation"] = {"n": 12, "pearson_r": float(stats.pearsonr(dA, dO)[0]), "pearson_p": float(stats.pearsonr(dA, dO)[1])}
json.dump(res, open(f"{V2}/t2_learned_noprnu_derived.json", "w"), indent=1); print(json.dumps(res, indent=1))
