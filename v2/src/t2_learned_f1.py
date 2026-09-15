"""Entry 18, follow-up F1 — is the learned-detector interaction PRNU-driven?

Adapter-level correlation (n = 24) between
  x: the learned own-body contrast of each primary adapter (out/t2_learned_ext.json; +score for
     A arms, -score for B arms; 250 generations),
  y: the archived NCC own-body paired contrast of the same adapter, rho(->K_own) - rho(->K_other),
     from the v1 per-row CSVs (data/csv_primary/*: s5_measure s0-s2, b3_measure s3-s5,
     sx2_measure s6-s11), restricted to the same first 250 generations; the ledger's 500-image
     per-adapter values are correlated as well.
Pearson r and Spearman rho with one-sided p (H1: positive). Writes out/t2_learned_f1.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, json, numpy as np, pandas as pd
from scipy import stats

V2 = EINV.V2
ext = json.load(open(os.path.join(V2, "out", "t2_learned_ext.json")))
led = json.load(open(os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')))["primary"]
csv = pd.concat([pd.read_csv(os.path.join(V2, "data", "csv_primary", p))[["tag", "gen_idx", "K", "rho_mult"]]
                 for p in ("P0v3/s5_measure.csv", "SEEDEXT/b3_measure.csv", "SEEDEXT2/sx2_measure.csv")])
csv = csv[csv.gen_idx < 250]
rows = []
for s in range(12):
    for arm in "AB":
        tag = f"{arm}_raw_s{s}_r16"; other = "B" if arm == "A" else "A"
        sub = csv[csv.tag == tag]
        own = sub[sub.K == arm].sort_values("gen_idx").rho_mult.values
        oth = sub[sub.K == other].sort_values("gen_idx").rho_mult.values
        assert len(own) == 250 and len(oth) == 250, (tag, len(own), len(oth))
        ncc250 = float((own - oth).mean())
        ncc500 = led["per_adapter_A"][s] if arm == "A" else led["per_adapter_B"][s]
        sc = ext["arms"][tag]["mean"]; learned = sc if arm == "A" else -sc
        rows.append({"tag": tag, "arm": arm, "seed": s, "learned_own_contrast": learned,
                     "ncc_own_contrast_250": ncc250, "ncc_own_contrast_500_ledger": ncc500})
x = np.array([r["learned_own_contrast"] for r in rows])
# remove each arm's additive part so that both variables are interaction-only per adapter
xa = x.copy(); xa[[r["arm"] == "A" for r in rows]] -= np.mean(x[[r["arm"] == "A" for r in rows]])
xa[[r["arm"] == "B" for r in rows]] -= np.mean(x[[r["arm"] == "B" for r in rows]])
out = {"n": len(rows), "rows": rows, "tests": {}}
for key in ("ncc_own_contrast_250", "ncc_own_contrast_500_ledger"):
    y = np.array([r[key] for r in rows])
    for xname, xv in (("learned_raw", x), ("learned_additive_removed", xa)):
        pr, pp = stats.pearsonr(xv, y); sr, sp = stats.spearmanr(xv, y)
        out["tests"][f"{xname}_vs_{key}"] = {"pearson_r": float(pr), "pearson_p_one_sided": float(pp / 2 if pr > 0 else 1 - pp / 2),
                                            "spearman_rho": float(sr), "spearman_p_one_sided": float(sp / 2 if sr > 0 else 1 - sp / 2)}
    # within-arm correlations too (12 each), since the additive part differs by arm
    for arm in "AB":
        m = np.array([r["arm"] == arm for r in rows])
        pr, pp = stats.pearsonr(x[m], y[m]); sr, sp = stats.spearmanr(x[m], y[m])
        out["tests"][f"within_{arm}_vs_{key}"] = {"pearson_r": float(pr), "spearman_rho": float(sr), "n": int(m.sum())}
json.dump(out, open(os.path.join(V2, "out", "t2_learned_f1.json"), "w"), indent=1)
for k, v in out["tests"].items(): print(f"{k:52s} {v}")
