"""Entry 48 F8 — do generations that reproduce training photographs more closely carry more of the
training body's fingerprint?

For every generation (first 250 per adapter) of the v2-environment unmarked adapters present:
  m = max NCC between its 256^2 grayscale thumbnail and the 50 training crops of its body
  c = own-body contrast rho(->K_own) - rho(->K_other), from the t1_measure rows
Within-adapter slope of c on m (adapter fixed effects), pooled and separately for A and B adapters;
one-sided permutation test shuffling m within adapter (2000 permutations); most-memorized decile minus
the rest. Writes out/t1/f8_memorization.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, glob, json, numpy as np, pandas as pd
from PIL import Image
T1 = os.path.join(EINV.V2, 'out', 't1'); G = 250; NPERM = 2000
ROWS = {"nomark": "measure_rows_nomark.csv", "nomarkB": "measure_rows_nomarkB.csv", "dose8k": "measure_rows_dose8k.csv",
        "dose16k": "measure_rows_dose16k.csv", "dose16krep": "measure_rows_dose16krep.csv", "dose16krep2": "measure_rows_dose16krep2.csv"}
rows = pd.concat([pd.read_csv(os.path.join(T1, f)) for f in ROWS.values() if os.path.exists(os.path.join(T1, f))])
def body(arm): return "B" if (arm.startswith("nomarkB") or "_B_" in arm) else "A"
def thumb(path):
    a = np.asarray(Image.open(path).convert("L").resize((256, 256), Image.BOX), np.float32).ravel(); a -= a.mean(); return a / (np.linalg.norm(a) + 1e-12)
train = {b: np.stack([thumb(f) for f in sorted(glob.glob(os.path.join(T1, "train_png", d, "*.png")))]) for b, d in (("A", "none_a0"), ("B", "noneB_a0"))}
arms = sorted(a for a in rows.arm.unique() if a.startswith(("nomark", "dose8k", "dose16k")))
data = []
for arm in arms:
    b = body(arm); g = rows[rows.arm == arm].copy(); g = g.sort_values("image").head(G)
    own = (g.rho_KA - g.rho_KB) if b == "A" else (g.rho_KB - g.rho_KA)
    m = np.array([float((train[b] @ thumb(os.path.join(T1, "gens", arm, im))).max()) for im in g.image])
    data.append({"arm": arm, "body": b, "m": m, "c": own.values.astype(float)})
    print(f"[f8] {arm}: n={len(m)} m mean {m.mean():.3f} c mean {own.mean():+.2e}", flush=True)

def slope(sel, perm_rng=None):
    num = den = 0.0
    for d in sel:
        m = d["m"] - d["m"].mean(); c = d["c"] - d["c"].mean()
        if perm_rng is not None: m = perm_rng.permutation(m)
        num += (m * c).sum(); den += (m * m).sum()
    return num / den
rng = np.random.default_rng(20260912)
res = {"arms": [{"arm": d["arm"], "body": d["body"], "n": len(d["m"]), "m_mean": float(d["m"].mean()), "c_mean": float(d["c"].mean()),
                 "top_decile_minus_rest": float(d["c"][d["m"] >= np.quantile(d["m"], 0.9)].mean() - d["c"][d["m"] < np.quantile(d["m"], 0.9)].mean())}
                for d in data]}
for name, sel in (("pooled", data), ("A_adapters", [d for d in data if d["body"] == "A"]), ("B_adapters", [d for d in data if d["body"] == "B"])):
    obs = slope(sel); null = np.array([slope(sel, rng) for _ in range(NPERM)])
    res[name] = {"n_adapters": len(sel), "slope_c_on_m": float(obs), "perm_p_one_sided": float((null >= obs).mean()), "null_sd": float(null.std())}
    print(f"[f8] {name}: slope {obs:+.3e} p {res[name]['perm_p_one_sided']:.4f}", flush=True)
res["registered_positive"] = bool(res["pooled"]["perm_p_one_sided"] < 0.01 and res["A_adapters"]["slope_c_on_m"] > 0 and res["B_adapters"]["slope_c_on_m"] > 0)
json.dump(res, open(os.path.join(T1, "f8_memorization.json"), "w"), indent=1); print("[f8] registered positive:", res["registered_positive"])
