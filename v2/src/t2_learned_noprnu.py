"""Entry 38(a) — the learned detector with the PRNU template projected out of every residual.

Identical to Entries 17-18 (four-block CNN, E1 u E2 real images of bodies A and B, 16 patches of 256^2,
20 epochs, AdamW 1e-3, seed 0; 24 primary adapters x 250 generations + base) except that every
residual W is replaced by W - X c, where X stacks the four templates Y*K_A(E1), Y*K_A(E2), Y*K_B(E1),
Y*K_B(E2) and c is the least-squares coefficient vector of W on X. Outputs out/t2_learned_noprnu.json,
weights out/t2_learned_noprnu_net.pt; real-image projected residuals cached in out/resid_cache_noprnu.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time, itertools
import numpy as np, torch, torch.nn as nn
from PIL import Image
from scipy import stats
sys.path.insert(0, EINV.SRC)
import t2_learned as L
from fingerprints import load_lum_crop, wavelet_residual, dv, splits

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp")
OUT = os.path.join(V2, "out", "t2_learned_noprnu.json"); NET = os.path.join(V2, "out", "t2_learned_noprnu_net.pt")
CACHE = os.path.join(V2, "out", "resid_cache_noprnu"); os.makedirs(CACHE, exist_ok=True)
KS = [np.load(os.path.join(FP, f"K_{d}_{e}.npy")).astype(np.float32) for d in ("A", "B") for e in ("E1", "E2")]
KA2 = np.load(os.path.join(FP, "K_A_E2.npy")).astype(np.float32)
G = 250
GENS = {"base": os.path.join(V2, "data", "gens", "base")}
for s in range(12):
    for arm in ("A", "B"):
        k = f"{arm}_raw_s{s}_r16"; GENS[k] = os.path.join(V2, "data", "gens" if s <= 2 else "gens_ext", k)

def lum_of(path):
    if path.lower().endswith(".png"):
        a = np.asarray(Image.open(path).convert("RGB"), np.float32)
        return (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    return load_lum_crop(path)

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def project(Y, W):
    X = np.stack([(Y * K).ravel() for K in KS]).astype(np.float64); w = W.ravel().astype(np.float64)
    c = np.linalg.solve(X @ X.T, X @ w)
    return (w - c @ X).reshape(W.shape).astype(np.float32)

def resid_np(path, key=None, with_check=False):
    if key and not with_check:
        p = os.path.join(CACHE, key + ".npy")
        if os.path.exists(p): return np.load(p).astype(np.float32)
    Y = lum_of(path); W = wavelet_residual(Y).astype(np.float32); Wp = project(Y, W)
    if key: np.save(os.path.join(CACHE, key + ".npy"), Wp.astype(np.float16))
    if with_check: return Wp, ncc(W, Y * KA2), ncc(Wp, Y * KA2)
    return Wp

def signflip_p(c):
    c = np.asarray(c); obs = c.mean(); cnt = 0
    for signs in itertools.product((1, -1), repeat=len(c)):
        if (c * np.array(signs)).mean() >= obs - 1e-15: cnt += 1
    return cnt / 2 ** len(c)

def main():
    t0 = time.time(); rng = np.random.default_rng(L.SEED); torch.manual_seed(L.SEED)
    A, B = splits(dv["Nikon_D200_1"]), splits(dv["Nikon_D200_0"])
    tr = [(f, 0) for f in A["E1"] + A["E2"]] + [(f, 1) for f in B["E1"] + B["E2"]]
    Xtr, ytr = [], []
    for i, (f, lab) in enumerate(tr):
        Xtr.append(L.patches(resid_np(f, f"real_{lab}_{i}"), rng).astype(np.float16)); ytr += [lab] * L.NP
    Xtr = torch.from_numpy(np.concatenate(Xtr))[:, None]; ytr = torch.tensor(ytr)
    print(f"[noprnu] {tuple(Xtr.shape)} training patches ({(time.time()-t0)/60:.1f} min)", flush=True)
    net = L.Net().to(L.DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(L.EPOCHS):
        perm = torch.randperm(len(ytr)); net.train()
        for k in range(0, len(perm), 64):
            idx = perm[k:k+64]; x, y = Xtr[idx].float().to(L.DEV), ytr[idx].to(L.DEV)
            loss = nn.functional.cross_entropy(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); torch.save(net.state_dict(), NET); print("[noprnu] trained", NET, flush=True)

    @torch.no_grad()
    def score(W):
        x = torch.from_numpy(L.patches(W, np.random.default_rng(L.SEED)))[:, None].to(L.DEV)
        lg = net(x); return float((lg[:, 0] - lg[:, 1]).mean())

    ho = [(f, 0) for f in A["H"]] + [(f, 1) for f in B["H"]]
    sc, chk = [], []
    for i, (f, lab) in enumerate(ho):
        Wp, before, after = resid_np(f, f"realH_{lab}_{i}", with_check=True)
        sc.append((score(Wp), lab)); chk.append((before, after))
    pos = np.array([s for s, l in sc if l == 0]); neg = np.array([s for s, l in sc if l == 1])
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()); R = float(pos.mean() - neg.mean())
    chk = np.array(chk)
    print(f"[noprnu] real AUC {auc:.4f} R {R:.3f}; |NCC(W, Y K_A)| before {np.abs(chk[:,0]).mean():.4f} after {np.abs(chk[:,1]).mean():.2e}", flush=True)

    arms = {}
    for k, d in GENS.items():
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        if len(files) < G: raise SystemExit(f"{k}: only {len(files)} images present, need {G}")
        s = np.array([score(resid_np(f)) for f in files[:G]])
        arms[k] = {"n": G, "mean": float(s.mean()), "se": float(s.std(ddof=1) / np.sqrt(G))}
        print(f"[noprnu] {k}: {arms[k]} ({(time.time()-t0)/60:.1f} min)", flush=True)
    mA = np.array([arms[f"A_raw_s{s}_r16"]["mean"] for s in range(12)]); mB = np.array([arms[f"B_raw_s{s}_r16"]["mean"] for s in range(12)])
    cA, cB = mA, -mB; thA, thB = cA.mean(), cB.mean(); th = 0.5 * (thA + thB); add = 0.5 * (mA.mean() + mB.mean())
    pA, pB = signflip_p(cA - add), signflip_p(cB + add)
    allv = np.concatenate([mA, mB]); obs = mA.mean() - mB.mean(); tot = allv.sum(); cnt = n = 0
    for comb in itertools.combinations(range(24), 12):
        sA = allv[list(comb)].sum(); n += 1
        if sA / 12 - (tot - sA) / 12 >= obs - 1e-15: cnt += 1
    vA, vB = cA.var(ddof=1) / 12, cB.var(ddof=1) / 12; se = 0.5 * np.sqrt(vA + vB); df = (vA + vB) ** 2 / (vA ** 2 / 11 + vB ** 2 / 11); t = th / se
    ref = json.load(open(os.path.join(V2, "out", "t2_learned_ext.json")))
    res = {"real_H_auc": auc, "real_paired_contrast_R": R,
           "template_ncc_on_H_before_after": {"mean_abs_before": float(np.abs(chk[:, 0]).mean()), "mean_abs_after": float(np.abs(chk[:, 1]).mean())},
           "base_mean": arms["base"]["mean"], "arms": arms, "A_means": mA.tolist(), "B_means": mB.tolist(),
           "theta_A": float(thA), "theta_B": float(thB), "theta_sym": float(th), "additive_part": float(add),
           "additive_over_R_pct": float(100 * abs(add) / R), "IU_signflip_p_A": pA, "IU_signflip_p_B": pB,
           "IU_rejects_at_0.01": bool(pA < 0.01 and pB < 0.01), "label_permutation_p_one_sided": cnt / n,
           "adapter_level_SE": float(se), "welch_df": float(df), "t": float(t), "p_t_one_sided": float(1 - stats.t.cdf(t, df)),
           "lambda_sym_pct": float(100 * th / R), "lambda_U_plugin_pct": float(100 * (th + stats.t.ppf(0.99, df) * se) / R),
           "reference_entry18": {k: ref[k] for k in ("real_H_auc", "theta_sym", "lambda_sym_pct", "lambda_U_plugin_pct", "label_permutation_p_one_sided", "t")}}
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("real_H_auc", "theta_sym", "additive_over_R_pct", "IU_signflip_p_A", "IU_signflip_p_B", "label_permutation_p_one_sided",
              "t", "p_t_one_sided", "lambda_sym_pct", "lambda_U_plugin_pct"): print(f"[noprnu] {k:30s} {res[k]}")
    print("[noprnu] written", OUT)

if __name__ == "__main__":
    main()
