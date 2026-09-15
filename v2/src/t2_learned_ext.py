"""Tier 2, Entry 17 — the learned detector at twelve adapters per arm.

Re-trains the Entry 09 network once (same seed, same data), saves its weights, and scores all
24 primary adapters (A_raw s0-s11, B_raw s0-s11) plus the base model from that single model,
using the first 250 generations of every arm so the unit is uniform. For s0-s2 the 500-image
means from the same model are also reported beside Entry 16's values.

Statistics (as registered, plus one addition made before any seed-3..11 image was scored):
  theta_A, theta_B      mean own-body contrast over the twelve adapters of each arm
  theta_sym             (theta_A + theta_B)/2 ; additive part (mean_A + mean_B)/2 of the raw scores
  IU sign-flip          exact 2^12 enumeration per arm on own-body contrasts after removing the
                        additive part; reject iff both one-sided p < 0.01
  label permutation     exact C(24,12) enumeration of mean_A - mean_B over adapter scores
                        (the addition: A/B labels exchangeable under no device-specific transfer)
  Welch t on theta_sym; plug-in one-sided 99 % upper limit lambda_U = (theta_sym + t SE)/R
Outputs: out/t2_learned_ext.json, out/t2_learned_net.pt
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time, itertools, numpy as np, torch, torch.nn as nn
sys.path.insert(0, EINV.SRC)
import t2_learned as L                      # __main__-guarded; brings Net, resid, patches, splits, dv
from scipy import stats

V2 = EINV.V2; OUT = os.path.join(V2, "out", "t2_learned_ext.json"); NET = os.path.join(V2, "out", "t2_learned_net.pt")
G = 250
GENS = {"base": os.path.join(V2, "data", "gens", "base")}
for s in range(12):
    for arm in ("A", "B"):
        k = f"{arm}_raw_s{s}_r16"
        GENS[k] = os.path.join(V2, "data", "gens" if s <= 2 else "gens_ext", k)

def train_and_save():
    rng = np.random.default_rng(L.SEED); torch.manual_seed(L.SEED)
    A, B = L.splits(L.dv["Nikon_D200_1"]), L.splits(L.dv["Nikon_D200_0"])
    tr = [(f, 0) for f in A["E1"] + A["E2"]] + [(f, 1) for f in B["E1"] + B["E2"]]
    Xtr, ytr = [], []
    for i, (f, lab) in enumerate(tr):
        W = L.resid(f, f"real_{lab}_{i}"); Xtr.append(L.patches(W, rng)); ytr += [lab] * L.NP
    Xtr = torch.from_numpy(np.concatenate(Xtr))[:, None]; ytr = torch.tensor(ytr)
    net = L.Net().to(L.DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(L.EPOCHS):
        perm = torch.randperm(len(ytr)); net.train()
        for k in range(0, len(perm), 64):
            idx = perm[k:k+64]; x, y = Xtr[idx].to(L.DEV), ytr[idx].to(L.DEV)
            loss = nn.functional.cross_entropy(net(x), y); opt.zero_grad(); loss.backward(); opt.step()
    net.eval(); torch.save(net.state_dict(), NET); print("[ext] trained and saved", NET, flush=True)
    ho = [(f, 0) for f in A["H"]] + [(f, 1) for f in B["H"]]
    return net, ho

def main():
    t0 = time.time()
    if os.path.exists(NET):
        net = L.Net().to(L.DEV); net.load_state_dict(torch.load(NET, map_location=L.DEV)); net.eval()
        A, B = L.splits(L.dv["Nikon_D200_1"]), L.splits(L.dv["Nikon_D200_0"])
        ho = [(f, 0) for f in A["H"]] + [(f, 1) for f in B["H"]]
        print("[ext] loaded", NET, flush=True)
    else:
        net, ho = train_and_save()

    @torch.no_grad()
    def score(W):
        x = torch.from_numpy(L.patches(W, np.random.default_rng(L.SEED)))[:, None].to(L.DEV)
        lg = net(x); return float((lg[:, 0] - lg[:, 1]).mean())

    sc = [(score(L.resid(f, f"realH_{lab}_{i}")), lab) for i, (f, lab) in enumerate(ho)]
    pos = np.array([s for s, l in sc if l == 0]); neg = np.array([s for s, l in sc if l == 1])
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())
    R = float(pos.mean() - neg.mean())
    print(f"[ext] real held-out AUC {auc:.4f}  R {R:.3f}", flush=True)

    arms = {}
    for k, d in GENS.items():
        files = sorted(glob.glob(os.path.join(d, "*.png")))
        if len(files) < G: raise SystemExit(f"{k}: only {len(files)} images present, need {G}")
        s = np.array([score(L.resid(f, f"gen_{k}_{os.path.basename(f)}")) for f in files[:G]])
        rec = {"n": G, "mean": float(s.mean()), "se": float(s.std(ddof=1) / np.sqrt(G))}
        if len(files) >= 500 and k != "base" and int(k.split("_s")[1].split("_")[0]) <= 2:
            s5 = np.array([score(L.resid(f, f"gen_{k}_{os.path.basename(f)}")) for f in files[:500]])
            rec["mean_500"] = float(s5.mean())
        arms[k] = rec; print(f"[ext] {k}: {rec}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    mA = np.array([arms[f"A_raw_s{s}_r16"]["mean"] for s in range(12)])
    mB = np.array([arms[f"B_raw_s{s}_r16"]["mean"] for s in range(12)])
    cA, cB = mA, -mB                                  # own-body contrasts
    thA, thB = cA.mean(), cB.mean(); th_sym = 0.5 * (thA + thB); additive = 0.5 * (mA.mean() + mB.mean())
    # IU sign-flip on corrected contrasts (exact 2^12)
    def signflip_p(c):
        c = np.asarray(c); n = len(c); obs = c.mean(); cnt = 0
        for signs in itertools.product((1, -1), repeat=n):
            if (c * np.array(signs)).mean() >= obs - 1e-15: cnt += 1
        return cnt / 2 ** n
    pA = signflip_p(cA - additive); pB = signflip_p(cB + additive)
    # label permutation, exact C(24,12)
    allv = np.concatenate([mA, mB]); obs = mA.mean() - mB.mean(); tot = allv.sum(); cnt = 0; ncomb = 0
    for comb in itertools.combinations(range(24), 12):
        sA = allv[list(comb)].sum(); statv = sA / 12 - (tot - sA) / 12; ncomb += 1
        if statv >= obs - 1e-15: cnt += 1
    p_perm = cnt / ncomb
    vA, vB = cA.var(ddof=1) / 12, cB.var(ddof=1) / 12
    se = 0.5 * np.sqrt(vA + vB); df = (vA + vB) ** 2 / (vA ** 2 / 11 + vB ** 2 / 11)
    t = th_sym / se; p_t = 1 - stats.t.cdf(t, df); UL = th_sym + stats.t.ppf(0.99, df) * se
    res = {"G_per_arm": G, "real_H_auc": auc, "real_paired_contrast_R": R, "base_mean": arms["base"]["mean"],
           "arms": arms, "A_means": mA.tolist(), "B_means": mB.tolist(),
           "theta_A": float(thA), "theta_B": float(thB), "theta_sym": float(th_sym), "additive_part": float(additive),
           "additive_over_R_pct": float(100 * abs(additive) / R),
           "IU_signflip_p_A": pA, "IU_signflip_p_B": pB, "IU_rejects_at_0.01": bool(pA < 0.01 and pB < 0.01),
           "label_permutation_p_one_sided": p_perm, "label_permutation_combinations": ncomb,
           "adapter_level_SE": float(se), "welch_df": float(df), "t": float(t), "p_t_one_sided": float(p_t),
           "lambda_sym_pct": float(100 * th_sym / R), "lambda_U_plugin_pct": float(100 * UL / R),
           "entry16_check_s0_s2_500": {k: [arms[k].get("mean_500"), None] for k in arms if k.endswith(("s0_r16", "s1_r16", "s2_r16"))}}
    e16 = json.load(open(os.path.join(V2, "out", "t2_learned.json")))["arms"]
    for k in res["entry16_check_s0_s2_500"]: res["entry16_check_s0_s2_500"][k][1] = e16[k]["mean_score_AminusB"]
    json.dump(res, open(OUT, "w"), indent=1)
    for k in ("theta_A", "theta_B", "theta_sym", "additive_part", "additive_over_R_pct", "IU_signflip_p_A", "IU_signflip_p_B",
              "label_permutation_p_one_sided", "adapter_level_SE", "t", "p_t_one_sided", "lambda_sym_pct", "lambda_U_plugin_pct"):
        print(f"[ext] {k:32s} {res[k]}")
    print("[ext] written", OUT)

if __name__ == "__main__":
    main()
