"""Tier 2 — a learned device detector, as a fourth detector family.

A small CNN is trained to tell device A from device B on 256x256 patches of the wavelet
residual of REAL photographs (E1 + E2 splits, 220 per body), validated on the held-out H
split (the positive control: same-model AUC must be high), and then applied to the archived
generations. Per image the score is mean over patches of logit(A) - logit(B); the paired
contrast for an arm trained on A is +score, for B is -score, exactly mirroring the paper's
statistic. This answers "a learned detector might see what correlation cannot".

Registered choices, fixed before any generation is scored: 4 conv blocks, 16 patches per
image, 20 epochs, AdamW 1e-3, seed 0, patch-mean pooling, no test-time augmentation.
Outputs: out/t2_learned.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time, numpy as np, torch, torch.nn as nn
from PIL import Image
sys.path.insert(0, EINV.SRC)
from fingerprints import load_lum_crop, wavelet_residual, MEAS, dv, splits

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); GENS = os.path.join(V2, "data", "gens")
OUT = os.path.join(V2, "out", "t2_learned.json"); CACHE = os.path.join(V2, "out", "resid_cache"); os.makedirs(CACHE, exist_ok=True)
ARMS = ["base", "A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16", "B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16"]
P, NP, EPOCHS, SEED = 256, 16, 20, 0
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(SEED); np.random.seed(SEED)

def resid(path, key):
    p = os.path.join(CACHE, key + ".npy")
    if os.path.exists(p): return np.load(p)
    if path.lower().endswith(".png"):
        a = np.asarray(Image.open(path).convert("RGB"), np.float32); lum = (0.299*a[...,0]+0.587*a[...,1]+0.114*a[...,2]).astype(np.float32)
    else: lum = load_lum_crop(path)
    W = wavelet_residual(lum); np.save(p, W.astype(np.float16)); return W

def patches(W, rng):
    ys = rng.integers(0, MEAS - P, NP); xs = rng.integers(0, MEAS - P, NP)
    return np.stack([W[y:y+P, x:x+P] for y, x in zip(ys, xs)]).astype(np.float32)

class Net(nn.Module):
    def __init__(s):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(c(1, 16), c(16, 32), c(32, 64), c(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, 2))
    def forward(s, x): return s.f(x)

def main():
    rng = np.random.default_rng(SEED)
    A, B = splits(dv["Nikon_D200_1"]), splits(dv["Nikon_D200_0"])
    tr = [(f, 0) for f in A["E1"] + A["E2"]] + [(f, 1) for f in B["E1"] + B["E2"]]
    ho = [(f, 0) for f in A["H"]] + [(f, 1) for f in B["H"]]
    t0 = time.time()
    Xtr = []; ytr = []
    for i, (f, lab) in enumerate(tr):
        W = resid(f, f"real_{lab}_{i}"); Xtr.append(patches(W, rng)); ytr += [lab]*NP
    Xtr = torch.from_numpy(np.concatenate(Xtr))[:, None]; ytr = torch.tensor(ytr)
    print(f"[learned] training patches {tuple(Xtr.shape)} from {len(tr)} real images ({(time.time()-t0)/60:.1f} min)", flush=True)
    net = Net().to(DEV); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(ytr)); tot = 0.0
        net.train()
        for k in range(0, len(perm), 64):
            idx = perm[k:k+64]; x, y = Xtr[idx].to(DEV), ytr[idx].to(DEV)
            loss = nn.functional.cross_entropy(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item()*len(idx)
        if (ep+1) % 5 == 0: print(f"[learned] epoch {ep+1} loss {tot/len(perm):.4f}", flush=True)
    net.eval()
    @torch.no_grad()
    def score(W):
        x = torch.from_numpy(patches(W, np.random.default_rng(SEED)))[:, None].to(DEV)
        lg = net(x); return float((lg[:, 0] - lg[:, 1]).mean())
    # positive control on held-out real images
    sc = [(score(resid(f, f"realH_{lab}_{i}")), lab) for i, (f, lab) in enumerate(ho)]
    pos = np.array([s for s, l in sc if l == 0]); neg = np.array([s for s, l in sc if l == 1])
    auc = float((pos[:, None] > neg[None, :]).mean() + 0.5*(pos[:, None] == neg[None, :]).mean())
    res = {"real_H_auc_A_vs_B": auc, "real_H_mean_score_A": float(pos.mean()), "real_H_mean_score_B": float(neg.mean()), "arms": {}}
    print(f"[learned] real held-out AUC {auc:.4f}", flush=True)
    for arm in ARMS:
        files = sorted(glob.glob(os.path.join(GENS, arm, "*.png")))
        s = np.array([score(resid(f, f"gen_{arm}_{os.path.basename(f)}")) for f in files])
        sign = 0 if arm == "base" else (1 if arm.startswith("A_") else -1)
        res["arms"][arm] = {"n": len(s), "mean_score_AminusB": float(s.mean()), "se": float(s.std(ddof=1)/np.sqrt(len(s))),
                            "paired_contrast_toward_own": None if not sign else float(sign*s.mean())}
        print(f"[learned] {arm}: n={len(s)} score {s.mean():+.4f} ± {s.std(ddof=1)/np.sqrt(len(s)):.4f}", flush=True)
    json.dump(res, open(OUT, "w"), indent=1); print("written", OUT)

if __name__ == "__main__":
    if os.path.exists(OUT):
        print("already written:", OUT, "(delete it to recompute)")
    else:
        main()
