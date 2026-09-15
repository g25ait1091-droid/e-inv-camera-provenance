"""Entry 38(b) — five-body closed-set attribution with a learned residual detector.

    python t2_learned_5body.py kodak | p20

Per group: 5-way four-block CNN (the Entry 09 architecture with 5 outputs) on wavelet-residual patches
(16 x 256^2 per image) from each body's E1 u E2 images; 20 epochs, AdamW 1e-3, seed 0; weights saved.
Gate: top-1 on the H split >= 0.90. Generations: archived 250 per adapter, 10 adapters. Score =
patch-mean logits. Attribution exactly as Entry 06: raw argmax and main-effect-corrected argmax
(subtract, for candidate d, its mean logit over generations of adapters not trained on d); blocks of
G consecutive generations, G in {1, 10, 50, 250}; adapter-level decision at G = 250; exact binomial.
Outputs out/t2_learned_5body_<group>.json, _logits.npz, _net.pt.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, csv, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
from multiprocessing import Pool
from scipy.stats import beta, binom
sys.path.insert(0, EINV.SRC)

V2 = EINV.V2; P, NP_, EPOCHS, SEED, GATE = 256, 16, 20, 0, 0.90
GROUPS = {"kodak": ["D0", "D1", "D2", "D3", "D4"], "p20": ["1101", "1102", "1103", "1104", "1105"]}
G_LIST = [1, 10, 50, 250]

def local(p): return p.replace("/content/drive/MyDrive/", EINV.MYDRIVE + "/")

def _winit():
    import torch as _t; _t.set_num_threads(1)

def patches_of(args):
    path, seed = args
    from fingerprints import load_lum_crop, wavelet_residual, MEAS
    if path.lower().endswith(".png"):
        a = np.asarray(Image.open(path).convert("RGB"), np.float32); Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    else:
        Y = load_lum_crop(path)
    W = wavelet_residual(Y); rng = np.random.default_rng(seed)
    ys = rng.integers(0, MEAS - P, NP_); xs = rng.integers(0, MEAS - P, NP_)
    return np.stack([W[y:y+P, x:x+P] for y, x in zip(ys, xs)]).astype(np.float16)

class NetK(nn.Module):
    def __init__(s, k):
        super().__init__()
        c = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(), nn.MaxPool2d(2))
        s.f = nn.Sequential(c(1, 16), c(16, 32), c(32, 64), c(64, 64), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(64, k))
    def forward(s, x): return s.f(x)

def cp(k, n, a=0.05):
    return (float(beta.ppf(a/2, k, n-k+1)) if k > 0 else 0.0, float(beta.ppf(1-a/2, k+1, n-k)) if k < n else 1.0)

def attribute(logits, devs):
    """logits: {tag: (250, 5)}; tag 'dev_sX'. Raw and corrected argmax, Entry 06 construction."""
    tags = sorted(logits); dev_of = {t: t.rsplit("_s", 1)[0] for t in tags}
    b = np.array([np.concatenate([logits[t] for t in tags if dev_of[t] != d])[:, j].mean() for j, d in enumerate(devs)])
    out = {"main_effects_b": dict(zip(devs, b.tolist()))}
    for score in ("raw", "corrected"):
        res = {}
        for G in G_LIST:
            correct = total = 0; adapters_right = 0; picks = {d: 0 for d in devs}
            for t in tags:
                Lg = logits[t] - (b if score == "corrected" else 0.0); nb = Lg.shape[0] // G; ok_t = 0
                for k in range(nb):
                    pred = devs[int(np.argmax(Lg[k*G:(k+1)*G].mean(0)))]; ok = pred == dev_of[t]; correct += ok; total += 1; ok_t += ok
                    if G == 250: picks[pred] += 1
                if ok_t * 2 > nb: adapters_right += 1
            n = len(tags)
            res[str(G)] = {"block_accuracy": correct / total, "n_blocks": total, "adapters_majority_correct": adapters_right,
                           "p_one_sided_binomial": float(binom.sf(adapters_right - 1, n, 0.2)), "adapter_CI95": cp(adapters_right, n)}
            if G == 250: res["selection_frequency_G250"] = picks
        out[score] = res
    return out

def main(group):
    devs = GROUPS[group]; man = list(csv.DictReader(open(os.path.join(V2, "data", "manifests", group, "manifest.csv"))))
    rows = [r for r in man if r["dev"] in devs]
    tr = [(local(r["path"]), devs.index(r["dev"])) for r in rows if r["split"] in ("E1", "E2")]
    ho = [(local(r["path"]), devs.index(r["dev"])) for r in rows if r["split"] == "H"]
    missing = [p for p, _ in tr + ho if not os.path.exists(p)]; assert not missing, missing[:5]
    t0 = time.time(); torch.manual_seed(SEED); dev = "cuda" if torch.cuda.is_available() else "cpu"
    with Pool(6, initializer=_winit) as pool:
        Xtr = pool.map(patches_of, [(p, SEED * 100000 + i) for i, (p, _) in enumerate(tr)], chunksize=4)
        Xho = pool.map(patches_of, [(p, SEED) for p, _ in ho], chunksize=4)
    ytr = torch.tensor(np.repeat([l for _, l in tr], NP_)); Xtr = torch.from_numpy(np.concatenate(Xtr))[:, None]
    print(f"[5body:{group}] {tuple(Xtr.shape)} training patches, {len(ho)} H images ({(time.time()-t0)/60:.1f} min)", flush=True)
    net = NetK(5).to(dev); opt = torch.optim.AdamW(net.parameters(), 1e-3, weight_decay=1e-4)
    for ep in range(EPOCHS):
        perm = torch.randperm(len(ytr)); net.train(); tot = 0.0
        for k in range(0, len(perm), 64):
            idx = perm[k:k+64]; x, y = Xtr[idx].float().to(dev), ytr[idx].to(dev)
            loss = nn.functional.cross_entropy(net(x), y); opt.zero_grad(); loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if (ep + 1) % 5 == 0: print(f"[5body:{group}] epoch {ep+1} loss {tot/len(perm):.4f}", flush=True)
    net.eval(); torch.save(net.state_dict(), os.path.join(V2, "out", f"t2_learned_5body_{group}_net.pt"))

    @torch.no_grad()
    def logit(p16): return net(torch.from_numpy(p16.astype(np.float32))[:, None].to(dev)).mean(0).cpu().numpy()

    Lh = np.array([logit(x) for x in Xho]); yh = np.array([l for _, l in ho])
    acc = float((Lh.argmax(1) == yh).mean()); per = {d: float((Lh[yh == j].argmax(1) == j).mean()) for j, d in enumerate(devs)}
    print(f"[5body:{group}] H top-1 {acc:.4f} per body {per} gate {'passed' if acc >= GATE else 'FAILED'}", flush=True)
    gens = {}
    tags = sorted(os.path.basename(d) for d in glob.glob(os.path.join(V2, "data", f"gens_{group}", "*")) if os.path.isdir(d))
    with Pool(6, initializer=_winit) as pool:
        for t in tags:
            files = sorted(glob.glob(os.path.join(V2, "data", f"gens_{group}", t, "*.png")))[:250]; assert len(files) == 250, (t, len(files))
            gens[t] = np.array([logit(x) for x in pool.map(patches_of, [(f, SEED) for f in files], chunksize=4)])
            print(f"[5body:{group}] {t}: mean logits {np.round(gens[t].mean(0), 3)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    np.savez_compressed(os.path.join(V2, "out", f"t2_learned_5body_{group}_logits.npz"), H=Lh, H_labels=yh, **gens)
    res = {"group": group, "devices": devs, "H_top1": acc, "H_per_body": per, "gate_passed": bool(acc >= GATE),
           "n_train_images": len(tr), "attribution": attribute(gens, devs)}
    json.dump(res, open(os.path.join(V2, "out", f"t2_learned_5body_{group}.json"), "w"), indent=1)
    a = res["attribution"]
    for s in ("raw", "corrected"):
        r = a[s]["250"]; print(f"[5body:{group}] {s}: G=250 adapters correct {r['adapters_majority_correct']}/10 p={r['p_one_sided_binomial']:.4f}; "
                              f"G=1 block acc {a[s]['1']['block_accuracy']:.3f}; picks {a[s]['selection_frequency_G250']}", flush=True)
    other = {"kodak": "p20", "p20": "kodak"}[group]; op = os.path.join(V2, "out", f"t2_learned_5body_{other}.json")
    if os.path.exists(op):
        o = json.load(open(op)); pooled = {}
        for s in ("raw", "corrected"):
            k = a[s]["250"]["adapters_majority_correct"] + o["attribution"][s]["250"]["adapters_majority_correct"]
            kmin = next(kk for kk in range(21) if binom.sf(kk - 1, 20, 0.2) < 0.01)
            pooled[s] = {"adapters_correct_of_20": k, "p_one_sided": float(binom.sf(k - 1, 20, 0.2)), "k_needed_for_p_below_0.01": kmin}
        json.dump(pooled, open(os.path.join(V2, "out", "t2_learned_5body_pooled.json"), "w"), indent=1); print("[5body] pooled", pooled, flush=True)

if __name__ == "__main__":
    main(sys.argv[1])
