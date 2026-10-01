"""G4 preparation (RESULTS.md Entry 78) - a second paired design on a modern smartphone model.

Huawei P10 Plus (VKY-AL00), Daxing devices 1604 (role A) and 1601 (role B), orientation 90 only, as the
P20 group uses one orientation. The local Daxing copy merges two shares and this family is ~50 %
byte-identical duplicates, so the file list is de-duplicated by SHA-256 before anything else: one file per
hash, the lexicographically first, and the hash of every kept file is recorded in the manifest.

Splits are the primary design's shape at this device's scale, with ten-image guard bands between them:
E1 60, E2 90, T 40, H 30 (220 images + 30 guard = 250, against 257 and 252 unique). Writes
out/fp_p10/{manifest.json, K_{A,B}_{E1,E2}.npy, H_{A,B}.npz}, the training crops
out/t1/train_png/{p10A_a0,p10B_a0}, and the fingerprint quality gates this study reports for every device
pair: cross-device correlation, split-half reliability, and the held-out AUC between the two bodies.
"""
import os, sys, json, glob, hashlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; OUT = os.path.join(V2, "out", "fp_p10"); T1 = os.path.join(V2, "out", "t1")
SRC = os.path.join(EINV.DAXING, 'image', '1601-1606')
DEV = {"A": "1604", "B": "1601"}
N = {"E1": 60, "E2": 90, "T": 40, "H": 30}; GUARD = 10
os.makedirs(OUT, exist_ok=True)

def dedup(dev):
    seen, keep = {}, []
    for f in sorted(glob.glob(os.path.join(SRC, dev, "90", "*.jpg"))):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if h not in seen:
            seen[h] = f; keep.append((f, h))
    return keep

def splits(files):
    out, i = {}, 0
    for name in ("E1", "E2", "T", "H"):
        out[name] = files[i:i + N[name]]; i += N[name] + GUARD
    return out

def main():
    from fingerprints import estimate_K, load_lum_crop, wavelet_residual
    import t1_ladder as L
    man = {"dataset": "Daxing", "model": "Huawei P10 Plus (VKY-AL00)", "orientation": 90,
           "dedup": "one file per SHA-256, lexicographically first", "splits": N, "guard": GUARD, "roles": {}}
    K = {}
    for role, dev in DEV.items():
        keep = dedup(dev)
        need = sum(N.values()) + 3 * GUARD
        assert len(keep) >= need, f"{dev}: {len(keep)} unique, need {need}"
        sp = splits([f for f, _ in keep])
        man["roles"][role] = {"device": dev, "unique_images": len(keep), "raw_files": len(glob.glob(os.path.join(SRC, dev, "90", "*.jpg"))),
                              **{k: [os.path.basename(x) for x in v] for k, v in sp.items()},
                              "sha256_first8": {os.path.basename(f): h[:8] for f, h in keep[:need]}}
        for e in ("E1", "E2"):
            p = os.path.join(OUT, f"K_{role}_{e}.npy")
            if not os.path.exists(p): np.save(p, estimate_K(sp[e]))
            K[f"{role}_{e}"] = np.load(p)
        ph = os.path.join(OUT, f"H_{role}.npz")
        if not os.path.exists(ph):
            Y = np.stack([load_lum_crop(f) for f in sp["H"]])
            np.savez_compressed(ph, Y=Y, W=np.stack([wavelet_residual(y) for y in Y]))
        d = os.path.join(T1, "train_png", f"p10{role}_a0"); os.makedirs(d, exist_ok=True)
        for i, f in enumerate(sp["T"]):
            Image.fromarray(np.clip(np.rint(L.load_rgb_crop(f)), 0, 255).astype(np.uint8)).save(
                os.path.join(d, f"{i:04d}.png"), compress_level=1)
        print(f"[g4] {role} = {dev}: {len(keep)} unique, splits built, {len(sp['T'])} training crops", flush=True)

    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    gates = {"kappa_model_E2": ncc(K["A_E2"], K["B_E2"]),
             "splithalf_A": ncc(K["A_E1"], K["A_E2"]), "splithalf_B": ncc(K["B_E1"], K["B_E2"])}
    for tag in ("E2",):
        own_all, oth_all, per = [], [], {}
        for role, other in (("A", "B"), ("B", "A")):
            z = np.load(os.path.join(OUT, f"H_{role}.npz")); Y, W = z["Y"], z["W"]
            own = [ncc(W[i], Y[i] * K[f"{role}_{tag}"]) for i in range(len(Y))]
            oth = [ncc(W[i], Y[i] * K[f"{other}_{tag}"]) for i in range(len(Y))]
            per[role] = float(np.mean(own) - np.mean(oth)); own_all += own; oth_all += oth
        sfx = "" if tag == "E2" else "_" + tag
        # canonical paired contrast, as in real_contrast() of g3_estimator_scale.py: the two bodies ADD
        gates[f"R_real{sfx}"] = 0.5 * (per["A"] + per["B"])
        gates[f"per_body_contrast{sfx}"] = per
        gates[f"AUC_held_out{sfx}"] = float(np.mean([(p > n) + 0.5 * (p == n)
                                                     for p in own_all for n in oth_all]))
    man["gates"] = gates
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("[g4] gates:", {k: round(v, 4) for k, v in gates.items()}, flush=True)

if __name__ == "__main__":
    main()
