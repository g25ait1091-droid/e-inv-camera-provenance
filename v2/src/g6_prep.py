"""G6 preparation (RESULTS.md Entry 80) - a third paired design: Apple iPhone 5c, VISION D05 and D14.

Native camera images only (images/nat); the Facebook-recompressed copies VISION also ships (natFBH,
natFBL) are excluded, and file lists are de-duplicated by SHA-256 before splitting. Splits take the
shape of the primary design at this pair's scale, ten-image guards: E1 40, E2 60, T 50, H 25 (Entry 87;
205 images of the 209 body B holds, with the 50 training images every arm in this study uses).

Two fingerprint estimates per body are built. The primary one follows the study's design and uses natural
photographs (E1 and E2). The second uses the flat-field images VISION provides, which are what PRNU
estimation normally uses and give a far better estimate; measuring the same generations with both tests
the estimator dependence of Entry 63 with data rather than statistics.

Writes out/fp_5c/{manifest.json, K_{A,B}_{E1,E2,FLAT}.npy, H_{A,B}.npz} and the training crops
out/t1/train_png/{p5cA_a0,p5cB_a0}.
"""
import os, sys, json, glob, shutil, hashlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; OUT = os.path.join(V2, "out", "fp_5c"); T1 = os.path.join(V2, "out", "t1")
SRC = os.path.join(EINV.DATASETS, 'vision', 'dataset')
DEV = {"A": "D05_Apple_iPhone5c", "B": "D14_Apple_iPhone5c"}
N = {"E1": 40, "E2": 60, "T": 50, "H": 25}; GUARD = 10   # Entry 87: T=50 as every arm trains on 50 (D14 holds 209)
os.makedirs(OUT, exist_ok=True)

def dedup(paths):
    seen, keep = set(), []
    for f in sorted(paths):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if h not in seen:
            seen.add(h); keep.append(f)
    return keep

def main():
    from fingerprints import estimate_K, load_lum_crop, wavelet_residual
    import t1_ladder as L
    man = {"dataset": "VISION", "model": "Apple iPhone 5c", "source": "images/nat (native camera only)",
           "dedup": "one file per SHA-256", "splits": N, "guard": GUARD, "roles": {}}
    K = {}
    for role, dev in DEV.items():
        nat = dedup(glob.glob(os.path.join(SRC, dev, "images", "nat", "*.jpg")))
        flat = dedup(glob.glob(os.path.join(SRC, dev, "images", "flat", "*.jpg")))
        need = sum(N.values()) + 3 * GUARD
        assert len(nat) >= need, f"{dev}: {len(nat)} unique natural images, need {need}"
        sp, i = {}, 0
        for k in ("E1", "E2", "T", "H"):
            sp[k] = nat[i:i + N[k]]; i += N[k] + GUARD
        for e in ("E1", "E2"):
            p = os.path.join(OUT, f"K_{role}_{e}.npy")
            if not os.path.exists(p): np.save(p, estimate_K(sp[e]))
            K[f"{role}_{e}"] = np.load(p)
        pf = os.path.join(OUT, f"K_{role}_FLAT.npy")
        if not os.path.exists(pf): np.save(pf, estimate_K(flat))
        K[f"{role}_FLAT"] = np.load(pf)
        ph = os.path.join(OUT, f"H_{role}.npz")
        if not os.path.exists(ph):
            Y = np.stack([load_lum_crop(f) for f in sp["H"]])
            np.savez_compressed(ph, Y=Y, W=np.stack([wavelet_residual(y) for y in Y]))
        d = os.path.join(T1, "train_png", f"p5c{role}_a0"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
        for j, f in enumerate(sp["T"]):
            Image.fromarray(np.clip(np.rint(L.load_rgb_crop(f)), 0, 255).astype(np.uint8)).save(
                os.path.join(d, f"{j:04d}.png"), compress_level=1)
        man["roles"][role] = {"device": dev, "unique_nat": len(nat), "unique_flat": len(flat),
                              **{k: [os.path.basename(x) for x in v] for k, v in sp.items()}}
        print(f"[g6] {role} = {dev}: {len(nat)} unique natural, {len(flat)} flat, crops written", flush=True)

    def ncc(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    gates = {"kappa_model_E2": ncc(K["A_E2"], K["B_E2"]), "kappa_model_FLAT": ncc(K["A_FLAT"], K["B_FLAT"]),
             "splithalf_A": ncc(K["A_E1"], K["A_E2"]), "splithalf_B": ncc(K["B_E1"], K["B_E2"]),
             "corr_E2_FLAT_A": ncc(K["A_E2"], K["A_FLAT"]), "corr_E2_FLAT_B": ncc(K["B_E2"], K["B_FLAT"])}
    for tag in ("E2", "FLAT"):
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
    print("[g6] gates:", {k: (round(v, 4) if isinstance(v, float) else
                              {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in gates.items()},
          flush=True)
    for tag in ("E2", "FLAT"):
        sfx = "" if tag == "E2" else "_" + tag
        ok = gates["AUC_held_out" + sfx] >= 0.90 and min(gates["splithalf_A"], gates["splithalf_B"]) >= 0.15
        print(f"[g6] {tag} gate {'MET' if ok else 'NOT met - descriptive only'}", flush=True)

if __name__ == "__main__":
    main()
