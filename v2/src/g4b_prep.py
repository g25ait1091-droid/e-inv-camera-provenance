"""G4b preparation (RESULTS.md Entry 85) - the modern-smartphone paired design, retried on Daxing P20.

G4 closed at its fingerprint gate: the P10 Plus pair's body 1601 does not correlate with its own held-out
photographs (own-minus-other +0.0030 against 1604's +0.0328, held-out AUC 0.809 against the registered
0.90), which the multi-frame captures in that family explain. The scope condition it was to close is still
open, so it is retried on a pair that can carry it.

Device selection is the rule registered in Entry 85, decided on real photographs alone with no generated
image involved: among the five duplicate-free Daxing P20 bodies, the two with the highest held-out
per-device contrast in the existing five-body arm that also hold at least 250 unique images. That selects
1104 (role A, contrast 0.075, 262 images) and 1103 (role B, 0.069, 280). Orientation 90 only, as the P20
group uses one orientation.

Design as registered, with Entry 87's training size: splits E1 60, E2 90, T 50, H 30 with ten-image
guards (260 images of 262), the 50 training images every arm in this study uses. The
file list is de-duplicated by SHA-256 first, as for every Daxing family. Gates use the corrected paired
definition of Entry 85 (defect D9): R_real = 0.5*(own-minus-other of A + own-minus-other of B), and the
held-out AUC separates own-fingerprint from other-fingerprint scores on the same photographs.

Writes out/fp_p20b/{manifest.json, K_{A,B}_{E1,E2}.npy, H_{A,B}.npz} and the training crops
out/t1/train_png/{p20bA_a0,p20bB_a0}.
"""
import os, sys, json, glob, shutil, hashlib
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; OUT = os.path.join(V2, "out", "fp_p20b"); T1 = os.path.join(V2, "out", "t1")
SRC = os.path.join(EINV.DAXING, 'image', '1101-1104')
DEV = {"A": "1104", "B": "1103"}
N = {"E1": 60, "E2": 90, "T": 50, "H": 30}; GUARD = 10   # Entry 87: T=50 as every arm trains on 50 (1104 holds 262)
os.makedirs(OUT, exist_ok=True)


def dedup(dev):
    seen, keep = set(), []
    for f in sorted(glob.glob(os.path.join(SRC, dev, "90", "*.jpg"))):
        h = hashlib.sha256(open(f, "rb").read()).hexdigest()
        if h not in seen:
            seen.add(h); keep.append((f, h))
    return keep


def splits(files):
    out, i = {}, 0
    for name in ("E1", "E2", "T", "H"):
        out[name] = files[i:i + N[name]]; i += N[name] + GUARD
    return out


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    from fingerprints import estimate_K, load_lum_crop, wavelet_residual
    import t1_ladder as L
    man = {"dataset": "Daxing", "model": "Huawei P20 (EML-AL00)", "orientation": 90,
           "entry": "RESULTS.md Entry 85 (G4b)",
           "selection_rule": ("highest held-out per-device contrast among the five duplicate-free P20 "
                              "bodies, subject to at least 250 unique images; decided on real photographs "
                              "only, before any adapter was trained"),
           "dedup": "one file per SHA-256, lexicographically first",
           "splits": N, "guard": GUARD, "roles": {}}
    K = {}
    for role, dev in DEV.items():
        keep = dedup(dev)
        need = sum(N.values()) + 3 * GUARD
        assert len(keep) >= need, f"{dev}: {len(keep)} unique, need {need}"
        sp = splits([f for f, _ in keep])
        man["roles"][role] = {"device": dev, "unique_images": len(keep),
                              "raw_files": len(glob.glob(os.path.join(SRC, dev, "90", "*.jpg"))),
                              **{k: [os.path.basename(x) for x in v] for k, v in sp.items()}}
        for e in ("E1", "E2"):
            p = os.path.join(OUT, f"K_{role}_{e}.npy")
            if not os.path.exists(p): np.save(p, estimate_K(sp[e]))
            K[f"{role}_{e}"] = np.load(p)
        ph = os.path.join(OUT, f"H_{role}.npz")
        if not os.path.exists(ph):
            Y = np.stack([load_lum_crop(f) for f in sp["H"]])
            np.savez_compressed(ph, Y=Y, W=np.stack([wavelet_residual(y) for y in Y]))
        d = os.path.join(T1, "train_png", f"p20b{role}_a0"); shutil.rmtree(d, ignore_errors=True); os.makedirs(d, exist_ok=True)
        for i, f in enumerate(sp["T"]):
            Image.fromarray(np.clip(np.rint(L.load_rgb_crop(f)), 0, 255).astype(np.uint8)).save(
                os.path.join(d, f"{i:04d}.png"), compress_level=1)
        print(f"[g4b] {role} = {dev}: {len(keep)} unique of {man['roles'][role]['raw_files']} files, "
              f"{len(sp['T'])} training crops", flush=True)

    gates = {"kappa_model_E2": ncc(K["A_E2"], K["B_E2"]),
             "splithalf_A": ncc(K["A_E1"], K["A_E2"]), "splithalf_B": ncc(K["B_E1"], K["B_E2"])}
    own_all, oth_all, per = [], [], {}
    for role, other in (("A", "B"), ("B", "A")):
        z = np.load(os.path.join(OUT, f"H_{role}.npz")); Y, W = z["Y"], z["W"]
        own = [ncc(W[i], Y[i] * K[f"{role}_E2"]) for i in range(len(Y))]
        oth = [ncc(W[i], Y[i] * K[f"{other}_E2"]) for i in range(len(Y))]
        per[role] = float(np.mean(own) - np.mean(oth)); own_all += own; oth_all += oth
    gates["R_real"] = 0.5 * (per["A"] + per["B"])
    gates["per_body_contrast"] = per
    gates["AUC_held_out"] = float(np.mean([(p > n) + 0.5 * (p == n) for p in own_all for n in oth_all]))
    gates["paired_own_gt_other"] = float(np.mean(np.array(own_all) > np.array(oth_all)))
    man["gates"] = gates
    json.dump(man, open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("[g4b] gates:", {k: (round(v, 4) if isinstance(v, float) else
                               {kk: round(vv, 4) for kk, vv in v.items()}) for k, v in gates.items()},
          flush=True)
    ok = gates["AUC_held_out"] >= 0.90 and min(gates["splithalf_A"], gates["splithalf_B"]) >= 0.15
    print(f"[g4b] registered gate {'MET - the arm may claim a limit' if ok else 'NOT met - descriptive only'}",
          flush=True)


if __name__ == "__main__":
    main()
