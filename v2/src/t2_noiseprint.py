"""Entry 28 — Noiseprint as a fifth detector family. Runs in the TF 1.15 environment:
    C:/Users/Administrator/anaconda3/envs/a4tf1/python.exe src/t2_noiseprint.py gate
    ... gens        (only if the gate passed; scores the 3,500 base-study generations)

Noiseprint (grip-unina, published per-QF weights) via D:/A4/src/a4_noiseprint.py (session-reuse
wrapper, verified bit-identical to the reference call). Image -> centre 1024^2 luminance in [0,1]
(the study's crop convention) -> noiseprint -> zero-mean. Fingerprint per body = mean over its E2
images. Score = zero-lag NCC against each fingerprint; paired contrast = own - other.
Gate: same-model AUC on the H split (40 + 40) >= 0.95.
Caches noiseprints as float16 in out/np_cache/. Outputs out/t2_noiseprint_gate.json / _gens.json.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time
import numpy as np
from PIL import Image
sys.path.insert(0, EINV.SRC)
import a4_noiseprint as A

V2 = EINV.V2; MEAS = 1024; CACHE = os.path.join(V2, "out", "np_cache"); os.makedirs(CACHE, exist_ok=True)
SPL = json.load(open(os.path.join(V2, "data", "manifests", "primary_splits.json")))
ARMS = ["base", "A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16", "B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16"]

def lum_crop(fp):
    with Image.open(fp) as im:
        im = im.convert("RGB"); W, H = im.size
        a = np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.float32)
    return ((0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]) / 255.0).astype(np.float32)

def nprint(fp, key):
    p = os.path.join(CACHE, key + ".npy")
    if os.path.exists(p): return np.load(p).astype(np.float32)
    qf = A.qf_of(fp) if fp.lower().endswith((".jpg", ".jpeg")) else 101
    r = A.extract(lum_crop(fp), qf).astype(np.float32); r = r - r.mean()
    np.save(p, r.astype(np.float16)); return r

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def fingerprint(body, tag):
    fs = SPL[body]["E2"]; acc = None; t0 = time.time()
    for i, f in enumerate(fs):
        r = nprint(f, f"real_{tag}_E2_{i}"); acc = r if acc is None else acc + r
        if (i+1) % 20 == 0: print(f"[np] {tag} E2 {i+1}/{len(fs)} ({(time.time()-t0)/60:.1f} min)", flush=True)
    F = acc / len(fs); return (F - F.mean()).astype(np.float32)

def gate():
    FA, FB = fingerprint("Nikon_D200_1", "A"), fingerprint("Nikon_D200_0", "B")
    np.save(os.path.join(V2, "out", "np_fingerprint_A.npy"), FA); np.save(os.path.join(V2, "out", "np_fingerprint_B.npy"), FB)
    rows = []
    for body, tag, lab in (("Nikon_D200_1", "A", 0), ("Nikon_D200_0", "B", 1)):
        for i, f in enumerate(SPL[body]["H"]):
            r = nprint(f, f"real_{tag}_H_{i}"); rows.append((lab, ncc(r, FA), ncc(r, FB)))
    rows = np.array(rows); sA = rows[rows[:, 0] == 0]; sB = rows[rows[:, 0] == 1]
    own_minus_other = np.concatenate([sA[:, 1] - sA[:, 2], sB[:, 2] - sB[:, 1]])
    dA = sA[:, 1] - sA[:, 2]; dB = sB[:, 1] - sB[:, 2]     # score toward A minus toward B; A images should be higher
    auc = float((dA[:, None] > dB[None, :]).mean() + 0.5 * (dA[:, None] == dB[None, :]).mean())
    res = {"n_H": [int(len(sA)), int(len(sB))], "same_model_AUC": auc, "gate_0.95_passed": bool(auc >= 0.95),
           "real_paired_contrast_own_minus_other": float(own_minus_other.mean()), "real_paired_contrast_se": float(own_minus_other.std(ddof=1)/np.sqrt(len(own_minus_other))),
           "mean_score_A_images_toward_A_minus_B": float(dA.mean()), "mean_score_B_images_toward_A_minus_B": float(dB.mean()),
           "fingerprint_cross_ncc_A_B": ncc(FA, FB), "qf_nets_used": sorted({A.qf_of(f) for f in SPL["Nikon_D200_1"]["E2"][:10]})}
    json.dump(res, open(os.path.join(V2, "out", "t2_noiseprint_gate.json"), "w"), indent=1); print(json.dumps(res, indent=1), flush=True)

def gens():
    FA, FB = np.load(os.path.join(V2, "out", "np_fingerprint_A.npy")), np.load(os.path.join(V2, "out", "np_fingerprint_B.npy"))
    out = {"arms": {}}; t0 = time.time()
    for arm in ARMS:
        fs = sorted(glob.glob(os.path.join(V2, "data", "gens", arm, "*.png")))
        s = np.array([[ncc(r, FA), ncc(r, FB)] for r in (nprint(f, f"gen_{arm}_{os.path.basename(f)}") for f in fs)])
        sign = 0 if arm == "base" else (1 if arm.startswith("A_") else -1)
        c = sign * (s[:, 0] - s[:, 1])
        out["arms"][arm] = {"n": len(fs), "mean_A": float(s[:, 0].mean()), "mean_B": float(s[:, 1].mean()),
                            "paired_contrast_toward_own": None if not sign else float(c.mean()), "se": float(c.std(ddof=1)/np.sqrt(len(c)))}
        print(f"[np] {arm}: {out['arms'][arm]} ({(time.time()-t0)/60:.1f} min)", flush=True)
        json.dump(out, open(os.path.join(V2, "out", "t2_noiseprint_gens.json"), "w"), indent=1)

if __name__ == "__main__":
    {"gate": gate, "gens": gens}[sys.argv[1] if len(sys.argv) > 1 else "gate"]()
