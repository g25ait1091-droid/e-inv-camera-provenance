"""Tier 2 — detector panel on the archive's own generations.

Detectors
  ncc    the paper's statistic: zero-lag NCC of the wavelet residual against Y*K (rho_mult)
  pce    classical Goljan PCE via prnu-python (its own extractor, its own K, neigh radius 2)
Also the Klier & Baier reproduction: unpaired PCE of every generation against each real
fingerprint at the conventional threshold of 60 -> false-positive rate, then the paired removal.

Inputs   out/fp/K_{A,B}_E2.npy, out/fp/H_{A,B}.npz, out/fp/manifest.json, data/gens/<arm>/*.png
Outputs  out/t2_rows.csv (one row per image x detector x fingerprint), out/t2_summary.json
Idempotent: rows already present are skipped, so it can be re-run as data arrives.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, time, csv
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(EINV.EXT, 'prnu-python'))
import prnu                                   # Bondi/Bestagini/Bonettini port of the Binghamton code
sys.path.insert(0, EINV.SRC)
from fingerprints import load_lum_crop, wavelet_residual, estimate_K, MEAS   # the paper's core

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); GENS = os.path.join(V2, "data", "gens")
OUT_ROWS = os.path.join(V2, "out", "t2_rows.csv"); OUT_SUM = os.path.join(V2, "out", "t2_summary.json")
ARMS = ["base", "A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16", "B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16"]
PCE_THRESH = 60.0
N_PER_ARM = int(os.environ.get("T2_N", "500"))

def main():
    man = json.load(open(os.path.join(FP, "manifest.json")))
    K = {r: np.load(os.path.join(FP, f"K_{r}_E2.npy")) for r in ("A", "B")}

    # ---- prnu-python fingerprints from the SAME E2 images, its own extractor -------------------------
    def rgb_crop(fp):
        with Image.open(fp) as im:
            im = im.convert("RGB"); W, H = im.size
            return np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.uint8)

    KP = {}
    for r in ("A", "B"):
        p = os.path.join(FP, f"Kpce_{r}_E2.npy")
        if not os.path.exists(p):
            t = time.time(); imgs = [rgb_crop(f) for f in man[r]["E2"]]
            np.save(p, prnu.extract_multiple_aligned(imgs, processes=8)); print(f"[t2] prnu-python K_{r} from {len(imgs)} E2 images ({time.time()-t:.0f}s)", flush=True)
        KP[r] = np.load(p)

    def ncc0(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

    # ---- low/mid DCT signature (notebook 10: 8x8 block DCT, keep 1 <= i+j <= 5, DC excluded) --------
    from scipy.fft import dctn, idctn
    from fingerprints import zero_mean, wiener_dft
    BLK, LO, HI = 8, 1, 5
    _M = np.zeros((BLK, BLK), np.float32)
    for i in range(BLK):
        for j in range(BLK):
            if LO <= i + j <= HI and not (i == 0 and j == 0): _M[i, j] = 1.0

    def lowmid_map(Y):
        n = MEAS // BLK
        B = Y.reshape(n, BLK, n, BLK).transpose(0, 2, 1, 3)
        D = dctn(B, axes=(2, 3), norm="ortho") * _M[None, None]
        R = idctn(D, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(MEAS, MEAS).astype(np.float32)
        R = R - R.mean(); return (R / float(R.std())).astype(np.float32)

    def estimate_L(files):
        acc = None
        for fp in files:
            m = lowmid_map(load_lum_crop(fp)); acc = m if acc is None else acc + m
        return wiener_dft(zero_mean((acc / len(files)).astype(np.float32)))

    L = {}
    for r in ("A", "B"):
        p = os.path.join(FP, f"L_{r}_E2.npy")
        if not os.path.exists(p):
            t = time.time(); np.save(p, estimate_L(man[r]["E2"])); print(f"[t2] low/mid L_{r} from E2 ({time.time()-t:.0f}s)", flush=True)
        L[r] = np.load(p)

    def measure(rgb_u8, lum):
        """Return dict of detector -> {A: value, B: value} for one image."""
        W = wavelet_residual(lum)
        out = {"ncc": {r: ncc0(W, lum*K[r]) for r in ("A", "B")}}
        w = prnu.extract_single(rgb_u8)
        out["pce"] = {}
        for r in ("A", "B"):
            cc = prnu.crosscorr_2d(KP[r], w); out["pce"][r] = float(prnu.pce(cc)["pce"])
        lm = lowmid_map(lum)
        out["lowmid"] = {r: ncc0(lm, L[r]) for r in ("A", "B")}
        return out

    done = set()
    if os.path.exists(OUT_ROWS):
        with open(OUT_ROWS) as f:
            for row in csv.DictReader(f): done.add((row["set"], row["image"]))
    fh = open(OUT_ROWS, "a", newline=""); wr = csv.writer(fh)
    if not done: wr.writerow(["set", "image", "ncc_A", "ncc_B", "pce_A", "pce_B", "lowmid_A", "lowmid_B"])

    # ---- real held-out photographs: the positive control for every detector -------------------------
    for r in ("A", "B"):
        for i, fp in enumerate(man[r]["H"]):
            key = (f"real_{r}", os.path.basename(fp))
            if key in done: continue
            m = measure(rgb_crop(fp), load_lum_crop(fp))
            wr.writerow([key[0], key[1], m["ncc"]["A"], m["ncc"]["B"], m["pce"]["A"], m["pce"]["B"], m["lowmid"]["A"], m["lowmid"]["B"]]); fh.flush()
    print("[t2] real H images done", flush=True)

    # ---- generations -------------------------------------------------------------------------------
    def gen_lum(rgb_u8):
        a = rgb_u8.astype(np.float32); return (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)

    for arm in ARMS:
        files = sorted(glob.glob(os.path.join(GENS, arm, "*.png")))[:N_PER_ARM]
        n = 0; t = time.time()
        for fp in files:
            key = (arm, os.path.basename(fp))
            if key in done: continue
            rgb = np.asarray(Image.open(fp).convert("RGB"), np.uint8)
            if rgb.shape[0] != MEAS: continue
            m = measure(rgb, gen_lum(rgb))
            wr.writerow([arm, key[1], m["ncc"]["A"], m["ncc"]["B"], m["pce"]["A"], m["pce"]["B"], m["lowmid"]["A"], m["lowmid"]["B"]]); fh.flush(); n += 1
        print(f"[t2] {arm}: {len(files)} files, {n} new rows ({time.time()-t:.0f}s)", flush=True)
    fh.close()

    # ---- summary -----------------------------------------------------------------------------------
    import pandas as pd
    d = pd.read_csv(OUT_ROWS)
    S = {}
    def auc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        return float((pos[:, None] > neg[None, :]).mean() + 0.5*(pos[:, None] == neg[None, :]).mean())
    for det in ("ncc", "pce", "lowmid"):
        rA, rB = d[d.set == "real_A"], d[d.set == "real_B"]
        S[det] = {"real_auc_same_model": auc(np.r_[rA[f"{det}_A"], rB[f"{det}_B"]], np.r_[rA[f"{det}_B"], rB[f"{det}_A"]]),
                  "real_paired_contrast": float(0.5*((rA[f"{det}_A"]-rA[f"{det}_B"]).mean() + (rB[f"{det}_B"]-rB[f"{det}_A"]).mean()))}
        arms = {}
        for arm in ARMS:
            g = d[d.set == arm]
            if not len(g): continue
            own, oth = ("A", "B") if arm.startswith("A_") else ("B", "A")
            arms[arm] = {"n": int(len(g)), "mean_A": float(g[f"{det}_A"].mean()), "mean_B": float(g[f"{det}_B"].mean()),
                         "paired_contrast": None if arm == "base" else float((g[f"{det}_{own}"] - g[f"{det}_{oth}"]).mean())}
            if det == "pce":
                arms[arm]["fpr_at_60_vs_A"] = float((g["pce_A"] > PCE_THRESH).mean())
                arms[arm]["fpr_at_60_vs_B"] = float((g["pce_B"] > PCE_THRESH).mean())
        S[det]["arms"] = arms
    json.dump(S, open(OUT_SUM, "w"), indent=1)
    print(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
