"""Parallel driver for the Tier 2 detector panel (same detectors, same outputs as t2_panel.py).
Workers load the fingerprints once (initializer) and measure one image per task.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, glob, csv, time
import numpy as np
from PIL import Image
from multiprocessing import Pool

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); GENS = os.path.join(V2, "data", "gens")
OUT_ROWS = os.path.join(V2, "out", "t2_rows.csv"); OUT_SUM = os.path.join(V2, "out", "t2_summary.json")
ARMS = ["base", "A_raw_s0_r16", "A_raw_s1_r16", "A_raw_s2_r16", "B_raw_s0_r16", "B_raw_s1_r16", "B_raw_s2_r16"]
PCE_THRESH = 60.0; N_PER_ARM = int(os.environ.get("T2_N", "500")); WORKERS = int(os.environ.get("T2_W", "6"))
HDR = ["set", "image", "ncc_A", "ncc_B", "pce_A", "pce_B", "lowmid_A", "lowmid_B"]

_G = {}
def _init():
    sys.path.insert(0, os.path.join(EINV.EXT, 'prnu-python')); sys.path.insert(0, EINV.SRC)
    import prnu, torch
    torch.set_num_threads(1)
    from fingerprints import load_lum_crop, wavelet_residual, MEAS
    from scipy.fft import dctn, idctn
    BLK = 8; _M = np.zeros((BLK, BLK), np.float32)
    for i in range(BLK):
        for j in range(BLK):
            if 1 <= i + j <= 5 and not (i == 0 and j == 0): _M[i, j] = 1.0
    def lowmid_map(Y):
        n = MEAS // BLK; B = Y.reshape(n, BLK, n, BLK).transpose(0, 2, 1, 3)
        D = dctn(B, axes=(2, 3), norm="ortho") * _M[None, None]
        R = idctn(D, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(MEAS, MEAS).astype(np.float32)
        R = R - R.mean(); return (R / float(R.std())).astype(np.float32)
    def ncc0(a, b):
        a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))
    def rgb_crop(fp):
        with Image.open(fp) as im:
            im = im.convert("RGB"); W, H = im.size
            return np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.uint8)
    _G.update(prnu=prnu, load_lum_crop=load_lum_crop, wavelet_residual=wavelet_residual, lowmid_map=lowmid_map,
              ncc0=ncc0, rgb_crop=rgb_crop, MEAS=MEAS,
              K={r: np.load(os.path.join(FP, f"K_{r}_E2.npy")) for r in "AB"},
              KP={r: np.load(os.path.join(FP, f"Kpce_{r}_E2.npy")) for r in "AB"},
              L={r: np.load(os.path.join(FP, f"L_{r}_E2.npy")) for r in "AB"})

def _measure(task):
    setname, path, is_real = task
    g = _G; prnu = g["prnu"]
    if is_real:
        rgb = g["rgb_crop"](path); lum = g["load_lum_crop"](path)
    else:
        rgb = np.asarray(Image.open(path).convert("RGB"), np.uint8)
        if rgb.shape[0] != g["MEAS"]: return None
        a = rgb.astype(np.float32); lum = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32)
    W = g["wavelet_residual"](lum); w = prnu.extract_single(rgb); lm = g["lowmid_map"](lum)
    row = [setname, os.path.basename(path)]
    row += [g["ncc0"](W, lum*g["K"][r]) for r in "AB"]
    row += [float(prnu.pce(prnu.crosscorr_2d(g["KP"][r], w))["pce"]) for r in "AB"]
    row += [g["ncc0"](lm, g["L"][r]) for r in "AB"]
    return row

def main():
    man = json.load(open(os.path.join(FP, "manifest.json")))
    done = set()
    if os.path.exists(OUT_ROWS):
        with open(OUT_ROWS) as f:
            for row in csv.DictReader(f): done.add((row["set"], row["image"]))
    tasks = []
    for r in "AB":
        for fp in man[r]["H"]:
            if (f"real_{r}", os.path.basename(fp)) not in done: tasks.append((f"real_{r}", fp, True))
    for arm in ARMS:
        for fp in sorted(glob.glob(os.path.join(GENS, arm, "*.png")))[:N_PER_ARM]:
            if (arm, os.path.basename(fp)) not in done: tasks.append((arm, fp, False))
    print(f"[t2p] {len(done)} rows present, {len(tasks)} to measure with {WORKERS} workers", flush=True)
    new = not os.path.exists(OUT_ROWS) or os.path.getsize(OUT_ROWS) == 0
    fh = open(OUT_ROWS, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(HDR)
    t0 = time.time(); n = 0
    with Pool(WORKERS, initializer=_init) as pool:
        for row in pool.imap_unordered(_measure, tasks, chunksize=4):
            if row is None: continue
            wr.writerow(row); n += 1
            if n % 100 == 0: fh.flush(); print(f"[t2p] {n}/{len(tasks)}  {(time.time()-t0)/60:.1f} min", flush=True)
    fh.close(); print(f"[t2p] done {n} rows in {(time.time()-t0)/60:.1f} min", flush=True)
    # summary identical to t2_panel.py
    import pandas as pd
    d = pd.read_csv(OUT_ROWS); S = {}
    def auc(pos, neg):
        pos, neg = np.asarray(pos), np.asarray(neg)
        return float((pos[:, None] > neg[None, :]).mean() + 0.5*(pos[:, None] == neg[None, :]).mean())
    for det in ("ncc", "pce", "lowmid"):
        rA, rB = d[d.set == "real_A"], d[d.set == "real_B"]
        S[det] = {"real_auc_same_model": auc(np.r_[rA[f"{det}_A"], rB[f"{det}_B"]], np.r_[rA[f"{det}_B"], rB[f"{det}_A"]]),
                  "real_paired_contrast": float(0.5*((rA[f"{det}_A"]-rA[f"{det}_B"]).mean() + (rB[f"{det}_B"]-rB[f"{det}_A"]).mean())), "arms": {}}
        for arm in ARMS:
            g = d[d.set == arm]
            if not len(g): continue
            own, oth = ("A", "B") if arm.startswith("A_") else ("B", "A")
            e = {"n": int(len(g)), "mean_A": float(g[f"{det}_A"].mean()), "mean_B": float(g[f"{det}_B"].mean()),
                 "paired_contrast": None if arm == "base" else float((g[f"{det}_{own}"] - g[f"{det}_{oth}"]).mean()),
                 "paired_contrast_se": None if arm == "base" else float((g[f"{det}_{own}"] - g[f"{det}_{oth}"]).std(ddof=1)/np.sqrt(len(g)))}
            if det == "pce":
                e["fpr_at_60_vs_A"] = float((g["pce_A"] > PCE_THRESH).mean()); e["fpr_at_60_vs_B"] = float((g["pce_B"] > PCE_THRESH).mean())
                e["pce_A_median"] = float(g["pce_A"].median()); e["pce_B_median"] = float(g["pce_B"].median())
            S[det]["arms"][arm] = e
    json.dump(S, open(OUT_SUM, "w"), indent=1); print(json.dumps(S, indent=1))

if __name__ == "__main__":
    main()
