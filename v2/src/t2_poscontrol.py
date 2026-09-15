"""Tier 2 positive controls in GENERATED images for PCE and the low/mid DCT signature
(with the paper's NCC statistic run through the same pipeline as a cross-check).

Why: only NCC was shown to be sensitive in generated images (supplement S2); PCE and low/mid
were gated on real photographs only. Here a known template is injected into already-generated
images by the S2 calibration procedure and each detector's paired contrast is measured.

Injection (supplement S2, notebook 14 convention): the decoder's pre-quantization float is modelled
as stored integer + U(-0.5, 0.5) per pixel and channel, the dither seeded per image and identical
across amplitudes and detectors (paired); G = F * (1 + alpha * T) on every RGB channel, clipped to
[0, 255] and rounded once.

Templates injected (all from body A's E1 split, disjoint from the E2 split every panel reference uses):
  ncc     T = K_A_E1 (out/fp, fingerprints.py estimator)        measured: NCC(W(Y), Y*K_{A,B}_E2)
  pce     T = prnu-python K from A's E1 images (computed here,  measured: PCE vs Kpce_{A,B}_E2
          same call t2_panel used for E2; not saved)                      (neigh radius 2)
  lowmid  T = L_A from A's E1 images (t2_panel's estimate_L,    measured: NCC(lowmid(Y), L_{A,B}_E2)
          computed here; not saved)
Paired contrast per image = value(->A) - value(->B), as in the panel.

Inputs   out/fp/{K_A_E1,K_{A,B}_E2,Kpce_{A,B}_E2,L_{A,B}_E2}.npy, out/fp/manifest.json,
         out/t1/gens/local_base/*.png
Outputs  out/t2_poscontrol.json; per-row log logs/t2_poscontrol_rows.log (resumable)
Env      T2PC_N (images, default 500), T2PC_W (workers, <= 6), T2PC_PILOT=1 (2 images, no JSON)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import sys, json, glob, time, csv
import numpy as np
from PIL import Image
from multiprocessing import Pool

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); GENS = os.path.join(V2, "out", "t1", "gens", "local_base")
OUT_JSON = os.path.join(V2, "out", "t2_poscontrol.json"); ROWS = os.path.join(V2, "logs", "t2_poscontrol_rows.log")
PILOT = os.environ.get("T2PC_PILOT") == "1"
N = 2 if PILOT else int(os.environ.get("T2PC_N", "500")); WORKERS = min(6, int(os.environ.get("T2PC_W", "6")))
DITHER_SEED = 20260913
GRID = [0.0, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0]
GRID_EXT = GRID + [2.0, 5.0, 10.0]
SUB = [0.001, 0.002, 0.005]   # added after the first pass: ncc and low/mid were already at t > 3 at 0.01
ALPHAS = {"ncc": sorted(SUB + GRID), "pce": GRID_EXT, "lowmid": sorted(SUB + GRID_EXT)}
# The three templates live on different native scales (prnu-python K std ~0.37, L std ~0.017,
# K_A_E1 std ~1.0e-3), so the PCE and low/mid templates are rescaled to the RMS of K_A_E1 before
# injection: nominal alpha then means the same injected multiplicative RMS for every detector, and
# alpha = 1 is the RMS of the study's own real-fingerprint estimate. NCC's template is unscaled
# (identical to the S2 calibration). native_alpha = alpha * scale is recorded per template.
MATCH_RMS_TO = "ncc"
PCE_THRESH = 60.0
# real-photograph paired contrasts R, from out/t2_summary.json (t2_panel_par.py); re-read at run time
R_PAPER = {"ncc": 0.03566, "pce": 367.8, "lowmid": 0.002154}
NCC_PAPER = {"n": 2500, "slope": 1.4142e-2, "slope_lower99": 1.3613e-2, "incr_at_0.02": 2.83e-4,
             "incr_pct_R_at_0.02": 0.78, "table": {0.0: -6.62e-5, 0.01: 7.52e-5, 0.02: 2.17e-4, 0.05: 6.41e-4,
                                                    0.1: 1.35e-3, 0.25: 3.47e-3}}
HDR = ["det", "idx", "image", "alpha", "vA", "vB", "rms_change", "peakA_y", "peakA_x"]

# ------------------------------------------------------------------------------------------------
# shared helpers, copied from t2_panel.py / t2_panel_par.py so the detectors are identical
BLK = 8
_M = np.zeros((BLK, BLK), np.float32)
for _i in range(BLK):
    for _j in range(BLK):
        if 1 <= _i + _j <= 5 and not (_i == 0 and _j == 0): _M[_i, _j] = 1.0

def lowmid_map(Y, MEAS=1024):
    from scipy.fft import dctn, idctn
    n = MEAS // BLK; B = Y.reshape(n, BLK, n, BLK).transpose(0, 2, 1, 3)
    D = dctn(B, axes=(2, 3), norm="ortho") * _M[None, None]
    R = idctn(D, axes=(2, 3), norm="ortho").transpose(0, 2, 1, 3).reshape(MEAS, MEAS).astype(np.float32)
    R = R - R.mean(); return (R / float(R.std())).astype(np.float32)

def ncc0(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def gen_lum(rgb):
    a = rgb.astype(np.float32); return (0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]).astype(np.float32)

def rgb_crop(fp, MEAS=1024):
    with Image.open(fp) as im:
        im = im.convert("RGB"); W, H = im.size
        return np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.uint8)

# ------------------------------------------------------------------------------------------------
_G = {}
def _init(T):
    sys.path.insert(0, os.path.join(EINV.EXT, 'prnu-python')); sys.path.insert(0, os.path.join(V2, "src"))
    import prnu, torch
    torch.set_num_threads(1)
    from fingerprints import wavelet_residual
    _G.update(prnu=prnu, wavelet_residual=wavelet_residual, T=T,
              K={r: np.load(os.path.join(FP, f"K_{r}_E2.npy")) for r in "AB"},
              KP={r: np.load(os.path.join(FP, f"Kpce_{r}_E2.npy")) for r in "AB"},
              L={r: np.load(os.path.join(FP, f"L_{r}_E2.npy")) for r in "AB"})

def _task(args):
    idx, path, todo = args            # todo: list of (det, alpha) still to measure for this image
    g = _G; prnu = g["prnu"]
    Q = np.asarray(Image.open(path).convert("RGB"), np.uint8).astype(np.float32)
    if Q.shape[0] != 1024 or Q.shape[1] != 1024: return []
    rng = np.random.default_rng(DITHER_SEED + idx)                 # per-image dither, shared by all alphas/detectors
    F = Q + (rng.random(Q.shape, dtype=np.float32) - np.float32(0.5))
    out = []
    for det, a in todo:
        T = g["T"][det]
        Qa = np.clip(F * (np.float32(1.0) + np.float32(a) * T[..., None]), 0, 255).round()
        rms = float(np.sqrt(np.mean((Qa - Q) ** 2)))
        py = px = -1
        if det == "ncc":
            lum = gen_lum(Qa); W = g["wavelet_residual"](lum)
            vA, vB = (ncc0(W, lum*g["K"][r]) for r in "AB")
        elif det == "pce":
            w = prnu.extract_single(Qa.astype(np.uint8))
            pA = prnu.pce(prnu.crosscorr_2d(g["KP"]["A"], w)); pB = prnu.pce(prnu.crosscorr_2d(g["KP"]["B"], w))
            vA, vB = float(pA["pce"]), float(pB["pce"]); py, px = int(pA["peak"][0]), int(pA["peak"][1])
        else:
            lm = lowmid_map(gen_lum(Qa))
            vA, vB = (ncc0(lm, g["L"][r]) for r in "AB")
        out.append([det, idx, os.path.basename(path), a, vA, vB, rms, py, px])
    return out

# ------------------------------------------------------------------------------------------------
def build_templates(man):
    sys.path.insert(0, os.path.join(EINV.EXT, 'prnu-python')); sys.path.insert(0, os.path.join(V2, "src"))
    import prnu
    from fingerprints import load_lum_crop, zero_mean, wiener_dft
    T = {"ncc": np.load(os.path.join(FP, "K_A_E1.npy")).astype(np.float32)}
    t = time.time(); imgs = [rgb_crop(f) for f in man["A"]["E1"]]
    T["pce"] = prnu.extract_multiple_aligned(imgs, processes=WORKERS).astype(np.float32); del imgs
    print(f"[pc] prnu-python K_A from {len(man['A']['E1'])} E1 images ({time.time()-t:.0f}s)", flush=True)
    t = time.time(); acc = None
    for fp in man["A"]["E1"]:
        m = lowmid_map(load_lum_crop(fp)); acc = m if acc is None else acc + m
    T["lowmid"] = wiener_dft(zero_mean((acc / len(man["A"]["E1"])).astype(np.float32))).astype(np.float32)
    print(f"[pc] low/mid L_A from E1 ({time.time()-t:.0f}s)", flush=True)
    native_std = {det: float(T[det].std()) for det in T}
    scale = {det: native_std[MATCH_RMS_TO] / native_std[det] for det in T}   # 1.0 for ncc
    for det in T: T[det] = (T[det] * np.float32(scale[det])).astype(np.float32)
    return T, native_std, scale

def analyse(rows, R, T_stats, n_img, runtime_min):
    from scipy import stats as sps
    import pandas as pd
    d = pd.DataFrame(rows, columns=HDR); d["c"] = d.vA - d.vB
    res = {}
    for det in ("ncc", "pce", "lowmid"):
        g = d[d.det == det]
        if not len(g): continue
        piv = g.pivot(index="idx", columns="alpha", values="c").dropna(); al = np.array(sorted(piv.columns))
        pA = g.pivot(index="idx", columns="alpha", values="vA").loc[piv.index]
        pB = g.pivot(index="idx", columns="alpha", values="vB").loc[piv.index]
        rms = g.pivot(index="idx", columns="alpha", values="rms_change").loc[piv.index]
        base = piv[0.0].to_numpy(); n = len(base); per = []
        for a in al:
            v = piv[a].to_numpy(); se = v.std(ddof=1)/np.sqrt(n); dv = v - base; dse = dv.std(ddof=1)/np.sqrt(n)
            tcrit = sps.t.ppf(0.995, n-1)
            e = {"alpha": float(a), "mean_contrast": float(v.mean()), "se": float(se), "t_vs_zero": float(v.mean()/se),
                 "mean_A": float(pA[a].mean()), "mean_B": float(pB[a].mean()),
                 "increment": float(dv.mean()), "increment_se": float(dse),
                 "increment_t": float(dv.mean()/dse) if dse > 0 else None,
                 "increment_ci99": [float(dv.mean()-tcrit*dse), float(dv.mean()+tcrit*dse)] if dse > 0 else None,
                 "increment_pct_R": float(100*dv.mean()/R[det]),
                 "stored_change_rms_gray": float(rms[a].mean())}
            if det == "pce":
                e["frac_pceA_gt60"] = float((pA[a] > PCE_THRESH).mean()); e["frac_pceB_gt60"] = float((pB[a] > PCE_THRESH).mean())
                e["median_pce_A"] = float(pA[a].median()); e["median_pce_B"] = float(pB[a].median())
                pk = g[g.alpha == a]; pk = pk[pk.idx.isin(piv.index)]
                e["frac_peakA_at_zero_lag"] = float(((pk.peakA_y == 1023) & (pk.peakA_x == 1023)).mean())
            per.append(e)
        def slope_fit(sub):
            x = sub - sub.mean(); Y = piv[sub].to_numpy()
            sl = ((Y - Y.mean(1, keepdims=True)) * x).sum(1) / (x**2).sum()
            m = float(sl.mean()); s = float(sl.std(ddof=1)/np.sqrt(len(sl)))
            return {"alphas": [float(a) for a in sub], "slope": m, "slope_se": s,
                    "slope_lower99": float(m - sps.norm.ppf(0.99)*s), "intercept_alpha0": float(base.mean())}
        fits = {"full_grid": slope_fit(al), "alpha_le_1": slope_fit(al[al <= 1 + 1e-9]),
                "alpha_le_0.25": slope_fit(al[al <= 0.25 + 1e-9])}
        det_a = next((e for e in per if e["alpha"] > 0 and e["increment_t"] is not None and e["increment_t"] > 3), None)
        det_z = next((e for e in per if e["alpha"] > 0 and e["t_vs_zero"] > 3), None)   # supplement-S2-style criterion
        res[det] = {"n_images": int(n), "R_real_paired_contrast": R[det], "template_injected": T_stats[det],
                    "per_alpha": per, "slope": fits,
                    "detection": None if det_a is None else
                        {"smallest_alpha_increment_t_gt_3": det_a["alpha"], "increment": det_a["increment"],
                         "increment_t": det_a["increment_t"], "increment_pct_R": det_a["increment_pct_R"],
                         "stored_change_rms_gray": det_a["stored_change_rms_gray"],
                         "at_smallest_grid_point": bool(det_a["alpha"] == min(a for a in al if a > 0))},
                    "detection_contrast_t_vs_zero_gt_3": None if det_z is None else
                        {"alpha": det_z["alpha"], "mean_contrast": det_z["mean_contrast"], "t_vs_zero": det_z["t_vs_zero"],
                         "increment_pct_R": det_z["increment_pct_R"]}}
    # NCC cross-check against supplement S2 (n = 2500, dither model, different image set)
    if "ncc" in res:
        e02 = next(e for e in res["ncc"]["per_alpha"] if abs(e["alpha"]-0.02) < 1e-9)
        sfit = res["ncc"]["slope"]["alpha_le_0.25"]
        res["ncc_crosscheck"] = {
            "paper": NCC_PAPER,
            "this_run_incr_at_0.02": e02["increment"], "this_run_incr_ci99_at_0.02": e02["increment_ci99"],
            "paper_incr_inside_this_ci99": bool(e02["increment_ci99"][0] <= NCC_PAPER["incr_at_0.02"] <= e02["increment_ci99"][1]),
            "this_run_slope_alpha_le_0.25": sfit["slope"], "slope_ratio_to_paper": sfit["slope"]/NCC_PAPER["slope"],
            "slope_z_vs_paper": (sfit["slope"]-NCC_PAPER["slope"])/sfit["slope_se"]}
    return res

def main():
    man = json.load(open(os.path.join(FP, "manifest.json")))
    summ = json.load(open(os.path.join(V2, "out", "t2_summary.json")))
    R = {det: float(summ[det]["real_paired_contrast"]) for det in ("ncc", "pce", "lowmid")}
    for det in R: assert abs(R[det]/R_PAPER[det] - 1) < 1e-3, (det, R[det], R_PAPER[det])
    files = sorted(glob.glob(os.path.join(GENS, "*.png")))[:N]
    print(f"[pc] {len(files)} images, {WORKERS} workers, R = {R}", flush=True)
    t0 = time.time()
    T, native_std, scale = build_templates(man)
    T_stats = {det: {"std_injected": float(T[det].std()), "absmax_injected": float(np.abs(T[det]).max()),
                     "std_native": native_std[det], "scale_injected_over_native": scale[det],
                     "native_alpha_per_nominal_alpha": scale[det],
                     "source": {"ncc": "out/fp/K_A_E1.npy (fingerprints.estimate_K, 80 E1 images)",
                                "pce": "prnu.extract_multiple_aligned on A's 80 E1 RGB centre crops (same call as t2_panel for E2)",
                                "lowmid": "t2_panel estimate_L on A's 80 E1 images"}[det],
                     "ncc_with_E2_reference_A": ncc0(T[det], np.load(os.path.join(FP, {"ncc": "K_A_E2", "pce": "Kpce_A_E2", "lowmid": "L_A_E2"}[det] + ".npy"))),
                     "ncc_with_E2_reference_B": ncc0(T[det], np.load(os.path.join(FP, {"ncc": "K_B_E2", "pce": "Kpce_B_E2", "lowmid": "L_B_E2"}[det] + ".npy")))}
               for det in T}
    print("[pc] templates:", json.dumps(T_stats), flush=True)

    rows_path = ROWS if not PILOT else ROWS.replace(".log", "_pilot.log")
    rows, done = [], set()
    if os.path.exists(rows_path):
        with open(rows_path) as f:
            for r in csv.DictReader(f):
                row = [r["det"], int(r["idx"]), r["image"], float(r["alpha"]), float(r["vA"]), float(r["vB"]),
                       float(r["rms_change"]), int(r["peakA_y"]), int(r["peakA_x"])]
                rows.append(row); done.add((row[0], row[1], row[3]))
    tasks = []
    for i, fp in enumerate(files):
        todo = [(det, a) for det in ("pce", "ncc", "lowmid") for a in ALPHAS[det] if (det, i, a) not in done]
        if todo: tasks.append((i, fp, todo))
    print(f"[pc] {len(rows)} rows present; {len(tasks)} images with work", flush=True)
    new = not os.path.exists(rows_path)
    fh = open(rows_path, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(HDR); fh.flush()
    t1 = time.time(); k = 0
    with Pool(WORKERS, initializer=_init, initargs=(T,)) as pool:
        for out in pool.imap_unordered(_task, tasks, chunksize=1):
            for r in out: wr.writerow(r); rows.append(r)
            fh.flush(); k += 1
            if k % 10 == 0 or PILOT:
                el = time.time()-t1; print(f"[pc] {k}/{len(tasks)} images  {el/60:.1f} min  eta {el/k*(len(tasks)-k)/60:.0f} min", flush=True)
    fh.close()
    runtime = (time.time()-t0)/60
    print(f"[pc] measurement done ({runtime:.1f} min this session)", flush=True)
    res = analyse(rows, R, T_stats, len(files), runtime)
    out = {"script": "src/t2_poscontrol.py", "date": time.strftime("%Y-%m-%d %H:%M"),
           "images": {"dir": "out/t1/gens/local_base", "n": len(files), "first": os.path.basename(files[0]), "last": os.path.basename(files[-1])},
           "settings": {"alphas": ALPHAS, "dither": "F = stored + U(-0.5,0.5) per pixel and channel, default_rng(%d + image index), shared across alphas and detectors" % DITHER_SEED,
                        "injection": "clip(F*(1+alpha*T[...,None]), 0, 255).round() on RGB; detectors then read the image exactly as t2_panel reads a generation",
                        "template_scaling": "PCE and low/mid templates rescaled to the RMS of K_A_E1 (NCC's template, unscaled); see results.<det>.template_injected",
                        "contrast": "value(->A) - value(->B); increment = contrast(alpha) - contrast(0) on the same image",
                        "t_crit_detection": 3.0, "slope": "per-image OLS slope over alphas, mean +/- SE, lower99 = mean - z_0.99*SE (notebook 14 convention)",
                        "workers": WORKERS, "cpu_only": True, "pilot": PILOT},
           "R_real_paired_contrast": R, "runtime_min_last_session": runtime, "results": res}
    if not PILOT:
        json.dump(out, open(OUT_JSON, "w"), indent=1)
    for det in ("ncc", "pce", "lowmid"):
        if det not in res: continue
        print(f"\n== {det}  n={res[det]['n_images']}  R={R[det]:.5g}")
        for e in res[det]["per_alpha"]:
            extra = f"  PCE_A>60 {e['frac_pceA_gt60']:.3f}  medA {e['median_pce_A']:.1f}  zero-lag {e['frac_peakA_at_zero_lag']:.2f}" if det == "pce" else ""
            print(f"  a={e['alpha']:<6g} C={e['mean_contrast']:+.4e}  incr={e['increment']:+.4e} (t {e['increment_t'] if e['increment_t'] is None else round(e['increment_t'],2)})  {e['increment_pct_R']:+.4f}%R  rms {e['stored_change_rms_gray']:.3f}{extra}")
        print("  slope", json.dumps(res[det]["slope"]), "\n  detection", res[det]["detection"])
    if "ncc_crosscheck" in res: print("\nncc cross-check", json.dumps(res["ncc_crosscheck"]))

if __name__ == "__main__":
    main()
