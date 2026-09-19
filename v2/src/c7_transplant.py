"""C7 (RESULTS.md Entry 58) — residual-transplant positive control for all four detectors.

Real noise: body A's (Nikon_D200_1) 40 held-out photographs (fingerprints.splits(...)["H"]), centred
1024^2 RGB crop (t1_ladder.load_rgb_crop convention). Per RGB CHANNEL, N_c = X_c - denoise(X_c), with the
study's wavelet residual operator (fingerprints.wavelet_residual) WITHOUT its final unit-std normalisation,
i.e. the residual in gray levels (n - mean(n)); the normalised version is checked equal to
fingerprints.wavelet_residual on image 0.
Generations: out/t1/gens/local_base/*.png, first 200 (sorted). Generation i is paired with residual i % 40.
Modified image Z_s = round(clip(Z + s * N, 0, 255)), rounded once, s in {0, 0.1, 0.25, 0.5, 1}; s = 0 is Z.
Detectors (paired contrast = value(->A) - value(->B)):
  ncc      rho(W(Y), Y*K_A_E2) - rho(W(Y), Y*K_B_E2)          as t1_measure._measure
  pce      PCE(Kpce_A_E2, w) - PCE(Kpce_B_E2, w), w = prnu.extract_single(rgb)   as t2_panel_par
  lowmid   ncc0(lowmid(Y), L_A_E2) - ncc0(lowmid(Y), L_B_E2)                   as t2_panel_par
  noiseprint  measured by src/c7_noiseprint_helper.py (a4tf1 env), fingerprints out/np_fingerprint_{A,B}.npy
Per detector and s: increment = contrast(s) - contrast(0) on the same image, mean, SE, paired t;
smallest s with t > 3. Reading (Entry 58): "insensitive in generated images" if nothing detected at s = 1.

Usage (base anaconda python):  c7_transplant.py residuals | measure | analyse
Env: C7_N (generations, 200), C7_W (workers, <= 2), C7_PILOT=1 (2 images, pilot rows file)
The residual stack (~0.5 GB) is scratch, not a result: it goes to C7_SCRATCH, else EINV_TMP, else
<EINV_V2>/tmp. Point one of them at a disk with room.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"     # "" deletes the variable on Windows; -1 hides every GPU (no CUDA call is made anyway)
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[_v] = "1"
import sys, json, glob, time, csv
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); GENS = os.path.join(V2, "out", "t1", "gens", "local_base")
OUT_JSON = os.path.join(V2, "out", "c7_transplant.json")
PILOT = os.environ.get("C7_PILOT") == "1"
ROWS = os.path.join(V2, "logs", "c7_rows_pilot.log" if PILOT else "c7_rows.log")
NP_ROWS = os.path.join(V2, "logs", "c7_noiseprint_rows.log")
SCR = os.environ.get("C7_SCRATCH", EINV.TMP)    # scratch only; nothing here is read back after a run
RES = os.path.join(SCR, "c7_residuals_A_H.npy"); RES_META = os.path.join(SCR, "c7_residuals_A_H.json")
S_GRID = [0.0, 0.1, 0.25, 0.5, 1.0]
N_GEN = 2 if PILOT else int(os.environ.get("C7_N", "200")); WORKERS = min(2, int(os.environ.get("C7_W", "1")))
N_RES = 40
HDR = ["idx", "image", "res_idx", "s", "ncc_A", "ncc_B", "pce_A", "pce_B", "peakA_y", "peakA_x",
       "lowmid_A", "lowmid_B", "rms_change", "frac_clipped", "checksum"]

def transplant(Z, N, s):
    """Z uint8 HxWx3, N float32 HxWx3 (gray levels). Rounded once. Identical arithmetic in the Noiseprint helper."""
    if s == 0.0: return Z.copy(), 0.0
    F = Z.astype(np.float32) + np.float32(s) * N
    clipped = float(((F < 0) | (F > 255)).mean())
    return np.clip(F, 0, 255).round().astype(np.uint8), clipped

# ------------------------------------------------------------------------------------------------ residuals
def raw_residual(img):
    """fingerprints.wavelet_residual without the final division by std: residual in the input's units."""
    import pywt
    from fingerprints import _conv2, SIG0, WAVELET, LEVELS
    s0 = np.float32(SIG0)**2
    co = pywt.wavedec2(img, WAVELET, level=LEVELS, mode="periodization"); out = [co[0]*0.0]
    for (cH, cV, cD) in co[1:]:
        band = []
        for c in (cH, cV, cD):
            c = c.astype(np.float32); vmin = None
            for w in (3, 5, 7, 9):
                v = np.maximum(_conv2(c*c, np.ones((w, w), np.float32)/(w*w)) - s0, 0.0)
                vmin = v if vmin is None else np.minimum(vmin, v)
            band.append(c*(s0/(vmin+s0)))
        out.append(tuple(band))
    n = pywt.waverec2(out, WAVELET, mode="periodization")[:img.shape[0], :img.shape[1]]
    return (n - n.mean()).astype(np.float32)

def residuals():
    import torch; torch.set_num_threads(1)
    from fingerprints import dv, splits, wavelet_residual, load_lum_crop, MEAS
    def load_rgb_crop(fp):          # verbatim t1_ladder.load_rgb_crop (not imported: t1_ladder creates dirs / loads CUDA config at import)
        with Image.open(fp) as im:
            im = im.convert("RGB"); W, H = im.size
            return np.asarray(im.crop(((W-MEAS)//2, (H-MEAS)//2, (W-MEAS)//2+MEAS, (H-MEAS)//2+MEAS)), np.float32)
    files = splits(dv["Nikon_D200_1"])["H"]
    man = json.load(open(os.path.join(FP, "manifest.json")))
    assert len(files) == N_RES and [os.path.basename(f) for f in files] == [os.path.basename(f) for f in man["A"]["H"]]
    t0 = time.time(); os.makedirs(SCR, exist_ok=True)
    Nst = np.lib.format.open_memmap(RES, mode="w+", dtype=np.float32, shape=(N_RES, 1024, 1024, 3))
    meta = {"files": [os.path.basename(f) for f in files], "rms_per_channel": [], "rms_lum_combination": []}
    # operator check: normalised raw residual == fingerprints.wavelet_residual (on the luminance crop of image 0)
    Y0 = load_lum_crop(files[0]); r0 = raw_residual(Y0); w0 = wavelet_residual(Y0)
    meta["operator_check_max_abs_diff"] = float(np.abs(r0/float(r0.std()) - w0).max())
    for k, f in enumerate(files):
        X = load_rgb_crop(f)
        for c in range(3): Nst[k, ..., c] = raw_residual(np.ascontiguousarray(X[..., c]))
        meta["rms_per_channel"].append([float(Nst[k, ..., c].std()) for c in range(3)])
        NL = 0.299*Nst[k, ..., 0] + 0.587*Nst[k, ..., 1] + 0.114*Nst[k, ..., 2]
        meta["rms_lum_combination"].append(float(NL.std()))
        print(f"[c7] residual {k+1}/{N_RES} rms {meta['rms_per_channel'][-1]} ({time.time()-t0:.0f}s)", flush=True)
    Nst.flush(); del Nst
    json.dump(meta, open(RES_META, "w"), indent=1)
    print(f"[c7] residuals done; operator check max|diff| = {meta['operator_check_max_abs_diff']:.3g}", flush=True)

# ------------------------------------------------------------------------------------------------ measure
_G = {}
def _init():
    sys.path.insert(0, os.path.join(EINV.EXT, 'prnu-python')); sys.path.insert(0, EINV.SRC)
    import prnu, torch
    torch.set_num_threads(1)
    from fingerprints import wavelet_residual
    from t2_poscontrol import lowmid_map, ncc0, gen_lum          # the panel's helpers (copied there from t2_panel)
    _G.update(prnu=prnu, wr=wavelet_residual, lowmid_map=lowmid_map, ncc0=ncc0, gen_lum=gen_lum,
              N=np.load(RES, mmap_mode="r"),
              K={r: np.load(os.path.join(FP, f"K_{r}_E2.npy")) for r in "AB"},
              KP={r: np.load(os.path.join(FP, f"Kpce_{r}_E2.npy")) for r in "AB"},
              L={r: np.load(os.path.join(FP, f"L_{r}_E2.npy")) for r in "AB"})

def _task(args):
    idx, path, todo = args; g = _G; prnu = g["prnu"]
    Z = np.asarray(Image.open(path).convert("RGB"), np.uint8)
    if Z.shape[:2] != (1024, 1024): return []
    ri = idx % N_RES; N = np.asarray(g["N"][ri], np.float32); out = []
    for s in todo:
        Zs, clipped = transplant(Z, N, s)
        rms = float(np.sqrt(np.mean((Zs.astype(np.float32) - Z.astype(np.float32))**2)))
        lum = g["gen_lum"](Zs); W = g["wr"](lum)
        nA, nB = (g["ncc0"](W, lum*g["K"][r]) for r in "AB")                 # = t1_measure._ncc(W, Y*K)
        w = prnu.extract_single(Zs)
        pA = prnu.pce(prnu.crosscorr_2d(g["KP"]["A"], w)); pB = prnu.pce(prnu.crosscorr_2d(g["KP"]["B"], w))
        lm = g["lowmid_map"](lum); lA, lB = (g["ncc0"](lm, g["L"][r]) for r in "AB")
        out.append([idx, os.path.basename(path), ri, s, nA, nB, float(pA["pce"]), float(pB["pce"]),
                    int(pA["peak"][0]), int(pA["peak"][1]), lA, lB, rms, clipped, int(Zs.sum(dtype=np.int64))])
    return out

def measure():
    from multiprocessing import Pool
    files = sorted(glob.glob(os.path.join(GENS, "*.png")))[:N_GEN]
    done = set()
    if os.path.exists(ROWS):
        for r in csv.DictReader(open(ROWS)): done.add((int(r["idx"]), float(r["s"])))
    tasks = [(i, f, [s for s in S_GRID if (i, s) not in done]) for i, f in enumerate(files)]
    tasks = [t for t in tasks if t[2]]
    print(f"[c7] {len(files)} generations, {len(done)} rows present, {len(tasks)} images with work, {WORKERS} worker(s)", flush=True)
    new = not os.path.exists(ROWS); fh = open(ROWS, "a", newline=""); wr = csv.writer(fh)
    if new: wr.writerow(HDR); fh.flush()
    t0 = time.time(); k = 0
    if WORKERS == 1:
        _init(); it = map(_task, tasks)
    else:
        pool = Pool(WORKERS, initializer=_init); it = pool.imap_unordered(_task, tasks, chunksize=1)
    for out in it:
        for r in out: wr.writerow(r)
        fh.flush(); k += 1
        if k % 10 == 0 or PILOT:
            el = time.time()-t0; print(f"[c7] {k}/{len(tasks)} images {el/60:.1f} min  eta {el/k*(len(tasks)-k)/60:.0f} min", flush=True)
    fh.close(); print(f"[c7] measure done in {(time.time()-t0)/60:.1f} min", flush=True)

# ------------------------------------------------------------------------------------------------ analyse
def analyse():
    import pandas as pd
    from scipy import stats as sps
    d = pd.read_csv(ROWS)
    d = d.drop_duplicates(["idx", "s"])
    frames = {det: d.assign(vA=d[f"{det}_A"], vB=d[f"{det}_B"]) for det in ("ncc", "pce", "lowmid")}
    npd = None
    if os.path.exists(NP_ROWS):
        npd = pd.read_csv(NP_ROWS).drop_duplicates(["idx", "s"]); frames["noiseprint"] = npd.assign(vA=npd["np_A"], vB=npd["np_B"])
    summ = json.load(open(os.path.join(V2, "out", "t2_summary.json")))
    gate = json.load(open(os.path.join(V2, "out", "t2_noiseprint_gate.json")))
    R = {det: float(summ[det]["real_paired_contrast"]) for det in ("ncc", "pce", "lowmid")}
    R["noiseprint"] = float(gate["real_paired_contrast_own_minus_other"])
    t2 = pd.read_csv(os.path.join(V2, "out", "t2_rows.csv")); rA = t2[t2.set == "real_A"]
    RA = {det: float((rA[f"{det}_A"] - rA[f"{det}_B"]).mean()) for det in ("ncc", "pce", "lowmid")}
    RA["noiseprint"] = float(gate["mean_score_A_images_toward_A_minus_B"])
    meta = json.load(open(RES_META))

    def block(g, det):
        g = g.assign(c=g.vA - g.vB)
        piv = g.pivot(index="idx", columns="s", values="c").dropna()
        pA = g.pivot(index="idx", columns="s", values="vA").loc[piv.index]; pB = g.pivot(index="idx", columns="s", values="vB").loc[piv.index]
        rms = g.pivot(index="idx", columns="s", values="rms_change").loc[piv.index] if "rms_change" in g else None
        base = piv[0.0].to_numpy(); n = len(base); ri = piv.index.to_numpy() % N_RES; per = []
        for s in sorted(piv.columns):
            v = piv[s].to_numpy(); dv = v - base
            e = {"s": float(s), "mean_contrast": float(v.mean()), "mean_A": float(pA[s].mean()), "mean_B": float(pB[s].mean())}
            if s > 0:
                se = float(dv.std(ddof=1)/np.sqrt(n)); tcrit = float(sps.t.ppf(0.995, n-1))
                cm = pd.Series(dv).groupby(ri).mean().to_numpy()                 # secondary: residual-clustered
                e.update({"increment": float(dv.mean()), "increment_se": se, "increment_t": float(dv.mean()/se) if se > 0 else None,
                          "df": n-1, "one_sided_p": float(sps.t.sf(dv.mean()/se, n-1)) if se > 0 else None,
                          "increment_ci99": [float(dv.mean()-tcrit*se), float(dv.mean()+tcrit*se)],
                          "increment_pct_R_real_paired": float(100*dv.mean()/R[det]),
                          "increment_pct_real_A_photo_contrast": float(100*dv.mean()/RA[det]),
                          "cluster_by_residual": {"n_clusters": int(len(cm)), "t": float(cm.mean()/(cm.std(ddof=1)/np.sqrt(len(cm))))}})
            if rms is not None: e["stored_change_rms_gray"] = float(rms[s].mean())
            if det == "pce":
                e["frac_pceA_gt60"] = float((pA[s] > 60).mean()); e["median_pce_A"] = float(pA[s].median())
                pk = g[(g.s == s) & g.idx.isin(piv.index)]
                e["frac_peakA_at_zero_lag"] = float(((pk.peakA_y == 1023) & (pk.peakA_x == 1023)).mean())
            per.append(e)
        det_s = next((e for e in per if e["s"] > 0 and e["increment_t"] is not None and e["increment_t"] > 3), None)
        at1 = next(e for e in per if e["s"] == 1.0)
        return {"n_images": int(n), "R_real_paired_contrast": R[det], "real_A_photo_contrast": RA[det], "per_s": per,
                "smallest_s_detected_t_gt_3": None if det_s is None else det_s["s"],
                "reading": "insensitive in generated images" if not (at1["increment_t"] is not None and at1["increment_t"] > 3)
                           else "detects the transplant"}
    res = {det: block(g, det) for det, g in frames.items()}
    checks = {"operator_check_max_abs_diff": meta["operator_check_max_abs_diff"]}
    # s = 0 NCC reproduces the archived t1 measurement of the same files (measure_rows_f5.csv, arm local_base)
    f5 = pd.read_csv(os.path.join(V2, "out", "t1", "measure_rows_f5.csv")); f5 = f5[f5.arm == "local_base"].set_index("image")
    z = d[d.s == 0.0].set_index("image")
    checks["ncc_s0_vs_measure_rows_f5_max_abs_diff"] = float(np.abs((z.ncc_A - z.ncc_B) - (f5.rho_KA - f5.rho_KB).loc[z.index]).max())
    if npd is not None:
        m = npd.merge(d[["idx", "s", "checksum"]], on=["idx", "s"], suffixes=("_np", ""))
        checks["noiseprint_images_identical_to_main"] = {"n_compared": int(len(m)), "all_equal": bool((m.checksum == m.checksum_np).all())}
        sub = sorted(set(npd.idx))
        if len(sub) < d.idx.nunique():   # other detectors on the Noiseprint subset, for like-for-like comparison
            res["same_images_as_noiseprint"] = {det: block(frames[det][frames[det].idx.isin(sub)], det) for det in ("ncc", "pce", "lowmid")}
    out = {"entry": "RESULTS.md Entry 58, C7", "script": "src/c7_transplant.py (+ src/c7_noiseprint_helper.py)", "date": time.strftime("%Y-%m-%d %H:%M"),
           "design": {"real_source": "body A Nikon_D200_1, H split (40), centred 1024^2 RGB crop",
                      "residual": "per RGB channel, X_c - denoise(X_c) with fingerprints.wavelet_residual's operator, un-normalised (gray levels)",
                      "residual_rms_gray_mean_per_channel": np.mean(meta["rms_per_channel"], 0).tolist(),
                      "residual_rms_gray_lum_combination_mean": float(np.mean(meta["rms_lum_combination"])),
                      "generations": "out/t1/gens/local_base, first %d sorted" % d.idx.nunique(), "pairing": "generation i <- residual i mod 40",
                      "transplant": "Z_s = round(clip(Z + s*N, 0, 255)), rounded once", "s": S_GRID,
                      "contrast": "value(->A) - value(->B); increment = contrast(s) - contrast(0), same image; paired t, df n-1",
                      "detection": "t > 3", "reading": "insensitive in generated images if increment t <= 3 at s = 1",
                      "secondary": "cluster_by_residual: t over the 40 residual means (each residual is reused 5x)",
                      "cpu_only": True},
           "checks": checks, "results": res}
    json.dump(out, open(OUT_JSON, "w"), indent=1)
    for det, r in res.items():
        if det == "same_images_as_noiseprint": continue
        print(f"\n== {det} n={r['n_images']} R={r['R_real_paired_contrast']:.4g} realA={r['real_A_photo_contrast']:.4g} smallest s={r['smallest_s_detected_t_gt_3']} -> {r['reading']}")
        for e in r["per_s"]:
            if e["s"] > 0: print(f"  s={e['s']:<5} incr {e['increment']:+.4e}  t {e['increment_t']:.2f}  {e['increment_pct_real_A_photo_contrast']:+.1f}% realA  clus t {e['cluster_by_residual']['t']:.2f}")
    print("\nchecks", json.dumps(checks))

if __name__ == "__main__":
    {"residuals": residuals, "measure": measure, "analyse": analyse}[sys.argv[1]]()
