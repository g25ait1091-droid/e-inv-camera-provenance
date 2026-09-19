"""C6(b) estimator swap, registered in RESULTS.md Entry 58.

The primary statistic recomputed on the archived primary generations (24 adapters, first 250
images each) with E1 instead of E2 fingerprint estimates. Per image, exactly as _measure in
src/t1_measure.py: luminance Y, W = wavelet_residual(Y), rho = NCC(W, Y*K). Per adapter
theta = mean(rho(K_own) - rho(K_other)). Symmetric statistic theta_sym = (mean_A + mean_B)/2,
Welch adapter-level SE, one-sided 99 % limit theta_sym + t_{0.99,df} SE, as % of R_real.
Reading (Entry 58): robust if the symmetric limit changes by less than 25 %.
CPU only, 2 workers. Output: out/c6_estimator_swap.json (per-image rows included).
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); OUT = os.path.join(V2, "out", "c6_estimator_swap.json")
LEDGER = os.path.join(EINV.REPO, 'analysis', 'FINAL_LEDGER.json')
GENS = EINV.GENS; GENS_EXT = EINV.GENS_EXT      # the archived primary generations; set EINV_DATA
R_REAL = 0.0356703416571125; N_IMG = 250; WORKERS = 2; SEEDS = range(12)
TEMPL = ["K_A_E2", "K_B_E2", "K_A_E1", "K_B_E1"]

_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    _G.update(wr=wavelet_residual, MEAS=MEAS, K={t: np.load(os.path.join(FP, t + ".npy")) for t in TEMPL})

def _ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def _measure(task):
    tag, path = task; g = _G
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != g["MEAS"]: return (tag, os.path.basename(path), None)
    Y = (0.299*a[...,0] + 0.587*a[...,1] + 0.114*a[...,2]).astype(np.float32); W = g["wr"](Y)
    return (tag, os.path.basename(path), [_ncc(W, Y*g["K"][t]) for t in TEMPL])

def adapter_dir(body, s):
    return os.path.join(GENS if s <= 2 else GENS_EXT, f"{body}_raw_s{s}_r16")

def sym_stats(A, B):
    from scipy import stats
    A, B = np.asarray(A), np.asarray(B); nA, nB = len(A), len(B)
    vA, vB = A.var(ddof=1)/nA, B.var(ddof=1)/nB
    th = 0.5*(A.mean() + B.mean()); se = 0.5*np.sqrt(vA + vB)
    df = (vA + vB)**2 / (vA**2/(nA-1) + vB**2/(nB-1))
    UL = th + stats.t.ppf(0.99, df)*se
    tc = stats.t.ppf(0.995, nA-1)
    UA = A.mean() + tc*A.std(ddof=1)/np.sqrt(nA); UB = B.mean() + stats.t.ppf(0.995, nB-1)*B.std(ddof=1)/np.sqrt(nB)
    return {"mean_A": float(A.mean()), "sd_A": float(A.std(ddof=1)), "mean_B": float(B.mean()), "sd_B": float(B.std(ddof=1)),
            "additive_part": float(0.5*(A.mean() - B.mean())), "theta_sym": float(th), "welch_se": float(se), "welch_df": float(df),
            "t": float(th/se), "one_sided_p": float(1 - stats.t.cdf(th/se, df)),
            "sym_limit99": float(UL), "sym_limit99_pct_Rreal": float(100*UL/R_REAL),
            "maxarm_U": float(max(UA, UB)), "maxarm_U_pct_Rreal": float(100*max(UA, UB)/R_REAL)}

def main():
    t0 = time.time()
    tasks = []
    for body in ("A", "B"):
        for s in SEEDS:
            tag = f"{body}_raw_s{s}_r16"
            files = sorted(glob.glob(os.path.join(adapter_dir(body, s), "*.png")))[:N_IMG]
            assert len(files) == N_IMG, (tag, len(files))
            tasks += [(tag, f) for f in files]
    print(f"[c6b] {len(tasks)} images, {WORKERS} workers", flush=True)
    rows = {}
    with Pool(WORKERS, initializer=_init) as pool:
        for n, (tag, name, r) in enumerate(pool.imap_unordered(_measure, tasks, chunksize=4), 1):
            if r is None: print(f"[c6b] skipped {tag}/{name} (size)", flush=True); continue
            rows.setdefault(tag, {})[name] = r
            if n % 250 == 0: print(f"[c6b] {n}/{len(tasks)} {(time.time()-t0)/60:.1f} min", flush=True)
    led = json.load(open(LEDGER))["primary"]
    per = {"E2": {"A": [], "B": []}, "E1": {"A": [], "B": []}}; adapters = {}
    for body in ("A", "B"):
        for s in SEEDS:
            tag = f"{body}_raw_s{s}_r16"; M = np.array([rows[tag][k] for k in sorted(rows[tag])])  # cols KA2 KB2 KA1 KB1
            sgn = 1.0 if body == "A" else -1.0
            th2 = float(np.mean(sgn*(M[:, 0] - M[:, 1]))); th1 = float(np.mean(sgn*(M[:, 2] - M[:, 3])))
            se2 = float(np.std(sgn*(M[:, 0] - M[:, 1]), ddof=1)/np.sqrt(len(M))); se1 = float(np.std(sgn*(M[:, 2] - M[:, 3]), ddof=1)/np.sqrt(len(M)))
            lv = led[f"per_adapter_{body}"][s]
            per["E2"][body].append(th2); per["E1"][body].append(th1)
            adapters[tag] = {"n": int(len(M)), "theta_E2": th2, "theta_E2_img_se": se2, "theta_E1": th1, "theta_E1_img_se": se1,
                             "ledger_theta": lv, "ledger_n_images": 500 if s <= 2 else 250, "E2_minus_ledger": th2 - lv}
    res = {"E2": sym_stats(per["E2"]["A"], per["E2"]["B"]), "E1": sym_stats(per["E1"]["A"], per["E1"]["B"]),
           "ledger_values": sym_stats(led["per_adapter_A"], led["per_adapter_B"])}
    chg = (res["E1"]["sym_limit99"] - res["E2"]["sym_limit99"]) / res["E2"]["sym_limit99"]
    e2 = np.array(per["E2"]["A"] + per["E2"]["B"]); e1 = np.array(per["E1"]["A"] + per["E1"]["B"])
    lv = np.array(led["per_adapter_A"] + led["per_adapter_B"])
    ext = np.array([s >= 3 for _ in "AB" for s in SEEDS])
    out = {"_spec": "RESULTS.md Entry 58, C6(b)", "R_real": R_REAL, "n_images_per_adapter": N_IMG,
           "templates": TEMPL, "statistic": "t1_measure._measure: Y luminance, W=wavelet_residual(Y), rho=NCC(W,Y*K); theta = mean(rho_own - rho_other)",
           "summary": res,
           "sym_limit_change_E1_vs_E2_same_images": float(chg),
           "reading_threshold": 0.25,
           "reading": "robust" if abs(chg) < 0.25 else "not robust (symmetric limit changes by >= 25 %)",
           "reproduction_E2_vs_ledger": {
               "corr_all": float(np.corrcoef(e2, lv)[0, 1]),
               "max_abs_diff_all": float(np.abs(e2 - lv).max()),
               "max_abs_diff_ext_250img_adapters": float(np.abs(e2 - lv)[ext].max()),
               "max_abs_diff_s0to2_500img_adapters": float(np.abs(e2 - lv)[~ext].max()),
               "sym_limit_change_E2_250_vs_ledger": float((res["E2"]["sym_limit99"] - res["ledger_values"]["sym_limit99"]) / res["ledger_values"]["sym_limit99"])},
           "corr_E1_E2_per_adapter": float(np.corrcoef(e1, e2)[0, 1]),
           "adapters": adapters, "runtime_min": (time.time() - t0)/60,
           "rows_cols": ["rho_KA_E2", "rho_KB_E2", "rho_KA_E1", "rho_KB_E1"], "rows": rows}
    json.dump(out, open(OUT, "w"), indent=1)
    print(json.dumps({k: out[k] for k in ("summary", "sym_limit_change_E1_vs_E2_same_images", "reading", "reproduction_E2_vs_ledger", "corr_E1_E2_per_adapter", "runtime_min")}, indent=1), flush=True)

if __name__ == "__main__":
    main()
