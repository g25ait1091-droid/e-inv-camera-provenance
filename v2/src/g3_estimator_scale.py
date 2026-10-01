"""G3 (RESULTS.md Entry 76, deviation recorded in Entry 79) - how much of the primary limit
depends on the fingerprint estimate.

The primary statistic recomputed over all 24 primary adapters and every generation archived locally
(500 for seeds 0-2 of each body, 250 beyond; 7,500 images), under
six estimates per body: E2 (140 photographs, the published one), E1 (80), pooled E1+E2 (220, same
estimator), and two disjoint 70-image halves of E2. Each is divided by its own real-image contrast,
measured on the held-out H split with the same estimate, so the estimates are compared on their own
scales rather than on E2's.

Per image, exactly as _measure in t1_measure.py: Y = luminance, W = wavelet_residual(Y),
rho = NCC(W, Y*K). Per adapter theta = mean(rho(K_own) - rho(K_other)); theta_sym = (mean_A + mean_B)/2
with a Welch adapter-level SE and a one-sided 99 % limit. CPU only. Writes out/g3_estimator_scale.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp"); OUT = os.path.join(V2, "out", "g3_estimator_scale.json")
GENS, GENS_EXT = EINV.GENS, EINV.GENS_EXT
N_IMG = 500; WORKERS = int(os.environ.get("G3_W", "6")); SEEDS = range(12)
EST = ["E2", "E1", "P", "H1", "H2"]                      # P = pooled E1+E2; H1/H2 = halves of E2
TEMPL = [f"K_{r}_{e}" for e in EST for r in ("A", "B")]

def build_templates():
    """E1 and E2 exist; build pooled and the two E2 halves once, with the study's estimator."""
    from fingerprints import dv, splits, estimate_K
    made = {}
    for role, did in (("A", "Nikon_D200_1"), ("B", "Nikon_D200_0")):
        sp = splits(dv[did])
        for tag, files in (("P", sp["E1"] + sp["E2"]), ("H1", sp["E2"][:70]), ("H2", sp["E2"][70:140])):
            p = os.path.join(FP, f"K_{role}_{tag}.npy")
            if not os.path.exists(p):
                np.save(p, estimate_K(files)); made[f"K_{role}_{tag}"] = len(files)
    return made

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def real_contrast(K):
    """Own-scale R_real: paired own-minus-other contrast on the 40 held-out photographs of each body."""
    out = {}
    for role, other in (("A", "B"), ("B", "A")):
        z = np.load(os.path.join(FP, f"H_{role}.npz")); Y, W = z["Y"], z["W"]
        out[role] = float(np.mean([ncc(W[i], Y[i] * K[role]) - ncc(W[i], Y[i] * K[other]) for i in range(len(Y))]))
    return 0.5 * (out["A"] + out["B"]), out

_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    _G.update(wr=wavelet_residual, MEAS=MEAS,
              K={t: np.load(os.path.join(FP, t + ".npy")) for t in TEMPL})

def _measure(task):
    tag, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != _G["MEAS"]: return (tag, None)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    W = _G["wr"](Y)
    return (tag, [ncc(W, Y * _G["K"][t]) for t in TEMPL])

def adapter_dir(body, s):
    return os.path.join(GENS if s <= 2 else GENS_EXT, f"{body}_raw_s{s}_r16")

def sym_stats(A, B, R):
    from scipy import stats
    A, B = np.asarray(A), np.asarray(B)
    vA, vB = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(B) - 1))
    UL = th + stats.t.ppf(0.99, df) * se
    tc = stats.t.ppf(0.995, len(A) - 1)
    U = max(A.mean() + tc * A.std(ddof=1) / np.sqrt(len(A)), B.mean() + tc * B.std(ddof=1) / np.sqrt(len(B)))
    return {"mean_A": float(A.mean()), "mean_B": float(B.mean()), "theta_sym": float(th),
            "additive_part": float(0.5 * (A.mean() - B.mean())), "welch_se": float(se), "welch_df": float(df),
            "t": float(th / se), "one_sided_p": float(1 - stats.t.cdf(th / se, df)),
            "R_real_own_scale": R, "sym_limit99": float(UL), "sym_limit99_pct": float(100 * UL / R),
            "maxarm_U": float(U), "maxarm_U_pct": float(100 * U / R)}

def main():
    t0 = time.time()
    made = build_templates(); print("[g3] built:", made or "all present", flush=True)
    tasks = []; counts = {}
    for s in SEEDS:
        for body in ("A", "B"):
            d = adapter_dir(body, s); fs = sorted(glob.glob(os.path.join(d, "*.png")))[:N_IMG]
            assert len(fs) >= 250, f"{d}: {len(fs)} images"      # 500 archived for seeds 0-2, 250 beyond
            counts[f"{body}_s{s}"] = len(fs)
            tasks += [(f"{body}_s{s}", f) for f in fs]
    print(f"[g3] {len(tasks)} images, {WORKERS} workers", flush=True)
    rows = {}
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (tag, v) in enumerate(pool.imap_unordered(_measure, tasks, chunksize=8), 1):
            if v is not None: rows.setdefault(tag, []).append(v)
            if i % 1000 == 0: print(f"[g3] {i}/{len(tasks)} {(time.time()-t0)/60:.1f} min", flush=True)
    K = {t: np.load(os.path.join(FP, t + ".npy")) for t in TEMPL}
    res = {"entry": "RESULTS.md Entry 76 (G3), deviation in Entry 79", "images_per_adapter": counts, "estimates": {}}
    for j, e in enumerate(EST):
        R, per_body = real_contrast({"A": K[f"K_A_{e}"], "B": K[f"K_B_{e}"]})
        iA, iB = 2 * j, 2 * j + 1                      # column of K_A_e and K_B_e in TEMPL
        A = [float(np.mean([r[iA] - r[iB] for r in rows[f"A_s{s}"]])) for s in SEEDS]
        B = [float(np.mean([r[iB] - r[iA] for r in rows[f"B_s{s}"]])) for s in SEEDS]
        st = sym_stats(A, B, R); st.update(per_adapter_A=A, per_adapter_B=B, R_real_per_body=per_body)
        res["estimates"][e] = st
        print(f"  {e:2s} R_real {R:.6f}  theta_sym {st['theta_sym']:+.3e}  sym99 {st['sym_limit99_pct']:.4f}%  "
              f"max-arm {st['maxarm_U_pct']:.4f}%  p {st['one_sided_p']:.3f}", flush=True)
    e2, pooled = res["estimates"]["E2"]["sym_limit99_pct"], res["estimates"]["P"]["sym_limit99_pct"]
    res["pooled_vs_E2_relative_change"] = (pooled - e2) / e2
    res["reading"] = ("the primary limit is stable under the estimate with the most data"
                      if abs(res["pooled_vs_E2_relative_change"]) < 0.25 else "spread reported; not stable")
    res["runtime_min"] = (time.time() - t0) / 60
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[g3] {res['reading']} (pooled vs E2 {100*res['pooled_vs_E2_relative_change']:+.1f} %); wrote {OUT}", flush=True)

if __name__ == "__main__":
    main()
