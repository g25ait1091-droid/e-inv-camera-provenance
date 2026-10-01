"""G4b measurement and readings (RESULTS.md Entries 85, 86) - the Huawei P20 pair (1104, 1103), scored only against its own
fingerprints. Nothing here touches the D200 estimates or any D200 result file.

Per image, exactly as t1_measure._measure: Y = luminance of the centred 1024^2 crop, W = wavelet_residual(Y),
rho = NCC(W, Y*K). Per adapter theta = mean(rho(K_own) - rho(K_other)) over its 500 generations.
Reports the intersection-union exact sign-flip test, the max-arm limit at t_{0.995,11}, and the symmetric
Welch limit, each as a fraction of this pair's own R_real. Applies the Entry 85 readings, including the
gate: if the held-out AUC is below 0.90 or split-half reliability below 0.15, the result is descriptive.
Writes out/g4b_p20.json.
"""
import os, sys, json, glob, time, itertools
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; FPP = os.path.join(V2, "out", "fp_p20b"); T1 = os.path.join(V2, "out", "t1")
OUT = os.path.join(V2, "out", "g4b_p20.json"); WORKERS = int(os.environ.get("G4B_W", "6"))
ARMS = [f"p20b_{b}_s{s}" for s in range(12) for b in ("A", "B")]

_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    _G.update(wr=wavelet_residual, MEAS=MEAS,
              K={r: np.load(os.path.join(FPP, f"K_{r}_E2.npy")) for r in ("A", "B")})

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _measure(task):
    arm, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != _G["MEAS"]: return (arm, None)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    W = _G["wr"](Y)
    return (arm, (ncc(W, Y * _G["K"]["A"]), ncc(W, Y * _G["K"]["B"])))

def exact_signflip(m):
    obs = float(np.mean(m))
    return sum(1 for sg in itertools.product([-1, 1], repeat=len(m))
               if float(np.mean(m * np.array(sg))) >= obs) / 2 ** len(m)

def main():
    man = json.load(open(os.path.join(FPP, "manifest.json"))); g = man["gates"]
    R = g["R_real"]; t0 = time.time()
    # Entry 85 / defect D9: the gate is decided on real photographs alone, so it is read before any
    # generation is measured. When it is not met the arm is reported descriptively, as registered in
    # Entry 78, and no generation is scored - there is no limit for them to inform.
    gate_ok = g["AUC_held_out"] >= 0.90 and min(g["splithalf_A"], g["splithalf_B"]) >= 0.15
    if not gate_ok:
        res = {"entry": "RESULTS.md Entry 85 (G4b)",
               "model": man.get("model", "Huawei P20 (EML-AL00)"),
               "devices": {r: d["device"] for r, d in man["roles"].items()},
               "gates": g, "gate_met": False, "generations_scored": 0,
               "reading": "descriptive only: the fingerprint gate was not met",
               "why": ("held-out AUC %.4f is below the registered 0.90; body %s contributes %+.4f of "
                       "own-minus-other contrast against body %s's %+.4f, so the pair cannot support a "
                       "transfer limit" % (g["AUC_held_out"], man["roles"]["A"]["device"],
                                           g["per_body_contrast"]["A"], man["roles"]["B"]["device"],
                                           g["per_body_contrast"]["B"])),
               "R_real": R}
        json.dump(res, open(OUT, "w"), indent=1)
        print("[g4b] gate not met (AUC %.4f): %s" % (g["AUC_held_out"], res["reading"]), flush=True)
        return
    tasks = [(a, f) for a in ARMS for f in sorted(glob.glob(os.path.join(T1, "gens", a, "*.png")))]
    print(f"[g4b] {len(tasks)} images over {len(ARMS)} arms", flush=True)
    rows = {}
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (arm, v) in enumerate(pool.imap_unordered(_measure, tasks, chunksize=8), 1):
            if v is not None: rows.setdefault(arm, []).append(v)
            if i % 2000 == 0: print(f"[g4b] {i}/{len(tasks)} {(time.time()-t0)/60:.1f} min", flush=True)
    theta = {}
    for arm, v in rows.items():
        v = np.asarray(v); own = 0 if "_A_" in arm else 1
        theta[arm] = float(np.mean(v[:, own] - v[:, 1 - own]))
    A = np.array([theta[f"p20b_A_s{s}"] for s in range(12)])
    B = np.array([theta[f"p20b_B_s{s}"] for s in range(12)])
    tc = stats.t.ppf(0.995, len(A) - 1)
    UA = A.mean() + tc * A.std(ddof=1) / np.sqrt(len(A)); UB = B.mean() + tc * B.std(ddof=1) / np.sqrt(len(B))
    vA, vB = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(B) - 1))
    p_sym = float(stats.t.sf(th / se, df)); lim = float(th + stats.t.ppf(0.99, df) * se)
    pA, pB = exact_signflip(A), exact_signflip(B)
    gate_ok = g["AUC_held_out"] >= 0.90 and min(g["splithalf_A"], g["splithalf_B"]) >= 0.15
    if not gate_ok:
        reading = "descriptive only: the fingerprint gate was not met"
    elif p_sym < 0.01 and A.mean() > 0 and B.mean() > 0:
        reading = "device-specific transfer on a modern smartphone pair"
    elif not (pA <= 0.01 and pB <= 0.01) and p_sym >= 0.01:
        reading = "no detectable transfer on a modern smartphone pair"
    else:
        reading = "neither reading applies: estimate and limits only"
    res = {"entry": "RESULTS.md Entries 85-87 (G4b), readings as registered for G4 in Entry 78",
           "model": man["model"], "devices": {r: d["device"] for r, d in man["roles"].items()},
           "additive_part": float(0.5 * (A.mean() - B.mean())),
           "n_positive": {"A": int((A > 0).sum()), "B": int((B > 0).sum())},
           "gates": g, "gate_met": bool(gate_ok), "per_adapter_A": A.tolist(), "per_adapter_B": B.tolist(),
           "mean_A": float(A.mean()), "mean_B": float(B.mean()),
           "iu_signflip": {"p_A": pA, "p_B": pB, "floor": 1 / 2 ** 12},
           "max_arm": {"U_A": float(UA), "U_B": float(UB), "U_device": float(max(UA, UB)),
                       "lambda_U_pct": float(100 * max(UA, UB) / R)},
           "symmetric": {"theta_sym": float(th), "welch_se": float(se), "welch_df": float(df),
                         "t": float(th / se), "one_sided_p": p_sym,
                         "limit99": lim, "lambda_sym_pct": float(100 * lim / R)},
           "R_real": R, "reading": reading, "runtime_min": (time.time() - t0) / 60}
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[g4b] theta_sym {th:+.3e} ({100*th/R:+.4f}%)  max-arm {100*max(UA,UB)/R:.4f}%  "
          f"sym99 {100*lim/R:.4f}%  p {p_sym:.4f}  ->  {reading}", flush=True)

if __name__ == "__main__":
    main()
