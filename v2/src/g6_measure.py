"""G6 measurement and readings (RESULTS.md Entries 80, 84, 85) - the iPhone 5c pair, scored only against
its own fingerprints. Nothing here touches the D200 estimates or any D200 result file.

Two estimators are carried through in parallel, which is the point of this arm: E2, built from natural
photographs exactly as the primary design builds it, and FLAT, built from the flat-field images VISION
ships, which is what PRNU estimation normally uses. Both score the identical generations, and each is
divided by the real-image contrast of its own estimate, so the pair of limits measures estimator
dependence with data rather than with the statistical argument of Entry 63.

Per image, exactly as t1_measure._measure: Y = luminance of the centred 1024^2 crop, W = wavelet_residual(Y),
rho = NCC(W, Y*K). Per adapter theta = mean(rho(K_own) - rho(K_other)) over its generations. Reports the
intersection-union exact sign-flip test, the max-arm limit at t_{0.995,11} and the symmetric Welch limit.
The Entry 80 gate applies per estimator: held-out AUC below 0.90, or split-half reliability below 0.15,
makes that estimator's result descriptive with no limit claimed. Writes out/g6_p5c.json.
"""
import os, sys, json, glob, time, itertools
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; FPP = os.path.join(V2, "out", "fp_5c"); T1 = os.path.join(V2, "out", "t1")
# Entry 106 (P1): G6_SUFFIX="_div" scores the diverse-prompt generations of the same adapters into their own file
SUFFIX = os.environ.get("G6_SUFFIX", "")
OUT = os.path.join(V2, "out", f"g6_p5c{SUFFIX}.json"); WORKERS = int(os.environ.get("G6_W", "6"))
ARMS = [f"p5c_{b}_s{s}" for s in range(12) for b in ("A", "B")]
TAGS = ("E2", "FLAT")
TEMPL = [f"K_{r}_{t}" for t in TAGS for r in ("A", "B")]

_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    _G.update(wr=wavelet_residual, MEAS=MEAS,
              K=[np.load(os.path.join(FPP, t + ".npy")) for t in TEMPL])

def ncc(a, b):
    a = a - a.mean(); b = b - b.mean(); return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

def _measure(task):
    arm, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != _G["MEAS"]: return (arm, None)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    W = _G["wr"](Y)
    return (arm, [ncc(W, Y * K) for K in _G["K"]])

def exact_signflip(m):
    obs = float(np.mean(m))
    return sum(1 for sg in itertools.product([-1, 1], repeat=len(m))
               if float(np.mean(m * np.array(sg))) >= obs) / 2 ** len(m)

def stats_for(A, B, R):
    """The study's three readings on one estimator's per-adapter values, against its own R_real."""
    tc = stats.t.ppf(0.995, len(A) - 1)
    UA = A.mean() + tc * A.std(ddof=1) / np.sqrt(len(A))
    UB = B.mean() + tc * B.std(ddof=1) / np.sqrt(len(B))
    vA, vB = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(B) - 1))
    p_sym = float(stats.t.sf(th / se, df)); lim = float(th + stats.t.ppf(0.99, df) * se)
    pA, pB = exact_signflip(A), exact_signflip(B)
    return {"per_adapter_A": A.tolist(), "per_adapter_B": B.tolist(),
            "mean_A": float(A.mean()), "mean_B": float(B.mean()),
            "iu_signflip": {"p_A": pA, "p_B": pB, "floor": 1 / 2 ** len(A)},
            "max_arm": {"U_A": float(UA), "U_B": float(UB), "U_device": float(max(UA, UB)),
                        "lambda_U_pct": float(100 * max(UA, UB) / R)},
            "symmetric": {"theta_sym": float(th), "welch_se": float(se), "welch_df": float(df),
                          "t": float(th / se), "one_sided_p": p_sym,
                          "limit99": lim, "lambda_sym_pct": float(100 * lim / R)},
            "R_real": R, "p_sym": p_sym, "pA": pA, "pB": pB}

def main():
    man = json.load(open(os.path.join(FPP, "manifest.json"))); g = man["gates"]
    t0 = time.time()
    tasks = [(a, f) for a in ARMS for f in sorted(glob.glob(os.path.join(T1, "gens", a + SUFFIX, "*.png")))]
    print(f"[g6] {len(tasks)} images over {len(ARMS)} arms, {len(TEMPL)} templates", flush=True)
    rows = {}
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (arm, v) in enumerate(pool.imap_unordered(_measure, tasks, chunksize=8), 1):
            if v is not None: rows.setdefault(arm, []).append(v)
            if i % 2000 == 0: print(f"[g6] {i}/{len(tasks)} {(time.time()-t0)/60:.1f} min", flush=True)

    res = {"entry": ("RESULTS.md Entry 106 (P1: diverse prompts on the G6 adapters)" if SUFFIX
                     else "RESULTS.md Entries 80, 84, 85 (G6)"),
           "prompt_bank": "diverse (five captions)" if SUFFIX else "uniform", "model": man["model"],
           "devices": {r: d["device"] for r, d in man["roles"].items()},
           "splits": man["splits"], "gates": g, "estimators": {}}
    for k, tag in enumerate(TAGS):
        iA, iB = 2 * k, 2 * k + 1                      # columns of TEMPL for this estimator
        theta = {}
        for arm, v in rows.items():
            v = np.asarray(v); own, oth = (iA, iB) if "_A_" in arm else (iB, iA)
            theta[arm] = float(np.mean(v[:, own] - v[:, oth]))
        A = np.array([theta[f"p5c_A_s{s}"] for s in range(12)])
        B = np.array([theta[f"p5c_B_s{s}"] for s in range(12)])
        sfx = "" if tag == "E2" else "_" + tag
        R = g["R_real" + sfx]
        st = stats_for(A, B, R)
        gate_ok = g["AUC_held_out" + sfx] >= 0.90 and (
            tag != "E2" or min(g["splithalf_A"], g["splithalf_B"]) >= 0.15)
        if not gate_ok:
            reading = "descriptive only: the fingerprint gate was not met"
        elif st["p_sym"] < 0.01 and st["mean_A"] > 0 and st["mean_B"] > 0:
            reading = "device-specific transfer on the iPhone 5c pair"
        elif not (st["pA"] <= 0.01 and st["pB"] <= 0.01) and st["p_sym"] >= 0.01:
            reading = "no detectable transfer on an iPhone 5c pair"
        else:
            reading = "neither reading applies: estimate and limits only"
        st.update(gate_met=bool(gate_ok), reading=reading,
                  AUC_held_out=g["AUC_held_out" + sfx], n_adapters_per_arm=len(A))
        res["estimators"][tag] = st
        print(f"[g6] {tag:4s} R_real {R:.6f}  theta_sym {st['symmetric']['theta_sym']:+.3e} "
              f"({100*st['symmetric']['theta_sym']/R:+.4f}%)  max-arm {st['max_arm']['lambda_U_pct']:.4f}%  "
              f"sym99 {st['symmetric']['lambda_sym_pct']:.4f}%  p {st['p_sym']:.4f}  ->  {reading}", flush=True)

    e2, fl = res["estimators"]["E2"], res["estimators"]["FLAT"]
    res["estimator_dependence"] = {
        "sym_limit_pct_E2": e2["symmetric"]["lambda_sym_pct"],
        "sym_limit_pct_FLAT": fl["symmetric"]["lambda_sym_pct"],
        "ratio_FLAT_over_E2": fl["symmetric"]["lambda_sym_pct"] / e2["symmetric"]["lambda_sym_pct"],
        "per_adapter_corr": float(np.corrcoef(
            np.concatenate([e2["per_adapter_A"], e2["per_adapter_B"]]),
            np.concatenate([fl["per_adapter_A"], fl["per_adapter_B"]]))[0, 1]),
        "agree_on_reading": e2["reading"] == fl["reading"]}
    res["runtime_min"] = (time.time() - t0) / 60
    json.dump(res, open(OUT, "w"), indent=1)
    d = res["estimator_dependence"]
    print(f"[g6] estimator dependence: FLAT/E2 limit ratio {d['ratio_FLAT_over_E2']:.2f}, "
          f"per-adapter r {d['per_adapter_corr']:.2f}, readings agree: {d['agree_on_reading']}", flush=True)

if __name__ == "__main__":
    main()
