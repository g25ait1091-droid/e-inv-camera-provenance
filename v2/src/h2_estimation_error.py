"""H2 (RESULTS.md Entry 88) - propagate fingerprint-estimation error into the limit.

Entry 83 showed the limit moves from 0.062 % to 0.152 % across five fingerprint estimates while the null
holds under all of them. That spread is real uncertainty the published construction does not carry: the
limit is computed as though K were known, when it is itself estimated from a finite set of photographs.

This resamples that set. Twenty-four bootstrap replicates of each body's E2 photographs, each giving its
own K, its own real-image contrast on the held-out split, and its own symmetric limit on the same
generations. Because estimate_K accumulates sum(W*Y) and sum(Y*Y) before its post-processing, every
replicate is an exactly weighted combination of one pass over the photographs - the bootstrap is exact
here, not an approximation of the estimator.

The images are held fixed across replicates (the same subsample of generations every time), so the only
thing moving is the estimate. Reported: the distribution of the limit, the null re-read under every
replicate, and an estimation-inflated limit that adds the between-replicate variance to the adapter-level
variance. CPU only. Writes out/h2_estimation_error.json.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
import sys, json, glob, time
import numpy as np
from PIL import Image
from multiprocessing import Pool
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV

V2 = EINV.V2; FP = os.path.join(V2, "out", "fp")
OUT = os.path.join(V2, "out", "h2_estimation_error.json")
GENS, GENS_EXT = EINV.GENS, EINV.GENS_EXT
N_REP = int(os.environ.get("H2_REP", "24"))
N_IMG = int(os.environ.get("H2_IMG", "200"))        # generations per adapter, fixed across replicates
WORKERS = int(os.environ.get("H2_W", "5"))          # leave cores for the GPU lanes' data loading
SEEDS = range(12)
BOOT_SEED = 880221
DEVICES = (("A", "Nikon_D200_1"), ("B", "Nikon_D200_0"))
TDIR = os.path.join(FP, "h2_boot")


def ncc(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def adapter_dir(body, s):
    return os.path.join(GENS if s <= 2 else GENS_EXT, f"{body}_raw_s{s}_r16")


def build_bootstrap_templates():
    """One pass over each body's E2 photographs; every replicate is a weighted sum of the same terms."""
    from fingerprints import dv, splits, load_lum_crop, wavelet_residual, zero_mean, wiener_dft
    os.makedirs(TDIR, exist_ok=True)
    meta = {}
    for role, did in DEVICES:
        files = splits(dv[did])["E2"]
        n = len(files)
        rng = np.random.default_rng(BOOT_SEED)        # same replicate seeds for both bodies
        counts = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(N_REP)])
        meta[role] = {"device": did, "n_photographs": n,
                      "mean_distinct_per_replicate": float(np.mean((counts > 0).sum(1)))}
        if all(os.path.exists(os.path.join(TDIR, f"K_{role}_b{r}.npy")) for r in range(N_REP)):
            print(f"[h2] {role}: bootstrap templates present", flush=True); continue
        num = den = None
        t0 = time.time()
        for i, fp in enumerate(files):
            Y = load_lum_crop(fp); W = wavelet_residual(Y)
            wy, yy = (W * Y).astype(np.float64), (Y * Y).astype(np.float64)
            if num is None:
                num = np.zeros((N_REP,) + wy.shape); den = np.zeros((N_REP,) + yy.shape)
            c = counts[:, i]
            nz = np.nonzero(c)[0]
            for r in nz:
                num[r] += c[r] * wy; den[r] += c[r] * yy
            if (i + 1) % 40 == 0:
                print(f"[h2] {role}: {i+1}/{n} photographs ({(time.time()-t0)/60:.1f} min)", flush=True)
        for r in range(N_REP):
            K = wiener_dft(zero_mean((num[r] / np.maximum(den[r], 1e-6)).astype(np.float32)))
            np.save(os.path.join(TDIR, f"K_{role}_b{r}.npy"), K)
        del num, den
        print(f"[h2] {role}: {N_REP} bootstrap templates written", flush=True)
    return meta


def real_contrast_all():
    """R_real per replicate: the paired own-minus-other contrast on each body's held-out photographs."""
    from fingerprints import dv, splits, load_lum_crop, wavelet_residual
    H = {}
    for role, did in DEVICES:
        files = splits(dv[did])["H"]
        Y = [load_lum_crop(f) for f in files]
        H[role] = (Y, [wavelet_residual(y) for y in Y])
    out = []
    for r in range(N_REP):
        K = {role: np.load(os.path.join(TDIR, f"K_{role}_b{r}.npy")) for role, _ in DEVICES}
        per = {}
        for role, other in (("A", "B"), ("B", "A")):
            Y, W = H[role]
            per[role] = float(np.mean([ncc(W[i], Y[i] * K[role]) - ncc(W[i], Y[i] * K[other])
                                       for i in range(len(Y))]))
        out.append({"R_real": 0.5 * (per["A"] + per["B"]), "per_body": per})
    return out


_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    _G.update(wr=wavelet_residual, MEAS=MEAS,
              K=[np.load(os.path.join(TDIR, f"K_{role}_b{r}.npy"))
                 for r in range(N_REP) for role in ("A", "B")])


def _measure(task):
    tag, path = task
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    if a.shape[0] != _G["MEAS"]: return (tag, None)
    Y = (0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]).astype(np.float32)
    W = _G["wr"](Y)
    return (tag, [ncc(W, Y * K) for K in _G["K"]])


def sym_stats(A, B, R):
    A, B = np.asarray(A), np.asarray(B)
    vA, vB = A.var(ddof=1) / len(A), B.var(ddof=1) / len(B)
    th = 0.5 * (A.mean() + B.mean()); se = 0.5 * np.sqrt(vA + vB)
    df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(B) - 1))
    return {"theta_sym": float(th), "welch_se": float(se), "welch_df": float(df),
            "one_sided_p": float(stats.t.sf(th / se, df)),
            "limit99": float(th + stats.t.ppf(0.99, df) * se),
            "limit99_pct": float(100 * (th + stats.t.ppf(0.99, df) * se) / R),
            "theta_sym_pct": float(100 * th / R), "se_pct": float(100 * se / R), "R_real": R}


def main():
    t0 = time.time()
    meta = build_bootstrap_templates()
    print(f"[h2] templates ready ({(time.time()-t0)/60:.1f} min); measuring held-out contrasts", flush=True)
    RC = real_contrast_all()
    print(f"[h2] R_real over replicates: {np.mean([r['R_real'] for r in RC]):.6f} "
          f"+- {np.std([r['R_real'] for r in RC], ddof=1):.6f}", flush=True)

    tasks, counts = [], {}
    for s in SEEDS:
        for body in ("A", "B"):
            fs = sorted(glob.glob(os.path.join(adapter_dir(body, s), "*.png")))[:N_IMG]
            assert len(fs) >= N_IMG, f"{body}_s{s}: {len(fs)} images, need {N_IMG}"
            counts[f"{body}_s{s}"] = len(fs)
            tasks += [(f"{body}_s{s}", f) for f in fs]
    print(f"[h2] {len(tasks)} generations x {2 * N_REP} templates, {WORKERS} workers", flush=True)
    rows = {}
    with Pool(WORKERS, initializer=_init) as pool:
        for i, (tag, v) in enumerate(pool.imap_unordered(_measure, tasks, chunksize=4), 1):
            if v is not None: rows.setdefault(tag, []).append(v)
            if i % 500 == 0: print(f"[h2] {i}/{len(tasks)} ({(time.time()-t0)/60:.1f} min)", flush=True)

    reps = []
    for r in range(N_REP):
        iA, iB = 2 * r, 2 * r + 1
        A = [float(np.mean([x[iA] - x[iB] for x in rows[f"A_s{s}"]])) for s in SEEDS]
        B = [float(np.mean([x[iB] - x[iA] for x in rows[f"B_s{s}"]])) for s in SEEDS]
        st = sym_stats(A, B, RC[r]["R_real"])
        st["replicate"] = r
        reps.append(st)

    lim = np.array([x["limit99_pct"] for x in reps])
    th = np.array([x["theta_sym_pct"] for x in reps])
    se = np.array([x["se_pct"] for x in reps])
    p = np.array([x["one_sided_p"] for x in reps])
    v_adapter = float(np.mean(se ** 2))              # adapter-level variance, averaged over replicates
    v_est = float(np.var(th, ddof=1))                # variance the estimate itself contributes
    df_bar = float(np.mean([x["welch_df"] for x in reps]))
    inflated = float(th.mean() + stats.t.ppf(0.99, df_bar) * np.sqrt(v_adapter + v_est))
    filed = 0.0815                                   # Entry 83, pooled estimate, per cent of R_real

    res = {"entry": "RESULTS.md Entry 88 (H2)", "n_replicates": N_REP, "images_per_adapter": N_IMG,
           "bootstrap_seed": BOOT_SEED, "photographs": meta, "images_used": counts,
           "replicates": reps, "R_real_per_replicate": RC,
           "limit_pct": {"mean": float(lim.mean()), "sd": float(lim.std(ddof=1)),
                         "min": float(lim.min()), "max": float(lim.max()),
                         "q05": float(np.percentile(lim, 5)), "q95": float(np.percentile(lim, 95))},
           "theta_sym_pct": {"mean": float(th.mean()), "sd": float(th.std(ddof=1))},
           "variance_components_pct2": {"adapter_level": v_adapter, "estimation": v_est,
                                        "estimation_share": v_est / (v_adapter + v_est)},
           "inflated_limit_pct": inflated, "filed_limit_pct": filed,
           "inflation_factor_vs_filed": inflated / filed,
           "null": {"max_p": float(p.max()), "min_p": float(p.min()),
                    "n_replicates_rejecting_at_01": int((p < 0.01).sum())},
           "runtime_min": (time.time() - t0) / 60}
    if res["null"]["n_replicates_rejecting_at_01"]:
        res["reading"] = (f"{res['null']['n_replicates_rejecting_at_01']} of {N_REP} replicates reject the "
                          "null at one-sided 0.01; reported")
    elif inflated / filed > 2:
        res["reading"] = ("estimation error more than doubles the limit; the paper quotes the inflated "
                          "limit as its headline and the filed value as the fixed-estimate case")
    else:
        res["reading"] = ("estimation error widens the limit by a stated factor below two; the filed "
                          "limit stands with the inflated value reported beside it")
    json.dump(res, open(OUT, "w"), indent=1)
    print(f"[h2] limit {lim.mean():.4f} % (sd {lim.std(ddof=1):.4f}, range {lim.min():.4f}-{lim.max():.4f})",
          flush=True)
    print(f"[h2] estimation share of variance {100*res['variance_components_pct2']['estimation_share']:.1f} %; "
          f"inflated limit {inflated:.4f} % vs filed {filed:.4f} % ({inflated/filed:.2f}x)", flush=True)
    print(f"[h2] null: p from {p.min():.3f} to {p.max():.3f}", flush=True)
    print(f"[h2] READING: {res['reading']}", flush=True)


if __name__ == "__main__":
    main()
