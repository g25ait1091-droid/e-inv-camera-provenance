"""C5 - Shifted-template mechanism (RESULTS.md Entry 58, registered before computation).

Statistic (as t1_measure._measure): Y = 0.299R+0.587G+0.114B, W = wavelet_residual(Y),
rho(d) = NCC(W, Y * roll(K, d)) with roll(K, (dy, dx)) = np.roll(np.roll(K, dy, 0), dx, 1),
K = body A's E2 fingerprint (out/fp/K_A_E2.npy).

Computed exactly for every lag at once (not an approximation): with a = W - mean(W), b_d = Y*K_d,
  sum a*b_d          = corr(W*Y, K)[d] - mean(W) * corr(Y, K)[d]
  sum (b_d-mean)^2   = corr(Y^2, K^2)[d] - corr(Y, K)[d]^2 / N
where corr(A, B)[d] = sum_p A(p) B(p-d) (circular), via real FFTs in float64. Checked against the
direct t1_measure-style NCC on a handful of displacements (reported in the JSON).

Design: every residue class (dx mod 16, dy mod 16) gets 3 displacements with the quotient drawn at
random (seed 20260915); lag (0,0) and lags within 32 px (circular, both axes) of it are excluded so
the unshifted fingerprint's own peak cannot enter; the v1 displacement (rows 137, cols 251) is added.
First 150 PNGs of each arm. Per arm: mean rho over images per displacement; excess of the
displacements with dx%8==0 and dy%8==0 over the others, SE across displacements (Welch form).
CPU only, 2 workers, 1 thread each. Output: out/c5_shift_grid.json
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"
import sys, json, glob, time, platform
import numpy as np
from PIL import Image
from multiprocessing import Pool

sys.path.insert(0, EINV.SRC)
V2 = EINV.V2; GENS = os.path.join(V2, "out", "t1", "gens")
KPATH = os.path.join(V2, "out", "fp", "K_A_E2.npy"); OUT = os.path.join(V2, "out", "c5_shift_grid.json")
BASE_ARM = "local_base"
ADAPTED = ["local_A_raw_s0", "local_B_raw_s0", "nomark_s0", "nomark_s1", "nomark_s2",
           "nomarkB_s0", "nomarkB_s1", "nomarkB_s2"]
ARMS = [BASE_ARM] + ADAPTED
N_IMG = 150; PER_CLASS = 3; SEED = 20260915; MIN_LAG = 32; WORKERS = 2; CHUNK = 15
V1_SHIFT = (137, 251)   # (rows = dy, cols = dx); notebooks/03_amp.ipynb: SHIFT = (137, 251), np.roll axis 0 then 1
V1_SOURCE = os.path.join(EINV.REPO, 'notebooks', '03_amp.ipynb (Config.SHIFT; shift_field = np.roll(np.roll(K, dr, 0), dc, 1))')
N = 1024

def circ(v): return min(v % N, (-v) % N)

def displacements():
    rng = np.random.default_rng(SEED); D = []
    for ry in range(16):
        for rx in range(16):
            got = set()
            while len(got) < PER_CLASS:
                dy = ry + 16*int(rng.integers(0, N//16)); dx = rx + 16*int(rng.integers(0, N//16))
                if max(circ(dy), circ(dx)) < MIN_LAG: continue
                got.add((dy, dx))
            D += sorted(got)
    if V1_SHIFT not in D: D.append(V1_SHIFT)
    return np.array(D, dtype=np.int64)   # columns: dy, dx

_G = {}
def _init():
    import torch; torch.set_num_threads(1)
    from fingerprints import wavelet_residual, MEAS
    assert MEAS == N
    K = np.load(KPATH).astype(np.float64)
    _G.update(wr=wavelet_residual, K32=np.load(KPATH), FK=np.fft.rfft2(K), FK2=np.fft.rfft2(K*K), D=displacements())

def _corr(A, FB):
    return np.fft.irfft2(np.fft.rfft2(A) * np.conj(FB), s=(N, N))

def _ncc(a, b):   # verbatim from t1_measure
    a = a - a.mean(); b = b - b.mean(); return float((a*b).sum() / (np.linalg.norm(a)*np.linalg.norm(b) + 1e-12))

def _surface(path):
    g = _G
    a = np.asarray(Image.open(path).convert("RGB"), np.float32)
    Y32 = (0.299*a[..., 0] + 0.587*a[..., 1] + 0.114*a[..., 2]).astype(np.float32); W32 = g["wr"](Y32)
    Y = Y32.astype(np.float64); W = W32.astype(np.float64)
    SYK = _corr(Y, g["FK"]); num = _corr(W*Y, g["FK"]) - W.mean()*SYK
    varb = np.maximum(_corr(Y*Y, g["FK2"]) - SYK*SYK/(N*N), 0.0)
    nW = np.sqrt(((W - W.mean())**2).sum())
    return num / (nW*np.sqrt(varb) + 1e-12), Y32, W32

def _task(t):
    arm, paths, verify = t; D = _G["D"]; ssum = np.zeros((N, N)); rows = []; chk = []
    for i, p in enumerate(paths):
        s, Y32, W32 = _surface(p); ssum += s; rows.append(s[D[:, 0], D[:, 1]])
        if verify and i == 0:
            for dy, dx in [(0, 0), V1_SHIFT, tuple(D[0]), tuple(D[len(D)//2]), (8, 16), (5, 3)]:
                direct = _ncc(W32, Y32*np.roll(np.roll(_G["K32"], dy, 0), dx, 1))
                chk.append({"image": os.path.basename(p), "dy": int(dy), "dx": int(dx), "direct": direct, "fft": float(s[dy, dx])})
    return arm, [os.path.basename(p) for p in paths], np.array(rows), ssum, chk

def excess(vals, on):
    a, b = vals[on], vals[~on]; e = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1)/len(a) + b.var(ddof=1)/len(b))
    return {"mean_on8": float(a.mean()), "mean_off": float(b.mean()), "excess": float(e), "se": float(se),
            "z": float(e/se), "n_on8": int(len(a)), "n_off": int(len(b))}

def full_lag_summary(surf):
    yy, xx = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    cy = np.minimum(yy, N-yy); cx = np.minimum(xx, N-xx); far = np.maximum(cy, cx) >= MIN_LAG
    on = (yy % 8 == 0) & (xx % 8 == 0)
    a, b = surf[on & far], surf[~on & far]
    return {"mean_on8": float(a.mean()), "mean_off": float(b.mean()), "excess": float(a.mean()-b.mean()),
            "sd_on8": float(a.std()), "sd_off": float(b.std()), "n_on8": int(a.size), "n_off": int(b.size),
            "z_naive_independent_lags": float((a.mean()-b.mean())/np.sqrt(a.var()/a.size + b.var()/b.size))}

def main():
    t0 = time.time(); D = displacements(); on8 = (D[:, 0] % 8 == 0) & (D[:, 1] % 8 == 0)
    on16 = (D[:, 0] % 16 == 0) & (D[:, 1] % 16 == 0)
    iv1 = int(np.where((D[:, 0] == V1_SHIFT[0]) & (D[:, 1] == V1_SHIFT[1]))[0][0])
    print(f"[c5] {len(D)} displacements, {on8.sum()} on the 8-px grid; v1 shift {V1_SHIFT} "
          f"mod8 {(V1_SHIFT[0] % 8, V1_SHIFT[1] % 8)} mod16 {(V1_SHIFT[0] % 16, V1_SHIFT[1] % 16)}", flush=True)
    tasks = []
    for arm in ARMS:
        files = sorted(glob.glob(os.path.join(GENS, arm, "*.png")))[:N_IMG]
        assert len(files) == N_IMG, (arm, len(files))
        for j in range(0, N_IMG, CHUNK): tasks.append((arm, files[j:j+CHUNK], arm == BASE_ARM and j == 0))
    per = {a: [] for a in ARMS}; names = {a: [] for a in ARMS}; surf = {a: np.zeros((N, N)) for a in ARMS}; checks = []
    with Pool(WORKERS, initializer=_init) as pool:
        for k, (arm, nm, rows, ssum, chk) in enumerate(pool.imap_unordered(_task, tasks), 1):
            per[arm].append(rows); names[arm] += nm; surf[arm] += ssum; checks += chk
            print(f"[c5] {k}/{len(tasks)} chunks  {(time.time()-t0)/60:.1f} min", flush=True)
    for c in checks: c["absdiff"] = abs(c["direct"] - c["fft"])
    print("[c5] verification max |direct - fft| =", max(c["absdiff"] for c in checks), flush=True)

    # K autocorrelation diagnostic: NCC(K, roll(K, d)) (zero-mean K)
    K = np.load(KPATH).astype(np.float64); K = K - K.mean(); FK = np.fft.rfft2(K)
    ac = _corr(K, FK) / (K*K).sum()
    res = {"entry": "RESULTS.md Entry 58 C5", "fingerprint": KPATH, "arms": ARMS, "n_images_per_arm": N_IMG,
           "v1_shift": {"dy_rows": V1_SHIFT[0], "dx_cols": V1_SHIFT[1], "source": V1_SOURCE,
                        "mod8": [V1_SHIFT[0] % 8, V1_SHIFT[1] % 8], "mod16": [V1_SHIFT[0] % 16, V1_SHIFT[1] % 16],
                        "is_multiple_of_8_both_axes": bool(V1_SHIFT[0] % 8 == 0 and V1_SHIFT[1] % 8 == 0)},
           "design": {"per_class": PER_CLASS, "seed": SEED, "min_circular_lag_px": MIN_LAG,
                      "n_displacements": int(len(D)), "n_on8": int(on8.sum()), "n_on16": int(on16.sum()),
                      "displacements_dy_dx": D.tolist()},
           "verification_direct_vs_fft": checks, "arm_results": {}}
    M = {}
    for arm in ARMS:
        R = np.concatenate(per[arm]); assert R.shape == (N_IMG, len(D)); m = R.mean(0); M[arm] = m
        ex = excess(m, on8)
        img_ex = R[:, on8].mean(1) - R[:, ~on8].mean(1)   # supplementary: SE across images
        cls16 = np.zeros((16, 16)); cls8 = np.zeros((8, 8))
        for ry in range(16):
            for rx in range(16): cls16[ry, rx] = m[(D[:, 0] % 16 == ry) & (D[:, 1] % 16 == rx)].mean()
        for ry in range(8):
            for rx in range(8): cls8[ry, rx] = m[(D[:, 0] % 8 == ry) & (D[:, 1] % 8 == rx)].mean()
        order16 = np.argsort(-cls16.ravel())
        rank16 = {f"{ry},{rx}": int(np.where(order16 == ry*16+rx)[0][0]) + 1 for ry, rx in [(0, 0), (0, 8), (8, 0), (8, 8), (9, 11)]}
        fs = surf[arm] / N_IMG
        fs_rank = int((fs > fs[V1_SHIFT]).sum() - (fs[0, 0] > fs[V1_SHIFT]))  # lags above v1, excluding (0,0)
        res["arm_results"][arm] = {
            "rho_unshifted_K_A": float(fs[0, 0]),
            "excess_on8_vs_off": ex,
            "on16_vs_on8_not16_vs_off": {"mean_on16": float(m[on16].mean()), "mean_on8_not16": float(m[on8 & ~on16].mean()),
                                         "mean_off": float(m[~on8].mean())},
            "excess_se_across_images_supplementary": {"excess": float(img_ex.mean()), "se": float(img_ex.std(ddof=1)/np.sqrt(N_IMG)),
                                                     "z": float(img_ex.mean()/(img_ex.std(ddof=1)/np.sqrt(N_IMG)))},
            "sd_across_displacements": float(m.std(ddof=1)),
            "v1_shift": {"rho": float(m[iv1]), "rank_among_sampled": int((m > m[iv1]).sum()) + 1, "n_sampled": int(len(D)),
                         "z_vs_sampled": float((m[iv1] - m.mean())/m.std(ddof=1)),
                         "rank_among_all_lags_excl_0_0": fs_rank + 1, "n_all_lags": N*N - 1},
            "mod8_class_means_rows_dy_cols_dx": cls8.tolist(),
            "mod16_class_means_rows_dy_cols_dx": cls16.tolist(),
            "mod16_class_ranks_of_256": rank16,
            "mod16_top5_classes": [[int(i // 16), int(i % 16), float(cls16.ravel()[i])] for i in order16[:5]],
            "full_lag_supplementary": full_lag_summary(fs)}
    # adapted minus base, per displacement
    res["adapted_minus_base"] = {a: excess(M[a] - M[BASE_ARM], on8) for a in ADAPTED}
    pooled = np.mean([M[a] for a in ADAPTED], 0)
    res["pooled_adapted"] = excess(pooled, on8); res["pooled_adapted_minus_base"] = excess(pooled - M[BASE_ARM], on8)
    res["K_autocorrelation_diagnostic"] = {"sampled": excess(ac[D[:, 0], D[:, 1]], on8), "full_lag": full_lag_summary(ac),
                                           "ncc_at_v1_shift": float(ac[V1_SHIFT])}
    # readings (Entry 58 C5, verbatim conditions)
    c1 = all(res["arm_results"][a]["excess_on8_vs_off"]["z"] > 3 for a in ADAPTED)
    c1_pooled = res["pooled_adapted"]["z"] > 3
    c2 = not (res["arm_results"][BASE_ARM]["excess_on8_vs_off"]["z"] > 3)
    c3 = res["v1_shift"]["is_multiple_of_8_both_axes"]
    res["reading"] = {
        "text_registered": ("'the 8-px latent grid explains the shifted-template artifact' if, in adapted arms, shifts that "
                            "are multiples of 8 px on both axes exceed the other shifts by more than 3 SE (SE across "
                            "displacements) while the base model shows no such excess, and the shift used in v1 is such a "
                            "multiple; 'not explained by the grid' otherwise."),
        "cond_adapted_all_arms_excess_gt_3SE": bool(c1), "cond_adapted_pooled_excess_gt_3SE_supplementary": bool(c1_pooled),
        "cond_base_no_excess_(z<=3)": bool(c2), "cond_v1_shift_multiple_of_8": bool(c3),
        "applies": "the 8-px latent grid explains the shifted-template artifact" if (c1 and c2 and c3) else "not explained by the grid"}
    res["runtime_min"] = (time.time() - t0)/60; res["platform"] = platform.platform(); res["workers"] = WORKERS
    json.dump(res, open(OUT, "w"), indent=1)
    for arm in ARMS:
        e = res["arm_results"][arm]["excess_on8_vs_off"]; v = res["arm_results"][arm]["v1_shift"]
        print(f"[c5] {arm:16s} on8 {e['mean_on8']:+.3e} off {e['mean_off']:+.3e} excess {e['excess']:+.3e} "
              f"SE {e['se']:.2e} z {e['z']:+.2f} | v1 rank {v['rank_among_sampled']}/{v['n_sampled']}", flush=True)
    for a in ADAPTED:
        e = res["adapted_minus_base"][a]; print(f"[c5] {a:16s} minus base: excess {e['excess']:+.3e} SE {e['se']:.2e} z {e['z']:+.2f}")
    print("[c5] K autocorr diag:", res["K_autocorrelation_diagnostic"]["sampled"])
    print("[c5] reading:", res["reading"]["applies"], f"  runtime {res['runtime_min']:.1f} min", flush=True)

if __name__ == "__main__":
    main()
