# %% E-POST-LOW — low-amplitude calibration with the CORRECT quantisation path
#
# The earlier E-POST injected into an already-quantised PNG and measured in float without
# requantising. That demonstrates detector sensitivity in a continuous-valued image, not after
# the generation-and-storage pipeline. This cell runs the honest path:
#
#     float image -> inject -> clip -> ROUND TO 8-BIT -> measure          (primary)
#     float image -> inject -> clip -> measure (no rounding)              (secondary, for the delta)
#
# Rounding to uint8 IS the PNG write; PNG is lossless, so no disk round-trip is needed.
#
# Injection uses K_B from E1, measurement uses K_B from E2 (disjoint), as everywhere else.
# Fits C(alpha) = b0 + b1*alpha WITH intercept, reports the lower confidence limit on b1, and
# finds the smallest alpha whose CI excludes zero — i.e. the DIRECTLY demonstrated sensitivity.

import pandas as pd
from scipy import stats as sps

ALPHAS_LOW  = (0.0, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25)
N_PRIMARY   = 2500     # sets the smallest directly-detectable alpha; ~55 min at 0.19 s/img
N_SECONDARY = 1000     # no-rounding arm, only to quantify what quantisation costs
CSVP = os.path.join(EXT_ROOT, "csv", "epost_low.csv")

KB1 = K_base("B", "E1")                                             # inject with this
KB2 = torch.from_numpy(np.load(os.path.join(EXT_ROOT, "fingerprints", "K_B.npy"))).to(_MEASURE_DEV)
KA2 = torch.from_numpy(np.load(os.path.join(EXT_ROOT, "fingerprints", "K_A.npy"))).to(_MEASURE_DEV)
KSTACK = torch.stack([KB2, KA2])
print(f"[LOW] NCC(K_B_E1, K_B_E2) = {ncc0(KB1, KB2.cpu().numpy()):.4f}  (disjoint estimates)")

# image pool: spread across all E-AMP tags so no single adapter's statistics dominate
pool = []
for t in all_tags():
    pool += sorted(glob.glob(os.path.join(gen_dir(t), "*.png")))
rng_pool = np.random.default_rng(0)
pool = list(rng_pool.permutation(pool))
print(f"[LOW] image pool: {len(pool)} generations across {len(all_tags())} tags")

hdr = ["alpha", "quant", "idx", "contrast"]
done = done_keys(CSVP, ["alpha", "quant", "idx"])

def _contrast(rgb, a, quantise):
    img = np.clip(rgb * (1.0 + a * KB1[..., None]), 0, 255)
    if quantise:
        img = np.round(img).astype(np.uint8).astype(np.float32)   # <-- the 8-bit write
    else:
        img = img.astype(np.float32)
    Y = (0.299*img[..., 0] + 0.587*img[..., 1] + 0.114*img[..., 2]).astype(np.float32)
    W = torch.from_numpy(wavelet_residual(Y)).to(_MEASURE_DEV)
    Z = torch.from_numpy(Y).to(_MEASURE_DEV)
    v = _ncc_batch(W, Z.unsqueeze(0) * KSTACK)
    return float(v[0] - v[1])

ex = ThreadPoolExecutor(max_workers=4)
try:
    for quant, N in ((1, N_PRIMARY), (0, N_SECONDARY)):
        for a in ALPHAS_LOW:
            todo = [i for i in range(N) if (str(a), str(quant), str(i)) not in done]
            if not todo:
                print(f"[LOW] alpha={a} quant={quant}: complete"); continue
            fut = {i: ex.submit(load_rgb_crop, pool[i]) for i in todo[:6]}
            t0 = time.time()
            for pos, i in enumerate(todo):
                rgb = fut.pop(i).result()
                nx = pos + 6
                if nx < len(todo): fut[todo[nx]] = ex.submit(load_rgb_crop, pool[todo[nx]])
                append_row(CSVP, hdr, [a, quant, i, f"{_contrast(rgb, a, bool(quant)):.6e}"])
            print(f"[LOW] alpha={a:6.3f} quant={quant} n={len(todo)} ({time.time()-t0:.0f}s)")
finally:
    ex.shutdown(wait=True)
    gc.collect()
    if DEV == "cuda": torch.cuda.empty_cache()

# ---------------- analysis ----------------
d = pd.read_csv(CSVP)
U_DEVICE = 8.880e-05          # simultaneous adapter-clustered upper limit from SEED_EXT
out = {"U_device": U_DEVICE, "alphas": list(ALPHAS_LOW), "arms": {}}

for quant, label in ((1, "quantised (full pipeline)"), (0, "not quantised")):
    g = d[d["quant"] == quant]
    if not len(g): continue
    print(f"\n── {label} ──")
    print(f"{'alpha':>7s} {'n':>6s} {'mean':>12s} {'SE':>11s} {'t':>7s} {'CI excl 0':>10s}")
    rows = []
    for a in ALPHAS_LOW:
        x = g[g["alpha"] == a]["contrast"].to_numpy(float)
        if not len(x): continue
        m, se = float(x.mean()), float(x.std(ddof=1)/np.sqrt(len(x)))
        lo = m - sps.norm.ppf(0.995)*se                       # two-sided 99%
        rows.append((a, len(x), m, se, m/se if se else np.nan, lo > 0))
        print(f"{a:7.3f} {len(x):6d} {m:+12.4e} {se:11.3e} {m/se if se else 0:7.2f} "
              f"{'YES' if lo > 0 else '-':>10s}")
    A = np.array([r[0] for r in rows]); M = np.array([r[2] for r in rows])
    Wt = 1.0/np.array([r[3] for r in rows])**2                # inverse-variance weights
    # weighted least squares WITH intercept
    X = np.vstack([np.ones_like(A), A]).T
    Wm = np.diag(Wt)
    beta = np.linalg.solve(X.T@Wm@X, X.T@Wm@M)
    cov = np.linalg.inv(X.T@Wm@X)
    b0, b1 = float(beta[0]), float(beta[1])
    se0, se1 = float(np.sqrt(cov[0, 0])), float(np.sqrt(cov[1, 1]))
    L_b1 = b1 - sps.norm.ppf(0.995)*se1                       # conservative lower CL on the slope
    a_eq = (U_DEVICE - b0)/L_b1
    detected = [r[0] for r in rows if r[5] and r[0] > 0]
    out["arms"][label] = {"b0": b0, "se_b0": se0, "b1": b1, "se_b1": se1, "L_b1": L_b1,
                          "alpha_equiv_conservative": a_eq,
                          "smallest_directly_detected": min(detected) if detected else None,
                          "per_level": [{"alpha": r[0], "n": r[1], "mean": r[2], "se": r[3]}
                                        for r in rows]}
    print(f"  fit  C = {b0:+.3e} + {b1:.4e}*alpha   (SE b0 {se0:.2e}, SE b1 {se1:.2e})")
    print(f"  lower 99% CL on slope         = {L_b1:.4e}")
    print(f"  alpha_equiv of U (conservative) = {a_eq:.4e} = {100*a_eq:.4f}% of natural")
    print(f"  margin below natural            = {1/a_eq:.0f}x")
    print(f"  smallest alpha DIRECTLY detected = "
          f"{min(detected) if detected else 'none'}"
          f"{'' if detected else ' (all low-dose levels are noise-limited)'}")

q = out["arms"].get("quantised (full pipeline)")
nq = out["arms"].get("not quantised")
if q and nq:
    print(f"\n  quantisation cost: slope {nq['b1']:.4e} -> {q['b1']:.4e} "
          f"({100*(1 - q['b1']/nq['b1']):+.1f}%)")

print("\n" + "="*70)
if q and q["smallest_directly_detected"] is not None and q["smallest_directly_detected"] <= 0.01:
    print("DIRECT: the measurement path detects the template at "
          f"{100*q['smallest_directly_detected']:.2f}% of natural amplitude, AFTER 8-bit")
    print("quantisation. The sensitivity claim is demonstrated, not extrapolated.")
else:
    print("EXTRAPOLATED: low-dose levels are noise-limited. Report as:")
    print("  'Extrapolation from the empirically linear calibration places the bound at an")
    print("   equivalent amplitude of ~X% of natural, with direct post-quantisation validation")
    print(f"   limited below {100*min([r for r in ALPHAS_LOW if r>0]):.2f}%.'")
print("="*70)
atomic_json_dump(out, os.path.join(EXT_ROOT, "epost_low_results.json"), indent=2)
