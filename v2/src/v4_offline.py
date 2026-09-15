"""Entry 50 (review follow-up, derived; no new generations). Writes out/v4_offline.json.

1. Symmetric statistic of the primary design (12 adapters per arm, supplement Table S5): theta_sym,
   Welch SE, one-sided 99 % and 99.5 % limits as a fraction of R_real, and its power curve
   (noncentral t, one-sided alpha 0.01) against true lambda.
2. Test of the channel prediction against theta_sym, with the prediction's own SE propagated.
3. Cross-spectrum weighting: the band energy of the fingerprint estimated as the cross-spectrum of two
   independent estimates (E1 x E2), which removes estimation noise that is independent between splits,
   in place of the power spectrum of one estimate. Predictions recomputed for the autoencoder
   (T_vae by band -> eta) and for the full pipeline (T_full by band -> non-repeating transmission),
   for body A and body B.
4. Kodak five-body exact one-sided sign-flip p over the device means.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import einv_paths as EINV
import os, sys, json, itertools, math
import numpy as np
from scipy import stats
sys.path.insert(0, EINV.SRC)
import t1_band as B

V2 = EINV.V2; T1 = os.path.join(V2, "out", "t1"); FP = os.path.join(V2, "out", "fp")
R_REAL = 0.0356703416571125
A = np.array([8.850e-6, -6.325e-6, 2.274e-6, -2.264e-5, 7.532e-6, 1.033e-4, 1.466e-5, -9.209e-5, -1.081e-5, -1.501e-5, -2.960e-5, -7.494e-5])
Bm = np.array([-3.418e-5, 1.363e-5, 4.924e-5, 4.068e-5, -1.620e-5, 3.312e-5, 6.443e-5, -3.437e-6, 1.021e-5, 2.735e-5, 8.484e-5, -4.446e-5])
out = {}

# 1. symmetric statistic
th = 0.5 * (A.mean() + Bm.mean()); vA, vB = A.var(ddof=1) / len(A), Bm.var(ddof=1) / len(Bm)
se = 0.5 * math.sqrt(vA + vB); df = (vA + vB) ** 2 / (vA ** 2 / (len(A) - 1) + vB ** 2 / (len(Bm) - 1))
sym = {"theta_sym": th, "se": se, "welch_df": df, "lambda_hat_pct": 100 * th / R_REAL}
for q in (0.99, 0.995):
    U = th + stats.t.ppf(q, df) * se; sym[f"U_{q}"] = U; sym[f"lambda_U_{q}_pct"] = 100 * U / R_REAL
crit = stats.t.ppf(0.99, df)
sym["power_curve"] = [{"lambda_pct": l, "power": float(stats.nct.sf(crit, df, l / 100 * R_REAL / se))} for l in (0.02, 0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15)]
out["symmetric"] = sym

# 3. cross-spectrum band energies
M = B.masks()
def spec(x): return np.fft.rfft2(x - x.mean())
def frac_power(x):
    X = np.abs(spec(x)) ** 2; return [float((X * m).sum() / X.sum()) for m in M], float(1 - sum((X * m).sum() for m in M) / X.sum())
def frac_cross(x1, x2):
    C = np.real(spec(x1) * np.conj(spec(x2))); tot = C.sum()
    return [float((C * m).sum() / tot) for m in M], float(1 - sum((C * m).sum() for m in M) / tot)
bands = {}
for body in ("A", "B"):
    k1, k2 = np.load(os.path.join(FP, f"K_{body}_E1.npy")), np.load(os.path.join(FP, f"K_{body}_E2.npy"))
    pw, pc = frac_power(k2); cr, cc = frac_cross(k1, k2)
    bands[body] = {"power_E2": pw, "power_E2_corner": pc, "cross_E1E2": cr, "cross_E1E2_corner": cc,
                   "split_corr": float(np.corrcoef((k1 - k1.mean()).ravel(), (k2 - k2.mean()).ravel())[0, 1])}
out["band_energy"] = bands

vae = json.load(open(os.path.join(T1, "band_vae.json")))["T_vae"]; Tv = [vae[f"band{b}"] for b in range(6)]
bs = json.load(open(os.path.join(T1, "band_summary.json")))["curve"]; Tf = [c["T_full"] for c in bs]; Tfse = [c["T_full_se"] for c in bs]
def pred(e, corner, T, Tse=None):
    lo = sum(ei * ti for ei, ti in zip(e, T)); hi = lo + corner * T[0]
    s = math.sqrt(sum((ei * si) ** 2 for ei, si in zip(e, Tse))) if Tse else None
    return {"low": lo, "high": hi, "se": s}
preds = {}
for body in ("A", "B"):
    b = bands[body]
    preds[body] = {"eta_power": pred(b["power_E2"], b["power_E2_corner"], Tv), "eta_cross": pred(b["cross_E1E2"], b["cross_E1E2_corner"], Tv),
                   "T_full_power": pred(b["power_E2"], b["power_E2_corner"], Tf, Tfse), "T_full_cross": pred(b["cross_E1E2"], b["cross_E1E2_corner"], Tf, Tfse)}
out["predictions"] = preds
out["eta_measured"] = {"eta": 0.36611, "ci95": [0.343672, 0.386646]}

# 2. prediction vs symmetric statistic (prediction SE propagated; corner energy at T_full(b0), as in Entry 46)
tests = {}
for name in ("T_full_power", "T_full_cross"):
    p = preds["A"][name]; lam = p["high"]; lam_se = p["se"]
    z = (lam * R_REAL - th) / math.sqrt(se ** 2 + (lam_se * R_REAL) ** 2)
    tests[name] = {"predicted_lambda_pct": 100 * lam, "predicted_se_pct": 100 * lam_se, "z_vs_theta_sym": z, "one_sided_p": float(stats.norm.sf(z))}
out["prediction_tests"] = tests

# 4. Kodak exact sign-flip
m = np.array([2.10e-4, -9.66e-6, -1.40e-4, -6.72e-5, 6.97e-5]); obs = m.mean()
cnt = sum(1 for s in itertools.product([1, -1], repeat=5) if (np.array(s) * m).mean() >= obs - 1e-15)
out["kodak_exact_signflip"] = {"count": cnt, "of": 32, "p": cnt / 32}

json.dump(out, open(os.path.join(V2, "out", "v4_offline.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "band_energy"}, indent=1))
print("band energy (A):", {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in bands["A"].items()})
print("band energy (B):", {k: ([round(x, 4) for x in v] if isinstance(v, list) else round(v, 4)) for k, v in bands["B"].items()})
