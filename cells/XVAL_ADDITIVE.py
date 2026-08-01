# %% X-VAL — is the additive decomposition mechanistic or merely descriptive?
#
# R^2 = 0.959 was fitted and assessed on the SAME 10x5 Kodak matrix: b_y is estimated from
# observations whose device contrasts are then predicted. That is an in-sample decomposition.
#
# Two genuinely held-out tests, both on data already collected:
#   XV1  CROSS-SEED   — fit mu + a_x + b_y on seed-0 adapters only, predict the seed-1 device
#                       contrasts; then reverse. Ten held-out predictions.
#   XV2  OFF-DIAGONAL — estimate b_y from OFF-DIAGONAL cells only, predict the held-out
#                       own-device diagonal cell. The strictest version: the quantity being
#                       predicted never enters the fit.
# Both get a permutation p from shuffling fingerprint labels.
#
# READ: if the effect holds out of sample, call it a mechanistic account. If it falls
# substantially, call it a descriptive additive decomposition. Say which.
import os, json, numpy as np, pandas as pd
from scipy import stats as sps

VARIANT = "raw"                      # "lodo" gives the same picture
d = pd.read_csv(os.path.join(ROOT, "csv", f"c3_measure_{VARIANT}.csv"))
M = d.groupby(["tag", "K"])["rho"].mean().unstack()          # 10 adapters x 5 fingerprints
dev_of = {t: dv for t, dv, _ in all_arms()}
seed_of = {t: s for t, _, s in all_arms()}
DEV = list(M.columns)

def fit_additive(sub):
    """Return (grand, a_x Series, b_y Series) from a sub-matrix of adapters."""
    g = sub.values.mean()
    return g, sub.mean(1) - g, sub.mean(0) - g

def theta_obs(tag):
    own = M.loc[tag, dev_of[tag]]
    oth = np.mean([M.loc[tag, o] for o in DEV if o != dev_of[tag]])
    return float(own - oth)

def theta_pred_from_b(b, dv):
    return float(b[dv] - np.mean([b[o] for o in DEV if o != dv]))

# ---------------- XV1 · cross-seed ----------------
print("=" * 72)
print("XV1  CROSS-SEED: fit on one seed, predict the other")
pred, obs, lab = [], [], []
for fit_seed in (0, 1):
    tr = [t for t in M.index if seed_of[t] == fit_seed]
    te = [t for t in M.index if seed_of[t] != fit_seed]
    _, _, b = fit_additive(M.loc[tr])
    for t in te:
        pred.append(theta_pred_from_b(b, dev_of[t])); obs.append(theta_obs(t))
        lab.append(f"{t}(fit s{fit_seed})")
pred, obs = np.array(pred), np.array(obs)
ss_res = float(((obs - pred) ** 2).sum())
ss_tot = float(((obs - obs.mean()) ** 2).sum())
r2_cv = 1 - ss_res / ss_tot
r, p_r = sps.pearsonr(pred, obs); rs, p_s = sps.spearmanr(pred, obs)
rmse = float(np.sqrt(((obs - pred) ** 2).mean()))
print(f"{'held-out adapter':24s} {'predicted':>12s} {'observed':>12s} {'resid':>11s}")
for l, pv, ov in zip(lab, pred, obs):
    print(f"{l:24s} {pv:+12.3e} {ov:+12.3e} {ov-pv:+11.3e}")
print(f"\n  cross-validated R^2 = {r2_cv:+.3f}    (in-sample was 0.959)")
print(f"  Pearson {r:+.3f} (p={p_r:.4f})   Spearman {rs:+.3f} (p={p_s:.4f})   RMSE {rmse:.3e}")
print(f"  observed spread {obs.max()-obs.min():.3e}  ->  RMSE is "
      f"{100*rmse/(obs.max()-obs.min()):.1f}% of it")

# permutation: shuffle fingerprint labels, redo the whole cross-seed procedure
rng = np.random.default_rng(0); NPERM = 5000; cnt = 0
for _ in range(NPERM):
    Mp = M.copy(); Mp.columns = list(rng.permutation(DEV)); Mp = Mp[DEV]
    pp, oo = [], []
    for fit_seed in (0, 1):
        tr = [t for t in Mp.index if seed_of[t] == fit_seed]
        te = [t for t in Mp.index if seed_of[t] != fit_seed]
        _, _, bp = fit_additive(Mp.loc[tr])
        for t in te:
            own = Mp.loc[t, dev_of[t]]
            oth = np.mean([Mp.loc[t, o] for o in DEV if o != dev_of[t]])
            pp.append(theta_pred_from_b(bp, dev_of[t])); oo.append(float(own - oth))
    pp, oo = np.array(pp), np.array(oo)
    cnt += (1 - ((oo - pp) ** 2).sum() / ((oo - oo.mean()) ** 2).sum()) >= r2_cv
p_perm = (cnt + 1) / (NPERM + 1)
print(f"  permutation p (shuffled fingerprint labels, {NPERM} draws) = {p_perm:.4g}")

# ---------------- XV2 · off-diagonal ----------------
print("\n" + "=" * 72)
print("XV2  OFF-DIAGONAL: estimate b_y from off-diagonal cells only, predict the own-device cell")
Moff = M.copy()
for t in M.index:
    Moff.loc[t, dev_of[t]] = np.nan                          # blank every own-device cell
g_off = float(np.nanmean(Moff.values))
a_off = Moff.mean(1, skipna=True) - g_off
b_off = Moff.mean(0, skipna=True) - g_off
pred_diag = np.array([g_off + a_off[t] + b_off[dev_of[t]] for t in M.index])
obs_diag = np.array([M.loc[t, dev_of[t]] for t in M.index])
resid_diag = obs_diag - pred_diag
r2_diag = 1 - float(((resid_diag) ** 2).sum() / ((obs_diag - obs_diag.mean()) ** 2).sum())
t_diag = resid_diag.mean() / (resid_diag.std(ddof=1) / np.sqrt(len(resid_diag)))
print(f"{'adapter':16s} {'predicted':>12s} {'observed':>12s} {'residual':>11s}")
for t, pv, ov, rv in zip(M.index, pred_diag, obs_diag, resid_diag):
    print(f"{t:16s} {pv:+12.3e} {ov:+12.3e} {rv:+11.3e}")
print(f"\n  held-out diagonal R^2 = {r2_diag:+.3f}")
print(f"  mean own-device residual = {resid_diag.mean():+.3e}  t = {t_diag:+.2f}")
print("  ^ THIS residual is the leakage estimate with all additive structure removed.")

# cluster bootstrap over adapters for the own-device residual (the review's ask)
bs = []
for _ in range(5000):
    idx = rng.integers(0, len(resid_diag), len(resid_diag))
    bs.append(resid_diag[idx].mean())
bs = np.array(bs)
lo, hi = np.percentile(bs, [0.5, 99.5])
print(f"  99% cluster-bootstrap CI on the own-device residual: [{lo:+.3e}, {hi:+.3e}]"
      f"  {'excludes 0' if lo > 0 or hi < 0 else 'includes 0'}")

out = {"variant": VARIANT,
       "xv1_cross_seed": {"r2_cv": r2_cv, "pearson": r, "spearman": rs, "rmse": rmse,
                          "p_perm": float(p_perm), "predicted": pred.tolist(),
                          "observed": obs.tolist()},
       "xv2_off_diagonal": {"r2": r2_diag, "mean_residual": float(resid_diag.mean()),
                            "t": float(t_diag), "ci99": [float(lo), float(hi)]},
       "in_sample_r2": 0.959}
json.dump(out, open(os.path.join(ROOT, "XVAL_additive.json"), "w"), indent=2)

print("\n" + "=" * 72)
if r2_cv > 0.7 and p_perm < 0.05:
    print("VERDICT: the decomposition holds OUT OF SAMPLE — 'mechanistic account' is earned.")
elif r2_cv > 0.4:
    print("VERDICT: partial. Call it a descriptive additive decomposition that predicts")
    print("         held-out device contrasts with moderate accuracy. State the CV R^2.")
else:
    print("VERDICT: does NOT hold out of sample. Report as an in-sample decomposition only,")
    print("         and drop 'mechanistic' from the contribution list.")
print("Report the CROSS-VALIDATED R^2 in the paper, not the in-sample 0.959.")
print("=" * 72)
