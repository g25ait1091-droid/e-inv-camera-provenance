"""Post-hoc check (RESULTS.md Entry 114, part A): the persistent term of the examiner power model.

CPU only, about three minutes (the coverage simulation and the statsmodels REML check take most of it).
Writes out/fv_sigma.json. Reads the archive rows from the Google Drive for desktop mirror (G:).
Not pre-specified: every quantity here is descriptive or a sensitivity analysis, computed after the manuscript
review asked for it. No registered reading changes.

WHAT THE POWER MODEL USES (main eq. power; supplement S11; out/t3_power_v4.json; verify_v2.tpr())
  theta_hat_G ~ N(lambda R_real, sigma_mu^2 + SE_500^2 * 500 / G)
  * sigma_mu = 5.23e-05 is the archive sanity block's SG2 quantity (cells/SANITY_BLOCK_cell.py,
    E_INV_P0_v3/sanity_block.json): ONE adapter (A_raw_s0_r16) against ONE fingerprint (K_A), the standard
    deviation over the off-peak lags of its 500-image mean correlation surface (sd_full 7.01e-05) and of the mean
    over half of its images (sd_half 8.42e-05). If the images averaged independently the ratio would be sqrt(2);
    it was 1.20, and the split sd_full^2 = sigma_mu^2 + s^2, sd_half^2 = sigma_mu^2 + 2 s^2 gives sigma_mu =
    5.23e-05 and s = 4.67e-05. It is the part of that surface that does not shrink with more images. It was never
    measured on the paired contrast (own-fingerprint minus other-fingerprint correlation at zero lag), whose
    common structure (the fingerprint main effect) the paired design and the model's examiner remove.
  * SE_500 = 7.035e-05 is the mean over A_raw_s0 and B_raw_s0 of the standard error of the 500-image mean of the
    per-image paired contrast, from the local re-measurement (out/t1/measure_rows.csv; Entry 50, D7).

WHAT THIS SCRIPT ESTIMATES
  The archive's per-image rows of the 24 primary adapters (12 per body, 500 generations each, generation seed
  770000 + j shared by every adapter) give the paired contrast d[a, j] = rho(gen_aj -> K_own) - rho(gen_aj ->
  K_other). Their adapter means reproduce FINAL_LEDGER primary.per_adapter_A/B exactly (asserted). Within each
  arm the crossed model
        d[a, j] = m_arm + u_a + v_j + e_aj        (adapter a, seed j; u, v, e independent, zero mean)
  separates a persistent per-adapter component sigma_u^2, a seed (content) effect sigma_v^2 shared by the arm's
  adapters, and the adapter-by-seed residual sigma_e^2. The design is balanced, so the ANOVA method of moments
  equals REML inside the parameter space; the constrained REML (closed form) and a statsmodels MixedLM REML fit are
  reported beside it. sigma_u is pooled over the two arms (common sigma_u and sigma_e, arm-specific m and sigma_v).
  An examiner holding ONE personalized model and generating G images with fresh seeds sees
        Var(theta_hat_G) = sigma_u^2 + (sigma_v^2 + sigma_e^2) / G,
  so sigma_u is the model's persistent term and (sigma_v^2 + sigma_e^2) / 500 is its SE_500^2.
  Intervals: exact normal-theory (F with 22 and 10,978 df), a bootstrap over adapters, and a bootstrap over
  adapters and seeds (seed multiplicities carried into the image-noise correction; the naive two-way resample,
  which double-counts image noise when a seed is drawn twice, is kept as a labelled diagnostic only). A seeded
  simulation from the fitted crossed model (200 data sets per cell, u normal or t with 3 df) measures the coverage
  of the three upper limits; the exact interval is the one carried into the macros.

  Secondary, descriptive: the same crossed model on the two local-stack D200 designs with per-image rows at hand
  (G2 second environment, 2000 steps, 6 + 6 adapters x 500 images; the 16000-step design, 6 + 6 x 250), and the
  image noise of a seed-matched symmetric contrast (one adapter per body, the same generation seeds).

  Power: verify_v2.tpr() imported from the repository (as in fv_derived.py), evaluated at the nominal and the
  calibrated max-arm limits for the archive sigma_mu, the paired-contrast estimate and its interval, and zero.

  Corrections after the independent check (Entry 114 addendum, 30 Sep 2026): the G2 outlier's distance with its
  own per-seed deviations and a family-wise p (largest_deviation_detail); the archive value's p and the exact
  interval's coverage under heavy-tailed u (heavy_tail_checks); the seed-matched examiner with a null-specific SE
  (seed_matched_reference.null_specific); the H6 training-set component as a persistent term
  (training_set_component_persistent); the bootstrap method named (bootstrap_method_note).

Run:  python src/fv/fv_sigma.py
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import einv_paths as EINV
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import time
import warnings

import numpy as np
import pandas as pd
from scipy import stats

V2 = EINV.V2
OUT = V2 + "/out"
T1 = OUT + "/t1"
REPO = EINV.REPO
LEDGER = REPO + "/analysis/FINAL_LEDGER.json"
DRIVE = (EINV.MYDRIVE + "/inv_channel")
NUMBERS = V2 + "/paper/fv/numbers.json"
DST = OUT + "/fv_sigma.json"
R_REAL = 0.0356703416571125
J = 500                                    # generations per primary adapter
ROWS = {                                   # archive per-image rows of the primary adapters (Drive mirror)
    "seeds 0-2": DRIVE + "/E_INV_P0_v3/csv/s5_measure.csv",
    "seeds 3-5": DRIVE + "/E_SEEDEXT/csv/b3_measure.csv",
    "seeds 6-11": DRIVE + "/E_SEEDEXT2/csv/sx2_measure.csv",
}
SANITY = DRIVE + "/E_INV_P0_v3/sanity_block.json"
N_BOOT = 20000
BOOT_SEED = 1140930


def load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ============================================================================ data
def primary_rows():
    """Per-image paired contrasts of the 24 primary adapters as two 12 x 500 matrices (seed order 0..11, gen 0..499),
    plus the base model's labelled contrast rho(K_A) - rho(K_B) on the same seeds."""
    frames = []
    for f in ROWS.values():
        d = pd.read_csv(f, usecols=["tag", "gen_idx", "K", "rho_mult"])
        frames.append(d[d.K.isin(["A", "B"])])
    d = pd.concat(frames, ignore_index=True)
    cnt = d.groupby(["tag", "K"]).gen_idx.agg(["count", "nunique", "min", "max"])
    P = d.pivot_table(index=["tag", "gen_idx"], columns="K", values="rho_mult", aggfunc="first")

    def series(tag, own, other):
        c = cnt.loc[(tag, own)]
        assert (c["count"], c["nunique"], c["min"], c["max"]) == (J, J, 0, J - 1), (tag, own, dict(c))
        x = P.loc[tag].sort_index()
        assert list(x.index) == list(range(J)), tag
        return (x[own] - x[other]).to_numpy(float)

    XA = np.array([series(f"A_raw_s{s}_r16", "A", "B") for s in range(12)])
    XB = np.array([series(f"B_raw_s{s}_r16", "B", "A") for s in range(12)])
    base = series("base", "A", "B")
    return XA, XB, base


def local_rows(files, arms_a, arms_b):
    """Local-stack per-image rows: rho_KA - rho_KB (arm A) and rho_KB - rho_KA (arm B), aligned by image index."""
    d = pd.concat([pd.read_csv(T1 + "/" + f, usecols=["arm", "image", "rho_KA", "rho_KB"]) for f in files])
    d["j"] = d.image.str.replace(".png", "", regex=False).astype(int)
    idx = None
    out = []
    for arms, sign in ((arms_a, 1.0), (arms_b, -1.0)):
        m = []
        for a in arms:
            x = d[d.arm == a].sort_values("j")
            assert x.j.is_unique, a
            if idx is None:
                idx = list(x.j)
            assert list(x.j) == idx, f"{a}: images not aligned with the other adapters"
            m.append(sign * (x.rho_KA - x.rho_KB).to_numpy(float))
        out.append(np.array(m))
    return out[0], out[1], len(idx)


# ============================================================================ crossed model
def two_way(X):
    """Balanced two-way ANOVA without replication (rows adapters, columns seeds)."""
    I, Jc = X.shape
    m = X.mean()
    ra, cs = X.mean(1), X.mean(0)
    res = X - ra[:, None] - cs[None, :] + m
    ss_a, ss_s, ss_e = Jc * ((ra - m) ** 2).sum(), I * ((cs - m) ** 2).sum(), (res ** 2).sum()
    return dict(I=I, J=Jc, ss_a=float(ss_a), ss_s=float(ss_s), ss_e=float(ss_e),
                df_a=I - 1, df_s=Jc - 1, df_e=(I - 1) * (Jc - 1))


def reml_constrained(ss_a, df_a, ss_e, df_e, Jc):
    """REML for the balanced model: lambda_A = s_e^2 + J s_u^2 >= lambda_E = s_e^2. The unconstrained maximum is
    the mean squares; if MS_A < MS_E the maximum on the boundary pools them (s_u^2 = 0)."""
    msa, mse = ss_a / df_a, ss_e / df_e
    if msa >= mse:
        return (msa - mse) / Jc, mse
    return 0.0, (ss_a + ss_e) / (df_a + df_e)


def components(mats, Jc):
    """Per-arm and pooled components. Pooled: common sigma_u and sigma_e, arm-specific mean and seed effect."""
    arms = {}
    for name, X in mats.items():
        t = two_way(X)
        msa, mss, mse = t["ss_a"] / t["df_a"], t["ss_s"] / t["df_s"], t["ss_e"] / t["df_e"]
        su2_r, se2_r = reml_constrained(t["ss_a"], t["df_a"], t["ss_e"], t["df_e"], Jc)
        F = msa / mse
        arms[name] = {"n_adapters": t["I"], "n_seeds": t["J"], "MS_adapter": msa, "MS_seed": mss, "MS_resid": mse,
                      "df_adapter": t["df_a"], "df_seed": t["df_s"], "df_resid": t["df_e"],
                      "sigma_u2_mom": (msa - mse) / Jc, "sigma_v2_mom": (mss - mse) / t["I"], "sigma_e2": mse,
                      "sigma_u2_reml": su2_r, "sigma_e2_reml": se2_r,
                      "sigma_u_reml": math.sqrt(su2_r), "sigma_v": math.sqrt(max((mss - mse) / t["I"], 0.0)),
                      "sigma_e": math.sqrt(mse),
                      "F_adapter": F, "F_p_upper": float(stats.f.sf(F, t["df_a"], t["df_e"])),
                      "adapter_means": [float(x) for x in X.mean(1)],
                      "sd_adapter_means": float(X.mean(1).std(ddof=1)),
                      "_ss": t}
    ss_a = sum(a["_ss"]["ss_a"] for a in arms.values())
    df_a = sum(a["_ss"]["df_a"] for a in arms.values())
    ss_e = sum(a["_ss"]["ss_e"] for a in arms.values())
    df_e = sum(a["_ss"]["df_e"] for a in arms.values())
    msa, mse = ss_a / df_a, ss_e / df_e
    su2_r, se2_r = reml_constrained(ss_a, df_a, ss_e, df_e, Jc)
    F = msa / mse
    sv2 = [a["sigma_v2_mom"] for a in arms.values()]
    pooled = {"MS_adapter": msa, "MS_resid": mse, "df_adapter": df_a, "df_resid": df_e,
              "sigma_u2_mom": (msa - mse) / Jc, "sigma_u2_reml": su2_r, "sigma_u_reml": math.sqrt(su2_r),
              "sigma_e2": mse, "sigma_e": math.sqrt(mse),
              "sigma_v2_mean_of_arms": float(np.mean(sv2)), "sigma_v": math.sqrt(max(float(np.mean(sv2)), 0.0)),
              "F_adapter": F, "F_p_upper": float(stats.f.sf(F, df_a, df_e)),
              "sd_adapter_means_pooled_within_arm": math.sqrt(ss_a / Jc / df_a)}
    for a in arms.values():
        a.pop("_ss")
    return arms, pooled


def exact_interval(pooled, Jc, conf=0.95):
    """Normal-theory interval for sigma_u from F = MS_A / MS_E ~ rho F(df_a, df_e), rho = 1 + J sigma_u^2 / sigma_e^2
    (sigma_e^2 taken at MS_E, which has 10,978 df)."""
    F, dfa, dfe, mse = pooled["F_adapter"], pooled["df_adapter"], pooled["df_resid"], pooled["MS_resid"]

    def su(rho):
        return math.sqrt(max(rho - 1.0, 0.0) * mse / Jc)

    a = 1 - conf
    return {"two_sided_95": [su(F / stats.f.ppf(1 - a / 2, dfa, dfe)), su(F / stats.f.ppf(a / 2, dfa, dfe))],
            "one_sided_95_upper": su(F / stats.f.ppf(0.05, dfa, dfe)),
            "one_sided_99_upper": su(F / stats.f.ppf(0.01, dfa, dfe))}


def p_value_for(sigma, pooled, Jc):
    """One-sided p of the hypothesis sigma_u >= sigma, given the observed F (small p: the data exclude sigma)."""
    rho = 1.0 + Jc * sigma ** 2 / pooled["MS_resid"]
    return float(stats.f.cdf(pooled["F_adapter"] / rho, pooled["df_adapter"], pooled["df_resid"]))


def heavy_tail_checks(pooled, Jc, sigmas, rng, n_rep=200000, I=12):
    """Correction (independent check, Entry 114 addendum). The one-sided p of a persistent component sigma and the
    coverage of the exact one-sided 95 % upper limit when u is not normal: u = sigma * t_nu / sd(t_nu) (nu = 5, 3)
    or normal; 2 arms x I adapters, J seeds; the adapter means carry image noise sigma_e / sqrt(J) and MS_E is
    drawn from its chi-square (10,978 df). p = P(F <= F_obs); coverage = P(exact upper >= sigma)."""
    F_obs, dfa, dfe = pooled["F_adapter"], pooled["df_adapter"], pooled["df_resid"]
    se = math.sqrt(pooled["MS_resid"])
    f05 = stats.f.ppf(0.05, dfa, dfe)
    out = {}
    for dist in ("normal", "t5", "t3"):
        row = {}
        for s in sigmas:
            if dist == "normal":
                u = rng.standard_normal((n_rep, 2, I))
            else:
                nu = int(dist[1:])
                u = rng.standard_t(nu, (n_rep, 2, I)) / math.sqrt(nu / (nu - 2))
            m = s * u + se / math.sqrt(Jc) * rng.standard_normal((n_rep, 2, I))
            msa = Jc * ((m - m.mean(2, keepdims=True)) ** 2).sum((1, 2)) / dfa
            mse = pooled["MS_resid"] * rng.chisquare(dfe, n_rep) / dfe
            F = msa / mse
            upper2 = np.clip(F / f05 - 1.0, 0.0, None) * mse / Jc
            row[f"{s:.4g}"] = {"p_F_le_Fobs": float((F <= F_obs).mean()),
                               "coverage_exact_one_sided_95": float((upper2 >= s ** 2).mean())}
        out[dist] = row
    return {"n_rep": n_rep, "cells": out,
            "what": ("u normal or Student t (5 or 3 df) scaled to sd sigma; p = P(F <= F_obs) is the one-sided p of "
                     "a persistent component as large as sigma; coverage of the exact one-sided 95 % upper limit")}


def bootstrap(XA, XB, mse_full, rng, n_boot, seeds_too, naive=False):
    """sigma_u^2* = pooled within-arm variance of the resampled adapter means - sigma_e^2 * sum_j w_j^2.
    Adapters are resampled within each arm; with seeds_too, the 500 generation seeds are resampled once per
    replicate for both arms (the seed bank is shared), w_j being the multiplicity of seed j over 500. sigma_e^2 is
    held at its full-data value (10,978 df). naive=True instead recomputes the two-way ANOVA on the resampled
    matrix as if every drawn seed were a new one - it counts a duplicated seed's image noise twice in the adapter
    means but not in the residual mean square, and is kept only as a labelled diagnostic."""
    n = XA.shape[0]
    Jc = XA.shape[1]
    est = np.empty(n_boot)
    for b in range(n_boot):
        ia, ib = rng.integers(0, n, n), rng.integers(0, n, n)
        if seeds_too:
            js = rng.integers(0, Jc, Jc)
            w = np.bincount(js, minlength=Jc) / Jc
        else:
            js, w = None, np.full(Jc, 1.0 / Jc)
        if naive:
            A, B = XA[ia][:, js], XB[ib][:, js]
            tA, tB = two_way(A), two_way(B)
            msa = (tA["ss_a"] + tB["ss_a"]) / (tA["df_a"] + tB["df_a"])
            mse = (tA["ss_e"] + tB["ss_e"]) / (tA["df_e"] + tB["df_e"])
            est[b] = (msa - mse) / Jc
            continue
        tha, thb = XA[ia] @ w, XB[ib] @ w
        s2 = 0.5 * (tha.var(ddof=1) + thb.var(ddof=1))
        est[b] = s2 - mse_full * float((w ** 2).sum())
    s = np.sqrt(np.clip(est, 0.0, None))
    return {"n_boot": n_boot, "P_sigma_u2_le_0": float((est <= 0).mean()),
            "percentile_95": [float(np.percentile(s, 2.5)), float(np.percentile(s, 97.5))],
            "percentile_95_upper_one_sided": float(np.percentile(s, 95)),
            "median": float(np.median(s))}


def interval_calibration(rng, sigma_v, sigma_e, n_rep=200, n_boot=600, I=12, Jc=J):
    """Coverage of the three one-sided 95 % upper limits for sigma_u on data simulated from the crossed model with
    the primary design's shape (two arms of 12 adapters x 500 seeds) and its fitted sigma_v and sigma_e; u normal or
    Student t with 3 df scaled to sd sigma_u (a heavy-tailed persistent component)."""
    dfa, dfe = 2 * (I - 1), 2 * (I - 1) * (Jc - 1)

    def upper_exact(XA, XB):
        tA, tB = two_way(XA), two_way(XB)
        msa, mse = (tA["ss_a"] + tB["ss_a"]) / dfa, (tA["ss_e"] + tB["ss_e"]) / dfe
        rho = (msa / mse) / stats.f.ppf(0.05, dfa, dfe)
        return math.sqrt(max(rho - 1.0, 0.0) * mse / Jc), mse

    out = {}
    for dist in ("normal", "t3"):
        for su in (0.0, 2e-05, 4e-05):
            cov, mean_up = np.zeros(3), np.zeros(3)
            for _ in range(n_rep):
                def u():
                    if dist == "normal":
                        return rng.normal(0.0, su, I)
                    return su * rng.standard_t(3, I) / math.sqrt(3.0)
                XA = u()[:, None] + rng.normal(0, sigma_v, Jc)[None, :] + rng.normal(0, sigma_e, (I, Jc))
                XB = u()[:, None] + rng.normal(0, sigma_v, Jc)[None, :] + rng.normal(0, sigma_e, (I, Jc))
                ue, mse = upper_exact(XA, XB)
                ua = bootstrap(XA, XB, mse, rng, n_boot, seeds_too=False)["percentile_95_upper_one_sided"]
                us = bootstrap(XA, XB, mse, rng, n_boot, seeds_too=True)["percentile_95_upper_one_sided"]
                ups = np.array([ue, ua, us])
                cov += ups >= su
                mean_up += ups
            out[f"{dist}_sigma_u_{su:.0e}"] = {
                "coverage_one_sided_95": dict(zip(("exact", "bootstrap_adapters", "bootstrap_adapters_and_seeds"),
                                                  (cov / n_rep).tolist())),
                "mean_upper": dict(zip(("exact", "bootstrap_adapters", "bootstrap_adapters_and_seeds"),
                                       (mean_up / n_rep).tolist()))}
    return {"n_rep": n_rep, "n_boot": n_boot, "sigma_v": sigma_v, "sigma_e": sigma_e, "cells": out,
            "reading": ("nominal coverage 0.95: the exact interval is calibrated when u is normal and slightly "
                        "anti-conservative with heavy tails; the adapter bootstrap under-covers; the bootstrap over "
                        "adapters and seeds over-covers, because a resampled seed bank holds about 316 distinct seeds "
                        "and so carries more image noise than the design")}


def statsmodels_reml(X, arm):
    """Crossed random effects fitted by statsmodels MixedLM (REML), as a check of the closed form."""
    import statsmodels
    import statsmodels.formula.api as smf
    scale = 1e4                                      # fit in units of 1e-4 for numerical conditioning
    I, Jc = X.shape
    df = pd.DataFrame({"d": (X * scale).ravel(), "adapter": np.repeat(np.arange(I), Jc),
                       "seed": np.tile(np.arange(Jc), I), "g": 1})
    md = smf.mixedlm("d ~ 1", df, groups="g", re_formula="0",
                     vc_formula={"adapter": "0 + C(adapter)", "seed": "0 + C(seed)"})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = md.fit(reml=True, method=["lbfgs"])
    names = list(r.model.exog_vc.names)
    vc = dict(zip(names, (float(v) / scale ** 2 for v in r.vcomp)))
    return {"statsmodels": statsmodels.__version__, "converged": bool(r.converged),
            "sigma_u2": vc["adapter"], "sigma_v2": vc["seed"], "sigma_e2": float(r.scale) / scale ** 2}


# ============================================================================ power model
def import_verifier():
    """Import verify_v2 for its tpr() model; it runs its checks at import and exits (both captured), as in
    fv_derived.py."""
    spec = importlib.util.spec_from_file_location("verify_v2", REPO + "/verify_v2.py")
    mod = importlib.util.module_from_spec(spec)
    buf = io.StringIO()
    status = "imported"
    with contextlib.redirect_stdout(buf):
        try:
            spec.loader.exec_module(mod)
        except SystemExit as e:
            status = f"module ran its checks and exited with code {e.code}"
        except Exception as e:  # tpr() is defined before most checks; keep it if it exists
            status = f"module raised {type(e).__name__} during its checks: {e}"
    assert hasattr(mod, "tpr"), "verify_v2.tpr not defined"
    last = [l for l in buf.getvalue().splitlines() if "checks agree" in l]
    return mod, status, (last[-1] if last else "")


def z_alpha(M):
    return float(stats.norm.ppf(1 - 0.01 / (M - 1)))


def make_power(vv):
    def tpr(U, G, sigma, se500, M):
        """verify_v2.tpr() at transfer U. The one limit it cannot evaluate, sigma = 0 with unlimited images
        (zero variance), is its limit, 1."""
        if sigma == 0.0 and math.isinf(G):
            return 1.0
        vv.U_DEVICE = U                 # tpr() reads the module-level U_DEVICE at call time
        return float(vv.tpr(G, sigma_mu=sigma, se500=se500, M=M))

    def g_needed(U, sigma, se500, M, target):
        """Exact images for TPR = target under the same model: sigma_G = U / (z_alpha + z_target); None if the
        persistent term alone exceeds it (target unattainable at any G)."""
        s_star = U / (z_alpha(M) + float(stats.norm.ppf(target)))
        room = s_star ** 2 - sigma ** 2
        if room <= 0:
            return None
        g = 500.0 * se500 ** 2 / room
        assert abs(tpr(U, g, sigma, se500, M) - target) < 1e-9
        return g
    return tpr, g_needed


def power_block(tpr, g_needed, U, sigma, se500):
    rows = {}
    for M in (2, 5, 50):
        r = {f"TPR_G{g}": tpr(U, g, sigma, se500, M) for g in (500, 5000)}
        r["TPR_G_inf"] = tpr(U, math.inf, sigma, se500, M)
        for tgt, k in ((0.5, "G_for_TPR50"), (0.9, "G_for_TPR90")):
            g = g_needed(U, sigma, se500, M, tgt)
            r[k] = g
            r[k + "_ceil"] = None if g is None else int(math.ceil(g))
        rows[f"M{M}"] = r
    return rows


# ============================================================================ main
def main():
    t0 = time.time()
    res = {"entry": "RESULTS.md Entry 114",
           "status": "post hoc; descriptive and sensitivity analyses; not pre-specified; no registered reading changes",
           "script": "src/fv/fv_sigma.py"}

    # ---------------------------------------------------------------- inputs of the power model as filed
    pw = load(OUT + "/t3_power_v4.json")
    inp = pw["inputs"]
    h6 = load(OUT + "/h6_calibrated_limit.json")["primary_d200"]
    sb = load(SANITY)["sg2"]
    s_img_sg2 = math.sqrt(sb["sd_half"] ** 2 - sb["sd_full"] ** 2)
    s_mu_sg2 = math.sqrt(2 * sb["sd_full"] ** 2 - sb["sd_half"] ** 2)
    assert abs(s_mu_sg2 - inp["sigma_mu"]) < 5e-8, (s_mu_sg2, inp["sigma_mu"])
    res["archive_sigma_mu"] = {
        "value": inp["sigma_mu"], "source": "out/t3_power_v4.json inputs.sigma_mu",
        "what_it_is": ("SG2 of the archive sanity block (cells/SANITY_BLOCK_cell.py): off-peak standard deviation "
                       "of the mean correlation surface of ONE adapter (A_raw_s0_r16) against K_A, over all 500 "
                       "images (sd_full) and over the even-indexed half (sd_half); sigma_mu^2 = 2 sd_full^2 - "
                       "sd_half^2 is the part that does not shrink with the number of images. A property of the "
                       "correlation surface at non-zero lags, not of the zero-lag paired contrast."),
        "sanity_block_sg2": sb, "sanity_block_file": SANITY,
        "recomputed_sigma_mu": s_mu_sg2, "recomputed_image_part": s_img_sg2,
        "SE_img_500_filed": inp["SE_img_500"],
        "SE_img_500_what": ("mean of the 500-image standard errors of the per-image paired contrast of A_raw_s0 and "
                            "B_raw_s0, local re-measurement (out/t1/measure_rows.csv; Entry 50, D7)")}

    # ---------------------------------------------------------------- primary per-image rows
    XA, XB, base = primary_rows()
    led = load(LEDGER)["primary"]
    dev = max(float(np.max(np.abs(XA.mean(1) - np.array(led["per_adapter_A"])))),
              float(np.max(np.abs(XB.mean(1) - np.array(led["per_adapter_B"])))))
    assert dev < 1e-15, dev
    res["data"] = {
        "rows": {k: {"file": v, "sha256": sha256(v)} for k, v in ROWS.items()},
        "adapters": {"A": [f"A_raw_s{s}_r16" for s in range(12)], "B": [f"B_raw_s{s}_r16" for s in range(12)]},
        "images_per_adapter": J, "generation_seeds": "770000 + j, j = 0..499, identical for every adapter",
        "paired_contrast": "arm A: rho_mult(K_A) - rho_mult(K_B); arm B: rho_mult(K_B) - rho_mult(K_A)",
        "ledger_reproduction_max_abs_diff": dev,
        "ledger_reproduction": "all 24 adapter means equal FINAL_LEDGER primary.per_adapter_A/B"}

    # ---------------------------------------------------------------- crossed model, primary design
    arms, pooled = components({"A": XA, "B": XB}, J)
    for name, X in (("A", XA), ("B", XB)):
        try:
            arms[name]["reml_statsmodels"] = statsmodels_reml(X, name)
        except Exception as e:  # the closed form is the estimate; the fit is a check
            arms[name]["reml_statsmodels"] = {"error": f"{type(e).__name__}: {e}"}
    ci = exact_interval(pooled, J)
    rng = np.random.default_rng(BOOT_SEED)
    boot_a = bootstrap(XA, XB, pooled["MS_resid"], rng, N_BOOT, seeds_too=False)
    boot_as = bootstrap(XA, XB, pooled["MS_resid"], rng, N_BOOT, seeds_too=True)
    boot_naive = bootstrap(XA, XB, pooled["MS_resid"], rng, 4000, seeds_too=True, naive=True)
    calib = interval_calibration(np.random.default_rng(BOOT_SEED + 1), pooled["sigma_v"], pooled["sigma_e"])

    # split-half reliability of the adapter means (within-arm centred; even vs odd seeds)
    def centred(X, cols):
        t = X[:, cols].mean(1)
        return t - t.mean()
    ev = np.concatenate([centred(XA, slice(0, None, 2)), centred(XB, slice(0, None, 2))])
    od = np.concatenate([centred(XA, slice(1, None, 2)), centred(XB, slice(1, None, 2))])
    split_r = float(np.corrcoef(ev, od)[0, 1])

    su_hat = pooled["sigma_u_reml"]
    sd_noise = math.sqrt(pooled["MS_resid"] / J)
    res["primary_crossed_model"] = {
        "model": "d[a,j] = m_arm + u_a + v_j + e_aj within each arm; common sigma_u, sigma_e; arm-specific m, sigma_v",
        "arms": arms, "pooled": pooled,
        "sigma_u_estimate": su_hat,
        "sigma_u_interval_exact": ci,
        "sigma_u_bootstrap_adapters": boot_a,
        "sigma_u_bootstrap_adapters_and_seeds": boot_as,
        "diagnostic_naive_two_way_bootstrap_not_used": dict(
            boot_naive, note=("recomputing the ANOVA on seed-resampled columns counts a duplicated seed's image noise "
                              "twice in each adapter mean but once in the residual mean square, so it adds about "
                              "sigma_e^2/500 to sigma_u^2 (sqrt = 4.7e-05); not an estimate of sigma_u")),
        "interval_calibration_simulation": calib,
        "interval_used_for_macros": ("sigma_u_interval_exact.two_sided_95: exact under the normal crossed model, where "
                                     "the seed effects cancel from the adapter-mean deviations, and calibrated in "
                                     "interval_calibration_simulation; the bootstraps are reported beside it"),
        "archive_sigma_mu_one_sided_p": p_value_for(inp["sigma_mu"], pooled, J),
        "heavy_tail_checks": heavy_tail_checks(pooled, J, (ci["two_sided_95"][1], inp["sigma_mu"]),
                                               np.random.default_rng(BOOT_SEED + 2)),
        "bootstrap_method_note": (
            "both bootstraps estimate sigma_u^2* as the pooled within-arm variance of the resampled adapter means "
            "minus sigma_e^2 * sum_j w_j^2, sigma_e^2 held at its full-data value (w_j = multiplicity of seed j / "
            "500), so a seed drawn twice counts its image noise once; recomputing the two-way ANOVA on each "
            "seed-resampled matrix (diagnostic_naive_two_way_bootstrap_not_used) double-counts it"),
        "archive_sigma_mu_one_sided_p_meaning": ("P(F <= F_obs / rho0), rho0 = 1 + 500 sigma_mu^2 / MS_resid: small p "
                                                 "means the spread of the 24 adapter means is too small for a "
                                                 "persistent component as large as the archive value"),
        "sd_adapter_means": {
            "observed_pooled_within_arm": pooled["sd_adapter_means_pooled_within_arm"],
            "predicted_image_noise_only": sd_noise,
            "predicted_with_archive_sigma_mu": math.sqrt(inp["sigma_mu"] ** 2 + sd_noise ** 2),
            "note": ("the seeds are shared, so the seed effect is common to an arm's adapters and does not spread "
                     "their means; image noise alone predicts sqrt(sigma_e^2/500)")},
        "split_half_r": split_r,
        "split_half_note": "correlation over the 24 adapters of their even-seed and odd-seed means, centred within arm",
    }

    # ---------------------------------------------------------------- seed effect and image-level noise
    sv2 = pooled["sigma_v2_mean_of_arms"]
    se2 = pooled["sigma_e2"]
    cA, cB = XA.mean(0), XB.mean(0)
    within_se = np.concatenate([XA.std(1, ddof=1), XB.std(1, ddof=1)]) / math.sqrt(J)
    pair_se, pair_r = [], []
    for a in range(12):
        for b in range(12):
            pair_se.append(((XA[a] + XB[b]) / 2).std(ddof=1) / math.sqrt(J))
            pair_r.append(np.corrcoef(XA[a], XB[b])[0, 1])
    pair_se = np.array(pair_se)
    res["seed_effect_and_image_noise"] = {
        "sigma_v_per_image": math.sqrt(sv2), "sigma_e_per_image": math.sqrt(se2),
        "seed_share_of_per_image_variance": sv2 / (sv2 + se2),
        "sigma_v_per_arm": {k: arms[k]["sigma_v"] for k in arms},
        "corr_seed_means_A_vs_B": float(np.corrcoef(cA, cB)[0, 1]),
        "corr_seed_means_A_vs_base_labelled": float(np.corrcoef(cA, base)[0, 1]),
        "corr_seed_means_B_vs_base_labelled": float(np.corrcoef(cB, base)[0, 1]),
        "mean_per_image_corr_A_adapter_vs_B_adapter": float(np.mean(pair_r)),
        "SE_500_single_adapter": {
            "rms_over_24": float(np.sqrt((within_se ** 2).mean())), "min": float(within_se.min()),
            "max": float(within_se.max()), "A_raw_s0": float(within_se[0]), "B_raw_s0": float(within_se[12]),
            "crossed_model_sqrt_(sv2+se2)/500": math.sqrt((sv2 + se2) / J),
            "filed_value_used_by_model": inp["SE_img_500"],
            "note": ("the filed 7.035e-05 comes from the local re-measurement of A_raw_s0 and B_raw_s0; the archive rows "
                     "give 7.17e-05 and 6.93e-05 for the same two adapters and 7.21e-05 over all 24")},
        "SE_500_seed_matched_symmetric": {
            "rms_over_144_pairs": float(np.sqrt((pair_se ** 2).mean())), "min": float(pair_se.min()),
            "max": float(pair_se.max()),
            "what": ("one adapter of each body, the same generation seeds, (d_A + d_B)/2 per image: the seed-dependent "
                     "content shared by the two adapters largely cancels, and so do the fingerprint main effects")},
        "examiner": ("An examiner holding ONE personalized model and generating G images with fresh seeds faces "
                     "sigma_u^2 + (sigma_v^2 + sigma_e^2)/G: u_a persists; the seed effect and the residual average "
                     "out. The seed effect is common to adapters of both bodies at a given seed (seed-mean "
                     "correlation between arms -0.84 in own-minus-other terms), so an examiner who also generates "
                     "with the same seeds from a reference adapter trained on the other candidate camera removes most "
                     "of it and needs no main-effect calibration."),
        "scope": ("sigma_u is the spread between adapters that share one training set and one fingerprint estimate "
                  "per body; training-set and estimation components common to an arm (H6: sd 2.2e-05 and 1.2e-05 per "
                  "arm) are not in it and are not in the power model either"),
    }
    res["seed_effect_and_image_noise"]["scope_h6_components"] = {
        "training_sd": load(OUT + "/h6_calibrated_limit.json")["training_sd"],
        "estimation_sd": load(OUT + "/h6_calibrated_limit.json")["estimation_sd"],
        "source": "out/h6_calibrated_limit.json"}

    # ---------------------------------------------------------------- secondary designs (local stack, descriptive)
    sec = {}
    GA, GB, jg = local_rows(["measure_rows_nomark.csv", "measure_rows_nomarkB.csv", "measure_rows_nomarkrep.csv"],
                            [f"nomark_s{i}" for i in range(6)], [f"nomarkB_s{i}" for i in range(6)])
    S = {}
    for f in ("summary_nomark.json", "summary_nomarkB.json", "summary_nomarkrep.json"):
        S.update(load(T1 + "/" + f)["arms"])
    chk = max(abs(GA[i].mean() - S[f"nomark_s{i}"]["natural_paired_KA_minus_KB"]) for i in range(6))
    chk = max(chk, max(abs(GB[i].mean() + S[f"nomarkB_s{i}"]["natural_paired_KA_minus_KB"]) for i in range(6)))
    assert chk < 1e-12, chk
    g_arms, g_pool = components({"A": GA, "B": GB}, jg)
    drop = {}
    for k in range(6):
        _, p_k = components({"A": np.delete(GA, k, 0), "B": GB}, jg)
        drop[f"nomark_s{k}"] = {"sigma_u": p_k["sigma_u_reml"], "F_p": p_k["F_p_upper"]}
    outlier = int(np.argmax(np.abs(GA.mean(1) - GA.mean())))
    # Correction (independent check, Entry 114 addendum): the distance of the outlier from the other five
    # measured with its OWN per-seed deviations (the seed effect cancels; its residual variance is larger than
    # the pooled value), with the SE of a difference under the pooled model, its persistence over the seed
    # bank, and a family-wise p for being the most extreme of the 48 adapters examined (24 primary, 12 G2,
    # 12 at 16000 steps).
    devs = GA[outlier] - np.delete(GA, outlier, 0).mean(0)
    se_own = float(devs.std(ddof=1) / math.sqrt(jg))
    z_own = float(devs.mean() / se_own)
    se_diff = math.sqrt(g_pool["MS_resid"] / jg * (1.0 + 1.0 / 5.0))
    p_one = float(stats.norm.sf(abs(z_own)) * 2.0)
    resid_ratio = []
    for a in range(6):
        dv = GA[a] - np.delete(GA, a, 0).mean(0)
        resid_ratio.append(float(dv.var(ddof=1) / (g_arms["A"]["MS_resid"] * (1.0 + 1.0 / 5.0))))
    outlier_detail = {
        "deviation_from_other_five": float(devs.mean()),
        "se_own_per_seed_deviations": se_own, "z_own": z_own,
        "z_pooled_difference_se": float(devs.mean() / se_diff),
        "ci95_deviation": [float(devs.mean() - 1.959964 * se_own), float(devs.mean() + 1.959964 * se_own)],
        "nominal_limit_in_ncc_units": None,             # filled below from t3_power_v4.json
        "even_seeds": float(devs[0::2].mean()), "odd_seeds": float(devs[1::2].mean()),
        "quarter_means": [float(devs[k * (jg // 4):(k + 1) * (jg // 4)].mean()) for k in range(4)],
        "two_sided_p_single": p_one,
        "familywise_p_most_extreme_of_48": float(1.0 - (1.0 - p_one) ** 48),
        "residual_variance_ratio_arm_A": resid_ratio,
        "reading": ("persistent over the seed bank (positive in every quarter), but as the most extreme of 48 "
                    "adapters its family-wise p is about 0.1; a persistent deviation larger than the headline "
                    "limit, and a heavy-tailed persistent term, are not established. The G2 F test is somewhat "
                    "anti-conservative because this adapter's residual variance exceeds the pooled value.")}
    sec["G2_local_2000_steps"] = {
        "adapters": "nomark_s0-5 (body A), nomarkB_s0-5 (body B); local stack; 500 images each; same seed bank",
        "summary_reproduction_max_abs_diff": chk,
        "pooled": g_pool, "per_arm_F_p": {k: v["F_p_upper"] for k, v in g_arms.items()},
        "per_arm_sigma_u": {k: v["sigma_u_reml"] for k, v in g_arms.items()},
        "adapter_means": {k: v["adapter_means"] for k, v in g_arms.items()},
        "sigma_u_interval_exact": exact_interval(g_pool, jg),
        "leave_one_out_arm_A": drop,
        "largest_deviation_adapter": f"nomark_s{outlier}",
        "largest_deviation_value": float(GA[outlier].mean()),
        "largest_deviation_pct_of_R_real": 100 * float(GA[outlier].mean()) / R_REAL,
        "largest_deviation_z_vs_image_noise": float((GA[outlier].mean() - np.delete(GA, outlier, 0).mean())
                                                    / math.sqrt(g_pool["MS_resid"] / jg)),
        "largest_deviation_z_note": ("divides by the SE of ONE 500-image mean; overstates the distance (see "
                                     "largest_deviation_detail.z_own, the corrected value)"),
        "largest_deviation_detail": outlier_detail}
    HA, HB, jh = local_rows(["measure_rows_dose16k.csv", "measure_rows_dose16krep.csv", "measure_rows_dose16krep2.csv"],
                            [f"dose16k_A_s{i}" for i in range(6)], [f"dose16k_B_s{i}" for i in range(6)])
    h_arms, h_pool = components({"A": HA, "B": HB}, jh)
    sec["dose_16000_steps"] = {
        "adapters": "dose16k_A_s0-5, dose16k_B_s0-5; local stack; 250 images each; same seed bank",
        "pooled": h_pool, "per_arm_F_p": {k: v["F_p_upper"] for k, v in h_arms.items()},
        "sigma_u_interval_exact": exact_interval(h_pool, jh)}
    res["secondary_designs_descriptive"] = sec

    # ---------------------------------------------------------------- power
    vv, status, summary = import_verifier()
    tpr, g_needed = make_power(vv)
    U_nom, U_cal = inp["U_device"], h6["U_device"]
    se500 = inp["SE_img_500"]
    sig = {"archive_sigma_mu": inp["sigma_mu"],
           "paired_estimate": su_hat,
           "paired_low_95": ci["two_sided_95"][0],
           "paired_high_95": ci["two_sided_95"][1],
           "paired_upper_99_one_sided": ci["one_sided_99_upper"],
           "G2_local_estimate": g_pool["sigma_u_reml"],
           "zero": 0.0}
    power = {"model": ("verify_v2.tpr(G, sigma_mu, se500, M): TPR = 1 - Phi(z_{1-0.01/(M-1)} - U / sqrt(sigma^2 + "
                       "se500^2 * 500 / G)); U set per row; sigma = 0 with G = inf is its limit, 1"),
             "verifier_import": status, "verifier_summary": summary,
             "inputs": {"U_nominal": U_nom, "U_calibrated": U_cal, "c": h6["c"], "SE_img_500": se500,
                        "sigma_cases": sig,
                        "sources": ["out/t3_power_v4.json inputs", "out/h6_calibrated_limit.json primary_d200"]},
             "nominal": {k: power_block(tpr, g_needed, U_nom, s, se500) for k, s in sig.items()},
             "calibrated": {k: power_block(tpr, g_needed, U_cal, s, se500) for k, s in sig.items()}}
    power["ten_times_nominal_G_for_TPR50"] = {
        k: {f"M{M}": g_needed(10 * U_nom, s, se500, M, 0.5) for M in (2, 5, 50)} for k, s in sig.items()}
    power["sigma_critical"] = {
        f"{lim}_M2_TPRinf_{int(t * 100)}": U / (z_alpha(2) + float(stats.norm.ppf(t)))
        for lim, U in (("nominal", U_nom), ("calibrated", U_cal)) for t in (0.5, 0.9)}
    power["sigma_critical_meaning"] = ("the persistent component above which no number of images reaches that TPR "
                                       "(two candidates, 1 % FPR)")
    # seed-matched examiner: (d_A + d_B)/2 over one adapter per body; persistent part (u_A + u_B)/2
    se_sm = res["seed_effect_and_image_noise"]["SE_500_seed_matched_symmetric"]["rms_over_144_pairs"]
    s_hi = ci["two_sided_95"][1]
    power["seed_matched_reference"] = {
        "SE_img_500": se_sm, "persistent_sigma": su_hat / math.sqrt(2),
        "nominal": power_block(tpr, g_needed, U_nom, su_hat / math.sqrt(2), se_sm)["M2"],
        "calibrated": power_block(tpr, g_needed, U_cal, su_hat / math.sqrt(2), se_sm)["M2"],
        "persistent_sigma_at_paired_high_95": s_hi / math.sqrt(2),
        "nominal_at_paired_high_95": power_block(tpr, g_needed, U_nom, s_hi / math.sqrt(2), se_sm)["M2"],
        "calibrated_at_paired_high_95": power_block(tpr, g_needed, U_cal, s_hi / math.sqrt(2), se_sm)["M2"],
        "what": ("an examiner who also holds a reference adapter trained on the other candidate camera and generates "
                 "from both with the same seeds and caption; image term from the 144 cross-arm adapter pairs; "
                 "persistent term sigma_u/sqrt(2) at the paired estimate; any training-set difference between the "
                 "suspect and the reference is not in this design")}
    # transfer (multiple of the nominal limit) at which fifty images give TPR 0.5, two candidates
    def mult50(sigma, s500, U_ref):
        s_star2 = sigma ** 2 + s500 ** 2 * 500.0 / 50.0
        return z_alpha(2) * math.sqrt(s_star2) / U_ref
    # Correction (independent check, Entry 114 addendum): the seed-matched contrast has a different SE under the
    # null. With the suspect trained on the SAME camera as the reference, both are adapters of one arm and the seed
    # term cancels exactly: per-image variance 2 sigma_e,arm^2 / 4. verify_v2.tpr() uses one SE for both
    # hypotheses, so the figures above are conservative. Two orientations (reference trained on body B or on
    # body A); with K reference adapters the null variance is sigma_e^2 (1 + 1/K) / 4.
    zA2 = z_alpha(2)
    ms_arm = {k: arms[k]["MS_resid"] for k in ("A", "B")}
    null_rows = {}
    for ref in ("B", "A"):
        se_null = math.sqrt(2.0 * ms_arm[ref] / 4.0 / J)
        se_null_many = math.sqrt(ms_arm[ref] / 4.0 / J)
        g50 = J * (zA2 * se_null / U_nom) ** 2
        g50_cal = J * (zA2 * se_null / U_cal) ** 2

        def tpr_sep(U, G, s0, s1):
            return float(stats.norm.sf((zA2 * s0 * math.sqrt(J / G) - U) / (s1 * math.sqrt(J / G))))
        null_rows[f"reference_{ref}"] = {
            "SE_null_500": se_null, "SE_null_500_many_references": se_null_many,
            "G_for_TPR50": g50, "G_for_TPR50_calibrated": g50_cal,
            "TPR_G500": tpr_sep(U_nom, 500.0, se_null, se_sm),
            "TPR_G500_calibrated": tpr_sep(U_cal, 500.0, se_null, se_sm),
            "x_nominal_50_images": zA2 * se_null * math.sqrt(J / 50.0) / U_nom,
            "G_for_TPR50_many_references": J * (zA2 * se_null_many / U_nom) ** 2,
            "x_nominal_50_images_many_references": zA2 * se_null_many * math.sqrt(J / 50.0) / U_nom}
    avg = {k: float(np.mean([null_rows[r][k] for r in null_rows])) for k in null_rows["reference_B"]}
    avg["SE_null_500"] = float(math.sqrt(np.mean([null_rows[r]["SE_null_500"] ** 2 for r in null_rows])))
    # training sets: the suspect's and the reference's differ even under the null (persistent sd sigma_t/sqrt 2)
    s_t = load(OUT + "/h6_calibrated_limit.json")["training_sd"]
    room = (U_nom / zA2) ** 2 - s_t ** 2 / 2.0
    avg["G_for_TPR50_with_training_sets"] = (J * avg["SE_null_500"] ** 2 / room) if room > 0 else None
    power["seed_matched_reference"]["null_specific"] = {
        "per_orientation": null_rows, "mean_of_orientations": avg,
        "what": ("SE under the null from the arm the reference was trained on (seed term cancels exactly); the "
                 "alternative keeps SE_img_500 of the 144 cross-arm pairs; TPR = 1 - Phi((z SE0_G - U) / SE1_G); "
                 "one half needs G = 500 (z SE0 / U)^2 whatever SE1 is. mean_of_orientations averages the two "
                 "reference cameras (SE_null_500 as their RMS). with_training_sets adds the difference between the "
                 "suspect's and the reference's training sets as a persistent term (sd training_sd / sqrt 2)")}
    # fresh-seed examiner with the training-set component of H6 treated as persistent: the suspect's own training
    # set is not a main effect any examiner can know
    power["training_set_component_persistent"] = {
        "sigma": s_t, "source": "out/h6_calibrated_limit.json training_sd (nLimTrainSD)",
        "nominal": power_block(tpr, g_needed, U_nom, s_t, se500),
        "calibrated": power_block(tpr, g_needed, U_cal, s_t, se500),
        "what": ("fresh-seed examiner of eq. (power) with the H6 training-set component as the persistent term; "
                 "H6 estimates it from two contrasts")}
    power["transfer_for_TPR50_with_50_images"] = {
        "fresh_seeds_sigma_zero": {"x_nominal": mult50(0.0, se500, U_nom), "x_calibrated": mult50(0.0, se500, U_cal)},
        "fresh_seeds_archive_sigma_mu": {"x_nominal": mult50(inp["sigma_mu"], se500, U_nom),
                                         "x_calibrated": mult50(inp["sigma_mu"], se500, U_cal)},
        "seed_matched_sigma_zero": {"x_nominal": mult50(0.0, se_sm, U_nom), "x_calibrated": mult50(0.0, se_sm, U_cal)},
        "meaning": "multiple of the nominal (or calibrated) limit at which G = 50 gives a two-candidate TPR of 0.5"}
    # sensitivity: the image term measured over all 24 archive adapters instead of the filed two-adapter value
    se24 = res["seed_effect_and_image_noise"]["SE_500_single_adapter"]["rms_over_24"]
    power["sensitivity_SE_500_all_24"] = {
        "SE_img_500": se24,
        "nominal_sigma_zero_M2": power_block(tpr, g_needed, U_nom, 0.0, se24)["M2"],
        "calibrated_sigma_zero_M2": power_block(tpr, g_needed, U_cal, 0.0, se24)["M2"]}

    # ---------------------------------------------------------------- reproduction of the filed values
    rep = {}
    for r in pw["rows"]:
        if r["transfer"] != "at the upper limit U_device":
            continue
        M = r["M_candidates"]
        mine = power["nominal"]["archive_sigma_mu"][f"M{M}"]
        for k in ("TPR_G500", "TPR_G5000", "TPR_G_inf"):
            rep[f"M{M}.{k}"] = {"file": r[k], "recomputed": mine[k], "abs_diff": abs(r[k] - mine[k])}
    for r in pw["rows_sigma_mu_zero"]:
        M = r["M_candidates"]
        mine = power["nominal"]["zero"][f"M{M}"]
        for k in ("TPR_G500", "TPR_G5000"):
            rep[f"zero.M{M}.{k}"] = {"file": r[k], "recomputed": mine[k], "abs_diff": abs(r[k] - mine[k])}
    nums = load(NUMBERS)
    macro_checks = {}
    for name, val in (("nAttTprTwoFiveHundred", power["nominal"]["archive_sigma_mu"]["M2"]["TPR_G500"]),
                      ("nAttTprTwoFiveThousand", power["nominal"]["archive_sigma_mu"]["M2"]["TPR_G5000"]),
                      ("nAttTprTwoInf", power["nominal"]["archive_sigma_mu"]["M2"]["TPR_G_inf"]),
                      ("nAttTprTwoFiveHundredCal", power["calibrated"]["archive_sigma_mu"]["M2"]["TPR_G500"]),
                      ("nAttTprTwoFiveThousandCal", power["calibrated"]["archive_sigma_mu"]["M2"]["TPR_G5000"]),
                      ("nAttTprTwoInfCal", power["calibrated"]["archive_sigma_mu"]["M2"]["TPR_G_inf"]),
                      ("nAttTprZeroTwoFiveHundred", power["nominal"]["zero"]["M2"]["TPR_G500"]),
                      ("nAttTprZeroTwoFiveThousand", power["nominal"]["zero"]["M2"]["TPR_G5000"]),
                      ("nAttZeroGFiftyExact", power["nominal"]["zero"]["M2"]["G_for_TPR50"]),
                      ("nAttZeroGNinetyExact", power["nominal"]["zero"]["M2"]["G_for_TPR90"]),
                      ("nAttTenxGTwoExact", power["ten_times_nominal_G_for_TPR50"]["archive_sigma_mu"]["M2"]),
                      ("nAttTenxGFiveExact", power["ten_times_nominal_G_for_TPR50"]["archive_sigma_mu"]["M5"]),
                      ("nAttTenxGFiftyExact", power["ten_times_nominal_G_for_TPR50"]["archive_sigma_mu"]["M50"])):
        mv = nums[name]["value"]
        macro_checks[name] = {"macro_value": mv, "macro_text": nums[name]["text"], "recomputed": val,
                              "rel_diff": abs(val - mv) / abs(mv)}
    assert all(v["abs_diff"] < 1e-6 for v in rep.values()), rep
    assert all(v["rel_diff"] < 1e-6 for v in macro_checks.values()), macro_checks
    res["power"] = power
    sec["G2_local_2000_steps"]["largest_deviation_detail"]["nominal_limit_in_ncc_units"] = U_nom
    res["reproduction"] = {"t3_power_v4_rows": rep, "numbers_json_macros": macro_checks,
                           "note": ("the file's G_inf column was evaluated at G = 1e9 and differs from the limit by "
                                    "< 1e-6; every filed TPR and exact image count is reproduced")}

    # ---------------------------------------------------------------- what the numbers say
    nom0 = power["nominal"]["zero"]["M2"]
    cal0 = power["calibrated"]["zero"]["M2"]
    hi = power["nominal"]["paired_high_95"]["M2"]
    res["reading"] = {
        "sigma_mu_is_not_the_paired_persistent_term": (
            f"The 24 adapter means spread by {pooled['sd_adapter_means_pooled_within_arm']:.2e} within arm, what image "
            f"noise alone predicts ({sd_noise:.2e}); the archive component would make it "
            f"{math.sqrt(inp['sigma_mu'] ** 2 + sd_noise ** 2):.2e}. REML sigma_u = {su_hat:.2e}; F = "
            f"{pooled['F_adapter']:.2f} (p {pooled['F_p_upper']:.2f}); exact 95 % interval {ci['two_sided_95'][0]:.2e}"
            f" to {ci['two_sided_95'][1]:.2e}; the archive 5.23e-05 is excluded (one-sided p "
            f"{res['primary_crossed_model']['archive_sigma_mu_one_sided_p']:.4f})."),
        "robust_bounds": (
            f"sigma only lowers the TPR, so the sigma = 0 values bound every case: at the nominal limit two candidates "
            f"reach at most {nom0['TPR_G500']:.3f} with 500 images and {nom0['TPR_G5000']:.2f} with 5,000; one half "
            f"needs at least {nom0['G_for_TPR50']:.0f} images and 0.9 {nom0['G_for_TPR90']:.0f}; at the calibrated "
            f"limit {cal0['TPR_G500']:.3f}, {cal0['TPR_G5000']:.2f}, {cal0['G_for_TPR50']:.0f} and "
            f"{cal0['G_for_TPR90']:.0f}."),
        "upper_end": (f"at the upper end of the interval ({ci['two_sided_95'][1]:.2e}) the unlimited-image TPR at the "
                      f"nominal limit is {hi['TPR_G_inf']:.3f} and one half is unattainable"),
        "g2": ("the G2 local design shows a persistent component, entirely from one adapter (see "
               "secondary_designs_descriptive); that adapter's deviation persists over the seed bank but, as the "
               "most extreme of 48, has a family-wise p of about 0.1, so a heavy-tailed persistent term is not "
               "established (corrected in the Entry 114 addendum)"),
        "seed_matched": (f"an examiner matching seeds against a reference adapter of the other camera has an image term "
                         f"of {se_sm:.2e} per 500 images; one half at the nominal limit then needs "
                         f"{power['seed_matched_reference']['nominal']['G_for_TPR50']:.0f} images"),
    }
    res["runtime_s"] = time.time() - t0
    with open(DST, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    for k, v in res["reading"].items():
        print(f"[fv_sigma] {k}: {v}")
    print(f"[fv_sigma] written {DST} ({res['runtime_s']:.0f} s)")


if __name__ == "__main__":
    main()
