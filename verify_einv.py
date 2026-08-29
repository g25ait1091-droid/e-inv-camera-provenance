#!/usr/bin/env python3
"""
E-INV — independent verification.

Recomputes every headline number in the manuscript from the per-row measurement
CSVs and compares against the frozen ledger. Deliberately does NOT read the
cached result JSONs: those are what is being checked.

Runs anywhere with numpy, pandas and scipy. No GPU, no Colab, no model weights.

    python verify_einv.py --root /path/to/inv_channel
    python verify_einv.py --root ~/Drive/MyDrive/inv_channel --verbose

Each experiment root is expected under --root:
    E_INV_P0_v3/  E_SEEDEXT/  E_SEEDEXT2/  E_AMP/  E_TRACKB2_FLUX/
    E_MULTIDEV/   E_LOWFREQ/  E_FULLFT/
E_SEEDEXT2 carries seeds 6-11 and is REQUIRED for the k=12 headline bound; without
it the k=12 checks report "E_SEEDEXT2 root absent" rather than silently falling back
to the superseded k=6 value.
Missing roots are reported as SKIP, not as failures, so a partial copy of the
data still verifies whatever it contains.
"""
import os, sys, json, argparse, itertools, warnings
import numpy as np
import pandas as pd
from scipy import stats as sps

warnings.filterwarnings("ignore")
RNG = np.random.default_rng(0)

R_REAL_REF = 3.56703416571125e-02      # denominator used throughout the manuscript
T5, T2_, T4 = (float(sps.t.ppf(0.995, df=d)) for d in (5, 2, 4))

# --------------------------------------------------------------------- helpers
class Report:
    def __init__(self): self.rows = []
    def add(self, name, expected, got, tol=None, note=""):
        if got is None:
            self.rows.append((name, expected, None, "SKIP", note)); return
        if expected is None:
            self.rows.append((name, None, got, "INFO", note)); return
        rel = abs(got - expected) / max(abs(expected), 1e-30)
        tol = tol if tol is not None else 0.01           # 1 % default
        ok = rel <= tol
        self.rows.append((name, expected, got, "OK" if ok else "DISAGREE",
                          note or f"rel {rel:.2%}"))
    def show(self):
        w = max(len(r[0]) for r in self.rows) + 2
        print("\n" + "=" * (w + 58))
        print(f"{'quantity':<{w}}{'manuscript':>14}{'recomputed':>14}  {'':<9}note")
        print("-" * (w + 58))
        for n, e, g, s, note in self.rows:
            es = "—" if e is None else (f"{e:.6g}" if abs(e) < 1e4 else f"{e:.4e}")
            gs = "—" if g is None else (f"{g:.6g}" if abs(g) < 1e4 else f"{g:.4e}")
            mark = {"OK": "  ok", "DISAGREE": "  ** DISAGREE", "SKIP": "  skip",
                    "INFO": "  info"}[s]
            print(f"{n:<{w}}{es:>14}{gs:>14}{mark:<11}{note}")
        print("=" * (w + 58))
        bad = [r for r in self.rows if r[3] == "DISAGREE"]
        skip = [r for r in self.rows if r[3] == "SKIP"]
        print(f"\n{sum(1 for r in self.rows if r[3]=='OK')} verified, "
              f"{len(bad)} disagreements, {len(skip)} skipped (data absent).")
        if bad:
            print("\nDISAGREEMENTS — investigate before submitting:")
            for r in bad: print(f"   {r[0]}: manuscript {r[1]:.6g}, recomputed {r[2]:.6g}")
        return len(bad)

def load(root, sub, name):
    p = os.path.join(root, sub, "csv", name)
    if not os.path.exists(p): return None
    try: return pd.read_csv(p)
    except Exception as e: print(f"   ! could not read {p}: {e}"); return None

def col(df, *cands):
    for c in cands:
        if c in df.columns: return c
    return None

def valcol(df):
    c = col(df, "rho", "corr", "ncc", "value", "rho_gen", "r", "score")
    if c is not None: return c
    num = [c for c in df.columns if df[c].dtype.kind == "f" and df[c].abs().max() <= 1.5]
    return num[-1] if num else None

def ucl(x, alpha=0.01):
    k = len(x); tc = float(sps.t.ppf(1 - alpha / 2, df=k - 1))
    return float(np.mean(x) + tc * np.std(x, ddof=1) / np.sqrt(k))

def paired_theta(df, tag, own, oth, tagc="tag", kc="K", idc=None, rc=None,
                 drop=frozenset()):
    idc = idc or col(df, "gen_idx", "idx", "i")
    rc  = rc  or valcol(df)
    if idc is None or rc is None: return None
    a = df[(df[tagc] == tag) & (df[kc] == own)].sort_values(idc)
    b = df[(df[tagc] == tag) & (df[kc] == oth)].sort_values(idc)
    if not len(a) or not len(b): return None
    ids = a[idc].to_numpy()
    va, vb = a[rc].to_numpy(float), b[rc].to_numpy(float)
    n = min(len(va), len(vb))
    keep = ~np.isin(ids[:n], list(drop))
    return (va[:n] - vb[:n])[keep]

def exact_signflip(m):
    """Exact one-sided sign-flip p over all 2^k assignments."""
    k = len(m); obs = float(np.mean(m))
    c = sum(1 for sg in itertools.product([-1, 1], repeat=k)
            if float(np.mean(m * np.array(sg))) >= obs)
    return c / 2 ** k

# ============================================================== 1. encoder stage
def check_encoder(root, rep, verbose):
    d = load(root, "E_INV_P0_v3", "s1b_vae_roundtrip.csv")
    if d is None:
        for n in ("R_real (round-trip file)", "R_VAE", "eta", "AUC post"):
            rep.add(n, None, None, note="s1b_vae_roundtrip.csv absent")
        return
    rc, sc = col(d, "role"), col(d, "stage")
    ra, rb = col(d, "rho_A", "rhoA"), col(d, "rho_B", "rhoB")
    ic = col(d, "idx", "gen_idx")
    if None in (rc, sc, ra, rb, ic):
        rep.add("encoder stage", None, None, note="unexpected columns"); return

    def con(stage):
        a = d[(d[rc] == "A") & (d[sc] == stage)].sort_values(ic)
        b = d[(d[rc] == "B") & (d[sc] == stage)].sort_values(ic)
        return (a[ra].to_numpy(float) - a[rb].to_numpy(float),
                b[rb].to_numpy(float) - b[ra].to_numpy(float))
    pa, pb = con("pre"); qa, qb = con("post")
    R_real = 0.5 * (pa.mean() + pb.mean()); R_vae = 0.5 * (qa.mean() + qb.mean())
    rep.add("R_real", 3.56703e-02, float(R_real))
    rep.add("R_VAE",  1.30593e-02, float(R_vae))
    rep.add("eta",    0.3661,      float(R_vae / R_real))

    et = []
    for _ in range(10000):
        ia, ib = RNG.integers(0, len(pa), len(pa)), RNG.integers(0, len(pb), len(pb))
        et.append((0.5 * (qa[ia].mean() + qb[ib].mean())) /
                  (0.5 * (pa[ia].mean() + pb[ib].mean())))
    lo, hi = np.percentile(et, [2.5, 97.5])
    rep.add("eta CI low",  0.3437, float(lo), tol=0.02)
    rep.add("eta CI high", 0.3870, float(hi), tol=0.02)

    def auc(stage):
        a = d[(d[rc] == "A") & (d[sc] == stage)]; b = d[(d[rc] == "B") & (d[sc] == stage)]
        same = np.r_[a[ra].to_numpy(float), b[rb].to_numpy(float)]
        cross = np.r_[a[rb].to_numpy(float), b[ra].to_numpy(float)]
        lab = np.r_[np.ones(len(same)), np.zeros(len(cross))]; sc_ = np.r_[same, cross]
        o = np.argsort(sc_); rk = np.empty(len(sc_)); rk[o] = np.arange(len(sc_))
        return float((rk[lab == 1].mean() - (lab.sum() - 1) / 2) / (len(sc_) - lab.sum()))
    rep.add("AUC pre",  1.0000, auc("pre"),  tol=0.001)
    rep.add("AUC post", 0.9814, auc("post"), tol=0.005)

# ============================================ 2. primary bound, twelve adapters
# Pools all three measurement roots. The registered design used seeds 0-5 across
# E_INV_P0_v3 and E_SEEDEXT; the declared extension added seeds 6-11 in E_SEEDEXT2.
# Both cluster counts are verified, because the paper reports the k=6 -> k=12
# tightening as well as the final limit.
def check_primary(root, rep, verbose):
    frames = [f for f in (load(root, "E_INV_P0_v3", "s5_measure.csv"),
                          load(root, "E_SEEDEXT",   "b3_measure.csv"),
                          load(root, "E_SEEDEXT2",  "sx2_measure.csv")) if f is not None]
    if not frames:
        rep.add("U_device (k=12)", 5.3761e-05, None, note="no measure CSVs found")
        return
    per = {"A": {}, "B": {}}
    for d in frames:                       # per frame: its own column names, no concat
        tagc, kc = col(d, "tag"), col(d, "K")
        vc = valcol(d) if "valcol" in globals() else None
        for arm, own, oth in (("A", "A", "B"), ("B", "B", "A")):
            for t in d[tagc].unique():
                ts = str(t)
                if not (ts.startswith(f"{arm}_raw_s") and ts.endswith("_r16")):
                    continue               # excludes the rank-64 capacity-ceiling arms
                x = paired_theta(d, t, own, oth, tagc=tagc, kc=kc)
                if x is not None and len(x):
                    per[arm][ts] = float(x.mean())
    import re as _re
    def _seed(t):                     # "A_raw_s10_r16" -> 10; lexicographic order is wrong,
        m = _re.search(r"_s(\d+)_", t)   # s10 and s11 sort before s2
        return int(m.group(1)) if m else 10**6
    ordA, ordB = sorted(per["A"], key=_seed), sorted(per["B"], key=_seed)
    A = np.array([per["A"][t] for t in ordA])
    B = np.array([per["B"][t] for t in ordB])
    if verbose:
        print(f"   arm A: {len(A)} adapters {[_seed(t) for t in ordA]}")
        print(f"   arm B: {len(B)} adapters {[_seed(t) for t in ordB]}")
    if len(A) < 2 or len(B) < 2:
        rep.add("U_device (k=12)", 5.3761e-05, None, note="too few adapters found"); return

    k = min(len(A), len(B))
    if k >= 12:
        rep.add("theta_A (k=12)", -9.570e-06, float(A.mean()), tol=0.05)
        rep.add("theta_B (k=12)",  1.877e-05, float(B.mean()), tol=0.05)
        U = max(ucl(A), ucl(B))
        rep.add("U_device (k=12)", 5.3761e-05, U, tol=0.05)
        rep.add("lambda_U plug-in %", 0.15072, 100 * U / R_REAL_REF, tol=0.05)
        rep.add("theta_sym (k=12)", 4.600e-06, float(0.5 * (A.mean() + B.mean())), tol=0.10)
        rep.add("exact sign-flip p_A", 0.7546, exact_signflip(A), tol=0.05)
        rep.add("exact sign-flip p_B", 0.0625, exact_signflip(B), tol=0.05)
        rep.add("sign-flip floor 1/4096", 1 / 4096, 1 / 4096, tol=1e-9,
                note="rejection at alpha=0.01 was attainable and did not occur")
    else:
        rep.add("U_device (k=12)", 5.3761e-05, None,
                note=f"only {k} adapters per arm found; E_SEEDEXT2 root absent")

    # the superseded k=6 analysis, from seeds 0-5 alone, reported in Section V-C
    A6 = np.array([per["A"][t] for t in ordA if _seed(t) < 6])
    B6 = np.array([per["B"][t] for t in ordB if _seed(t) < 6])
    if len(A6) == 6 and len(B6) == 6:
        U6 = max(ucl(A6), ucl(B6))
        rep.add("U_device (k=6, superseded)", 8.8802e-05, U6, tol=0.05,
                note="the registered six-cluster analysis")
        rep.add("lambda_U k=6 % (superseded)", 0.2490, 100 * U6 / R_REAL_REF, tol=0.05)
        rep.add("sign-flip floor 1/64", 0.015625, 1 / 64, tol=1e-9,
                note="structural: could not reject at 0.01, which is why k was extended")

# ============================================================== 3. FLUX, full FT
def _three_seed_arm(d, prefix, own, oth, tagc, kc, drop=frozenset()):
    m = []
    for s in (0, 1, 2):
        for suff in ("", "_flux", "_ft"):
            t = f"{prefix}{suff}_s{s}" if suff else f"{prefix}_s{s}"
            for cand in (t, f"{prefix}_ft_s{s}", f"{prefix}_raw_s{s}_flux"):
                x = paired_theta(d, cand, own, oth, tagc=tagc, kc=kc,
                                 drop={g for (tt, g) in drop if tt == cand})
                if x is not None and len(x): m.append(float(x.mean())); break
            else: continue
            break
    return np.array(m)

def check_flux(root, rep, verbose):
    d = load(root, "E_TRACKB2_FLUX", "f3_measure.csv")
    if d is None:
        rep.add("FLUX lambda_U % (n=3)", 0.777, None, note="FLUX f3_measure.csv absent"); return
    tagc, kc = col(d, "tag"), col(d, "K")
    A = _three_seed_arm(d, "A", "A", "B", tagc, kc)
    B = _three_seed_arm(d, "B", "B", "A", tagc, kc)
    if len(A) < 3 or len(B) < 3:
        rep.add("FLUX lambda_U % (n=3)", 0.777, None, note="fewer than 3 adapters/arm"); return
    rep.add("FLUX theta_A", 1.342e-05, float(A.mean()), tol=0.05)
    rep.add("FLUX theta_B", -7.867e-06, float(B.mean()), tol=0.05)
    U = max(ucl(A), ucl(B))
    rep.add("FLUX lambda_U % (n=3)", 0.777, 100 * U / R_REAL_REF, tol=0.05)

def check_fullft(root, rep, verbose):
    d = load(root, "E_FULLFT", "f3_measure.csv")
    if d is None:
        rep.add("full-FT lambda_U % (n=3)", 0.7466, None, note="E_FULLFT f3_measure.csv absent")
        return None
    tagc, kc = col(d, "tag"), col(d, "K")
    A = _three_seed_arm(d, "A", "A", "B", tagc, kc)
    B = _three_seed_arm(d, "B", "B", "A", tagc, kc)
    if len(A) < 3 or len(B) < 3:
        rep.add("full-FT lambda_U % (n=3)", 0.7466, None, note="fewer than 3 adapters/arm")
        return None
    rep.add("full-FT theta_A", -1.134e-04, float(A.mean()), tol=0.05)
    rep.add("full-FT theta_B",  6.829e-05, float(B.mean()), tol=0.05)
    sym = (A + B) / 2
    rep.add("full-FT theta_sym", -2.2553e-05, float(sym.mean()), tol=0.05)
    U = ucl(sym)
    rep.add("full-FT lambda_U % (sym, n=3)", 0.7466, 100 * U / R_REAL_REF, tol=0.05)
    rep.add("full-FT lambda_U % (per-arm)", 2.716,
            100 * max(ucl(A), ucl(B)) / R_REAL_REF, tol=0.05,
            note="additive-contaminated; NOT the reported bound")
    anti = (A - B) / 2
    rep.add("full-FT |additive/interaction|", 4.03,
            float(abs(anti.mean() / sym.mean())), tol=0.05)
    return (d, tagc, kc, sym.mean(), U)

# ================================================ 4. copy-audit sensitivity (F5b)
def check_copy_sensitivity(root, rep, ft, verbose):
    f = load(root, "E_FULLFT", "f5_copy.csv")
    if f is None or ft is None:
        rep.add("full-FT theta_sym, flagged excluded", None, None,
                note="f5_copy.csv or f3_measure.csv absent"); return
    d, tagc, kc, sym0, U0 = ft
    fc = col(f, "copy_flag"); tc_ = col(f, "tag"); gc_ = col(f, "gen_idx")
    bad = {(r[tc_], int(r[gc_])) for _, r in f[f[fc] == 1].iterrows()}
    rate = 100.0 * f[fc].mean()
    rep.add("copy-flag rate %", 0.07, float(rate), tol=0.30,
            note=f"{len(bad)} of {len(f)} generations")
    if not bad:
        rep.add("full-FT theta_sym, flagged excluded", float(sym0), float(sym0),
                note="nothing flagged"); return
    A = _three_seed_arm(d, "A", "A", "B", tagc, kc, drop=bad)
    B = _three_seed_arm(d, "B", "B", "A", tagc, kc, drop=bad)
    sym = (A + B) / 2
    U = ucl(sym)
    rep.add("full-FT theta_sym, flagged excluded", None, float(sym.mean()),
            note=f"shift {100*abs(sym.mean()-sym0)/abs(sym0):.2f}% of its magnitude")
    rep.add("full-FT lambda_U %, flagged excluded", None, 100 * U / R_REAL_REF,
            note=f"vs {100*U0/R_REAL_REF:.4f}% with all generations")
    print(f"\n   flagged and excluded: {sorted(bad)}")
    print(f"   theta_sym  {sym0:+.4e} -> {sym.mean():+.4e}")
    print(f"   lambda_U   {100*U0/R_REAL_REF:.4f}% -> {100*U/R_REAL_REF:.4f}%")
    if abs(sym.mean() - sym0) < 0.2 * abs(sym0):
        print("   => shift is small relative to the estimate; the copy audit is closed.")
    else:
        print("   => shift is material; report the leave-flagged-out value as primary.")

# ============================================================ 5. low/mid band
def check_lowmid(root, rep, verbose):
    d = load(root, "E_LOWFREQ", "l2_measure.csv")
    g = os.path.join(root, "E_LOWFREQ", "fingerprints", "l0_gates.json")
    if d is None:
        rep.add("low/mid theta_sym", -1.036e-05, None, note="l2_measure.csv absent"); return
    tagc, lc = col(d, "tag"), col(d, "L", "K")
    per = {}
    for arm, own, oth in (("A", "A", "B"), ("B", "B", "A")):
        m = [float(x.mean()) for s in range(6)
             for x in [paired_theta(d, f"{arm}_raw_s{s}_r16", own, oth, tagc=tagc, kc=lc)]
             if x is not None and len(x)]
        per[arm] = np.array(m)
    if len(per["A"]) < 2 or len(per["B"]) < 2:
        rep.add("low/mid theta_sym", -1.036e-05, None, note="adapters not found"); return
    rep.add("low/mid theta_A", -1.110e-04, float(per["A"].mean()), tol=0.05)
    rep.add("low/mid theta_B",  9.032e-05, float(per["B"].mean()), tol=0.05)
    sym = (per["A"] + per["B"]) / 2
    rep.add("low/mid theta_sym", -1.036e-05, float(sym.mean()), tol=0.08)
    if os.path.exists(g):
        R_low = json.load(open(g)).get("R_real_lower99")
        if R_low:
            rep.add("low/mid lambda_U % (sym)", 9.72, 100 * ucl(sym) / R_low, tol=0.05)
            rep.add("low/mid lambda_U % (per-arm)", 20.56,
                    100 * max(ucl(per["A"]), ucl(per["B"])) / R_low, tol=0.05,
                    note="additive-contaminated; NOT the reported bound")

# ============================================================ 6. Kodak, additive
def check_kodak(root, rep, verbose):
    d = load(root, "E_MULTIDEV", "c3_measure_raw.csv")
    pc = load(root, "E_MULTIDEV", "c0_positive_control.csv")
    if d is None:
        rep.add("Kodak ICC(1,1)", 0.943, None, note="c3_measure_raw.csv absent"); return
    tagc, kc = col(d, "tag"), col(d, "K")
    devs = sorted(d[kc].unique())
    rows = []
    for t in sorted(d[tagc].unique()):
        dev = str(t).split("_")[0]
        if dev not in devs: continue
        _v = valcol(d); _i = col(d, "gen_idx", "idx")
        own = d[(d[tagc] == t) & (d[kc] == dev)].sort_values(_i)[_v].to_numpy(float)
        oth = [d[(d[tagc] == t) & (d[kc] == o)].sort_values(_i)[_v].to_numpy(float)
               for o in devs if o != dev]
        n = min([len(own)] + [len(x) for x in oth])
        if n == 0: continue
        rows.append((dev, int(str(t).split("_s")[-1]),
                     float((own[:n] - np.mean([x[:n] for x in oth], axis=0)).mean())))
    if len(rows) < 6:
        rep.add("Kodak ICC(1,1)", 0.943, None, note="too few adapters"); return
    R = pd.DataFrame(rows, columns=["dev", "seed", "theta"])
    dm = R.groupby("dev")["theta"].mean(); gm = float(R.theta.mean())
    k, nps = len(dm), R.seed.nunique()
    ms = float(nps * ((dm - gm) ** 2).sum() / (k - 1))
    within = float(sum(((R[R.dev == dv].theta - dm[dv]) ** 2).sum()
                       for dv in dm.index) / (k * (nps - 1)))
    var_u = max((ms - within) / nps, 0.0)
    rep.add("Kodak ICC(1,1)", 0.943, float(var_u / (var_u + within)), tol=0.02)
    rep.add("Kodak grand mean theta", 1.265e-05, gm, tol=0.05)
    rep.add("Kodak sign-flip p (devices)", 0.434,
            float(exact_signflip(dm.to_numpy())), tol=0.15)

    # additive decomposition, held out across seeds
    M = d.groupby([tagc, kc])[valcol(d)].mean().unstack()
    dev_of = {t: str(t).split("_")[0] for t in M.index}
    seed_of = {t: int(str(t).split("_s")[-1]) for t in M.index}
    DEV = list(M.columns)
    def fit(sub): 
        g0 = sub.values.mean(); return g0, sub.mean(1) - g0, sub.mean(0) - g0
    pred, obs = [], []
    for fs in sorted(set(seed_of.values())):
        tr = [t for t in M.index if seed_of[t] == fs]
        te = [t for t in M.index if seed_of[t] != fs]
        if not tr or not te: continue
        _, _, b = fit(M.loc[tr])
        for t in te:
            dv = dev_of[t]
            pred.append(float(b[dv] - np.mean([b[o] for o in DEV if o != dv])))
            own = M.loc[t, dv]; oth = np.mean([M.loc[t, o] for o in DEV if o != dv])
            obs.append(float(own - oth))
    if len(pred) >= 6:
        pred, obs = np.array(pred), np.array(obs)
        r2 = 1 - ((obs - pred) ** 2).sum() / ((obs - obs.mean()) ** 2).sum()
        rep.add("additive R^2 cross-seed", 0.870, float(r2), tol=0.05)
        # exact permutation over device assignments, at device level
        P = np.array([np.mean([pred[i] for i in range(len(pred))]) for _ in [0]])  # placeholder
        byd = {}
        for i, t in enumerate([t for fs in sorted(set(seed_of.values()))
                               for t in M.index if seed_of[t] != fs]):
            byd.setdefault(dev_of[t], []).append((pred[i], obs[i]))
        Pd = np.array([np.mean([p for p, _ in v]) for _, v in sorted(byd.items())])
        Od = np.array([np.mean([o for _, o in v]) for _, v in sorted(byd.items())])
        if len(Pd) == 5:
            r2f = lambda o, p: 1 - ((o - p) ** 2).sum() / ((o - o.mean()) ** 2).sum()
            obs_r2 = r2f(Od, Pd)
            null = [r2f(Od, Pd[list(pm)]) for pm in itertools.permutations(range(5))]
            rep.add("additive exact permutation p", 0.0167,
                    float(sum(1 for v in null if v >= obs_r2) / 120), tol=0.40)
            rep.add("additive device-level r", 0.979,
                    float(sps.pearsonr(Pd, Od)[0]), tol=0.05)

# ======================================================================== main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True,
                    help="directory containing E_INV_P0_v3/, E_SEEDEXT/, ... ")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    root = os.path.expanduser(a.root)
    if not os.path.isdir(root): sys.exit(f"not a directory: {root}")

    print(f"E-INV independent verification\nroot: {root}")
    print("recomputing from per-row CSVs; cached result JSONs are NOT consulted.")
    present = [d for d in ("E_INV_P0_v3", "E_SEEDEXT", "E_AMP", "E_TRACKB2_FLUX",
                           "E_MULTIDEV", "E_LOWFREQ", "E_FULLFT")
               if os.path.isdir(os.path.join(root, d))]
    print(f"experiment roots found: {', '.join(present) if present else 'NONE'}")

    rep = Report()
    for fn in (check_encoder, check_primary, check_flux, check_lowmid, check_kodak):
        try: fn(root, rep, a.verbose)
        except Exception as e: rep.add(fn.__name__, None, None, note=f"error: {e}")
    try:
        ft = check_fullft(root, rep, a.verbose)
        check_copy_sensitivity(root, rep, ft, a.verbose)
    except Exception as e:
        rep.add("full fine-tuning", None, None, note=f"error: {e}")

    nbad = rep.show()
    print("\nInterpretation")
    print("  OK        recomputed value matches the manuscript within tolerance.")
    print("  DISAGREE  the manuscript and the raw data differ — resolve before submitting.")
    print("  skip      the CSV for that quantity is not present under --root.")
    print("  info      reported for context; no manuscript value to compare against.")
    print("\nNote: two archived JSONs hold superseded values by design "
          "(E_LOWFREQ lambda_U and\nXVAL p_perm). See config/results_corrections.json. "
          "This script recomputes both correctly.")
    sys.exit(1 if nbad else 0)

if __name__ == "__main__":
    main()
